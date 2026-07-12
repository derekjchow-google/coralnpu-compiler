// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "compiler/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

// MLIR headers
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "coralnpu-vector-to-matrix"

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUVECTORTOMATRIX
#include "compiler/Transforms/Passes.h.inc"

namespace {

struct AccumulatorChain {
  Value initVal;
  SmallVector<Operation *> updates;
  vector::TransferWriteOp writeback;
  int64_t tileId = -1;
  Type inputElTy;
};

Value traceToSourceValue(Value val) {
  while (Operation *defOp = val.getDefiningOp()) {
    if (auto extOp = dyn_cast<arith::ExtFOp>(defOp)) {
      val = extOp.getIn();
    } else if (auto extOp = dyn_cast<arith::ExtSIOp>(defOp)) {
      val = extOp.getIn();
    } else if (auto extOp = dyn_cast<arith::ExtUIOp>(defOp)) {
      val = extOp.getIn();
    } else if (auto broadcastOp = dyn_cast<vector::BroadcastOp>(defOp)) {
      val = broadcastOp.getSource();
    } else if (auto shapeCastOp = dyn_cast<vector::ShapeCastOp>(defOp)) {
      val = shapeCastOp.getSource();
    } else {
      break;
    }
  }
  return val;
}

Type traceToSourceType(Value val) { return traceToSourceValue(val).getType(); }

int64_t getDimPosition(AffineMap map, int64_t dimIdx) {
  for (unsigned i = 0; i < map.getNumResults(); ++i) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i))) {
      if (dimExpr.getPosition() == dimIdx) {
        return i;
      }
    }
  }
  return -1;
}

bool isSupportedMatrixContraction(vector::ContractionOp contractOp) {
  auto lhsType = cast<VectorType>(contractOp.getLhs().getType());
  auto rhsType = cast<VectorType>(contractOp.getRhs().getType());
  auto accType = cast<VectorType>(contractOp.getAcc().getType());

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
      accType.getRank() != 2) {
    return false;
  }

  // Find reduction dimension index
  int64_t reductionDimIdx = -1;
  auto iterTypes = contractOp.getIteratorTypes().getValue();
  for (int64_t i = 0; i < iterTypes.size(); ++i) {
    if (vector::isReductionIterator(iterTypes[i])) {
      reductionDimIdx = i;
      break;
    }
  }
  if (reductionDimIdx == -1) return false;

  auto mapsAttr = contractOp.getIndexingMaps();
  auto lhsMap = cast<AffineMapAttr>(mapsAttr[0]).getValue();
  auto rhsMap = cast<AffineMapAttr>(mapsAttr[1]).getValue();

  int64_t lhsKPos = getDimPosition(lhsMap, reductionDimIdx);
  int64_t rhsKPos = getDimPosition(rhsMap, reductionDimIdx);
  if (lhsKPos == -1 || rhsKPos == -1) return false;

  int64_t M = lhsType.getDimSize(1 - lhsKPos);
  int64_t K = lhsType.getDimSize(lhsKPos);
  int64_t N = rhsType.getDimSize(1 - rhsKPos);
  int64_t KR = rhsType.getDimSize(rhsKPos);

  if (K != KR) return false;

  Type inputElTy =
      cast<VectorType>(traceToSourceValue(contractOp.getLhs()).getType())
          .getElementType();
  Type accElTy = accType.getElementType();

  LLVM_DEBUG(llvm::dbgs() << "MATRIX_PASS: checking contract: M=" << M
                          << ", N=" << N << ", K=" << K << ", KR=" << KR
                          << ", inputElTy=" << inputElTy << "\n");

  if (inputElTy.isF32() && accElTy.isF32()) {
    return M == 4 && N == 4 && K == 1;
  }
  if (inputElTy.isBF16() && accElTy.isF32()) {
    return M == 4 && N == 4 && K == 1;
  }
  if (inputElTy.isInteger(8) && accElTy.isInteger(32)) {
    return M == 4 && N == 4 && K == 1;
  }

  return false;
}

// Traverses up the SSA chain to find the initialization value and all updates.
std::optional<AccumulatorChain> analyzeChain(vector::TransferWriteOp writeOp) {
  AccumulatorChain chain;
  chain.writeback = writeOp;

  Value val = writeOp.getVector();

  auto vecType = dyn_cast<VectorType>(val.getType());
  if (!vecType || vecType.getRank() != 2) {
    return std::nullopt;
  }
  Type elemTy = vecType.getElementType();
  if (!elemTy.isF32() && !elemTy.isInteger(32)) {
    return std::nullopt;
  }

  // Keep track of active loop we are tracing inside
  scf::ForOp activeLoop = nullptr;
  unsigned activeLoopArgNum = 0;
  Value activeLoopInitVal = nullptr;

  while (val) {
    // Check if it is a block argument first
    if (auto blockArg = dyn_cast<BlockArgument>(val)) {
      Operation *owner = blockArg.getOwner()->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        if (forOp == activeLoop &&
            blockArg.getArgNumber() == activeLoopArgNum) {
          // We reached the block argument corresponding to the loop result we
          // started with. Transition to the initial value outside the loop.
          val = activeLoopInitVal;
          activeLoop = nullptr;  // exit loop context
          continue;
        }
      }
      // If it is another block argument (e.g. function argument), we stop.
      if (!activeLoop) {
        chain.initVal = val;
      }
      break;
    }

    Operation *defOp = val.getDefiningOp();
    if (!defOp) {
      break;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      if (activeLoop) {
        // Nested loops not supported yet
        return std::nullopt;
      }
      // We are entering a loop from its result.
      auto opResult = cast<OpResult>(val);
      unsigned resNum = opResult.getResultNumber();

      activeLoop = forOp;
      // Block arguments: IV is 0, iter_args start at 1.
      activeLoopArgNum = resNum + 1;
      activeLoopInitVal = forOp.getInitArgs()[resNum];

      // Transition to the yielded value inside the loop
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      val = yieldOp.getOperand(resNum);
      continue;
    }

    if (auto contractOp = dyn_cast<vector::ContractionOp>(defOp)) {
      if (!isSupportedMatrixContraction(contractOp)) {
        return std::nullopt;
      }
      chain.updates.push_back(contractOp);
      chain.inputElTy = cast<VectorType>(traceToSourceType(contractOp.getLhs()))
                            .getElementType();
      val = contractOp.getAcc();  // trace through accumulator
      continue;
    }

    if (auto outerProdOp = dyn_cast<vector::OuterProductOp>(defOp)) {
      // Outer product not supported for Matrix
      return std::nullopt;
    }

    // If it is a constant or transfer_read, it is the initialization.
    if (isa<arith::ConstantOp>(defOp) || isa<vector::TransferReadOp>(defOp)) {
      if (!activeLoop) {  // Only set initVal if we are not inside a loop (or we
                          // already exited it)
        chain.initVal = val;
      }
      break;
    }

    // Unknown op in chain, abort
    LLVM_DEBUG(llvm::dbgs()
               << "Unknown op in accumulator chain: " << *defOp << "\n");
    return std::nullopt;
  }

  // Reverse updates so they are in execution order
  std::reverse(chain.updates.begin(), chain.updates.end());

  if (chain.updates.empty() || !chain.initVal) {
    return std::nullopt;
  }

  return chain;
}

bool isConstantZero(Value val) {
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      if (attr.isSplat()) {
        auto splatVal = attr.getSplatValue<Attribute>();
        if (auto floatAttr = dyn_cast<FloatAttr>(splatVal)) {
          return floatAttr.getValue().isZero();
        }
        if (auto intAttr = dyn_cast<IntegerAttr>(splatVal)) {
          return intAttr.getValue().isZero();
        }
      }
    }
  }
  return false;
}

func::FuncOp getOrDeclareWritebackFunc(ModuleOp moduleOp, OpBuilder &builder,
                                       Type memrefType) {
  auto memrefValType = cast<MemRefType>(memrefType);
  auto elementTy = memrefValType.getElementType();
  auto ctx = moduleOp.getContext();
  auto dynamicVal = ShapedType::kDynamic;
  auto layout =
      StridedLayoutAttr::get(ctx, dynamicVal, {dynamicVal, dynamicVal});
  auto genericMemrefType = MemRefType::get({dynamicVal, dynamicVal}, elementTy,
                                           layout, /*memorySpace=*/0);

  unsigned width = elementTy.getIntOrFloatBitWidth();
  std::string funcName = "coralnpu.matrix.writeback.w" + std::to_string(width);
  if (auto func = moduleOp.lookupSymbol<func::FuncOp>(funcName)) {
    return func;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto funcType =
      builder.getFunctionType({genericMemrefType, builder.getIndexType(),
                               builder.getIndexType(), builder.getI32Type()},
                              TypeRange{});
  auto funcOp =
      func::FuncOp::create(builder, moduleOp.getLoc(), funcName, funcType);
  funcOp.setPrivate();
  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  func::ReturnOp::create(builder, moduleOp.getLoc());
  return funcOp;
}

func::FuncOp getOrDeclareLoadFunc(ModuleOp moduleOp, OpBuilder &builder,
                                  Type memrefType) {
  auto memrefValType = cast<MemRefType>(memrefType);
  auto elementTy = memrefValType.getElementType();
  auto ctx = moduleOp.getContext();
  auto dynamicVal = ShapedType::kDynamic;
  auto layout =
      StridedLayoutAttr::get(ctx, dynamicVal, {dynamicVal, dynamicVal});
  auto genericMemrefType = MemRefType::get({dynamicVal, dynamicVal}, elementTy,
                                           layout, /*memorySpace=*/0);

  unsigned width = elementTy.getIntOrFloatBitWidth();
  std::string funcName = "coralnpu.matrix.load.w" + std::to_string(width);
  if (auto func = moduleOp.lookupSymbol<func::FuncOp>(funcName)) {
    return func;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(moduleOp.getBody());
  auto funcType =
      builder.getFunctionType({genericMemrefType, builder.getIndexType(),
                               builder.getIndexType(), builder.getI32Type()},
                              TypeRange{});
  auto funcOp =
      func::FuncOp::create(builder, moduleOp.getLoc(), funcName, funcType);
  funcOp.setPrivate();
  Block *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);
  func::ReturnOp::create(builder, moduleOp.getLoc());
  return funcOp;
}

Value castToScalable(OpBuilder &builder, Location loc, Value val) {
  VectorType valType = cast<VectorType>(val.getType());
  int64_t size = valType.getNumElements();
  VectorType flatType = VectorType::get({size}, valType.getElementType());
  Value flatVal = vector::ShapeCastOp::create(builder, loc, flatType, val);

  int64_t minElements = std::max<int64_t>(1, size / 2);
  VectorType scalableType =
      VectorType::get({minElements}, valType.getElementType(), {true});
  Value undef = LLVM::UndefOp::create(builder, loc, scalableType);
  Value zero = LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                        builder.getI64IntegerAttr(0));

  SmallVector<Value> args = {undef, flatVal, zero};
  auto castOp = LLVM::CallIntrinsicOp::create(
      builder, loc, static_cast<Type>(scalableType),
      builder.getStringAttr("llvm.vector.insert"), ValueRange(args));
  return castOp.getResult(0);
}

bool hasZvtFeature(mlir::FunctionOpInterface funcOp) {
  auto targetAttr =
      mlir::iree_compiler::IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!targetAttr) return false;

  auto config = targetAttr.getConfiguration();
  if (!config) return false;

  auto attr = config.getAs<mlir::StringAttr>("cpu_features");
  if (!attr) return false;

  llvm::StringRef cpuFeatures = attr.getValue();
  return cpuFeatures.contains("+zvtbase");
}

void insertMatrixConfiguration(mlir::FunctionOpInterface funcOp,
                               OpBuilder &builder, int64_t M, int64_t N,
                               Type inputElTy) {
  auto concreteFuncOp = dyn_cast<func::FuncOp>(funcOp.getOperation());
  if (!concreteFuncOp) return;

  OpBuilder::InsertionGuard guard(builder);
  Block &entryBlock = concreteFuncOp.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);
  Location loc = funcOp.getLoc();

  auto i32Type = builder.getI32Type();

  int64_t K = 1;
  int64_t vtype = 16;  // default SEW=32
  int64_t mtwiden = 1;

  if (inputElTy.isF32()) {
    K = 1;
    vtype = 16;  // SEW=32, LMUL=1
    mtwiden = 1;
  } else if (inputElTy.isBF16()) {
    K = 1;
    vtype = 8;    // SEW=16, LMUL=1
    mtwiden = 2;  // 2x widening
  } else if (inputElTy.isInteger(8)) {
    K = 1;
    vtype = 0;    // SEW=8, LMUL=1
    mtwiden = 3;  // 4x widening
  } else {
    llvm_unreachable("unsupported input element type for Matrix");
  }

  // Compute mtype: tm = M, tk = K, mtwiden
  int64_t mtype = (M << 10) | (K << 5) | mtwiden;
  LLVM_DEBUG(llvm::dbgs() << "MATRIX_CONF: M=" << M << ", N=" << N
                          << ", K=" << K << ", mtwiden=" << mtwiden
                          << ", mtype=" << mtype << "\n");
  Value mtypeVal =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(mtype));
  Value vtypeVal =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(vtype));
  // tn = N
  Value tnVal =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(N));

  LLVM::CallIntrinsicOp::create(
      builder, loc, builder.getStringAttr("llvm.riscv.zvt.msetmtype"),
      ValueRange{mtypeVal, vtypeVal});
  LLVM::CallIntrinsicOp::create(builder, loc, i32Type,
                                builder.getStringAttr("llvm.riscv.zvt.msettn"),
                                ValueRange{tnVal});
}

struct CoralNPUVectorToMatrixPass
    : public impl::CoralNPUVectorToMatrixBase<CoralNPUVectorToMatrixPass> {
  using CoralNPUVectorToMatrixBase::CoralNPUVectorToMatrixBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!hasZvtFeature(funcOp)) {
      return;
    }
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    LLVM_DEBUG(llvm::dbgs() << "Running CoralNPUVectorToMatrixPass on: "
                            << funcOp.getName() << "\n");

    SmallVector<AccumulatorChain> chains;
    funcOp.walk([&](vector::TransferWriteOp writeOp) {
      if (auto chain = analyzeChain(writeOp)) {
        chains.push_back(*chain);
      }
    });

    bool rewritten = false;
    int64_t nextTileId = 0;
    int64_t configM = 4;
    int64_t configN = 4;
    Type inputElTy = nullptr;
    for (auto &chain : chains) {
      auto vecType = cast<VectorType>(chain.writeback.getVector().getType());
      configM = vecType.getDimSize(0);
      configN = vecType.getDimSize(1);
      inputElTy = chain.inputElTy;

      // Statically assign tile ID
      chain.tileId = nextTileId++;
      if (chain.tileId >= 16) {
        funcOp.emitError("Matrix only supports up to 16 tile registers");
        return signalPassFailure();
      }

      Location loc = chain.writeback.getLoc();

      auto readOp = chain.initVal.getDefiningOp<vector::TransferReadOp>();
      if (!readOp && !isConstantZero(chain.initVal)) {
        funcOp.emitWarning(
            "Matrix lowering only supports constant zero or transfer_read "
            "initialization, skipping");
        continue;
      }

      // 1. Generate initialization before the loop (or before the first update)
      Operation *insertPoint = nullptr;
      if (auto forOp =
              chain.writeback.getVector().getDefiningOp<scf::ForOp>()) {
        insertPoint = forOp;
      } else {
        insertPoint = chain.updates.front();
      }

      builder.setInsertionPoint(insertPoint);
      Value tileIdVal =
          LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                   builder.getI32IntegerAttr(chain.tileId));

      if (readOp) {
        auto moduleOp = funcOp->getParentOfType<ModuleOp>();
        func::FuncOp loadFunc =
            getOrDeclareLoadFunc(moduleOp, builder, readOp.getBase().getType());
        auto memrefType = cast<MemRefType>(readOp.getBase().getType());

        // 1. Memory space cast to 0
        auto spaceCastType =
            MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                            memrefType.getLayout(), /*memorySpace=*/0);
        Value spaceCastMemref = memref::MemorySpaceCastOp::create(
            builder, loc, spaceCastType, readOp.getBase());

        // 2. Create 2D subview
        int64_t rank = memrefType.getRank();
        assert(rank >= 2 && "expected read memref rank to be at least 2");

        SmallVector<OpFoldResult> offsets;
        for (auto index : readOp.getIndices()) {
          offsets.push_back(index);
        }

        SmallVector<OpFoldResult> sizes;
        for (int64_t i = 0; i < rank - 2; ++i) {
          sizes.push_back(builder.getIndexAttr(1));
        }
        sizes.push_back(builder.getIndexAttr(configM));
        sizes.push_back(builder.getIndexAttr(configN));

        SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

        SmallVector<int64_t> resultShape = {configM, configN};
        auto subViewType =
            cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
                resultShape, spaceCastType, offsets, sizes, strides));

        Value subViewMemref =
            memref::SubViewOp::create(builder, loc, subViewType,
                                      spaceCastMemref, offsets, sizes, strides);

        // 3. Cast to generic 2D
        auto ctx = builder.getContext();
        auto dynamicVal = ShapedType::kDynamic;
        auto layout =
            StridedLayoutAttr::get(ctx, dynamicVal, {dynamicVal, dynamicVal});
        auto genericMemrefType = MemRefType::get(
            {dynamicVal, dynamicVal}, memrefType.getElementType(), layout);
        Value castMemref = memref::CastOp::create(
            builder, loc, genericMemrefType, subViewMemref);

        // 4. Call load with offset 0, 0
        Value zeroIndex = arith::ConstantIndexOp::create(builder, loc, 0);
        func::CallOp::create(
            builder, loc, loadFunc,
            ValueRange{castMemref, zeroIndex, zeroIndex, tileIdVal});
      } else {
        // Constant zero: generate vtzero
        SmallVector<Value> vtzeroArgs = {tileIdVal};
        LLVM::CallIntrinsicOp::create(
            builder, loc, builder.getStringAttr("llvm.riscv.zvt.vtzero"),
            ValueRange(vtzeroArgs));
      }

      // 2. Rewrite loop if present
      if (auto forOp =
              chain.writeback.getVector().getDefiningOp<scf::ForOp>()) {
        // Find which iter_arg is the accumulator
        unsigned accIdx = 0;
        for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
          if (forOp.getInitArgs()[i] == chain.initVal) {
            accIdx = i;
            break;
          }
        }

        // Create new loop without the accumulator
        builder.setInsertionPoint(forOp);
        auto inits = llvm::to_vector(forOp.getInitArgs());
        inits.erase(inits.begin() + accIdx);
        auto newForOp =
            scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                               forOp.getUpperBound(), forOp.getStep(), inits);

        // Move body and replace contract
        Block *newBody = newForOp.getBody();
        if (!newBody->empty() && isa<scf::YieldOp>(newBody->back())) {
          newBody->back().erase();
        }
        Block *oldBody = forOp.getBody();

        IRMapping mapping;
        mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
        for (unsigned i = 0; i < inits.size(); ++i) {
          unsigned oldIdx = i < accIdx ? i + 1 : i + 2;
          mapping.map(oldBody->getArgument(oldIdx),
                      newBody->getArgument(i + 1));
        }

        builder.setInsertionPointToStart(newBody);
        for (auto &op : oldBody->without_terminator()) {
          auto contractOp = dyn_cast<vector::ContractionOp>(&op);
          if (contractOp && llvm::is_contained(chain.updates, contractOp)) {
            Value origLhs = traceToSourceValue(contractOp.getLhs());
            Value origRhs = traceToSourceValue(contractOp.getRhs());
            Value lhs = mapping.lookupOrDefault(origLhs);
            Value rhs = mapping.lookupOrDefault(origRhs);
            Value lhsScalable =
                castToScalable(builder, contractOp.getLoc(), lhs);
            Value rhsScalable =
                castToScalable(builder, contractOp.getLoc(), rhs);

            std::string intrinsicName;
            if (inputElTy.isF32()) {
              intrinsicName = "llvm.riscv.zvt.vtfmm.tvv";
            } else if (inputElTy.isF16()) {
              intrinsicName = "llvm.riscv.zvt.vtfmm.tvv";
            } else if (inputElTy.isBF16()) {
              intrinsicName = "llvm.riscv.zvt.vtfmm.alt.tvv";
            } else if (inputElTy.isInteger(8)) {
              intrinsicName = "llvm.riscv.zvt.vtmms.tvv";
            }

            SmallVector<Value> vtfmmArgs = {tileIdVal, lhsScalable,
                                            rhsScalable};
            LLVM::CallIntrinsicOp::create(builder, contractOp.getLoc(),
                                          builder.getStringAttr(intrinsicName),
                                          ValueRange(vtfmmArgs));
          } else {
            builder.clone(op, mapping);
          }
        }

        // Terminator
        auto yieldOp = cast<scf::YieldOp>(oldBody->getTerminator());
        auto yieldOperands = llvm::to_vector(yieldOp.getOperands());
        yieldOperands.erase(yieldOperands.begin() + accIdx);
        scf::YieldOp::create(builder, yieldOp.getLoc(), yieldOperands);

        // Replace uses
        for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
          if (i < accIdx) {
            forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
          } else if (i > accIdx) {
            forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i - 1));
          } else {
            builder.setInsertionPoint(forOp);
            Value dummyVec = arith::ConstantOp::create(
                builder, forOp.getLoc(),
                builder.getZeroAttr(forOp.getResult(i).getType()));
            forOp.getResult(i).replaceAllUsesWith(dummyVec);
          }
        }
        forOp.erase();
      } else {
        // No loop, just lower contract
        auto contractOp = cast<vector::ContractionOp>(chain.updates.front());
        builder.setInsertionPoint(contractOp);
        Value lhsScalable =
            castToScalable(builder, contractOp.getLoc(), contractOp.getLhs());
        Value rhsScalable =
            castToScalable(builder, contractOp.getLoc(), contractOp.getRhs());

        std::string intrinsicName;
        if (inputElTy.isF32()) {
          intrinsicName = "llvm.riscv.zvt.vtfmm.tvv";
        } else if (inputElTy.isF16()) {
          intrinsicName = "llvm.riscv.zvt.vtfmm.tvv";
        } else if (inputElTy.isBF16()) {
          intrinsicName = "llvm.riscv.zvt.vtfmm.alt.tvv";
        } else if (inputElTy.isInteger(8)) {
          intrinsicName = "llvm.riscv.zvt.vtmms.tvv";
        }

        SmallVector<Value> vtfmmArgs = {tileIdVal, lhsScalable, rhsScalable};
        LLVM::CallIntrinsicOp::create(builder, contractOp.getLoc(),
                                      builder.getStringAttr(intrinsicName),
                                      ValueRange(vtfmmArgs));
        Value dummyVec = arith::ConstantOp::create(
            builder, contractOp.getLoc(),
            builder.getZeroAttr(contractOp.getType()));
        contractOp.replaceAllUsesWith(dummyVec);
        contractOp.erase();
      }

      // 3. Replace transfer_write with dummy call
      builder.setInsertionPoint(chain.writeback);
      func::FuncOp writebackFunc = getOrDeclareWritebackFunc(
          moduleOp, builder, chain.writeback.getBase().getType());

      auto memrefType = cast<MemRefType>(chain.writeback.getBase().getType());

      // 1. Memory space cast to 0
      auto spaceCastType =
          MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                          memrefType.getLayout(), /*memorySpace=*/0);
      Value spaceCastMemref = memref::MemorySpaceCastOp::create(
          builder, loc, spaceCastType, chain.writeback.getBase());

      // 2. Create 2D subview to drop unit dimensions and handle offsets
      int64_t rank = memrefType.getRank();
      assert(rank >= 2 && "expected writeback memref rank to be at least 2");

      SmallVector<OpFoldResult> offsets;
      for (auto index : chain.writeback.getIndices()) {
        offsets.push_back(index);
      }

      SmallVector<OpFoldResult> sizes;
      for (int64_t i = 0; i < rank - 2; ++i) {
        sizes.push_back(builder.getIndexAttr(1));
      }
      sizes.push_back(builder.getIndexAttr(configM));
      sizes.push_back(builder.getIndexAttr(configN));

      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

      SmallVector<int64_t> resultShape = {configM, configN};
      auto subViewType =
          cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
              resultShape, spaceCastType, offsets, sizes, strides));

      Value subViewMemref = memref::SubViewOp::create(
          builder, loc, subViewType, spaceCastMemref, offsets, sizes, strides);

      // 3. Cast to generic 2D (dynamic strided)
      auto ctx = builder.getContext();
      auto dynamicVal = ShapedType::kDynamic;
      auto layout =
          StridedLayoutAttr::get(ctx, dynamicVal, {dynamicVal, dynamicVal});
      auto genericMemrefType = MemRefType::get(
          {dynamicVal, dynamicVal}, memrefType.getElementType(), layout);
      Value castMemref = memref::CastOp::create(builder, loc, genericMemrefType,
                                                subViewMemref);

      // 4. Call writeback with offset 0, 0 because subview is already offset
      Value zeroIndex = arith::ConstantIndexOp::create(builder, loc, 0);
      func::CallOp::create(
          builder, loc, writebackFunc,
          ValueRange{castMemref, zeroIndex, zeroIndex, tileIdVal});

      chain.writeback.erase();
      rewritten = true;
    }
    if (rewritten) {
      assert(inputElTy && "expected inputElTy to be set");
      insertMatrixConfiguration(funcOp, builder, configM, configN, inputElTy);
    }
  }
};

}  // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCoralNPUVectorToMatrixPass() {
  return std::make_unique<CoralNPUVectorToMatrixPass>();
}

}  // namespace mlir::coralnpu_compiler
