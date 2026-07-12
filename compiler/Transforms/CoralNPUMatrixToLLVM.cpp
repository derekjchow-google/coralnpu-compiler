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

// MLIR headers
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "coralnpu-matrix-to-llvm"

namespace mlir::coralnpu_compiler {

#define GEN_PASS_DEF_CORALNPUMATRIXTOLLVM
#include "compiler/Transforms/Passes.h.inc"

namespace {

struct CoralNPUMatrixToLLVMPass
    : public impl::CoralNPUMatrixToLLVMBase<CoralNPUMatrixToLLVMPass> {
  using CoralNPUMatrixToLLVMBase::CoralNPUMatrixToLLVMBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    LLVM_DEBUG(llvm::dbgs() << "Running CoralNPUMatrixToLLVMPass\n");

    SmallVector<LLVM::CallOp> writebackCalls;
    SmallVector<LLVM::CallOp> loadCalls;
    moduleOp.walk([&](LLVM::CallOp callOp) {
      if (auto callee = callOp.getCallee()) {
        if (callee->starts_with("coralnpu.matrix.writeback.w")) {
          writebackCalls.push_back(callOp);
        } else if (callee->starts_with("coralnpu.matrix.load.w")) {
          loadCalls.push_back(callOp);
        }
      }
    });

    auto lowerMatrixMemOp = [&](LLVM::CallOp callOp) {
      Location loc = callOp.getLoc();
      builder.setInsertionPoint(callOp);

      if (callOp.getNumOperands() != 10) {
        callOp.emitOpError("expected 10 operands after LLVM conversion");
        return failure();
      }

      Value alignedPtr = callOp.getOperand(1);
      Value offset = callOp.getOperand(2);
      Value stride0 = callOp.getOperand(5);
      Value stride1 = callOp.getOperand(6);
      Value idx0 = callOp.getOperand(7);
      Value idx1 = callOp.getOperand(8);
      Value tileId = callOp.getOperand(9);

      // total_offset = offset + idx0 * stride0 + idx1 * stride1
      Value val0 = LLVM::MulOp::create(builder, loc, idx0, stride0);
      Value val1 = LLVM::MulOp::create(builder, loc, idx1, stride1);
      Value totalOffset = LLVM::AddOp::create(builder, loc, offset, val0);
      totalOffset = LLVM::AddOp::create(builder, loc, totalOffset, val1);

      Type ptrType = alignedPtr.getType();
      Type indexType = stride0.getType();

      // Get constant size0 (rows)
      int64_t numRows = 4;
      if (auto constOp =
              callOp.getOperand(3).getDefiningOp<LLVM::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          numRows = intAttr.getInt();
        }
      }

      StringRef callee = callOp.getCallee().value();
      bool isLoad = callee.contains("load");
      unsigned width = 32;
      StringRef widthStr = isLoad ? callee.substr(19) : callee.substr(24);
      widthStr.getAsInteger(10, width);

      Type gepElementType;
      std::string intrinsicName;
      if (width == 8) {
        gepElementType = builder.getI8Type();
        intrinsicName =
            isLoad ? "llvm.riscv.zvt.vtle8" : "llvm.riscv.zvt.vtse8";
      } else if (width == 16) {
        gepElementType = builder.getI16Type();
        intrinsicName =
            isLoad ? "llvm.riscv.zvt.vtle16" : "llvm.riscv.zvt.vtse16";
      } else if (width == 32) {
        gepElementType = builder.getF32Type();
        intrinsicName =
            isLoad ? "llvm.riscv.zvt.vtle32" : "llvm.riscv.zvt.vtse32";
      } else if (width == 64) {
        gepElementType = builder.getI64Type();
        intrinsicName =
            isLoad ? "llvm.riscv.zvt.vtle64" : "llvm.riscv.zvt.vtse64";
      } else {
        callOp.emitOpError("unsupported width");
        return failure();
      }

      for (int64_t row = 0; row < numRows; ++row) {
        // 1. Compute tss descriptor: (tileId << 27) | row
        Value shiftAmt = LLVM::ConstantOp::create(
            builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(27));
        Value shiftedTileId =
            LLVM::ShlOp::create(builder, loc, tileId, shiftAmt);
        Value rowVal = LLVM::ConstantOp::create(
            builder, loc, builder.getI32Type(), builder.getI32IntegerAttr(row));
        Value tss = LLVM::OrOp::create(builder, loc, shiftedTileId, rowVal);

        // 2. Compute row offset: totalOffset + row * stride0
        Value rowValIdx = LLVM::ConstantOp::create(
            builder, loc, indexType, builder.getIntegerAttr(indexType, row));
        Value rowOffset = LLVM::MulOp::create(builder, loc, rowValIdx, stride0);
        Value finalOffset =
            LLVM::AddOp::create(builder, loc, totalOffset, rowOffset);

        // 3. GEP
        Value gep = LLVM::GEPOp::create(builder, loc, ptrType, gepElementType,
                                        alignedPtr, ValueRange{finalOffset});

        // 4. Call intrinsic
        SmallVector<Value> intrinsicArgs = {tss, gep};
        LLVM::CallIntrinsicOp::create(builder, loc,
                                      builder.getStringAttr(intrinsicName),
                                      ValueRange(intrinsicArgs));
      }

      // Erase the call
      callOp.erase();
      return success();
    };

    for (auto callOp : writebackCalls) {
      if (failed(lowerMatrixMemOp(callOp))) {
        return signalPassFailure();
      }
    }

    for (auto callOp : loadCalls) {
      if (failed(lowerMatrixMemOp(callOp))) {
        return signalPassFailure();
      }
    }

    // Also erase the dummy function declarations if they exist
    SmallVector<LLVM::LLVMFuncOp> funcsToErase;
    moduleOp.walk([&](LLVM::LLVMFuncOp funcOp) {
      if (funcOp.getName().starts_with("coralnpu.matrix.writeback") ||
          funcOp.getName().starts_with("coralnpu.matrix.load")) {
        funcsToErase.push_back(funcOp);
      }
    });
    for (auto funcOp : funcsToErase) {
      funcOp.erase();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createCoralNPUMatrixToLLVMPass() {
  return std::make_unique<CoralNPUMatrixToLLVMPass>();
}

}  // namespace mlir::coralnpu_compiler
