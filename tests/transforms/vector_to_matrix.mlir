// RUN: %iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(coralnpu-vector-to-matrix)))))" %s | FileCheck %s

#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {
  cpu_features = "+zvtbase"
}>

// CHECK-LABEL: hal.executable private @matmul
module {
  hal.executable private @matmul {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        // CHECK: func.func private @coralnpu.matrix.writeback.w32
        // CHECK: func.func private @coralnpu.matrix.load.w32
        
        // CHECK-LABEL: func.func @matmul
        func.func @matmul(%arg0: memref<4x1xf32>, %arg1: memref<1x4xf32>, %arg2: memref<4x4xf32>) {
          // CHECK-DAG: %[[MTYPE:.+]] = arith.constant 4129 : i32
          // CHECK-DAG: %[[VTYPE:.+]] = arith.constant 16 : i32
          // CHECK-DAG: %[[TN:.+]] = arith.constant 4 : i32
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.msetmtype"(%[[MTYPE]], %[[VTYPE]])
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.msettn"(%[[TN]])
          
          %c0 = arith.constant 0 : index
          %cst_0 = arith.constant 0.0 : f32
          
          %lhs = vector.transfer_read %arg0[%c0, %c0], %cst_0 : memref<4x1xf32>, vector<4x1xf32>
          %rhs = vector.transfer_read %arg1[%c0, %c0], %cst_0 : memref<1x4xf32>, vector<1x4xf32>
          %acc = vector.transfer_read %arg2[%c0, %c0], %cst_0 : memref<4x4xf32>, vector<4x4xf32>
          
          // CHECK: %[[TILE_ID:.+]] = llvm.mlir.constant(0 : i32) : i32
          // CHECK: %[[MEMCAST_LOAD:.+]] = memref.memory_space_cast %arg2
          // CHECK: %[[SUBVIEW_LOAD:.+]] = memref.subview %[[MEMCAST_LOAD]]
          // CHECK: %[[CAST_LOAD:.+]] = memref.cast %[[SUBVIEW_LOAD]]
          // CHECK: call @coralnpu.matrix.load.w32(%[[CAST_LOAD]], %{{.+}}, %{{.+}}, %[[TILE_ID]])
          
          // CHECK: %[[LHS_FLAT:.+]] = vector.shape_cast %{{.+}} : vector<4x1xf32> to vector<4xf32>
          // CHECK: %[[LHS_INS:.+]] = llvm.call_intrinsic "llvm.vector.insert"(%{{.+}}, %[[LHS_FLAT]], %{{.+}})
          // CHECK: %[[RHS_FLAT:.+]] = vector.shape_cast %{{.+}} : vector<1x4xf32> to vector<4xf32>
          // CHECK: %[[RHS_INS:.+]] = llvm.call_intrinsic "llvm.vector.insert"(%{{.+}}, %[[RHS_FLAT]], %{{.+}})
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.vtfmm.tvv"(%[[TILE_ID]], %[[LHS_INS]], %[[RHS_INS]])
          
          %res = vector.contract {
            indexing_maps = [
              affine_map<(d0, d1, d2) -> (d0, d2)>,
              affine_map<(d0, d1, d2) -> (d2, d1)>,
              affine_map<(d0, d1, d2) -> (d0, d1)>
            ],
            iterator_types = ["parallel", "parallel", "reduction"]
          } %lhs, %rhs, %acc : vector<4x1xf32>, vector<1x4xf32> into vector<4x4xf32>
          
          // CHECK: %[[MEMCAST_STORE:.+]] = memref.memory_space_cast %arg2
          // CHECK: %[[SUBVIEW_STORE:.+]] = memref.subview %[[MEMCAST_STORE]]
          // CHECK: %[[CAST_STORE:.+]] = memref.cast %[[SUBVIEW_STORE]]
          // CHECK: call @coralnpu.matrix.writeback.w32(%[[CAST_STORE]], %{{.+}}, %{{.+}}, %[[TILE_ID]])
          
          vector.transfer_write %res, %arg2[%c0, %c0] : vector<4x4xf32>, memref<4x4xf32>
          return
        }
      }
    }
  }
}
