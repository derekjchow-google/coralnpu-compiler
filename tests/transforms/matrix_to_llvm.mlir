// RUN: %iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(coralnpu-matrix-to-llvm))))" %s | FileCheck %s

#target = #hal.executable.target<"coralnpu", "coralnpu-elf", {cpu_features = "+zvtbase"}>

// CHECK-LABEL: hal.executable private @matmul
module {
  hal.executable private @matmul {
    hal.executable.variant public @coralnpu_elf target(#target) {
      builtin.module {
        llvm.func external @coralnpu.matrix.load.w32(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i32)
        llvm.func external @coralnpu.matrix.writeback.w32(!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i32)
        
        // CHECK-LABEL: llvm.func @matmul
        llvm.func @matmul(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
          %c0 = llvm.mlir.constant(0 : i64) : i64
          %c4 = llvm.mlir.constant(4 : i64) : i64
          %c1 = llvm.mlir.constant(1 : i64) : i64
          %tileId = llvm.mlir.constant(0 : i32) : i32
          
          // CHECK: %[[TILE_ID:.+]] = llvm.mlir.constant(0 : i32) : i32
          
          // CHECK: %[[TSS0:.+]] = llvm.or %{{.+}}, %{{.+}} : i32
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.vtle32"(%[[TSS0]], %{{.+}})
          // CHECK: %[[TSS1:.+]] = llvm.or %{{.+}}, %{{.+}} : i32
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.vtle32"(%[[TSS1]], %{{.+}})
          // CHECK: %[[TSS2:.+]] = llvm.or %{{.+}} : i32
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.vtle32"(%[[TSS2]], %{{.+}})
          // CHECK: %[[TSS3:.+]] = llvm.or %{{.+}} : i32
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.vtle32"(%[[TSS3]], %{{.+}})
          llvm.call @coralnpu.matrix.load.w32(%arg0, %arg0, %c0, %c4, %c4, %c4, %c1, %c0, %c0, %tileId) 
            : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i32) -> ()

          // CHECK: %[[TSS0_W:.+]] = llvm.or %{{.+}}, %{{.+}} : i32
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.vtse32"(%[[TSS0_W]], %{{.+}})
          // CHECK: %[[TSS1_W:.+]] = llvm.or %{{.+}} : i32
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.vtse32"(%[[TSS1_W]], %{{.+}})
          // CHECK: %[[TSS2_W:.+]] = llvm.or %{{.+}} : i32
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.vtse32"(%[[TSS2_W]], %{{.+}})
          // CHECK: %[[TSS3_W:.+]] = llvm.or %{{.+}} : i32
          // CHECK: llvm.call_intrinsic "llvm.riscv.zvt.vtse32"(%[[TSS3_W]], %{{.+}})
          llvm.call @coralnpu.matrix.writeback.w32(%arg2, %arg2, %c0, %c4, %c4, %c4, %c1, %c0, %c0, %tileId) 
            : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i32) -> ()
          llvm.return
        }
        
        // CHECK-NOT: llvm.func external @coralnpu.matrix.load.w32
        // CHECK-NOT: llvm.func external @coralnpu.matrix.writeback.w32
      }
    }
  }
}
