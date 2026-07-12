#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np

# MLIR Template for mmt4d
MLIR_TEMPLATE = """
#config = #iree_cpu.lowering_config<cache_parallel = [1, 1, 0, 8, 8, 0], cache_reduction = [0, 0, 1, 0, 0, 8], distribution = [1, 1, 0, 8, 8, 0], vector_common_parallel = [1, 1, 0, 4, 4, 0], vector_reduction = [0, 0, 1, 0, 0, 1]>
#translation = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>
#compilation_info = #iree_codegen.compilation_info<lowering_config = #config, translation_info = #translation>

func.func @main(%arg0: tensor<1x1x16x16x{in_type}>, %arg1: tensor<1x1x16x16x{in_type}>, %arg2: tensor<1x1x16x16x{acc_type}>) -> tensor<1x1x16x16x{acc_type}> {{
  %1 = linalg.mmt4d {{compilation_info = #compilation_info}}
    ins(%arg0, %arg1 : tensor<1x1x16x16x{in_type}>, tensor<1x1x16x16x{in_type}>)
    outs(%arg2 : tensor<1x1x16x16x{acc_type}>) -> tensor<1x1x16x16x{acc_type}>
  return %1 : tensor<1x1x16x16x{acc_type}>
}}
"""

SIZE = 256  # 16x16


def generate_data(dtype, tmpdir):
  np.random.seed(42)

  if dtype == 'f32':
    lhs = np.random.randint(-3, 4, SIZE).astype(np.float32)
    rhs = np.random.randint(-3, 4, SIZE).astype(np.float32)
    acc = np.random.randint(-3, 4, SIZE).astype(np.float32)
  elif dtype == 'f16':
    lhs = np.random.randint(-3, 4, SIZE).astype(np.float16)
    rhs = np.random.randint(-3, 4, SIZE).astype(np.float16)
    acc = np.random.randint(-3, 4, SIZE).astype(np.float32)  # acc is f32
  elif dtype == 'bf16':
    lhs_f32 = np.random.randint(-3, 4, SIZE).astype(np.float32)
    rhs_f32 = np.random.randint(-3, 4, SIZE).astype(np.float32)

    def to_bf16(x):
      x_u32 = x.view(np.uint32)
      return (x_u32 & 0xFFFF0000).view(np.float32)

    lhs = to_bf16(lhs_f32)
    rhs = to_bf16(rhs_f32)
    acc = np.random.randint(-3, 4, SIZE).astype(np.float32)
  elif dtype == 'i8':
    lhs = np.random.randint(-3, 4, SIZE).astype(np.int8)
    rhs = np.random.randint(-3, 4, SIZE).astype(np.int8)
    acc = np.random.randint(-3, 4, SIZE).astype(np.int32)
  else:
    raise ValueError(f"Unsupported dtype: {dtype}")

  lhs_bin = os.path.join(tmpdir, f"lhs_{dtype}.bin")
  rhs_bin = os.path.join(tmpdir, f"rhs_{dtype}.bin")
  acc_bin = os.path.join(tmpdir, f"acc_{dtype}.bin")

  if dtype == 'bf16':
    lhs_u16 = (lhs.view(np.uint32) >> 16).astype(np.uint16)
    rhs_u16 = (rhs.view(np.uint32) >> 16).astype(np.uint16)
    lhs_u16.tofile(lhs_bin)
    rhs_u16.tofile(rhs_bin)
  else:
    lhs.tofile(lhs_bin)
    rhs.tofile(rhs_bin)
  acc.tofile(acc_bin)

  return lhs_bin, rhs_bin, acc_bin


def write_mlir_file(dtype, tmpdir):
  in_type = dtype
  acc_type = 'f32' if dtype in ['f32', 'f16', 'bf16'] else 'i32'
  content = MLIR_TEMPLATE.format(in_type=in_type, acc_type=acc_type)
  mlir_file = os.path.join(tmpdir, f"mmt4d_{dtype}.mlir")
  with open(mlir_file, "w") as f:
    f.write(content)
  return mlir_file


def compile_model(compile_tool, mlir_file, vmfb_file, features):
  cmd = [
      compile_tool, mlir_file, "-o", vmfb_file,
      "--iree-hal-target-backends=coralnpu",
      f"--coralnpu-target-cpu-features={features}", "--mlir-disable-threading"
  ]
  res = subprocess.run(cmd, capture_output=True, text=True)
  if res.returncode != 0:
    print(f"Compilation failed for {mlir_file}:")
    print(res.stderr)
    raise RuntimeError("Compilation failed")


def run_model(run_tool, vmfb, dtype, lhs_bin, rhs_bin, acc_bin):
  input_types = {
      'f32': ('f32', 'f32', 'f32'),
      'f16': ('f16', 'f16', 'f32'),
      'bf16': ('bf16', 'bf16', 'f32'),
      'i8': ('i8', 'i8', 'i32')
  }
  t0, t1, t2 = input_types[dtype]

  cmd = [
      run_tool, f"--module={vmfb}", "--device=coralnpu", "--function=main",
      f"--input=1x1x16x16x{t0}=@{lhs_bin}",
      f"--input=1x1x16x16x{t1}=@{rhs_bin}",
      f"--input=1x1x16x16x{t2}=@{acc_bin}", "--output_max_element_count=20000"
  ]
  res = subprocess.run(cmd, capture_output=True, text=True)
  if res.returncode != 0:
    print(f"Execution failed for {vmfb}:")
    print(res.stderr)
    raise RuntimeError("Execution failed")
  return res.stdout


def parse_output(out_str):
  equal_idx = out_str.find('=')
  if equal_idx == -1:
    raise ValueError(f"Could not find '=' in output:\n{out_str}")
  start = out_str.find('[', equal_idx)
  end = out_str.rfind(']')
  if start == -1 or end == -1 or end <= start:
    raise ValueError(
        f"Could not find bracketed tensor data in output:\n{out_str}")
  data_str = out_str[start + 1:end]
  clean = re.sub(r'[\[\]]', ' ', data_str)
  elements = clean.split()
  return [float(x) for x in elements]


def find_tools(build_dir):
  compile_tool = None
  run_tool = None
  script_dir = os.path.dirname(os.path.abspath(__file__))

  if not build_dir:
    # Try Bazel auto-detect first
    bazel_bin = os.path.join(script_dir, "../bazel-bin")
    if os.path.exists(bazel_bin):
      import glob
      c_tool = os.path.join(bazel_bin, "compiler/tools/coralnpu-compile")
      r_tools = glob.glob(
          os.path.join(bazel_bin, "external/*/tools/iree-run-module"))
      if os.path.exists(c_tool) and r_tools:
        print(f"Auto-detected Bazel build at {bazel_bin}")
        return c_tool, r_tools[0]

    # Fallback to CMake auto-detect
    possible_dirs = [
        os.path.join(script_dir, "../build-out"),
        os.path.join(script_dir, "../build"),
        os.path.join(script_dir, "../../coralnpu-compiler-build"),
        os.path.join(script_dir, "../coralnpu-compiler-build"),
    ]
    for d in possible_dirs:
      c_tool = os.path.join(d, "third_party/iree/tools/coralnpu-compile")
      r_tool = os.path.join(d, "third_party/iree/tools/iree-run-module")
      if os.path.exists(c_tool) and os.path.exists(r_tool):
        build_dir = d
        compile_tool = c_tool
        run_tool = r_tool
        break
  else:
    # If build_dir is specified, check if it matches Bazel structure
    c_tool = os.path.join(build_dir, "compiler/tools/coralnpu-compile")
    if os.path.exists(c_tool):
      import glob
      r_tools = glob.glob(
          os.path.join(build_dir, "external/*/tools/iree-run-module"))
      if r_tools:
        compile_tool = c_tool
        run_tool = r_tools[0]

    if not compile_tool:
      # Try CMake structure
      compile_tool = os.path.join(build_dir,
                                  "third_party/iree/tools/coralnpu-compile")
      run_tool = os.path.join(build_dir,
                              "third_party/iree/tools/iree-run-module")

  if not compile_tool or not run_tool or not os.path.exists(
      compile_tool) or not os.path.exists(run_tool):
    print("Error: Could not find build directory containing tools.")
    print("Please specify a valid build directory with --build-dir")
    if build_dir:
      print(f"Checked: {build_dir}")
      print(f"Expected: {compile_tool} and {run_tool}")
    sys.exit(1)

  return compile_tool, run_tool


def verify_dtype(dtype,
                 compile_tool,
                 run_tool,
                 verbose=False,
                 keep_temps=False):
  print(f"\n==================================================")
  print(f" Verifying Data Type: {dtype.upper()}")
  print(f"==================================================")

  # Configure features
  if dtype == 'f32':
    ref_features = "+m,+a,+f,+d,+v,+zve32f,+zvl128b"
    matrix_features = ("+m,+a,+f,+d,+v,+zve32f,+zvl128b,"
                       "+zvtbase,+zvtf32f32mm")
  elif dtype == 'f16':
    ref_features = "+m,+a,+f,+d,+v,+zve32f,+zvl128b,+zvfhmin"
    # FP16 falls back to Zve on Matrix configuration
    matrix_features = ("+m,+a,+f,+d,+v,+zve32f,+zvl128b,"
                       "+zvtbase,+zvtf16f32mm,+zvfhmin")
  elif dtype == 'bf16':
    ref_features = "+m,+a,+f,+d,+v,+zve32f,+zvl128b,+zvfbfmin"
    matrix_features = ("+m,+a,+f,+d,+v,+zve32f,+zvl128b,"
                       "+zvtbase,+zvtf16f32mm,+zvfbfmin")
  elif dtype == 'i8':
    ref_features = "+m,+a,+f,+d,+v,+zve32f,+zvl128b"
    matrix_features = ("+m,+a,+f,+d,+v,+zve32f,+zvl128b,"
                       "+zvtbase,+zvti8i32mm")
  else:
    raise ValueError(f"Unknown dtype: {dtype}")

  # Use a temporary directory for this run
  temp_dir_manager = tempfile.TemporaryDirectory(
      prefix=f"matrix_verify_{dtype}_")
  tmpdir = temp_dir_manager.name

  try:
    lhs_bin, rhs_bin, acc_bin = generate_data(dtype, tmpdir)
    mlir_file = write_mlir_file(dtype, tmpdir)

    ref_vmfb = os.path.join(tmpdir, f"matmul_ref_{dtype}.vmfb")
    matrix_vmfb = os.path.join(tmpdir, f"matmul_matrix_{dtype}.vmfb")

    print("Compiling models...")
    compile_model(compile_tool, mlir_file, ref_vmfb, ref_features)
    compile_model(compile_tool, mlir_file, matrix_vmfb, matrix_features)

    print("Running models on simulator...")
    out_ref = run_model(run_tool, ref_vmfb, dtype, lhs_bin, rhs_bin, acc_bin)
    out_matrix = run_model(run_tool, matrix_vmfb, dtype, lhs_bin, rhs_bin,
                           acc_bin)

    val_ref = parse_output(out_ref)
    val_matrix = parse_output(out_matrix)

    if len(val_ref) != len(val_matrix):
      print(
          f"ERROR: Dimension mismatch! Ref: {len(val_ref)}, Matrix: {len(val_matrix)}"
      )
      return False

    if verbose:
      print("\nRef Output (16x16):")
      for r in range(16):
        row_vals = val_ref[r * 16:(r + 1) * 16]
        print("  " + " ".join(f"{x:10.6f}" for x in row_vals))

      print("\nMatrix Output (16x16):")
      for r in range(16):
        row_vals = val_matrix[r * 16:(r + 1) * 16]
        print("  " + " ".join(f"{x:10.6f}" for x in row_vals))

    mismatches = 0
    max_diff = 0.0
    tol = 1e-4 if dtype in ['f32', 'f16', 'bf16'] else 0

    for r, m in zip(val_ref, val_matrix):
      diff = abs(r - m)
      if diff > max_diff:
        max_diff = diff
      if diff > tol:
        mismatches += 1

    print(f"Max difference: {max_diff}")
    if mismatches == 0:
      print(f"SUCCESS: Outputs match for {dtype.upper()}.")
      return True
    else:
      print(f"FAILED: Found {mismatches} mismatches for {dtype.upper()}.")
      return False

  except Exception as e:
    print(f"EXCEPTION occurred during verification: {e}")
    return False
  finally:
    if keep_temps:
      print(f"Preserving temporary directory: {tmpdir}")
    else:
      temp_dir_manager.cleanup()


def main():
  parser = argparse.ArgumentParser(
      description="Verify Matrix end-to-end execution on simulator.")
  parser.add_argument(
      '--dtype',
      type=str,
      default='all',
      choices=['f32', 'f16', 'bf16', 'i8', 'all'],
      help="Data type to verify. Use 'all' (default) to test f32, bf16, and i8."
  )
  parser.add_argument(
      '--build-dir',
      type=str,
      default=None,
      help="Path to the compiler build directory. Auto-detects if omitted.")
  parser.add_argument('--verbose',
                      action='store_true',
                      help="Print full input and output matrices.")
  parser.add_argument(
      '--keep-temps',
      action='store_true',
      help="Keep temporary files after execution (stored in /tmp).")
  args = parser.parse_args()

  compile_tool, run_tool = find_tools(args.build_dir)

  if args.dtype == 'all':
    # Verify Matrix operations (f32, bf16, i8). f16 falls back to Zve.
    dtypes_to_test = ['f32', 'bf16', 'i8']
  else:
    dtypes_to_test = [args.dtype]

  results = {}
  for dtype in dtypes_to_test:
    success = verify_dtype(dtype, compile_tool, run_tool, args.verbose,
                           args.keep_temps)
    results[dtype] = success

  print("\n==================================================")
  print(" SUMMARY")
  print("==================================================")
  all_success = True
  for dtype, success in results.items():
    status = "PASSED" if success else "FAILED"
    print(f"  {dtype.upper():6} : {status}")
    if not success:
      all_success = False

  if all_success:
    print("\nAll verifications completed successfully.")
    sys.exit(0)
  else:
    print("\nSome verifications failed. Check output logs.")
    sys.exit(1)


if __name__ == "__main__":
  main()
