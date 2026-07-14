// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>

extern "C" {

struct ExceptionInfo {
  uint32_t mcause;
  uint32_t mepc;
  uint32_t mtval;
};

__attribute__((section(".ddr_exception_info"))) volatile ExceptionInfo exception_info;

void __attribute__((weak)) coralnpu_exception_handler() {
  uint32_t mcause, mepc, mtval;
  asm volatile("csrr %0, mcause" : "=r"(mcause));
  asm volatile("csrr %0, mepc" : "=r"(mepc));
  asm volatile("csrr %0, mtval" : "=r"(mtval));
  exception_info.mcause = mcause;
  exception_info.mepc = mepc;
  exception_info.mtval = mtval;

  asm volatile("ebreak");
  while (1) {}
}
}