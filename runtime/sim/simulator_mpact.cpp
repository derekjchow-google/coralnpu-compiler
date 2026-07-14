/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>

#include "runtime/sim/simulator_api.h"

#ifdef USE_MPACT_PREBUILT

struct CoralNPUMailbox {
  uint32_t message[4] = {0, 0, 0, 0};
};

class CoralNPUSimulator {
 public:
  static CoralNPUSimulator *Create();

  virtual ~CoralNPUSimulator() = default;

  // Functions for reading/writing memory and Mailbox.
  virtual void ReadMem(uint32_t addr, size_t size, char *data) = 0;
  virtual const CoralNPUMailbox &ReadMailbox() = 0;
  virtual void WriteMem(uint32_t addr, size_t size, const char *data) = 0;
  virtual void WriteMailbox(const CoralNPUMailbox &mailbox) = 0;

  // Wait for interrupt
  virtual bool WaitForTermination(int timeout) = 0;

  // Begin executing starting with the PC set to the specified address. Returns
  // when the core halts.
  virtual void Run(uint32_t start_addr) = 0;
};

#else // USE_VERILATOR

#include "hw_sim/coralnpu_simulator.h"

#endif

static CoralNPUSimulator *sim = NULL;

void simulator_create(void) { sim = CoralNPUSimulator::Create(); }

void simulator_write_mem(uint32_t addr, const void *data, size_t size) {
#ifdef USE_MPACT_PREBUILT
  sim->WriteMem(addr, size, static_cast<const char *>(data));
#else
  sim->WriteTCM(addr, size, static_cast<const char *>(data));
#endif
}

void simulator_read_mem(uint32_t addr, void *data, size_t size) {
#ifdef USE_MPACT_PREBUILT
  sim->ReadMem(addr, size, static_cast<char *>(data));
#else
  sim->ReadTCM(addr, size, static_cast<char *>(data));
#endif
}

#include <cinttypes>
#include <unistd.h>
#include <cstdlib>
#include <chrono>

void simulator_run(uint32_t start_pc) {
  sim->Run(start_pc);

  bool show_progress = isatty(fileno(stderr));
  if (const char* env = std::getenv("CORALNPU_PROGRESS")) {
    show_progress = (env[0] != '0');
  }

  const char spinner[] = {'|', '/', '-', '\\'};
  int spinner_idx = 0;
  auto start_time = std::chrono::steady_clock::now();
  auto last_print_time = start_time;
  uint64_t total_steps = 0;
  const int step_size = 1; // Step by 1 for cycle-exact tracking
  
#ifdef USE_VERILATOR
  const char* unit = "cycles";
#else
  const char* unit = "steps";
#endif

  if (show_progress) {
    fprintf(stderr, "\rSimulation running [%c] %" PRIu64 " %s (0.0s)",
            spinner[spinner_idx], total_steps, unit);
    fflush(stderr);
  }
  
  while (!sim->WaitForTermination(step_size)) {
    total_steps += step_size;
    auto now = std::chrono::steady_clock::now();
    
    if (show_progress) {
      auto elapsed_from_last = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print_time).count();
      if (elapsed_from_last >= 100) { // Limit to 10 Hz
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        double elapsed_secs = elapsed / 1000.0;
        
        fprintf(stderr, "\rSimulation running [%c] %" PRIu64 " %s (%.1fs)",
                spinner[spinner_idx], total_steps, unit, elapsed_secs);
        fflush(stderr);
        spinner_idx = (spinner_idx + 1) % 4;
        last_print_time = now;
      }
    }
  }
  // No need to add step_size here because we stepped by 1 and the loop terminated when it halted,
  // so total_steps is already cycle-exact.

  auto end_time = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  double elapsed_secs = elapsed / 1000.0;
  double steps_per_sec = 0.0;
  if (elapsed_secs > 0.0) {
    steps_per_sec = total_steps / elapsed_secs;
  }

  const char* prefix = show_progress ? "\r" : "";
  // Use \033[K to clear to the end of the line to prevent trailing chars
  fprintf(stderr, "%sSimulation finished: %" PRIu64 " %s (%.1fs, %.0f %s/s)\033[K\n",
          prefix, total_steps, unit, elapsed_secs, steps_per_sec, unit);
  fflush(stderr);

  // Read exception info from DDR (0x8F000000)
  uint32_t exception_info[3] = {0, 0, 0};
#ifdef USE_MPACT_PREBUILT
  sim->ReadMem(0x8F000000, sizeof(exception_info), reinterpret_cast<char*>(exception_info));
#else
  sim->ReadTCM(0x8F000000, sizeof(exception_info), reinterpret_cast<char*>(exception_info));
#endif
  if (exception_info[0] != 0) {
    fprintf(stderr, "=== NPU Firmware Exception! ===\n");
    fprintf(stderr, "mcause: 0x%08" PRIx32 "\n", exception_info[0]);
    fprintf(stderr, "mepc:   0x%08" PRIx32 "\n", exception_info[1]);
    fprintf(stderr, "mtval:  0x%08" PRIx32 "\n", exception_info[2]);
    fprintf(stderr, "===============================\n");
    fflush(stderr);
  }
}
