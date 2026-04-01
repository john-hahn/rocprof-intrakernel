// test_wave_info.hip — Verify CU ID and wave ID reads return valid values.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <rocprof_intrakernel/arch/wave_info.hpp>

struct WaveData {
  uint16_t cu_id;
  uint16_t wave_id;
  uint16_t simd_id;
  uint32_t wave_index;
  uint32_t block_id;
};

__global__ void wave_info_kernel(WaveData* out) {
  if (rocprof_intrakernel::arch::is_wave_leader()) {
    uint32_t idx = blockIdx.x * (blockDim.x / rocprof_intrakernel::arch::kWaveSize) +
                   rocprof_intrakernel::arch::wave_index_in_workgroup();
    out[idx].cu_id = rocprof_intrakernel::arch::read_cu_id();
    out[idx].wave_id = rocprof_intrakernel::arch::read_wave_id();
    out[idx].simd_id = rocprof_intrakernel::arch::read_simd_id();
    out[idx].wave_index = rocprof_intrakernel::arch::wave_index_in_workgroup();
    out[idx].block_id = blockIdx.x;
  }
}

int main() {
  constexpr int BLOCKS = 4;
  constexpr int THREADS = 256;
  constexpr int WAVES_PER_BLOCK =
      THREADS / rocprof_intrakernel::arch::kWaveSize;
  constexpr int TOTAL_WAVES = BLOCKS * WAVES_PER_BLOCK;

  WaveData* d_out;
  hipMalloc(&d_out, sizeof(WaveData) * TOTAL_WAVES);
  hipMemset(d_out, 0, sizeof(WaveData) * TOTAL_WAVES);

  wave_info_kernel<<<BLOCKS, THREADS>>>(d_out);
  hipDeviceSynchronize();

  std::vector<WaveData> h_out(TOTAL_WAVES);
  hipMemcpy(h_out.data(), d_out, sizeof(WaveData) * TOTAL_WAVES, hipMemcpyDeviceToHost);

  int failures = 0;
  for (int i = 0; i < TOTAL_WAVES; ++i) {
    const auto& w = h_out[i];
    // CU ID should be a small positive number (< 256 for any current GPU).
    if (w.cu_id > 255) {
      std::fprintf(stderr, "FAIL: wave %d has cu_id=%u (too large)\n", i, w.cu_id);
      ++failures;
    }
    // Wave index should be within [0, WAVES_PER_BLOCK).
    if (w.wave_index >= static_cast<uint32_t>(WAVES_PER_BLOCK)) {
      std::fprintf(stderr, "FAIL: wave %d has wave_index=%u (expected < %d)\n",
                   i, w.wave_index, WAVES_PER_BLOCK);
      ++failures;
    }
  }

  hipFree(d_out);

  if (failures == 0) {
    std::printf("PASS: %d waves, all CU/wave IDs valid\n", TOTAL_WAVES);
    return 0;
  } else {
    std::fprintf(stderr, "FAIL: %d failures\n", failures);
    return 1;
  }
}
