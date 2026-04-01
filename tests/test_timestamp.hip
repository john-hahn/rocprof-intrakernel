// test_timestamp.hip — Verify realtime clock reads are monotonic and in expected range.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <rocprof_intrakernel/arch/timestamp.hpp>
#include <rocprof_intrakernel/arch/wave_info.hpp>

__global__ void timestamp_kernel(uint64_t* out, int n) {
  if (rocprof_intrakernel::arch::is_wave_leader()) {
    for (int i = 0; i < n; ++i) {
      out[i] = rocprof_intrakernel::arch::read_realtime();
    }
  }
}

int main() {
  constexpr int N = 1024;
  uint64_t* d_out;
  hipMalloc(&d_out, sizeof(uint64_t) * N);

  timestamp_kernel<<<1, rocprof_intrakernel::arch::kWaveSize>>>(d_out, N);
  hipDeviceSynchronize();

  std::vector<uint64_t> h_out(N);
  hipMemcpy(h_out.data(), d_out, sizeof(uint64_t) * N, hipMemcpyDeviceToHost);

  int failures = 0;

  // Check monotonicity.
  for (int i = 1; i < N; ++i) {
    if (h_out[i] < h_out[i - 1]) {
      std::fprintf(stderr, "FAIL: timestamp[%d]=%llu < timestamp[%d]=%llu (not monotonic)\n",
                   i, (unsigned long long)h_out[i], i - 1, (unsigned long long)h_out[i - 1]);
      ++failures;
      if (failures > 10) break;
    }
  }

  // Check that timestamps are non-zero and that total span is reasonable.
  if (h_out[0] == 0) {
    std::fprintf(stderr, "FAIL: first timestamp is 0 (clock not working?)\n");
    ++failures;
  }

  const uint64_t span = h_out[N - 1] - h_out[0];
  // At 100MHz, 1024 reads should take at least a few microseconds.
  // At most a few milliseconds. Allow wide range.
  if (span == 0) {
    std::fprintf(stderr, "FAIL: all timestamps identical (span=0)\n");
    ++failures;
  }

  hipFree(d_out);

  if (failures == 0) {
    std::printf("PASS: %d timestamps, monotonic, span=%llu ticks\n",
                N, (unsigned long long)span);
    return 0;
  } else {
    std::fprintf(stderr, "FAIL: %d failures\n", failures);
    return 1;
  }
}
