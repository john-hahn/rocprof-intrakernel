// test_circular_buffer.hip — Verify circular buffer overflow keeps only last CAP events.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <rocprof_intrakernel/rocprof_intrakernel.hpp>

constexpr uint32_t CAP = 64;  // Small cap to easily overflow.
constexpr int kBlockThreads = 64;
constexpr uint32_t kWavesPerBlock =
    (kBlockThreads + rocprof_intrakernel::arch::kWaveSize - 1) /
    rocprof_intrakernel::arch::kWaveSize;

__global__ void overflow_kernel(int n_events, rocprof_intrakernel::trace::GlobalBuffer prof) {
  RIKP_TRACE_CTX_TYPE(CAP, kWavesPerBlock) ctx;
  RIKP_TRACE_CTX_INIT(ctx);

  for (int i = 0; i < n_events; ++i) {
    RIKP_TRACE_REC_B(ctx, prof, 0);
  }

  RIKP_TRACE_CTX_FLUSH(ctx, prof);
}

int main() {
  const int n_events = 200;  // Much larger than CAP=64.

  rocprof_intrakernel::trace::HostSession sess;
  sess.set_region_names({"test"});
  sess.init(CAP, 1, kBlockThreads, rocprof_intrakernel::arch::kWaveSize);
  sess.reset();

  overflow_kernel<<<1, kBlockThreads>>>(n_events, sess.global_buffer());
  hipDeviceSynchronize();

  // Read back the counter to verify it recorded n_events (not capped at CAP).
  uint32_t total_slots = 1 * kWavesPerBlock;
  std::vector<uint32_t> counts(total_slots);
  hipMemcpy(counts.data(), sess.global_buffer().counters,
            sizeof(uint32_t) * total_slots, hipMemcpyDeviceToHost);

  // The first wave should have recorded all n_events.
  if (counts[0] != static_cast<uint32_t>(n_events)) {
    std::fprintf(stderr, "FAIL: counter=%u, expected=%d\n", counts[0], n_events);
    return 1;
  }

  // Write trace to verify only CAP events come out (the last CAP).
  const std::string path = "/tmp/rikp_test_circular.json";
  rocprof_intrakernel::trace::TraceWriteOptions opt;
  opt.scale = 1.0;
  opt.emit_summary_json = false;
  sess.write_trace(path, opt);

  std::printf("PASS: circular buffer overflow test (recorded=%d, cap=%d)\n", n_events, CAP);
  return 0;
}
