// test_trace_roundtrip.hip — Full trace record/read cycle test.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include <rocprof_intrakernel/rocprof_intrakernel.hpp>

constexpr int kBlockThreads = 64;
constexpr uint32_t kWavesPerBlock =
    (kBlockThreads + rocprof_intrakernel::arch::kWaveSize - 1) /
    rocprof_intrakernel::arch::kWaveSize;

__global__ void test_kernel(rocprof_intrakernel::trace::GlobalBuffer prof) {
  RIKP_TRACE_CTX_TYPE(1024, kWavesPerBlock) ctx;
  RIKP_TRACE_CTX_INIT(ctx);

  RIKP_TRACE_REC_B(ctx, prof, 0);
  RIKP_TRACE_REC_B(ctx, prof, 1);
  RIKP_TRACE_REC_E(ctx, prof, 1);
  RIKP_TRACE_REC_E(ctx, prof, 0);
  RIKP_TRACE_REC_M(ctx, prof, 2);

  RIKP_TRACE_CTX_FLUSH(ctx, prof);
}

int main() {
  rocprof_intrakernel::trace::HostSession sess;
  sess.set_region_names({"outer", "inner", "marker"});
  sess.init(1024, 1, kBlockThreads, rocprof_intrakernel::arch::kWaveSize);
  sess.reset();

  test_kernel<<<1, kBlockThreads>>>(sess.global_buffer());
  hipDeviceSynchronize();

  const std::string path = "/tmp/rikp_test_roundtrip.json";
  rocprof_intrakernel::trace::TraceWriteOptions opt;
  opt.scale = 1.0;
  opt.emit_summary_json = true;
  sess.write_trace(path, opt);

  // Verify the trace file was created and is valid JSON-ish.
  std::ifstream in(path);
  if (!in.good()) {
    std::fprintf(stderr, "FAIL: trace file not created at %s\n", path.c_str());
    return 1;
  }

  std::string content((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  if (content.find("\"traceEvents\"") == std::string::npos) {
    std::fprintf(stderr, "FAIL: trace file missing traceEvents key\n");
    return 1;
  }
  if (content.find("\"outer\"") == std::string::npos) {
    std::fprintf(stderr, "FAIL: trace file missing 'outer' region name\n");
    return 1;
  }
  if (content.find("\"inner\"") == std::string::npos) {
    std::fprintf(stderr, "FAIL: trace file missing 'inner' region name\n");
    return 1;
  }

  std::printf("PASS: trace roundtrip test\n");
  return 0;
}
