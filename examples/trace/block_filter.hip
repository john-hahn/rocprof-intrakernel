#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <rocprof_intrakernel/rocprof_intrakernel.hpp>

#ifndef PROFILE_PER_WAVE_CAP
#define PROFILE_PER_WAVE_CAP 32768
#endif
static_assert((PROFILE_PER_WAVE_CAP & (PROFILE_PER_WAVE_CAP - 1u)) == 0u,
              "PROFILE_PER_WAVE_CAP must be power of 2");

constexpr int kBlockThreads = 128;
constexpr uint32_t kWavesPerBlock =
    (kBlockThreads + rocprof_intrakernel::arch::kWaveSize - 1) /
    rocprof_intrakernel::arch::kWaveSize;

__global__ void FilterDemoKernel(int iters, rocprof_intrakernel::trace::GlobalBuffer prof) {
  RIKP_TRACE_CTX_TYPE(PROFILE_PER_WAVE_CAP, kWavesPerBlock) ctx;
  RIKP_TRACE_CTX_INIT(ctx);

  RIKP_TRACE_REC_B(ctx, prof, 0);
  for (int i = 0; i < iters; ++i) {
    // Tiny dummy op to keep the loop body non-empty.
    uint32_t tmp = uint32_t(i);
    asm volatile("v_mov_b32 %0, %0" : "+v"(tmp));
  }
  RIKP_TRACE_REC_E(ctx, prof, 0);

  RIKP_TRACE_CTX_FLUSH(ctx, prof);
}

static void ck(hipError_t e, const char* msg) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", msg, hipGetErrorString(e));
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  int iters = 100000;
  const char* out_path = "block_filter_trace.json";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!a) continue;
    if (std::strncmp(a, "--iters=", 8) == 0) {
      iters = std::atoi(a + 8);
    } else if (std::strncmp(a, "--out=", 6) == 0) {
      out_path = a + 6;
    } else if (i == 1 && a[0] != '-') {
      iters = std::atoi(a);
    }
  }
  if (iters < 0) iters = 0;

  dim3 block(kBlockThreads, 1, 1);
  dim3 grid(8, 1, 1);

  rocprof_intrakernel::trace::HostSession sess;
  sess.set_region_names({"demo"});
  sess.init(uint32_t(PROFILE_PER_WAVE_CAP), uint32_t(grid.x), uint32_t(block.x),
            rocprof_intrakernel::arch::kWaveSize);
  sess.reset();

  FilterDemoKernel<<<grid, block>>>(iters, sess.global_buffer());
  ck(hipDeviceSynchronize(), "hipDeviceSynchronize");

  sess.set_block_filter({0u, 7u});

  rocprof_intrakernel::trace::TraceWriteOptions opt;
  opt.scale = 1e-3;
  sess.write_trace(out_path, opt);

  return 0;
}
