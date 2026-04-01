#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

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

__global__ void SampledLoopKernel(int iters, int sample_mask,
                                   rocprof_intrakernel::trace::GlobalBuffer prof) {
  RIKP_TRACE_CTX_TYPE(PROFILE_PER_WAVE_CAP, kWavesPerBlock) ctx;
  RIKP_TRACE_CTX_INIT(ctx);

  RIKP_TRACE_REC_M(ctx, prof, 2);

  for (int i = 0; i < iters; ++i) {
    const bool sample = (i & sample_mask) == 0;
    RIKP_TRACE_REC_IF(ctx, prof, 0, 0, sample);  // begin
    uint32_t tmp = uint32_t(i);
    asm volatile("v_mov_b32 %0, %0" : "+v"(tmp));
    RIKP_TRACE_REC_IF(ctx, prof, 0, 1, sample);  // end
  }

  RIKP_TRACE_REC_M(ctx, prof, 3);
  RIKP_TRACE_CTX_FLUSH(ctx, prof);
}

static void ck(hipError_t e, const char* msg) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "%s: %s\n", msg, hipGetErrorString(e));
    std::exit(1);
  }
}

int main(int argc, char** argv) {
  int iters = 1 << 20;
  int sample_shift = 10;
  const char* out_path = "sampled_loop_trace.json";
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (!a) continue;
    if (std::strncmp(a, "--iters=", 8) == 0) {
      iters = std::atoi(a + 8);
    } else if (std::strncmp(a, "--sample_shift=", 15) == 0) {
      sample_shift = std::atoi(a + 15);
    } else if (std::strncmp(a, "--out=", 6) == 0) {
      out_path = a + 6;
    } else if (i == 1 && a[0] != '-') {
      iters = std::atoi(a);
    } else if (i == 2 && a[0] != '-') {
      sample_shift = std::atoi(a);
    }
  }
  if (sample_shift < 0) sample_shift = 0;
  if (sample_shift > 30) sample_shift = 30;
  const int sample_mask = (1 << sample_shift) - 1;
  if (iters < 0) iters = 0;

  dim3 block(kBlockThreads, 1, 1);
  dim3 grid(1, 1, 1);

  rocprof_intrakernel::trace::HostSession sess;
  sess.set_region_names({"sampled_region", "unused", "kernel_start", "kernel_end"});
  sess.init(uint32_t(PROFILE_PER_WAVE_CAP), uint32_t(grid.x), uint32_t(block.x),
            rocprof_intrakernel::arch::kWaveSize);
  sess.reset();

  SampledLoopKernel<<<grid, block>>>(iters, sample_mask, sess.global_buffer());
  ck(hipDeviceSynchronize(), "hipDeviceSynchronize");

  rocprof_intrakernel::trace::TraceWriteOptions opt;
  opt.scale = 1e-3;
  sess.write_trace(out_path, opt);

  return 0;
}
