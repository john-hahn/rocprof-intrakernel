// tiled_gemm.hip — Shared-memory tiled GEMM with comprehensive intra-kernel profiling.
//
// Build:
//   hipcc -O3 -std=c++17 --offload-arch=gfx1151 -I ../../include tiled_gemm.hip -o tiled_gemm
//
// Run:
//   ./tiled_gemm --m=2048 --n=2048 --k=2048 --out=gemm_trace.json
//
// Profiling regions (5 regions, IDs 1-5):
//   1: total      — entire tile computation envelope
//   2: load_A     — global → shared: load A tile
//   3: load_B     — global → shared: load B tile
//   4: compute    — shared → registers: multiply-accumulate
//   5: store      — registers → global: write-back C

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include <rocprof_intrakernel/rocprof_intrakernel.hpp>

// Tile configuration
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS = (BM / TM) * (BN / TN);  // 256

#ifndef PROFILE_CAP
#define PROFILE_CAP 8192
#endif
static_assert((PROFILE_CAP & (PROFILE_CAP - 1)) == 0, "PROFILE_CAP must be power of 2");
constexpr uint32_t kWavesPerBlock =
    (THREADS + rocprof_intrakernel::arch::kWaveSize - 1) /
    rocprof_intrakernel::arch::kWaveSize;

enum Region : uint16_t {
  kTotal   = 1,
  kLoadA   = 2,
  kLoadB   = 3,
  kCompute = 4,
  kStore   = 5,
};

__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    int tiles_n,
    rocprof_intrakernel::trace::GlobalBuffer prof) {

  RIKP_TRACE_CTX_TYPE(PROFILE_CAP, kWavesPerBlock) ctx;
  RIKP_TRACE_CTX_INIT(ctx);

  const int tile_row = blockIdx.x / tiles_n;
  const int tile_col = blockIdx.x % tiles_n;

  const int tx = threadIdx.x % (BN / TN);
  const int ty = threadIdx.x / (BN / TN);

  __shared__ float sA[BM][BK];
  __shared__ float sB[BK][BN];

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      acc[i][j] = 0.0f;

  const int row_base = tile_row * BM + ty * TM;
  const int col_base = tile_col * BN + tx * TN;

  RIKP_TRACE_REC_B(ctx, prof, kTotal);

  const int num_k_tiles = (K + BK - 1) / BK;
  for (int t = 0; t < num_k_tiles; ++t) {

    RIKP_TRACE_REC_B(ctx, prof, kLoadA);
    #pragma unroll
    for (int load = 0; load < (BM * BK) / THREADS; ++load) {
      int linear = load * THREADS + threadIdx.x;
      int sr = linear / BK;
      int sc = linear % BK;
      int gr = tile_row * BM + sr;
      int gc = t * BK + sc;
      sA[sr][sc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
    }
    RIKP_TRACE_REC_E(ctx, prof, kLoadA);

    RIKP_TRACE_REC_B(ctx, prof, kLoadB);
    #pragma unroll
    for (int load = 0; load < (BK * BN) / THREADS; ++load) {
      int linear = load * THREADS + threadIdx.x;
      int sr = linear / BN;
      int sc = linear % BN;
      int gr = t * BK + sr;
      int gc = tile_col * BN + sc;
      sB[sr][sc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
    }
    RIKP_TRACE_REC_E(ctx, prof, kLoadB);

    __syncthreads();

    RIKP_TRACE_REC_B(ctx, prof, kCompute);
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += sA[ty * TM + i][k] * sB[k][tx * TN + j];
        }
      }
    }
    RIKP_TRACE_REC_E(ctx, prof, kCompute);

    __syncthreads();
  }

  RIKP_TRACE_REC_B(ctx, prof, kStore);
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      int r = row_base + i;
      int c = col_base + j;
      if (r < M && c < N)
        C[r * N + c] = acc[i][j];
    }
  }
  RIKP_TRACE_REC_E(ctx, prof, kStore);

  RIKP_TRACE_REC_E(ctx, prof, kTotal);

  RIKP_TRACE_CTX_FLUSH(ctx, prof);
}

static void ck(hipError_t e, const char* msg) {
  if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", msg, hipGetErrorString(e)); std::exit(1); }
}
static int get_int(int argc, char** argv, const char* key, int def) {
  std::string pfx = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i)
    if (std::strncmp(argv[i], pfx.c_str(), pfx.size()) == 0)
      return std::atoi(argv[i] + pfx.size());
  return def;
}
static const char* get_str(int argc, char** argv, const char* key, const char* def) {
  std::string pfx = std::string("--") + key + "=";
  for (int i = 1; i < argc; ++i)
    if (std::strncmp(argv[i], pfx.c_str(), pfx.size()) == 0)
      return argv[i] + pfx.size();
  return def;
}

int main(int argc, char** argv) {
  const int M     = get_int(argc, argv, "m", 1024);
  const int N     = get_int(argc, argv, "n", 1024);
  const int K     = get_int(argc, argv, "k", 1024);
  const int iters = get_int(argc, argv, "iters", 3);
  const char* out = get_str(argc, argv, "out", "gemm_trace.json");

  const int tiles_m = (M + BM - 1) / BM;
  const int tiles_n = (N + BN - 1) / BN;
  const int total_blocks = tiles_m * tiles_n;

  std::printf("=== rocprof-intrakernel: Tiled GEMM Demo ===\n");
  std::printf("Problem:  M=%d N=%d K=%d\n", M, N, K);
  std::printf("Tiling:   BM=%d BN=%d BK=%d  TM=%d TN=%d  threads=%d\n",
              BM, BN, BK, TM, TN, THREADS);
  std::printf("Grid:     %d blocks (%d x %d tiles)\n", total_blocks, tiles_m, tiles_n);

  size_t szA = size_t(M) * K, szB = size_t(K) * N, szC = size_t(M) * N;
  std::vector<float> hA(szA), hB(szB), hC(szC, 0.f);
  for (size_t i = 0; i < szA; ++i) hA[i] = float((i * 17) % 101) / 101.f;
  for (size_t i = 0; i < szB; ++i) hB[i] = float((i * 29) % 103) / 103.f;

  float *dA, *dB, *dC;
  ck(hipMalloc(&dA, szA * sizeof(float)), "malloc A");
  ck(hipMalloc(&dB, szB * sizeof(float)), "malloc B");
  ck(hipMalloc(&dC, szC * sizeof(float)), "malloc C");
  ck(hipMemcpy(dA, hA.data(), szA * sizeof(float), hipMemcpyHostToDevice), "cpy A");
  ck(hipMemcpy(dB, hB.data(), szB * sizeof(float), hipMemcpyHostToDevice), "cpy B");

  dim3 grid(total_blocks);
  dim3 block(THREADS);

  rocprof_intrakernel::trace::HostSession sess;
  sess.set_region_names({"_outside", "total", "load_A", "load_B", "compute", "store"});
  sess.set_block_filter({0, 1, 2, 3});
  sess.init(PROFILE_CAP, total_blocks, THREADS, rocprof_intrakernel::arch::kWaveSize);

  std::printf("Profiling: blocks 0-3 (of %d), cap=%d events/wave\n\n", total_blocks, PROFILE_CAP);

  // Warmup
  {
    rocprof_intrakernel::trace::GlobalBuffer null_prof{};
    tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K, tiles_n, null_prof);
    ck(hipDeviceSynchronize(), "warmup");
  }

  // Profiled run
  ck(hipMemset(dC, 0, szC * sizeof(float)), "clear C");
  sess.reset();
  tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K, tiles_n, sess.global_buffer());
  ck(hipDeviceSynchronize(), "profile sync");

  rocprof_intrakernel::trace::TraceWriteOptions opt;
  opt.scale               = 1.0;
  opt.group_by_cu         = true;
  opt.emit_summary_json   = true;
  opt.emit_complete_events = true;
  sess.write_trace(out, opt);

  // Benchmark (profiling disabled)
  hipEvent_t ev0, ev1;
  ck(hipEventCreate(&ev0), "ev0");
  ck(hipEventCreate(&ev1), "ev1");
  ck(hipEventRecord(ev0), "rec0");
  for (int i = 0; i < iters; ++i) {
    rocprof_intrakernel::trace::GlobalBuffer null_prof{};
    tiled_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K, tiles_n, null_prof);
  }
  ck(hipEventRecord(ev1), "rec1");
  ck(hipEventSynchronize(ev1), "sync");
  float ms = 0;
  ck(hipEventElapsedTime(&ms, ev0, ev1), "elapsed");
  double gflops = 2.0 * M * N * K * iters / (ms * 1e6);
  std::printf("Benchmark: %d iters, %.3f ms total, %.3f ms/iter, %.1f GFLOP/s\n",
              iters, ms, ms / iters, gflops);

  // Quick verify
  ck(hipMemcpy(hC.data(), dC, szC * sizeof(float), hipMemcpyDeviceToHost), "cpy C");
  double ref = 0;
  for (int k = 0; k < K; ++k) ref += double(hA[k]) * double(hB[k * N]);
  double err = std::fabs(double(hC[0]) - ref);
  std::printf("Verify:    C[0,0]=%.4f ref=%.4f err=%.2e %s\n",
              hC[0], ref, err, (err < 0.1) ? "PASS" : "FAIL");

  hipFree(dA); hipFree(dB); hipFree(dC);
  hipEventDestroy(ev0); hipEventDestroy(ev1);
  return 0;
}
