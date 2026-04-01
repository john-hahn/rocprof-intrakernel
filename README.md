# rocprof-intrakernel

Region-level GPU profiling for AMD RDNA and CDNA architectures.

An AMD/ROCm port of [yao-jz/intra-kernel-profiler](https://github.com/yao-jz/intra-kernel-profiler), adapted for HIP, rocprofiler-sdk, and AMD GPU ISAs.

## What it does

Traditional GPU profilers report aggregate metrics per kernel dispatch. `rocprof-intrakernel` breaks down performance data by **named code regions within kernels**, enabling identification of which specific computational phases cause bottlenecks.

## Architecture support

| Architecture | GPU | Timer instruction | Wave size |
|---|---|---|---|
| RDNA 3 | gfx1100, gfx1101 | `s_sendmsg_rtn_b64 MSG_GET_REALTIME` | 32 |
| RDNA 3.5 | gfx1150, gfx1151 | `s_sendmsg_rtn_b64 MSG_GET_REALTIME` | 32 |
| CDNA 3 | gfx942 | `s_memrealtime` | 64 |
| CDNA 3.5 | gfx950 | `s_memrealtime` | 64 |

## Components

### 1. Trace backend (header-only)
Per-wave circular buffers with nanosecond timestamps. Zero external dependencies beyond HIP. Outputs Chrome Trace JSON viewable in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev).

### 2. rocProfiler counter collection (replaces CUPTI)
rocprofiler-sdk tool libraries for hardware counter collection and PC sampling. Loaded via `ROCP_TOOL_LIB` environment variable.

### 3. Code object analysis (replaces NVBit)
Static ISA disassembly and instruction classification. Correlates PC samples with instruction categories (VALU, SALU, VMEM, LDS, etc.) to provide per-region instruction mix profiles.

## Quick start

```bash
# Build (trace examples only — just needs ROCm)
mkdir build && cd build
cmake .. -DCMAKE_HIP_ARCHITECTURES=gfx1151
make -j

# Run
./rikp_trace_record --iters=10000 --out=trace.json
./rikp_gemm_demo --m=1024 --n=1024 --k=1024 --out=gemm_trace.json

# View in browser
# Open chrome://tracing and load trace.json
```

### Build with tools (requires rocprofiler-sdk)
```bash
cmake .. -DRIKP_BUILD_TOOLS=ON -DCMAKE_HIP_ARCHITECTURES="gfx1151;gfx942"
make -j

# Run with counter collection
ROCP_TOOL_LIB=./tools/rocprof_counter_profiler/librikp_counter_client.so \
  rocprofv3 ./rikp_gemm_demo
```

### Build with tests
```bash
cmake .. -DRIKP_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151
make -j
ctest
```

## Usage in your kernels

```cpp
#include <rocprof_intrakernel/rocprof_intrakernel.hpp>

constexpr uint32_t kWavesPerBlock =
    (THREADS + rocprof_intrakernel::arch::kWaveSize - 1) /
    rocprof_intrakernel::arch::kWaveSize;

__global__ void my_kernel(rocprof_intrakernel::trace::GlobalBuffer prof) {
  RIKP_TRACE_CTX_TYPE(8192, kWavesPerBlock) ctx;
  RIKP_TRACE_CTX_INIT(ctx);

  RIKP_TRACE_REC_B(ctx, prof, 0);  // begin region 0
  // ... computation ...
  RIKP_TRACE_REC_E(ctx, prof, 0);  // end region 0

  RIKP_TRACE_CTX_FLUSH(ctx, prof);
}

// Host side
rocprof_intrakernel::trace::HostSession sess;
sess.set_region_names({"my_region"});
sess.init(8192, grid_size, block_size, rocprof_intrakernel::arch::kWaveSize);
sess.reset();

my_kernel<<<grid, block>>>(sess.global_buffer());
hipDeviceSynchronize();
sess.write_trace("output.json");
```

## Dependencies

- **Required**: ROCm 6.0+ (HIP runtime, hipcc)
- **Optional for tools**: rocprofiler-sdk
- **Optional for analysis**: Python 3.8+
- **Build**: CMake 3.21+

## Credits

Based on [Intra-Kernel Profiler](https://github.com/yao-jz/intra-kernel-profiler) by Jizhao Yao et al. Ported to AMD/ROCm with architectural adaptations for RDNA and CDNA GPU families.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
