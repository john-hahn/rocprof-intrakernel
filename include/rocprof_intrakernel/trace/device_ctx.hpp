#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>

#include "rocprof_intrakernel/arch/timestamp.hpp"
#include "rocprof_intrakernel/arch/wave_info.hpp"
#include "rocprof_intrakernel/trace/event.hpp"

namespace rocprof_intrakernel::trace {

// Per-wave circular buffer recorder. Only the wave leader (lane 0) calls
// init/record/flush to keep overhead low.
//
// CAP must be a power of two. If a wave records more than CAP events, only the
// last CAP are kept.
template <uint32_t CAP, uint32_t WAVES_PER_WORKGROUP>
struct WaveContext {
  static_assert((CAP & (CAP - 1u)) == 0u, "CAP must be power of 2");
  static constexpr uint32_t kMask = CAP - 1u;

  uint32_t base = 0;  // slot * CAP
  uint32_t cnt = 0;   // local event counter
  uint32_t slot = 0;  // slot index in global buffer
  uint32_t info = 0;  // packed: block(lo16) | (wave|cu)(hi16)

  __device__ __forceinline__ void init() {
    const uint32_t w = arch::wave_index_in_workgroup();
    const uint32_t b = static_cast<uint32_t>(blockIdx.x);
    const uint16_t cu = arch::read_cu_id();

    slot = b * WAVES_PER_WORKGROUP + w;
    base = slot * CAP;
    cnt = 0;
    const uint16_t packed = pack_wave_cu(static_cast<uint16_t>(w), cu);
    info = b | (static_cast<uint32_t>(packed) << 16);
  }

  // Record a trace event. Uses a regular store (no cache-bypass optimization yet).
  __device__ __forceinline__ void record(GlobalBuffer& buf, uint16_t id, uint16_t type) {
    if (!buf.events) return;
    const uint64_t ts = arch::read_realtime();
    const uint32_t idx = base + (cnt & kMask);

    buf.events[idx] = Event{
        ts,
        id,
        type,
        static_cast<uint16_t>(info & 0xFFFFu),
        static_cast<uint16_t>(info >> 16),
    };
    ++cnt;
  }

  __device__ __forceinline__ void flush(GlobalBuffer& buf) {
    if (!buf.counters) return;
    buf.counters[slot] = cnt;
  }
};

}  // namespace rocprof_intrakernel::trace
