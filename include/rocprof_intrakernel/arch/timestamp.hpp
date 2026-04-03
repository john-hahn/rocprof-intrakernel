#pragma once

#include <cstdint>

namespace rocprof_intrakernel::arch {

// Read the GPU realtime clock (nanosecond-resolution, constant-frequency).
//
// Architecture dispatch:
//   RDNA 3/3.5 (gfx1100, gfx1151): s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME  (~100 MHz)
//   CDNA 3/3.5 (gfx942, gfx950):   s_memrealtime                       (~100 MHz or 25 MHz)
//   Fallback (older GCN):           s_memtime (deprecated)

__device__ __forceinline__ uint64_t read_realtime() {
  uint64_t ts;
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
  // RDNA 3/3.5
  asm volatile(
      "s_sendmsg_rtn_b64 %0, sendmsg(MSG_RTN_GET_REALTIME)\n"
      "s_waitcnt lgkmcnt(0)"
      : "=s"(ts)
      :
      : "memory");
#elif defined(__gfx942__) || defined(__gfx950__) || defined(__gfx940__) || \
    defined(__gfx90a__)
  // CDNA 2/3
  asm volatile(
      "s_memrealtime %0\n"
      "s_waitcnt lgkmcnt(0)"
      : "=s"(ts)
      :
      : "memory");
#else
  // Fallback: s_memtime (deprecated but widely available on older GCN/RDNA)
  asm volatile(
      "s_memtime %0\n"
      "s_waitcnt lgkmcnt(0)"
      : "=s"(ts)
      :
      : "memory");
#endif
  return ts;
}

}  // namespace rocprof_intrakernel::arch
