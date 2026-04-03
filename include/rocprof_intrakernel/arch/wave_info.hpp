#pragma once

#include <cstdint>

namespace rocprof_intrakernel::arch {

// Wave size is a compile-time constant set by the HIP compiler based on
// -mwavefrontsize32 / -mwavefrontsize64 and the target architecture.
//   RDNA (wave32 mode): 32
//   CDNA (always):      64
#if defined(__AMDGCN_WAVEFRONT_SIZE)
static constexpr uint32_t kWaveSize = __AMDGCN_WAVEFRONT_SIZE;
#else
static constexpr uint32_t kWaveSize = 32;
#endif
static constexpr uint32_t kWaveSizeLog2 = (kWaveSize == 64) ? 6 : 5;
static constexpr uint32_t kWaveMask = kWaveSize - 1;

// Read HW_ID1 register via s_getreg_b32.
//
// RDNA 3/3.5 HW_ID1 layout:
//   [3:0]   WAVE_ID   (wave slot within SIMD)
//   [7:4]   SIMD_ID
//   [13:8]  WGP_ID    (Workgroup Processor — AMD equivalent of NVIDIA SM)
//   [17:14] SA_ID     (Shader Array)
//   [21:18] SE_ID     (Shader Engine)
//
// CDNA (gfx9xx) HW_ID1 layout:
//   [3:0]   WAVE_ID
//   [7:4]   SIMD_ID
//   [11:8]  CU_ID     (Compute Unit)
//   [15:12] SH_ID     (Shader Half)
//   [19:16] SE_ID     (Shader Engine)

__device__ __forceinline__ uint32_t read_hw_id1() {
  uint32_t hw_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID1)" : "=s"(hw_id));
  return hw_id;
}

// Read Compute Unit / Workgroup Processor ID.
// On RDNA this is the WGP_ID; on CDNA this is the CU_ID.
__device__ __forceinline__ uint16_t read_cu_id() {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
  // RDNA 3/3.5: WGP_ID in bits [13:8]
  return static_cast<uint16_t>((read_hw_id1() >> 8) & 0x3Fu);
#else
  // CDNA / older: CU_ID in bits [11:8]
  return static_cast<uint16_t>((read_hw_id1() >> 8) & 0xFu);
#endif
}

// Read wave slot ID within the SIMD (bits [3:0] on all architectures).
__device__ __forceinline__ uint16_t read_wave_id() {
  return static_cast<uint16_t>(read_hw_id1() & 0xFu);
}

// Read SIMD ID (bits [7:4] on all architectures).
__device__ __forceinline__ uint16_t read_simd_id() {
  return static_cast<uint16_t>((read_hw_id1() >> 4) & 0xFu);
}

// Check if the current lane is the wave leader (lane 0).
__device__ __forceinline__ bool is_wave_leader() {
  return (threadIdx.x & kWaveMask) == 0;
}

// Get the wave index within the current workgroup.
__device__ __forceinline__ uint32_t wave_index_in_workgroup() {
  return static_cast<uint32_t>(threadIdx.x >> kWaveSizeLog2);
}

}  // namespace rocprof_intrakernel::arch
