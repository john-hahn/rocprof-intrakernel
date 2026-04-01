#pragma once

#include <cstdint>

namespace rocprof_intrakernel::trace {

// Pack (wave, cu_id) into 16 bits:
//   wave:  6 bits  (0..63)
//   cu_id: 10 bits (0..1023)
__host__ __device__ __forceinline__ uint16_t pack_wave_cu(uint16_t wave, uint16_t cu_id) {
  return static_cast<uint16_t>((wave & 0x3Fu) | ((cu_id & 0x3FFu) << 6));
}
__host__ __device__ __forceinline__ uint16_t unpack_wave(uint16_t packed) {
  return static_cast<uint16_t>(packed & 0x3Fu);
}
__host__ __device__ __forceinline__ uint16_t unpack_cu(uint16_t packed) {
  return static_cast<uint16_t>(packed >> 6);
}

// Event: 16-byte record written via a single 128-bit store.
struct alignas(16) Event {
  uint64_t ts;      // realtime clock (raw ticks)
  uint16_t id;      // region id
  uint16_t type;    // 0=begin, 1=end, 2=instant(mark)
  uint16_t block;   // workgroup id (blockIdx.x)
  uint16_t wave;    // packed (wave_in_workgroup, cu_id)
};
static_assert(sizeof(Event) == 16, "Event must be exactly 16 bytes");
static_assert(alignof(Event) == 16, "Event must be 16-byte aligned");

struct GlobalBuffer {
  Event* events = nullptr;
  uint32_t* counters = nullptr;  // per-(block,wave) final count (for circular buffer)
};

}  // namespace rocprof_intrakernel::trace
