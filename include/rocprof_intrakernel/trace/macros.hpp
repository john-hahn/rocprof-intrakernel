#pragma once

#include "rocprof_intrakernel/arch/wave_info.hpp"
#include "rocprof_intrakernel/trace/device_ctx.hpp"
#include "rocprof_intrakernel/trace/event.hpp"

// Device-side convenience macros.
//
// Only the wave leader (lane 0) records to keep overhead low.
// Use RIKP_* prefix to avoid macro name collisions.

#define RIKP_TRACE_CTX_TYPE(cap, waves_per_workgroup) \
  ::rocprof_intrakernel::trace::WaveContext<(cap), (waves_per_workgroup)>

#define RIKP_TRACE_CTX_INIT(ctx) \
  do { \
    if (::rocprof_intrakernel::arch::is_wave_leader()) { \
      (ctx).init(); \
    } \
  } while (0)

#define RIKP_TRACE_CTX_FLUSH(ctx, gbuf) \
  do { \
    if (::rocprof_intrakernel::arch::is_wave_leader()) { \
      (ctx).flush((gbuf)); \
    } \
  } while (0)

#define RIKP_TRACE_REC(ctx, gbuf, id, type) \
  do { \
    if (::rocprof_intrakernel::arch::is_wave_leader()) { \
      (ctx).record((gbuf), static_cast<uint16_t>(id), static_cast<uint16_t>(type)); \
    } \
  } while (0)

#define RIKP_TRACE_REC_B(ctx, gbuf, id) RIKP_TRACE_REC((ctx), (gbuf), (id), 0)
#define RIKP_TRACE_REC_E(ctx, gbuf, id) RIKP_TRACE_REC((ctx), (gbuf), (id), 1)
#define RIKP_TRACE_REC_M(ctx, gbuf, id) RIKP_TRACE_REC((ctx), (gbuf), (id), 2)

#define RIKP_TRACE_REC_IF(ctx, gbuf, id, type, cond) \
  do { \
    if (::rocprof_intrakernel::arch::is_wave_leader() && (cond)) { \
      (ctx).record((gbuf), static_cast<uint16_t>(id), static_cast<uint16_t>(type)); \
    } \
  } while (0)
