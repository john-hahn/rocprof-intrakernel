#pragma once

// Public entry for the intra-kernel timeline trace recorder:
// - device: wave-leader timestamped begin/end/mark events into per-wave circular buffers
// - host:   reconstruct ordered events, emit Chrome Trace JSON + summary JSON

#include "rocprof_intrakernel/trace/device_ctx.hpp"
#include "rocprof_intrakernel/trace/event.hpp"
#include "rocprof_intrakernel/trace/host_session.hpp"
#include "rocprof_intrakernel/trace/macros.hpp"
