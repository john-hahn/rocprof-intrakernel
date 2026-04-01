#pragma once

// Common utilities shared between rocprofiler-sdk tool libraries.

#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace rikp {

// Check rocprofiler status and abort on failure.
inline void check_rocprof(rocprofiler_status_t status, const char* msg) {
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    std::fprintf(stderr, "rocprofiler error at %s: %d\n", msg, static_cast<int>(status));
    std::exit(1);
  }
}

// Kernel invocation record.
struct KernelInvocation {
  uint64_t dispatch_id = 0;
  uint64_t correlation_id = 0;
  std::string kernel_name;
  uint32_t grid_x = 0, grid_y = 0, grid_z = 0;
  uint32_t block_x = 0, block_y = 0, block_z = 0;
  uint32_t shared_mem_bytes = 0;
};

// Thread-safe invocation tracker.
class InvocationTracker {
 public:
  void record(KernelInvocation inv) {
    std::lock_guard<std::mutex> lock(mu_);
    invocations_.push_back(std::move(inv));
  }

  std::vector<KernelInvocation> get_all() const {
    std::lock_guard<std::mutex> lock(mu_);
    return invocations_;
  }

 private:
  mutable std::mutex mu_;
  std::vector<KernelInvocation> invocations_;
};

// JSON string escaping.
inline std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '\"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c >= 0x20) out.push_back(static_cast<char>(c));
        break;
    }
  }
  return out;
}

}  // namespace rikp
