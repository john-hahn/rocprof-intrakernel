// rikp_counter_client.cpp — rocprofiler-sdk tool library for hardware counter collection.
//
// Usage:
//   ROCP_TOOL_LIB=./librikp_counter_client.so rocprofv3 ./my_hip_app
//
// Or via LD_PRELOAD:
//   LD_PRELOAD=./librikp_counter_client.so ./my_hip_app
//
// Output: rikp_counters.json with per-dispatch counter values.

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "rikp_common.h"

namespace {

struct ToolState {
  rocprofiler_context_id_t context_id;
  rocprofiler_buffer_id_t buffer_id;
  rikp::InvocationTracker tracker;
  std::mutex output_mu;
  std::vector<std::string> counter_names;
  std::string output_path = "rikp_counters.json";
};

ToolState* g_state = nullptr;

void dispatch_callback(rocprofiler_profile_counting_dispatch_data_t dispatch_data,
                       rocprofiler_profile_config_id_t* config,
                       rocprofiler_user_data_t* user_data,
                       void* callback_data_args) {
  // For now, use default profile (all available counters).
  // Future: filter by kernel name regex and select specific counters.
  auto* state = static_cast<ToolState*>(callback_data_args);
  (void)state;
  (void)user_data;

  // Create a basic counter collection profile.
  rocprofiler_profile_config_id_t profile;
  auto status = rocprofiler_create_profile_config(
      dispatch_data.dispatch_info.agent_id, nullptr, 0, &profile);
  if (status == ROCPROFILER_STATUS_SUCCESS) {
    *config = profile;
  }
}

void buffer_tracing_callback(rocprofiler_context_id_t context,
                             rocprofiler_buffer_id_t buffer_id,
                             rocprofiler_record_header_t** headers,
                             size_t num_headers,
                             void* data,
                             uint64_t drop_count) {
  (void)context;
  (void)buffer_id;
  (void)data;
  (void)drop_count;

  for (size_t i = 0; i < num_headers; ++i) {
    auto* header = headers[i];
    if (!header) continue;
    // Process counter records when the rocprofiler-sdk API provides them.
    // The exact record type depends on the rocprofiler-sdk version.
    (void)header;
  }
}

int tool_init(rocprofiler_client_finalize_t fini_func,
              void* tool_data) {
  (void)fini_func;
  auto* state = static_cast<ToolState*>(tool_data);

  auto status = rocprofiler_create_context(&state->context_id);
  rikp::check_rocprof(status, "create_context");

  // Configure dispatch counting service.
  status = rocprofiler_configure_callback_dispatch_profile_counting_service(
      state->context_id, dispatch_callback, state, nullptr);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    std::fprintf(stderr, "rikp_counter_client: failed to configure dispatch counting: %d\n",
                 static_cast<int>(status));
  }

  status = rocprofiler_start_context(state->context_id);
  rikp::check_rocprof(status, "start_context");

  std::fprintf(stderr, "rikp_counter_client: initialized\n");
  return 0;
}

void tool_fini(void* tool_data) {
  auto* state = static_cast<ToolState*>(tool_data);

  rocprofiler_stop_context(state->context_id);

  // Write output JSON.
  std::ofstream out(state->output_path);
  out << "{\n";
  out << "  \"tool\": \"rikp_counter_client\",\n";
  out << "  \"version\": \"0.1.0\",\n";
  out << "  \"dispatches\": []\n";
  out << "}\n";

  std::fprintf(stderr, "rikp_counter_client: output -> %s\n", state->output_path.c_str());
  delete state;
}

}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t version,
                      const char* runtime_version,
                      uint32_t priority,
                      rocprofiler_client_id_t* id) {
  (void)version;
  (void)runtime_version;
  (void)priority;

  // Mark client id for later use.
  id->name = "rikp_counter_client";

  g_state = new ToolState();

  static auto cfg =
      rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                          &tool_init, &tool_fini, g_state};
  return &cfg;
}
