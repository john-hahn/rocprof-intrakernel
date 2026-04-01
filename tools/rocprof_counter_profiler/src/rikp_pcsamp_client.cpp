// rikp_pcsamp_client.cpp — rocprofiler-sdk tool library for PC sampling.
//
// Usage:
//   ROCP_TOOL_LIB=./librikp_pcsamp_client.so rocprofv3 ./my_hip_app
//
// Output: rikp_pcsamp.json with per-PC sample data.

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rikp_common.h"

namespace {

struct PcSample {
  uint64_t pc_offset = 0;
  uint64_t code_object_id = 0;
  uint64_t timestamp = 0;
  uint64_t exec_mask = 0;
  uint32_t workgroup_x = 0;
  uint32_t workgroup_y = 0;
  uint32_t workgroup_z = 0;
};

struct ToolState {
  rocprofiler_context_id_t context_id;
  std::mutex samples_mu;
  std::vector<PcSample> samples;
  std::string output_path = "rikp_pcsamp.json";
  bool pc_sampling_available = false;
};

ToolState* g_state = nullptr;

int tool_init(rocprofiler_client_finalize_t fini_func,
              void* tool_data) {
  (void)fini_func;
  auto* state = static_cast<ToolState*>(tool_data);

  auto status = rocprofiler_create_context(&state->context_id);
  rikp::check_rocprof(status, "create_context");

  // PC sampling configuration depends on hardware support.
  // On unsupported hardware, this will fail gracefully.
  // TODO: Enumerate agents and check PC sampling support.
  //   rocprofiler_query_pc_sampling_agent_configurations(...)

  std::fprintf(stderr, "rikp_pcsamp_client: initialized (PC sampling support: %s)\n",
               state->pc_sampling_available ? "yes" : "checking...");

  status = rocprofiler_start_context(state->context_id);
  rikp::check_rocprof(status, "start_context");

  return 0;
}

void tool_fini(void* tool_data) {
  auto* state = static_cast<ToolState*>(tool_data);

  rocprofiler_stop_context(state->context_id);

  // Write output JSON.
  std::ofstream out(state->output_path);
  out << "{\n";
  out << "  \"tool\": \"rikp_pcsamp_client\",\n";
  out << "  \"version\": \"0.1.0\",\n";
  out << "  \"pc_sampling_available\": " << (state->pc_sampling_available ? "true" : "false")
      << ",\n";
  out << "  \"samples\": [\n";

  {
    std::lock_guard<std::mutex> lock(state->samples_mu);
    for (size_t i = 0; i < state->samples.size(); ++i) {
      const auto& s = state->samples[i];
      out << "    {\"pc_offset\": " << s.pc_offset
          << ", \"code_object_id\": " << s.code_object_id
          << ", \"timestamp\": " << s.timestamp
          << ", \"exec_mask\": " << s.exec_mask
          << ", \"workgroup\": [" << s.workgroup_x << ", " << s.workgroup_y << ", "
          << s.workgroup_z << "]}";
      if (i + 1 < state->samples.size()) out << ",";
      out << "\n";
    }
  }

  out << "  ]\n}\n";
  std::fprintf(stderr, "rikp_pcsamp_client: %zu samples -> %s\n",
               state->samples.size(), state->output_path.c_str());
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

  id->name = "rikp_pcsamp_client";

  g_state = new ToolState();

  static auto cfg =
      rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                          &tool_init, &tool_fini, g_state};
  return &cfg;
}
