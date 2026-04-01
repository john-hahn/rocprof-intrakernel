// codeobj_profiler_client.cpp — rocprofiler-sdk tool for code object ISA analysis.
//
// Intercepts code object loading, disassembles kernel ISA, and classifies
// instructions. When combined with PC sampling data, provides per-region
// instruction mix profiles.
//
// Usage:
//   ROCP_TOOL_LIB=./librikp_codeobj_profiler.so rocprofv3 ./my_hip_app
//
// Output: rikp_codeobj_analysis.json

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "isa_classifier.h"
#include "region_mapper.h"

namespace {

struct KernelInfo {
  std::string name;
  uint64_t code_object_id = 0;
  uint64_t kernel_id = 0;
  uint64_t code_offset = 0;
  uint64_t code_size = 0;
};

struct ToolState {
  rocprofiler_context_id_t context_id;
  std::mutex mu;
  std::vector<KernelInfo> kernels;
  std::string output_path = "rikp_codeobj_analysis.json";
};

ToolState* g_state = nullptr;

void codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                               rocprofiler_user_data_t* user_data,
                               void* callback_data) {
  (void)user_data;
  auto* state = static_cast<ToolState*>(callback_data);

  // Intercept code object load events.
  // The exact API depends on rocprofiler-sdk version.
  // This is a skeleton that will be filled in with the actual
  // code object introspection logic.
  (void)state;
  (void)record;
}

int tool_init(rocprofiler_client_finalize_t fini_func,
              void* tool_data) {
  (void)fini_func;
  auto* state = static_cast<ToolState*>(tool_data);

  auto status = rocprofiler_create_context(&state->context_id);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    std::fprintf(stderr, "rikp_codeobj_profiler: failed to create context: %d\n",
                 static_cast<int>(status));
    return -1;
  }

  // Register for code object tracing events.
  // TODO: Configure ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT when available.

  status = rocprofiler_start_context(state->context_id);
  if (status != ROCPROFILER_STATUS_SUCCESS) {
    std::fprintf(stderr, "rikp_codeobj_profiler: failed to start context: %d\n",
                 static_cast<int>(status));
  }

  std::fprintf(stderr, "rikp_codeobj_profiler: initialized\n");
  return 0;
}

void tool_fini(void* tool_data) {
  auto* state = static_cast<ToolState*>(tool_data);

  rocprofiler_stop_context(state->context_id);

  // Write analysis output.
  std::ofstream out(state->output_path);
  out << "{\n";
  out << "  \"tool\": \"rikp_codeobj_profiler\",\n";
  out << "  \"version\": \"0.1.0\",\n";
  out << "  \"kernels\": [\n";

  {
    std::lock_guard<std::mutex> lock(state->mu);
    for (size_t i = 0; i < state->kernels.size(); ++i) {
      const auto& k = state->kernels[i];
      out << "    {\"name\": \"" << k.name << "\""
          << ", \"code_object_id\": " << k.code_object_id
          << ", \"kernel_id\": " << k.kernel_id
          << ", \"code_offset\": " << k.code_offset
          << ", \"code_size\": " << k.code_size << "}";
      if (i + 1 < state->kernels.size()) out << ",";
      out << "\n";
    }
  }

  out << "  ]\n}\n";
  std::fprintf(stderr, "rikp_codeobj_profiler: %zu kernels -> %s\n",
               state->kernels.size(), state->output_path.c_str());
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

  id->name = "rikp_codeobj_profiler";

  g_state = new ToolState();

  static auto cfg =
      rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                          &tool_init, &tool_fini, g_state};
  return &cfg;
}
