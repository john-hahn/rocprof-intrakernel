add_library(rikp_codeobj_profiler SHARED
  src/codeobj_profiler_client.cpp
  src/isa_classifier.cpp
  src/region_mapper.cpp
)
target_link_libraries(rikp_codeobj_profiler PRIVATE rocprofiler-sdk::rocprofiler-sdk)
set_target_properties(rikp_codeobj_profiler PROPERTIES
  OUTPUT_NAME "rikp_codeobj_profiler"
  PREFIX "lib"
)
