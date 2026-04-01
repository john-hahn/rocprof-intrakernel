if(RIKP_BUILD_TOOLS AND rocprofiler-sdk_FOUND)
  add_subdirectory(rocprof_counter_profiler)
  add_subdirectory(codeobj_region_profiler)
endif()
