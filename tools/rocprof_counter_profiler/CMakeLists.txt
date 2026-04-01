add_library(rikp_counter_client SHARED
  src/rikp_counter_client.cpp
)
target_link_libraries(rikp_counter_client PRIVATE rocprofiler-sdk::rocprofiler-sdk)
set_target_properties(rikp_counter_client PROPERTIES
  OUTPUT_NAME "rikp_counter_client"
  PREFIX "lib"
)

add_library(rikp_pcsamp_client SHARED
  src/rikp_pcsamp_client.cpp
)
target_link_libraries(rikp_pcsamp_client PRIVATE rocprofiler-sdk::rocprofiler-sdk)
set_target_properties(rikp_pcsamp_client PROPERTIES
  OUTPUT_NAME "rikp_pcsamp_client"
  PREFIX "lib"
)
