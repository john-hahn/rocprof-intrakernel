cmake_minimum_required(VERSION 3.21)

project(rocprof_intrakernel LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# GPU architecture targets: RDNA 3, RDNA 3.5, CDNA 3, CDNA 3.5
set(RIKP_GPU_TARGETS "gfx1100;gfx1151;gfx942;gfx950" CACHE STRING
    "Semicolon-separated list of GPU architectures to target")
set(CMAKE_HIP_ARCHITECTURES ${RIKP_GPU_TARGETS})

option(RIKP_BUILD_EXAMPLES "Build examples" ON)
option(RIKP_BUILD_TOOLS    "Build rocprofiler tools" OFF)
option(RIKP_BUILD_TESTS    "Build tests" OFF)

find_package(hip REQUIRED)

if(RIKP_BUILD_TOOLS)
  find_package(rocprofiler-sdk QUIET)
  if(NOT rocprofiler-sdk_FOUND)
    message(WARNING "rocprofiler-sdk not found — tools will not be built")
    set(RIKP_BUILD_TOOLS OFF)
  endif()
endif()

# Header-only library target
add_library(rocprof_intrakernel_headers INTERFACE)
target_include_directories(rocprof_intrakernel_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(rocprof_intrakernel_headers INTERFACE hip::device)

if(RIKP_BUILD_EXAMPLES)
  add_executable(rikp_trace_record examples/trace/record.hip)
  target_link_libraries(rikp_trace_record PRIVATE rocprof_intrakernel_headers hip::host)

  add_executable(rikp_trace_block_filter examples/trace/block_filter.hip)
  target_link_libraries(rikp_trace_block_filter PRIVATE rocprof_intrakernel_headers hip::host)

  add_executable(rikp_trace_sampled_loop examples/trace/sampled_loop.hip)
  target_link_libraries(rikp_trace_sampled_loop PRIVATE rocprof_intrakernel_headers hip::host)

  add_executable(rikp_gemm_demo examples/gemm/tiled_gemm.hip)
  target_link_libraries(rikp_gemm_demo PRIVATE rocprof_intrakernel_headers hip::host)
endif()

if(RIKP_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

if(RIKP_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Install
include(GNUInstallDirs)

install(DIRECTORY include/rocprof_intrakernel
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS rocprof_intrakernel_headers
  EXPORT rocprof_intrakernel-targets
)

install(EXPORT rocprof_intrakernel-targets
  FILE rocprof_intrakernel-targets.cmake
  NAMESPACE rocprof_intrakernel::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rocprof_intrakernel
)
