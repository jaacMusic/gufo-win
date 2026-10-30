# Windows NPU unit tests run on the CPU against fake XRT/HIP backends; the
# "_real" probes read live HAL/memory state and stay informative on fail.
# Mirrors bubo tests/npu registration; paths point at our npu module.
if(NOT GUFO_ENABLE_NPU)
  return()
endif()
set(NPU_DIR ${CMAKE_SOURCE_DIR}/src/models/qwen38_flash_next/npu)
set(GUFO_NPU_FAULT_SOURCES ${NPU_DIR}/fault_telemetry.cpp ${NPU_DIR}/hal_fault_monitor.cpp
  ${NPU_DIR}/chip_power.cpp ${NPU_DIR}/runtime_status.cpp ${NPU_DIR}/mem_trace.cpp
  ${NPU_DIR}/guard.cpp ${NPU_DIR}/platform_info.cpp)

add_executable(npu_xrt_runtime_test
  xrt_runtime_test.cpp ${NPU_DIR}/xrt_runtime_policy.cpp)
add_test(NAME npu_xrt_runtime_test COMMAND npu_xrt_runtime_test)
set_tests_properties(npu_xrt_runtime_test PROPERTIES TIMEOUT 30)

add_executable(npu_xrt_delay_guard_test xrt_delay_guard_test.cpp)
npu_attach_runtime(npu_xrt_delay_guard_test)
target_link_libraries(npu_xrt_delay_guard_test PRIVATE gufo_core)
add_test(NAME npu_xrt_delay_guard_test COMMAND npu_xrt_delay_guard_test)
set_tests_properties(npu_xrt_delay_guard_test PROPERTIES TIMEOUT 30)

# Same adapter and pinned public XRT classes, with tooling-derived fake
# implementations. No import library/runtime loader is linked into this test.
add_executable(npu_private_bo_adapter_test
  private_bo_adapter_test.cpp fake_xrt.cpp
  ${NPU_DIR}/private_bo_adapter.cpp)
target_include_directories(npu_private_bo_adapter_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(npu_private_bo_adapter_test SYSTEM PRIVATE
  "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
target_compile_definitions(npu_private_bo_adapter_test PRIVATE
  XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
target_link_libraries(npu_private_bo_adapter_test PRIVATE gufo_core)
set(GUFO_NPU_FAKE_CASES "${CMAKE_BINARY_DIR}/npu-fake-cases"
  CACHE PATH "Prepared bad-artifact fixtures for CPU-only NPU tests")
add_test(NAME npu_private_bo_adapter_test COMMAND npu_private_bo_adapter_test
  "${GUFO_XRT_TOOLING_ROOT}" "${GUFO_NPU_FAKE_CASES}")
set_tests_properties(npu_private_bo_adapter_test PROPERTIES TIMEOUT 30)

add_executable(npu_bfp_test bfp_test.cpp ${NPU_DIR}/bfp.cpp)
target_include_directories(npu_bfp_test PRIVATE "${CMAKE_SOURCE_DIR}")
add_test(NAME npu_bfp_test COMMAND npu_bfp_test)
set_tests_properties(npu_bfp_test PROPERTIES TIMEOUT 30)

# fake_hip needs the HIP SDK headers; when HIP is off (hosted CPU presets),
# the HIP-backed targets are skipped and the pure-CPU ones still build.
if(TARGET hip::amdhip64)
add_executable(npu_linear_z_test linear_z_test.cpp
  ${NPU_DIR}/linear_z.cpp ${NPU_DIR}/staging.cpp ${NPU_DIR}/bfp.cpp
  ${NPU_DIR}/session.cpp ${NPU_DIR}/private_bo_adapter.cpp
  fake_xrt.cpp)
target_include_directories(npu_linear_z_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(npu_linear_z_test SYSTEM PRIVATE
  "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
target_compile_definitions(npu_linear_z_test PRIVATE
  XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
target_link_libraries(npu_linear_z_test PRIVATE Threads::Threads gufo_core gufo_logging)
add_test(NAME npu_linear_z_test COMMAND npu_linear_z_test "${GUFO_XRT_TOOLING_ROOT}")
set_tests_properties(npu_linear_z_test PROPERTIES TIMEOUT 30)

add_executable(npu_guard_latch_test guard_latch_test.cpp
  ${NPU_DIR}/linear_z.cpp ${NPU_DIR}/staging.cpp ${NPU_DIR}/bfp.cpp
  ${NPU_DIR}/session.cpp ${NPU_DIR}/private_bo_adapter.cpp
  fake_xrt.cpp)
target_include_directories(npu_guard_latch_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(npu_guard_latch_test SYSTEM PRIVATE
  "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
target_compile_definitions(npu_guard_latch_test PRIVATE
  XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
target_link_libraries(npu_guard_latch_test PRIVATE Threads::Threads gufo_core gufo_logging)
add_test(NAME npu_guard_latch_test COMMAND npu_guard_latch_test "${GUFO_XRT_TOOLING_ROOT}")
set_tests_properties(npu_guard_latch_test PROPERTIES TIMEOUT 30)

add_executable(npu_projection_test projection_test.cpp
  ${NPU_DIR}/projection.cpp ${NPU_DIR}/staging.cpp ${NPU_DIR}/bfp.cpp
  ${NPU_DIR}/session.cpp ${NPU_DIR}/private_bo_adapter.cpp
  fake_xrt.cpp)
target_include_directories(npu_projection_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(npu_projection_test SYSTEM PRIVATE
  "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
target_compile_definitions(npu_projection_test PRIVATE
  XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
target_link_libraries(npu_projection_test PRIVATE Threads::Threads gufo_core gufo_logging)
add_test(NAME npu_projection_test COMMAND npu_projection_test "${GUFO_XRT_TOOLING_ROOT}")
set_tests_properties(npu_projection_test PROPERTIES TIMEOUT 30)
endif(TARGET hip::amdhip64)

# Fault telemetry + HAL fault monitor (detection only). CPU tests use fakes;
# nothing here reads the Windows event log.
add_executable(npu_hal_fault_monitor_test hal_fault_monitor_test.cpp
  ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_hal_fault_monitor_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_hal_fault_monitor_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_hal_fault_monitor_test PRIVATE Threads::Threads gufo_logging)
add_executable(npu_hal_source_real_test hal_source_real_test.cpp
  ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_hal_source_real_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_hal_source_real_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_hal_source_real_test PRIVATE Threads::Threads gufo_logging)
add_executable(npu_chip_power_test chip_power_test.cpp
  ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_chip_power_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_chip_power_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_chip_power_test PRIVATE Threads::Threads gufo_logging)
add_executable(npu_guard_test guard_test.cpp ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_guard_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_guard_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_guard_test PRIVATE Threads::Threads gufo_logging)
add_test(NAME npu_guard_test COMMAND npu_guard_test)
set_tests_properties(npu_guard_test PROPERTIES TIMEOUT 30)
add_executable(npu_platform_info_test platform_info_test.cpp ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_platform_info_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_platform_info_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_platform_info_test PRIVATE Threads::Threads gufo_logging)
add_test(NAME npu_platform_info_test COMMAND npu_platform_info_test)
set_tests_properties(npu_platform_info_test PROPERTIES TIMEOUT 30)
add_executable(npu_bo_address_test bo_address_test.cpp)
target_include_directories(npu_bo_address_test PRIVATE "${CMAKE_SOURCE_DIR}")
add_test(NAME npu_bo_address_test COMMAND npu_bo_address_test)
set_tests_properties(npu_bo_address_test PROPERTIES TIMEOUT 30)
add_executable(npu_chunk_retry_test chunk_retry_test.cpp
  ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_chunk_retry_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_chunk_retry_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_chunk_retry_test PRIVATE Threads::Threads gufo_logging)
add_test(NAME npu_chunk_retry_test COMMAND npu_chunk_retry_test)
set_tests_properties(npu_chunk_retry_test PROPERTIES TIMEOUT 30)
add_executable(npu_runtime_status_test runtime_status_test.cpp
  ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_runtime_status_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_runtime_status_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_runtime_status_test PRIVATE Threads::Threads gufo_logging)
add_test(NAME npu_runtime_status_test COMMAND npu_runtime_status_test)
set_tests_properties(npu_runtime_status_test PROPERTIES TIMEOUT 30)
add_executable(npu_mem_trace_test mem_trace_test.cpp
  ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_mem_trace_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_mem_trace_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_mem_trace_test PRIVATE Threads::Threads gufo_logging)
add_test(NAME npu_mem_trace_test COMMAND npu_mem_trace_test)
set_tests_properties(npu_mem_trace_test PROPERTIES TIMEOUT 60)
add_executable(npu_mem_trace_real_test mem_trace_real_test.cpp
  ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_mem_trace_real_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_mem_trace_real_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_mem_trace_real_test PRIVATE Threads::Threads gufo_logging)
add_test(NAME npu_mem_trace_real_test COMMAND npu_mem_trace_real_test)
set_tests_properties(npu_mem_trace_real_test PROPERTIES TIMEOUT 60)
add_test(NAME npu_chip_power_test COMMAND npu_chip_power_test)
set_tests_properties(npu_chip_power_test PROPERTIES TIMEOUT 30)
add_executable(npu_chip_power_real_test chip_power_real_test.cpp
  ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_chip_power_real_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_compile_definitions(npu_chip_power_real_test PRIVATE GUFO_NPU_TESTING)
target_link_libraries(npu_chip_power_real_test PRIVATE Threads::Threads gufo_logging)
add_test(NAME npu_chip_power_real_test COMMAND npu_chip_power_real_test)
set_tests_properties(npu_chip_power_real_test PROPERTIES TIMEOUT 60)
add_test(NAME npu_hal_source_real_test COMMAND npu_hal_source_real_test)
set_tests_properties(npu_hal_source_real_test PROPERTIES TIMEOUT 30)
add_test(NAME npu_hal_fault_monitor_test COMMAND npu_hal_fault_monitor_test)
set_tests_properties(npu_hal_fault_monitor_test PROPERTIES TIMEOUT 30)

add_executable(npu_host_bench host_bench.cpp fake_xrt.cpp ${NPU_DIR}/private_bo_adapter.cpp ${NPU_DIR}/bfp.cpp)
target_include_directories(npu_host_bench PRIVATE "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(npu_host_bench SYSTEM PRIVATE "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
target_compile_definitions(npu_host_bench PRIVATE XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
target_link_libraries(npu_host_bench PRIVATE gufo_core)

# fake_hip needs the HIP SDK headers; when HIP is off, only the HIP-backed
# targets are skipped (hosted CPU presets build the rest).
if(TARGET hip::amdhip64)
add_executable(npu_session_test session_test.cpp fake_xrt.cpp
  fake_hip.cpp ${NPU_DIR}/linear_z_hip.cpp ${NPU_DIR}/projection.cpp
  ${NPU_DIR}/private_bo_adapter.cpp ${NPU_DIR}/session.cpp ${NPU_DIR}/server_prefill.cpp
  ${NPU_DIR}/linear_z.cpp ${NPU_DIR}/staging.cpp ${NPU_DIR}/bfp.cpp
  ${GUFO_NPU_FAULT_SOURCES})
target_include_directories(npu_session_test PRIVATE "${CMAKE_SOURCE_DIR}")
target_include_directories(npu_session_test SYSTEM PRIVATE "${GUFO_XRT_GENERATED_INCLUDE_DIR}" "${GUFO_XRT_INCLUDE_DIR}")
target_compile_definitions(npu_session_test PRIVATE XRT_STATIC_BUILD _ITERATOR_DEBUG_LEVEL=0 GUFO_NPU_TESTING)
target_link_libraries(npu_session_test PRIVATE gufo_core gufo_logging)
# SDK headers only: fake_hip.cpp supplies symbols; never link hip::host.
get_target_property(npu_hip_headers hip::amdhip64 INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(npu_session_test SYSTEM PRIVATE ${npu_hip_headers})
target_compile_definitions(npu_session_test PRIVATE __HIP_PLATFORM_AMD__=1)
foreach(npu_unit npu_linear_z_test npu_projection_test npu_guard_latch_test)
  target_sources(${npu_unit} PRIVATE fake_hip.cpp
    ${NPU_DIR}/linear_z_hip.cpp ${NPU_DIR}/server_prefill.cpp ${GUFO_NPU_FAULT_SOURCES})
  target_include_directories(${npu_unit} SYSTEM PRIVATE ${npu_hip_headers})
  target_compile_definitions(${npu_unit} PRIVATE __HIP_PLATFORM_AMD__=1)
endforeach()
target_compile_options(npu_session_test PRIVATE -include "${CMAKE_CURRENT_SOURCE_DIR}/file_io_hook.hpp")
foreach(npu_case clean timeout error layer single stream_timeout invalid
    startup_missing_dir startup_missing_insts startup_missing_xclbin startup_hash
    startup_version startup_runtime_missing startup_device startup_pre_failed denied
    power_unavailable power_turbo hal_event invalid_hal)
  add_test(NAME npu_session_${npu_case}_test COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/failure_log_test.py" "$<TARGET_FILE:npu_session_test>" "${npu_case}" "${GUFO_XRT_TOOLING_ROOT}")
  set_tests_properties(npu_session_${npu_case}_test PROPERTIES TIMEOUT 120)
endforeach()
add_custom_target(npu_cpu_checks DEPENDS npu_session_test npu_private_bo_adapter_test
  npu_linear_z_test npu_projection_test npu_guard_latch_test
  npu_bfp_test npu_host_bench npu_hal_fault_monitor_test npu_hal_source_real_test npu_runtime_status_test npu_bo_address_test npu_chunk_retry_test npu_platform_info_test npu_guard_test npu_mem_trace_test npu_mem_trace_real_test npu_chip_power_test
  npu_chip_power_real_test npu_runtime)
if(TARGET npu_bfp_hip)
  add_dependencies(npu_cpu_checks npu_bfp_hip)
endif()
endif(TARGET hip::amdhip64)

# vcpkg OpenSSL links dynamically on Windows; tests live outside the DLLs
# CMake otherwise places next to only a few targets.
if(WIN32 AND EXISTS "${OPENSSL_INCLUDE_DIR}")
  cmake_path(GET OPENSSL_INCLUDE_DIR PARENT_PATH _npu_openssl_prefix)
  set(_npu_openssl_dll "${_npu_openssl_prefix}/bin/libcrypto-3-x64.dll")
  if(EXISTS "${_npu_openssl_dll}")
    foreach(npu_linked_target npu_xrt_delay_guard_test npu_private_bo_adapter_test
        npu_linear_z_test npu_guard_latch_test npu_projection_test npu_session_test)
      if(TARGET ${npu_linked_target})
        add_custom_command(TARGET ${npu_linked_target} POST_BUILD COMMAND
          "${CMAKE_COMMAND}" -E copy_if_different "${_npu_openssl_dll}"
          "$<TARGET_FILE_DIR:${npu_linked_target}>")
      endif()
    endforeach()
  endif()
endif()
