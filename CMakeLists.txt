cmake_minimum_required(VERSION 3.18)
project(GPU_Fortran_Lib LANGUAGES CXX CUDA Fortran)

find_package(CUDAToolkit REQUIRED)

option(ENABLE_OPENACC "Enable OpenACC interop layer" ON)
option(BUILD_TESTS "Build the test suite" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# 全域 Fortran 編譯選項
add_compile_options($<$<COMPILE_LANGUAGE:Fortran>:-cuda>)
if(ENABLE_OPENACC)
  add_compile_options($<$<COMPILE_LANGUAGE:Fortran>:-acc>)
endif()

# 優化選項
add_compile_options(-O3)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

# ------------------------------
# 1. GPU Backend (C++ / CUDA)
# ------------------------------
file(GLOB CUDA_SOURCES "src/cuda/*.cu")

add_library(gpu_backend SHARED ${CUDA_SOURCES})
set_target_properties(gpu_backend PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(gpu_backend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_link_libraries(gpu_backend PRIVATE CUDA::cudart)

if(ENABLE_OPENACC)
  target_sources(gpu_backend PRIVATE src/openacc/acc_interop.cpp)
  set_source_files_properties(src/openacc/acc_interop.cpp PROPERTIES COMPILE_OPTIONS "-acc")
  target_link_options(gpu_backend PRIVATE $<$<LINK_LANGUAGE:CXX>:-acc>)
  set_target_properties(gpu_backend PROPERTIES LINKER_LANGUAGE CXX)
endif()

# ------------------------------
# 2. Fortran Interface
# ------------------------------
add_library(gpu_interface STATIC src/fortran/Device_Interface.F90)
target_link_libraries(gpu_interface PUBLIC gpu_backend)
target_include_directories(gpu_interface PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY})

if(ENABLE_OPENACC)
  target_sources(gpu_interface PRIVATE src/fortran/Device_Interface_Openacc.F90)
endif()

# ------------------------------
# 3. Tests & Profiles (Auto-Discovery)
# ------------------------------
if(BUILD_TESTS)
  # Standard Unit Test
  add_executable(run_tests tests/Test_Suite.F90)
  target_link_libraries(run_tests PRIVATE gpu_interface CUDA::cudart)
  set_target_properties(run_tests PROPERTIES LINK_FLAGS "-cuda")
  target_link_options(run_tests PRIVATE "-Wl,-rpath,$<TARGET_FILE_DIR:CUDA::cudart>")
  if(ENABLE_OPENACC)
    target_link_options(run_tests PRIVATE "-acc")
  endif()

  # Dynamic Profiles: 自動抓取 profile 資料夾下的所有 .F90
  file(GLOB PROFILE_SOURCES "profile/*.F90")
  
  foreach(PROFILE_SRC ${PROFILE_SOURCES})
    # 取得檔名 (不含路徑和副檔名)，例如 profile_memaccess
    get_filename_component(PROFILE_NAME ${PROFILE_SRC} NAME_WE)
    
    # 建立執行檔 Target
    add_executable(${PROFILE_NAME} ${PROFILE_SRC})
    
    # 連結必要的庫
    target_link_libraries(${PROFILE_NAME} PRIVATE gpu_interface CUDA::cudart)
    
    # 設定編譯屬性和 RPATH
    set_target_properties(${PROFILE_NAME} PROPERTIES Fortran_FORMAT FREE LINK_FLAGS "-cuda")
    target_link_options(${PROFILE_NAME} PRIVATE "-Wl,-rpath,$<TARGET_FILE_DIR:CUDA::cudart>")

    # 如果開啟 OpenACC，補上 flag
    if(ENABLE_OPENACC)
      target_link_options(${PROFILE_NAME} PRIVATE "-acc")
    endif()
    
    message(STATUS "👉 Configured Profile Target: ${PROFILE_NAME}")
  endforeach()
endif()