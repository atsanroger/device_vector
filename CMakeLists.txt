cmake_minimum_required(VERSION 3.18)
project(GPU_Fortran_Lib LANGUAGES CXX CUDA Fortran)

find_package(CUDAToolkit REQUIRED)

option(ENABLE_OPENACC "Enable OpenACC interop layer" ON)
option(BUILD_TESTS "Build the test suite" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Fortran flags:
# -cuda: enable CUDA Fortran / linking mode for nvfortran
# If you will compile OpenACC directives in Fortran, also add -acc.
add_compile_options($<$<COMPILE_LANGUAGE:Fortran>:-cuda>)
if(ENABLE_OPENACC)
  add_compile_options($<$<COMPILE_LANGUAGE:Fortran>:-acc>)
endif()

add_compile_options(-O3)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

# ------------------------------
# gpu_backend (CUDA + optional OpenACC bridge)
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
  # Add OpenACC interop source (compiled by CXX compiler, which should be nvc++)
  target_sources(gpu_backend PRIVATE src/openacc/acc_interop.cpp)

  set_source_files_properties(src/openacc/acc_interop.cpp PROPERTIES COMPILE_OPTIONS "-acc")
  target_link_options(gpu_backend PRIVATE $<$<LINK_LANGUAGE:CXX>:-acc>)
  set_target_properties(gpu_backend PROPERTIES LINKER_LANGUAGE CXX)
endif()

# ------------------------------
# Fortran interface
# ------------------------------
add_library(gpu_interface STATIC src/fortran/Device_Interface.F90)
target_link_libraries(gpu_interface PUBLIC gpu_backend)
target_include_directories(gpu_interface PUBLIC ${CMAKE_Fortran_MODULE_DIRECTORY})

# If you want the OpenACC wrapper module too, you can add it:
if(ENABLE_OPENACC)
  target_sources(gpu_interface PRIVATE src/fortran/Device_Interface_Openacc.F90)
endif()

# ------------------------------
# Tests / Profiles
# ------------------------------
if(BUILD_TESTS)
  add_executable(run_tests tests/Test_Suite.F90)
  target_link_libraries(run_tests PRIVATE gpu_interface CUDA::cudart)
  set_target_properties(run_tests PROPERTIES LINK_FLAGS "-cuda")
  target_link_options(run_tests PRIVATE "-Wl,-rpath,$<TARGET_FILE_DIR:CUDA::cudart>")

  file(GLOB PROFILE_SOURCES "profile/*.F90")
  foreach(PROFILE_SRC ${PROFILE_SOURCES})
    get_filename_component(PROFILE_NAME ${PROFILE_SRC} NAME_WE)
    add_executable(${PROFILE_NAME} ${PROFILE_SRC})
    target_link_libraries(${PROFILE_NAME} PRIVATE gpu_interface CUDA::cudart)
    set_target_properties(${PROFILE_NAME} PROPERTIES Fortran_FORMAT FREE LINK_FLAGS "-cuda")
    target_link_options(${PROFILE_NAME} PRIVATE "-Wl,-rpath,$<TARGET_FILE_DIR:CUDA::cudart>")
  endforeach()
endif()
