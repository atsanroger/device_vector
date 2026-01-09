cmake_minimum_required(VERSION 3.18)
project(GPU_Fortran_Lib LANGUAGES CXX CUDA Fortran)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")

add_compile_options(-O3)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

file(GLOB CUDA_SOURCES "src/*.cu")

add_library(gpu_backend SHARED ${CUDA_SOURCES})
set_target_properties(gpu_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_library(gpu_interface STATIC src/Device_Interface.F90)

target_link_libraries(gpu_interface PUBLIC gpu_backend)
target_link_libraries(gpu_backend PRIVATE CUDA::cudart)

option(BUILD_TESTS "Build the test suite" ON)

if(BUILD_TESTS)
    add_executable(run_tests tests/Test_Suite.F90)
    target_link_libraries(run_tests PRIVATE gpu_interface)
    target_link_libraries(run_tests PRIVATE CUDA::cudart)
    target_link_options(run_tests PRIVATE "-Wl,-rpath,$<TARGET_FILE_DIR:CUDA::cudart>")
endif()