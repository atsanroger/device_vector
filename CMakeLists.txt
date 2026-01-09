cmake_minimum_required(VERSION 3.18)
project(GPU_Fortran_Lib LANGUAGES CXX CUDA Fortran)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_compile_options($<$<COMPILE_LANGUAGE:Fortran>:-cuda>)
add_compile_options(-O3)

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)

file(GLOB CUDA_SOURCES "src/*.cu")
add_library(gpu_backend SHARED ${CUDA_SOURCES})
set_target_properties(gpu_backend PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(gpu_backend PRIVATE CUDA::cudart)

add_library(gpu_interface STATIC src/Device_Interface.F90)
target_link_libraries(gpu_interface PUBLIC gpu_backend)

option(BUILD_TESTS "Build the test suite" ON)

if(BUILD_TESTS)
    include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

    add_executable(run_tests tests/Test_Suite.F90)
    target_link_libraries(run_tests PRIVATE gpu_interface CUDA::cudart)
    set_target_properties(run_tests PROPERTIES LINK_FLAGS "-cuda")
    target_link_options(run_tests PRIVATE "-Wl,-rpath,$<TARGET_FILE_DIR:CUDA::cudart>")

    file(GLOB PROFILE_SOURCES "profile/*.F90")
    foreach(PROFILE_SRC ${PROFILE_SOURCES})
        get_filename_component(PROFILE_NAME ${PROFILE_SRC} NAME_WE)
        
        add_executable(${PROFILE_NAME} ${PROFILE_SRC})
        target_link_libraries(${PROFILE_NAME} PRIVATE gpu_interface CUDA::cudart)
        
        set_target_properties(${PROFILE_NAME} PROPERTIES 
            Fortran_FORMAT FREE
            LINK_FLAGS "-cuda"
        )
        
        target_link_options(${PROFILE_NAME} PRIVATE "-Wl,-rpath,$<TARGET_FILE_DIR:CUDA::cudart>")
        
        message(STATUS "Configured Profile: ${PROFILE_NAME}")
    endforeach()
endif()