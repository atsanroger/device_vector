cmake_minimum_required(VERSION 3.20)
project(DeviceVector_Fortran LANGUAGES CXX Fortran CUDA)

# ==========================================
# 1. 全域設定
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

# [RTX 4090 專用暴力 Flags]
set(CUDA_HARDCODED_FLAGS "-gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_HARDCODED_FLAGS}")

# [🔥🔥🔥 核心修正 🔥🔥🔥] 
# 同時開啟 -acc 與 -cuda。
# -acc: 處理 OpenACC 指令
# -cuda: 處理 CUDA Fortran (ATTRIBUTES, DEVICE, <<<>>>)
set(FORTRAN_GPU_FLAGS "-acc" "-cuda" "-gpu=cc89")

# ==========================================
# 2. 核心程式庫
# ==========================================
include_directories(src/cuda src/include src/openacc)

add_library(DeviceVector_Lib STATIC
    src/cuda/c_interface.cu
    src/cuda/DeviceEnv.cu
    src/openacc/acc_interop.cpp
)
set_target_properties(DeviceVector_Lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Fortran 介面
add_library(DeviceVector_Mod STATIC
    src/fortran/Device_Interface_Openacc.F90
    src/fortran/Device_Interface.F90
)

# [這裡修正] 將 -cuda 注入編譯階段
target_compile_options(DeviceVector_Mod PUBLIC $<$<COMPILE_LANGUAGE:Fortran>:${FORTRAN_GPU_FLAGS}>)
target_link_libraries(DeviceVector_Mod PUBLIC DeviceVector_Lib)

# ==========================================
# 3. Tests & Profiles (合併邏輯)
# ==========================================
set(ALL_SOURCES ${TEST_SOURCES} ${PROFILE_SOURCES})

# 這裡建立一個清單方便處理
file(GLOB TEST_SOURCES "tests/*.F90")
file(GLOB PROFILE_SOURCES "profile/*.F90")

macro(ADD_FORTRAN_GPU_EXE target_name source_file)
    add_executable(${target_name} ${source_file})
    
    # 連結 DeviceVector_Mod (它已經帶有 -acc -cuda)
    target_link_libraries(${target_name} PUBLIC 
        DeviceVector_Mod
        "-Wl,--whole-archive" DeviceVector_Lib "-Wl,--no-whole-archive"
        "cudart"
    )

    set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE Fortran)
    
    # [這裡修正] 強制給予編譯選項
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:${FORTRAN_GPU_FLAGS}>)

    if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
        # 再次強調連結時也要帶上，確保 runtime 對齊
        set_property(TARGET ${target_name} PROPERTY LINK_FLAGS "-acc -cuda -gpu=cc89 -c++libs")
    endif()
endmacro()

# 批量生成
foreach(source ${TEST_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    ADD_FORTRAN_GPU_EXE(${name} ${source})
endforeach()

foreach(source ${PROFILE_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    ADD_FORTRAN_GPU_EXE(${name} ${source})
endforeach()

message(STATUS "Auto-detected Tests: ${TEST_SOURCES}")
message(STATUS "Auto-detected Profiles: ${PROFILE_SOURCES}")