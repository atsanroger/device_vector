cmake_minimum_required(VERSION 3.20)
project(DeviceVector_Fortran LANGUAGES CXX Fortran CUDA)

# ==========================================
# 1. 徹底歸零所有 CMake 內建的隱藏 Flag
# ==========================================
set(CMAKE_Fortran_FLAGS "")
set(CMAKE_Fortran_FLAGS_RELEASE "")
set(CMAKE_Fortran_FLAGS_DEBUG "")
set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "")

# ==========================================
# 2. 全域 GPU 設定
# ==========================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

set(FORTRAN_GPU_BASE "-v" "-acc" "-cuda" "-gpu=cc89" "-Minfo=all" "-v" "-mp" "-Mnoinline" "-O0")
# ==========================================
# 3. 核心程式庫 DeviceVector
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

# 核心庫使用
target_compile_options(DeviceVector_Mod PUBLIC 
    $<$<COMPILE_LANGUAGE:Fortran>:${FORTRAN_GPU_BASE};>
)
target_link_libraries(DeviceVector_Mod PUBLIC DeviceVector_Lib)

# ==========================================
# 4. Tests & Profiles 巨集 (保持通用)
# ==========================================

macro(ADD_FORTRAN_GPU_EXE target_name source_files)
    # [修正] 這裡使用傳入的參數 source_files，支援單檔或多檔
    add_executable(${target_name} ${source_files})
    
    target_compile_options(${target_name} PRIVATE 
        $<$<COMPILE_LANGUAGE:Fortran>:${FORTRAN_GPU_BASE};${LOCAL_OPT}>
    )

    target_link_libraries(${target_name} PUBLIC 
        DeviceVector_Mod
        "-Wl,--whole-archive" DeviceVector_Lib "-Wl,--no-whole-archive"
        "cudart"
    )

    set_target_properties(${target_name} PROPERTIES LINKER_LANGUAGE Fortran)
    
    if(CMAKE_Fortran_COMPILER_ID MATCHES "NVHPC")
        string(REPLACE ";" " " LINK_STR "${FORTRAN_GPU_BASE}")
        set_property(TARGET ${target_name} PROPERTY LINK_FLAGS "${LINK_STR} ${LOCAL_OPT} -c++libs")
    endif()
endmacro()

# ==========================================
# 5. 自動掃描與個別處理
# ==========================================
file(GLOB TEST_SOURCES "tests/*.F90")
file(GLOB PROFILE_SOURCES "profile/*.F90")

# --- 處理一般單檔測試 ---
foreach(source ${TEST_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    
    # [重要修正] 跳過 test_rk4_kinaco，因為它需要特殊的多檔案編譯，不能在這裡處理
    if(name STREQUAL "test_rk4_kinaco")
        continue()
    endif()

    ADD_FORTRAN_GPU_EXE(${name} ${source})
endforeach()

# --- 處理 Profile ---
foreach(source ${PROFILE_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    ADD_FORTRAN_GPU_EXE(${name} ${source})
endforeach()

# ==========================================
# 6. 特殊 Target: test_rk4_kinaco
# ==========================================
set(RK4_KINACO_SRCS
    src/particle/rand.F90 
    src/particle/cross_ckeck_logic.F90 
    src/particle/particle_init.F90
    src/particle/physics_mod.F90
    src/particle/ppmain.F90 
    tests/test_rk4_kinaco.F90
)

ADD_FORTRAN_GPU_EXE(test_rk4_kinaco "${RK4_KINACO_SRCS}")

set_source_files_properties(tests/test_rk4_dv.F90 PROPERTIES COMPILE_OPTIONS "-acc;-cuda;-gpu=cc89;-O0")