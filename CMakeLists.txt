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

# [修正點] 使用分號分隔 List，確保 CMake 正確拆解參數
set(FORTRAN_GPU_BASE "-acc" "-cuda" "-gpu=cc89" "-Minfo=accel")

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

# 核心庫使用 -O2
target_compile_options(DeviceVector_Mod PUBLIC 
    $<$<COMPILE_LANGUAGE:Fortran>:${FORTRAN_GPU_BASE};-O2>
)
target_link_libraries(DeviceVector_Mod PUBLIC DeviceVector_Lib)

# ==========================================
# 4. Tests & Profiles 批量生成
# ==========================================

macro(ADD_FORTRAN_GPU_EXE target_name source_file)
    add_executable(${target_name} ${source_file})
    
    # 判斷是否需要降級優化
    if("${source_file}" MATCHES "test_rk4_dv.F90" OR "${source_file}" MATCHES "profile_rk4_interp.F90")
        set(LOCAL_OPT "-O1")
    else()
        set(LOCAL_OPT "-O2")
    endif()

    # [修正點] 確保這裡也是獨立參數傳遞
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
        # Link flags 必須是空白分隔的字串
        string(REPLACE ";" " " LINK_STR "${FORTRAN_GPU_BASE}")
        set_property(TARGET ${target_name} PROPERTY LINK_FLAGS "${LINK_STR} ${LOCAL_OPT} -c++libs")
    endif()
endmacro()

# 檔案掃描與生成
file(GLOB TEST_SOURCES "tests/*.F90")
file(GLOB PROFILE_SOURCES "profile/*.F90")

foreach(source ${TEST_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    ADD_FORTRAN_GPU_EXE(${name} ${source})
endforeach()

foreach(source ${PROFILE_SOURCES})
    get_filename_component(name ${source} NAME_WE)
    ADD_FORTRAN_GPU_EXE(${name} ${source})
endforeach()

set_source_files_properties(tests/test_rk4_dv.F90 PROPERTIES COMPILE_OPTIONS "-acc;-cuda;-gpu=cc89;-O0")