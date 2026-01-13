#!/bin/bash

BUILD_DIR="build"
set -e

echo "=========================================="
echo "🎯 Select Language:"
echo "   1) Fortran"
echo "   2) C++"
echo "=========================================="
read -p "Enter choice [1-2]: " LANG_NUM

echo ""
echo "=========================================="
echo "🎯 Select Execution Mode:"
echo "   1) Run Tests"
echo "   2) Run Profile (Auto-detects files)"
echo "   3) Skip Execution"
echo "=========================================="
read -p "Enter choice [1-3]: " EXEC_MODE

echo "=========================================="
echo "🔧 Starting Build Process..."
echo "=========================================="

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
mkdir "$BUILD_DIR"

cd "$BUILD_DIR"

echo "⚙️  Configuring..."
FC=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvfortran \
CXX=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvc++ \
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DENABLE_OPENACC=ON

echo "🔨 Compiling..."

TARGETS=""
if [[ "$EXEC_MODE" == "1" ]]; then
    # 模式 1：只編譯測試
    TARGETS="run_tests"
elif [[ "$EXEC_MODE" == "2" ]]; then
    # 模式 2：自動掃描 ../profile 資料夾下的所有 .F90 檔案
    # 並將其檔名 (不含副檔名) 作為 make 的目標
    echo "   -> Scanning profile directory for targets..."
    for f in ../profile/*.F90; do
        if [ -e "$f" ]; then
            filename=$(basename "$f" .F90)
            TARGETS="$TARGETS $filename"
        fi
    done
    echo "   -> Detected Profile Targets: $TARGETS"
else
    # 模式 3：預設編譯所有東西
    TARGETS="all"
fi

# 執行 make
make -j$(nproc) $TARGETS

echo "✅ Build Successful!"
echo "=========================================="

if [[ "$EXEC_MODE" == "1" ]]; then
    echo "🚀 Running Standard Tests..."
    [ -f "./run_tests" ] && ./run_tests || echo "❌ Error: run_tests not found."

elif [[ "$EXEC_MODE" == "2" ]]; then
    # Profile 執行選單
    if [[ "$LANG_NUM" == "1" ]]; then
        # 重新掃描一次以建立選單列表
        RAW_FILES=(../profile/*.F90)
        
        if [ ! -e "${RAW_FILES[0]}" ]; then
            echo "❌ Error: No .F90 files found in ../profile/"
        else
            echo "------------------------------------------"
            echo "Select a profile to run:"
            # 動態列出所有抓到的檔案
            for i in "${!RAW_FILES[@]}"; do
                echo "   $((i+1))) $(basename "${RAW_FILES[$i]}" .F90)"
            done
            echo "------------------------------------------"
            read -p "Enter choice [1-${#RAW_FILES[@]}]: " P_IDX
            
            # 取得使用者選擇的檔案名稱
            SELECTED_BASE=$(basename "${RAW_FILES[$((P_IDX-1))]}" .F90)
            
            echo "🚀 Running $SELECTED_BASE ..."
            [ -f "./$SELECTED_BASE" ] && ./"$SELECTED_BASE" || echo "❌ Error: ./$SELECTED_BASE not found."
        fi
    fi
fi