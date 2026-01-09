#!/bin/bash

BUILD_DIR="build"
set -e

# ==========================================
# 1. 語言選單
# ==========================================
echo "=========================================="
echo "🎯 Select Language:"
echo "   1) Fortran"
echo "   2) C++"
echo "=========================================="
read -p "Enter choice [1-2]: " LANG_NUM

# ==========================================
# 2. 執行模式選單
# ==========================================
echo ""
echo "=========================================="
echo "🎯 Select Execution Mode:"
echo "   1) Run Tests"
echo "   2) Run Profile"
echo "   3) Skip Execution"
echo "=========================================="
read -p "Enter choice [1-3]: " EXEC_MODE

echo "=========================================="
echo "🔧 Starting Build Process..."
echo "=========================================="

# ⚠️ 關鍵：如果 build 資料夾已經存在，先砍掉確保 CMake 重新偵測編譯器
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
mkdir "$BUILD_DIR"

cd "$BUILD_DIR"

echo "⚙️  Configuring..."
# ⚠️ 這裡直接把你的 nvfortran 路徑鎖死給 CMake
FC=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvfortran \
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

echo "🔨 Compiling..."
make -j$(nproc)

echo "✅ Build Successful!"
echo "=========================================="

# ==========================================
# 3. 執行判斷邏輯 (後續邏輯維持不變)
# ==========================================
if [[ "$EXEC_MODE" == "1" ]]; then
    echo "🚀 Running Standard Tests..."
    [ -f "./run_tests" ] && ./run_tests || echo "❌ Error: run_tests not found."

elif [[ "$EXEC_MODE" == "2" ]]; then
    if [[ "$LANG_NUM" == "1" ]]; then
        RAW_FILES=(../profile/*.F90)
        if [ ! -e "${RAW_FILES[0]}" ]; then
            echo "❌ Error: No .F90 files found in ../profile/"
        else
            echo "------------------------------------------"
            echo "Select a profile to run:"
            for i in "${!RAW_FILES[@]}"; do
                echo "   $((i+1))) $(basename "${RAW_FILES[$i]}" .F90)"
            done
            echo "------------------------------------------"
            read -p "Enter choice [1-${#RAW_FILES[@]}]: " P_IDX
            SELECTED_BASE=$(basename "${RAW_FILES[$((P_IDX-1))]}" .F90)
            [ -f "./$SELECTED_BASE" ] && ./"$SELECTED_BASE" || echo "❌ Error: ./$SELECTED_BASE not found."
        fi
    fi
fi