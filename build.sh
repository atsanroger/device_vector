#!/bin/bash

BUILD_DIR="build"
set -e

# ==========================================
# 1. 選單設定
# ==========================================
echo "=========================================="
echo "🎯 Select Language:"
echo "   1) Fortran "
echo "   2) C++     "
echo "=========================================="
read -p "Enter choice [1-2]: " LANG_NUM

# 設定副檔名
if [[ "$LANG_NUM" == "1" ]]; then
    EXT=".F90"
else
    EXT=".cpp"
fi

echo ""
echo "=========================================="
echo "🎯 Select Execution Mode:"
echo "   1) Run Tests   (Scans ../tests/)"
echo "   2) Run Profile (Scans ../profile/)"
echo "   3) Build Only  (Skip Execution)"
echo "=========================================="
read -p "Enter choice [1-3]: " EXEC_MODE

# ==========================================
# 2. 建置環境與 CMake
# ==========================================
echo "=========================================="
echo "🔧 Starting Build Process..."
echo "=========================================="

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

echo "⚙️  Configuring CMake..."
FC=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvfortran \
CXX=/opt/nvidia/hpc_sdk/Linux_x86_64/24.7/compilers/bin/nvc++ \
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON -DENABLE_OPENACC=ON

# ==========================================
# 3. 自動掃描與編譯目標
# ==========================================
TARGETS=""
SEARCH_DIR=""

if [[ "$EXEC_MODE" == "1" ]]; then
    SEARCH_DIR="../tests"
    echo " 🔍 Scanning TEST directory ($SEARCH_DIR) for *$EXT ..."
elif [[ "$EXEC_MODE" == "2" ]]; then
    SEARCH_DIR="../profile"
    echo " 🔍 Scanning PROFILE directory ($SEARCH_DIR) for *$EXT ..."
else
    TARGETS="all"
fi

if [[ "$EXEC_MODE" == "1" || "$EXEC_MODE" == "2" ]]; then
    shopt -s nullglob
    for f in "$SEARCH_DIR"/*"$EXT"; do
        filename=$(basename "$f" "$EXT")
        TARGETS="$TARGETS $filename"
    done
    shopt -u nullglob

    if [ -z "$TARGETS" ]; then
        echo " ⚠️  No files found in $SEARCH_DIR! Defaulting to 'all'."
        TARGETS="all"
    else
        echo " 📝 Detected Targets: $TARGETS"
    fi
fi

echo "🔨 Compiling..."
make -j$(nproc) $TARGETS

echo "✅ Build Successful!"
echo "=========================================="

# ==========================================
# 4. 執行選單 (照你的要求修改 Mode 1)
# ==========================================

# --- MODE 1: Run Tests (修改後：跟 Profile 一樣列出選單) ---
if [[ "$EXEC_MODE" == "1" ]]; then
    shopt -s nullglob
    RAW_FILES=(../tests/*"$EXT") # 這裡改掃 tests
    shopt -u nullglob

    if [ ${#RAW_FILES[@]} -eq 0 ]; then
        echo "❌ Error: No $EXT files found in ../tests/"
    else
        echo "------------------------------------------"
        echo "Select a TEST to run:"
        for i in "${!RAW_FILES[@]}"; do
            echo "   $((i+1))) $(basename "${RAW_FILES[$i]}" "$EXT")"
        done
        echo "------------------------------------------"
        read -p "Enter choice [1-${#RAW_FILES[@]}]: " P_IDX
        
        # 檢查輸入有效性
        if [[ "$P_IDX" =~ ^[0-9]+$ ]] && [ "$P_IDX" -ge 1 ] && [ "$P_IDX" -le "${#RAW_FILES[@]}" ]; then
             SELECTED_BASE=$(basename "${RAW_FILES[$((P_IDX-1))]}" "$EXT")
             echo "🚀 Running $SELECTED_BASE ..."
             [ -f "./$SELECTED_BASE" ] && ./"$SELECTED_BASE" || echo "❌ Error: ./$SELECTED_BASE not found."
        else
             echo "❌ Invalid selection."
        fi
    fi

# --- MODE 2: Run Profile (保留原樣) ---
elif [[ "$EXEC_MODE" == "2" ]]; then
    shopt -s nullglob
    RAW_FILES=(../profile/*"$EXT")
    shopt -u nullglob

    if [ ${#RAW_FILES[@]} -eq 0 ]; then
        echo "❌ Error: No $EXT files found in ../profile/"
    else
        echo "------------------------------------------"
        echo "Select a PROFILE to run:"
        for i in "${!RAW_FILES[@]}"; do
            echo "   $((i+1))) $(basename "${RAW_FILES[$i]}" "$EXT")"
        done
        echo "------------------------------------------"
        read -p "Enter choice [1-${#RAW_FILES[@]}]: " P_IDX
        
        if [[ "$P_IDX" =~ ^[0-9]+$ ]] && [ "$P_IDX" -ge 1 ] && [ "$P_IDX" -le "${#RAW_FILES[@]}" ]; then
            SELECTED_BASE=$(basename "${RAW_FILES[$((P_IDX-1))]}" "$EXT")
            echo "🚀 Running $SELECTED_BASE ..."
            [ -f "./$SELECTED_BASE" ] && ./"$SELECTED_BASE" || echo "❌ Error: ./$SELECTED_BASE not found."
        else
            echo "❌ Invalid selection."
        fi
    fi
fi