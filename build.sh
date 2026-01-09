#!/bin/bash

BUILD_DIR="build"
set -e

echo -n "Select Language (fortran/c++): "
read LANG_CHOICE

echo -n "Run Tests? (y/n): "
read RUN_CHOICE

echo "=========================================="
echo "🔧 Starting Build Process..."
echo "=========================================="

if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

cd "$BUILD_DIR"

echo "⚙️  Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

echo "🔨 Compiling..."
make -j$(nproc)

echo "✅ Build Successful!"
echo "=========================================="


if [[ "$RUN_CHOICE" == "y" || "$RUN_CHOICE" == "Y" ]]; then

    if [[ "$LANG_CHOICE" == "fortran" || "$LANG_CHOICE" == "f" ]]; then
        echo "🚀 Running Fortran Tests..."
        if [ -f "./run_tests" ]; then
            ./run_tests
        else
            echo "❌ Error: ./run_tests not found."
        fi

    elif [[ "$LANG_CHOICE" == "c++" || "$LANG_CHOICE" == "cpp" || "$LANG_CHOICE" == "c" ]]; then
        echo "🚀 Running C++ Tests..."
        if [ -f "./run_tests" ]; then
            ./run_tests
        else
            echo "⚠️  C++ test executable (run_tests) not found."
        fi

    else
        echo "❌ Unknown language selection."
    fi

else
    echo "ℹ️  Skipping tests."
fi