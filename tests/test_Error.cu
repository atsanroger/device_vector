#include <iostream>
#include <cuda_runtime.h>
#include <stdexcept>
#include "DeviceVector.cuh"
#include "DeviceEnv.cuh"

using namespace GPU;

template <typename T>
using DeviceVector = DeviceVectorImpl<T, PageableAllocator<T>>;

void run_error_test() {
    std::cout << ">> [Principle 5] Error Handling & Exception Safety Test" << std::endl;

    // 1. Setup a valid vector
    DeviceVector<float> vec(100);
    vec.set_value(7.0f);
    float* original_ptr = vec.device_ptr();

    std::cout << "   Original Vector Valid. Pointer: " << original_ptr << std::endl;

    // 2. Attempt a MASSIVE allocation that should fail
    // Asking for ~1 TB of memory
    size_t huge_size = 1024ULL * 1024ULL * 1024ULL * 250ULL; 
    
    std::cout << "   Attempting to resize to ~1 TB (Expect Failure)..." << std::endl;

    bool caught_exception = false;

    try {
        vec.resize(huge_size);
    } catch (const std::exception& e) {
        std::cout << "   [SUCCESS] Caught expected exception: " << e.what() << std::endl;
        caught_exception = true;
    }

    if (!caught_exception) {
        std::cerr << "   [WARN] Did not catch exception! (System might have Unified Memory enabled)" << std::endl;
    }

    // 3. Strong Exception Guarantee Check
    // Verify that the original data remains valid after a failed resize.
    // The class should NOT free the old memory if the new allocation failed.
    std::cout << "   Verifying original data integrity..." << std::endl;
    
    if (vec.logical_size() == 100) {
        float val = vec.sum();
        if (abs(val - 700.0f) < 0.1f) {
            std::cout << "   [PASS] Strong Exception Guarantee held. Data preserved." << std::endl;
        } else {
            std::cerr << "   [FAIL] Data corrupted after failed resize!" << std::endl;
            exit(1);
        }
    } else {
        std::cerr << "   [FAIL] Logical size changed despite failure!" << std::endl;
        exit(1);
    }
}

int main() {
    try {
        DeviceEnv::instance().init();
        run_error_test();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}