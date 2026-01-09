#include <iostream>
#include <cuda_runtime.h>
#include "DeviceVector.cuh"
#include "DeviceEnv.cuh"

using namespace GPU;

// Alias for easier usage
template <typename T>
using DeviceVector = DeviceVectorImpl<T, PageableAllocator<T>>;

void run_lifecycle_test() {
    std::cout << ">> [Principle 1] Lifecycle Safety Test (Deep Copy & Scope)" << std::endl;

    // 1. Initialize Vector A
    DeviceVector<float> vec_a(100);
    vec_a.set_value(1.0f);
    
    float sum_a_initial = vec_a.sum();
    std::cout << "   Vector A Initial Sum: " << sum_a_initial << std::endl;

    {
        std::cout << "   --- Entering Inner Scope ---" << std::endl;
        
        // 2. Clone Vector A to Vector B (Deep Copy)
        // Since copy constructor is deleted, we use clone() interface.
        IDeviceVector<float>* vec_b_ptr = vec_a.clone();
        
        // 3. Modify Vector B
        std::cout << "   Modifying Vector B (Set to 2.0)..." << std::endl;
        vec_b_ptr->set_value(2.0f);

        // Check B
        float sum_b = vec_b_ptr->sum();
        if (abs(sum_b - 200.0f) > 0.1f) {
            std::cerr << "   [FAIL] Vector B modification failed!" << std::endl;
            exit(1);
        }

        // 4. Verify Vector A is UNTOUCHED
        // If Deep Copy failed (Shallow Copy), A would have been modified too.
        float sum_a_check = vec_a.sum();
        if (abs(sum_a_check - 100.0f) > 0.1f) {
            std::cerr << "   [FAIL] Shallow Copy Detected! Vector A was modified by B." << std::endl;
            exit(1);
        }

        // 5. Destroy Vector B
        delete vec_b_ptr;
        std::cout << "   --- Exiting Inner Scope (Vector B Destroyed) ---" << std::endl;
    }

    // 6. Verify Vector A is still alive
    // If Double Free occurred, A's pointer would be dangling now.
    float sum_a_final = vec_a.sum();
    
    if (abs(sum_a_final - 100.0f) < 0.1f) {
        std::cout << "   [PASS] Vector A survived Vector B's destruction." << std::endl;
    } else {
        std::cerr << "   [FAIL] Vector A corrupted after scope exit!" << std::endl;
        exit(1);
    }
}

int main() {
    try {
        DeviceEnv::instance().init();
        run_lifecycle_test();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}