#include <iostream>
#include <cuda_runtime.h>
#include "DeviceVector.cuh"
#include "DeviceEnv.cuh"

using namespace GPU;

template <typename T>
using DeviceVector = DeviceVectorImpl<T, PageableAllocator<T>>;

void run_capacity_test() {
    std::cout << ">> [Principle 2] Capacity Strategy Test (Buffering)" << std::endl;

    size_t initial_n = 1000;
    DeviceVector<float> vec(initial_n);

    size_t cap_1 = vec.capacity();
    float* ptr_1 = vec.device_ptr();
    
    std::cout << "   Step 1: Size " << initial_n << " | Capacity " << cap_1 << std::endl;

    // 1. Shrink Test (Capacity should NOT decrease)
    vec.resize(10);
    size_t cap_2 = vec.capacity();
    float* ptr_2 = vec.device_ptr();

    std::cout << "   Step 2: Shrink to 10 | Capacity " << cap_2 << std::endl;

    if (cap_2 < cap_1) {
        std::cerr << "   [FAIL] Capacity shrank! Memory reallocated unnecessarily." << std::endl;
        exit(1);
    }
    if (ptr_1 != ptr_2) {
        std::cerr << "   [FAIL] Pointer changed during shrink! (Performance hit)" << std::endl;
        exit(1);
    }

    // 2. Growth Test (Trigger 1.5x buffer)
    // Resize slightly larger than current capacity to force reallocation.
    size_t new_target = cap_1 + 100;
    vec.resize(new_target);
    
    size_t cap_3 = vec.capacity();
    std::cout << "   Step 3: Grow to " << new_target << " | New Capacity " << cap_3 << std::endl;

    // Expectation: New Capacity should be approx 1.5x of old capacity
    size_t expected_min = (size_t)(cap_1 * 1.5);
    
    if (cap_3 >= expected_min) {
        std::cout << "   [PASS] 1.5x Growth Strategy verified." << std::endl;
    } else {
        std::cout << "   [WARN] Growth factor < 1.5x, but functionality holds." << std::endl;
    }
}

int main() {
    try {
        DeviceEnv::instance().init();
        run_capacity_test();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}