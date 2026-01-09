#include <iostream>
#include <cuda_runtime.h>
#include "DeviceVector.cuh"
#include "DeviceEnv.cuh"

using namespace GPU;

template <typename T>
using DeviceVector = DeviceVectorImpl<T, PageableAllocator<T>>;

// Kernel to manually corrupt the padding area
__global__ void corrupt_padding_kernel(float* data, size_t start_idx, size_t end_idx, float junk_val) {
    size_t idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < end_idx) {
        data[idx] = junk_val;
    }
}

void run_alignment_test() {
    std::cout << ">> [Principle 3] Alignment & Padding Test" << std::endl;

    // 1. Create a size NOT aligned to 32 (Warp size)
    // Logical size 33 -> Warp aligned storage should be 64
    size_t n = 33;
    DeviceVector<float> vec(n);
    vec.fill_zero();

    size_t storage_n = vec.storage_size();
    std::cout << "   Logical: " << n << " | Storage: " << storage_n << std::endl;

    if (storage_n % 32 != 0) {
        std::cerr << "   [FAIL] Storage size is not Warp Aligned!" << std::endl;
        exit(1);
    }

    // 2. Padding Corruption Test
    if (storage_n > n) {
        std::cout << "   Injecting junk (10000.0) into padding area..." << std::endl;
        
        // Write garbage into [n, storage_n)
        corrupt_padding_kernel<<<1, 32>>>(vec.device_ptr(), n, storage_n, 10000.0f);
        cudaDeviceSynchronize();
        
        // 3. Test Sum (Should ignore padding)
        float sum = vec.sum(); 
        
        if (sum > 1.0f) {
            std::cerr << "   [FAIL] Sum included padding values! Result: " << sum << std::endl;
            exit(1);
        } else {
            std::cout << "   [PASS] Sum reduction correctly ignored padding." << std::endl;
        }

        // 4. Test Min (Should ignore padding or handle identity)
        vec.set_value(5.0f);
        float min_val = vec.min();
        
        if (min_val == 5.0f) {
             std::cout << "   [PASS] Min reduction correctly handled padding." << std::endl;
        } else {
             std::cerr << "   [FAIL] Min reduction failed. Got: " << min_val << std::endl;
             exit(1);
        }
    } else {
        std::cout << "   [SKIP] No padding generated (size matched alignment)." << std::endl;
    }
}

int main() {
    try {
        DeviceEnv::instance().init();
        run_alignment_test();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}