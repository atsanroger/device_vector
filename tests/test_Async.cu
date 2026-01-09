#include <iostream>
#include <cuda_runtime.h>
#include "DeviceVector.cuh"
#include "DeviceEnv.cuh"

using namespace GPU;

template <typename T>
using DeviceVector = DeviceVectorImpl<T, PageableAllocator<T>>;

// A slow kernel to simulate workload (busy wait)
__global__ void slow_set_kernel(float* data, size_t n, float val, long long iterations) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Busy wait to ensure host runs ahead if not synchronized
        long long start = clock64();
        while (clock64() - start < iterations); 
        data[idx] = val;
    }
}

void run_async_test() {
    std::cout << ">> [Principle 4] Asynchronous Consistency Test" << std::endl;

    size_t n = 1000;
    DeviceVector<float> vec(n);
    vec.set_value(0.0f); // Init to 0

    std::cout << "   Launching slow kernel (Async)..." << std::endl;
    
    cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
    // Delay for ~1 million cycles
    slow_set_kernel<<<1, 1, 0, stream>>>(vec.device_ptr(), 1, 999.0f, 1000000);

    std::cout << "   Reading back to host immediately..." << std::endl;
    
    // This MUST internally block/sync until the kernel finishes
    vec.update_host(); 
    
    float val = vec.host_ptr()[0];
    std::cout << "   Value read: " << val << std::endl;

    if (val == 999.0f) {
        std::cout << "   [PASS] Update synchronized correctly." << std::endl;
    } else {
        std::cerr << "   [FAIL] Race Condition! Read old value (" << val << ") instead of 999.0." << std::endl;
        exit(1);
    }
}

int main() {
    try {
        DeviceEnv::instance().init();
        run_async_test();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}