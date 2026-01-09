#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// 1. Include necessary headers
#include "DeviceVector.cuh" 
#include "DeviceEnv.cuh"     // <--- FIX: Include DeviceEnv header

// 2. Use the namespace defined in your header
using namespace GPU;

// 3. Define Alias
// Map the implementation class "DeviceVectorImpl" to a simple name "DeviceVector".
template <typename T>
using DeviceVector = DeviceVectorImpl<T, PageableAllocator<T>>;

// Test Kernel: Checks if data matches the expected value
__global__ void check_memory_kernel(float* data, size_t n, float expected_val, int* error_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple float comparison with tolerance
        if (abs(data[idx] - expected_val) > 1e-5) {
            atomicAdd(error_count, 1);
        }
    }
}

void run_tests() {
    int* d_errors;
    cudaMalloc(&d_errors, sizeof(int));
    cudaMemset(d_errors, 0, sizeof(int));
    
    std::cout << "=== DeviceVector Integration Test ===\n" << std::endl;

    // --- Test 1: Construction & Getters ---
    size_t logical_n = 100;
    std::cout << "[Test 1] Create Vector size " << logical_n << "..." << std::endl;
    
    // The constructor checks if DeviceEnv is initialized. 
    // If not initialized in main(), this line throws the exception you saw.
    DeviceVector<float> vec(logical_n); 
    
    std::cout << "  Logical Size: " << vec.logical_size() << std::endl;
    std::cout << "  Storage Size: " << vec.storage_size() << std::endl; 
    std::cout << "  Capacity    : " << vec.capacity() << std::endl;

    // --- Test 2: Set Value ---
    std::cout << "\n[Test 2] Set Value to 1.23f..." << std::endl;
    vec.set_value(1.23f);
    
    // Verify results on GPU
    check_memory_kernel<<<(logical_n + 255) / 256, 256>>>(vec.device_ptr(), logical_n, 1.23f, d_errors);
    cudaDeviceSynchronize();

    int h_errors = 0;
    cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "  Errors found: " << h_errors << (h_errors == 0 ? " [PASS]" : " [FAIL]") << std::endl;

    // --- Test 3: Resize & Growth Strategy ---
    std::cout << "\n[Test 3] Resize to 150 (Trigger 1.5x Growth)..." << std::endl;
    vec.resize(150);
    vec.set_value(5.55f); // Fill new size with value

    std::cout << "  New Capacity: " << vec.capacity() << std::endl;
    
    cudaMemset(d_errors, 0, sizeof(int));
    check_memory_kernel<<<(150 + 255) / 256, 256>>>(vec.device_ptr(), 150, 5.55f, d_errors);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_errors, d_errors, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "  Errors found: " << h_errors << (h_errors == 0 ? " [PASS]" : " [FAIL]") << std::endl;

    cudaFree(d_errors);
}

int main() {
    try {
        // =========================================================
        // FIX: Initialize Device Environment BEFORE running tests
        // =========================================================
        std::cout << "[Setup] Initializing DeviceEnv..." << std::endl;
        
        // Assuming initialize takes no args or a device ID. 
        // If your API requires a device ID, pass 0.
        DeviceEnv::instance().init(); 

        run_tests();

        // Optional: Cleanup if your class has a finalize method
        // DeviceEnv::instance().finalize();

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}