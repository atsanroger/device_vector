#include "DeviceEnv.cuh"
#include "DevicePtrManager.cuh"
#include <mutex>
#include <cuda_runtime.h>

namespace GPU {

    // ==========================================
    // DeviceEnv Implementation
    // ==========================================
    DeviceEnv& DeviceEnv::instance() {
        static DeviceEnv inst;
        return inst;
    }

    void DeviceEnv::init(int rank, int gpus_per_node) {
        if (initialized) return;
        
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        if (gpus_per_node > 0) {
            device_id = rank % gpus_per_node;
        } else {
            device_id = 0; 
        }

        if (device_id >= device_count) device_id = 0;
        
        cudaSetDevice(device_id);
        
        cudaStreamCreate(&compute_stream);
        cudaStreamCreate(&transfer_stream);
        
        initialized = true;
    }

    void DeviceEnv::finalize() {
        if (!initialized) return;
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(transfer_stream);
        initialized = false;
    }

    cudaStream_t DeviceEnv::get_compute_stream() const {
        return compute_stream;
    }

    cudaStream_t DeviceEnv::get_transfer_stream() const {
        return transfer_stream;
    }

    // ==========================================
    // DevicePtrManager Implementation
    // ==========================================
    DevicePtrManager& DevicePtrManager::instance() {
        static DevicePtrManager inst;
        return inst;
    }

    void DevicePtrManager::register_ptr(void* host_ptr, void* device_ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mu);
        ptr_map[host_ptr] = {device_ptr, size};
    }

    void DevicePtrManager::unregister_ptr(void* host_ptr) {
        std::lock_guard<std::mutex> lock(mu);
        ptr_map.erase(host_ptr);
    }

} // namespace GPU