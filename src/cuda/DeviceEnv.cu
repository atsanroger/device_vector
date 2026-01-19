#include "DeviceEnv.cuh"
#include "DevicePtrManager.cuh"

#include <mutex>
#include <cstdio>

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
        if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
            std::fprintf(stderr, "Error: cudaGetDeviceCount failed.\n");
            return;
        }
        
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
    
    void* DevicePtrManager::get_dev_ptr(void* h_ptr) {
        std::lock_guard<std::mutex> lock(mu);
        auto it = ptr_map.find(h_ptr);
        if (it != ptr_map.end()) {
            return it->second.device_ptr;
        }
        return nullptr;
    }

} // namespace GPU