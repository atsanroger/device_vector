#ifndef DEVICE_ENV_CUH
#define DEVICE_ENV_CUH

#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <mutex>

namespace GPU {

    class DeviceEnv {
    private:
        bool initialized = false;
        int device_id    = 0;
        
        cudaStream_t compute_stream  = 0;
        cudaStream_t transfer_stream = 0;

        void* workspace_ptr_   = nullptr;
        size_t workspace_size_ = 0;      
        
        DeviceEnv() = default;

    public:
        DeviceEnv(const DeviceEnv&)      = delete;
        void operator=(const DeviceEnv&) = delete;

        static DeviceEnv& instance();

        void init(int rank, int gpus_per_node);
        
        void finalize() {
            if (!initialized) return;

            if (workspace_ptr_) {
                cudaFree(workspace_ptr_); 
                workspace_ptr_ = nullptr;
            }
            workspace_size_ = 0;

            initialized     = false;
        }
        
        bool is_initialized() const { return initialized; }
        int get_device_id() const { return device_id; }
        
        cudaStream_t get_compute_stream() const;
        cudaStream_t get_transfer_stream() const;

        // =========================================================
        // Workspace (Scratchpad) 
        // =========================================================
        void* get_workspace(size_t required_bytes, cudaStream_t stream) {
            if (required_bytes == 0) return nullptr;

            if (required_bytes > workspace_size_) {
                if (workspace_ptr_) {
                    cudaFreeAsync(workspace_ptr_, stream);
                }

                workspace_size_ = static_cast<size_t>(required_bytes * 1.5);
                if (cudaMallocAsync(&workspace_ptr_, workspace_size_, stream) != cudaSuccess) {
                    workspace_ptr_  = nullptr;
                    workspace_size_ = 0;
                    return nullptr;
                }
            }
            return workspace_ptr_;
        }
    };

}

#endif