#ifndef DEVICE_ENV_CUH
#define DEVICE_ENV_CUH

#pragma once
#include <cuda_runtime.h>

namespace GPU {

    class DeviceEnv {
    private:
        // [修正] 補上這些變數，不然 .cu 檔找不到它們
        bool initialized = false;
        int device_id = 0;
        cudaStream_t compute_stream = 0;
        cudaStream_t transfer_stream = 0;
        
        DeviceEnv() = default;

    public:
        DeviceEnv(const DeviceEnv&) = delete;
        void operator=(const DeviceEnv&) = delete;

        static DeviceEnv& instance();

        void init(int rank, int gpus_per_node);
        void finalize();
        
        bool is_initialized() const { return initialized; }
        int get_device_id() const { return device_id; }
        
        // [修正] 宣告這兩個函數
        cudaStream_t get_compute_stream() const;
        cudaStream_t get_transfer_stream() const;
    };

}

#endif