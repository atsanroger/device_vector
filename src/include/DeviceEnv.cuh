#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace GPU {

    class DeviceEnv {
    private:
        int device_id_ = 0;
        cudaStream_t compute_stream_ = nullptr;
        cudaStream_t comm_stream_    = nullptr;
        bool initialized_ = false;

        DeviceEnv();
        ~DeviceEnv();

    public:
        static DeviceEnv& instance();

        DeviceEnv(const DeviceEnv&)      = delete;
        void operator=(const DeviceEnv&) = delete;

        #ifdef USE_MPI
            void init(int rank, int gpus_per_node = 1); 
        #else
            void init(int rank = 0, int gpus_per_node = 1); 
        #endif
        
        void finalize();

        // Fail-fast helper: many components assume streams exist.
        bool is_initialized() const { return initialized_; }

        cudaStream_t get_compute_stream() const { return compute_stream_; }
        cudaStream_t get_comm_stream() const { return comm_stream_; }
        int get_device_id() const { return device_id_; }
    };
    
};