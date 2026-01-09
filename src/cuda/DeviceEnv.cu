#include "DeviceEnv.cuh"

namespace GPU {

    DeviceEnv::DeviceEnv() {}

    DeviceEnv::~DeviceEnv() {
        finalize(); 
    }

    DeviceEnv& DeviceEnv::instance() {
        static DeviceEnv inst;
        return inst;
    }

    void DeviceEnv::init(int rank, int gpus_per_node) {

        // if single card, must set rank = 0!

        if (initialized_) return; 

        int actual_gpu_count = 0;
        cudaError_t count_err = cudaGetDeviceCount(&actual_gpu_count);
            
        if (count_err != cudaSuccess || actual_gpu_count <= 0) {
            fprintf(stderr, "[Error] No CUDA devices found or driver error!\n");
            exit(EXIT_FAILURE);
        }

        this->device_id_ = rank % gpus_per_node;
        cudaError_t err = cudaSetDevice(this->device_id_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Error] Rank %d failed to set device %d: %s\n", 
                    rank, device_id_, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        int priority_high, priority_low;
        cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);

        cudaStreamCreateWithPriority(&compute_stream_, cudaStreamNonBlocking, 0);
        cudaStreamCreateWithPriority(&comm_stream_,    cudaStreamNonBlocking, priority_low);

        initialized_ = true;
        
        // printf("[DeviceEnv] Rank %d bound to GPU %d (Comm Stream Priority: %d)\n", 
        //        rank, device_id_, priority_low);
    };

    void DeviceEnv::finalize() {
        if (!initialized_) return; 
        
        if (compute_stream_) {
            cudaStreamSynchronize(compute_stream_);
            cudaStreamDestroy(compute_stream_);
            compute_stream_ = nullptr; 
        }
        if (comm_stream_) {
            cudaStreamSynchronize(comm_stream_);
            cudaStreamDestroy(comm_stream_);
            comm_stream_ = nullptr;
        }
        initialized_ = false;
    }

};