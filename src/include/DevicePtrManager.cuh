#ifndef DEVICE_PTR_MANAGER_CUH
#define DEVICE_PTR_MANAGER_CUH

#pragma once
#include <unordered_map>
#include <mutex>

namespace GPU {

    struct PtrInfo {
        void* device_ptr;
        size_t size;
    };

    class DevicePtrManager {
    private:

        std::unordered_map<void*, PtrInfo> ptr_map;
        std::mutex mu;
        
        DevicePtrManager() = default;

    public:
        DevicePtrManager(const DevicePtrManager&) = delete;
        void operator=(const DevicePtrManager&) = delete;

        static DevicePtrManager& instance();

        void register_ptr(void* host_ptr, void* device_ptr, size_t size);
        void unregister_ptr(void* host_ptr);
        void* get_dev_ptr(void* host_ptr);    
    };

}

#endif