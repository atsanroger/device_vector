#pragma once
#include <map>
#include <mutex>
#include <cuda_runtime.h>

namespace GPU {

    class DevicePtrManager {
    private:

        struct MemInfo {
            void* dev_start;
            size_t size;
        };

        std::map<void*, MemInfo> ptr_map; // Host Ptr -> {Device Ptr, Size}
        std::mutex mtx;

        DevicePtrManager() {}

    public:
        static DevicePtrManager& instance();

        DevicePtrManager(const DevicePtrManager&) = delete;
        void operator=(const DevicePtrManager&)   = delete;

        void register_ptr(void* h_ptr, void* d_ptr, size_t size);
        void unregister_ptr(void* h_ptr);

        void* get_dev_ptr(void* phost);
    };

} // namespace GPU