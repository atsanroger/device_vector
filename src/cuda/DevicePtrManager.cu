#include "DevicePtrManager.cuh"

namespace GPU {

    // =========================================================
    // Singleton Instance
    // =========================================================
    DevicePtrManager& DevicePtrManager::instance() {
        static DevicePtrManager inst;
        return inst;
    }

    // =========================================================
    // Register
    // =========================================================
    void DevicePtrManager::register_ptr(void* h_ptr, void* d_ptr, size_t size) {
        if (!h_ptr) return;
        std::lock_guard<std::mutex> lock(mtx);
        ptr_map[h_ptr] = {d_ptr, size};
    }

    // =========================================================
    // Unregister
    // =========================================================
    void DevicePtrManager::unregister_ptr(void* h_ptr) {
        if (!h_ptr) return;
        std::lock_guard<std::mutex> lock(mtx);
        ptr_map.erase(h_ptr);
    }

    // =========================================================
    // Lookup
    // =========================================================
    void* DevicePtrManager::get_dev_ptr(void* phost) {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (ptr_map.empty()) return nullptr;

        auto it = ptr_map.upper_bound(phost);

        if (it == ptr_map.begin()) return nullptr;

        --it; 

        char* h_start = (char*)it->first;
        char* d_start = (char*)it->second.dev_start;
        size_t size   = it->second.size;
        
        size_t offset = (char*)phost - h_start;

        if (offset >= size) return nullptr;

        return (void*)(d_start + offset);
    }

} // namespace gpu