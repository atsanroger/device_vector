#include <cub/cub.cuh>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "DeviceVector.cuh"
#include "acc_interop.h"

using namespace GPU;

#define ACC_GUARD(SUFFIX, HANDLE, WHAT) \
    if (vec_acc_is_mapped_##SUFFIX((HANDLE))) { \
        fprintf(stderr, "[FATAL] %s: vector is OpenACC-mapped.\n", (WHAT)); abort(); \
    }

// =========================================================
// Semantic Interface Definition
// =========================================================
#define DEFINE_VEC_INTERFACE(SUFFIX, TYPE) \
    /* 1. 計算專用: DeviceVector (Pure Compute) */ \
    void *vec_new_vector_##SUFFIX(size_t n) { \
        /* 強制使用 PageableAllocator 但關閉 Mirror (use_host_mirror=false) */ \
        return new DeviceVectorImpl<TYPE, PageableAllocator<TYPE>>(n, false); \
    } \
    /* 2. 傳輸專用: DeviceBuffer (Transfer/IO) */ \
    void *vec_new_buffer_##SUFFIX(size_t n, bool pinned) { \
        /* 開啟 Mirror (use_host_mirror=true) */ \
        if (pinned) { \
            return new DeviceVectorImpl<TYPE, PinnedAllocator<TYPE>>(n, true); \
        } else { \
            return new DeviceVectorImpl<TYPE, PageableAllocator<TYPE>>(n, true); \
        } \
    } \
    /* 以下通用函數保持不變，因為它們操作的是介面 IDeviceVector */ \
    void vec_delete_##SUFFIX(void *h) { \
        ACC_GUARD(SUFFIX, h, "delete"); delete (IDeviceVector<TYPE> *)h; \
    } \
    void vec_resize_##SUFFIX(void *h, size_t n) { \
        ACC_GUARD(SUFFIX, h, "resize"); ((IDeviceVector<TYPE> *)h)->resize(n); \
    } \
    void vec_reserve_##SUFFIX(void *h, size_t n) { \
        ACC_GUARD(SUFFIX, h, "reserve"); ((IDeviceVector<TYPE> *)h)->reserve(n); \
    } \
    void vec_copy_from_##SUFFIX(void *dst, void *src) { \
        ((IDeviceVector<TYPE> *)dst)->copy_from((IDeviceVector<TYPE> *)src); \
    } \
    size_t vec_size_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->size(); } \
    size_t vec_capacity_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->capacity(); } \
    TYPE *vec_host_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->host_ptr(); } \
    void *vec_dev_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->device_ptr(); } \
    void vec_upload_##SUFFIX(void *h) { ((IDeviceVector<TYPE> *)h)->update_device(); } \
    void vec_download_##SUFFIX(void *h) { ((IDeviceVector<TYPE> *)h)->update_host(); } \
    void vec_upload_part_##SUFFIX(void *h, size_t o, size_t c) { ((IDeviceVector<TYPE> *)h)->update_device(o, c); } \
    void vec_download_part_##SUFFIX(void *h, size_t o, size_t c) { ((IDeviceVector<TYPE> *)h)->update_host(o, c); } \
    void vec_fill_zero_##SUFFIX(void *h) { ((IDeviceVector<TYPE> *)h)->fill_zero(); } \
    void vec_set_value_##SUFFIX(void *h, TYPE v) { ((IDeviceVector<TYPE> *)h)->set_value(v); } \
    void *vec_clone_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->clone(); } \
    TYPE vec_sum_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->sum(); } \
    TYPE vec_min_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->min(); } \
    TYPE vec_max_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->max(); } \
    TYPE vec_sum_partial_##SUFFIX(void *h, size_t n) { return ((IDeviceVector<TYPE> *)h)->sum_partial(n); } \
    TYPE vec_min_partial_##SUFFIX(void *h, size_t n) { return ((IDeviceVector<TYPE> *)h)->min_partial(n); } \
    TYPE vec_max_partial_##SUFFIX(void *h, size_t n) { return ((IDeviceVector<TYPE> *)h)->max_partial(n); }

extern "C" {
    void device_env_init(int r, int g) { GPU::DeviceEnv::instance().init(r, g); }
    void device_env_finalize() { GPU::DeviceEnv::instance().finalize(); }
    void device_synchronize() { cudaDeviceSynchronize(); }

    // Sort Wrapper (unchanged)
    static thread_local void *g_sort_temp_storage = nullptr;
    static thread_local size_t g_sort_temp_storage_bytes = 0;
    static thread_local size_t g_sort_max_items = 0;

    void vec_sort_pairs_i4_c(void *k_in, void *k_buf, void *v_in, void *v_buf, size_t n) {

        if (n==0) return;

        cudaStream_t s = GPU::DeviceEnv::instance().get_compute_stream();

        int *k_ptr     = (int*)((IDeviceVector<int>*)k_in)->device_ptr();
        int *k_buf_ptr = (int*)((IDeviceVector<int>*)k_buf)->device_ptr();
        int *v_ptr     = (int*)((IDeviceVector<int>*)v_in)->device_ptr();
        int *v_buf_ptr = (int*)((IDeviceVector<int>*)v_buf)->device_ptr();

        cub::DoubleBuffer<int> dk(k_ptr, k_buf_ptr), dv(v_ptr, v_buf_ptr);

        if (n > g_sort_max_items) {
            if (g_sort_temp_storage) cudaFreeAsync(g_sort_temp_storage, s);
            size_t nb=0; cub::DeviceRadixSort::SortPairs(nullptr, nb, dk, dv, n, 0, 32, s);
            size_t pb = (size_t)(nb*1.5);
            cudaMallocAsync(&g_sort_temp_storage, pb, s);
            g_sort_temp_storage_bytes = pb; g_sort_max_items = (size_t)(n*1.5);
        }
        cub::DeviceRadixSort::SortPairs(g_sort_temp_storage, g_sort_temp_storage_bytes, dk, dv, n, 0, 32, s);
        if (dk.Current() != k_ptr) {
            cudaMemcpyAsync(k_ptr, dk.Current(), n*4, cudaMemcpyDeviceToDevice, s);
            cudaMemcpyAsync(v_ptr, dv.Current(), n*4, cudaMemcpyDeviceToDevice, s);
        }
    }

    template <typename ValT>
    void sort_by_key(IDeviceVector<ValT>* values_vec) {
        if (storage_size_ == 0 || values_vec == nullptr) return;
        if (values_vec->size() != this->logical_size_) {
             throw std::runtime_error("SortPairs: Value vector size mismatch");
        }

        cudaStream_t stream = DeviceEnv::instance().get_compute_stream();

        T* d_keys_alt = nullptr;
        cudaMallocAsync(&d_keys_alt, storage_size_ * sizeof(T), stream);
        cub::DoubleBuffer<T> d_keys(d_ptr_, d_keys_alt);

        ValT* d_vals_ptr = values_vec->device_ptr();
        ValT* d_vals_alt = nullptr;
        cudaMallocAsync(&d_vals_alt, values_vec->size() * sizeof(ValT), stream);
        cub::DoubleBuffer<ValT> d_values(d_vals_ptr, d_vals_alt);

        size_t temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, d_keys, d_values, logical_size_, 0, sizeof(T) * 8, stream);

        void* d_temp_storage = DeviceEnv::instance().get_workspace(temp_storage_bytes, stream);

        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, logical_size_, 0, sizeof(T) * 8, stream);

        if (d_keys.Current() != d_ptr_) {
            cudaMemcpyAsync(d_ptr_, d_keys.Current(), logical_size_ * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
        if (d_values.Current() != d_vals_ptr) {
            cudaMemcpyAsync(d_vals_ptr, d_values.Current(), logical_size_ * sizeof(ValT), cudaMemcpyDeviceToDevice, stream);
        }

        cudaFreeAsync(d_keys_alt, stream);
        cudaFreeAsync(d_vals_alt, stream);
    }

    DEFINE_VEC_INTERFACE(r4, float)
    DEFINE_VEC_INTERFACE(r8, double)
    DEFINE_VEC_INTERFACE(i8, long long)
    DEFINE_VEC_INTERFACE(i4, int)
}