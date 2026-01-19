#include <cub/cub.cuh>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "DeviceVector.cuh"
#include "acc_interop.h"

using namespace GPU;

// =========================================================
// Safety Guard
// =========================================================
#define ACC_GUARD(SUFFIX, HANDLE, WHAT) \
    if (vec_acc_is_mapped_##SUFFIX((HANDLE))) { \
        fprintf(stderr, "[FATAL] %s: vector is OpenACC-mapped.\n", (WHAT)); abort(); \
    }

// =========================================================
// Semantic Interface Definition
// =========================================================
#define DEFINE_VEC_INTERFACE(SUFFIX, TYPE) \
    void *vec_new_vector_##SUFFIX(size_t n) { \
        return new DeviceVectorImpl<TYPE, PageableAllocator<TYPE>>(n, false); \
    } \
    void *vec_new_buffer_##SUFFIX(size_t n, bool pinned) { \
        if (pinned) { \
            return new DeviceVectorImpl<TYPE, PinnedAllocator<TYPE>>(n, true); \
        } else { \
            return new DeviceVectorImpl<TYPE, PageableAllocator<TYPE>>(n, true); \
        } \
    } \
    /* --- Common Operations --- */ \
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
    /* Getters */ \
    size_t vec_size_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->size(); } \
    size_t vec_capacity_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->capacity(); } \
    TYPE *vec_host_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->host_ptr(); } \
    void *vec_dev_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->device_ptr(); } \
    /* Transfers */ \
    void vec_upload_##SUFFIX(void *h) { ((IDeviceVector<TYPE> *)h)->update_device(); } \
    void vec_download_##SUFFIX(void *h) { ((IDeviceVector<TYPE> *)h)->update_host(); } \
    void vec_upload_part_##SUFFIX(void *h, size_t o, size_t c) { ((IDeviceVector<TYPE> *)h)->update_device(o, c); } \
    void vec_download_part_##SUFFIX(void *h, size_t o, size_t c) { ((IDeviceVector<TYPE> *)h)->update_host(o, c); } \
    /* Utils */ \
    void vec_fill_zero_##SUFFIX(void *h) { ((IDeviceVector<TYPE> *)h)->fill_zero(); } \
    void vec_set_value_##SUFFIX(void *h, TYPE v) { ((IDeviceVector<TYPE> *)h)->set_value(v); } \
    void *vec_clone_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->clone(); } \
    /* Reductions */ \
    TYPE vec_sum_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->sum(); } \
    TYPE vec_min_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->min(); } \
    TYPE vec_max_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->max(); } \
    TYPE vec_sum_partial_##SUFFIX(void *h, size_t n) { return ((IDeviceVector<TYPE> *)h)->sum_partial(n); } \
    TYPE vec_min_partial_##SUFFIX(void *h, size_t n) { return ((IDeviceVector<TYPE> *)h)->min_partial(n); } \
    TYPE vec_max_partial_##SUFFIX(void *h, size_t n) { return ((IDeviceVector<TYPE> *)h)->max_partial(n); } \
    void vec_sort_##SUFFIX(void *h) { ((IDeviceVector<TYPE> *)h)->sort(); }

// =========================================================
// Sort Pairs Wrapper Generator
// =========================================================
#define DEFINE_SORT_PAIRS_WRAPPER(SUFFIX, KEY_TYPE) \
    void vec_sort_pairs_##SUFFIX##_c(void *k_in, void *k_buf, void *v_in, void *v_buf, size_t n) { \
        if (n == 0) return; \
        auto* keys_vec = (DeviceVectorImpl<KEY_TYPE, PageableAllocator<KEY_TYPE>>*)k_in; \
        auto* vals_vec = (IDeviceVector<int>*)v_in; \
        \
        keys_vec->sort_by_key(vals_vec); \
        \
        (void)k_buf; (void)v_buf; \
    }
    
extern "C" {

    // --- Environment ---
    void device_env_init(int r, int g) { GPU::DeviceEnv::instance().init(r, g); }
    void device_env_finalize() { GPU::DeviceEnv::instance().finalize(); }
    void device_synchronize() { cudaDeviceSynchronize(); }

    // --- Instantiations (Standard Interface) ---
    DEFINE_VEC_INTERFACE(r4, float)
    DEFINE_VEC_INTERFACE(r8, double)
    DEFINE_VEC_INTERFACE(i8, long long)
    DEFINE_VEC_INTERFACE(i4, int)

    DEFINE_SORT_PAIRS_WRAPPER(i4, int)
    DEFINE_SORT_PAIRS_WRAPPER(i8, long long)
    DEFINE_SORT_PAIRS_WRAPPER(r4, float)
    DEFINE_SORT_PAIRS_WRAPPER(r8, double)
}