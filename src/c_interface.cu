#include <cub/cub.cuh>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "DeviceVector.cuh"
#include "Device_Constant.cuh"

// 這個 header 只放 extern "C" 宣告，不要 include <openacc.h>（避免 nvcc 麻煩）
#include "acc_interop.h"

using namespace GPU;

// =========================================================
// Error Checking Macro
// =========================================================
#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t err = call;                               \
        if (err != cudaSuccess)                               \
        {                                                     \
            printf("[C++ ERROR] %s failed at line %d: %s\n",  \
                   #call, __LINE__, cudaGetErrorString(err)); \
            std::fflush(stdout);                              \
            std::abort();                                     \
        }                                                     \
    } while (0)

// =========================================================
// OpenACC-mapped Guard (Strategy A)
// mapped 期間禁止：delete / resize / reserve
// =========================================================
#define ACC_GUARD(SUFFIX, HANDLE, WHAT)                                            \
    do                                                                             \
    {                                                                              \
        if (vec_acc_is_mapped_##SUFFIX((HANDLE)))                                  \
        {                                                                          \
            std::fprintf(stderr,                                                   \
                         "[C++ FATAL] %s: vector(i.e. handle) is OpenACC-mapped. " \
                         "Call vec_acc_unmap_%s(handle) before %s.\n",             \
                         (WHAT), #SUFFIX, (WHAT));                                 \
            std::fflush(stderr);                                                   \
            std::abort();                                                          \
        }                                                                          \
    } while (0)

// =========================================================
// Vector Interface Macro
// =========================================================
#define DEFINE_VEC_INTERFACE(SUFFIX, TYPE)                                              \
    /* Create Vector */                                                                 \
    void *vec_create_##SUFFIX(size_t n, int mode)                                       \
    {                                                                                   \
        if (mode == 0)                                                                  \
        {                                                                               \
            return new DeviceVectorImpl<TYPE, PinnedAllocator<TYPE>>(n, false);         \
        }                                                                               \
        else if (mode == 2)                                                             \
        {                                                                               \
            return new DeviceVectorImpl<TYPE, MappedAllocator<TYPE>>(n, true);          \
        }                                                                               \
        else                                                                            \
        {                                                                               \
            return new DeviceVectorImpl<TYPE, PageableAllocator<TYPE>>(n, false);       \
        }                                                                               \
    }                                                                                   \
    void vec_delete_##SUFFIX(void *h)                                                   \
    {                                                                                   \
        /* mapped 期間禁止 delete（不然 OpenACC 仍持有舊映射會炸） */                   \
        ACC_GUARD(SUFFIX, h, "vec_delete_" #SUFFIX);                                    \
        delete (IDeviceVector<TYPE> *)h;                                                \
    }                                                                                   \
    void vec_resize_##SUFFIX(void *h, size_t n)                                         \
    {                                                                                   \
        /* mapped 期間禁止 resize（device ptr 可能變，直接野指標） */                   \
        ACC_GUARD(SUFFIX, h, "vec_resize_" #SUFFIX);                                    \
        ((IDeviceVector<TYPE> *)h)->resize(n);                                          \
    }                                                                                   \
    void vec_reserve_##SUFFIX(void *h, size_t n)                                        \
    {                                                                                   \
        /* mapped 期間禁止 reserve（也可能 realloc/換 ptr） */                          \
        ACC_GUARD(SUFFIX, h, "vec_reserve_" #SUFFIX);                                   \
        ((IDeviceVector<TYPE> *)h)->reserve(n);                                         \
    }                                                                                   \
    size_t vec_size_##SUFFIX(void *h)                                                   \
    {                                                                                   \
        return ((IDeviceVector<TYPE> *)h)->size();                                      \
    }                                                                                   \
    size_t vec_capacity_##SUFFIX(void *h)                                               \
    {                                                                                   \
        return ((IDeviceVector<TYPE> *)h)->capacity();                                  \
    }                                                                                   \
    TYPE *vec_host_##SUFFIX(void *h)                                                    \
    {                                                                                   \
        return ((IDeviceVector<TYPE> *)h)->host_ptr();                                  \
    }                                                                                   \
    void *vec_dev_##SUFFIX(void *h)                                                     \
    {                                                                                   \
        return ((IDeviceVector<TYPE> *)h)->device_ptr();                                \
    }                                                                                   \
    /* Full Update */                                                                   \
    void vec_upload_##SUFFIX(void *h)                                                   \
    {                                                                                   \
        ((IDeviceVector<TYPE> *)h)->update_device();                                    \
    }                                                                                   \
    void vec_download_##SUFFIX(void *h)                                                 \
    {                                                                                   \
        ((IDeviceVector<TYPE> *)h)->update_host();                                      \
    }                                                                                   \
    /* Partial Update */                                                                \
    void vec_upload_part_##SUFFIX(void *h, size_t off, size_t cnt)                      \
    {                                                                                   \
        ((IDeviceVector<TYPE> *)h)->update_device(off, cnt);                            \
    }                                                                                   \
    void vec_download_part_##SUFFIX(void *h, size_t off, size_t cnt)                    \
    {                                                                                   \
        ((IDeviceVector<TYPE> *)h)->update_host(off, cnt);                              \
    }                                                                                   \
    /* Fast Fill / Set */                                                               \
    void vec_fill_zero_##SUFFIX(void *h)                                                \
    {                                                                                   \
        ((IDeviceVector<TYPE> *)h)->fill_zero();                                        \
    }                                                                                   \
    void vec_set_value_##SUFFIX(void *h, TYPE val)                                      \
    {                                                                                   \
        ((IDeviceVector<TYPE> *)h)->set_value(val);                                     \
    }                                                                                   \
    /* Gather */                                                                        \
    void vec_gather_##SUFFIX(void *h_src, void *h_map, void *h_dst)                     \
    {                                                                                   \
        IDeviceVector<TYPE> *src = (IDeviceVector<TYPE> *)h_src;                        \
        IDeviceVector<int> *map = (IDeviceVector<int> *)h_map;                          \
        IDeviceVector<TYPE> *dst = (IDeviceVector<TYPE> *)h_dst;                        \
        size_t n = src->size();                                                         \
        if (n == 0)                                                                     \
            return;                                                                     \
        if (src->device_ptr() == dst->device_ptr())                                     \
        {                                                                               \
            std::fprintf(stderr, "[C++ Error] vec_gather: src and dst must differ!\n"); \
            std::fflush(stderr);                                                        \
            std::abort();                                                               \
        }                                                                               \
        cudaStream_t stream = GPU::DeviceEnv::instance().get_compute_stream();          \
        int block = VECTOR_LENGTH;                                                      \
        int num_sm = getNumSMs();                                                       \
        int grid = std::min((size_t)(WARP_LENGTH * num_sm), (n + block - 1) / block);   \
        gather_kernel<TYPE><<<grid, block, 0, stream>>>(                                \
            src->device_ptr(), map->device_ptr(), dst->device_ptr(), n);                \
        CUDA_CHECK(cudaGetLastError());                                                 \
    }                                                                                   \
    /* Deep Copy */                                                                     \
    void *vec_clone_##SUFFIX(void *h)                                                   \
    {                                                                                   \
        return ((IDeviceVector<TYPE> *)h)->clone();                                     \
    }                                                                                   \
    /* Reductions (Full & Partial) */                                                   \
    TYPE vec_sum_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->sum(); }        \
    TYPE vec_min_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->min(); }        \
    TYPE vec_max_##SUFFIX(void *h) { return ((IDeviceVector<TYPE> *)h)->max(); }        \
    TYPE vec_sum_partial_##SUFFIX(void *h, size_t n)                                    \
    {                                                                                   \
        return ((IDeviceVector<TYPE> *)h)->sum_partial(n);                              \
    }                                                                                   \
    TYPE vec_min_partial_##SUFFIX(void *h, size_t n)                                    \
    {                                                                                   \
        return ((IDeviceVector<TYPE> *)h)->min_partial(n);                              \
    }                                                                                   \
    TYPE vec_max_partial_##SUFFIX(void *h, size_t n)                                    \
    {                                                                                   \
        return ((IDeviceVector<TYPE> *)h)->max_partial(n);                              \
    }

extern "C"
{

    // --- Environment Management ---
    void device_env_init(int rank, int gpus_per_node)
    {
        GPU::DeviceEnv::instance().init(rank, gpus_per_node);
    }

    void device_env_finalize()
    {
        GPU::DeviceEnv::instance().finalize();
    }

    void device_synchronize()
    {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // =================================================================
    // Static Workspace Cache for CUB Sort
    // =================================================================
    static thread_local void *g_sort_temp_storage = nullptr; // openMP safety
    static thread_local size_t g_sort_temp_storage_bytes = 0;
    static thread_local size_t g_sort_max_items = 0;

    // --- CUB Radix Sort Wrapper (Optimized & Cached) ---
    void vec_sort_pairs_i4_c(void *d_keys_in, void *d_keys_buf,
                             void *d_vals_in, void *d_vals_buf,
                             size_t num_items)
    {

        if (num_items == 0)
            return;

        cudaStream_t stream = GPU::DeviceEnv::instance().get_compute_stream();

        cub::DoubleBuffer<int> d_keys((int *)((IDeviceVector<int> *)d_keys_in)->device_ptr(),
                                      (int *)((IDeviceVector<int> *)d_keys_buf)->device_ptr());

        cub::DoubleBuffer<int> d_values((int *)((IDeviceVector<int> *)d_vals_in)->device_ptr(),
                                        (int *)((IDeviceVector<int> *)d_vals_buf)->device_ptr());

        // 1) Check & Reallocate Workspace if needed
        if (num_items > g_sort_max_items)
        {

            // Free old
            if (g_sort_temp_storage)
            {
                CUDA_CHECK(cudaFreeAsync(g_sort_temp_storage, stream));
                g_sort_temp_storage = nullptr;
            }

            // Query new size
            size_t new_bytes = 0;
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, new_bytes,
                                                       d_keys, d_values, num_items,
                                                       0, sizeof(int) * 8, stream));

            // Allocate with padding
            size_t padded_bytes = (size_t)(new_bytes * 1.5);
            CUDA_CHECK(cudaMallocAsync(&g_sort_temp_storage, padded_bytes, stream));

            g_sort_temp_storage_bytes = padded_bytes;
            g_sort_max_items = (size_t)(num_items * 1.5);
        }

        // 2) Execute Sort
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(g_sort_temp_storage, g_sort_temp_storage_bytes,
                                                   d_keys, d_values, num_items,
                                                   0, sizeof(int) * 8, stream));

        // 3) Ping-Pong Handling
        if (d_keys.Current() != (int *)((IDeviceVector<int> *)d_keys_in)->device_ptr())
        {
            CUDA_CHECK(cudaMemcpyAsync(((IDeviceVector<int> *)d_keys_in)->device_ptr(),
                                       d_keys.Current(), num_items * sizeof(int),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(((IDeviceVector<int> *)d_vals_in)->device_ptr(),
                                       d_values.Current(), num_items * sizeof(int),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }

    // --- Template Instantiations ---
    DEFINE_VEC_INTERFACE(r4, float)
    DEFINE_VEC_INTERFACE(r8, double)
    DEFINE_VEC_INTERFACE(i8, long long)
    DEFINE_VEC_INTERFACE(i4, int)
}
