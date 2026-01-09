#ifndef DEVICE_VECTOR_CUH
#define DEVICE_VECTOR_CUH

#pragma once

// c++ library
#include <vector>
#include <algorithm>
#include <cstddef>
#include <limits>

// CUDA library
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <stdexcept>
#include <cstdio>
#include <cstdlib>

// Relatives
#include "DeviceEnv.cuh"
#include "DevicePtrManager.cuh"
#include "Device_Constant.cuh"
#include "Device_Kernels.cuh"

namespace GPU {

    // =========================================================
    // Allocators
    // =========================================================

    // Mode 0: Pinned Memory
    template <typename T>
    struct PinnedAllocator {
        using value_type = T;
        T* allocate(std::size_t n) {
            T* ptr;
            if (cudaMallocHost((void**)&ptr, n * sizeof(T)) != cudaSuccess) throw std::bad_alloc();
            return ptr;
        }
        void deallocate(T* ptr, std::size_t) { cudaFreeHost(ptr); }
    };

    // Mode 1: Pageable Memory
    template <typename T>
    using PageableAllocator = std::allocator<T>;

    // Mode 2: Mapped Memory (Zero-Copy)
    template <typename T>
    struct MappedAllocator {
        using value_type = T;
        T* allocate(std::size_t n) {
            T* ptr;
            unsigned int flags = cudaHostAllocMapped | cudaHostAllocPortable;
            if (cudaHostAlloc((void**)&ptr, n * sizeof(T), flags) != cudaSuccess) throw std::bad_alloc();
            return ptr;
        }
        void deallocate(T* ptr, std::size_t) { cudaFreeHost(ptr); }
    };

    // =========================================================
    // 1. Abstract Base Class (Interface)
    // =========================================================
    template <typename T>
    class IDeviceVector {
    public:
        virtual ~IDeviceVector() {}

        virtual void resize(size_t n)  = 0;
        virtual void reserve(size_t n) = 0;

        virtual void update_device() = 0;
        virtual void update_host()   = 0;

        virtual void update_device(size_t offset, size_t count) = 0;
        virtual void update_host(size_t offset, size_t count)   = 0;

        virtual void fill_zero()          = 0;
        virtual void set_value(T val)     = 0;
        virtual IDeviceVector<T>* clone() = 0;

        virtual T sum_partial(size_t n) = 0;
        virtual T min_partial(size_t n) = 0;
        virtual T max_partial(size_t n) = 0;

        virtual T sum() = 0;
        virtual T min() = 0;
        virtual T max() = 0;

        virtual T* host_ptr()   = 0;
        virtual T* device_ptr() = 0;

        virtual size_t logical_size() const = 0; // original requested length (no padding)
        virtual size_t size() const         = 0; // padded size (storage length, may be >= logical size)
        virtual size_t capacity() const     = 0; // device capacity (elements)
    };

    // =========================================================
    // 2. Implementation
    // =========================================================
    template <typename T, typename Alloc>
    class DeviceVectorImpl : public IDeviceVector<T> {
    private:
        std::vector<T, Alloc> h_data_;
        T* d_ptr_ = nullptr;

        // logical_size_  : user-visible length (Fortran side should see this)
        // storage_size_  : aligned length actually stored in h_data_ (and used for pointer registration)
        // capacity_      : allocated device capacity (elements)
        size_t logical_size_ = 0;
        size_t storage_size_ = 0;
        size_t capacity_     = 0;

        bool is_mapped_ = false;
        int device_id_  = 0;

        static size_t align_len(size_t n) {
            // padding
            return ceil_warp_length(n);
        }

        void unregister_mapping_() {
            if (storage_size_ == 0) return;
            if (h_data_.data()) DevicePtrManager::instance().unregister_ptr(h_data_.data());
        }

        void register_mapping_() {
            if (storage_size_ == 0) return;
            if (h_data_.data() && d_ptr_) {
                DevicePtrManager::instance().register_ptr(h_data_.data(), d_ptr_, storage_size_ * sizeof(T));
            }
        }

        void ensure_device_capacity_(size_t required_storage) {
            if (is_mapped_) return;

            required_storage = align_len(required_storage);
            if (required_storage <= capacity_) return;

            if (!DeviceEnv::instance().is_initialized()) {
                throw std::runtime_error("DeviceEnv is not initialized.");
            }

            size_t new_capacity = std::max(required_storage, (size_t)(capacity_ * 1.5));
            new_capacity = align_len(new_capacity);

            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();

            T* new_d_ptr = nullptr;
            if (cudaMallocAsync(&new_d_ptr, new_capacity * sizeof(T), stream) != cudaSuccess) {
                throw std::runtime_error("GPU out of memory in ensure_device_capacity_!");
            }

            // Preserve existing content (including padding; harmless and simpler)
            if (d_ptr_ && storage_size_ > 0) {
                size_t copy_elems = std::min(storage_size_, required_storage);
                if (copy_elems > 0) {
                    cudaMemcpyAsync(new_d_ptr, d_ptr_, copy_elems * sizeof(T), cudaMemcpyDeviceToDevice, stream);
                }
                cudaFreeAsync(d_ptr_, stream);
            }

            d_ptr_    = new_d_ptr;
            capacity_ = new_capacity;
        }

        void refresh_mapped_pointer_() {
            if (!is_mapped_) return;

            if (storage_size_ == 0 || h_data_.empty()) {
                d_ptr_ = nullptr;
                capacity_ = 0;
                return;
            }

            if (cudaHostGetDevicePointer((void**)&d_ptr_, h_data_.data(), 0) != cudaSuccess) {
                throw std::runtime_error("Failed to get device pointer for mapped memory");
            }
            capacity_ = h_data_.capacity();
        }

        // Internal helper for reductions
        T reduce_impl_(int op_type, size_t num_items) {
            // op_type: 0=sum, 1=min, 2=max
            if (storage_size_ == 0 || num_items == 0) return static_cast<T>(0);

            // Clamp to storage size to avoid out-of-range reductions
            num_items = std::min(num_items, storage_size_);

            const auto identity_for_min = []() -> T {
                if constexpr (std::numeric_limits<T>::has_infinity) {
                    return std::numeric_limits<T>::infinity();
                } else {
                    return std::numeric_limits<T>::max();
                }
            };
            const auto identity_for_max = []() -> T {
                if constexpr (std::numeric_limits<T>::has_infinity) {
                    return -std::numeric_limits<T>::infinity();
                } else {
                    return std::numeric_limits<T>::lowest();
                }
            };

            auto fill_padding_device = [&](T val, cudaStream_t stream) {

                if (is_mapped_) return;
                if (d_ptr_ == nullptr) return;
                if (logical_size_ >= storage_size_) return;

                // Only touch the tail [logical_size_, storage_size_)
                const size_t pad_len   = storage_size_ - logical_size_;
                const int block        = VECTOR_LENGTH;
                const size_t grid_need = (pad_len + (size_t)block - 1) / (size_t)block;
                const size_t grid_cap  = (size_t)(WARP_LENGTH * getNumSMs());
                const int grid = (int)std::min(grid_need, grid_cap);

                set_value_kernel<T><<<grid, block, 0, stream>>>(d_ptr_ + logical_size_, val, pad_len);
            };

            // -----------------------------
            // Mapped (zero-copy): do host-side reduction over logical items only.
            // -----------------------------
            if (is_mapped_) {
                if (DeviceEnv::instance().is_initialized()) {
                    cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                }
                const size_t n = std::min(logical_size_, num_items);
                if (n == 0) return static_cast<T>(0);

                if (op_type == 0) {
                    T result = static_cast<T>(0);
                    for (size_t i = 0; i < n; ++i) result += h_data_[i];
                    return result;
                } else if (op_type == 1) {
                    T result = h_data_[0];
                    for (size_t i = 1; i < n; ++i) if (h_data_[i] < result) result = h_data_[i];
                    return result;
                } else {
                    T result = h_data_[0];
                    for (size_t i = 1; i < n; ++i) if (h_data_[i] > result) result = h_data_[i];
                    return result;
                }
            }

            if (!DeviceEnv::instance().is_initialized()) {
                throw std::runtime_error("DeviceEnv not initialized");
            }

            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();

            // -----------------------------
            // [SAFE-B contract]
            // Keep padding in a predictable state (0) after returning from any API.
            // For full min/max over the padded size, we temporarily set the tail to identity,
            // run the reduction over storage_size_, then reset the tail back to 0.
            // -----------------------------
            const bool touches_padding = (logical_size_ < storage_size_) && (num_items == storage_size_);
            if (touches_padding && (op_type == 1)) {
                fill_padding_device(identity_for_min(), stream);
            } else if (touches_padding && (op_type == 2)) {
                fill_padding_device(identity_for_max(), stream);
            }

            T* d_out = nullptr;
            void* d_temp = nullptr;
            size_t temp_bytes = 0;

            if (cudaMallocAsync(&d_out, sizeof(T), stream) != cudaSuccess) {
                throw std::runtime_error("cudaMallocAsync(d_out) failed");
            }

            // query temp bytes
            if (op_type == 0)      cub::DeviceReduce::Sum(nullptr, temp_bytes, d_ptr_, d_out, num_items, stream);
            else if (op_type == 1) cub::DeviceReduce::Min(nullptr, temp_bytes, d_ptr_, d_out, num_items, stream);
            else                   cub::DeviceReduce::Max(nullptr, temp_bytes, d_ptr_, d_out, num_items, stream);

            if (cudaMallocAsync(&d_temp, temp_bytes, stream) != cudaSuccess) {
                cudaFreeAsync(d_out, stream);
                throw std::runtime_error("cudaMallocAsync(temp) failed");
            }

            // execute
            if (op_type == 0)      cub::DeviceReduce::Sum(d_temp, temp_bytes, d_ptr_, d_out, num_items, stream);
            else if (op_type == 1) cub::DeviceReduce::Min(d_temp, temp_bytes, d_ptr_, d_out, num_items, stream);
            else                   cub::DeviceReduce::Max(d_temp, temp_bytes, d_ptr_, d_out, num_items, stream);

            // reset padding to 0 for safety
            if (touches_padding && (op_type == 1 || op_type == 2)) {
                fill_padding_device(static_cast<T>(0), stream);
            }

            T h_out{};
            cudaMemcpyAsync(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            cudaFreeAsync(d_out, stream);
            cudaFreeAsync(d_temp, stream);

            return h_out;
        }

        void free_device_ptr_safe_() {
            if (is_mapped_ || d_ptr_ == nullptr) return;

            // Prefer the library stream when available; otherwise create a local stream.
            if (DeviceEnv::instance().is_initialized()) {

                cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
                cudaFreeAsync(d_ptr_, stream);
                cudaStreamSynchronize(stream);

            } else {

                // Avoid leaks if device_env_finalize() was called before vec_delete().
                cudaSetDevice(device_id_);
                cudaStream_t tmp{};
                cudaStreamCreateWithFlags(&tmp, cudaStreamNonBlocking);
                cudaFreeAsync(d_ptr_, tmp);
                cudaStreamSynchronize(tmp);
                cudaStreamDestroy(tmp);
                
            }

            d_ptr_ = nullptr;
        }

    public:
        // [Safety] Delete copy/move constructors
        DeviceVectorImpl(const DeviceVectorImpl&)            = delete;
        DeviceVectorImpl& operator=(const DeviceVectorImpl&) = delete;
        DeviceVectorImpl(DeviceVectorImpl&&)                 = delete;
        DeviceVectorImpl& operator=(DeviceVectorImpl&&)      = delete;

        DeviceVectorImpl(size_t n = 0, bool is_mapped = false) : is_mapped_(is_mapped) {
            // Record device id for safe cleanup even if DeviceEnv is finalized early.
            if (DeviceEnv::instance().is_initialized()) {
                device_id_ = DeviceEnv::instance().get_device_id();
            } else {
                int dev = 0;
                if (cudaGetDevice(&dev) == cudaSuccess) device_id_ = dev;
            }

            if (n > 0) resize(n);
        }

        ~DeviceVectorImpl() override {
            unregister_mapping_();
            free_device_ptr_safe_();
        }

        void resize(size_t new_size) override {
            size_t new_storage = align_len(new_size);

            if (new_storage == storage_size_){
                logical_size_ = new_size;
                return;
            }
            // [CRITICAL FIX] 
            // If using Mapped Memory, std::vector reallocation will invalidate 
            // the memory physically. Any running kernel using d_ptr_ will crash.
            // We MUST synchronize before touching h_data_.
            if (is_mapped_) {
                if (DeviceEnv::instance().is_initialized()) {
                    cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                }
            }

            unregister_mapping_();

            if (is_mapped_) {
                h_data_.resize(new_storage);
                storage_size_ = new_storage;
                refresh_mapped_pointer_();
            } else {
                // Device memory resize (cudaMallocAsync) is stream-ordered, 
                // so generally safe without explicit sync IF on same stream.
                if (new_storage > 0) ensure_device_capacity_(new_storage);
                h_data_.resize(new_storage);
                storage_size_ = new_storage;
            }

            logical_size_ = new_size;

            // Keep padding predictable
            if (logical_size_ < storage_size_) {
                std::fill(h_data_.begin() + (ptrdiff_t)logical_size_, h_data_.end(), static_cast<T>(0));
            }

            register_mapping_();
        }

        void reserve(size_t n) override {

            size_t required_storage = align_len(n);
            const bool need_host = (required_storage > h_data_.capacity());
            const bool need_dev  = (!is_mapped_ && required_storage > capacity_);

            if (!need_host && !need_dev) return;

            unregister_mapping_();

            if (need_host) h_data_.reserve(required_storage);

            if (is_mapped_) {
                refresh_mapped_pointer_();
            } else if (need_dev) {
                ensure_device_capacity_(required_storage);
            }

            // storage_size_ (current size) is unchanged, but h_data_.data() may have moved.
            register_mapping_();
        }

        // =========================================================
        // Data Transfer - Full Update (logical size only)
        // =========================================================

        void update_device() override {
            if (storage_size_ == 0) return;
            if (is_mapped_) return;

            if (!DeviceEnv::instance().is_initialized()) throw std::runtime_error("DeviceEnv not initialized");

            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
            cudaMemcpyAsync(d_ptr_, h_data_.data(), storage_size_ * sizeof(T), cudaMemcpyHostToDevice, stream);
        }

        // void update_host() override {
        //     if (storage_size_ == 0) return;

        //     if (is_mapped_) {
        //         if (DeviceEnv::instance().is_initialized()) {
        //             cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
        //         }
        //         return;
        //     }

        //     cudaMemcpy(h_data_.data(), d_ptr_, storage_size_ * sizeof(T), cudaMemcpyDeviceToHost);

        //     // Keep host padding predictable as well
        //     if (logical_size_ < storage_size_) {
        //         std::fill(h_data_.begin() + (ptrdiff_t)logical_size_, h_data_.end(), static_cast<T>(0));
        //     }
        // }
        void update_host() override {
            if (storage_size_ == 0) return;

            // 1. If mapped (Zero-Copy), just sync to ensure previous kernels are done
            if (is_mapped_) {
                if (DeviceEnv::instance().is_initialized()) {
                    cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                }
                return;
            }

            // 2. If standard memory, we MUST use the compute stream to copy
            //    to ensure serialization with previous kernels.
            if (DeviceEnv::instance().is_initialized()) {
                cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
                
                // Use Async copy on the SPECIFIC stream
                cudaMemcpyAsync(
                    h_data_.data(), 
                    d_ptr_, 
                    storage_size_ * sizeof(T), 
                    cudaMemcpyDeviceToHost, 
                    stream
                );

                // BLOCK until the copy (and all preceding kernels) are finished.
                // This guarantees the host sees the latest data.
                cudaStreamSynchronize(stream);
            } else {
                // Fallback (Safe but slow)
                cudaMemcpy(
                    h_data_.data(), 
                    d_ptr_, 
                    storage_size_ * sizeof(T), 
                    cudaMemcpyDeviceToHost
                );
            }

            // Keep host padding predictable as well
            if (logical_size_ < storage_size_) {
                std::fill(h_data_.begin() + (ptrdiff_t)logical_size_, h_data_.end(), static_cast<T>(0));
            }
        }
        // =========================================================
        // Data Transfer - Partial Update (range checked on logical size)
        // =========================================================

        void update_device(size_t offset, size_t count) override {
            if (storage_size_ == 0 || count == 0) return;
            if (offset + count > storage_size_) throw std::out_of_range("update_device range error");
            if (is_mapped_) return;
            if (!DeviceEnv::instance().is_initialized()) throw std::runtime_error("DeviceEnv not initialized");

            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();

            cudaMemcpyAsync(
                d_ptr_ + offset,
                h_data_.data() + offset,
                count * sizeof(T),
                cudaMemcpyHostToDevice,
                stream
            );
        }

        void update_host(size_t offset, size_t count) override {
            if (storage_size_ == 0 || count == 0) return;
            if (offset + count > storage_size_) throw std::out_of_range("update_host range error");

            if (is_mapped_) {
                if (DeviceEnv::instance().is_initialized()) {
                    cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                }
                return;
            }

            cudaMemcpy(
                h_data_.data() + offset,
                d_ptr_ + offset,
                count * sizeof(T),
                cudaMemcpyDeviceToHost
            );
        }

        // =========================================================
        // Fast Fill / Set (logical size only)
        // =========================================================

        void fill_zero() override {
            if (storage_size_ == 0) return;

            if (is_mapped_) {
                if (DeviceEnv::instance().is_initialized()) {
                    cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                }
            }

            std::fill(h_data_.begin(), h_data_.end(), static_cast<T>(0));

            if (is_mapped_) return;
            if (!DeviceEnv::instance().is_initialized()) throw std::runtime_error("DeviceEnv not initialized");

            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
            cudaMemsetAsync(d_ptr_, 0, storage_size_ * sizeof(T), stream);
        }

        void set_value(T val) override {
            if (storage_size_ == 0) return;

            if (is_mapped_) {
                if (DeviceEnv::instance().is_initialized()) {
                    cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                }
            }

            std::fill(h_data_.begin(), h_data_.begin() + (ptrdiff_t)logical_size_, val);
            if (logical_size_ < storage_size_) {
                std::fill(h_data_.begin() + (ptrdiff_t)logical_size_, h_data_.end(), static_cast<T>(0));
            }

            if (is_mapped_) return;
            if (!DeviceEnv::instance().is_initialized()) throw std::runtime_error("DeviceEnv not initialized");

            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
            int block = VECTOR_LENGTH;
            int grid  = std::min((size_t)(WARP_LENGTH * getNumSMs()), (logical_size_ + block - 1) / (size_t)block);
            set_value_kernel<T><<<grid, block, 0, stream>>>(d_ptr_, val, logical_size_);
            if (logical_size_ < storage_size_) {
                const size_t pad_len = storage_size_ - logical_size_;
                const size_t grid_need2 = (pad_len + (size_t)block - 1) / (size_t)block;
                const size_t grid_cap2  = (size_t)(WARP_LENGTH * getNumSMs());
                const int grid2 = (int)std::min(grid_need2, grid_cap2);
                set_value_kernel<T><<<grid2, block, 0, stream>>>(d_ptr_ + logical_size_, static_cast<T>(0), pad_len);
            }
        }

        // =========================================================
        // Deep Copy (Clone) - preserves logical size
        // =========================================================
        IDeviceVector<T>* clone() override {
            auto* new_vec = new DeviceVectorImpl<T, Alloc>(this->logical_size(), is_mapped_);

            if (this->size() > 0) {
                if (is_mapped_) {
                    if (DeviceEnv::instance().is_initialized()) {
                        cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                    }
                    std::copy(this->h_data_.begin(),
                              this->h_data_.begin() + (ptrdiff_t)this->size(),
                              new_vec->host_ptr());
                } else {
                    cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
                    cudaMemcpyAsync(new_vec->device_ptr(), this->d_ptr_,
                                    this->size() * sizeof(T), cudaMemcpyDeviceToDevice, stream);
                }
            }
            return new_vec;
        }

        // =========================================================
        // Reductions
        // =========================================================
        T sum_partial(size_t n) override { return reduce_impl_(0, n); }
        T min_partial(size_t n) override { return reduce_impl_(1, n); }
        T max_partial(size_t n) override { return reduce_impl_(2, n); }

        T sum() override { return reduce_impl_(0, this->size()); }
        T min() override { return reduce_impl_(1, this->size()); }
        T max() override { return reduce_impl_(2, this->size()); }

        // =========================================================
        // Getters
        // =========================================================
        T* host_ptr() override { return h_data_.data(); }
        T* device_ptr() override { return d_ptr_; }
        // logical_size_ =  size before padding
        //size_t size() const override { return storage_size_; }
        size_t size() const override { return logical_size_; }
        size_t logical_size() const override { return logical_size_; }
        size_t storage_size() const { return storage_size_; }
        size_t capacity() const override { return capacity_; }
    };

} // namespace GPU

#endif // DEVICE_VECTOR_CUH