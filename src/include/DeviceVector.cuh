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

    // Mode 1: Pageable Memory (Standard)
    template <typename T>
    using PageableAllocator = std::allocator<T>;

    // Mode 3: Mapped Memory (Zero-Copy) - Note: Shifted to 3 to allow Mode 2 for Pure Device
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

        virtual size_t logical_size() const = 0;
        virtual size_t size() const         = 0;
        virtual size_t capacity() const     = 0;
    };

    // =========================================================
    // 2. Implementation
    // =========================================================
    template <typename T, typename Alloc>
    class DeviceVectorImpl : public IDeviceVector<T> {
    private:
        std::vector<T, Alloc> h_data_;
        T* d_ptr_ = nullptr;

        size_t logical_size_ = 0;
        size_t storage_size_ = 0;
        size_t capacity_     = 0;

        int mode_       = 1; // 0=Pinned, 1=Pageable, 2=PureDevice, 3=Mapped
        bool is_mapped_ = false;
        int device_id_  = 0;

        static size_t align_len(size_t n) {
            return ceil_warp_length(n);
        }

        void unregister_mapping_() {
            if (mode_ == 2) return; // Pure device has no host mirror
            if (storage_size_ == 0) return;
            if (h_data_.data()) DevicePtrManager::instance().unregister_ptr(h_data_.data());
        }

        void register_mapping_() {
            if (mode_ == 2) return; // Pure device has no host mirror
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

            // [Strategy] Grow by 1.5x to amortize allocation cost
            size_t new_capacity = std::max(required_storage, (size_t)(capacity_ * 1.5));
            new_capacity = align_len(new_capacity);

            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();

            T* new_d_ptr = nullptr;
            if (cudaMallocAsync(&new_d_ptr, new_capacity * sizeof(T), stream) != cudaSuccess) {
                throw std::runtime_error("GPU out of memory in ensure_device_capacity_!");
            }

            // Preserve existing content (Device to Device Copy) - Extremely Fast
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

        T reduce_impl_(int op_type, size_t num_items) {
            if (storage_size_ == 0 || num_items == 0) return static_cast<T>(0);
            num_items = std::min(num_items, storage_size_);

            // Helper lambdas for identity
            const auto identity_for_min = []() -> T {
                if constexpr (std::numeric_limits<T>::has_infinity) return std::numeric_limits<T>::infinity();
                else return std::numeric_limits<T>::max();
            };
            const auto identity_for_max = []() -> T {
                if constexpr (std::numeric_limits<T>::has_infinity) return -std::numeric_limits<T>::infinity();
                else return std::numeric_limits<T>::lowest();
            };

            auto fill_padding_device = [&](T val, cudaStream_t stream) {
                if (is_mapped_ || d_ptr_ == nullptr) return;
                if (logical_size_ >= storage_size_) return;
                const size_t pad_len = storage_size_ - logical_size_;
                const int block = VECTOR_LENGTH;
                const int grid = (int)std::min((pad_len + block - 1) / block, (size_t)(WARP_LENGTH * getNumSMs()));
                set_value_kernel<T><<<grid, block, 0, stream>>>(d_ptr_ + logical_size_, val, pad_len);
            };

            // Mapped logic (CPU reduction)
            if (is_mapped_) {
                if (DeviceEnv::instance().is_initialized()) cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                const size_t n = std::min(logical_size_, num_items);
                if (n == 0) return static_cast<T>(0);
                T result = h_data_[0];
                if (op_type == 0) { result = 0; for(size_t i=0; i<n; ++i) result += h_data_[i]; }
                else if (op_type == 1) { for(size_t i=1; i<n; ++i) if(h_data_[i] < result) result = h_data_[i]; }
                else { for(size_t i=1; i<n; ++i) if(h_data_[i] > result) result = h_data_[i]; }
                return result;
            }

            // Device reduction
            if (!DeviceEnv::instance().is_initialized()) throw std::runtime_error("DeviceEnv not initialized");
            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();

            const bool touches_padding = (logical_size_ < storage_size_) && (num_items == storage_size_);
            if (touches_padding) {
                if (op_type == 1) fill_padding_device(identity_for_min(), stream);
                else if (op_type == 2) fill_padding_device(identity_for_max(), stream);
            }

            T* d_out = nullptr;
            void* d_temp = nullptr;
            size_t temp_bytes = 0;

            if (cudaMallocAsync(&d_out, sizeof(T), stream) != cudaSuccess) throw std::runtime_error("cudaMallocAsync failed");

            if (op_type == 0) cub::DeviceReduce::Sum(nullptr, temp_bytes, d_ptr_, d_out, num_items, stream);
            else if (op_type == 1) cub::DeviceReduce::Min(nullptr, temp_bytes, d_ptr_, d_out, num_items, stream);
            else cub::DeviceReduce::Max(nullptr, temp_bytes, d_ptr_, d_out, num_items, stream);

            if (cudaMallocAsync(&d_temp, temp_bytes, stream) != cudaSuccess) {
                cudaFreeAsync(d_out, stream); throw std::runtime_error("cudaMallocAsync temp failed");
            }

            if (op_type == 0) cub::DeviceReduce::Sum(d_temp, temp_bytes, d_ptr_, d_out, num_items, stream);
            else if (op_type == 1) cub::DeviceReduce::Min(d_temp, temp_bytes, d_ptr_, d_out, num_items, stream);
            else cub::DeviceReduce::Max(d_temp, temp_bytes, d_ptr_, d_out, num_items, stream);

            if (touches_padding && (op_type == 1 || op_type == 2)) fill_padding_device(static_cast<T>(0), stream);

            T h_out{};
            cudaMemcpyAsync(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            cudaFreeAsync(d_out, stream);
            cudaFreeAsync(d_temp, stream);
            return h_out;
        }

        void free_device_ptr_safe_() {
            if (is_mapped_ || d_ptr_ == nullptr) return;
            if (DeviceEnv::instance().is_initialized()) {
                cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
                cudaFreeAsync(d_ptr_, stream);
                cudaStreamSynchronize(stream);
            } else {
                cudaFree(d_ptr_); // Fallback
            }
            d_ptr_ = nullptr;
        }

    public:
        DeviceVectorImpl(const DeviceVectorImpl&) = delete;
        DeviceVectorImpl& operator=(const DeviceVectorImpl&) = delete;
        DeviceVectorImpl(DeviceVectorImpl&&) = delete;
        DeviceVectorImpl& operator=(DeviceVectorImpl&&) = delete;

        // [MODIFIED] Constructor now takes an integer mode
        // 0: Pinned, 1: Pageable, 2: Pure Device, 3: Mapped
        DeviceVectorImpl(size_t n = 0, int mode = 1) : mode_(mode) {
            if (DeviceEnv::instance().is_initialized()) {
                device_id_ = DeviceEnv::instance().get_device_id();
            } else {
                cudaGetDevice(&device_id_);
            }
            
            is_mapped_ = (mode == 3); // Map "3" to IsMapped

            if (n > 0) resize(n);
        }

        ~DeviceVectorImpl() override {
            unregister_mapping_();
            free_device_ptr_safe_();
        }

        // ---------------------------------------------------------------------
        // [CRITICAL OPTIMIZATION] Mode 2 (Pure Device) skips Host operations
        // ---------------------------------------------------------------------
        void resize(size_t new_size) override {
            size_t new_storage_req = align_len(new_size);

            if (new_storage_req <= storage_size_) {
                logical_size_ = new_size;
                return;
            }

            // [MODE 2: Pure Device Path]
            // Zero CPU overhead. Only GPU allocation and D2D copy.
            if (mode_ == 2) {
                if (new_storage_req > capacity_) {
                    ensure_device_capacity_(new_storage_req);
                }
                logical_size_ = new_size;
                storage_size_ = new_storage_req; // Update tracked size, though h_data_ is empty
                return; 
            }

            // [Standard Modes 0, 1, 3]
            if (is_mapped_ && DeviceEnv::instance().is_initialized()) {
                cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
            }

            unregister_mapping_();

            storage_size_ = new_storage_req;

            if (is_mapped_) {
                h_data_.resize(storage_size_);
                refresh_mapped_pointer_();
            } else {
                if (storage_size_ > 0) ensure_device_capacity_(storage_size_);
                h_data_.resize(storage_size_); // CPU Cost happens here
            }

            logical_size_ = new_size;
            register_mapping_();
        }

        void reserve(size_t n) override {
            size_t required_storage = align_len(n);
            
            // [MODE 2: Pure Device]
            if (mode_ == 2) {
                if (required_storage > capacity_) ensure_device_capacity_(required_storage);
                return;
            }

            const bool need_host = (required_storage > h_data_.capacity());
            const bool need_dev  = (!is_mapped_ && required_storage > capacity_);

            if (!need_host && !need_dev) return;

            unregister_mapping_();
            if (need_host) h_data_.reserve(required_storage);
            if (is_mapped_) refresh_mapped_pointer_();
            else if (need_dev) ensure_device_capacity_(required_storage);
            register_mapping_();
        }

        // =========================================================
        // Data Transfer
        // =========================================================

        void update_device() override {
            if (mode_ == 2) return; // No host data to copy from
            if (storage_size_ == 0 || logical_size_ == 0) return;
            if (is_mapped_) return;
            
            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
            cudaMemcpyAsync(d_ptr_, h_data_.data(), logical_size_ * sizeof(T), cudaMemcpyHostToDevice, stream);
        }

        void update_host() override {
            if (mode_ == 2) return; // No host data to copy to
            if (storage_size_ == 0 || logical_size_ == 0) return;

            if (is_mapped_) {
                if (DeviceEnv::instance().is_initialized()) cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                return;
            }

            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
            cudaMemcpyAsync(h_data_.data(), d_ptr_, logical_size_ * sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        void update_device(size_t offset, size_t count) override {
            if (mode_ == 2) return;
            if (storage_size_ == 0 || count == 0) return;
            if (is_mapped_) return;
            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
            cudaMemcpyAsync(d_ptr_ + offset, h_data_.data() + offset, count * sizeof(T), cudaMemcpyHostToDevice, stream);
        }

        void update_host(size_t offset, size_t count) override {
            if (mode_ == 2) return;
            if (storage_size_ == 0 || count == 0) return;
            if (is_mapped_) {
                if (DeviceEnv::instance().is_initialized()) cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                return;
            }
            cudaMemcpy(h_data_.data() + offset, d_ptr_ + offset, count * sizeof(T), cudaMemcpyDeviceToHost);
        }

        // =========================================================
        // Fast Fill / Set
        // =========================================================

        void fill_zero() override {
            if (storage_size_ == 0) return;

            // [MODE 2] Skip host fill
            if (mode_ != 2) {
                if (is_mapped_) {
                    if (DeviceEnv::instance().is_initialized()) cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                    std::fill(h_data_.begin(), h_data_.end(), static_cast<T>(0));
                    return;
                }
                std::fill(h_data_.begin(), h_data_.begin() + logical_size_, static_cast<T>(0));
            }

            // Device Fill (Always do this)
            if (!DeviceEnv::instance().is_initialized()) throw std::runtime_error("DeviceEnv not initialized");
            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
            cudaMemsetAsync(d_ptr_, 0, storage_size_ * sizeof(T), stream);
        }

        void set_value(T val) override {
            if (storage_size_ == 0) return;

            // [MODE 2] Skip host fill
            if (mode_ != 2) {
                if (is_mapped_) {
                    if (DeviceEnv::instance().is_initialized()) cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                    std::fill(h_data_.begin(), h_data_.begin() + (ptrdiff_t)logical_size_, val);
                    return;
                }
                std::fill(h_data_.begin(), h_data_.begin() + (ptrdiff_t)logical_size_, val);
            }
            
            if (!DeviceEnv::instance().is_initialized()) throw std::runtime_error("DeviceEnv not initialized");
            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
            
            int block = VECTOR_LENGTH;
            int grid  = std::min((size_t)(WARP_LENGTH * getNumSMs()), (logical_size_ + block - 1) / (size_t)block);
            set_value_kernel<T><<<grid, block, 0, stream>>>(d_ptr_, val, logical_size_);
            
            // Zero Padding for safety
            if (logical_size_ < storage_size_) {
                const size_t pad_len = storage_size_ - logical_size_;
                const size_t grid2 = std::min((size_t)(WARP_LENGTH * getNumSMs()), (pad_len + block - 1) / (size_t)block);
                set_value_kernel<T><<<grid2, block, 0, stream>>>(d_ptr_ + logical_size_, static_cast<T>(0), pad_len);
            }
        }

        // =========================================================
        // Clone
        // =========================================================
        IDeviceVector<T>* clone() override {
            auto* new_vec = new DeviceVectorImpl<T, Alloc>(this->logical_size(), mode_); // Pass mode_

            if (this->size() > 0) {
                if (is_mapped_) {
                    // Mapped logic...
                    if (DeviceEnv::instance().is_initialized()) cudaStreamSynchronize(DeviceEnv::instance().get_compute_stream());
                    std::copy(this->h_data_.begin(), this->h_data_.begin() + (ptrdiff_t)this->size(), new_vec->host_ptr());
                } else {
                    // Standard & Mode 2 logic (D2D copy)
                    cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
                    cudaMemcpyAsync(new_vec->device_ptr(), this->d_ptr_, this->size() * sizeof(T), cudaMemcpyDeviceToDevice, stream);
                }
            }
            return new_vec;
        }

        // =========================================================
        // Reductions (Unchanged - they use d_ptr_)
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
        T* host_ptr() override { return (mode_ == 2) ? nullptr : h_data_.data(); }
        T* device_ptr() override { return d_ptr_; }
        size_t size() const override { return logical_size_; }
        size_t logical_size() const override { return logical_size_; }
        size_t capacity() const override { return capacity_; }
    };

} // namespace GPU

#endif // DEVICE_VECTOR_CUH