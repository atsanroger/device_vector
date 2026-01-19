#ifndef DEVICE_VECTOR_CUH
#define DEVICE_VECTOR_CUH

#pragma once

#include <vector>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

#include "DeviceEnv.cuh"
#include "DevicePtrManager.cuh"
#include "Device_Constant.cuh"
#include "Device_Kernels.cuh"

namespace GPU {

    // =========================================================
    // Allocators
    // =========================================================

    // Mode 0: Pinned Memory (For fast D2H/H2D)
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

    // =========================================================
    // Interface
    // =========================================================
    template <typename T>
    class IDeviceVector {
    public:
        virtual ~IDeviceVector() {}

        virtual void resize(size_t n)  = 0;
        virtual void reserve(size_t n) = 0;
        virtual void copy_from(IDeviceVector<T>* other) = 0; // GPU-GPU Deep Copy

        virtual void update_device() = 0;
        virtual void update_host()   = 0;
        virtual void update_device(size_t offset, size_t count) = 0;
        virtual void update_host(size_t offset, size_t count)   = 0;

        virtual void fill_zero()          = 0;
        virtual void set_value(T val)     = 0;
        virtual IDeviceVector<T>* clone() = 0;

        // --- Algorithms ---
        virtual void sort() = 0;  // [New] Sort Interface

        // --- Reductions ---
        virtual T sum_partial(size_t n) = 0;
        virtual T min_partial(size_t n) = 0;
        virtual T max_partial(size_t n) = 0;

        virtual T sum() = 0;
        virtual T min() = 0;
        virtual T max() = 0;

        virtual T* host_ptr()   = 0;
        virtual T* device_ptr() = 0;

        virtual size_t size() const         = 0;
        virtual size_t logical_size() const = 0;
        virtual size_t capacity() const     = 0;
    };

    // =========================================================
    // Implementation
    // =========================================================
    template <typename T, typename Alloc>
    class DeviceVectorImpl : public IDeviceVector<T> {
    private:
        std::vector<T, Alloc> h_data_; // Host Mirror (Empty if use_host_mirror_ == false)
        T* d_ptr_ = nullptr;

        size_t logical_size_ = 0;
        size_t storage_size_ = 0;
        size_t capacity_     = 0;

        bool use_host_mirror_ = true; 
        int device_id_        = 0;

        static size_t align_len(size_t n) {
            return ceil_warp_length(n);
        }

        void unregister_mapping_() {
            if (!use_host_mirror_) return; 
            if (storage_size_ > 0 && h_data_.data()) 
                DevicePtrManager::instance().unregister_ptr(h_data_.data());
        }

        void register_mapping_() {
            if (!use_host_mirror_) return;
            if (storage_size_ > 0 && h_data_.data() && d_ptr_) {
                DevicePtrManager::instance().register_ptr(h_data_.data(), d_ptr_, storage_size_ * sizeof(T));
            }
        }

        void ensure_device_capacity_(size_t required_storage) {
            required_storage = align_len(required_storage);
            if (required_storage <= capacity_) return;

            if (!DeviceEnv::instance().is_initialized()) throw std::runtime_error("DeviceEnv not initialized");

            // 1.5x Growth Strategy
            size_t new_capacity = std::max(required_storage, (size_t)(capacity_ * 1.5));
            new_capacity = align_len(new_capacity);

            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();
            T* new_d_ptr = nullptr;
            
            if (cudaMallocAsync(&new_d_ptr, new_capacity * sizeof(T), stream) != cudaSuccess) {
                throw std::runtime_error("GPU OOM in ensure_device_capacity_");
            }

            // D2D Copy (Preserve old data)
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

        // Helper for reductions
        T reduce_impl_(int op_type, size_t num_items) {
            if (storage_size_ == 0 || num_items == 0) return static_cast<T>(0);
            num_items = std::min(num_items, storage_size_);

            if (!DeviceEnv::instance().is_initialized()) throw std::runtime_error("DeviceEnv not initialized");
            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();

            // Handle padding for min/max safety
            if (logical_size_ < storage_size_ && num_items == storage_size_) {
                if (op_type == 1) { // Min
                     T val = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
                     size_t pad = storage_size_ - logical_size_;
                     int grid = std::min((size_t)(WARP_LENGTH * getNumSMs()), (pad + 255)/256);
                     set_value_kernel<T><<<grid, 256, 0, stream>>>(d_ptr_ + logical_size_, val, pad);
                } else if (op_type == 2) { // Max
                     T val = std::numeric_limits<T>::has_infinity ? -std::numeric_limits<T>::infinity() : std::numeric_limits<T>::lowest();
                     size_t pad = storage_size_ - logical_size_;
                     int grid = std::min((size_t)(WARP_LENGTH * getNumSMs()), (pad + 255)/256);
                     set_value_kernel<T><<<grid, 256, 0, stream>>>(d_ptr_ + logical_size_, val, pad);
                }
            }

            T* d_out = nullptr;
            void* d_temp = nullptr;
            size_t temp_bytes = 0;

            cudaMallocAsync(&d_out, sizeof(T), stream);

            if (op_type == 0) cub::DeviceReduce::Sum(nullptr, temp_bytes, d_ptr_, d_out, num_items, stream);
            else if (op_type == 1) cub::DeviceReduce::Min(nullptr, temp_bytes, d_ptr_, d_out, num_items, stream);
            else cub::DeviceReduce::Max(nullptr, temp_bytes, d_ptr_, d_out, num_items, stream);

            // Using Workspace if implemented for reduce too, otherwise malloc
            // For now keeping malloc for reduce to match your previous code
            cudaMallocAsync(&d_temp, temp_bytes, stream);

            if (op_type == 0) cub::DeviceReduce::Sum(d_temp, temp_bytes, d_ptr_, d_out, num_items, stream);
            else if (op_type == 1) cub::DeviceReduce::Min(d_temp, temp_bytes, d_ptr_, d_out, num_items, stream);
            else cub::DeviceReduce::Max(d_temp, temp_bytes, d_ptr_, d_out, num_items, stream);

            T h_out{};
            cudaMemcpyAsync(&h_out, d_out, sizeof(T), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            cudaFreeAsync(d_out, stream);
            cudaFreeAsync(d_temp, stream);
            return h_out;
        }

    public:
        DeviceVectorImpl(size_t n = 0, bool use_host_mirror = true) 
            : use_host_mirror_(use_host_mirror) {
            
            if (DeviceEnv::instance().is_initialized()) device_id_ = DeviceEnv::instance().get_device_id();
            else cudaGetDevice(&device_id_);
            
            if (n > 0) resize(n);
        }

        ~DeviceVectorImpl() override {
            unregister_mapping_();
            if (d_ptr_) {
                if (DeviceEnv::instance().is_initialized()) {
                    cudaStream_t s = DeviceEnv::instance().get_compute_stream();
                    cudaFreeAsync(d_ptr_, s);
                    cudaStreamSynchronize(s);
                } else {
                    cudaFree(d_ptr_);
                }
            }
        }

        // =========================================================
        // Core Logic: Resize
        // =========================================================
        void resize(size_t new_size) override {
            size_t new_storage_req = align_len(new_size);

            if (new_storage_req <= storage_size_) {
                logical_size_ = new_size;
                return;
            }

            if (!use_host_mirror_) {
                if (new_storage_req > capacity_) {
                    ensure_device_capacity_(new_storage_req);
                }
                logical_size_ = new_size;
                storage_size_ = new_storage_req;
                return;
            }

            unregister_mapping_();
            storage_size_ = new_storage_req;

            if (storage_size_ > 0) ensure_device_capacity_(storage_size_);
            
            h_data_.resize(storage_size_); 

            logical_size_ = new_size;
            register_mapping_();
        }

        void reserve(size_t n) override {
            size_t required_storage = align_len(n);

            if (!use_host_mirror_) {
                if (required_storage > capacity_) ensure_device_capacity_(required_storage);
                return;
            }

            if (required_storage <= h_data_.capacity() && required_storage <= capacity_) return;

            unregister_mapping_();
            if (required_storage > h_data_.capacity()) h_data_.reserve(required_storage);
            if (required_storage > capacity_) ensure_device_capacity_(required_storage);
            register_mapping_();
        }

        // =========================================================
        // Copy From (Deep GPU-GPU Copy)
        // =========================================================
        void copy_from(IDeviceVector<T>* other) override {
            if (!other) return;
            this->resize(other->size());
            if (this->size() == 0) return;

            if (DeviceEnv::instance().is_initialized()) {
                cudaMemcpyAsync(this->device_ptr(), other->device_ptr(), 
                                this->size() * sizeof(T), cudaMemcpyDeviceToDevice, 
                                DeviceEnv::instance().get_compute_stream());
            } else {
                cudaMemcpy(this->device_ptr(), other->device_ptr(), 
                           this->size() * sizeof(T), cudaMemcpyDeviceToDevice);
            }
        }

        // =========================================================
        // Data Transfer
        // =========================================================
        void update_device() override {
            if (!use_host_mirror_) return; 
            if (storage_size_ == 0) return;
            cudaMemcpyAsync(d_ptr_, h_data_.data(), logical_size_ * sizeof(T), cudaMemcpyHostToDevice, DeviceEnv::instance().get_compute_stream());
        }

        void update_host() override {
            if (!use_host_mirror_) return; 
            if (storage_size_ == 0) return;
            cudaStream_t s = DeviceEnv::instance().get_compute_stream();
            cudaMemcpyAsync(h_data_.data(), d_ptr_, logical_size_ * sizeof(T), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
        }

        void update_device(size_t offset, size_t count) override {
            if (!use_host_mirror_) return;
            cudaMemcpyAsync(d_ptr_ + offset, h_data_.data() + offset, count * sizeof(T), cudaMemcpyHostToDevice, DeviceEnv::instance().get_compute_stream());
        }

        void update_host(size_t offset, size_t count) override {
            if (!use_host_mirror_) return;
            cudaStream_t s = DeviceEnv::instance().get_compute_stream();
            cudaMemcpyAsync(h_data_.data() + offset, d_ptr_ + offset, count * sizeof(T), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
        }

        // =========================================================
        // Utils
        // =========================================================
        void fill_zero() override {
            if (storage_size_ == 0) return;
            if (use_host_mirror_) {
                std::fill(h_data_.begin(), h_data_.begin() + logical_size_, static_cast<T>(0));
            }
            cudaMemsetAsync(d_ptr_, 0, storage_size_ * sizeof(T), DeviceEnv::instance().get_compute_stream());
        }

        void set_value(T val) override {
            if (storage_size_ == 0) return;
            if (use_host_mirror_) {
                std::fill(h_data_.begin(), h_data_.begin() + (ptrdiff_t)logical_size_, val);
            }
            cudaStream_t s = DeviceEnv::instance().get_compute_stream();
            int block = 256;
            int grid = std::min((size_t)(WARP_LENGTH * getNumSMs()), (logical_size_ + block - 1) / block);
            set_value_kernel<T><<<grid, block, 0, s>>>(d_ptr_, val, logical_size_);
            
            if (logical_size_ < storage_size_) {
                size_t pad = storage_size_ - logical_size_;
                int grid2 = std::min((size_t)(WARP_LENGTH * getNumSMs()), (pad + block - 1) / block);
                set_value_kernel<T><<<grid2, block, 0, s>>>(d_ptr_ + logical_size_, static_cast<T>(0), pad);
            }
        }

        IDeviceVector<T>* clone() override {
            auto* new_vec = new DeviceVectorImpl<T, Alloc>(this->size(), use_host_mirror_);
            if (this->size() > 0) {
                cudaMemcpyAsync(new_vec->device_ptr(), this->d_ptr_, this->size() * sizeof(T), cudaMemcpyDeviceToDevice, DeviceEnv::instance().get_compute_stream());
                if (use_host_mirror_) {
                    std::copy(this->h_data_.begin(), this->h_data_.begin() + (ptrdiff_t)this->size(), new_vec->host_ptr());
                }
            }
            return new_vec;
        }

        // =========================================================
        // Algorithms (Sort)
        // =========================================================
        void sort() override {
            if (storage_size_ == 0) return;
            
            cudaStream_t stream = DeviceEnv::instance().get_compute_stream();

            // 1. Allocate Double Buffer (Ping-Pong)
            T* d_alt_ptr = nullptr;
            if (cudaMallocAsync(&d_alt_ptr, storage_size_ * sizeof(T), stream) != cudaSuccess) {
                throw std::runtime_error("Sort: Failed to allocate ping-pong buffer");
            }

            cub::DoubleBuffer<T> d_keys(d_ptr_, d_alt_ptr);

            // 2. Query Workspace
            size_t temp_storage_bytes = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, d_keys, logical_size_, 0, sizeof(T) * 8, stream);

            // 3. Get Workspace from DeviceEnv
            void* d_temp_storage = DeviceEnv::instance().get_workspace(temp_storage_bytes, stream);

            // 4. Execute Sort
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, logical_size_, 0, sizeof(T) * 8, stream);

            // 5. Handle Ping-Pong result
            // If the valid data ended up in the temp buffer (d_alt_ptr), copy it back to d_ptr_
            if (d_keys.Current() != d_ptr_) {
                cudaMemcpyAsync(d_ptr_, d_keys.Current(), logical_size_ * sizeof(T), cudaMemcpyDeviceToDevice, stream);
            }

            // Free the ping-pong buffer (workspace is managed by DeviceEnv, so don't free d_temp_storage)
            cudaFreeAsync(d_alt_ptr, stream);
        }

        template <typename ValT>
        void sort_by_key(IDeviceVector<ValT>* values_vec) {
            if (storage_size_ == 0 || values_vec == nullptr) return;
            if (values_vec->size() != this->logical_size_) {
                throw std::runtime_error("Size problem occur for sort_by_key");
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

        // =========================================================
        // Getters & Reductions
        // =========================================================
        T* host_ptr() override { return use_host_mirror_ ? h_data_.data() : nullptr; }
        T* device_ptr() override { return d_ptr_; }
        
        size_t size() const override { return logical_size_; }
        size_t logical_size() const override { return logical_size_; }
        size_t capacity() const override { return capacity_; }

        T sum_partial(size_t n) override { return reduce_impl_(0, n); }
        T min_partial(size_t n) override { return reduce_impl_(1, n); }
        T max_partial(size_t n) override { return reduce_impl_(2, n); }
        T sum() override { return reduce_impl_(0, this->size()); }
        T min() override { return reduce_impl_(1, this->size()); }
        T max() override { return reduce_impl_(2, this->size()); }
    };
} 

#endif // DEVICE_VECTOR_CUH