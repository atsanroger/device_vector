namespace GPU {

    // =========================================================
    // CUDA Kernels (Template Helper Functions)
    // =========================================================
    
    // 1. Set Value Kernel (Stride Loop)
    template <typename T>
    __global__ void set_value_kernel(T* ptr, T val, size_t n) {
        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = gridDim.x * blockDim.x;
        
        for (; idx < n; idx += stride) {
            ptr[idx] = val;
        }
    }    

    // 2. Gather Kernel (Stride Loop)
    // dst[i] = src[map[i]]
    template <typename T>
    __global__ void gather_kernel(const T* __restrict__ src, 
                                  const int* __restrict__ map, 
                                  T* __restrict__ dst, 
                                  size_t n) {

        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = gridDim.x * blockDim.x;

        for (; idx < n; idx += stride) {
            dst[idx] = src[map[idx]];
        }
    }

    template <typename T>
    __global__ void scal_kernel(T* ptr, T alpha, size_t n) {

        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = gridDim.x * blockDim.x;

        for (; idx < n; idx += stride) {
            ptr[idx] *= alpha;
        }
    }

    template <typename T>
    __global__ void axpy_kernel(T alpha, const T* __restrict__ x, T* y, size_t n) {

        size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = gridDim.x * blockDim.x;

        for (; idx < n; idx += stride) {
            y[idx] = alpha * x[idx] + y[idx];
        }
    }

} // namespace GPU