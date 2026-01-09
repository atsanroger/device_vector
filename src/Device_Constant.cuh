#pragma once
#include <cuda_runtime.h>

constexpr int VECTOR_LENGTH = 256;
constexpr int WARP_LENGTH   = 32; // Dont change this number till Nvidai change it

inline constexpr size_t ceil_warp_length(size_t n) {
    return (n + WARP_LENGTH - 1) & ~(size_t)(WARP_LENGTH - 1);
}

inline int getNumSMs() {
    static int numSMs = []() {
        int count = 0;
        cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, 0);
        return count;
    }();
    return numSMs;
}