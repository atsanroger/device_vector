#pragma once

// 1. Basic CUDA library
#include <cuda_runtime.h>
#include <cstdio>

// 2. Core library
#include "Device_Constant.cuh"  // Costant only for GPU
#include "DeviceEnv.cuh"        // Singleton
#include "DevicePtrManager.cuh" // Pointer Mapping (Host<->Device Map)
#include "DeviceVector.cuh"     // Container (Template Class)
//#include "DeviceComm.cuh"       // MultiGPU enviroment 

// 3. NVIDIA THIRD PARTY LIBRARY
#include <cub/cub.cuh>          // High performance radix sort 