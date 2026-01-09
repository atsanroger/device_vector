# GPU DeviceVector Library Guide (Development Principles / Usage / File & Class Overview)

> **Scope**: This document describes a **Fortran ↔ C/CUDA** DeviceVector wrapper: Fortran holds an opaque `void*` handle, while C++/CUDA manages host/device memory, streams, and common vector operations (fill/set/scal/axpy/gather/reduce/sort).  
> **Sources used here**: In the current sandbox, the files available for inspection are:
> - `DeviceVector.cuh.safeB2.patched`
> - `c_interface.cu.patched`
>
> ⚠️ The other filenames you mentioned earlier (`DeviceEnv.*`, `DevicePtrManager.*`, `Device_Kernels.cuh`, `Device_Interface.F90`, `DeviceComm.*`) are **not present** in this sandbox folder, so Section 3 describes them only as **expected roles / inferred interfaces**. If you drop those files into the same folder, I can update this doc with exact API lists and code-level cross references.

---

## What this wrapper is trying to solve

1. **Fortran does not deal with CUDA memory management**: it only holds a `TYPE(C_PTR)` handle and (optionally) a host pointer view.
2. **Multiple host memory modes**:
   - `mode=0`: Pinned host (`cudaHostAlloc`)
   - `mode=1`: Pageable host (regular `new/delete` / `std::vector`)
   - `mode=2`: Mapped / Zero-copy (`cudaHostAllocMapped` + `cudaHostGetDevicePointer`)
3. **Warp-friendly padding**: storage length is rounded up to a multiple of `WARP_LENGTH (=32)` so kernels can run at a fixed aligned size.
4. **Correctness-first (Safe-B)**: for reductions that can be corrupted by padding (notably `min/max`), we use:
   **fill padding tail with identity → reduce → immediately reset padding tail back to 0**, so that:
   - results are equivalent to reducing only the logical range
   - after the API returns, padding remains predictable (0), which helps debugging/dumping and prevents accidental downstream misuse from exploding

---

# 1. Development Principles (Design / Coding Principles)

## 1.1 External ABI: Fortran-friendly, minimal surface, explicit types
- **Expose only C ABI**: every function callable from Fortran should be in `extern "C"` and use C-compatible types (`size_t`, `int`, `float`, `double`, `void*`).
- **Handle = `void*`**: Fortran stores an opaque pointer (actually `IDeviceVector<T>*`).
- **No exceptions across ABI**: C++ may throw internally, but C ABI should convert failures into error codes or `fprintf + exit`. (The current code mixes styles; for robustness, unify one strategy.)

## 1.2 Padding Strategy & Contract (Safe-B)
### Terminology
- `logical_size`: the user’s real number of valid elements (e.g., 1000)
- `storage_size`: aligned length after padding (e.g., 1024)
- padding tail: index range `[logical_size, storage_size)`

### Safe-B contract (recommended to document explicitly)
1. **GPU kernels may run over `storage_size`** (aligned, consistent throughput).
2. **After any public API returns, padding tail is in a predictable state** (in this version: padding tail is 0).
3. **Correctness of `min/max` is guaranteed by the library**:
   - Before reduce: temporarily fill padding tail with the identity value  
     - min: `+INF` (float) or `numeric_limits<T>::max()` (integer)
     - max: `-INF` (float) or `numeric_limits<T>::lowest()` (integer)
   - After reduce: reset padding tail back to 0 (safety)
4. **`sum` identity is already 0**: with padding==0, reducing `storage_size` gives the same result as reducing the logical range.

> Important: Safe-B does **not** mean “padding is part of the data.” It means “even if someone dumps/uses the padded array by mistake, it is less likely to blow up.”

## 1.3 Memory Lifetime: must be safe to free (avoid finalize-order pitfalls)
- **Do not depend on `DeviceEnv` being alive to free**: `DeviceVectorImpl` records `device_id_` and provides a safe free path (even if `DeviceEnv` has been finalized, it can allocate a temporary stream to `cudaFreeAsync + sync`).
- **Mapped mode does not own a device allocation**: zero-copy uses `cudaHostGetDevicePointer`, so it must not call `cudaFree` on that pointer.

## 1.4 Streams & Synchronization: enforce ordering on a single stream
- All device ops (kernels / memcpy / CUB reduce / CUB sort) should be enqueued on the same compute stream.
- Reductions that return a host scalar should use `cudaMemcpyAsync(..., stream)` + `cudaStreamSynchronize(stream)` to avoid subtle default-stream mode differences.

## 1.5 Reliability-first Debug Invariants
Recommended invariants (useful for assertions / debugging):
- `storage_size % 32 == 0` (unless `storage_size == 0`)
- `padding tail == 0` after any public API returns (Safe-B’s core guarantee)
- `vec_size()` returns storage size; but you should also expose logical size (the class has `logical_size()`, but C ABI currently doesn’t export it—add `vec_logical_size_*`)

---

# 2. Usage (How to Use)

This section describes the **C ABI** view; Fortran uses `BIND(C)` to declare these symbols.

## 2.1 Init / Finalize
```c
void device_env_init(int rank, int gpus_per_node);
void device_env_finalize(void);
void device_synchronize(void);
```

- Minimal flow: call `device_env_init(...)` at program start and `device_env_finalize()` before exit.
- If you are not doing MPI/multi-GPU yet: `rank=0, gpus_per_node=1`.

## 2.2 Create / Destroy vectors (example: `r8` = double)
The C ABI is generated by macros (one set per type):

- Create: `void* vec_create_<suffix>(size_t n, int mode)`
- Destroy: `void  vec_delete_<suffix>(void* h)`
- Resize/reserve: `vec_resize_<suffix>`, `vec_reserve_<suffix>`

Suffix mapping (from `c_interface.cu.patched`):
- `r4` = `float`
- `r8` = `double`
- `i8` = `long long`
- `i4` = `int`

Host `mode`:
- `0`: Pinned host (recommended for frequent H2D/D2H)
- `1`: Pageable host (simplest; transfers typically slower)
- `2`: Mapped / Zero-copy (no explicit copy, but often slower on discrete GPUs for large bandwidth workloads)

Example:
```c
void* v = vec_create_r8(1000, 0);   // logical=1000, storage will be padded to a multiple of 32
...
vec_delete_r8(v);
```

## 2.3 Host pointer view (Fortran typically uses `c_f_pointer`)
```c
double* vec_host_r8(void* h);
size_t  vec_size_r8(void* h);      // returns padded size (storage length)
size_t  vec_capacity_r8(void* h);  // device capacity
```

Typical pattern:
1. Get host pointer via `vec_host_*`, fill data (at least the logical region).
2. Upload to device with `vec_upload_*`.

> ⚠️ `vec_size_*()` is the **padded** size. If Fortran uses it as the shape for `c_f_pointer`, the view length will be padded.  
> Safe-B helps because the tail stays 0 after APIs return, but semantically you should still track `logical_size` for I/O and comparisons.

## 2.4 Upload / Download (full / partial)
```c
void vec_upload_r8(void* h);
void vec_download_r8(void* h);
void vec_upload_part_r8(void* h, size_t off, size_t cnt);
void vec_download_part_r8(void* h, size_t off, size_t cnt);
```

## 2.5 Common device operations (enqueued on compute stream)
```c
void vec_fill_zero_r8(void* h);
void vec_set_value_r8(void* h, double val);
void vec_scal_r8(void* h, double alpha);        // v *= alpha
void vec_axpy_r8(void* hx, void* hy, double a); // y += a*x
void vec_gather_r8(void* h_src, void* h_map_i4, void* h_dst);
```

Notes:
- `vec_gather` requires `src` and `dst` device pointers to be different (the code checks and calls `exit(1)` on violation).

## 2.6 Reductions (full / partial)
```c
double vec_sum_r8(void* h);
double vec_min_r8(void* h);
double vec_max_r8(void* h);

double vec_sum_partial_r8(void* h, size_t n);
double vec_min_partial_r8(void* h, size_t n);
double vec_max_partial_r8(void* h, size_t n);
```

### Recommended under Safe-B
- For maximum semantic clarity: prefer `*_partial(h, logical_size)`.
- For convenience: full `min/max` is still safe—Safe-B ensures the result matches the logical range, and the padding tail is reset to 0 before returning.

## 2.7 Sorting (CUB radix sort, key/value pairs)
```c
void vec_sort_pairs_i4_c(void* d_keys_in, void* d_keys_buf,
                         void* d_vals_in, void* d_vals_buf,
                         size_t num_items);
```

- Wrapper around CUB radix sort using `DoubleBuffer` ping-pong buffers, caching temp workspace to avoid repeated allocations.
- ⚠️ `num_items` should usually be the **logical** item count, not the padded size (unless you deliberately want the tail included).

---

# 3. File & Class Overview

## 3.1 `DeviceVector.cuh.safeB2.patched`
### Main types
- `template<typename T> class IDeviceVector`
  - Unified interface: `resize/reserve`, `update_device/update_host`, `fill_zero/set_value`,
    `scal/axpy/gather`, `sum/min/max`, `host_ptr/device_ptr`, and `size/logical_size/capacity`.

- `template<typename T, typename HostAlloc> class DeviceVectorImpl : public IDeviceVector<T>`
  - Implementation details:
    - Host buffer: `std::vector<T, HostAlloc> h_data_`
    - Device pointer: `T* d_ptr_` (mapped mode uses `cudaHostGetDevicePointer`)
    - Sizes:
      - `logical_size_`: original requested length
      - `storage_size_`: padded length via `ceil_warp_length(n)`
    - Safe-B invariant: after any public API returns, padding tail is 0.

- `PinnedAllocator<T>`
  - Uses `cudaHostAlloc` pinned memory to improve transfer bandwidth.

- `MappedAllocator<T>`
  - Uses `cudaHostAllocMapped` + `cudaHostGetDevicePointer`.
  - Performance depends heavily on hardware topology; on discrete GPUs, large streaming workloads often perform poorly.

### Safe-B reduction core
`reduce_impl_(op_type, num_items)`:
- If `num_items == storage_size_` and `logical < storage`:
  - For `min/max`: run a tiny tail-fill kernel to set `[logical, storage)` to identity
  - Call CUB reduce
  - Reset `[logical, storage)` back to 0 (safety)
- In mapped mode: reduce is performed on host (logical range only)

## 3.2 `c_interface.cu.patched`
### Main role
- Exposes C ABI wrappers around the C++ classes for Fortran `BIND(C)`.
- Macro `DEFINE_VEC_INTERFACE(SUFFIX, TYPE)` generates a full API set per type:
  - create/delete/resize/reserve
  - size/capacity
  - host_ptr/device_ptr
  - upload/download (full/partial)
  - fill_zero/set_value
  - scal/axpy/gather
  - clone
  - sum/min/max (full/partial)

### Sorting wrapper
- `vec_sort_pairs_i4_c(...)`: CUB radix sort + cached temp storage.

---

## 3.3 (To be filled) Other files: expected roles
> ⚠️ Needs the actual source files for exact API lists.

- `DeviceEnv.cuh/.cu` (expected)
  - Singleton managing CUDA device id and compute/comm streams.
  - Provides `get_compute_stream()` etc.

- `Device_Kernels.cuh` (expected)
  - Small kernels: `set_value_kernel`, `scal_kernel`, `axpy_kernel`, `gather_kernel`, etc.

- `DevicePtrManager.cuh/.cu` (expected)
  - Host↔device pointer mapping manager (if you need host pointer → device pointer translation).

- `Device_Interface.F90` (expected)
  - Fortran module with `interface, bind(C)` declarations for the C ABI.
  - Wraps `vec_host_*` pointer into a Fortran array via `c_f_pointer`.

- `DeviceComm.cuh/.cu` (expected)
  - MPI/NCCL/halo exchange (you said MPI/multi-GPU is not the focus for now; can be excluded from build).

---

## 4. Suggested Next APIs (Quality-of-life + safety)
1. `vec_logical_size_<suffix>(handle)`
   - The class has `logical_size()`, but C ABI does not export it.
   - Exporting it avoids “remember n in Fortran” mistakes, and makes partial reduce and I/O safer.

2. `vec_set_logical_size_<suffix>(handle, n)` (optional)
   - If you want to keep storage fixed and change only the valid range (vector view semantics).

---

## Version Info
- Generated: 2026-01-09 (Asia/Tokyo)
- Based on:
  - `DeviceVector.cuh.safeB2.patched`
  - `c_interface.cu.patched`
