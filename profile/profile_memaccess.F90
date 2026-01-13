MODULE physics_kernels
    USE cudafor
    IMPLICIT NONE
CONTAINS
    ATTRIBUTES(GLOBAL) SUBROUTINE physics_update_kernel(arr, n_val, dt)
        REAL(4), DEVICE :: arr(*)
        INTEGER(8), VALUE :: n_val
        REAL(4), VALUE :: dt
        INTEGER :: i
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        IF (i <= n_val) THEN
            arr(i) = arr(i) + dt
        END IF
    END SUBROUTINE physics_update_kernel
END MODULE physics_kernels

PROGRAM profile_kernel
    USE Device_Vector
    USE cudafor
    USE physics_kernels
    IMPLICIT NONE

    INTEGER(8), PARAMETER :: N = 10000003_8 
    INTEGER, PARAMETER :: BLOCK_SIZE = 128
    INTEGER :: GRID_SIZE
    INTEGER :: i_iter, istat
    INTEGER, PARAMETER :: ITERATIONS = 10000 

    TYPE(device_vector_r4_t) :: vec
    REAL(4), DEVICE, ALLOCATABLE :: d_raw(:)
    
    ! ⚠️ 關鍵：宣告一個 local 的 device pointer 來接
    REAL(4), DEVICE, POINTER :: d_ptr_fix(:) => NULL()
    
    INTEGER(8) :: count_start, count_end, count_rate
    REAL(8) :: time_raw, time_vec

    CALL device_env_init(0, 1)
    GRID_SIZE = (INT(N) + BLOCK_SIZE - 1) / BLOCK_SIZE
    
    PRINT *, "=========================================================="
    PRINT *, "   KERNEL BENCHMARK: Memory Alignment Impact"
    PRINT *, "   N =", N, " (Odd number, unaligned tail)"
    PRINT *, "=========================================================="

    ! --- [A] Raw CUDA Fortran ---
    ALLOCATE(d_raw(N))
    d_raw = 0.0
    istat = cudaDeviceSynchronize()

    CALL SYSTEM_CLOCK(COUNT=count_start, COUNT_RATE=count_rate)
    DO i_iter = 1, ITERATIONS
        CALL physics_update_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_raw, N, 0.01)
    END DO
    istat = cudaDeviceSynchronize()
    CALL SYSTEM_CLOCK(COUNT=count_end)
    
    time_raw = REAL(count_end - count_start) / REAL(count_rate)
    PRINT '(A, F10.6, A)', " [A] Raw CUDA Fortran Time : ", time_raw, " s"

    ! --- [B] DeviceVector ---
    CALL vec%create(N, 0)
    
    !$acc kernels deviceptr(vec%data)
    vec%data = 0.0
    !$acc end kernels
    
    d_ptr_fix => vec%data 
    
    istat = cudaDeviceSynchronize()

    CALL SYSTEM_CLOCK(COUNT=count_start)
    DO i_iter = 1, ITERATIONS
        CALL physics_update_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_ptr_fix(1), N, 0.01)
    END DO
    istat = cudaDeviceSynchronize()
    CALL SYSTEM_CLOCK(COUNT=count_end)
    
    time_vec = REAL(count_end - count_start) / REAL(count_rate)
    PRINT '(A, F10.6, A)', " [B] DeviceVector Time     : ", time_vec, " s"

    PRINT *, "----------------------------------------------------------"
    PRINT '(A, F10.4, A)', " Speedup (Vector / Raw): ", time_raw / time_vec, " X"
    PRINT *, "=========================================================="

    IF (ALLOCATED(d_raw)) DEALLOCATE(d_raw)
    CALL vec%free()
    CALL device_env_finalize()

END PROGRAM profile_kernel