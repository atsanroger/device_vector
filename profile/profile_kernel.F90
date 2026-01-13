MODULE physics_kernels
    USE cudafor
    IMPLICIT NONE
CONTAINS
    ! [CUDA Kernel] 手寫的 CUDA Kernel，用於 CUDA Fortran 和 DeviceVector
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
    USE iso_c_binding
    USE openacc
    IMPLICIT NONE

    INTEGER(8), PARAMETER :: N = 10000003_8 
    INTEGER, PARAMETER :: BLOCK_SIZE = 128
    INTEGER :: GRID_SIZE
    INTEGER :: i_iter, istat
    INTEGER, PARAMETER :: ITERATIONS = 10000 

    ! 變數宣告
    REAL(4), ALLOCATABLE :: h_acc(:)        ! OpenACC 用 (Host Array)
    REAL(4), DEVICE, ALLOCATABLE :: d_raw(:) ! CUDA Fortran 用 (Device Array)
    TYPE(device_vector_r4_t) :: vec          ! DeviceVector 用
    
    ! DeviceVector 指標轉換用
    REAL(4), DEVICE, POINTER :: d_ptr_fix(:) => NULL()
    TYPE(C_PTR) :: cptr
    
    INTEGER(8) :: count_start, count_end, count_rate
    REAL(8) :: t_acc, t_cuda, t_vec
    INTEGER :: k

    CALL device_env_init(0, 1)
    GRID_SIZE = (INT(N) + BLOCK_SIZE - 1) / BLOCK_SIZE
    
    PRINT *, "=========================================================="
    PRINT *, "   KERNEL BENCHMARK: The Holy Trinity"
    PRINT *, "   N =", N
    PRINT *, "=========================================================="

    ! ==================================================================
    ! [1] Pure OpenACC (Compiler Generated Kernel)
    ! ==================================================================
    PRINT *, "Running [1] OpenACC..."
    ALLOCATE(h_acc(N))
    !$acc enter data create(h_acc)
    
    ! 初始化
    !$acc kernels present(h_acc)
    h_acc(:) = 0.0
    !$acc end kernels
    
    CALL device_synchronize()
    CALL SYSTEM_CLOCK(COUNT=count_start, COUNT_RATE=count_rate)
    
    DO i_iter = 1, ITERATIONS
        ! 這裡讓編譯器自動生成 Kernel，這是 OpenACC 的標準用法
        !$acc parallel loop present(h_acc)
        DO k = 1, N
            h_acc(k) = h_acc(k) + 0.01
        END DO
        !$acc end parallel loop
    END DO
    
    CALL device_synchronize()
    CALL SYSTEM_CLOCK(COUNT=count_end)
    t_acc = REAL(count_end - count_start) / REAL(count_rate)
    
    !$acc exit data delete(h_acc)
    DEALLOCATE(h_acc)

    ! ==================================================================
    ! [2] Raw CUDA Fortran (Explicit Kernel)
    ! ==================================================================
    PRINT *, "Running [2] Raw CUDA Fortran..."
    ALLOCATE(d_raw(N))
    
    ! 使用 OpenACC 初始化 (確保公平)
    !$acc kernels deviceptr(d_raw)
    d_raw(:) = 0.0
    !$acc end kernels
    
    CALL device_synchronize()
    CALL SYSTEM_CLOCK(COUNT=count_start)
    
    DO i_iter = 1, ITERATIONS
        ! 呼叫手寫的 CUDA Kernel
        CALL physics_update_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_raw, N, 0.01)
    END DO
    
    CALL device_synchronize()
    CALL SYSTEM_CLOCK(COUNT=count_end)
    t_cuda = REAL(count_end - count_start) / REAL(count_rate)
    
    DEALLOCATE(d_raw)

    ! ==================================================================
    ! [3] DeviceVector (Mode 2) + Explicit Kernel
    ! ==================================================================
    PRINT *, "Running [3] DeviceVector..."
    CALL vec%create_vector(N)
    
    ! 取得指標
    cptr = vec%device_ptr()
    CALL C_F_POINTER(cptr, d_ptr_fix, [N])
    
    ! 初始化
    !$acc kernels deviceptr(d_ptr_fix)
    d_ptr_fix(:) = 0.0
    !$acc end kernels
    
    CALL device_synchronize()
    CALL SYSTEM_CLOCK(COUNT=count_start)
    
    DO i_iter = 1, ITERATIONS
        ! 傳入 DeviceVector 的指標給 CUDA Kernel
        CALL physics_update_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_ptr_fix(1), N, 0.01)
    END DO
    
    CALL device_synchronize()
    CALL SYSTEM_CLOCK(COUNT=count_end)
    t_vec = REAL(count_end - count_start) / REAL(count_rate)
    
    CALL vec%free()

    ! ==================================================================
    ! Final Report
    ! ==================================================================
    PRINT *, "=========================================================="
    PRINT '(A, F10.6, A)', " [1] OpenACC Time      : ", t_acc,  " s"
    PRINT '(A, F10.6, A)', " [2] Raw CUDA Time     : ", t_cuda, " s"
    PRINT '(A, F10.6, A)', " [3] DeviceVector Time : ", t_vec,  " s"
    PRINT *, "----------------------------------------------------------"
    PRINT '(A, F10.4, A)', " Speedup (openACC vs DV): ", t_acc / t_vec, " X"
    PRINT '(A, F10.4, A)', " Speedup (CUDA F  vs DV): ", t_cuda / t_vec, " X"
    PRINT *, "=========================================================="

    CALL device_env_finalize()

END PROGRAM profile_kernel