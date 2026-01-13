PROGRAM profile_resize
  USE Device_Vector
  USE cudafor
  USE iso_c_binding
  IMPLICIT NONE

  ! 1. 增加迭代次數，避免時間太短
  INTEGER(8), PARAMETER :: N_SMALL = 1024 * 1024 * 10
  INTEGER(8), PARAMETER :: N_LARGE = 1024 * 1024 * 20
  INTEGER, PARAMETER    :: ITERATIONS = 2000 

  REAL(4), ALLOCATABLE, DEVICE :: d_cf(:)
  REAL(4), ALLOCATABLE :: h_acc(:)
  TYPE(device_vector_r4_t) :: vec

  INTEGER :: i, istat
  INTEGER(8) :: c_start, c_end, c_rate
  REAL(8) :: t_acc, t_cf, t_dv_smart
  
  ! 加入一個 dummy 變數防止優化
  REAL(4) :: dummy_val

  CALL device_env_init(0, 1)

  ! [2] CUDA Fortran 修正版
  PRINT *, "Testing CUDA Fortran Resize..."
  ALLOCATE(d_cf(N_SMALL))
  istat = cudaDeviceSynchronize()
  
  CALL SYSTEM_CLOCK(COUNT=c_start, COUNT_RATE=c_rate)

  DO i = 1, ITERATIONS
     DEALLOCATE(d_cf)
     ALLOCATE(d_cf(N_LARGE))
     d_cf(1) = 1.0 
     
     DEALLOCATE(d_cf)
     ALLOCATE(d_cf(N_SMALL))
     d_cf(1) = 1.0
  END DO
  
  istat = cudaDeviceSynchronize()
  CALL SYSTEM_CLOCK(COUNT=c_end)
  
  dummy_val = d_cf(1) 
  
  t_cf = REAL(c_end - c_start) / REAL(c_rate)
  PRINT *, "CUDA Fortran time:", t_cf

  ! [3] DeviceVector 修正版
  PRINT *, "Testing DeviceVector..."
  CALL vec%create(N_SMALL, 0)
  
  CALL SYSTEM_CLOCK(COUNT=c_start)
  DO i = 1, ITERATIONS
    CALL vec%resize(N_LARGE)
    CALL vec%resize(N_SMALL)
  END DO
  istat = cudaDeviceSynchronize()
  CALL SYSTEM_CLOCK(COUNT=c_end)
  t_dv_smart = REAL(c_end - c_start) / REAL(c_rate)

  CALL vec%free()
  CALL device_env_finalize()

  ! ------------------------------------------------------------------
  ! Report
  ! ------------------------------------------------------------------
  PRINT *, "=========================================================="
  PRINT *, "                  PERFORMANCE RESULTS                     "
  PRINT *, "=========================================================="
  PRINT '(A, F10.4, A)', " [1] OpenACC Total Time      : ", t_acc, " s"
  PRINT '(A, F10.4, A)', " [2] CUDA Fortran Total Time : ", t_cf,  " s"
  PRINT '(A, F10.4, A)', " [3] DeviceVector            : ", t_dv_smart, " s"
  PRINT *, "----------------------------------------------------------"
  PRINT '(A, F10.2, A)', " Smart Speedup vs CUDA Fortran: ", t_cf / t_dv_smart, " x"
  PRINT *, "=========================================================="
  
  CALL device_env_finalize()

END PROGRAM profile_resize