PROGRAM profile_resize
  USE Device_Vector
  USE cudafor        ! 這裡需要編譯器開啟 -cuda
  USE iso_c_binding
  USE openacc        ! 這裡需要編譯器開啟 -acc
  IMPLICIT NONE

  ! 參數設定
  INTEGER(8), PARAMETER :: N_SMALL = 1024 * 1024 * 10
  INTEGER(8), PARAMETER :: N_LARGE = 1024 * 1024 * 20
  INTEGER, PARAMETER    :: ITERATIONS = 100  ! 建議先調小一點，測過再調大

  ! OpenACC Host Array
  REAL(4), ALLOCATABLE :: h_acc(:)

  ! CUDA Fortran Device Array
  REAL(4), ALLOCATABLE, DEVICE :: d_cf(:)
  
  ! DeviceVector
  TYPE(device_vector_r4_t) :: vec

  INTEGER :: i, istat
  INTEGER(8) :: c_start, c_end, c_rate
  REAL(8) :: t_acc, t_cf, t_dv_smart
  
  REAL(4) :: dummy_val

  CALL device_env_init(0, 1)

  PRINT *, "=========================================================="
  PRINT *, "   RESIZE BENCHMARK: OpenACC vs CUDA Fortran vs DeviceVector"
  PRINT *, "=========================================================="

  ! ==================================================================
  ! [1] OpenACC (Manual Resize Overhead)
  ! ==================================================================
  PRINT *, "Testing OpenACC Resize..."
  ALLOCATE(h_acc(N_SMALL))
  !$acc enter data create(h_acc)
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(COUNT=c_start, COUNT_RATE=c_rate)
  
  DO i = 1, ITERATIONS
      !$acc exit data delete(h_acc)
      DEALLOCATE(h_acc)
      ALLOCATE(h_acc(N_LARGE))
      !$acc enter data create(h_acc)
      
      !$acc kernels present(h_acc)
      h_acc(1) = 1.0 
      !$acc end kernels
      
      !$acc exit data delete(h_acc)
      DEALLOCATE(h_acc)
      ALLOCATE(h_acc(N_SMALL))
      !$acc enter data create(h_acc)
      
      !$acc kernels present(h_acc)
      h_acc(1) = 1.0
      !$acc end kernels
  END DO
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(COUNT=c_end)
  t_acc = REAL(c_end - c_start) / REAL(c_rate)
  
  IF (ALLOCATED(h_acc)) THEN
      !$acc exit data delete(h_acc)
      DEALLOCATE(h_acc)
  END IF

  ! ==================================================================
  ! [2] CUDA Fortran (Manual Alloc/Dealloc)
  ! ==================================================================
  PRINT *, "Testing CUDA Fortran Resize..."
  ALLOCATE(d_cf(N_SMALL))
  istat = cudaDeviceSynchronize()
  
  CALL SYSTEM_CLOCK(COUNT=c_start)

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
  
  t_cf = REAL(c_end - c_start) / REAL(c_rate)
  IF (ALLOCATED(d_cf)) DEALLOCATE(d_cf)

  ! ==================================================================
  ! [3] DeviceVector (Compute Mode)
  ! ==================================================================
  PRINT *, "Testing DeviceVector (Compute Mode)..."
  CALL vec%create_vector(N_SMALL)
  
  CALL SYSTEM_CLOCK(COUNT=c_start)
  DO i = 1, ITERATIONS
    CALL vec%resize(N_LARGE)
    CALL vec%resize(N_SMALL)
  END DO
  ! DeviceVector 內部是 C++，我們用我們寫好的同步
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(COUNT=c_end)
  t_dv_smart = REAL(c_end - c_start) / REAL(c_rate)

  CALL vec%free()
  CALL device_env_finalize()

  ! Report (略，維持原樣)
  PRINT *, "=========================================================="
  PRINT *, "                   PERFORMANCE RESULTS                   "
  PRINT *, "=========================================================="
  PRINT '(A, F10.4, A)', " [1] OpenACC Total Time      : ", t_acc,      " s"
  PRINT '(A, F10.4, A)', " [2] CUDA Fortran Total Time : ", t_cf,       " s"
  PRINT '(A, F10.4, A)', " [3] DeviceVector Total Time : ", t_dv_smart, " s"
  PRINT *, "----------------------------------------------------------"
  PRINT '(A, F10.2, A)', " Speedup (DV vs OpenACC): ", t_acc / t_dv_smart, " x"
  PRINT '(A, F10.2, A)', " Speedup (DV vs CF)     : ", t_cf / t_dv_smart, " x"
  PRINT *, "=========================================================="

END PROGRAM profile_resize