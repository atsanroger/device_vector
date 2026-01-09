PROGRAM profile_all
  USE Device_Vector
  USE cudafor
  IMPLICIT NONE

  INTEGER(8), PARAMETER :: iterations = 100000 
  INTEGER(8), PARAMETER :: base_n = 10000003
  
  REAL(4), ALLOCATABLE :: h_a(:)
  REAL(4), DEVICE, ALLOCATABLE :: d_cf(:)
  TYPE(device_vector_r4_t) :: vec_dv
  
  ! 局部指標：徹底解決 S-0519 與 S-0528
  REAL(4), DEVICE, POINTER :: d_ptr_dv(:)
  
  ! 運算暫存
  REAL(4) :: tmp_val
  INTEGER(8) :: i, c_val, c_rate
  REAL(8) :: t1, t2, time_acc, time_cf, time_dv
  INTEGER :: istat

  CALL device_env_init(0, 1)
  PRINT *, "=========================================================="
  PRINT *, "   ULTIMATE PERFORMANCE SHOWDOWN (N =", base_n, ")"
  PRINT *, "=========================================================="

  ! --- [1] OpenACC ---
  PRINT *, "Running OpenACC..."
  ALLOCATE(h_a(base_n))
  h_a = 0.0
  CALL SYSTEM_CLOCK(c_val, c_rate)
  t1 = REAL(c_val) / REAL(c_rate)
  
  DO i = 1, iterations
     !$acc enter data copyin(h_a)
     !$acc serial deviceptr(h_a)
     ! 拆開寫，避免 S-0519
     tmp_val = h_a(1)
     h_a(1) = tmp_val + 1.0
     !$acc end serial
     !$acc exit data delete(h_a)
  END DO
  
  CALL SYSTEM_CLOCK(c_val)
  t2 = REAL(c_val) / REAL(c_rate)
  time_acc = t2 - t1
  DEALLOCATE(h_a)

  ! --- [2] CUDA Fortran ---
  PRINT *, "Running CUDA Fortran..."
  CALL SYSTEM_CLOCK(c_val)
  t1 = REAL(c_val) / REAL(c_rate)
  
  DO i = 1, iterations
     ALLOCATE(d_cf(base_n))
     !$acc serial deviceptr(d_cf)
     tmp_val = d_cf(1)
     d_cf(1) = tmp_val + 1.0
     !$acc end serial
     DEALLOCATE(d_cf)
  END DO
  
  CALL SYSTEM_CLOCK(c_val)
  t2 = REAL(c_val) / REAL(c_rate)
  time_cf = t2 - t1

  ! --- [3] DeviceVector ---
  PRINT *, "Running DeviceVector..."
  CALL vec_dv%create(base_n, 0)
  ! 把指標洗出來
  d_ptr_dv => vec_dv%data 
  
  CALL SYSTEM_CLOCK(c_val)
  t1 = REAL(c_val) / REAL(c_rate)
  
  DO i = 1, iterations
     CALL vec_dv%resize(base_n)
     
     !$acc serial deviceptr(d_ptr_dv)
     tmp_val = d_ptr_dv(1)
     d_ptr_dv(1) = tmp_val + 1.0
     !$acc end serial
  END DO
  
  istat = cudaDeviceSynchronize()
  CALL SYSTEM_CLOCK(c_val)
  t2 = REAL(c_val) / REAL(c_rate)
  time_dv = t2 - t1

  PRINT *, "----------------------------------------------------------"
  PRINT '(A, F10.4, A)', " [1] OpenACC Time      : ", time_acc, " s"
  PRINT '(A, F10.4, A)', " [2] CUDA Fortran Time : ", time_cf,  " s"
  PRINT '(A, F10.4, A)', " [3] DeviceVector Time : ", time_dv,  " s"
  PRINT *, "----------------------------------------------------------"
  PRINT '(A, F10.2, A)', " >>> Speedup vs ACC: ", time_acc / time_dv, " X"
  PRINT '(A, F10.2, A)', " >>> Speedup vs CF : ", time_cf / time_dv, " X"
  PRINT *, "=========================================================="

  CALL vec_dv%free()
  CALL device_env_finalize()
END PROGRAM profile_all