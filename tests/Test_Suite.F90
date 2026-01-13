PROGRAM Test_Suite
  USE Device_Vector
  USE iso_c_binding
  IMPLICIT NONE

  INTEGER :: failures = 0

  PRINT *, "=========================================================="
  PRINT *, "🚀 STARTING GPU FORTRAN TEST SUITE (SEMANTIC API)"
  PRINT *, "=========================================================="

  CALL device_env_init(0, 1)

  CALL test_lifecycle_buffer()
  CALL test_lifecycle_vector()
  CALL test_copy_vector_to_buffer()
  CALL test_reductions()
  CALL test_sort()
  CALL test_double_precision()
  
  CALL device_env_finalize()

  PRINT *, "=========================================================="
  IF (failures == 0) THEN
      PRINT *, "✅ ALL TESTS PASSED!"
  ELSE
      PRINT *, "❌ FAILED TESTS: ", failures
      ! 這裡是測試腳本，可以保留 STOP 1 讓 CI/CD 抓到錯誤
      STOP 1
  END IF
  PRINT *, "=========================================================="

CONTAINS
  
  ! ====================================================================
  ! Test 1: DeviceBuffer - 測試標準 Host 指針存取
  ! ====================================================================
  SUBROUTINE test_lifecycle_buffer()
    TYPE(device_vector_i4_t) :: buf
    PRINT *, "--- [Test 1] DeviceBuffer (Transfer Mode) ---"
    
    CALL buf%create_buffer(100_8, pinned=.TRUE.)
    
    IF (.NOT. ASSOCIATED(buf%ptr)) CALL assert_fail("Buffer host pointer missing")
    IF (SIZE(buf%ptr) /= 100) CALL assert_fail("Buffer size mismatch")
    
    CALL buf%resize(200_8)
    IF (SIZE(buf%ptr) /= 200) CALL assert_fail("Buffer resize failed")
    
    CALL buf%free()
    IF (ASSOCIATED(buf%ptr)) CALL assert_fail("Buffer pointer not nullified")
    PRINT *, "   [PASS]"
  END SUBROUTINE test_lifecycle_buffer

  ! ====================================================================
  ! Test 2: DeviceVector - 測試純 GPU 模式（不應有關聯的 Host 指針）
  ! ====================================================================
  SUBROUTINE test_lifecycle_vector()
    TYPE(device_vector_i4_t) :: vec
    PRINT *, "--- [Test 2] DeviceVector (Compute Mode) ---"
    
    CALL vec%create_vector(100_8)
    
    ! 在 Mode 2 (純 Device) 下，sync_ptr 會找不到 host 地址，ptr 應為空
    IF (ASSOCIATED(vec%ptr)) CALL assert_fail("Compute Vector should not have host ptr")
    
    IF (vec%size() /= 100) CALL assert_fail("Vector size query failed")
    
    CALL vec%resize(200_8)
    IF (vec%size() /= 200) CALL assert_fail("Vector resize failed")
    
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_lifecycle_vector

  ! ====================================================================
  ! Test 3: Copy (Vector -> Buffer) - 測試 GPU 內部拷貝與下載
  ! ====================================================================
  SUBROUTINE test_copy_vector_to_buffer()
    TYPE(device_vector_i4_t) :: comp_vec
    TYPE(device_vector_i4_t) :: io_buf
    INTEGER :: i
    
    PRINT *, "--- [Test 3] Copy Vector -> Buffer ---"
    
    CALL io_buf%create_buffer(10_8)
    io_buf%ptr(:) = [(i, i=1, 10)]
    CALL io_buf%upload() 
    
    CALL comp_vec%create_vector(10_8)
    CALL comp_vec%copy_from(io_buf) 
    
    io_buf%ptr(:) = 0
    CALL io_buf%upload()
    
    CALL io_buf%copy_from(comp_vec) 
    CALL io_buf%download()          
    
    IF (io_buf%ptr(5) /= 5) THEN
        PRINT *, "Got:", io_buf%ptr(5), " Expected: 5"
        CALL assert_fail("Data copy failed")
    END IF
    
    CALL comp_vec%free()
    CALL io_buf%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_copy_vector_to_buffer

  ! ====================================================================
  ! Test 4: Reductions - 測試 GPU 數學加速介面
  ! ====================================================================
  SUBROUTINE test_reductions()
    TYPE(device_vector_i4_t) :: vec
    INTEGER :: min_val, max_val, s_val
    PRINT *, "--- [Test 4] Reductions ---"
    
    CALL vec%create_buffer(100_8)
    vec%ptr(:) = 10
    CALL vec%upload()
    
    min_val = vec%min() 
    max_val = vec%max()
    s_val   = vec%sum()
    
    IF (min_val /= 10) CALL assert_fail("Min failed")
    IF (max_val /= 10) CALL assert_fail("Max failed")
    IF (s_val /= 1000) CALL assert_fail("Sum failed")
    
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_reductions

  ! ====================================================================
  ! Test 5: Sort - 測試 C++ 呼叫
  ! ====================================================================
  SUBROUTINE test_sort()
    TYPE(device_vector_i4_t) :: k, v, kb, vb
    INTEGER :: i
    PRINT *, "--- [Test 5] Sorting ---"
    
    CALL k%create_buffer(10_8)
    CALL v%create_buffer(10_8)
    CALL kb%create_buffer(10_8)
    CALL vb%create_buffer(10_8)
    
    k%ptr(:) = [(11-i, i=1, 10)] 
    v%ptr(:) = 0
    CALL k%upload(); CALL v%upload()
    
    ! 注意：這裡直接傳 handle 
    CALL vec_sort_i4(k%get_handle(), kb%get_handle(), v%get_handle(), vb%get_handle())
    
    CALL k%download()
    IF (k%ptr(1) /= 1 .OR. k%ptr(10) /= 10) CALL assert_fail("Sort result incorrect")
    
    CALL k%free(); CALL v%free(); CALL kb%free(); CALL vb%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_sort

  ! ====================================================================
  ! Test 6: Double Precision - 測試 R8 通道
  ! ====================================================================
  SUBROUTINE test_double_precision()
    TYPE(device_vector_r8_t) :: vec
    REAL(8) :: val
    
    PRINT *, "--- [Test 6] Double Precision (r8) ---"
    CALL vec%create_buffer(10_8)
    
    vec%ptr(:) = 3.141592653589793_8
    CALL vec%upload()
    
    val = vec%max()
    IF (ABS(val - 3.141592653589793_8) > 1.0e-12) CALL assert_fail("R8 Max accuracy failed")
    
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_double_precision

  SUBROUTINE assert_fail(msg)
    CHARACTER(LEN=*), INTENT(IN) :: msg
    PRINT *, "❌ FAIL: ", msg
    failures = failures + 1
  END SUBROUTINE assert_fail

END PROGRAM Test_Suite