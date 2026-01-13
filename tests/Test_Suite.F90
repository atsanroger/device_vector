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
     STOP 1
  END IF
  PRINT *, "=========================================================="


CONTAINS
  
  ! ====================================================================
  ! Test 1: DeviceBuffer (IO Mode) - Should act like standard host array
  ! ====================================================================
  SUBROUTINE test_lifecycle_buffer()
    TYPE(device_vector_i4_t) :: buf
    PRINT *, "--- [Test 1] DeviceBuffer (Transfer Mode) ---"
    
    ! 1. Create Buffer (Pinned)
    CALL buf%create_buffer(100_8)
    
    ! Check: Host pointer should be valid
    IF (.NOT. ASSOCIATED(buf%ptr)) CALL assert_fail("Buffer host pointer missing")
    IF (SIZE(buf%ptr) /= 100) CALL assert_fail("Buffer size mismatch")
    
    ! 2. Resize Buffer
    CALL buf%resize(200_8)
    IF (SIZE(buf%ptr) /= 200) CALL assert_fail("Buffer resize failed")
    
    ! 3. Free
    CALL buf%free()
    IF (ASSOCIATED(buf%ptr)) CALL assert_fail("Buffer pointer not nullified")
    PRINT *, "   [PASS]"
  END SUBROUTINE test_lifecycle_buffer

  ! ====================================================================
  ! Test 2: DeviceVector (Compute Mode) - Should NOT have host pointer
  ! ====================================================================
  SUBROUTINE test_lifecycle_vector()
    TYPE(device_vector_i4_t) :: vec
    PRINT *, "--- [Test 2] DeviceVector (Compute Mode) ---"
    
    ! 1. Create Vector (Pure Device)
    CALL vec%create_vector(100_8)
    
    ! Check: Host pointer should be NULL (Mode 2 behavior)
    IF (ASSOCIATED(vec%ptr)) CALL assert_fail("Compute Vector should not have host ptr")
    
    ! Check: Size query works
    IF (vec%size() /= 100) CALL assert_fail("Vector size query failed")
    
    ! 2. Resize (GPU only)
    CALL vec%resize(200_8)
    IF (vec%size() /= 200) CALL assert_fail("Vector resize failed")
    
    ! 3. Free
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_lifecycle_vector

  ! ====================================================================
  ! Test 3: Copy (Vector -> Buffer) - The Core Workflow
  ! ====================================================================
  SUBROUTINE test_copy_vector_to_buffer()
    TYPE(device_vector_i4_t) :: comp_vec
    TYPE(device_vector_i4_t) :: io_buf
    INTEGER :: i
    
    PRINT *, "--- [Test 3] Copy Vector -> Buffer ---"
    
    ! 1. Setup Compute Vector (Fake some GPU data)
    ! We use a temp buffer to upload data since we can't write to comp_vec directly from host
    CALL io_buf%create_buffer(10_8)
    io_buf%ptr(:) = [(i, i=1, 10)]
    CALL io_buf%upload() ! Host -> GPU
    
    ! 2. Create Compute Vector and initialize it
    CALL comp_vec%create_vector(10_8)
    CALL comp_vec%copy_from(io_buf) ! GPU -> GPU (Init)
    
    ! 3. Modify Buffer to ensure we are not sharing memory
    io_buf%ptr(:) = 0
    CALL io_buf%upload()
    
    ! 4. The Real Test: Pull data back from Compute Vector
    CALL io_buf%copy_from(comp_vec) ! GPU (Comp) -> GPU (Buffer)
    CALL io_buf%download()          ! GPU (Buffer) -> Host
    
    ! 5. Verify
    IF (io_buf%ptr(5) /= 5) THEN
        PRINT *, "Got:", io_buf%ptr(5), " Expected: 5"
        CALL assert_fail("Data copy failed")
    END IF
    
    CALL comp_vec%free()
    CALL io_buf%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_copy_vector_to_buffer

  ! ====================================================================
  ! Test 4: Reductions (Works on both modes, testing Buffer here)
  ! ====================================================================
  SUBROUTINE test_reductions()
    TYPE(device_vector_i4_t) :: vec
    INTEGER :: min_val, max_val, s_val
    PRINT *, "--- [Test 4] Reductions ---"
    
    CALL vec%create_buffer(100_8)
    vec%ptr(:) = 10
    CALL vec%upload()
    
    min_val = vec%min() ! Using OOP interface
    max_val = vec%max()
    s_val   = vec%sum()
    
    IF (min_val /= 10) CALL assert_fail("Min failed")
    IF (max_val /= 10) CALL assert_fail("Max failed")
    IF (s_val /= 1000) CALL assert_fail("Sum failed")
    
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_reductions

  ! ====================================================================
  ! Test 5: Sort (Using Buffers for IO)
  ! ====================================================================
  SUBROUTINE test_sort()
    TYPE(device_vector_i4_t) :: k, v, kb, vb
    INTEGER :: i
    PRINT *, "--- [Test 5] Sorting ---"
    
    CALL k%create_buffer(10_8)
    CALL v%create_buffer(10_8)
    CALL kb%create_buffer(10_8)
    CALL vb%create_buffer(10_8)
    
    k%ptr(:) = [(11-i, i=1, 10)] ! 10, 9, ... 1
    v%ptr(:) = 0
    CALL k%upload(); CALL v%upload()
    
    ! Sort API takes handles
    CALL vec_sort_i4(k%get_handle(), kb%get_handle(), v%get_handle(), vb%get_handle())
    
    CALL k%download()
    IF (k%ptr(1) /= 1 .OR. k%ptr(10) /= 10) CALL assert_fail("Sort result incorrect")
    
    CALL k%free(); CALL v%free(); CALL kb%free(); CALL vb%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_sort

  ! ====================================================================
  ! Test 6: Double Precision
  ! ====================================================================
  SUBROUTINE test_double_precision()
    TYPE(device_vector_r8_t) :: vec
    REAL(8) :: val
    
    PRINT *, "--- [Test 6] Double Precision (r8) ---"
    CALL vec%create_buffer(10_8)
    
    vec%ptr(:) = 3.14159_8
    CALL vec%upload()
    
    val = vec%max()
    IF (ABS(val - 3.14159_8) > 1.0e-5) CALL assert_fail("R8 Max failed")
    
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_double_precision

  SUBROUTINE assert_fail(msg)
    CHARACTER(LEN=*), INTENT(IN) :: msg
    PRINT *, "❌ FAIL: ", msg
    failures = failures + 1
  END SUBROUTINE assert_fail

END PROGRAM Test_Suite