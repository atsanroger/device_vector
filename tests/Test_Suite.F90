PROGRAM Test_Suite
  USE Device_Vector
  USE iso_c_binding
  IMPLICIT NONE

  INTEGER :: failures = 0

  PRINT *, "=========================================================="
  PRINT *, "🚀 STARTING GPU FORTRAN TEST SUITE"
  PRINT *, "=========================================================="

  ! 👇 [修正] 初始化環境 (Rank=0, GPUs_per_node=1)
  CALL device_env_init(0, 1)

  CALL test_lifecycle()
  CALL test_resize_safety()
  CALL test_reductions()
  CALL test_sort()
  CALL test_double_precision()
  CALL test_shrink_and_fill()
  CALL test_empty_vector()

  ! 👇 [修正] 釋放環境
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
  
  SUBROUTINE test_lifecycle()
    TYPE(device_vector_i4_t) :: vec
    PRINT *, "--- [Test 1] Create & Free ---"
    CALL vec%create(100_8)
    IF (SIZE(vec%data) /= 100) CALL assert_fail("Size mismatch")
    CALL vec%free()
    IF (ASSOCIATED(vec%data)) CALL assert_fail("Pointer not nullified")
    PRINT *, "   [PASS]"
  END SUBROUTINE test_lifecycle

  SUBROUTINE test_resize_safety()
    TYPE(device_vector_i4_t) :: vec
    INTEGER :: i
    PRINT *, "--- [Test 2] Resize & Pointer Sync ---"
    CALL vec%create(10_8)
    vec%data(:) = [(i, i=1, 10)]
    CALL vec%upload()
    CALL vec%resize(20_8)
    IF (SIZE(vec%data) /= 20) CALL assert_fail("Resize failed to update size")
    CALL vec%download()
    IF (vec%data(5) /= 5) CALL assert_fail("Old data lost after resize")
    vec%data(15) = 999
    CALL vec%upload()
    vec%data(:) = 0
    CALL vec%download()
    IF (vec%data(15) /= 999) CALL assert_fail("Pointer not synced to new memory")
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_resize_safety

  SUBROUTINE test_reductions()
    TYPE(device_vector_i4_t) :: vec
    INTEGER :: min_val, max_val
    PRINT *, "--- [Test 3] Reductions (Padding Check) ---"
    CALL vec%create(100_8)
    vec%data(:) = 10
    CALL vec%upload()
    min_val = vec_min_i4(vec%get_handle())
    max_val = vec_max_i4(vec%get_handle())
    IF (min_val /= 10) THEN 
        PRINT *, "Got Min:", min_val, " Expected: 10"
        CALL assert_fail("Min reduction affected by padding")
    END IF
    IF (max_val /= 10) CALL assert_fail("Max reduction failed")
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_reductions

  SUBROUTINE test_sort()
    TYPE(device_vector_i4_t) :: k, v, kb, vb
    INTEGER :: i
    PRINT *, "--- [Test 4] Sorting ---"
    CALL k%create(10_8); CALL v%create(10_8)
    CALL kb%create(10_8); CALL vb%create(10_8)
    k%data(:) = [(11-i, i=1, 10)]
    v%data(:) = 0
    CALL k%upload(); CALL v%upload()
    CALL vec_sort_i4(k%get_handle(), kb%get_handle(), v%get_handle(), vb%get_handle())
    CALL k%download()
    IF (k%data(1) /= 1 .OR. k%data(10) /= 10) CALL assert_fail("Sort result incorrect")
    CALL k%free(); CALL v%free(); CALL kb%free(); CALL vb%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_sort

  SUBROUTINE assert_fail(msg)
    CHARACTER(LEN=*), INTENT(IN) :: msg
    PRINT *, "❌ FAIL: ", msg
    failures = failures + 1
  END SUBROUTINE assert_fail

SUBROUTINE test_double_precision()
    TYPE(device_vector_r8_t) :: vec
    REAL(8) :: val
    
    PRINT *, "--- [Test 5] Double Precision (r8) ---"
    CALL vec%create(10_8)
    
    ! 測試浮點數運算
    vec%data(:) = 3.14159_8
    CALL vec%upload()
    
    val = vec_max_r8(vec%get_handle())
    
    IF (ABS(val - 3.14159_8) > 1.0e-5) CALL assert_fail("R8 Max failed")
    
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_double_precision

SUBROUTINE test_shrink_and_fill()
    TYPE(device_vector_i4_t) :: vec
    INTEGER :: s
    
    PRINT *, "--- [Test 6] Shrink & Fill Zero ---"
    CALL vec%create(100_8)
    vec%data(:) = 99
    CALL vec%upload()
    
    ! 1. 縮小測試 (100 -> 5)
    CALL vec%resize(5_8)
    IF (SIZE(vec%data) /= 5) CALL assert_fail("Shrink size incorrect")
    
    CALL vec%download()
    IF (vec%data(1) /= 99) CALL assert_fail("Data lost after shrink")
    
    ! 2. 歸零測試
    CALL vec%fill_zero()
    CALL vec%download()
    
    s = SUM(vec%data)
    IF (s /= 0) CALL assert_fail("Fill zero failed")
    
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_shrink_and_fill

SUBROUTINE test_empty_vector()
    TYPE(device_vector_i4_t) :: vec
    
    PRINT *, "--- [Test 7] Empty Vector Safety ---"
    ! 故意建立 0 大小，看會不會崩潰
    CALL vec%create(0_8) 
    
    IF (ASSOCIATED(vec%data)) THEN
        IF (SIZE(vec%data) /= 0) CALL assert_fail("Size 0 create failed")
    END IF
    
    ! 測試對空陣列 resize
    CALL vec%resize(10_8)
    IF (SIZE(vec%data) /= 10) CALL assert_fail("Resize from 0 failed")
    
    CALL vec%free()
    PRINT *, "   [PASS]"
  END SUBROUTINE test_empty_vector

END PROGRAM Test_Suite