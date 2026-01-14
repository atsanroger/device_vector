PROGRAM test_reorder_dv_fixed
  USE Device_Vector 
  USE openacc
  USE iso_c_binding
  IMPLICIT NONE

  ! 確保 N 是 INTEGER(8) 以匹配 create_buffer
  INTEGER(8), PARAMETER :: N = 100_8 
  INTEGER,    PARAMETER :: GX=1000, GY=1000, GZ=1000

  ! 1. 粒子資料 (Real4)
  TYPE(device_vector_r4_t) :: px, py, pz     ! 原始
  TYPE(device_vector_r4_t) :: tx, ty, tz     ! 暫存
  REAL(4), POINTER :: ax(:), ay(:), az(:)
  REAL(4), POINTER :: atx(:), aty(:), atz(:)

  ! 2. 排序 Buffer (Key/Value + Double Buffering)
  TYPE(device_vector_i4_t) :: dv_codes, dv_codes_buf
  TYPE(device_vector_i4_t) :: dv_ids,   dv_ids_buf
  INTEGER, POINTER :: codes(:), ids(:) 

  ! 3. 驗證用 Host
  INTEGER, ALLOCATABLE :: h_codes(:)
  INTEGER :: i
  LOGICAL :: is_sorted

  CALL device_env_init(0, 1)
  PRINT *, "[Init] Creating DeviceVectors..."

  ! 建立粒子 Buffer (注意：傳入 N，它是 INT8)
  CALL px%create_buffer(N); CALL py%create_buffer(N); CALL pz%create_buffer(N)
  CALL tx%create_buffer(N); CALL ty%create_buffer(N); CALL tz%create_buffer(N)

  ! 建立排序 Buffer
  CALL dv_codes%create_buffer(N); CALL dv_codes_buf%create_buffer(N)
  CALL dv_ids%create_buffer(N);   CALL dv_ids_buf%create_buffer(N)

  ! Map
  CALL px%acc_map(ax); CALL py%acc_map(ay); CALL pz%acc_map(az)
  CALL tx%acc_map(atx); CALL ty%acc_map(aty); CALL tz%acc_map(atz)
  CALL dv_codes%acc_map(codes)
  CALL dv_ids%acc_map(ids)

  ! 1. 初始化資料
  !$acc parallel loop present(ax, ay, az)
  DO i = 1, N
     ax(i) = REAL(MOD(i * 17, 100), 4) 
     ay(i) = REAL(MOD(i * 31, 100), 4)
     az(i) = REAL(MOD(i * 13, 100), 4)
  END DO

  PRINT *, "[Step 1] Calculating Morton Codes..."
  !$acc parallel loop present(ax, ay, az, codes, ids)
  DO i = 1, N
     ids(i) = INT(i, 4) 
     codes(i) = get_morton_code(INT(ax(i),4), INT(ay(i),4), INT(az(i),4))
  END DO
  !$acc wait

  ! =========================================================
  ! Step 2: GPU Sort (使用 Handle)
  ! =========================================================
  PRINT *, "[Step 2] Sorting..."
  
  ! 解除 Map 以便 C++ 使用
  CALL dv_codes%acc_unmap() 
  CALL dv_ids%acc_unmap()
  
  ! 呼叫 C++ 排序
  CALL vec_sort_i4(dv_codes%get_handle(), dv_codes_buf%get_handle(), &
                   dv_ids%get_handle(),   dv_ids_buf%get_handle())

  ! 重新 Map 回來
  CALL dv_codes%acc_map(codes)
  CALL dv_ids%acc_map(ids)

  ! =========================================================
  ! Step 3: Gather (重排)
  ! =========================================================
  PRINT *, "[Step 3] Gathering..."
  
  CALL gather_particles(N, ids, ax, atx)
  CALL gather_particles(N, ids, ay, aty)
  CALL gather_particles(N, ids, az, atz)

  ! 更新回 ax
  !$acc parallel loop present(ax, ay, az, atx, aty, atz)
  DO i = 1, N
     ax(i) = atx(i)
     ay(i) = aty(i)
     az(i) = atz(i)
  END DO

  ! =========================================================
  ! 驗證
  ! =========================================================
  ALLOCATE(h_codes(N))
  !$acc update host(codes)
  h_codes = codes(1:N)

  PRINT *, " "
  PRINT *, "Index | Sorted Code"
  DO i = 1, 10
     WRITE(*,*) i, h_codes(i)
  END DO

  is_sorted = .TRUE.
  DO i = 1, N-1
     IF (h_codes(i) > h_codes(i+1)) THEN
        PRINT *, ">>> ERROR: ", h_codes(i), " > ", h_codes(i+1)
        is_sorted = .FALSE.
        EXIT
     END IF
  END DO

  IF (is_sorted) PRINT *, "✅ PASS: GPU Sort & Reorder 成功！"

  ! 清理
  CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
  CALL tx%acc_unmap(); CALL ty%acc_unmap(); CALL tz%acc_unmap()
  CALL dv_codes%acc_unmap(); CALL dv_ids%acc_unmap()
  
  CALL px%free(); CALL py%free(); CALL pz%free()
  CALL tx%free(); CALL ty%free(); CALL tz%free()
  CALL dv_codes%free(); CALL dv_codes_buf%free()
  CALL dv_ids%free();   CALL dv_ids_buf%free()
  
  CALL device_env_finalize()

CONTAINS

  ! ---------------------------------------------------------
  ! Device Function: 計算 Morton Code
  ! ---------------------------------------------------------
  !$acc routine seq
  FUNCTION get_morton_code(ix, iy, iz) RESULT(code)
    INTEGER(4), INTENT(IN) :: ix, iy, iz
    INTEGER(4) :: code
    INTEGER(4) :: x, y, z
    INTEGER :: i

    x = ix; y = iy; z = iz
    code = 0
    DO i = 0, 9
       IF (BTEST(x, i)) code = IBSET(code, 3*i)
       IF (BTEST(y, i)) code = IBSET(code, 3*i + 1)
       IF (BTEST(z, i)) code = IBSET(code, 3*i + 2)
    END DO
  END FUNCTION get_morton_code

  ! ---------------------------------------------------------
  ! Kernel: Gather
  ! ---------------------------------------------------------
  SUBROUTINE gather_particles(n_pts, ids_in, src, dst)
    INTEGER(8), VALUE :: n_pts
    INTEGER(4), INTENT(IN) :: ids_in(*) 
    REAL(4),    INTENT(IN) :: src(*) 
    REAL(4),    INTENT(OUT):: dst(*) 
    
    INTEGER(8) :: i_loop
    INTEGER(4) :: old_idx

    !$acc parallel loop gang vector present(ids_in, src, dst)
    DO i_loop = 1, n_pts
       old_idx = ids_in(i_loop)
       dst(i_loop) = src(old_idx)
    END DO
  END SUBROUTINE gather_particles

END PROGRAM