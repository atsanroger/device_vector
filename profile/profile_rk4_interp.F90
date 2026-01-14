PROGRAM test_reorder_dv_fixed
  USE Device_Vector 
  USE openacc
  USE iso_c_binding
  IMPLICIT NONE

  ! 參數設定
  INTEGER(8), PARAMETER :: N = 100_8 
  INTEGER,    PARAMETER :: GX=1000, GY=1000, GZ=1000
  INTEGER(8) :: n_cells

  ! 1. 粒子資料
  TYPE(device_vector_r4_t) :: px, py, pz, tx, ty, tz
  REAL(4), POINTER :: ax(:), ay(:), az(:), atx(:), aty(:), atz(:)

  ! 2. 排序 Buffer
  TYPE(device_vector_i4_t) :: dv_codes, dv_codes_buf
  TYPE(device_vector_i4_t) :: dv_ids,   dv_ids_buf
  INTEGER, POINTER :: codes(:), ids(:) 

  ! 3. 網格 (Edge Finding 用)
  TYPE(device_vector_i4_t) :: dv_cell_ptr, dv_cell_cnt
  INTEGER, POINTER :: cell_ptr(:), cell_cnt(:)

  ! 4. 驗證與計時變數 ★★★
  INTEGER, ALLOCATABLE :: h_codes(:), h_cell_cnt(:)
  INTEGER(8) :: i, total_particles
  INTEGER    :: non_empty_cells
  LOGICAL    :: is_sorted
  
  INTEGER(8) :: t1, t2, t_rate
  REAL(8)    :: t_total

  CALL device_env_init(0, 1)
  PRINT *, "[Init] Creating DeviceVectors..."

  n_cells = int(GX,8) * int(GY,8) * int(GZ,8)
  
  ! 建立 Buffer
  CALL dv_cell_ptr%create_buffer(n_cells)
  CALL dv_cell_cnt%create_buffer(n_cells)
  CALL dv_cell_ptr%acc_map(cell_ptr)
  CALL dv_cell_cnt%acc_map(cell_cnt)

  CALL px%create_buffer(N); CALL py%create_buffer(N); CALL pz%create_buffer(N)
  CALL tx%create_buffer(N); CALL ty%create_buffer(N); CALL tz%create_buffer(N)
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
     ax(i) = REAL(MOD(i * 17, 100), 4) + 0.5
     ay(i) = REAL(MOD(i * 31, 100), 4) + 0.5
     az(i) = REAL(MOD(i * 13, 100), 4) + 0.5
  END DO
  
  ! 預熱 GPU 並同步，確保計時準確
  CALL device_synchronize()

  ! ★★★ 計時開始 ★★★
  PRINT *, "[Run] Starting Full Reorder Timer..."
  CALL SYSTEM_CLOCK(t1, t_rate)

  ! --- Step 1: Morton Codes ---
  !$acc parallel loop present(ax, ay, az, codes, ids)
  DO i = 1, N
     ids(i) = INT(i, 4) 
     codes(i) = get_morton_code(INT(ax(i),4), INT(ay(i),4), INT(az(i),4))
  END DO

  ! --- Step 2: GPU Sort ---
  CALL dv_codes%acc_unmap(); CALL dv_ids%acc_unmap()
  CALL vec_sort_i4(dv_codes%get_handle(), dv_codes_buf%get_handle(), &
                   dv_ids%get_handle(),   dv_ids_buf%get_handle())
  CALL dv_codes%acc_map(codes); CALL dv_ids%acc_map(ids)

  ! --- Step 3: Gather ---
  CALL gather_particles(N, ids, ax, atx)
  CALL gather_particles(N, ids, ay, aty)
  CALL gather_particles(N, ids, az, atz)
  ! 更新回 ax
  !$acc parallel loop present(ax, ay, az, atx, aty, atz)
  DO i = 1, N
     ax(i) = atx(i); ay(i) = aty(i); az(i) = atz(i)
  END DO

  ! --- Step 4: Edge Finding ---
  CALL find_cell_edges(N, codes, cell_ptr, cell_cnt, n_cells)

  ! 等待 GPU 完成所有工作
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t2)
  ! ★★★ 計時結束 ★★★

  t_total = REAL(t2 - t1, 8) / REAL(t_rate, 8)
  PRINT *, "-------------------------------------------"
  PRINT '(A, F10.6, A)', " [Total Reorder] Time: ", t_total, " s"
  PRINT *, "-------------------------------------------"

  ! =========================================================
  ! 驗證邏輯
  ! =========================================================
  ALLOCATE(h_codes(N))
  !$acc update host(codes)
  h_codes = codes(1:N)
  is_sorted = .TRUE.
  DO i = 1, N-1
     IF (h_codes(i) > h_codes(i+1)) THEN
        PRINT *, ">>> Sort ERROR at ", i
        is_sorted = .FALSE.
        EXIT
     END IF
  END DO
  IF (is_sorted) PRINT *, "✅ 1. Sort Accuracy: PASS"

  ALLOCATE(h_cell_cnt(n_cells))
  !$acc update host(cell_cnt)
  h_cell_cnt = cell_cnt(1:n_cells)
  non_empty_cells = 0
  total_particles = 0
  DO i = 1, n_cells
     IF (h_cell_cnt(i) > 0) THEN
        non_empty_cells = non_empty_cells + 1
        total_particles = total_particles + h_cell_cnt(i)
     END IF
  END DO

  IF (total_particles == N) THEN
     PRINT *, "✅ 2. Particle Conservation: PASS (Sum == N)"
  ELSE
     PRINT *, "❌ 2. Particle Conservation: FAIL!"
  END IF
  PRINT '(A, I10)', "    Non-Empty Cells: ", non_empty_cells

  ! 清理 (同前，省略以保持簡潔，請記得加回你的程式碼中)
  ! ... (Cleanup 區塊) ...
  CALL device_env_finalize()

CONTAINS

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

  SUBROUTINE find_cell_edges(n_pts, sorted_codes, cell_ptr, cell_cnt, n_cells)
    INTEGER(8), VALUE :: n_pts, n_cells
    INTEGER(4), INTENT(IN) :: sorted_codes(*)
    INTEGER(4), INTENT(OUT):: cell_ptr(*), cell_cnt(*)
    INTEGER(8) :: i
    INTEGER(4) :: code, prev_code, next_code, start_idx

    !$acc parallel loop present(cell_ptr, cell_cnt)
    DO i = 1, n_cells
       cell_ptr(i) = 0; cell_cnt(i) = 0
    END DO

    ! Pass 1: Find Start (Edge Finding) [cite: 262]
    !$acc parallel loop gang vector present(sorted_codes, cell_ptr)
    DO i = 1, n_pts
       code = sorted_codes(i) + 1 
       prev_code = -1
       IF (i > 1) prev_code = sorted_codes(i-1) + 1
       IF (code /= prev_code) THEN
          IF (code <= n_cells) cell_ptr(code) = INT(i, 4)
       END IF
    END DO

    ! Pass 2: Find End & Calculate Count (Counting) [cite: 263]
    !$acc parallel loop gang vector present(sorted_codes, cell_ptr, cell_cnt)
    DO i = 1, n_pts
       code = sorted_codes(i) + 1
       next_code = -1
       IF (i < n_pts) next_code = sorted_codes(i+1) + 1
       IF (code /= next_code) THEN
          IF (code <= n_cells) THEN
             start_idx = cell_ptr(code)
             cell_cnt(code) = INT(i, 4) - start_idx + 1
          END IF
       END IF
    END DO
  END SUBROUTINE find_cell_edges

END PROGRAM