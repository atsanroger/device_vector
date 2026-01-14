PROGRAM test_reorder_dv_fixed
  USE Device_Vector 
  USE openacc
  USE iso_c_binding
  IMPLICIT NONE

  ! ==========================================
  ! 1. Namelist 與參數設定 (絕對保留)
  ! ==========================================
  INTEGER(8) :: N, GX, GY, GZ, GXYZ, n_cells
  INTEGER    :: ios
  NAMELIST /sim_config/ N, GX, GY, GZ

  ! ==========================================
  ! 2. 變數宣告
  ! ==========================================
  ! 粒子資料
  TYPE(device_vector_r4_t) :: px, py, pz, tx, ty, tz
  REAL(4), POINTER :: ax(:), ay(:), az(:), atx(:), aty(:), atz(:)

  ! 排序 Buffer
  TYPE(device_vector_i4_t) :: dv_codes, dv_codes_buf
  TYPE(device_vector_i4_t) :: dv_ids,   dv_ids_buf
  INTEGER, POINTER :: codes(:), ids(:) 

  ! 網格 (Edge Finding 用)
  TYPE(device_vector_i4_t) :: dv_cell_ptr, dv_cell_cnt
  INTEGER, POINTER :: cell_ptr(:), cell_cnt(:)

  ! 驗證用 Host 變數
  INTEGER, ALLOCATABLE :: h_codes(:)
  INTEGER, ALLOCATABLE :: h_cell_cnt(:) 
  INTEGER(8) :: i, total_particles
  INTEGER    :: non_empty_cells 
  LOGICAL    :: is_sorted

  ! ★★★ 計時變數 (Profiler) ★★★
  INTEGER(8) :: t1, t2, t_rate
  REAL(8)    :: t_morton, t_sort, t_gather, t_edge, t_total

  CALL device_env_init(0, 1)

  ! ==========================================
  ! Step 0: 讀取 Namelist
  ! ==========================================
  ! 設定預設值 (萬一讀不到檔)
  N = 1000000_8; GX = 128; GY = 128; GZ = 128
  
  OPEN(UNIT=10, FILE='../configs/test_reorder.nml', STATUS='OLD', IOSTAT=ios)
  IF (ios == 0) THEN
     READ(10, NML=sim_config)
     CLOSE(10)
     PRINT *, "[Init] Namelist Loaded: N =", N, " Grid =", GX, "x", GY, "x", GZ
  ELSE
     PRINT *, "[Init] Namelist not found, using defaults: N =", N
  END IF
  
  GXYZ = int(GX,8) * int(GY,8) * int(GZ,8)
  n_cells = GXYZ

  PRINT *, "[Init] Creating DeviceVectors..."
  
  ! 建立所有 Buffer
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

  ! 初始化資料
  !$acc parallel loop present(ax, ay, az)
  DO i = 1, N
     ax(i) = REAL(MOD(i * 17, GX), 4) + 0.5
     ay(i) = REAL(MOD(i * 31, GY), 4) + 0.5
     az(i) = REAL(MOD(i * 13, GZ), 4) + 0.5
  END DO
  
  ! 預熱並取得時鐘頻率
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(count_rate=t_rate)
  PRINT *, "[Run] Starting Profiler..."

  ! =========================================================
  ! Step 1: Calculating Morton Codes
  ! =========================================================
  CALL SYSTEM_CLOCK(t1)
  
  !$acc parallel loop present(ax, ay, az, codes, ids)
  DO i = 1, N
     ids(i) = INT(i, 4) 
     codes(i) = get_morton_code(INT(ax(i),4), INT(ay(i),4), INT(az(i),4))
  END DO
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t2)
  t_morton = REAL(t2 - t1, 8) / REAL(t_rate, 8)
  PRINT *, "[Step 1] Morton Done."

  ! =========================================================
  ! Step 2: Sorting
  ! =========================================================
  CALL SYSTEM_CLOCK(t1)

  CALL dv_codes%acc_unmap(); CALL dv_ids%acc_unmap()
  CALL vec_sort_i4(dv_codes%get_handle(), dv_codes_buf%get_handle(), &
                   dv_ids%get_handle(),   dv_ids_buf%get_handle())
  CALL dv_codes%acc_map(codes); CALL dv_ids%acc_map(ids)
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t2)
  t_sort = REAL(t2 - t1, 8) / REAL(t_rate, 8)
  PRINT *, "[Step 2] Sort Done."

  ! =========================================================
  ! Step 3: Gathering
  ! =========================================================
  CALL SYSTEM_CLOCK(t1)

  CALL gather_particles(N, ids, ax, atx)
  CALL gather_particles(N, ids, ay, aty)
  CALL gather_particles(N, ids, az, atz)

  ! 更新回 ax
  !$acc parallel loop present(ax, ay, az, atx, aty, atz)
  DO i = 1, N
     ax(i) = atx(i); ay(i) = aty(i); az(i) = atz(i)
  END DO

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t2)
  t_gather = REAL(t2 - t1, 8) / REAL(t_rate, 8)
  PRINT *, "[Step 3] Gather Done."

  ! 驗證 Sort (保留驗證邏輯)
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
  IF (is_sorted) PRINT *, "✅ PASS: GPU Sort & Reorder 成功！"

  ! =========================================================
  ! Step 4: Edge Finding
  ! =========================================================
  CALL SYSTEM_CLOCK(t1)

  CALL find_cell_edges(N, codes, cell_ptr, cell_cnt, n_cells)

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t2)
  t_edge = REAL(t2 - t1, 8) / REAL(t_rate, 8)
  PRINT *, "[Step 4] Edge Finding Done."

  ! =========================================================
  ! 報告與驗證
  ! =========================================================
  t_total = t_morton + t_sort + t_gather + t_edge

  PRINT *, " "
  PRINT *, "================== PROFILING REPORT =================="
  PRINT '(A, F10.6, A)', " [1] Morton Code:  ", t_morton, " s"
  PRINT '(A, F10.6, A)', " [2] Sort (CUB):   ", t_sort,   " s"
  PRINT '(A, F10.6, A)', " [3] Gather:       ", t_gather, " s"
  PRINT '(A, F10.6, A)', " [4] Edge Finding: ", t_edge,   " s"
  PRINT *, "------------------------------------------------------"
  PRINT '(A, F10.6, A)', " TOTAL GPU TIME:   ", t_total,  " s"
  PRINT '(A, F10.2, A)', " Throughput:       ", REAL(N)/t_total/1e6, " M/s"
  PRINT *, "======================================================"

  ALLOCATE(h_cell_cnt(n_cells))
  !$acc update host(cell_cnt)
  h_cell_cnt = cell_cnt(1:n_cells)

  ! 統計
  non_empty_cells = 0
  total_particles = 0
  DO i = 1, n_cells
     IF (h_cell_cnt(i) > 0) THEN
        non_empty_cells = non_empty_cells + 1
        total_particles = total_particles + h_cell_cnt(i)
     END IF
  END DO

  PRINT *, "Edge Finding Validation:"
  PRINT '(A, I10)', "  Total Grid Cells: ", n_cells
  PRINT '(A, I10)', "  Non-Empty Cells:  ", non_empty_cells
  PRINT '(A, I10)', "  Total Particles:  ", total_particles

  IF (total_particles == N) THEN
     PRINT *, "✅ PASS: 所有粒子都被正確歸入網格！ (Count sum == N)"
  ELSE
     PRINT *, "❌ FAIL: 粒子總數不符！ Edge Finding 有問題。"
     PRINT *, "Expected: ", N, ", Got: ", total_particles
  END IF

  ! 清理
  CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
  CALL tx%acc_unmap(); CALL ty%acc_unmap(); CALL tz%acc_unmap()
  CALL dv_codes%acc_unmap(); CALL dv_ids%acc_unmap()
  CALL dv_cell_ptr%acc_unmap(); CALL dv_cell_cnt%acc_unmap()
  
  CALL px%free(); CALL py%free(); CALL pz%free()
  CALL tx%free(); CALL ty%free(); CALL tz%free()
  CALL dv_codes%free(); CALL dv_codes_buf%free()
  CALL dv_ids%free();   CALL dv_ids_buf%free()
  CALL dv_cell_ptr%free(); CALL dv_cell_cnt%free()
  
  IF (ALLOCATED(h_codes)) DEALLOCATE(h_codes)
  IF (ALLOCATED(h_cell_cnt)) DEALLOCATE(h_cell_cnt)
  
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

  ! Two-Pass Edge Finding
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

    ! Pass 1: Find Start
    !$acc parallel loop gang vector present(sorted_codes, cell_ptr)
    DO i = 1, n_pts
       code = sorted_codes(i) + 1 
       prev_code = -1
       IF (i > 1) prev_code = sorted_codes(i-1) + 1
       IF (code /= prev_code) THEN
          IF (code <= n_cells) cell_ptr(code) = INT(i, 4)
       END IF
    END DO

    ! Pass 2: Find End & Count
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