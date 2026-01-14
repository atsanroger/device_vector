MODULE pure_openacc_sort_mod
  USE openacc
  IMPLICIT NONE
CONTAINS

  ! =========================================================
  ! 純 OpenACC 手寫 Radix Sort (Binary LSB)
  ! 原理：
  !   1. 針對每個 Bit (0~29) 做一次分群。
  !   2. GPU 算出每個元素的 Bit (0或1)。
  !   3. CPU 幫忙算前綴和 (Scan) 決定搬家位置 (因為 GPU 純手寫 Scan 太難)。
  !   4. GPU 根據位置搬家 (Scatter)。
  ! =========================================================
  SUBROUTINE openacc_radix_sort_i4(n, keys, vals, keys_buf, vals_buf)
    INTEGER(8), VALUE :: n
    INTEGER(4), INTENT(INOUT) :: keys(:), vals(:)
    INTEGER(4), INTENT(INOUT) :: keys_buf(:), vals_buf(:)
    
    ! 內部暫存
    INTEGER(4), ALLOCATABLE :: bits(:), offsets(:)
    INTEGER(4), ALLOCATABLE :: h_bits(:), h_offsets(:)
    INTEGER :: bit, i, zeros, total_zeros
    
    ! 分配記憶體
    ALLOCATE(bits(n), offsets(n))
    ALLOCATE(h_bits(n), h_offsets(n))
    
    ! 上傳到 GPU
    !$acc enter data create(bits, offsets)

    ! Radix Sort: 處理 30 個 bits (Morton Code 範圍)
    DO bit = 0, 29
       
       ! [GPU] Step 1: 提取當前 Bit (0 或 1)
       !$acc parallel loop present(keys, bits)
       DO i = 1, n
          bits(i) = IAND(ISHFT(keys(i), -bit), 1)
       END DO
       
       ! [CPU Helper] Step 2: 下載 Bit 資訊並計算 Offset (Scan)
       ! (這是 Pure OpenACC 最痛的點：手寫 GPU Scan 太難，只好借過 CPU)
       !$acc update host(bits)
       
       h_bits = bits(1:n)
       
       ! CPU Serial Scan (計算 0 的位置)
       zeros = 0
       DO i = 1, n
          IF (h_bits(i) == 0) THEN
             zeros = zeros + 1
             h_offsets(i) = zeros ! 0 的目標位置
          ELSE
             h_offsets(i) = 0     ! 1 暫時不管
          END IF
       END DO
       total_zeros = zeros
       
       ! CPU Serial Scan (計算 1 的位置)
       ! 1 的位置 = 總 0 數 + 當前 1 的序號
       DO i = 1, n
          IF (h_bits(i) == 1) THEN
             zeros = zeros + 1
             h_offsets(i) = zeros
          END IF
       END DO
       
       ! [GPU] Step 3: 上傳 Offset 並重排 (Scatter)
       offsets(1:n) = h_offsets(1:n)
       !$acc update device(offsets)
       
       ! 搬家：根據 Offset 把資料搬到 Buffer
       !$acc parallel loop present(keys, vals, keys_buf, vals_buf, bits, offsets)
       DO i = 1, n
          keys_buf(offsets(i)) = keys(i)
          vals_buf(offsets(i)) = vals(i)
       END DO
       
       ! 寫回原陣列 (Ping-Pong)
       !$acc parallel loop present(keys, vals, keys_buf, vals_buf)
       DO i = 1, n
          keys(i) = keys_buf(i)
          vals(i) = vals_buf(i)
       END DO
       
    END DO

    ! 清理
    !$acc exit data delete(bits, offsets)
    DEALLOCATE(bits, offsets, h_bits, h_offsets)
    
  END SUBROUTINE openacc_radix_sort_i4
END MODULE pure_openacc_sort_mod


PROGRAM test_pure_vs_oop
  USE Device_Vector 
  USE pure_openacc_sort_mod
  USE openacc
  IMPLICIT NONE

  ! 1. Namelist
  INTEGER(8) :: N, GX, GY, GZ, n_cells
  INTEGER    :: ios, alloc_stat
  NAMELIST /sim_config/ N, GX, GY, GZ

  ! 2. 變數
  INTEGER(8) :: t1, t2, t_rate
  REAL(8)    :: t_dv, t_acc 

  ! OOP 變數
  TYPE(device_vector_r4_t) :: dv_px, dv_py, dv_pz, dv_tx, dv_ty, dv_tz
  TYPE(device_vector_i4_t) :: dv_codes, dv_codes_buf, dv_ids, dv_ids_buf
  TYPE(device_vector_i4_t) :: dv_cell_ptr, dv_cell_cnt
  REAL(4), POINTER :: ptr_ax(:), ptr_ay(:), ptr_az(:), ptr_atx(:), ptr_aty(:), ptr_atz(:)
  INTEGER, POINTER :: ptr_codes(:), ptr_ids(:), ptr_cell_ptr(:), ptr_cell_cnt(:)

  ! Raw 變數
  REAL(4),    ALLOCATABLE, TARGET :: raw_ax(:), raw_ay(:), raw_az(:)
  REAL(4),    ALLOCATABLE, TARGET :: raw_tx(:), raw_ty(:), raw_tz(:)
  INTEGER(4), ALLOCATABLE, TARGET :: raw_codes(:), raw_codes_buf(:)
  INTEGER(4), ALLOCATABLE, TARGET :: raw_ids(:),   raw_ids_buf(:)
  INTEGER(4), ALLOCATABLE, TARGET :: raw_cell_ptr(:), raw_cell_cnt(:)
  INTEGER(8) :: i

  CALL device_env_init(0, 1)

  ! 讀取設定 (預設 N=100萬，純手寫 Sort 很慢，別跑太大)
  N = 1000000_8; GX = 128; GY = 128; GZ = 128
  OPEN(UNIT=10, FILE='../configs/test_reorder.nml', STATUS='OLD', IOSTAT=ios)
  IF (ios == 0) THEN
     READ(10, NML=sim_config)
     CLOSE(10)
     PRINT *, "[Init] Namelist Loaded: N =", N
  ELSE
     PRINT *, "[Init] Using Default N =", N
  END IF
  n_cells = int(GX,8) * int(GY,8) * int(GZ,8)

  ! =================================================================
  ! ROUND 1: Device Vector (CUB Sort)
  ! =================================================================
  PRINT *, " "
  PRINT *, ">>> ROUND 1: Device Vector (CUB Sort) <<<"
  
  CALL dv_px%create_buffer(N); CALL dv_py%create_buffer(N); CALL dv_pz%create_buffer(N)
  CALL dv_tx%create_buffer(N); CALL dv_ty%create_buffer(N); CALL dv_tz%create_buffer(N)
  CALL dv_codes%create_buffer(N); CALL dv_codes_buf%create_buffer(N)
  CALL dv_ids%create_buffer(N);   CALL dv_ids_buf%create_buffer(N)
  CALL dv_cell_ptr%create_buffer(n_cells); CALL dv_cell_cnt%create_buffer(n_cells)

  CALL dv_px%acc_map(ptr_ax); CALL dv_py%acc_map(ptr_ay); CALL dv_pz%acc_map(ptr_az)
  CALL dv_tx%acc_map(ptr_atx); CALL dv_ty%acc_map(ptr_aty); CALL dv_tz%acc_map(ptr_atz)
  CALL dv_codes%acc_map(ptr_codes); CALL dv_ids%acc_map(ptr_ids)
  CALL dv_cell_ptr%acc_map(ptr_cell_ptr); CALL dv_cell_cnt%acc_map(ptr_cell_cnt)

  !$acc parallel loop present(ptr_ax, ptr_ay, ptr_az)
  DO i = 1, N
     ptr_ax(i) = REAL(MOD(i * 17, GX), 4) + 0.5
     ptr_ay(i) = REAL(MOD(i * 31, GY), 4) + 0.5
     ptr_az(i) = REAL(MOD(i * 13, GZ), 4) + 0.5
  END DO
  CALL device_synchronize()

  CALL SYSTEM_CLOCK(t1, t_rate)

  ! (A) Morton
  !$acc parallel loop present(ptr_ax, ptr_ay, ptr_az, ptr_codes, ptr_ids)
  DO i = 1, N
     ptr_ids(i) = INT(i, 4) 
     ptr_codes(i) = get_morton_code(INT(ptr_ax(i),4), INT(ptr_ay(i),4), INT(ptr_az(i),4))
  END DO

  ! (B) Sort (OOP -> Thrust)
  CALL dv_codes%acc_unmap(); CALL dv_ids%acc_unmap()
  CALL vec_sort_i4(dv_codes%get_handle(), dv_codes_buf%get_handle(), &
                   dv_ids%get_handle(),   dv_ids_buf%get_handle())
  CALL dv_codes%acc_map(ptr_codes); CALL dv_ids%acc_map(ptr_ids)

  ! (C) Gather
  CALL gather_particles(N, ptr_ids, ptr_ax, ptr_atx)
  CALL gather_particles(N, ptr_ids, ptr_ay, ptr_aty)
  CALL gather_particles(N, ptr_ids, ptr_az, ptr_atz)

  ! (D) Edge Finding
  CALL find_cell_edges(N, ptr_codes, ptr_cell_ptr, ptr_cell_cnt, n_cells)
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t2)
  t_dv = REAL(t2 - t1, 8) / REAL(t_rate, 8)
  
  CALL dv_px%acc_unmap(); CALL dv_py%acc_unmap(); CALL dv_pz%acc_unmap()
  CALL dv_tx%acc_unmap(); CALL dv_ty%acc_unmap(); CALL dv_tz%acc_unmap()
  CALL dv_codes%acc_unmap(); CALL dv_codes_buf%acc_unmap()
  CALL dv_ids%acc_unmap();   CALL dv_ids_buf%acc_unmap()
  CALL dv_cell_ptr%acc_unmap(); CALL dv_cell_cnt%acc_unmap()

  CALL dv_px%free(); CALL dv_py%free(); CALL dv_pz%free()
  CALL dv_tx%free(); CALL dv_ty%free(); CALL dv_tz%free()
  CALL dv_codes%free(); CALL dv_codes_buf%free()
  CALL dv_ids%free();   CALL dv_ids_buf%free()
  CALL dv_cell_ptr%free(); CALL dv_cell_cnt%free()



  ! =================================================================
  ! ROUND 2: Pure OpenACC (Hand-written Sort)
  ! =================================================================
  PRINT *, " "
  PRINT *, ">>> ROUND 2: Pure OpenACC (Hand-written Sort) <<<"

  ALLOCATE(raw_ax(N), raw_ay(N), raw_az(N))
  ALLOCATE(raw_tx(N), raw_ty(N), raw_tz(N))
  ALLOCATE(raw_codes(N), raw_codes_buf(N))
  ALLOCATE(raw_ids(N), raw_ids_buf(N))
  ALLOCATE(raw_cell_ptr(n_cells), raw_cell_cnt(n_cells))

  DO i = 1, N
     raw_ax(i) = REAL(MOD(i * 17, GX), 4) + 0.5
     raw_ay(i) = REAL(MOD(i * 31, GY), 4) + 0.5
     raw_az(i) = REAL(MOD(i * 13, GZ), 4) + 0.5
  END DO

  !$acc enter data copyin(raw_ax, raw_ay, raw_az) &
  !$acc            create(raw_tx, raw_ty, raw_tz) &
  !$acc            create(raw_codes, raw_codes_buf) &
  !$acc            create(raw_ids, raw_ids_buf) &
  !$acc            create(raw_cell_ptr, raw_cell_cnt)

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t1, t_rate)

  ! (A) Morton
  !$acc parallel loop present(raw_ax, raw_ay, raw_az, raw_codes, raw_ids)
  DO i = 1, N
     raw_ids(i) = INT(i, 4)
     raw_codes(i) = get_morton_code(INT(raw_ax(i),4), INT(raw_ay(i),4), INT(raw_az(i),4))
  END DO

  ! (B) Sort (使用手寫的 Pure Fortran Sort)
  ! 注意：這個函數會頻繁做 CPU-GPU 資料傳輸 (PCIe bottleneck)，預期會比 Thrust 慢很多
  CALL openacc_radix_sort_i4(N, raw_codes, raw_ids, raw_codes_buf, raw_ids_buf)

  ! (C) Gather
  CALL gather_particles(N, raw_ids, raw_ax, raw_tx)
  CALL gather_particles(N, raw_ids, raw_ay, raw_ty)
  CALL gather_particles(N, raw_ids, raw_az, raw_tz)

  ! (D) Edge Finding
  CALL find_cell_edges(N, raw_codes, raw_cell_ptr, raw_cell_cnt, n_cells)

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t2)
  t_acc = REAL(t2 - t1, 8) / REAL(t_rate, 8)

  ! Cleanup
  !$acc exit data delete(raw_ax, raw_ay, raw_az, raw_tx, raw_ty, raw_tz) &
  !$acc           delete(raw_codes, raw_codes_buf, raw_ids, raw_ids_buf) &
  !$acc           delete(raw_cell_ptr, raw_cell_cnt)
  DEALLOCATE(raw_ax, raw_ay, raw_az, raw_tx, raw_ty, raw_tz)
  DEALLOCATE(raw_codes, raw_codes_buf, raw_ids, raw_ids_buf, raw_cell_ptr, raw_cell_cnt)

  ! =================================================================
  ! 戰報
  ! =================================================================
  PRINT *, " "
  PRINT *, "============== Library vs Hand-written =============="
  PRINT '(A, F10.6, A)', " Device Vector (CUB): ", t_dv,  " s"
  PRINT '(A, F10.6, A)', " Pure OpenACC (Manual):  ", t_acc, " s"
  PRINT *, "---------------------------------------------------"
  PRINT '(A, F10.2, A)', " Optimization Speedup:   ", t_acc/t_dv, " x FASTER"
  PRINT *, "==================================================="
  
  CALL device_env_finalize()

CONTAINS

  !$acc routine seq
  FUNCTION get_morton_code(ix, iy, iz) RESULT(code)
    INTEGER(4), INTENT(IN) :: ix, iy, iz
    INTEGER(4) :: code, x, y, z, i
    x = ix; y = iy; z = iz; code = 0
    DO i = 0, 9
       IF (BTEST(x, i)) code = IBSET(code, 3*i)
       IF (BTEST(y, i)) code = IBSET(code, 3*i + 1)
       IF (BTEST(z, i)) code = IBSET(code, 3*i + 2)
    END DO
  END FUNCTION

  SUBROUTINE gather_particles(n_pts, ids_in, src, dst)
    INTEGER(8), VALUE :: n_pts
    INTEGER(4), INTENT(IN) :: ids_in(*) 
    REAL(4),    INTENT(IN) :: src(*) 
    REAL(4),    INTENT(OUT):: dst(*) 
    INTEGER(8) :: i_loop
    !$acc parallel loop gang vector present(ids_in, src, dst)
    DO i_loop = 1, n_pts
       dst(i_loop) = src(ids_in(i_loop))
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
    
    !$acc parallel loop gang vector present(sorted_codes, cell_ptr)
    DO i = 1, n_pts
       code = sorted_codes(i) + 1 
       prev_code = -1
       IF (i > 1) prev_code = sorted_codes(i-1) + 1
       IF (code /= prev_code .and. code <= n_cells) cell_ptr(code) = INT(i, 4)
    END DO
    
    !$acc parallel loop gang vector present(sorted_codes, cell_ptr, cell_cnt)
    DO i = 1, n_pts
       code = sorted_codes(i) + 1
       next_code = -1
       IF (i < n_pts) next_code = sorted_codes(i+1) + 1
       IF (code /= next_code .and. code <= n_cells) then
           start_idx = cell_ptr(code)
           cell_cnt(code) = INT(i, 4) - start_idx + 1
       END IF
    END DO
  END SUBROUTINE find_cell_edges

END PROGRAM