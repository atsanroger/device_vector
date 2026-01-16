MODULE constants_mod
  IMPLICIT NONE
  INTEGER, PARAMETER :: WARP_LENGTH = 128
END MODULE constants_mod

MODULE cpu_legacy_mod
  USE omp_lib
  IMPLICIT NONE

CONTAINS

  SUBROUTINE sort_mt(h_keys, pin, pout, n, nt)
    INTEGER, INTENT(IN) :: nt, h_keys(:), pin(:)
    INTEGER(8), INTENT(IN) :: n
    INTEGER(8) :: idx
    INTEGER, INTENT(OUT):: pout(:)
    INTEGER :: cnt_mt(0:255, 0:nt-1), off_mt(0:255, 0:nt-1)
    INTEGER :: cnt(0:255), off(0:255), j, tid

!$OMP PARALLEL PRIVATE(tid, idx, j)
    tid = omp_get_thread_num()
    cnt_mt(:,tid) = 0
!$OMP DO SCHEDULE(STATIC)
    DO idx=1, n
       j = h_keys(idx)
       cnt_mt(j,tid) = cnt_mt(j,tid) + 1
    END DO
!$OMP SINGLE
    DO idx=0, 255
       cnt(idx) = sum(cnt_mt(idx,:))
    END DO
    off(0) = 1
    DO idx=1, 255
       off(idx) = off(idx-1) + cnt(idx-1)
    END DO
    DO idx=0, 255
       off_mt(idx,0) = off(idx)
       DO j=1, nt-1
          off_mt(idx,j) = off_mt(idx,j-1) + cnt_mt(idx,j-1)
       END DO
    END DO
!$OMP END SINGLE
!$OMP DO SCHEDULE(STATIC)
    DO idx=1, n
       j = h_keys(idx)
       pout(off_mt(j,tid)) = pin(idx)
       off_mt(j,tid) = off_mt(j,tid) + 1
    END DO
!$OMP END PARALLEL
  END SUBROUTINE

  SUBROUTINE original_cpu_build_cell_mt(n, ijk_arr, cell_ptr, cell_cnt, cell_gen, nt, GX, GY, GZ)
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN)    :: nt, ijk_arr(3,n)
    INTEGER(8), INTENT(IN) :: GX, GY, GZ
    INTEGER, INTENT(OUT)   :: cell_ptr(GX,GY,GZ), cell_cnt(GX,GY,GZ), cell_gen(GX,GY,GZ)
    INTEGER :: cnt_local(0:1, 0:nt-1), ptr_local(0:1, 0:nt-1), ijk_local(3, 0:1, 0:nt-1)
    LOGICAL :: flag(0:1, 0:nt-1)
    INTEGER :: ptr_start(0:nt), key_start(0:nt), tid, a, b, i, j, k, h_keys, m, ptr

    a = n / nt ; b = n - a*nt
    DO tid=0, nt-1
       ptr_start(tid) = tid*a + min(tid,b) + 1
       i=ijk_arr(1,ptr_start(tid)); j=ijk_arr(2,ptr_start(tid)); k=ijk_arr(3,ptr_start(tid))
       key_start(tid) = key_morton3_cpu(i,j,k, GX, GY, GZ)
    END DO
    ptr_start(nt) = n+1 ; key_start(nt) = 2147483647
    flag(:,:) = .FALSE.

!$OMP PARALLEL PRIVATE(i,j,k,ptr,h_keys,tid) NUM_THREADS(nt)
    tid = omp_get_thread_num()
    DO ptr=ptr_start(tid), ptr_start(tid+1)-1
       i=ijk_arr(1,ptr); j=ijk_arr(2,ptr); k=ijk_arr(3,ptr); h_keys = key_morton3_cpu(i,j,k, GX, GY, GZ)
       IF (h_keys==key_start(tid)) THEN
          IF (flag(0,tid)) THEN ; cnt_local(0,tid) = cnt_local(0,tid) + 1
          ELSE ; cnt_local(0,tid) = 1; ptr_local(0,tid) = ptr ; END IF
          ijk_local(:,0,tid) = (/i,j,k/) ; flag(0,tid) = .TRUE.
       ELSE IF (h_keys==key_start(tid+1)) THEN
          IF (flag(1,tid)) THEN ; cnt_local(1,tid) = cnt_local(1,tid) + 1
          ELSE ; cnt_local(1,tid) = 1; ptr_local(1,tid) = ptr ; END IF
          ijk_local(:,1,tid) = (/i,j,k/) ; flag(1,tid) = .TRUE.
       ELSE
          IF (cell_gen(i,j,k) < 1) THEN
             cell_ptr(i,j,k) = ptr; cell_cnt(i,j,k) = 1; cell_gen(i,j,k) = 1
          ELSE ; cell_cnt(i,j,k) = cell_cnt(i,j,k) + 1 ; END IF
       END IF
    END DO
!$OMP END PARALLEL

    DO tid=0, nt-1
       DO m=0, 1
          IF (flag(m,tid)) THEN
             i=ijk_local(1,m,tid); j=ijk_local(2,m,tid); k=ijk_local(3,m,tid)
             IF (cell_gen(i,j,k) < 1) THEN
                cell_ptr(i,j,k) = ptr_local(m,tid); cell_cnt(i,j,k) = cnt_local(m,tid); cell_gen(i,j,k) = 1
             ELSE
                cell_ptr(i,j,k) = min(cell_ptr(i,j,k), ptr_local(m,tid))
                cell_cnt(i,j,k) = cell_cnt(i,j,k) + cnt_local(m,tid)
             END IF
          END IF
       END DO
    END DO
  END SUBROUTINE

  PURE FUNCTION key_morton3_cpu(i, j, k, GX, GY, GZ) RESULT(h_keys)
     INTEGER, INTENT(IN) :: i, j, k
     INTEGER(8), INTENT(IN) :: GX, GY, GZ 
     INTEGER :: h_keys, ih, il, jh, jl, kh, kl
     INTEGER, PARAMETER :: tbl(0:7) = (/0, 1, 8, 9, 64, 65, 72, 73/)
     ih = ISHFT(i-1, -3); jh = ISHFT(j-1, -3); kh = ISHFT(k-1, -3)
     il = IAND(i-1, 7_4); jl = IAND(j-1, 7_4); kl = IAND(k-1, 7_4)
     h_keys = (kh*ISHFT(GY+7,-3) + jh)*ISHFT(GX+7,-3) + ih
     h_keys = ISHFT(h_keys, 9)
     h_keys = IOR(h_keys, ISHFT(tbl(kl),2))
     h_keys = IOR(h_keys, ISHFT(tbl(jl),1))
     h_keys = IOR(h_keys, tbl(il))
  END FUNCTION

END MODULE cpu_legacy_mod

MODULE pure_openacc_sort_mod
  USE openacc
  USE constants_mod
  IMPLICIT NONE

CONTAINS

  SUBROUTINE openacc_radix_sort_i4(n, keys, vals, keys_buf, vals_buf)
    INTEGER(8), VALUE :: n
    INTEGER(4), INTENT(INOUT) :: keys(:), vals(:)
    INTEGER(4), INTENT(INOUT) :: keys_buf(:), vals_buf(:)
    
    INTEGER(4), ALLOCATABLE :: bits(:), offsets(:)
    INTEGER(4), ALLOCATABLE :: h_bits(:), h_offsets(:)
    INTEGER :: bit, i, zeros, total_zeros
    
    ALLOCATE(bits(n), offsets(n))
    ALLOCATE(h_bits(n), h_offsets(n))
    
    !$acc enter data create(bits, offsets)

    DO bit = 0, 29
       
       !$acc parallel loop present(keys, bits)
       DO i = 1, n
          bits(i) = IAND(ISHFT(keys(i), -bit), 1)
       END DO
       
       !$acc update host(bits)
       
       h_bits = bits(1:n)
       
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
       
       DO i = 1, n
          IF (h_bits(i) == 1) THEN
             zeros = zeros + 1
             h_offsets(i) = zeros
          END IF
       END DO
       
       offsets(1:n) = h_offsets(1:n)
       !$acc update device(offsets)
       
       !$acc parallel loop present(keys, vals, keys_buf, vals_buf, bits, offsets)
       DO i = 1, n
          keys_buf(offsets(i)) = keys(i)
          vals_buf(offsets(i)) = vals(i)
       END DO
       
       !$acc parallel loop gang vector_length(WARP_LENGTH) &
       !$acc present(keys, vals, keys_buf, vals_buf)
       DO i = 1, n
          keys(i) = keys_buf(i)
          vals(i) = vals_buf(i)
       END DO
       
    END DO

    !$acc exit data delete(bits, offsets)
    DEALLOCATE(bits, offsets, h_bits, h_offsets)
    
  END SUBROUTINE openacc_radix_sort_i4
END MODULE pure_openacc_sort_mod

PROGRAM test_pure_vs_oop
  USE Device_Vector 
  USE pure_openacc_sort_mod
  USE cpu_legacy_mod
  USE openacc
  USE omp_lib
  USE constants_mod
  IMPLICIT NONE

  ! 1. Namelist
  INTEGER(8) :: N, GX, GY, GZ, n_cells, idx
  INTEGER    :: ios, alloc_stat, m_code
  NAMELIST /sim_config/ N, GX, GY, GZ

  ! 2. 變數
  INTEGER(8) :: t1, t2, t_rate
  REAL(8)    :: t_dv, t_acc, t_cpu 

  ! OOP 變數
  TYPE(device_vector_r4_t) :: dv_px, dv_py, dv_pz, dv_tx, dv_ty, dv_tz, io_buf
  TYPE(device_vector_i4_t) :: dv_codes, dv_codes_buf, dv_ids, dv_ids_buf
  TYPE(device_vector_i4_t) :: dv_cell_ptr, dv_cell_cnt
  REAL(4), POINTER :: dp_ax(:), dp_ay(:), dp_az(:), dp_atx(:), dp_aty(:), dp_atz(:)
  INTEGER, POINTER :: dp_codes(:), dp_ids(:), dp_cell_ptr(:), dp_cell_cnt(:)

  ! Raw 變數
  REAL(4),    ALLOCATABLE, TARGET :: raw_ax(:), raw_ay(:), raw_az(:)
  REAL(4),    ALLOCATABLE, TARGET :: raw_tx(:), raw_ty(:), raw_tz(:)
  INTEGER(4), ALLOCATABLE, TARGET :: raw_codes(:), raw_codes_buf(:)
  INTEGER(4), ALLOCATABLE, TARGET :: raw_ids(:),   raw_ids_buf(:)
  INTEGER(4), ALLOCATABLE, TARGET :: raw_cell_ptr(:), raw_cell_cnt(:)
  INTEGER(8) :: i, j, k

  REAL(4), ALLOCATABLE :: h_x(:), h_y(:), h_z(:)
  INTEGER, ALLOCATABLE :: h_ids(:), h_ijk(:,:)          ! h_ids -> h_ids
  INTEGER, ALLOCATABLE :: h_keys(:), h_keys_buf(:)      ! h_keys -> h_keys
  INTEGER, ALLOCATABLE :: h_perm(:), h_work(:)          ! h_work -> h_work
  INTEGER, ALLOCATABLE :: legacy_cell_ptr(:,:,:), legacy_cell_cnt(:,:,:), legacy_cell_gen(:,:,:)
  INTEGER :: nt, ptr
  INTEGER, ALLOCATABLE :: h_ijk_sorted(:,:)
  INTEGER :: err_count, diff_val, idx_1d
  INTEGER(8) :: sum_gpu, sum_cpu, val_gpu, val_cpu

  nt = omp_get_max_threads()

  CALL device_env_init(0, 1)
  
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
  PRINT *, "---------------------------------------"
  PRINT *, " CONFIG: N =", N, " Grid =", GX, GY, GZ
  PRINT *, "---------------------------------------"

  ! =================================================================
  ! ROUND 1: Device Vector (CUB Sort)
  ! =================================================================
  PRINT *, " "
  PRINT *, ">>> ROUND 1: Device Vector (CUB Sort) <<<"

  CALL device_synchronize()

  ! gpu buffer
  CALL dv_px%create_buffer(N);    CALL dv_py%create_buffer(N); CALL dv_pz%create_buffer(N)
  CALL dv_tx%create_buffer(N);    CALL dv_ty%create_buffer(N); CALL dv_tz%create_buffer(N)
  CALL dv_codes%create_buffer(N); CALL dv_codes_buf%create_buffer(N)
  CALL dv_ids%create_buffer(N);   CALL dv_ids_buf%create_buffer(N)
  CALL dv_cell_ptr%create_buffer(n_cells); CALL dv_cell_cnt%create_buffer(n_cells)

  ! IO Buffer
  CALL io_buf%create_buffer(N)

  CALL dv_px%acc_map(dp_ax);       CALL dv_py%acc_map(dp_ay); CALL dv_pz%acc_map(dp_az)
  CALL dv_tx%acc_map(dp_atx);      CALL dv_ty%acc_map(dp_aty); CALL dv_tz%acc_map(dp_atz)
  CALL dv_codes%acc_map(dp_codes); CALL dv_ids%acc_map(dp_ids)
  CALL dv_cell_ptr%acc_map(dp_cell_ptr); CALL dv_cell_cnt%acc_map(dp_cell_cnt)

  !$acc parallel loop gang vector_length(WARP_LENGTH) &
  !$acc present(dp_ax, dp_ay, dp_az)
  DO i = 1, N
     dp_ax(i) = REAL(MOD(i * 17, GX), 4) + 0.5
     dp_ay(i) = REAL(MOD(i * 31, GY), 4) + 0.5
     dp_az(i) = REAL(MOD(i * 13, GZ), 4) + 0.5
  END DO

  PRINT *, "Downloading Data to CPU for Verification..."

  ALLOCATE(h_x(N), h_y(N), h_z(N))
  ALLOCATE(h_ids(N))        ! 這裡原本是 h_ids(N)
  ALLOCATE(h_ijk(3,N))
  ALLOCATE(h_keys(N))       ! 這裡原本是 h_keys(N)
  ALLOCATE(h_keys_buf(N))   ! 這裡原本是 h_keys_buf(N)
  ALLOCATE(h_perm(N))
  ALLOCATE(h_work(N))       ! 這裡原本是 h_work(N)
  
  ALLOCATE(legacy_cell_ptr(GX,GY,GZ), legacy_cell_cnt(GX,GY,GZ), legacy_cell_gen(GX,GY,GZ))
  legacy_cell_ptr = 0
  legacy_cell_cnt = 0
  legacy_cell_gen = 0

  ! --- Download X ---
  CALL io_buf%copy_from(dv_px); CALL io_buf%download(); h_x(1:N) = io_buf%ptr(1:N)
  CALL io_buf%copy_from(dv_py); CALL io_buf%download(); h_y(1:N) = io_buf%ptr(1:N)
  CALL io_buf%copy_from(dv_pz); CALL io_buf%download(); h_z(1:N) = io_buf%ptr(1:N)

  !$OMP PARALLEL DO
  DO i = 1, N
     h_ijk(1,i) = INT(h_x(i)) +1; 
     h_ijk(2,i) = INT(h_y(i)) +1; 
     h_ijk(3,i) = INT(h_z(i)) +1;
     h_ids(i)  = i; 
     h_perm(i) = i
  END DO
  
  PRINT *, "Data Sync Complete. Starting Profile."
  PRINT *, "---------------------------------------"

  CALL SYSTEM_CLOCK(t1, t_rate)

  ! (A) Morton
  !$acc parallel loop gang vector_length(WARP_LENGTH) &
  !$acc present(dp_ax, dp_ay, dp_az, dp_codes, dp_ids)
  DO i = 1, N
     dp_ids(i)   = INT(i, 4) 
     dp_codes(i) = get_morton_code(INT(dp_ax(i),4), INT(dp_ay(i),4), INT(dp_az(i),4))
  END DO

  ! (B) Sort (OOP -> Thrust)
  !CALL dv_codes%acc_unmap(); CALL dv_ids%acc_unmap()
  CALL vec_sort_i4(dv_codes%get_handle(), dv_codes_buf%get_handle(), &
                   dv_ids%get_handle(),   dv_ids_buf%get_handle())
  !CALL dv_codes%acc_map(dp_codes); CALL dv_ids%acc_map(dp_ids)

  ! (C) Gather
  CALL gather_particles(N, dp_ids, dp_ax, dp_atx)
  CALL gather_particles(N, dp_ids, dp_ay, dp_aty)
  CALL gather_particles(N, dp_ids, dp_az, dp_atz)

  ! (D) Edge Finding
  CALL find_cell_edges(N, dp_codes, dp_cell_ptr, dp_cell_cnt, n_cells)
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t2)
  t_dv = REAL(t2 - t1, 8) / REAL(t_rate, 8)
  
  PRINT *, "Downloading GPU results..."
  CALL dv_cell_cnt%download()
  CALL dv_cell_ptr%download()

  CALL dv_px%acc_unmap();    CALL dv_py%acc_unmap(); CALL dv_pz%acc_unmap()
  CALL dv_tx%acc_unmap();    CALL dv_ty%acc_unmap(); CALL dv_tz%acc_unmap()
  CALL dv_codes%acc_unmap(); CALL dv_codes_buf%acc_unmap()
  CALL dv_ids%acc_unmap();   CALL dv_ids_buf%acc_unmap()
  CALL dv_cell_ptr%acc_unmap(); CALL dv_cell_cnt%acc_unmap()

  PRINT *, " "
  PRINT *, ">>> ROUND 2: Kinaco CPU Execution <<<"

  CALL SYSTEM_CLOCK(t1, t_rate)

  !$OMP PARALLEL DO
  DO i = 1, N
     h_keys(i) = key_morton3_cpu(h_ijk(1,i), h_ijk(2,i), h_ijk(3,i), GX, GY, GZ)
  END DO

  DO idx = 1, 4
     !$OMP PARALLEL DO
     DO ptr = 1, N
        h_keys_buf(ptr) = IAND(ISHFT(h_keys(h_perm(ptr)), -(idx-1)*8), 255)
     END DO
     CALL sort_mt(h_keys_buf, h_perm, h_work, N, nt)
     h_perm = h_work
  END DO

  ALLOCATE(h_ijk_sorted(3, N))

  !$OMP PARALLEL DO
  DO i = 1, N
     h_ijk_sorted(1, i) = h_ijk(1, h_perm(i))
     h_ijk_sorted(2, i) = h_ijk(2, h_perm(i))
     h_ijk_sorted(3, i) = h_ijk(3, h_perm(i))
  END DO

  CALL original_cpu_build_cell_mt(N, h_ijk_sorted, legacy_cell_ptr, legacy_cell_cnt, legacy_cell_gen, nt, GX, GY, GZ)

  CALL SYSTEM_CLOCK(t2, t_rate)
  t_cpu = REAL(t2 - t1, 8) / REAL(t_rate, 8)

  err_count = 0
  sum_gpu = 0
  sum_cpu = 0

  DO k = 1, GZ
     DO j = 1, GY
        DO i = 1, GX
           
           ! [修正點 1] 使用 get_morton_code_host 產生 GPU 懂的 Key
           ! 傳入 0-based 座標 (i-1)
           m_code = get_morton_code_host(INT(i-1, 4), INT(j-1, 4), INT(k-1, 4))
           
           ! [修正點 2] 查詢 GPU 結果 (Key + 1)
           IF (m_code + 1 > n_cells) THEN
               val_gpu = 0
           ELSE
               val_gpu = dp_cell_cnt(m_code + 1)
           END IF
           
           ! 查詢 CPU 結果 (直接查三維陣列)
           val_cpu = legacy_cell_cnt(i,j,k)           
           sum_gpu = sum_gpu + val_gpu
           sum_cpu = sum_cpu + val_cpu
           
           IF (val_gpu /= val_cpu) THEN
              err_count = err_count + 1
              IF (err_count <= 5) THEN
                 PRINT '(A,3I4, A,I8, A,I8, A,I8)', &
                    "Mismatch @(", i, j, k, ") Morton:", m_code, " GPU:", val_gpu, " CPU:", val_cpu
              END IF
           END IF
        END DO
     END DO
  END DO

  PRINT *, "---------------------------------------------"
  PRINT *, "Total Particles (GPU):", sum_gpu
  PRINT *, "Total Particles (CPU):", sum_cpu
  PRINT *, "Mismatched Cells     :", err_count
  
  IF (err_count == 0 .AND. sum_gpu == N) THEN
     PRINT *, ">>> VERIFICATION PASSED! Results match perfectly. <<<"
  ELSE
     PRINT *, ">>> VERIFICATION FAILED! <<<"
     IF (sum_gpu == sum_cpu .AND. err_count > 0) THEN
        PRINT *, "Note: Total count matches but distribution differs."
        PRINT *, "      Check if CPU h_ijk was reordered using h_perm before building cells."
     END IF
  END IF
  PRINT *, "============================================="

  ! =================================================================
  ! Benchmark Result
  ! =================================================================
  PRINT *, " "
  PRINT *, "============== GPU vs openMP =============="
  PRINT '(A, F10.6, A)', " Device Vector (CUB):   ", t_dv,  " s"
  PRINT '(A, F10.6, A)', " Kinaco CPU   (openMP): ", t_cpu, " s"
  PRINT *, "---------------------------------------------------"
  PRINT '(A, F10.2, A)', " Optimization Speedup (to openMP) :", t_cpu/t_dv, "times FASTER"
  PRINT *, "==================================================="

  DEALLOCATE(h_ijk_sorted)

  CALL dv_px%free()       
  CALL dv_py%free() 
  CALL dv_pz%free()
  CALL dv_tx%free()       
  CALL dv_ty%free() 
  CALL dv_tz%free()
  CALL dv_codes%free()    
  CALL dv_codes_buf%free()
  CALL dv_ids%free()      
  CALL dv_ids_buf%free()
  CALL dv_cell_ptr%free()
  CALL dv_cell_cnt%free()

  CALL device_env_finalize()

CONTAINS

  FUNCTION get_morton_code_host(ix, iy, iz) RESULT(code)
    INTEGER(4), INTENT(IN) :: ix, iy, iz
    INTEGER(4) :: code, h_x, h_y, h_z, i
    
    ! 模擬 GPU 的行為 (與 get_morton_code 完全一樣)
    h_x = ix; h_y = iy; h_z = iz
    code = 0
    DO i = 0, 9
       IF (BTEST(h_x, i)) code = IBSET(code, 3*i)
       IF (BTEST(h_y, i)) code = IBSET(code, 3*i + 1)
       IF (BTEST(h_z, i)) code = IBSET(code, 3*i + 2)
    END DO
  END FUNCTION

  !$acc routine seq
  FUNCTION get_morton_code(ix, iy, iz) RESULT(code)
    INTEGER(4), INTENT(IN) :: ix, iy, iz
    INTEGER(4) :: code, h_x, h_y, h_z, i
    h_x = ix; h_y = iy; h_z = iz; code = 0
    DO i = 0, 9
       IF (BTEST(h_x, i)) code = IBSET(code, 3*i)
       IF (BTEST(h_y, i)) code = IBSET(code, 3*i + 1)
       IF (BTEST(h_z, i)) code = IBSET(code, 3*i + 2)
    END DO
  END FUNCTION

  SUBROUTINE gather_particles(n_pts, ids_in, src, dst)
    INTEGER(8), VALUE :: n_pts
    INTEGER(4), INTENT(IN) :: ids_in(*) 
    REAL(4),    INTENT(IN) :: src(*) 
    REAL(4),    INTENT(OUT):: dst(*) 
    INTEGER(8) :: i_loop
    !$acc parallel loop gang vector_length(WARP_LENGTH) &
    !$acc present(ids_in, src, dst)
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
    
    !$acc parallel loop gang vector_length(WARP_LENGTH) &
    !$acc present(sorted_codes, cell_ptr)
    DO i = 1, n_pts
       code = sorted_codes(i) + 1 
       prev_code = -1
       IF (i > 1) prev_code = sorted_codes(i-1) + 1
       IF (code /= prev_code .and. code <= n_cells) cell_ptr(code) = INT(i, 4)
    END DO
    
    !$acc parallel loop gang vector_length(WARP_LENGTH) & 
    !$acc present(sorted_codes, cell_ptr, cell_cnt)
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