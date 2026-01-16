MODULE cpu_legacy_mod
  USE omp_lib
  IMPLICIT NONE

  INTEGER, PARAMETER :: GX=128, GY=128, GZ=128
  INTEGER :: isize=GX, jsize=GY, ksize=GZ

CONTAINS

  SUBROUTINE sort_mt(key, pin, pout, n, nt)
    INTEGER, INTENT(IN) :: n, nt, key(:), pin(:)
    INTEGER, INTENT(OUT):: pout(:)
    INTEGER :: cnt_mt(0:255, 0:nt-1), off_mt(0:255, 0:nt-1)
    INTEGER :: cnt(0:255), off(0:255), i, j, tid

!$OMP PARALLEL PRIVATE(tid, i, j)
    tid = omp_get_thread_num()
    cnt_mt(:,tid) = 0
!$OMP DO SCHEDULE(STATIC)
    DO i=1, n
       j = key(i)
       cnt_mt(j,tid) = cnt_mt(j,tid) + 1
    END DO
!$OMP SINGLE
    DO i=0, 255
       cnt(i) = sum(cnt_mt(i,:))
    END DO
    off(0) = 1
    DO i=1, 255
       off(i) = off(i-1) + cnt(i-1)
    END DO
    DO i=0, 255
       off_mt(i,0) = off(i)
       DO j=1, nt-1
          off_mt(i,j) = off_mt(i,j-1) + cnt_mt(i,j-1)
       END DO
    END DO
!$OMP END SINGLE
!$OMP DO SCHEDULE(STATIC)
    DO i=1, n
       j = key(i)
       pout(off_mt(j,tid)) = pin(i)
       off_mt(j,tid) = off_mt(j,tid) + 1
    END DO
!$OMP END PARALLEL
  END SUBROUTINE

  SUBROUTINE original_cpu_build_cell_mt(n, ijk_arr, cell_ptr, cell_cnt, cell_gen, nt)
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN)    :: nt, ijk_arr(3,n)
    INTEGER, INTENT(OUT)   :: cell_ptr(GX,GY,GZ), cell_cnt(GX,GY,GZ), cell_gen(GX,GY,GZ)
    INTEGER :: cnt_local(0:1, 0:nt-1), ptr_local(0:1, 0:nt-1), ijk_local(3, 0:1, 0:nt-1)
    LOGICAL :: flag(0:1, 0:nt-1)
    INTEGER :: ptr_start(0:nt), key_start(0:nt), tid, a, b, i, j, k, key, m, ptr

    a = n / nt ; b = n - a*nt
    DO tid=0, nt-1
       ptr_start(tid) = tid*a + min(tid,b) + 1
       i=ijk_arr(1,ptr_start(tid)); j=ijk_arr(2,ptr_start(tid)); k=ijk_arr(3,ptr_start(tid))
       key_start(tid) = key_morton3_cpu(i,j,k)
    END DO
    ptr_start(nt) = n+1 ; key_start(nt) = 2147483647
    flag(:,:) = .FALSE.

!$OMP PARALLEL PRIVATE(i,j,k,ptr,key,tid) NUM_THREADS(nt)
    tid = omp_get_thread_num()
    DO ptr=ptr_start(tid), ptr_start(tid+1)-1
       i=ijk_arr(1,ptr); j=ijk_arr(2,ptr); k=ijk_arr(3,ptr); key = key_morton3_cpu(i,j,k)
       IF (key==key_start(tid)) THEN
          IF (flag(0,tid)) THEN ; cnt_local(0,tid) = cnt_local(0,tid) + 1
          ELSE ; cnt_local(0,tid) = 1; ptr_local(0,tid) = ptr ; END IF
          ijk_local(:,0,tid) = (/i,j,k/) ; flag(0,tid) = .TRUE.
       ELSE IF (key==key_start(tid+1)) THEN
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

  PURE FUNCTION key_morton3_cpu(i, j, k) RESULT(key)
     INTEGER, INTENT(IN) :: i, j, k
     INTEGER :: key, ih, il, jh, jl, kh, kl
     INTEGER, PARAMETER :: tbl(0:7) = (/0, 1, 8, 9, 64, 65, 72, 73/)
     ih = ISHFT(i-1, -3); jh = ISHFT(j-1, -3); kh = ISHFT(k-1, -3)
     il = IAND(i-1, 7_4); jl = IAND(j-1, 7_4); kl = IAND(k-1, 7_4)
     key = (kh*ISHFT(jsize+7,-3) + jh)*ISHFT(isize+7,-3) + ih
     key = ISHFT(key, 9)
     key = IOR(key, ISHFT(tbl(kl),2))
     key = IOR(key, ISHFT(tbl(jl),1))
     key = IOR(key, tbl(il))
  END FUNCTION

END MODULE cpu_legacy_mod

MODULE pure_openacc_sort_mod
  USE openacc
  IMPLICIT NONE

  ! GPU Hardware constant
  INTEGER, PARAMETER :: WARP_LENGTH = 128

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
  IMPLICIT NONE

  ! 1. Namelist
  INTEGER(8) :: N, GX, GY, GZ, n_cells
  INTEGER    :: ios, alloc_stat
  NAMELIST /sim_config/ N, GX, GY, GZ

  ! 2. 變數
  INTEGER(8) :: t1, t2, t_rate
  REAL(8)    :: t_dv, t_acc, t_cpu 

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

  ALLOCATE(x(N), y(N), z(N), id(N), ijk(3,N), key(N), key_(N), perm(N), work(N))
  ALLOCATE(c_ptr(GX,GY,GZ), c_cnt(GX,GY,GZ), c_gen(GX,GY,GZ))

  !$OMP PARALLEL DO
  DO i = 1, N
     x(i) = REAL(MOD(i, GX) + 1)
     y(i) = REAL(MOD(i, GY) + 1)
     z(i) = REAL(MOD(i, GZ) + 1)
     ijk(1,i) = INT(x(i)); ijk(2,i) = INT(y(i)); ijk(3,i) = INT(z(i))
     id(i) = i; perm(i) = i
  END DO

  ! =================================================================
  ! ROUND 1: Device Vector (CUB Sort)
  ! =================================================================
  PRINT *, " "
  PRINT *, ">>> ROUND 1: Device Vector (CUB Sort) <<<"

  CALL device_synchronize()

  CALL dv_px%create_buffer(N);    CALL dv_py%create_buffer(N); CALL dv_pz%create_buffer(N)
  CALL dv_tx%create_buffer(N);    CALL dv_ty%create_buffer(N); CALL dv_tz%create_buffer(N)
  CALL dv_codes%create_buffer(N); CALL dv_codes_buf%create_buffer(N)
  CALL dv_ids%create_buffer(N);   CALL dv_ids_buf%create_buffer(N)
  CALL dv_cell_ptr%create_buffer(n_cells); CALL dv_cell_cnt%create_buffer(n_cells)

  CALL dv_px%acc_map(ptr_ax);       CALL dv_py%acc_map(ptr_ay); CALL dv_pz%acc_map(ptr_az)
  CALL dv_tx%acc_map(ptr_atx);      CALL dv_ty%acc_map(ptr_aty); CALL dv_tz%acc_map(ptr_atz)
  CALL dv_codes%acc_map(ptr_codes); CALL dv_ids%acc_map(ptr_ids)
  CALL dv_cell_ptr%acc_map(ptr_cell_ptr); CALL dv_cell_cnt%acc_map(ptr_cell_cnt)

  !$acc parallel loop gang vector_length(WARP_LENGTH) &
  !$acc present(ptr_ax, ptr_ay, ptr_az)
  DO i = 1, N
     ptr_ax(i) = REAL(MOD(i * 17, GX), 4) + 0.5
     ptr_ay(i) = REAL(MOD(i * 31, GY), 4) + 0.5
     ptr_az(i) = REAL(MOD(i * 13, GZ), 4) + 0.5
  END DO

  ! (A) Morton
  !$acc parallel loop gang vector_length(WARP_LENGTH) &
  !$acc present(ptr_ax, ptr_ay, ptr_az, ptr_codes, ptr_ids)
  DO i = 1, N
     ptr_ids(i)   = INT(i, 4) 
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
  
  CALL dv_px%acc_unmap();    CALL dv_py%acc_unmap(); CALL dv_pz%acc_unmap()
  CALL dv_tx%acc_unmap();    CALL dv_ty%acc_unmap(); CALL dv_tz%acc_unmap()
  CALL dv_codes%acc_unmap(); CALL dv_codes_buf%acc_unmap()
  CALL dv_ids%acc_unmap();   CALL dv_ids_buf%acc_unmap()
  CALL dv_cell_ptr%acc_unmap(); CALL dv_cell_cnt%acc_unmap()

  CALL dv_px%free();    CALL dv_py%free(); CALL dv_pz%free()
  CALL dv_tx%free();    CALL dv_ty%free(); CALL dv_tz%free()
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
  !$acc parallel loop gang vector_length(WARP_LENGTH) &
  !$acc present(raw_ax, raw_ay, raw_az, raw_codes, raw_ids)
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