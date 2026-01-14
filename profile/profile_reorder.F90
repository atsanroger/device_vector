PROGRAM test_reorder_dv_nml
  USE Device_Vector 
  USE openacc
  USE iso_c_binding
  IMPLICIT NONE

  ! 1. 改為變數，移除 PARAMETER
  INTEGER(8) :: N 
  INTEGER(8) :: GX, GY, GZ
  INTEGER    :: ios

  ! 2. 定義 Namelist
  NAMELIST /sim_config/ N, GX, GY, GZ

  ! Device Vectors
  TYPE(device_vector_r4_t) :: px, py, pz     ! 原始
  TYPE(device_vector_r4_t) :: tx, ty, tz     ! 暫存
  REAL(4), POINTER :: ax(:), ay(:), az(:)
  REAL(4), POINTER :: atx(:), aty(:), atz(:)

  ! 排序 Buffer
  TYPE(device_vector_i4_t) :: dv_codes, dv_codes_buf
  TYPE(device_vector_i4_t) :: dv_ids,   dv_ids_buf
  INTEGER, POINTER :: codes(:), ids(:) 

  ! 驗證用 & 計時
  INTEGER, ALLOCATABLE :: h_codes(:)
  INTEGER(8) :: i
  LOGICAL :: is_sorted
  INTEGER(8) :: t1, t2, t_rate
  REAL(8)    :: t_reorder

  CALL device_env_init(0, 1)

  PRINT *, "[Init] Reading test_reorder.nml..."
  OPEN(UNIT=10, FILE='../configs/test_reorder.nml', STATUS='OLD', IOSTAT=ios)
  IF (ios == 0) THEN
     READ(10, NML=sim_config)
     CLOSE(10)
     PRINT *, "  -> Loaded Configuration:"
     PRINT '(A,I12)', "     N  = ", N
     PRINT '(A,I12)', "     GX = ", GX
  ELSE
     PRINT *, "  You can create ../configs/test_reorder.nml with &sim_config group)"
  END IF

  PRINT *, "[Init] Allocating GPU memory..."

  ! 建立粒子 Buffer
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
     ax(i) = REAL(MOD(i * 17, GX), 4) 
     ay(i) = REAL(MOD(i * 31, GY), 4)
     az(i) = REAL(MOD(i * 13, GZ), 4)
  END DO
  
  CALL device_synchronize()

  ! ★★★ 計時開始 ★★★
  PRINT *, "[Run] Starting Reorder Timer..."
  CALL SYSTEM_CLOCK(t1, t_rate)

  ! Step 1: Calculating Morton Codes
  !$acc parallel loop present(ax, ay, az, codes, ids)
  DO i = 1, N
     ids(i) = INT(i, 4) 
     codes(i) = get_morton_code(INT(ax(i),4), INT(ay(i),4), INT(az(i),4))
  END DO
  !$acc wait

  ! Step 2: GPU Sort
  CALL dv_codes%acc_unmap() 
  CALL dv_ids%acc_unmap()
  
  CALL vec_sort_i4(dv_codes%get_handle(), dv_codes_buf%get_handle(), &
                   dv_ids%get_handle(),   dv_ids_buf%get_handle())

  CALL dv_codes%acc_map(codes)
  CALL dv_ids%acc_map(ids)

  ! Step 3: Gather
  CALL gather_particles(N, ids, ax, atx)
  CALL gather_particles(N, ids, ay, aty)
  CALL gather_particles(N, ids, az, atz)

  ! Update back
  !$acc parallel loop present(ax, ay, az, atx, aty, atz)
  DO i = 1, N
     ax(i) = atx(i)
     ay(i) = aty(i)
     az(i) = atz(i)
  END DO

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(t2)
  ! ★★★ 計時結束 ★★★

  t_reorder = REAL(t2 - t1, 8) / REAL(t_rate, 8)
  PRINT *, "------------------------------------------------"
  PRINT '(A, F10.6, A)', " [Reorder] Total Time: ", t_reorder, " s"
  PRINT '(A, F10.2, A)', " [Perf] Throughput:    ", REAL(N)/t_reorder/1e6, " M/s"
  PRINT *, "------------------------------------------------"

  ! 驗證 (只印前10筆確認)
  ALLOCATE(h_codes(N))
  !$acc update host(codes)
  h_codes = codes(1:N)

  PRINT *, "Index | Sorted Code (First 10)"
  DO i = 1, 10
     WRITE(*,*) i, h_codes(i)
  END DO

  is_sorted = .TRUE.
  DO i = 1, N-1
     IF (h_codes(i) > h_codes(i+1)) THEN
        is_sorted = .FALSE.
        PRINT *, ">>> ERROR at ", i
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

END PROGRAM