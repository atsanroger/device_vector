PROGRAM profile_particles_soa
  USE Device_Vector
  USE cudafor
  USE iso_c_binding
  USE openacc
  IMPLICIT NONE

  ! ------------------------------------------------------------------
  ! 測試參數
  ! ------------------------------------------------------------------
  INTEGER(8), PARAMETER :: N_START   = 1000
  INTEGER(8), PARAMETER :: N_END     = 5000000     ! 5M 粒子
  INTEGER(8), PARAMETER :: STEP_SIZE = 50000       ! 每次增加 5萬
  INTEGER(8), PARAMETER :: ITERATIONS = (N_END - N_START) / STEP_SIZE
  
  INTEGER, PARAMETER :: N_PROP = 6

  ! ------------------------------------------------------------------
  ! [1] OpenACC 變數
  ! ------------------------------------------------------------------
  INTEGER(8), ALLOCATABLE :: h_id(:), h_id_tmp(:)
  INTEGER(4), ALLOCATABLE :: h_ijk(:), h_ijk_tmp(:)
  REAL(4),    ALLOCATABLE :: h_xyz(:), h_xyz_tmp(:)
  REAL(4),    ALLOCATABLE :: h_prop(:), h_prop_tmp(:)
  INTEGER(4), ALLOCATABLE :: h_seed(:), h_seed_tmp(:)
  INTEGER(4), ALLOCATABLE :: h_alive(:), h_alive_tmp(:)
  
  INTEGER(8) :: cap_acc

  ! ------------------------------------------------------------------
  ! [2] CUDA Fortran 變數
  ! ------------------------------------------------------------------
  INTEGER(8), ALLOCATABLE, DEVICE, TARGET :: d_id(:), d_id_tmp(:)
  INTEGER(4), ALLOCATABLE, DEVICE, TARGET :: d_ijk(:), d_ijk_tmp(:)
  REAL(4),    ALLOCATABLE, DEVICE, TARGET :: d_xyz(:), d_xyz_tmp(:)
  REAL(4),    ALLOCATABLE, DEVICE, TARGET :: d_prop(:), d_prop_tmp(:)
  INTEGER(4), ALLOCATABLE, DEVICE, TARGET :: d_seed(:), d_seed_tmp(:)
  INTEGER(4), ALLOCATABLE, DEVICE, TARGET :: d_alive(:), d_alive_tmp(:)
  
  INTEGER(8) :: cap_cf

  ! ------------------------------------------------------------------
  ! [3] DeviceVector
  ! ------------------------------------------------------------------
  TYPE :: particle_soa_t
     TYPE(device_vector_i8_t) :: id
     TYPE(device_vector_i4_t) :: ijk
     TYPE(device_vector_r4_t) :: xyz
     TYPE(device_vector_r4_t) :: prop
     TYPE(device_vector_i4_t) :: seed
     TYPE(device_vector_i4_t) :: alive
  END TYPE particle_soa_t
  TYPE(particle_soa_t) :: ptcl
  
  ! 用於初始化的臨時 Buffer (因為 Compute Vector 不能直接寫入)
  ! 我們需要不同型別的 Buffer 來對應不同欄位
  TYPE(device_vector_i8_t) :: init_buf_i8
  TYPE(device_vector_i4_t) :: init_buf_i4
  TYPE(device_vector_r4_t) :: init_buf_r4
  
  INTEGER(8) :: cap_dv  ! 新增 DeviceVector 的容量追蹤

  ! 計時與輔助
  INTEGER(8) :: c_start, c_end, c_rate
  REAL(8)    :: t_acc, t_cf, t_dv
  INTEGER    :: i
  INTEGER(8) :: current_n
  INTEGER(8) :: new_cap

  CALL device_env_init(0, 1)

  PRINT *, "=========================================================="
  PRINT *, "   FINAL SHOWDOWN: CAPACITY-BASED RESIZE (SEMANTIC API)   "
  PRINT *, "   Growth: ", N_START, " -> ", N_END
  PRINT *, "=========================================================="

  ! ==================================================================
  ! [1] OpenACC (Manual Capacity Growth)
  ! ==================================================================
  PRINT *, "Running [1] OpenACC..."
  cap_acc = N_START
  ALLOCATE(h_id(cap_acc), h_ijk(cap_acc*3), h_xyz(cap_acc*3))
  ALLOCATE(h_prop(cap_acc*N_PROP), h_seed(cap_acc), h_alive(cap_acc))
  
  ! 初始化資料 (公平起見)
  h_id = 1
  h_ijk = 0
  h_xyz = 0.0
  h_prop = 0.0
  h_seed = 0
  h_alive = 1
  
  !$acc enter data copyin(h_id, h_ijk, h_xyz, h_prop, h_seed, h_alive)
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(COUNT=c_start, COUNT_RATE=c_rate)

  current_n = N_START
  DO i = 1, ITERATIONS
     current_n = current_n + STEP_SIZE
     
     IF (current_n > cap_acc) THEN
        new_cap = cap_acc + cap_acc / 2
        if (new_cap < current_n) new_cap = current_n * 1.5
        
        ALLOCATE(h_id_tmp(new_cap)); h_id_tmp(1:cap_acc) = h_id(1:cap_acc)
        !$acc exit data delete(h_id); DEALLOCATE(h_id)
        CALL MOVE_ALLOC(h_id_tmp, h_id)
        !$acc enter data copyin(h_id)
        
        ALLOCATE(h_ijk_tmp(new_cap*3)); h_ijk_tmp(1:cap_acc*3) = h_ijk(1:cap_acc*3)
        !$acc exit data delete(h_ijk); DEALLOCATE(h_ijk)
        CALL MOVE_ALLOC(h_ijk_tmp, h_ijk)
        !$acc enter data copyin(h_ijk)

        ALLOCATE(h_xyz_tmp(new_cap*3)); h_xyz_tmp(1:cap_acc*3) = h_xyz(1:cap_acc*3)
        !$acc exit data delete(h_xyz); DEALLOCATE(h_xyz)
        CALL MOVE_ALLOC(h_xyz_tmp, h_xyz)
        !$acc enter data copyin(h_xyz)

        ALLOCATE(h_prop_tmp(new_cap*N_PROP)); h_prop_tmp(1:cap_acc*N_PROP) = h_prop(1:cap_acc*N_PROP)
        !$acc exit data delete(h_prop); DEALLOCATE(h_prop)
        CALL MOVE_ALLOC(h_prop_tmp, h_prop)
        !$acc enter data copyin(h_prop)

        ALLOCATE(h_seed_tmp(new_cap)); h_seed_tmp(1:cap_acc) = h_seed(1:cap_acc)
        !$acc exit data delete(h_seed); DEALLOCATE(h_seed)
        CALL MOVE_ALLOC(h_seed_tmp, h_seed)
        !$acc enter data copyin(h_seed)

        ALLOCATE(h_alive_tmp(new_cap)); h_alive_tmp(1:cap_acc) = h_alive(1:cap_acc)
        !$acc exit data delete(h_alive); DEALLOCATE(h_alive)
        CALL MOVE_ALLOC(h_alive_tmp, h_alive)
        !$acc enter data copyin(h_alive)

        cap_acc = new_cap
     END IF
  END DO
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(COUNT=c_end)
  t_acc = REAL(c_end - c_start) / REAL(c_rate)

  !$acc exit data delete(h_id, h_ijk, h_xyz, h_prop, h_seed, h_alive)
  DEALLOCATE(h_id, h_ijk, h_xyz, h_prop, h_seed, h_alive)


  ! ==================================================================
  ! [2] CUDA Fortran (Manual Capacity Growth)
  ! ==================================================================
  PRINT *, "Running [2] CUDA Fortran..."
  cap_cf = N_START
  ALLOCATE(d_id(cap_cf), d_ijk(cap_cf*3), d_xyz(cap_cf*3))
  ALLOCATE(d_prop(cap_cf*N_PROP), d_seed(cap_cf), d_alive(cap_cf))
  
  ! 初始化 (Device Kernel or Assignment)
  d_id = 1
  d_ijk = 0
  d_xyz = 0.0
  d_prop = 0.0
  d_seed = 0
  d_alive = 1
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(COUNT=c_start)

  current_n = N_START
  DO i = 1, ITERATIONS
     current_n = current_n + STEP_SIZE
     
     IF (current_n > cap_cf) THEN
        new_cap = cap_cf + cap_cf / 2
        if (new_cap < current_n) new_cap = current_n * 1.5
        
        ALLOCATE(d_id_tmp(new_cap)); d_id_tmp(1:cap_cf) = d_id(1:cap_cf)
        DEALLOCATE(d_id); CALL MOVE_ALLOC(d_id_tmp, d_id)
        
        ALLOCATE(d_ijk_tmp(new_cap*3)); d_ijk_tmp(1:cap_cf*3) = d_ijk(1:cap_cf*3)
        DEALLOCATE(d_ijk); CALL MOVE_ALLOC(d_ijk_tmp, d_ijk)

        ALLOCATE(d_xyz_tmp(new_cap*3)); d_xyz_tmp(1:cap_cf*3) = d_xyz(1:cap_cf*3)
        DEALLOCATE(d_xyz); CALL MOVE_ALLOC(d_xyz_tmp, d_xyz)

        ALLOCATE(d_prop_tmp(new_cap*N_PROP)); d_prop_tmp(1:cap_cf*N_PROP) = d_prop(1:cap_cf*N_PROP)
        DEALLOCATE(d_prop); CALL MOVE_ALLOC(d_prop_tmp, d_prop)

        ALLOCATE(d_seed_tmp(new_cap)); d_seed_tmp(1:cap_cf) = d_seed(1:cap_cf)
        DEALLOCATE(d_seed); CALL MOVE_ALLOC(d_seed_tmp, d_seed)

        ALLOCATE(d_alive_tmp(new_cap)); d_alive_tmp(1:cap_cf) = d_alive(1:cap_cf)
        DEALLOCATE(d_alive); CALL MOVE_ALLOC(d_alive_tmp, d_alive)

        cap_cf = new_cap
     END IF
  END DO

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(COUNT=c_end)
  t_cf = REAL(c_end - c_start) / REAL(c_rate)
  DEALLOCATE(d_id, d_ijk, d_xyz, d_prop, d_seed, d_alive)


  ! ==================================================================
  ! [3] DeviceVector (Mode 2: Compute Vector)
  ! ==================================================================
  PRINT *, "Running [3] DeviceVector (Compute Mode)..."
  cap_dv = N_START
  
  ! 1. 建立計算用 Vector (Pure Device)
  CALL ptcl%id%create_vector(cap_dv)
  CALL ptcl%ijk%create_vector(cap_dv * 3)
  CALL ptcl%xyz%create_vector(cap_dv * 3)
  CALL ptcl%prop%create_vector(cap_dv * N_PROP)
  CALL ptcl%seed%create_vector(cap_dv)
  CALL ptcl%alive%create_vector(cap_dv)
  
  ! 2. 初始化資料 (這也是標準流程：用 Buffer 傳輸，copy_from 給 Vector)
  ! (A) ID (Integer 8)
  CALL init_buf_i8%create_buffer(cap_dv)
  init_buf_i8%ptr(:) = 1_8
  CALL init_buf_i8%upload()
  CALL ptcl%id%copy_from(init_buf_i8)
  CALL init_buf_i8%free()

  ! (B) IJK, Seed, Alive (Integer 4)
  CALL init_buf_i4%create_buffer(cap_dv * 3) ! 共用大小，反正只是初始化
  init_buf_i4%ptr(:) = 0
  CALL init_buf_i4%upload()
  CALL ptcl%ijk%copy_from(init_buf_i4)
  
  ! Reuse buf for seed
  CALL init_buf_i4%resize(cap_dv)
  init_buf_i4%ptr(:) = 0
  CALL init_buf_i4%upload()
  CALL ptcl%seed%copy_from(init_buf_i4)
  
  ! Reuse buf for alive
  init_buf_i4%ptr(:) = 1
  CALL init_buf_i4%upload()
  CALL ptcl%alive%copy_from(init_buf_i4)
  CALL init_buf_i4%free()

  ! (C) XYZ, Prop (Real 4)
  CALL init_buf_r4%create_buffer(cap_dv * 3)
  init_buf_r4%ptr(:) = 0.0
  CALL init_buf_r4%upload()
  CALL ptcl%xyz%copy_from(init_buf_r4)
  
  CALL init_buf_r4%resize(cap_dv * N_PROP)
  init_buf_r4%ptr(:) = 0.0
  CALL init_buf_r4%upload()
  CALL ptcl%prop%copy_from(init_buf_r4)
  CALL init_buf_r4%free()
  
  
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(COUNT=c_start)

  current_n = N_START
  DO i = 1, ITERATIONS
     current_n = current_n + STEP_SIZE
     
     IF (current_n > cap_dv) THEN
        new_cap = cap_dv + cap_dv / 2
        if (new_cap < current_n) new_cap = current_n * 1.5
        
        ! 這裡呼叫 resize，底層是純 GPU 操作 (Async + Capacity)
        CALL ptcl%id%resize(new_cap)
        CALL ptcl%ijk%resize(new_cap * 3)
        CALL ptcl%xyz%resize(new_cap * 3)
        CALL ptcl%prop%resize(new_cap * N_PROP)
        CALL ptcl%seed%resize(new_cap)
        CALL ptcl%alive%resize(new_cap)
        
        cap_dv = new_cap
     END IF
  END DO

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(COUNT=c_end)
  t_dv = REAL(c_end - c_start) / REAL(c_rate)
  
  CALL ptcl%id%free()
  CALL ptcl%ijk%free()
  CALL ptcl%xyz%free()
  CALL ptcl%prop%free()
  CALL ptcl%seed%free()
  CALL ptcl%alive%free()
  CALL device_env_finalize()

  ! ==================================================================
  ! Report
  ! ==================================================================
  PRINT *, "=========================================================="
  PRINT *, "                 FINAL TRUE RESULTS                       "
  PRINT *, "=========================================================="
  PRINT '(A, F10.4, A)', " [1] OpenACC Time      : ", t_acc, " s"
  PRINT '(A, F10.4, A)', " [2] CUDA Fortran Time : ", t_cf,  " s"
  PRINT '(A, F10.4, A)', " [3] DeviceVector Time : ", t_dv,  " s"
  PRINT *, "----------------------------------------------------------"
  PRINT '(A, F10.2, A)', " Speedup (DV vs OpenACC): ", t_acc / t_dv, " x"
  PRINT '(A, F10.2, A)', " Speedup (DV vs CUDA F) : ", t_cf / t_dv,  " x"
  PRINT *, "=========================================================="

END PROGRAM profile_particles_soa