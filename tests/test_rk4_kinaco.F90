PROGRAM main_driver
    USE Device_Vector
    USE particles_acc
    IMPLICIT NONE

    INTEGER :: ni = 100, nj = 100, nk = 50
    INTEGER :: n_steps = 10, n_props = 10
    INTEGER :: n_part = 1000 
    REAL(4) :: dt = 60.0
    
    REAL(4) :: dx_val = 100.0, dy_val = 100.0, dz_val = 10.0, mld_val = 30.0

    INTEGER :: idx_age=1, idx_zmin=2, idx_zmax=3, idx_uvexp2=4, idx_uvexp10=5
    INTEGER :: idx_ml_ages=6, idx_ml_aged=7, idx_sfc_ages=8, idx_sfc_aged=9

    NAMELIST /test_config/ ni, nj, nk, n_steps, n_props, n_part, dt, &
                           dx_val, dy_val, dz_val, mld_val, &
                           idx_age, idx_zmin, idx_zmax, idx_uvexp2, idx_uvexp10, &
                           idx_ml_ages, idx_ml_aged, idx_sfc_ages, idx_sfc_aged
 
    REAL(4), ALLOCATABLE    :: host_data(:)
    REAL(4), ALLOCATABLE    :: disph(:,:,:), dispv(:,:,:), dispt(:,:,:)
    INTEGER(1), ALLOCATABLE :: rmvmask(:,:)
    REAL(4), ALLOCATABLE    :: zlev(:), mld(:,:), props(:,:)
    INTEGER(1), ALLOCATABLE :: alive(:)
    
    REAL(4), ALLOCATABLE :: dx(:,:), dy(:,:), dz(:)

    INTEGER :: i, j, k, n, sz_field, io_status
    PRINT *, "--- [Main] Initializing Data ---"

    PRINT *, "--- [Main] Reading configuration from input.nml ---"
    OPEN(UNIT=10, FILE='../configs/test_geometry.nml', STATUS='OLD', IOSTAT=io_status)
    IF (io_status == 0) THEN
        READ(10, NML=test_config)
        CLOSE(10)
    ELSE
        PRINT *, "Warning: test_geometry.nml not found, using default values."
    END IF

    PRINT *, "Grid:", ni, nj, nk
    PRINT *, "Particles:", n_part, " Steps:", n_steps


    ! --- 2. 配置記憶體 ---
    ! Flow Field (3 variables * grid size)
    sz_field = (ni+2)*(nj+2)*(nk+2) * 3
    ALLOCATE(host_data(sz_field))
    host_data = 1.0 !

    ! Diffusion & Mask
    ALLOCATE(disph(0:ni+1, 0:nj+1, 0:nk+1)); disph = 0.1
    ALLOCATE(dispv(0:ni+1, 0:nj+1, 0:nk+1)); dispv = 0.1
    ALLOCATE(dispt(0:ni+1, 0:nj+1, 0:nk+1)); dispt = 0.01
    ALLOCATE(rmvmask(ni, nj)); rmvmask = 0
 
    IF (.NOT. ALLOCATED(dx)) ALLOCATE(dx(ni, nj))
    dx = dx_val
    
    IF (.NOT. ALLOCATED(dy)) ALLOCATE(dy(ni, nj))
    dy = dy_val
    
    IF (.NOT. ALLOCATED(dz)) ALLOCATE(dz(nk))
    dz = dz_val

    ! Environment
    ALLOCATE(zlev(nk))
      DO k=1, nk; zlev(k) = -10.0 * REAL(k); 
    END DO
    
    ALLOCATE(mld(ni, nj)); mld = 30.0
    mld = mld_val

    ! Particles
    ALLOCATE(props(n_props, n_part)); props = 0.0
    ALLOCATE(alive(n_part)); alive = 1_1 

    IF (.NOT. ALLOCATED(dx)) ALLOCATE(dx(ni, nj)); dx = 100.0
    IF (.NOT. ALLOCATED(dy)) ALLOCATE(dy(ni, nj)); dy = 100.0
    IF (.NOT. ALLOCATED(dz)) ALLOCATE(dz(nk)); dz = 10.0

    IF (.NOT. ALLOCATED(mask_packed)) THEN
       ALLOCATE(mask_packed(-1:ni+2, -1:nj+2, -1:nk+2))
       mask_packed = 0_1
       DO k=1,nk; DO j=1,nj; DO i=1,ni
          mask_packed(i,j,k) = IBSET(63_1, 7)
       END DO; END DO; END DO
       !$acc enter data copyin(mask_packed)
    END IF

    PRINT *, "--- [Main] Calling RK4 Subroutine ---"

    ! --- 3. 呼叫 Subroutine ---
    CALL rk4(host_data, disph, dispv, dispt, rmvmask, &
             zlev, mld, props, alive, &
             ni, nj, nk, n_steps, n_props, &
             idx_age, idx_zmin, idx_zmax, idx_uvexp2, idx_uvexp10, &
             idx_ml_ages, idx_ml_aged, idx_sfc_ages, idx_sfc_aged)

    PRINT *, "--- [Main] RK4 Completed ---"
    
    !$acc update host(props)
    PRINT *, "Particle 1 Age:", props(idx_age, 1)

    ! 清理
    DEALLOCATE(host_data, disph, dispv, dispt, rmvmask, zlev, mld, props, alive)
    !$acc exit data delete(mask_packed)

CONTAINS

    INCLUDE '../src/particle/ppmain.inc'

END PROGRAM main_driver