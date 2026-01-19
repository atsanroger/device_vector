PROGRAM main_driver
    USE Device_Vector
    USE openacc
    USE Random_Mod
    USE Physics_Routines_Mod
    USE Particle_Tracker_Mod ! The module we just defined above
    IMPLICIT NONE

    INTEGER :: ni = 100, nj = 100, nk = 50
    INTEGER :: n_steps = 10, n_props = 10, n_part = 1000 
    REAL(4) :: dt = 60.0
    
    REAL(4) :: dx_val = 100.0, dy_val = 100.0, dz_val = 10.0, mld_val = 30.0


    NAMELIST /test_config/ ni, nj, nk, n_steps, n_props, n_part, dt, &
                           dx_val, dy_val, dz_val, mld_val, &
                           idx_age, idx_zmin, idx_zmax, idx_uvexp2, idx_uvexp10, &
                           idx_ml_ages, idx_ml_aged, idx_sfc_ages, idx_sfc_aged


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

    REAL(4), ALLOCATABLE :: host_data(:)
    REAL(4), ALLOCATABLE :: disph(:,:,:), dispv(:,:,:), dispt(:,:,:)
    INTEGER(1), ALLOCATABLE :: rmvmask(:,:)
    REAL(4), ALLOCATABLE :: zlev(:), mld(:,:), props(:,:)
    INTEGER(1), ALLOCATABLE :: alive(:)
    
    ! Grid arrays
    REAL(4), ALLOCATABLE :: dx(:,:), dy(:,:), dz(:)
    REAL(4), ALLOCATABLE :: idx0(:,:), idy0(:,:), idz0(:), dz_ref(:,:,:)

    INTEGER :: sz_field
    
    ! --- 1. Init Data ---
    ALLOCATE(host_data((ni+2)*(nj+2)*(nk+2)*3)); host_data = 1.0
    ALLOCATE(disph(0:ni+1,0:nj+1,0:nk+1)); disph = 0.1
    ALLOCATE(dispv(0:ni+1,0:nj+1,0:nk+1)); dispv = 0.1
    ALLOCATE(dispt(0:ni+1,0:nj+1,0:nk+1)); dispt = 0.01
    ALLOCATE(rmvmask(ni,nj)); rmvmask = 0
    
    ALLOCATE(dx(ni,nj)); dx = dx_val
    ALLOCATE(dy(ni,nj)); dy = dy_val
    ALLOCATE(dz(nk)); dz = dz_val

    ! Init Helper Grid Arrays (Inverse metrics)
    ALLOCATE(idx0(ni+1,nj)); idx0 = 1.0/dx_val
    ALLOCATE(idy0(ni,nj+1)); idy0 = 1.0/dy_val
    ALLOCATE(idz0(nk+1)); idz0 = 1.0/dz_val
    ALLOCATE(dz_ref(ni,nj,nk)); dz_ref = dz_val

    ALLOCATE(zlev(nk)); zlev = -10.0
    ALLOCATE(mld(ni,nj)); mld = mld_val
    ALLOCATE(props(n_props, n_part)); props = 0.0
    ALLOCATE(alive(n_part)); alive = 1_1

    ! --- 2. Setup Module Pointers ---
    ! Pass DT and constants to the module
    DT_MOD = dt
    DX_AVG = dx_val; DY_AVG = dy_val; DZ_AVG = dz_val
    
    ! Upload the grid data to the module's GPU pointers
    PRINT *, "--- [Main] Init Grid Device ---"
    CALL init_grid_device(dx, dy, dz, idx0, idy0, idz0, dz_ref)

    ! --- 3. Run Simulation ---
    PRINT *, "--- [Main] Calling RK4 ---"
    CALL rk4(host_data, disph, dispv, dispt, rmvmask, &
             zlev, mld, props, alive, &
             ni, nj, nk, n_steps, n_props, n_part, &
             1, 2, 3, 4, 5, 6, 7, 8, 9) ! Passing indices

    ! --- 4. Cleanup ---
    CALL free_grid_device()
    DEALLOCATE(host_data, disph, dispv, dispt, rmvmask, zlev, mld, props, alive)
    DEALLOCATE(dx, dy, dz, idx0, idy0, idz0, dz_ref)

    PRINT *, "--- Done ---"

END PROGRAM main_driver