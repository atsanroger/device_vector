MODULE Particle_Tracker_Mod
    USE OpenACC
    USE Device_Vector 
    USE Physics_Routines_Mod      
    USE Boundary_Logic_Mod
    USE Random_Mod          
    USE Mask_Manager_Mod
    IMPLICIT NONE

    ! --- Module Pointers ---
    REAL(4), POINTER, CONTIGUOUS :: p_dx(:), p_dy(:), p_dz(:)
    REAL(4), POINTER, CONTIGUOUS :: p_idx0(:), p_idy0(:), p_idz0(:), p_dz_ref(:)

    ! Device Vectors
    TYPE(device_vector_r4_t), PRIVATE :: dv_dx, dv_dy, dv_dz
    TYPE(device_vector_r4_t), PRIVATE :: dv_idx0, dv_idy0, dv_idz0, dv_dz_ref

    REAL(4), PUBLIC :: DT_MOD
    REAL(4), PUBLIC :: DX_AVG=100.0, DY_AVG=100.0, DZ_AVG=10.0

    PRIVATE
    PUBLIC :: rk4, init_grid_device, free_grid_device

CONTAINS

    SUBROUTINE init_grid_device(dx_h, dy_h, dz_h, idx0_h, idy0_h, idz0_h, dz_ref_h)
        REAL(4), INTENT(IN) :: dx_h(:,:), dy_h(:,:), dz_h(:)
        REAL(4), INTENT(IN) :: idx0_h(:,:), idy0_h(:,:), idz0_h(:)
        REAL(4), INTENT(IN) :: dz_ref_h(:,:,:)

        REAL(4), ALLOCATABLE :: tmp_flat(:)

        ! 1. DX
        ALLOCATE(tmp_flat(SIZE(dx_h)))
        tmp_flat = RESHAPE(dx_h, [SIZE(dx_h)])
        CALL dv_dx%create_buffer(INT(SIZE(dx_h), 8))
        CALL dv_dx%upload(tmp_flat)
        CALL dv_dx%acc_map(p_dx)
        DEALLOCATE(tmp_flat)

        ! 2. DY
        ALLOCATE(tmp_flat(SIZE(dy_h)))
        tmp_flat = RESHAPE(dy_h, [SIZE(dy_h)])
        CALL dv_dy%create_buffer(INT(SIZE(dy_h), 8))
        CALL dv_dy%upload(tmp_flat)
        CALL dv_dy%acc_map(p_dy)
        DEALLOCATE(tmp_flat)

        ! 3. DZ
        ALLOCATE(tmp_flat(SIZE(dz_h)))
        tmp_flat = dz_h
        CALL dv_dz%create_buffer(INT(SIZE(dz_h), 8))
        CALL dv_dz%upload(tmp_flat)
        CALL dv_dz%acc_map(p_dz)
        DEALLOCATE(tmp_flat)

        ! 4. IDX0
        ALLOCATE(tmp_flat(SIZE(idx0_h)))
        tmp_flat = RESHAPE(idx0_h, [SIZE(idx0_h)])
        CALL dv_idx0%create_buffer(INT(SIZE(idx0_h), 8))
        CALL dv_idx0%upload(tmp_flat)
        CALL dv_idx0%acc_map(p_idx0)
        DEALLOCATE(tmp_flat)

        ! 5. IDY0
        ALLOCATE(tmp_flat(SIZE(idy0_h)))
        tmp_flat = RESHAPE(idy0_h, [SIZE(idy0_h)])
        CALL dv_idy0%create_buffer(INT(SIZE(idy0_h), 8))
        CALL dv_idy0%upload(tmp_flat)
        CALL dv_idy0%acc_map(p_idy0)
        DEALLOCATE(tmp_flat)

        ! 6. IDZ0
        ALLOCATE(tmp_flat(SIZE(idz0_h)))
        tmp_flat = idz0_h
        CALL dv_idz0%create_buffer(INT(SIZE(idz0_h), 8))
        CALL dv_idz0%upload(tmp_flat)
        CALL dv_idz0%acc_map(p_idz0)
        DEALLOCATE(tmp_flat)

        ! 7. DZ_REF (3D -> 1D)
        ALLOCATE(tmp_flat(SIZE(dz_ref_h)))
        tmp_flat = RESHAPE(dz_ref_h, [SIZE(dz_ref_h)])
        CALL dv_dz_ref%create_buffer(INT(SIZE(dz_ref_h), 8))
        CALL dv_dz_ref%upload(tmp_flat)
        CALL dv_dz_ref%acc_map(p_dz_ref)
        DEALLOCATE(tmp_flat)

    END SUBROUTINE init_grid_device

    SUBROUTINE free_grid_device()
        CALL dv_dx%free(); CALL dv_dy%free(); CALL dv_dz%free()
        CALL dv_idx0%free(); CALL dv_idy0%free(); CALL dv_idz0%free(); CALL dv_dz_ref%free()
    END SUBROUTINE

    ! =========================================================
    ! RK4 主程式
    ! =========================================================
    SUBROUTINE rk4(host_data, disph_in, dispv_in, dispt_in, rmvmask_in, &
                   zlev_in, mld_in, props_in, alive_in, &                
                   ni, nj, nk, n_steps, n_props, n_part, &               
                   idx_age, idx_zmin, idx_zmax, idx_uvexp2, idx_uvexp10, & 
                   idx_ml_ages, idx_ml_aged, idx_sfc_ages, idx_sfc_aged)       

        INTEGER, INTENT(IN) :: ni, nj, nk, n_steps, n_props, n_part
        REAL(4), INTENT(IN) :: host_data(:)
        REAL(4), INTENT(IN) :: disph_in(0:ni+1, 0:nj+1, 0:nk+1)
        REAL(4), INTENT(IN) :: dispv_in(0:ni+1, 0:nj+1, 0:nk+1)
        REAL(4), INTENT(IN) :: dispt_in(0:ni+1, 0:nj+1, 0:nk+1)
        INTEGER(1), INTENT(IN) :: rmvmask_in(ni, nj)
        REAL(4), INTENT(IN) :: zlev_in(nk), mld_in(ni, nj)
        REAL(4), INTENT(IN) :: props_in(n_props, n_part)
        INTEGER(1), INTENT(IN) :: alive_in(n_part)
        INTEGER, INTENT(IN) :: idx_age, idx_zmin, idx_zmax, idx_uvexp2, idx_uvexp10
        INTEGER, INTENT(IN) :: idx_ml_ages, idx_ml_aged, idx_sfc_ages, idx_sfc_aged

        TYPE(device_vector_r4_t) :: dv_f1d, dv_disph, dv_dispv, dv_dispt
        TYPE(device_vector_r4_t) :: dv_zlev, dv_mld, dv_props
        TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
        
        REAL(4), POINTER, CONTIGUOUS :: pf1d(:), p_disph(:), p_dispv(:), p_dispt(:)
        REAL(4), POINTER, CONTIGUOUS :: p_zlev(:), p_mld(:), p_props(:)
        REAL(4), POINTER, CONTIGUOUS :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
        INTEGER(8) :: n
        
        ! [解決方案] 宣告臨時陣列
        REAL(4), ALLOCATABLE :: tmp_flat(:)

        ! --- Upload Data ---
        
        ! host_data
        ALLOCATE(tmp_flat(SIZE(host_data)))
        tmp_flat = host_data
        CALL dv_f1d%create_buffer(INT(SIZE(host_data), 8))
        CALL dv_f1d%upload(tmp_flat)
        CALL dv_f1d%acc_map(pf1d)
        DEALLOCATE(tmp_flat)

        ! disph
        ALLOCATE(tmp_flat(SIZE(disph_in)))
        tmp_flat = RESHAPE(disph_in, [SIZE(disph_in)])
        CALL dv_disph%create_buffer(INT(SIZE(disph_in), 8))
        CALL dv_disph%upload(tmp_flat)
        CALL dv_disph%acc_map(p_disph)
        DEALLOCATE(tmp_flat)

        ! dispv
        ALLOCATE(tmp_flat(SIZE(dispv_in)))
        tmp_flat = RESHAPE(dispv_in, [SIZE(dispv_in)])
        CALL dv_dispv%create_buffer(INT(SIZE(dispv_in), 8))
        CALL dv_dispv%upload(tmp_flat)
        CALL dv_dispv%acc_map(p_dispv)
        DEALLOCATE(tmp_flat)

        ! dispt
        ALLOCATE(tmp_flat(SIZE(dispt_in)))
        tmp_flat = RESHAPE(dispt_in, [SIZE(dispt_in)])
        CALL dv_dispt%create_buffer(INT(SIZE(dispt_in), 8))
        CALL dv_dispt%upload(tmp_flat)
        CALL dv_dispt%acc_map(p_dispt)
        DEALLOCATE(tmp_flat)

        ! zlev
        ALLOCATE(tmp_flat(SIZE(zlev_in)))
        tmp_flat = zlev_in
        CALL dv_zlev%create_buffer(INT(SIZE(zlev_in), 8))
        CALL dv_zlev%upload(tmp_flat)
        CALL dv_zlev%acc_map(p_zlev)
        DEALLOCATE(tmp_flat)

        ! mld
        ALLOCATE(tmp_flat(SIZE(mld_in)))
        tmp_flat = RESHAPE(mld_in, [SIZE(mld_in)])
        CALL dv_mld%create_buffer(INT(SIZE(mld_in), 8))
        CALL dv_mld%upload(tmp_flat)
        CALL dv_mld%acc_map(p_mld)
        DEALLOCATE(tmp_flat)

        ! props
        ALLOCATE(tmp_flat(SIZE(props_in)))
        tmp_flat = RESHAPE(props_in, [SIZE(props_in)])
        CALL dv_props%create_buffer(INT(SIZE(props_in), 8))
        CALL dv_props%upload(tmp_flat)
        CALL dv_props%acc_map(p_props)
        DEALLOCATE(tmp_flat)

        CALL px%create_buffer(INT(n_part, 8)); CALL px%acc_map(ax)
        CALL py%create_buffer(INT(n_part, 8)); CALL py%acc_map(ay)
        CALL pz%create_buffer(INT(n_part, 8)); CALL pz%acc_map(az)
        CALL vx%create_buffer(INT(n_part, 8)); CALL vx%acc_map(aux)
        CALL vy%create_buffer(INT(n_part, 8)); CALL vy%acc_map(auy) 
        CALL vz%create_buffer(INT(n_part, 8)); CALL vz%acc_map(auz)

        ! Init
        !$acc parallel loop present(ax, ay, az, aux, auy, auz)
        DO n = 1, n_part
            ax(n)=32.5; ay(n)=32.5; az(n)=32.5; aux(n)=1.0; auy(n)=0.0; auz(n)=0.0
        END DO

        CALL rk4_time_integration( &
            n_part, n_steps, DT_MOD, &
            ni, nj, nk, &
            DX_AVG, DY_AVG, DZ_AVG, &
            pf1d, p_disph, p_dispv, p_dispt, &
            p_zlev, p_mld, p_props, n_props, &
            ax, ay, az, &
            p_dx, p_dy, p_dz, p_idx0, p_idy0, p_idz0, p_dz_ref, &
            idx_age)

        ! Cleanup
        CALL dv_f1d%free(); CALL dv_disph%free(); CALL dv_dispv%free(); CALL dv_dispt%free()
        CALL dv_zlev%free(); CALL dv_mld%free(); CALL dv_props%free()
        CALL px%free(); CALL py%free(); CALL pz%free(); CALL vx%free(); CALL vy%free(); CALL vz%free()
    END SUBROUTINE rk4

END MODULE Particle_Tracker_Mod