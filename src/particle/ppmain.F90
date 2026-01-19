MODULE Particle_Tracker_Mod
    USE OpenACC
    USE Device_Vector       
    USE Boundary_Logic_Mod  
    USE Random_Mod          
    USE Mask_Manager_Mod
    USE Physics_Routines_Mod 
    IMPLICIT NONE

    ! Device Vectors (保持全域，因為資料存在這裡)
    TYPE(device_vector_r4_t), PRIVATE :: dv_dx, dv_dy, dv_dz
    TYPE(device_vector_r4_t), PRIVATE :: dv_idx0, dv_idy0, dv_idz0, dv_dz_ref

    ! [修改] 移除全域 Pointer，避免跨 Subroutine 分析失敗
    ! REAL(4), POINTER ... (移除)

    REAL(4), PUBLIC :: DT_MOD
    REAL(4), PUBLIC :: DX_AVG=100.0, DY_AVG=100.0, DZ_AVG=10.0

    PRIVATE
    PUBLIC :: rk4, init_grid_device, free_grid_device

CONTAINS

    ! =========================================================
    ! 初始化：只負責分配記憶體與上傳 (模仿 profile_rk4_dv 的行為)
    ! =========================================================
    SUBROUTINE init_grid_device(dx_h, dy_h, dz_h, idx0_h, idy0_h, idz0_h, dz_ref_h)
        REAL(4), INTENT(IN) :: dx_h(:,:), dy_h(:,:), dz_h(:)
        REAL(4), INTENT(IN) :: idx0_h(:,:), idy0_h(:,:), idz0_h(:)
        REAL(4), INTENT(IN) :: dz_ref_h(:,:,:)
        
        REAL(4), ALLOCATABLE :: tmp_flat(:)

        ! 1. DX
        ALLOCATE(tmp_flat(SIZE(dx_h)))
        tmp_flat = RESHAPE(dx_h, [SIZE(dx_h)])
        CALL dv_dx%create_buffer(INT(SIZE(dx_h), 8))
        CALL dv_dx%upload(tmp_flat) ! 只上傳，不 Map
        DEALLOCATE(tmp_flat)

        ! 2. DY
        ALLOCATE(tmp_flat(SIZE(dy_h)))
        tmp_flat = RESHAPE(dy_h, [SIZE(dy_h)])
        CALL dv_dy%create_buffer(INT(SIZE(dy_h), 8))
        CALL dv_dy%upload(tmp_flat)
        DEALLOCATE(tmp_flat)

        ! 3. DZ
        ALLOCATE(tmp_flat(SIZE(dz_h)))
        tmp_flat = dz_h
        CALL dv_dz%create_buffer(INT(SIZE(dz_h), 8))
        CALL dv_dz%upload(tmp_flat)
        DEALLOCATE(tmp_flat)

        ! 4. IDX0
        ALLOCATE(tmp_flat(SIZE(idx0_h)))
        tmp_flat = RESHAPE(idx0_h, [SIZE(idx0_h)])
        CALL dv_idx0%create_buffer(INT(SIZE(idx0_h), 8))
        CALL dv_idx0%upload(tmp_flat)
        DEALLOCATE(tmp_flat)

        ! 5. IDY0
        ALLOCATE(tmp_flat(SIZE(idy0_h)))
        tmp_flat = RESHAPE(idy0_h, [SIZE(idy0_h)])
        CALL dv_idy0%create_buffer(INT(SIZE(idy0_h), 8))
        CALL dv_idy0%upload(tmp_flat)
        DEALLOCATE(tmp_flat)

        ! 6. IDZ0
        ALLOCATE(tmp_flat(SIZE(idz0_h)))
        tmp_flat = idz0_h
        CALL dv_idz0%create_buffer(INT(SIZE(idz0_h), 8))
        CALL dv_idz0%upload(tmp_flat)
        DEALLOCATE(tmp_flat)

        ! 7. DZ_REF
        ALLOCATE(tmp_flat(SIZE(dz_ref_h)))
        tmp_flat = RESHAPE(dz_ref_h, [SIZE(dz_ref_h)])
        CALL dv_dz_ref%create_buffer(INT(SIZE(dz_ref_h), 8))
        CALL dv_dz_ref%upload(tmp_flat)
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

        ! Local Device Vectors (Particles)
        TYPE(device_vector_r4_t) :: dv_f1d, dv_disph, dv_dispv, dv_dispt
        TYPE(device_vector_r4_t) :: dv_zlev, dv_mld, dv_props
        TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
        
        REAL(4), POINTER :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
        REAL(4), POINTER :: pf1d(:), p_disph(:), p_dispv(:), p_dispt(:)
        REAL(4), POINTER :: p_zlev(:), p_mld(:), p_props(:)
        
        ! Grid pointers (Local)
        REAL(4), POINTER :: l_dx(:), l_dy(:), l_dz(:)
        REAL(4), POINTER :: l_idx0(:), l_idy0(:), l_idz0(:), l_dz_ref(:)

        REAL(4), ALLOCATABLE :: tmp_flat(:)
        INTEGER(8) :: n

        ! --- Upload & Local Map ---
        
        ! Host Data
        ALLOCATE(tmp_flat(SIZE(host_data)))
        tmp_flat = host_data
        CALL dv_f1d%create_buffer(INT(SIZE(host_data), 8))
        CALL dv_f1d%upload(tmp_flat)
        CALL dv_f1d%acc_map(pf1d) ! Local Map
        DEALLOCATE(tmp_flat)

        ! Disph
        ALLOCATE(tmp_flat(SIZE(disph_in)))
        tmp_flat = RESHAPE(disph_in, [SIZE(disph_in)])
        CALL dv_disph%create_buffer(INT(SIZE(disph_in), 8))
        CALL dv_disph%upload(tmp_flat)
        CALL dv_disph%acc_map(p_disph) ! Local Map
        DEALLOCATE(tmp_flat)

        ! Dispv
        ALLOCATE(tmp_flat(SIZE(dispv_in)))
        tmp_flat = RESHAPE(dispv_in, [SIZE(dispv_in)])
        CALL dv_dispv%create_buffer(INT(SIZE(dispv_in), 8))
        CALL dv_dispv%upload(tmp_flat)
        CALL dv_dispv%acc_map(p_dispv) ! Local Map
        DEALLOCATE(tmp_flat)

        ! Dispt
        ALLOCATE(tmp_flat(SIZE(dispt_in)))
        tmp_flat = RESHAPE(dispt_in, [SIZE(dispt_in)])
        CALL dv_dispt%create_buffer(INT(SIZE(dispt_in), 8))
        CALL dv_dispt%upload(tmp_flat)
        CALL dv_dispt%acc_map(p_dispt) ! Local Map
        DEALLOCATE(tmp_flat)

        ! Zlev
        ALLOCATE(tmp_flat(SIZE(zlev_in)))
        tmp_flat = zlev_in
        CALL dv_zlev%create_buffer(INT(SIZE(zlev_in), 8))
        CALL dv_zlev%upload(tmp_flat)
        CALL dv_zlev%acc_map(p_zlev)
        DEALLOCATE(tmp_flat)

        ! Mld
        ALLOCATE(tmp_flat(SIZE(mld_in)))
        tmp_flat = RESHAPE(mld_in, [SIZE(mld_in)])
        CALL dv_mld%create_buffer(INT(SIZE(mld_in), 8))
        CALL dv_mld%upload(tmp_flat)
        CALL dv_mld%acc_map(p_mld)
        DEALLOCATE(tmp_flat)

        ! Props
        ALLOCATE(tmp_flat(SIZE(props_in)))
        tmp_flat = RESHAPE(props_in, [SIZE(props_in)])
        CALL dv_props%create_buffer(INT(SIZE(props_in), 8))
        CALL dv_props%upload(tmp_flat)
        CALL dv_props%acc_map(p_props)
        DEALLOCATE(tmp_flat)

        ! Particles (Create & Local Map)
        CALL px%create_buffer(INT(n_part, 8)); CALL px%acc_map(ax)
        CALL py%create_buffer(INT(n_part, 8)); CALL py%acc_map(ay)
        CALL pz%create_buffer(INT(n_part, 8)); CALL pz%acc_map(az)
        CALL vx%create_buffer(INT(n_part, 8)); CALL vx%acc_map(aux)
        CALL vy%create_buffer(INT(n_part, 8)); CALL vy%acc_map(auy)
        CALL vz%create_buffer(INT(n_part, 8)); CALL vz%acc_map(auz)

        ! --- [關鍵修復] 這裡進行 Grid 的 Local Map ---
        ! 因為這些 DV 是全域的，我們在這裡把這一次的區域指針綁上去
        ! 這樣編譯器就知道 l_dx 到底指向哪裡了
        CALL dv_dx%acc_map(l_dx)
        CALL dv_dy%acc_map(l_dy)
        CALL dv_dz%acc_map(l_dz)
        CALL dv_idx0%acc_map(l_idx0)
        CALL dv_idy0%acc_map(l_idy0)
        CALL dv_idz0%acc_map(l_idz0)
        CALL dv_dz_ref%acc_map(l_dz_ref)

        ! Init
        !$acc parallel loop present(ax, ay, az, aux, auy, auz)
        DO n = 1, n_part
            ax(n)=32.5; ay(n)=32.5; az(n)=32.5; aux(n)=1.0; auy(n)=0.0; auz(n)=0.0
        END DO

        ! --- 呼叫物理運算 ---
        ! 現在傳入的是剛剛才 Map 好的 Local Pointer
        ! 這完全模擬了 run_device_vector_ver2 的成功模式
        CALL rk4_time_integration( &
            n_part, n_steps, DT_MOD, &
            ni, nj, nk, &
            DX_AVG, DY_AVG, DZ_AVG, &
            pf1d, p_disph, p_dispv, p_dispt, &
            p_zlev, p_mld, p_props, n_props, &
            ax, ay, az, &
            l_dx, l_dy, l_dz, l_idx0, l_idy0, l_idz0, l_dz_ref, &
            idx_age)

        ! Cleanup
        CALL dv_f1d%free(); CALL dv_disph%free(); CALL dv_dispv%free(); CALL dv_dispt%free()
        CALL dv_zlev%free(); CALL dv_mld%free(); CALL dv_props%free()
        CALL px%free(); CALL py%free(); CALL pz%free(); CALL vx%free(); CALL vy%free(); CALL vz%free()
        
        ! Grid 不用 free，因為它們是 module level 的，要在 free_grid_device 釋放
    END SUBROUTINE rk4

END MODULE Particle_Tracker_Mod