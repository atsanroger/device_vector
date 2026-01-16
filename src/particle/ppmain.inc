SUBROUTINE rk4(host_data, disph_in, dispv_in, dispt_in, rmvmask_in, &
               zlev_in, mld_in, props_in, alive_in, &                
               ni, nj, nk, n_steps, n_props, n_part, &               
               idx_age, idx_zmin, idx_zmax, idx_uvexp2, idx_uvexp10, & 
               idx_ml_ages, idx_ml_aged, idx_sfc_ages, idx_sfc_aged)       
               
    USE Device_Vector
    IMPLICIT NONE
    
    ! --- [Input] 從 Main 傳進來的 CPU 資料 ---
    INTEGER, INTENT(IN) :: ni, nj, nk, n_steps, n_props, n_part

    REAL(4), INTENT(IN) :: host_data(:)
    REAL(4), INTENT(IN) :: disph_in(0:ni+1, 0:nj+1, 0:nk+1)
    REAL(4), INTENT(IN) :: dispv_in(0:ni+1, 0:nj+1, 0:nk+1)
    REAL(4), INTENT(IN) :: dispt_in(0:ni+1, 0:nj+1, 0:nk+1)
    INTEGER(1), INTENT(IN) :: rmvmask_in(ni, nj)

    REAL(4), INTENT(IN) :: zlev_in(nk)           
    REAL(4), INTENT(IN) :: mld_in(ni, nj)        
    REAL(4), INTENT(IN) :: props_in(n_props, n_part) 
    INTEGER(1), INTENT(IN) :: alive_in(n_part)      
    
    INTEGER, INTENT(IN) :: idx_age, idx_zmin, idx_zmax, idx_uvexp2, idx_uvexp10
    INTEGER, INTENT(IN) :: idx_ml_ages, idx_ml_aged, idx_sfc_ages, idx_sfc_aged
    ! --- Device Vectors (GPU 物件) ---
    ! 1. 流場 (已存在)
    TYPE(device_vector_r4_t) :: dv_f1d
    REAL(4), POINTER, CONTIGUOUS :: pf1d(:)

    ! 2. 輔助陣列 (已存在)
    TYPE(device_vector_r4_t) :: dv_disph, dv_dispv, dv_dispt, dv_rmvmask
    REAL(4), POINTER, CONTIGUOUS :: p_disph(:), p_dispv(:), p_dispt(:), p_rmvmask(:)    
    
    ! 【新增】3. 環境與屬性 Device Vectors
    TYPE(device_vector_r4_t) :: dv_zlev, dv_mld, dv_props
    !TYPE(device_vector_i1_t) :: dv_alive
    
    REAL(4), POINTER, CONTIGUOUS :: p_zlev(:), p_mld(:), p_props(:)
    INTEGER(4), POINTER, CONTIGUOUS :: p_alive(:)
    
    ! 4. PARTICLE 位置 (已存在)
    TYPE(device_vector_r4_t) :: px, py, pz
    REAL(4), POINTER, CONTIGUOUS :: ax(:), ay(:), az(:)
    
    INTEGER(8) :: n, idx_1d, prop_offset
    INTEGER :: i, j, k, ip, jp, kp
    REAL(4) :: xpos, ypos, zpos, xpos_rel, ypos_rel, zpos_rel
    REAL(4) :: z_phys, mld_val
    REAL(4) :: z_term, val_grid_z, days_inc

    ! RK4 TEMP
    REAL(4) :: start_x, start_y, start_z
    REAL(4) :: tk1x, tk1y, tk1z, tk2x, tk2y, tk2z
    REAL(4) :: tk3x, tk3y, tk3z, tk4x, tk4y, tk4z
    REAL(4) :: fx, fy, fz, wx, wy, wz, w00, w10, w0, w1
    INTEGER(8) :: ii, jj, kk, off8

    INTEGER(1) :: m_curr
    REAL(4) :: d_val, idx_n, idx_p, idy_n, idy_p
    REAL(4) :: val_dz, val_idx_now, val_idx_nxt, val_idx_prv, val_ref
    LOGICAL :: i_chg, j_chg

    ! --- 擴散與隨機數專用變數 ---
    INTEGER :: n_sub, m_steps, nsplit_local
    INTEGER :: my_seed !
    INTEGER(8) :: sz_layer, idx_base, idx_k, idx_km1, idx_kp1
    REAL(4) :: kt(0:2), kt_max, f_step, dtsplit
    REAL(4) :: val_k, val_km1, val_kp1, term_bot, term_top
    REAL(4) :: idx0_sq, term_diff, rnd_val

    ! --- (A) 建立與載入流場 (f1d) ---
    sz_field = SIZE(host_data)
    CALL dv_f1d%create_buffer(sz_field)
    CALL dv_f1d%acc_map(pf1d)
    
    !$acc parallel loop present(pf1d) copyin(host_data)
    DO n = 1, sz_field
       pf1d(n) = host_data(n)
    END DO

    ! --- (B) 建立與載入擴散係數 (3D -> 1D Flattening) ---
    sz_grid_3d = (ni+2_8) * (nj+2_8) * (nk+2_8)
    CALL dv_disph%create_buffer(sz_grid_3d); CALL dv_disph%acc_map(p_disph)
    CALL dv_dispv%create_buffer(sz_grid_3d); CALL dv_dispv%acc_map(p_dispv)
    CALL dv_dispt%create_buffer(sz_grid_3d); CALL dv_dispt%acc_map(p_dispt)

    ! 把 CPU 的 3D 陣列壓扁塞入 GPU
    !$acc parallel loop collapse(3) present(p_disph, p_dispv, p_dispt) &
    !$acc& copyin(disph_in, dispv_in, dispt_in)
    DO k = 0, nk+1
       DO j = 0, nj+1
          DO i = 0, ni+1
             ! 算出 1D 索引 (跟 RK4 裡的 off8 邏輯要一致)
             idx_1d = k * (nj+2) * (ni+2) + j * (ni+2) + i + 1
             p_disph(idx_1d) = disph_in(i,j,k)
             p_dispv(idx_1d) = dispv_in(i,j,k)
             p_dispt(idx_1d) = dispt_in(i,j,k)
          END DO
       END DO
    END DO

    ! --- (C) 建立與載入 Mask (2D -> 1D) ---
    sz_mask_2d = ni * nj
    CALL dv_rmvmask%create_buffer(sz_mask_2d)
    CALL dv_rmvmask%acc_map(p_rmvmask)

    !$acc parallel loop collapse(2) present(p_rmvmask) copyin(rmvmask_in)
    DO j = 1, nj
       DO i = 1, ni
          idx_1d = (j-1)*ni + i
          p_rmvmask(idx_1d) = REAL(rmvmask_in(i,j)) ! 轉成 REAL 存
       END DO
    END DO

    ! --- (D) 建立粒子 ---
    CALL px%create_buffer(n_part); CALL px%acc_map(ax)
    CALL py%create_buffer(n_part); CALL py%acc_map(ay)
    CALL pz%create_buffer(n_part); CALL pz%acc_map(az)
    CALL vx%create_buffer(n_part); CALL vx%acc_map(aux)
    CALL vy%create_buffer(n_part); CALL vy%acc_map(auy)
    CALL vz%create_buffer(n_part); CALL vz%acc_map(auz)

    ! --- (E) 載入 Zlev (1D) ---
    CALL dv_zlev%create_buffer(nk)
    CALL dv_zlev%acc_map(p_zlev)
    !$acc parallel loop present(p_zlev) copyin(zlev_in)
    DO k = 1, nk
       p_zlev(k) = zlev_in(k)
    END DO

    ! --- (F) 載入 MLD (2D -> 1D) ---
    CALL dv_mld%create_buffer(ni * nj)
    CALL dv_mld%acc_map(p_mld)
    !$acc parallel loop collapse(2) present(p_mld) copyin(mld_in)
    DO j = 1, nj
       DO i = 1, ni
          idx_1d = (j-1)*ni + i
          p_mld(idx_1d) = mld_in(i,j)
       END DO
    END DO

    ! --- (G) 載入 Properties (2D -> 1D Flattened) ---
    ! 假設 props 是 (n_props, n_part)
    CALL dv_props%create_buffer(SIZE(props_in))
    CALL dv_props%acc_map(p_props)
    !$acc parallel loop present(p_props) copyin(props_in)
    DO n = 1, SIZE(props_in)
       p_props(n) = props_in(n, 1)
    END DO

    ! 3. 初始化粒子
    !$acc parallel loop present(ax, ay, az, aux, auy, auz)
    DO n = 1, n_part
        ax(n)=32.5; ay(n)=32.5; az(n)=32.5; aux(n)=1.0; auy(n)=0.0; auz(n)=0.0
    END DO
    CALL device_synchronize() 

    ! 4. 時間積分主迴圈 (RK4 內嵌)
    !$acc data present(pf1d, ax, ay, az)
    DO i_step = 1, N_S
        
        !$acc parallel loop gang vector_length(WARP_LENGTH) &
        !$acc private(xpos, ypos, zpos, start_x, start_y, start_z) &
        !$acc private(fx, fy, fz, ii, jj, kk, wx, wy, wz, off8, w00, w10, w0, w1) &
        !$acc private(tk1x, tk1y, tk1z, tk2x, tk2y, tk2z) &
        !$acc private(tk3x, tk3y, tk3z, tk4x, tk4y, tk4z)

        DO n = 1, n_part

            IF (p_alive(n) == 0_1) CYCLE
            ! ================= RK4 START =================
            ! 1. 紀錄起點
            start_x = ax(n); 
            start_y = ay(n); 
            start_z = az(n)

            ! ---------------------------------------------
            ! Step 1: K1
            ! ---------------------------------------------
            CALL get_field_deriv(start_x, start_y, start_z, pf1d, &
                                 DX, DY, DZ, GX, GY, GZ, GXY, GXYZ, &
                                 tk1x, tk1y, tk1z)
            ! ---------------------------------------------
            ! Step 2: K2
            ! ---------------------------------------------
            xpos = start_x + 0.5_4 * DT * tk1x
            ypos = start_y + 0.5_4 * DT * tk1y
            zpos = start_z + 0.5_4 * DT * tk1z
            CALL get_field_deriv(xpos, ypos, zpos, pf1d, &
                                 DX, DY, DZ, GX, GY, GZ, GXY, GXYZ, &
                                 tk2x, tk2y, tk2z)
            ! ---------------------------------------------
            ! Step 3: K3
            ! ---------------------------------------------
            xpos = start_x + 0.5_4 * DT * tk2x
            ypos = start_y + 0.5_4 * DT * tk2y
            zpos = start_z + 0.5_4 * DT * tk2z
            CALL get_field_deriv(xpos, ypos, zpos, pf1d, &
                                 DX, DY, DZ, GX, GY, GZ, GXY, GXYZ, &
                                 tk3x, tk3y, tk3z)
            ! ---------------------------------------------
            ! Step 4: K4
            ! ---------------------------------------------
            xpos = start_x + DT * tk3x
            ypos = start_y + DT * tk3y
            zpos = start_z + DT * tk3z
            CALL get_field_deriv(xpos, ypos, zpos, pf1d, &
                                 DX, DY, DZ, GX, GY, GZ, GXY, GXYZ, &
                                 tk4x, tk4y, tk4z)
            ! ---------------------------------------------
            ! 更新位置
            ! ---------------------------------------------
            xpos = start_x + (DT/6.0_4) * (tk1x + 2.0_4*tk2x + 2.0_4*tk3x + tk4x)
            ypos = start_y + (DT/6.0_4) * (tk1y + 2.0_4*tk2y + 2.0_4*tk3y + tk4y)
            zpos = start_z + (DT/6.0_4) * (tk1z + 2.0_4*tk2z + 2.0_4*tk3z + tk4z)
            ip = INT(xpos/DX)
            jp = INT(ypos/DY)
            kp = INT(zpos/DZ)

            xpos_rel = (xpos / DX) - REAL(ip, 4)
            ypos_rel = (ypos / DY) - REAL(jp, 4)
            zpos_rel = (zpos / DZ) - REAL(kp, 4)

            m_curr = mask_packed(ip, jp, kp)

            IF (BTEST(m_curr, 7)) THEN 
               
               nsplit_local = 5 
               dtsplit = DT / REAL(nsplit_local, 4)
               n_sub = 0
               
               sz_layer = (ni + 2) * (nj + 2)
               idx_base = jp * (ni + 2) + ip + 1  ! 基礎平面偏移

               my_seed = IEOR(INT(n, 4), INT(i_step, 4) * 1664525)
               IF (my_seed == 0) my_seed = 123456789

               DO WHILE (n_sub < nsplit_local)
                  
                  idx_k   = kp * sz_layer + idx_base
                  idx_km1 = (kp-1) * sz_layer + idx_base
                  idx_kp1 = (kp+1) * sz_layer + idx_base
                  
                  val_k   = p_dispt(idx_k)
                  val_km1 = p_dispt(idx_km1)
                  val_kp1 = p_dispt(idx_kp1)
                  
                  !    Bit 5: -Z (wmask(k-1)), Bit 4: +Z (wmask(k))
                  term_bot = MERGE(1.0_4, 0.0_4, BTEST(m_curr, 5))
                  term_top = MERGE(1.0_4, 0.0_4, BTEST(m_curr, 4))
                  
                  idx0_sq = idz0(kp)**2
                  
                  kt(0) = 0.5_4 * term_bot * (val_k + val_km1) * dtsplit * idx0_sq
                  kt(1) =         val_k                        * dtsplit * idx0_sq
                  kt(2) = 0.5_4 * term_top * (val_k + val_kp1) * dtsplit * idx0_sq

                  kt_max = max(kt(0), kt(1), kt(2))
                  IF (kt_max <= 1.0E-6_4) THEN
                     m_steps = nsplit_local
                  ELSE
                     m_steps = INT(0.01_4 / kt_max)
                  END IF
                  m_steps = max(min(m_steps, nsplit_local - n_sub), 1)
                  
                  f_step = REAL(m_steps, 4)
                  kt(0) = kt(0) * f_step
                  kt(2) = kt(2) * f_step
                  
                  rnd_val = nrand(my_seed)  

                  term_diff = sqrt(2.0_4 * max((1.0_4 - zpos_rel)*kt(0) + zpos_rel*kt(2), 0.0_4))
                  
                  zpos_rel = zpos_rel + rnd_val * term_diff
                  zpos_rel = zpos_rel + (kt(2) - kt(0))
                  
                  val_dz      = dz(kp)
                  val_idx_now = idz0(kp)
                  val_idx_nxt = idz0(kp+1)
                  val_idx_prv = idz0(kp-1)
                  val_ref     = dz_ref(ip, jp, kp)

                  CALL cross_z_branchless(kp, zpos_rel, m_curr, val_dz, val_idx_now, &
                                          val_idx_nxt, val_idx_prv, val_ref, &
                                          reflc_sfc, reflc_btm, repul_sfc, repul_btm)
                                          
                  n_sub = n_sub + m_steps
                  
               END DO
            END IF

            ! --- X Check ---
            IF (xpos_rel >= 1.0_4 .OR. xpos_rel < 0.0_4) THEN
              d_val = dx(ip, jp)
              idx_n = idx0(ip+1, jp); idx_p = idx0(ip-1, jp)
              CALL cross_x_branchless(ip, xpos_rel, m_curr, d_val, idx_n, idx_p, reflc, repul)
              i_chg = .TRUE.
            ELSE
              i_chg = .FALSE.
            END IF

            ! --- Y Check ---
            IF (i_chg) m_curr = mask_packed(ip, jp, kp)
            IF (ypos_rel >= 1.0_4 .OR. ypos_rel < 0.0_4) THEN
              d_val = dy(ip, jp)
              idy_n = idy0(ip, jp+1); idy_p = idy0(ip, jp-1)
              CALL cross_y_branchless(jp, ypos_rel, m_curr, d_val, idy_n, idy_p, reflc, repul)
              j_chg = .TRUE.
            ELSE
              j_chg = .FALSE.
            END IF

            ! --- Z Check ---
            IF (j_chg) m_curr = mask_packed(ip, jp, kp)
            IF (zpos_rel >= 1.0_4 .OR. zpos_rel < 0.0_4) THEN
                val_dz      = dz(kp)
                val_idx_now = idz0(kp)
                val_idx_nxt = idz0(kp+1)
                val_idx_prv = idz0(kp-1)
                val_ref     = dz_ref(ip, jp, kp) !
            
                CALL cross_z_branchless(kp, zpos_rel, m_curr, val_dz, val_idx_now, &
                                        val_idx_nxt, val_idx_prv, val_ref, &
                                        reflc_sfc, reflc_btm, repul_sfc, repul_btm)

            END IF

            ax(n) = (REAL(ip, 4) + xpos_rel) * DX
            ay(n) = (REAL(jp, 4) + ypos_rel) * DY
            az(n) = (REAL(kp, 4) + zpos_rel) * DZ

            prop_offset = (n - 1) * n_props
            z_term = p_zlev(kp) - (1.0_4 - zpos_rel) * dz(kp)

            val_grid_z = REAL(kp, 4) - 1.0_4 + zpos_rel

            IF (idx_age > 0) THEN
                p_props(prop_offset + idx_age) = p_props(prop_offset + idx_age) + DT
            END IF

            IF (idx_uvexp2 > 0) THEN
               p_props(prop_offset + idx_uvexp2) = p_props(prop_offset + idx_uvexp2) + &
                                                   DT * EXP(1.05_4 * z_term)
            END IF
            
            IF (idx_uvexp10 > 0) THEN
               p_props(prop_offset + idx_uvexp10) = p_props(prop_offset + idx_uvexp10) + &
                                                    DT * EXP(0.23_4 * z_term)
            END IF
            
            IF (idx_zmin > 0) THEN
               p_props(prop_offset + idx_zmin) = min(p_props(prop_offset + idx_zmin), val_grid_z)
            END IF
            
            IF (idx_zmax > 0) THEN
               p_props(prop_offset + idx_zmax) = max(p_props(prop_offset + idx_zmax), val_grid_z)
            END IF

            idx_1d = (jp-1)*ni + ip
            
            IF (p_mld(idx_1d) > -z_term) THEN
               
               IF (idx_ml_ages > 0) THEN
                  ! 1. 累積秒數
                  p_props(prop_offset + idx_ml_ages) = p_props(prop_offset + idx_ml_ages) + DT
                  
                  IF (idx_ml_aged > 0) THEN
                     days_inc = INT(p_props(prop_offset + idx_ml_ages) / 86400.0_4)
                     
                     IF (days_inc > 0.0_4) THEN
                        p_props(prop_offset + idx_ml_aged) = p_props(prop_offset + idx_ml_aged) + days_inc
                        p_props(prop_offset + idx_ml_ages) = p_props(prop_offset + idx_ml_ages) - (days_inc * 86400.0_4)
                     END IF
                  END IF

               ELSE IF (idx_ml_aged > 0) THEN
                  p_props(prop_offset + idx_ml_aged) = p_props(prop_offset + idx_ml_aged) + (DT / 86400.0_4)
               END IF
            END IF

            IF (kp == nk) THEN
               
               IF (idx_sfc_ages > 0) THEN
                  p_props(prop_offset + idx_sfc_ages) = p_props(prop_offset + idx_sfc_ages) + DT
                  
                  IF (idx_sfc_aged > 0) THEN
                     days_inc = INT(p_props(prop_offset + idx_sfc_ages) / 86400.0_4)
                     
                     IF (days_inc > 0.0_4) THEN
                        p_props(prop_offset + idx_sfc_aged) = p_props(prop_offset + idx_sfc_aged) + days_inc
                        p_props(prop_offset + idx_sfc_ages) = p_props(prop_offset + idx_sfc_ages) - (days_inc * 86400.0_4)
                     END IF
                  END IF

               ELSE IF (idx_sfc_aged > 0) THEN
                   p_props(prop_offset + idx_sfc_aged) = p_props(prop_offset + idx_sfc_aged) + (DT / 86400.0_4)
               END IF
            END IF

        END DO ! loop over particle

    END DO ! rk4 time loop
    !$acc end data

    CALL device_synchronize() 
    
    CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
    CALL vx%acc_unmap(); CALL vy%acc_unmap(); CALL vz%acc_unmap()
    CALL dv_f1d%free()
    CALL dv_disph%free(); 
    CALL dv_dispv%free(); 
    CALL dv_dispt%free()
    CALL dv_rmvmask%free()
    CALL dv_zlev%free()
    CALL dv_mld%free()
    CALL dv_props%free()    
    CALL px%free(); CALL py%free(); CALL pz%free()
    CALL vx%free(); CALL vy%free(); CALL vz%free()  

END SUBROUTINE rk4

!$acc routine seq
SUBROUTINE get_field_deriv(x_in, y_in, z_in, f1d_in, &
                           dx_val, dy_val, dz_val, &         ! 新增：網格間距
                           gx_val, gy_val, gz_val, &         ! 新增：網格點數
                           gxy_val, gxyz_val, &              ! 新增：偏移量
                           kx, ky, kz)
    
    IMPLICIT NONE

    ! --- 輸入 ---
    REAL(4), INTENT(IN) :: x_in, y_in, z_in
    REAL(4), INTENT(IN) :: f1d_in(*) 
    
    ! --- 新增的網格參數 (全部 INTENT(IN)) ---
    REAL(4), INTENT(IN) :: dx_val, dy_val, dz_val
    INTEGER, INTENT(IN) :: gx_val, gy_val, gz_val
    INTEGER(8), INTENT(IN) :: gxy_val, gxyz_val  ! 這裡要注意原本 GXY 的型別，通常是 INTEGER(8)

    ! --- 輸出 ---
    REAL(4), INTENT(OUT) :: kx, ky, kz

    ! --- 區域變數 ---
    REAL(4) :: fx, fy, fz, wx, wy, wz
    REAL(4) :: w00, w10, w0, w1
    INTEGER(8) :: ii, jj, kk, off8

    ! 1. 使用傳入的 dx_val, dy_val, dz_val
    fx = x_in / dx_val
    fy = y_in / dy_val
    fz = z_in / dz_val

    ii = INT(fx, 8)
    jj = INT(fy, 8)
    kk = INT(fz, 8)

    ! 2. 使用傳入的 gx_val... 進行邊界檢查
    ! 注意：這裡假設 gx_val 是整數，與 INT(8) 比較時會自動轉型，或建議轉型
    IF (ii < 0) ii = 0
    IF (ii > INT(gx_val, 8)-2) ii = INT(gx_val, 8)-2
    IF (jj < 0) jj = 0
    IF (jj > INT(gy_val, 8)-2) jj = INT(gy_val, 8)-2
    IF (kk < 0) kk = 0
    IF (kk > INT(gz_val, 8)-2) kk = INT(gz_val, 8)-2

    wx = fx - REAL(ii, 4)
    wy = fy - REAL(jj, 4)
    wz = fz - REAL(kk, 4)

    ! 3. 使用傳入的 gxy_val 計算偏移
    ! 注意：這裡要確認 gx_val 也要轉成 INT(8) 參與運算以防溢位
    off8 = kk * gxy_val + jj * INT(gx_val, 8) + ii + 1_8

    ! 4. 插值 (使用傳入的 gx_val, gxy_val, gxyz_val)
    ! GX -> INT(gx_val, 8)
    ! GXY -> gxy_val
    ! GXYZ -> gxyz_val
    
    ! --- U component ---
    w00 = f1d_in(off8)               * (1.-wx) + f1d_in(off8+1)               * wx
    w10 = f1d_in(off8+INT(gx_val,8)) * (1.-wx) + f1d_in(off8+INT(gx_val,8)+1) * wx
    w0  = w00 * (1.-wy) + w10 * wy
    
    w00 = f1d_in(off8+gxy_val)                * (1.-wx) + f1d_in(off8+gxy_val+1)                * wx
    w10 = f1d_in(off8+gxy_val+INT(gx_val,8))  * (1.-wx) + f1d_in(off8+gxy_val+INT(gx_val,8)+1)  * wx
    w1  = w00 * (1.-wy) + w10 * wy
    
    kx  = w0 * (1.-wz) + w1 * wz

    ! --- V component ---
    off8 = off8 + gxyz_val
    w00 = f1d_in(off8)               * (1.-wx) + f1d_in(off8+1)               * wx
    w10 = f1d_in(off8+INT(gx_val,8)) * (1.-wx) + f1d_in(off8+INT(gx_val,8)+1) * wx
    w0  = w00 * (1.-wy) + w10 * wy
    
    w00 = f1d_in(off8+gxy_val)                * (1.-wx) + f1d_in(off8+gxy_val+1)                * wx
    w10 = f1d_in(off8+gxy_val+INT(gx_val,8))  * (1.-wx) + f1d_in(off8+gxy_val+INT(gx_val,8)+1)  * wx
    w1  = w00 * (1.-wy) + w10 * wy
    
    ky  = w0 * (1.-wz) + w1 * wz

    ! --- W component ---
    off8 = off8 + gxyz_val
    w00 = f1d_in(off8)               * (1.-wx) + f1d_in(off8+1)               * wx
    w10 = f1d_in(off8+INT(gx_val,8)) * (1.-wx) + f1d_in(off8+INT(gx_val,8)+1) * wx
    w0  = w00 * (1.-wy) + w10 * wy
    
    w00 = f1d_in(off8+gxy_val)                * (1.-wx) + f1d_in(off8+gxy_val+1)                * wx
    w10 = f1d_in(off8+gxy_val+INT(gx_val,8))  * (1.-wx) + f1d_in(off8+gxy_val+INT(gx_val,8)+1)  * wx
    w1  = w00 * (1.-wy) + w10 * wy
    
    kz  = w0 * (1.-wz) + w1 * wz

END SUBROUTINE get_field_deriv