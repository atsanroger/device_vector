MODULE Physics_Routines_Mod
    USE OpenACC
    USE Device_Vector       
    USE Boundary_Logic_Mod
    USE Random_Mod
    USE Mask_Manager_Mod 
    IMPLICIT NONE

    PRIVATE
    PUBLIC :: get_field_deriv, do_vertical_diffusion, rk4_time_integration

CONTAINS

    ! =========================================================
    ! RK4 時間積分核心 (已拆分為三個獨立的平行迴圈)
    ! =========================================================
    SUBROUTINE rk4_time_integration( &
        n_part, n_steps, dt, &
        ni, nj, nk, &
        dx_avg, dy_avg, dz_avg, &
        pf1d, p_disph, p_dispv, p_dispt, &
        p_zlev, p_mld, p_props, n_props, &
        ax, ay, az, &
        p_dx, p_dy, p_dz, p_idx0, p_idy0, p_idz0, p_dz_ref, &
        idx_age)

        ! --- Scalars ---
        INTEGER, INTENT(IN) :: n_part, n_steps, ni, nj, nk, n_props, idx_age
        REAL(4), INTENT(IN) :: dt, dx_avg, dy_avg, dz_avg

        ! --- Data Arrays (1D Views) ---
        REAL(4), INTENT(IN) :: pf1d(*)
        REAL(4), INTENT(IN) :: p_disph(*), p_dispv(*), p_dispt(*)
        REAL(4), INTENT(IN) :: p_zlev(*), p_mld(*)
        REAL(4), INTENT(INOUT):: p_props(*)
        
        ! --- Particles ---
        REAL(4), INTENT(INOUT) :: ax(*), ay(*), az(*)

        ! --- Grid Arrays (1D Pointers) ---
        REAL(4), INTENT(IN) :: p_dx(*), p_dy(*), p_dz(*)
        REAL(4), INTENT(IN) :: p_idx0(*), p_idy0(*), p_idz0(*), p_dz_ref(*)

        ! --- Locals ---
        INTEGER(8) :: n, idx_flat, prop_offset_8
        INTEGER :: i_step, ip, jp, kp, n_sub
        REAL(4) :: start_x, start_y, start_z, xpos, ypos, zpos
        REAL(4) :: tk1x, tk1y, tk1z, tk2x, tk2y, tk2z, tk3x, tk3y, tk3z, tk4x, tk4y, tk4z
        REAL(4) :: xpos_rel, ypos_rel, zpos_rel
        REAL(4) :: d_val, idx_n, idx_p, idy_n, idy_p
        REAL(4) :: val_dz, val_idx_now, val_idx_nxt, val_idx_prv, val_ref, z_term
        INTEGER(1) :: m_curr
        LOGICAL :: i_chg, j_chg
        REAL(4) :: reflc=1.0, repul=0.01

        !$acc data present(pf1d, p_disph, p_dispv, p_dispt, p_zlev, p_mld, p_props) &
        !$acc present(ax, ay, az) &
        !$acc present(p_dx, p_dy, p_dz, p_idx0, p_idy0, p_idz0, p_dz_ref)
        
        DO i_step = 1, n_steps
            
            ! -----------------------------------------------------------
            ! Loop 1: 純 RK4 積分 (Advection)
            ! -----------------------------------------------------------
            !$acc parallel loop gang vector &
            !$acc private(start_x, start_y, start_z, xpos, ypos, zpos) &
            !$acc private(tk1x, tk1y, tk1z, tk2x, tk2y, tk2z, tk3x, tk3y, tk3z, tk4x, tk4y, tk4z)
            DO n = 1, n_part
                start_x = ax(n); start_y = ay(n); start_z = az(n)
                
                CALL get_field_deriv(start_x, start_y, start_z, pf1d, dx_avg, dy_avg, dz_avg, ni, nj, nk, tk1x, tk1y, tk1z)
                
                xpos = start_x + 0.5*dt*tk1x; ypos = start_y + 0.5*dt*tk1y; zpos = start_z + 0.5*dt*tk1z
                CALL get_field_deriv(xpos, ypos, zpos, pf1d, dx_avg, dy_avg, dz_avg, ni, nj, nk, tk2x, tk2y, tk2z)
                
                xpos = start_x + 0.5*dt*tk2x; ypos = start_y + 0.5*dt*tk2y; zpos = start_z + 0.5*dt*tk2z
                CALL get_field_deriv(xpos, ypos, zpos, pf1d, dx_avg, dy_avg, dz_avg, ni, nj, nk, tk3x, tk3y, tk3z)
                
                xpos = start_x + dt*tk3x; ypos = start_y + dt*tk3y; zpos = start_z + dt*tk3z
                CALL get_field_deriv(xpos, ypos, zpos, pf1d, dx_avg, dy_avg, dz_avg, ni, nj, nk, tk4x, tk4y, tk4z)

                ax(n) = start_x + (dt/6.0)*(tk1x + 2.0*tk2x + 2.0*tk3x + tk4x)
                ay(n) = start_y + (dt/6.0)*(tk1y + 2.0*tk2y + 2.0*tk3y + tk4y)
                az(n) = start_z + (dt/6.0)*(tk1z + 2.0*tk2z + 2.0*tk3z + tk4z)
            END DO

            ! -----------------------------------------------------------
            ! Loop 2: 垂直擴散與邊界檢查 (Diffusion & BC)
            ! -----------------------------------------------------------
            !$acc parallel loop gang vector &
            !$acc private(xpos, ypos, zpos, ip, jp, kp, xpos_rel, ypos_rel, zpos_rel) &
            !$acc private(m_curr, i_chg, j_chg, d_val, idx_n, idx_p, idy_n, idy_p) &
            !$acc private(val_dz, val_idx_now, val_idx_nxt, val_idx_prv, val_ref, n_sub, idx_flat)
            DO n = 1, n_part
                xpos = ax(n); ypos = ay(n); zpos = az(n)
                
                ! 計算索引
                ip = INT(xpos/dx_avg); jp = INT(ypos/dy_avg); kp = INT(zpos/dz_avg)
                xpos_rel = (xpos/dx_avg) - REAL(ip,4); ypos_rel = (ypos/dy_avg) - REAL(jp,4)
                zpos_rel = (zpos/dz_avg) - REAL(kp,4)

                m_curr = 0 ! 暫時 Placeholder，若有 mask_packed 應在此讀取
                
                ! 1. 垂直擴散
                IF (BTEST(m_curr, 7)) THEN
                    CALL do_vertical_diffusion(INT(n,8), i_step, ip, jp, kp, ni, nj, dt, m_curr, &
                        zpos_rel, p_dispt, p_dz, p_idz0, p_dz_ref, &
                        1.0, 1.0, 0.01, 0.01, n_sub)
                END IF

                ! 2. X 邊界 (使用 1D 索引)
                IF (xpos_rel >= 1.0 .OR. xpos_rel < 0.0) THEN
                    idx_flat = (jp-1)*ni + ip
                    d_val = p_dx(idx_flat)  
                    idx_n = p_idx0((jp-1)*(ni+1) + (ip+1))
                    idx_p = p_idx0((jp-1)*(ni+1) + (ip-1))
                    CALL cross_x_branchless(ip, xpos_rel, m_curr, d_val, idx_n, idx_p, reflc, repul)
                    i_chg = .TRUE.
                ELSE
                    i_chg = .FALSE.
                END IF

                ! 3. Y 邊界 (使用 1D 索引)
                IF (ypos_rel >= 1.0 .OR. ypos_rel < 0.0) THEN
                    idx_flat = (jp-1)*ni + ip
                    d_val = p_dy(idx_flat)
                    idy_n = p_idy0( jp   *ni + ip) 
                    idy_p = p_idy0((jp-2)*ni + ip) 
                    CALL cross_y_branchless(jp, ypos_rel, m_curr, d_val, idy_n, idy_p, reflc, repul)
                    j_chg = .TRUE.
                ELSE
                    j_chg = .FALSE.
                END IF

                ! 4. Z 邊界 (使用 1D 索引)
                IF (zpos_rel >= 1.0 .OR. zpos_rel < 0.0) THEN
                    val_dz = p_dz(kp); val_idx_now = p_idz0(kp)
                    val_idx_nxt = p_idz0(kp+1); val_idx_prv = p_idz0(kp-1)
                    ! p_dz_ref 是 3D 壓扁: (k-1)*ni*nj + (j-1)*ni + i
                    idx_flat = (INT(kp,8)-1_8)*INT(ni,8)*INT(nj,8) + (INT(jp,8)-1_8)*INT(ni,8) + INT(ip,8)
                    val_ref = p_dz_ref(idx_flat)
                    CALL cross_z_branchless(kp, zpos_rel, m_curr, val_dz, val_idx_now, &
                        val_idx_nxt, val_idx_prv, val_ref, 1.0, 1.0, 0.01, 0.01)
                END IF

                ! 更新回全域座標
                ax(n) = (REAL(ip,4) + xpos_rel) * dx_avg
                ay(n) = (REAL(jp,4) + ypos_rel) * dy_avg
                az(n) = (REAL(kp,4) + zpos_rel) * dz_avg
            END DO

            ! -----------------------------------------------------------
            ! Loop 3: Property Update
            ! -----------------------------------------------------------
            !$acc parallel loop gang vector private(kp, zpos_rel, prop_offset_8, z_term)
            DO n = 1, n_part
               kp = INT(az(n)/dz_avg)
               zpos_rel = (az(n)/dz_avg) - REAL(kp, 4)
               prop_offset_8 = (n - 1) * INT(n_props, 8)
               z_term = p_zlev(kp) - (1.0 - zpos_rel) * p_dz(kp)
               
               IF (idx_age > 0) THEN
                   p_props(prop_offset_8 + idx_age) = p_props(prop_offset_8 + idx_age) + dt
               END IF
            END DO

        END DO
        !$acc end data

    END SUBROUTINE rk4_time_integration

    ! =========================================================
    ! Helper Routines
    ! =========================================================
    !$acc routine seq
    SUBROUTINE get_field_deriv(x, y, z, f, dx, dy, dz, gx, gy, gz, kx, ky, kz)
        REAL(4), INTENT(IN) :: x, y, z, f(*), dx, dy, dz
        INTEGER, INTENT(IN) :: gx, gy, gz
        REAL(4), INTENT(OUT) :: kx, ky, kz
        
        REAL(4) :: fx_loc, fy_loc, fz_loc, wx, wy, wz
        REAL(4) :: w00, w10, w0, w1
        INTEGER :: ii, jj, kk, off, stride_z

        fx_loc = x / dx; fy_loc = y / dy; fz_loc = z / dz
        ii = INT(fx_loc); jj = INT(fy_loc); kk = INT(fz_loc)

        IF (ii < 0) ii = 0
        IF (ii > gx - 2) ii = gx - 2
        IF (jj < 0) jj = 0
        IF (jj > gy - 2) jj = gy - 2
        IF (kk < 0) kk = 0
        IF (kk > gz - 2) kk = gz - 2

        wx = fx_loc - REAL(ii, 4)
        wy = fy_loc - REAL(jj, 4)
        wz = fz_loc - REAL(kk, 4)

        stride_z = gx * gy
        off = kk * stride_z + jj * gx + ii + 1

        ! U
        w00 = f(off)       * (1.-wx) + f(off+1)       * wx
        w10 = f(off+gx)    * (1.-wx) + f(off+gx+1)    * wx
        w0  = w00 * (1.-wy) + w10 * wy
        w00 = f(off+stride_z)    * (1.-wx) + f(off+stride_z+1)    * wx
        w10 = f(off+stride_z+gx) * (1.-wx) + f(off+stride_z+gx+1) * wx
        w1  = w00 * (1.-wy) + w10 * wy
        kx  = w0 * (1.-wz) + w1 * wz

        ! V
        off = off + (stride_z * gz)
        w00 = f(off)       * (1.-wx) + f(off+1)       * wx
        w10 = f(off+gx)    * (1.-wx) + f(off+gx+1)    * wx
        w0  = w00 * (1.-wy) + w10 * wy
        w00 = f(off+stride_z)    * (1.-wx) + f(off+stride_z+1)    * wx
        w10 = f(off+stride_z+gx) * (1.-wx) + f(off+stride_z+gx+1) * wx
        w1  = w00 * (1.-wy) + w10 * wy
        ky  = w0 * (1.-wz) + w1 * wz

        ! W
        off = off + (stride_z * gz)
        w00 = f(off)       * (1.-wx) + f(off+1)       * wx
        w10 = f(off+gx)    * (1.-wx) + f(off+gx+1)    * wx
        w0  = w00 * (1.-wy) + w10 * wy
        w00 = f(off+stride_z)    * (1.-wx) + f(off+stride_z+1)    * wx
        w10 = f(off+stride_z+gx) * (1.-wx) + f(off+stride_z+gx+1) * wx
        w1  = w00 * (1.-wy) + w10 * wy
        kz  = w0 * (1.-wz) + w1 * wz
    END SUBROUTINE get_field_deriv

    !$acc routine seq
    SUBROUTINE do_vertical_diffusion(n, step, ip, jp, kp, ni, nj, dt_val, m_packed_val, &
                                     zpos_rel_inout, p_dispt, dz_arr, idz0_arr, dz_ref_arr, &
                                     reflc_sfc, reflc_btm, repul_sfc, repul_btm, steps_taken)
        
        INTEGER(8), INTENT(IN) :: n
        INTEGER, INTENT(IN) :: step, ip, jp, kp, ni, nj
        REAL(4), INTENT(IN) :: dt_val
        INTEGER(1), INTENT(IN) :: m_packed_val
        REAL(4), INTENT(INOUT) :: zpos_rel_inout
        REAL(4), INTENT(IN) :: p_dispt(*) 
        REAL(4), INTENT(IN) :: dz_arr(*), idz0_arr(*), dz_ref_arr(*)
        REAL(4), INTENT(IN) :: reflc_sfc, reflc_btm, repul_sfc, repul_btm
        INTEGER, INTENT(OUT) :: steps_taken

        INTEGER :: iter, n_sub, m_steps, nsplit_local, my_seed
        INTEGER(8) :: sz_layer, idx_base, idx_k, idx_km1, idx_kp1, idx_flat
        REAL(4) :: dtsplit, val_k, val_km1, val_kp1, term_bot, term_top
        REAL(4) :: idx0_sq, kt(0:2), kt_max, f_step, rnd_val, term_diff
        REAL(4) :: val_dz, val_idx_now, val_idx_nxt, val_idx_prv, val_ref

        nsplit_local = 5 
        dtsplit = dt_val / REAL(nsplit_local, 4)
        n_sub = 0
        
        sz_layer = INT(ni + 2, 8) * INT(nj + 2, 8)
        idx_base = INT(jp, 8) * INT(ni + 2, 8) + INT(ip, 8) + 1_8 

        my_seed = IEOR(INT(n, 4), step * 1664525)
        IF (my_seed == 0) my_seed = 123456789

        DO iter = 1, nsplit_local
            IF (n_sub >= nsplit_local) EXIT

            idx_k   = INT(kp, 8) * sz_layer + idx_base
            idx_km1 = INT(kp-1, 8) * sz_layer + idx_base
            idx_kp1 = INT(kp+1, 8) * sz_layer + idx_base
            
            val_k   = p_dispt(idx_k)
            val_km1 = p_dispt(idx_km1)
            val_kp1 = p_dispt(idx_kp1)
            
            term_bot = MERGE(1.0_4, 0.0_4, BTEST(m_packed_val, 5))
            term_top = MERGE(1.0_4, 0.0_4, BTEST(m_packed_val, 4))
            
            idx0_sq = idz0_arr(kp)**2
            
            kt(0) = 0.5_4 * term_bot * (val_k + val_km1) * dtsplit * idx0_sq
            kt(1) =                  val_k                  * dtsplit * idx0_sq
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

            term_diff = sqrt(2.0_4 * max((1.0_4 - zpos_rel_inout)*kt(0) + zpos_rel_inout*kt(2), 0.0_4))
            
            zpos_rel_inout = zpos_rel_inout + rnd_val * term_diff
            zpos_rel_inout = zpos_rel_inout + (kt(2) - kt(0))
            
            val_dz      = dz_arr(kp)
            val_idx_now = idz0_arr(kp)
            val_idx_nxt = idz0_arr(kp+1)
            val_idx_prv = idz0_arr(kp-1)
            
            ! dz_ref is 3D flattened: (k-1)*ni*nj + (j-1)*ni + i
            idx_flat = (INT(kp,8)-1_8)*INT(ni,8)*INT(nj,8) + (INT(jp,8)-1_8)*INT(ni,8) + INT(ip,8)
            val_ref     = dz_ref_arr(idx_flat)

            CALL cross_z_branchless(kp, zpos_rel_inout, m_packed_val, val_dz, val_idx_now, &
                                    val_idx_nxt, val_idx_prv, val_ref, &
                                    reflc_sfc, reflc_btm, repul_sfc, repul_btm)
                                    
            n_sub = n_sub + m_steps
        END DO
        steps_taken = n_sub
    END SUBROUTINE do_vertical_diffusion

END MODULE Physics_Routines_Mod