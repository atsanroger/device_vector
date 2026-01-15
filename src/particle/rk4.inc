SUBROUTINE rk4(host_data)
    USE Device_Vector
    USE PARAMETERS
    IMPLICIT NONE
    
    ! 輸入資料
    REAL(4), INTENT(IN) :: host_data(:)
    
    ! 本地變數
    REAL(4), ALLOCATABLE, TARGET :: f1d_local(:)
    
    ! Device Vector 結構
    TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
    REAL(4), POINTER, CONTIGUOUS :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
    
    ! 迴圈變數
    INTEGER(8) :: i_step, n
    
    ! RK4 運算需要的暫存變數
    REAL(4) :: xpos, ypos, zpos
    REAL(4) :: start_x, start_y, start_z
    REAL(4) :: tk1x, tk1y, tk1z, tk2x, tk2y, tk2z
    REAL(4) :: tk3x, tk3y, tk3z, tk4x, tk4y, tk4z
    REAL(4) :: fx, fy, fz, wx, wy, wz, w00, w10, w0, w1
    INTEGER(8) :: ii, jj, kk, off8

    INTEGER :: ip, jp, kp
    INTEGER(1) :: m_curr
    REAL(4) :: d_val, idx_n, idx_p, idy_n, idy_p
    REAL(4) :: val_dz, val_idx_now, val_idx_nxt, val_idx_prv, val_ref
    LOGICAL :: i_chg, j_chg

    ! 1. 準備資料
    ALLOCATE(f1d_local(SIZE(host_data)))
    f1d_local = host_data
    $acc enter data copyin(f1d_local)

    ! 2. 建立與映射 Device Vector
    CALL px%create_buffer(N_P);  CALL py%create_buffer(N_P);  CALL pz%create_buffer(N_P)
    CALL vx%create_buffer(N_P);  CALL vy%create_buffer(N_P);  CALL vz%create_buffer(N_P)
    CALL px%acc_map(ax);    CALL py%acc_map(ay);    CALL pz%acc_map(az)
    CALL vx%acc_map(aux);   CALL vy%acc_map(auy);   CALL vz%acc_map(auz)

    ! 3. 初始化粒子
    !$acc parallel loop present(ax, ay, az, aux, auy, auz)
    DO n = 1, N_P
        ax(n)=32.5; ay(n)=32.5; az(n)=32.5; aux(n)=1.0; auy(n)=0.0; auz(n)=0.0
    END DO
    CALL device_synchronize() 

    ! 4. 時間積分主迴圈 (RK4 內嵌)
    !$acc data present(f1d_local, ax, ay, az)
    DO i_step = 1, N_S
        
        !$acc parallel loop gang vector_length(WARP_LENGTH) &
        !$acc private(curr_x, curr_y, curr_z, start_x, start_y, start_z) &
        !$acc private(fx, fy, fz, ii, jj, kk, wx, wy, wz, off8, w00, w10, w0, w1) &
        !$acc private(tk1x, tk1y, tk1z, tk2x, tk2y, tk2z) &
        !$acc private(tk3x, tk3y, tk3z, tk4x, tk4y, tk4z)
        DO n = 1, N_P
            
            ! ================= RK4 START =================
            ! 1. 紀錄起點
            start_x = ax(n); start_y = ay(n); start_z = az(n)

            ! ---------------------------------------------
            ! Step 1: K1 (at Start)
            ! ---------------------------------------------
            curr_x = start_x; curr_y = start_y; curr_z = start_z
            
            fx=curr_x/DX; fy=curr_y/DY; fz=curr_z/DZ
            ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
            IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2
            IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2
            IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
            
            wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
            off8=kk*GXY+jj*GX+ii+1_8
            
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk1x=w0*(1.-wz)+w1*wz
            
            off8=off8+GXYZ
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk1y=w0*(1.-wz)+w1*wz
            
            off8=off8+GXYZ
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk1z=w0*(1.-wz)+w1*wz

            ! ---------------------------------------------
            ! Step 2: K2 (at Start + 0.5*dt*K1)
            ! ---------------------------------------------
            curr_x=start_x+0.5*DT*tk1x
            curr_y=start_y+0.5*DT*tk1y
            curr_z=start_z+0.5*DT*tk1z
            
            fx=curr_x/DX; fy=curr_y/DY; fz=curr_z/DZ
            ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
            IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2
            IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2
            IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
            
            wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
            off8=kk*GXY+jj*GX+ii+1_8
            
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk2x=w0*(1.-wz)+w1*wz
            
            off8=off8+GXYZ
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk2y=w0*(1.-wz)+w1*wz
            
            off8=off8+GXYZ
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk2z=w0*(1.-wz)+w1*wz

            ! ---------------------------------------------
            ! Step 3: K3 (at Start + 0.5*dt*K2)
            ! ---------------------------------------------
            curr_x=start_x+0.5*DT*tk2x
            curr_y=start_y+0.5*DT*tk2y
            curr_z=start_z+0.5*DT*tk2z
            
            fx=curr_x/DX; fy=curr_y/DY; fz=curr_z/DZ
            ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
            IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2
            IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2
            IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
            
            wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
            off8=kk*GXY+jj*GX+ii+1_8
            
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk3x=w0*(1.-wz)+w1*wz
            
            off8=off8+GXYZ
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk3y=w0*(1.-wz)+w1*wz
            
            off8=off8+GXYZ
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk3z=w0*(1.-wz)+w1*wz

            ! ---------------------------------------------
            ! Step 4: K4 (at Start + dt*K3)
            ! ---------------------------------------------
            curr_x=start_x+DT*tk3x
            curr_y=start_y+DT*tk3y
            curr_z=start_z+DT*tk3z
            
            fx=curr_x/DX; fy=curr_y/DY; fz=curr_z/DZ
            ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
            IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2
            IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2
            IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
            
            wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
            off8=kk*GXY+jj*GX+ii+1_8
            
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk4x=w0*(1.-wz)+w1*wz
            
            off8=off8+GXYZ
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk4y=w0*(1.-wz)+w1*wz
            
            off8=off8+GXYZ
            w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx
            w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx
            w0=w00*(1.-wy)+w10*wy
            w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx
            w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx
            w1=w00*(1.-wy)+w10*wy
            tk4z=w0*(1.-wz)+w1*wz

            ! ---------------------------------------------
            ! 更新位置
            ! ---------------------------------------------
            curr_x = start_x + (DT/6.0_4)*(tk1x+2.*tk2x+2.*tk3x+tk4x)
            curr_y = start_y + (DT/6.0_4)*(tk1y+2.*tk2y+2.*tk3y+tk4y)
            curr_z = start_z + (DT/6.0_4)*(tk1z+2.*tk2z+2.*tk3z+tk4z)            ! ================= RK4 END =================

            ip = INT(curr_x/DX); jp = INT(curr_y/DY); kp = INT(curr_z/DZ)
            m_curr = mask_packed(ip, jp, kp)

            ! --- X Check ---
            IF (curr_x >= REAL(GX-1,4) .OR. curr_x < 0.0_4) THEN
              d_val = dx(ip, jp)
              idx_n = idx0(ip+1, jp); idx_p = idx0(ip-1, jp)
              CALL cross_x_branchless(ip, curr_x, m_curr, d_val, idx_n, idx_p, reflc, repul)
              i_chg = .TRUE.
            ELSE
              i_chg = .FALSE.
            END IF

            ! --- Y Check ---
            IF (i_chg) m_curr = mask_packed(ip, jp, kp)
            IF (curr_y >= REAL(GY-1,4) .OR. curr_y < 0.0_4) THEN
              d_val = dy(ip, jp)
              idy_n = idy0(ip, jp+1); idy_p = idy0(ip, jp-1)
              CALL cross_y_branchless(jp, curr_y, m_curr, d_val, idy_n, idy_p, reflc, repul)
              j_chg = .TRUE.
            ELSE
              j_chg = .FALSE.
            END IF

            IF (zpos >= 1.0_4 .OR. zpos < 0.0_4) THEN
                val_dz      = dz(kp)
                val_idx_now = idz0(kp)
                val_idx_nxt = idz0(kp+1)
                val_idx_prv = idz0(kp-1)
                val_ref     = dz_ref(ip, jp, kp) !
            
                CALL cross_z_branchless(kp, zpos, m_curr, val_dz, val_idx_now, &
                                        val_idx_nxt, val_idx_prv, val_ref, &
                                        reflc_sfc, reflc_btm, repul_sfc, repul_btm)

            END IF

            ax(n) = curr_x
            ay(n) = curr_y
            az(n) = curr_z

        END DO ! loop over particle

    END DO ! rk4 time loop
    !$acc end data

    !CALL device_synchronize() 
    
    !CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
    !CALL vx%acc_unmap(); CALL vy%acc_unmap(); CALL vz%acc_unmap()
    !CALL px%free(); CALL py%free(); CALL pz%free()
    !CALL vx%free(); CALL vy%free(); CALL vz%free()
    
    !!$acc exit data delete(f1d_local)
    !DEALLOCATE(f1d_local)

END SUBROUTINE rk4

CONTAINS
  INCLUDE "cross_ckeck_logic.inc"
  INCLUDE "rand.int"
  INCLUDE "particle_init.inc"