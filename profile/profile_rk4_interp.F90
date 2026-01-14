PROGRAM profile_rk4_dv
  USE Device_Vector
  USE openacc
  !USE cudafor
  USE iso_c_binding
  IMPLICIT NONE

  INTEGER(8) :: N_P, N_S
  REAL(4)    :: DT, DX, DY, DZ
  INTEGER(8) :: GX, GY, GZ, GXY, GXYZ
  INTEGER    :: ios

  TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
  TYPE(device_vector_r4_t) :: k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z
  
  ! 指針映射
  REAL(4), POINTER, CONTIGUOUS :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
  REAL(4), POINTER, CONTIGUOUS :: ak1x(:), ak1y(:), ak1z(:), ak2x(:), ak2y(:), ak2z(:)
  REAL(4), POINTER, CONTIGUOUS :: ak3x(:), ak3y(:), ak3z(:), ak4x(:), ak4y(:), ak4z(:)
  REAL(4), ALLOCATABLE, TARGET :: f1d(:)
  
  INTEGER(8) :: i_step, n, t1, t2, t_rate, off8
  REAL(4)    :: tx, ty, tz, fx, fy, fz, wx, wy, wz, w00, w10, w0, w1
  REAL(8)    :: t_dv, t_acc, t_dv2
  INTEGER(8) :: ii,     jj, kk

  ! GPU Hardware constant
  INTEGER, PARAMETER :: WARP_LENGTH = 128

  NAMELIST /sim_config/ N_P, N_S, DT, DX, DY, DZ, GX, GY, GZ

  PRINT *, "[Init] Reading input.nml..."
   OPEN(UNIT=10, FILE='../configs/test_rk4.nml', STATUS='OLD', IOSTAT=ios)
   IF (ios == 0) THEN
      READ(10, NML=sim_config)
      GXY  = GX * GY    
      GXYZ = GXY * GZ   
      CLOSE(10)
      PRINT *, "[Config] Parameters loaded successfully."
   ELSE
      PRINT *, "[Config] input.nml not found, exit."
  END IF

  PRINT *, "[Init] Generating Random Velocity Field..."
  ALLOCATE(f1d(GXYZ * 3))

  CALL device_env_init(0, 1)

  CALL RANDOM_SEED()
  CALL RANDOM_NUMBER(f1d)
  !$acc enter data copyin(f1d)

  CALL SYSTEM_CLOCK(t1, t_rate)
  CALL run_device_vector()
  CALL SYSTEM_CLOCK(t2, t_rate)
  CALL SYSTEM_CLOCK(t2)

  t_dv = REAL(t2-t1,8)/REAL(t_rate,8)
  PRINT *, " [DeviceVector] Total Time (Full Physics): ", t_dv, " s"

  CALL SYSTEM_CLOCK(t1, t_rate)
  CALL run_device_vector_ver2(f1d)
  CALL SYSTEM_CLOCK(t2, t_rate)
  CALL SYSTEM_CLOCK(t2)

  t_dv2 = REAL(t2-t1,8)/REAL(t_rate,8)
  PRINT *, " [DeviceVector_Ver2] Total Time (Full Physics): ", t_dv2, " s"

  CALL SYSTEM_CLOCK(t1, t_rate)
  CALL run_raw_openacc()
  CALL SYSTEM_CLOCK(t2, t_rate)
  CALL SYSTEM_CLOCK(t2)

  t_acc = REAL(t2-t1,8)/REAL(t_rate,8)
  PRINT *, " [openACC] Total Time (Full Physics): ", t_acc, " s"

  ! ==================================================================
  ! Report
  ! ==================================================================
  PRINT *, "=========================================================="
  PRINT *, "                 FINAL TRUE RESULTS                       "
  PRINT *, "=========================================================="
  PRINT '(A, F10.4, A)', " [1] OpenACC Time              : ", t_acc, " s"
  PRINT '(A, F10.4, A)', " [2] DeviceVector Time         : ", t_dv,  " s"
  PRINT '(A, F10.4, A)', " [3] DeviceVector_Improve Time : ", t_dv2, " s"
  PRINT *, "----------------------------------------------------------"
  PRINT '(A, F10.2, A)', " Speedup (DV_improve vs OpenACC): ", t_acc / t_dv2, " x"
  PRINT '(A, F10.2, A)', " Speedup (DV_improve vs DV_odd) : ", t_dv  / t_dv2, " x"
  PRINT *, "=========================================================="

CONTAINS

  ! ---------------------------------------------------------
  ! Main Subroutine: Device Vector Control
  ! ---------------------------------------------------------
  SUBROUTINE run_device_vector_ver2(host_data)
IMPLICIT NONE
    REAL(4), INTENT(IN) :: host_data(:)
    REAL(4), ALLOCATABLE, TARGET :: f1d_local(:)
    
    INTEGER(8) :: i_step, n, off8, ii, jj, kk
    REAL(4) :: fx, fy, fz, wx, wy, wz, w00, w10, w0, w1
    REAL(4) :: tx, ty, tz

    CALL device_env_init(0, 1)
    
    ALLOCATE(f1d_local(SIZE(host_data)))
    f1d_local = host_data
    !$acc enter data copyin(f1d_local)

    ! 建立 Buffer
    CALL px%create_buffer(N_P);  CALL py%create_buffer(N_P);  CALL pz%create_buffer(N_P)
    CALL vx%create_buffer(N_P);  CALL vy%create_buffer(N_P);  CALL vz%create_buffer(N_P)
    CALL k1x%create_buffer(N_P); CALL k1y%create_buffer(N_P); CALL k1z%create_buffer(N_P)
    CALL k2x%create_buffer(N_P); CALL k2y%create_buffer(N_P); CALL k2z%create_buffer(N_P)
    CALL k3x%create_buffer(N_P); CALL k3y%create_buffer(N_P); CALL k3z%create_buffer(N_P)
    CALL k4x%create_buffer(N_P); CALL k4y%create_buffer(N_P); CALL k4z%create_buffer(N_P)

    ! 映射指針
    CALL px%acc_map(ax);    CALL py%acc_map(ay);    CALL pz%acc_map(az)
    CALL vx%acc_map(aux);   CALL vy%acc_map(auy);   CALL vz%acc_map(auz)
    CALL k1x%acc_map(ak1x); CALL k1y%acc_map(ak1y); CALL k1z%acc_map(ak1z)
    CALL k2x%acc_map(ak2x); CALL k2y%acc_map(ak2y); CALL k2z%acc_map(ak2z)
    CALL k3x%acc_map(ak3x); CALL k3y%acc_map(ak3y); CALL k3z%acc_map(ak3z)
    CALL k4x%acc_map(ak4x); CALL k4y%acc_map(ak4y); CALL k4z%acc_map(ak4z)

    ! 初始化位置
    !$acc parallel loop present(ax, ay, az, aux, auy, auz)
    DO n = 1, N_P
       ax(n)=32.5; ay(n)=32.5; az(n)=32.5; aux(n)=1.0; auy(n)=0.0; auz(n)=0.0
    END DO
    
    CALL device_synchronize() 

    ! ★★★ 關鍵優化：Data Region Hoisting ★★★
    ! 在迴圈外查一次表，迴圈內就不需要再查了，省去數千次 Overhead
    !$acc data present(f1d_local, ax, ay, az) &
    !$acc      present(ak1x, ak1y, ak1z, ak2x, ak2y, ak2z) &
    !$acc      present(ak3x, ak3y, ak3z, ak4x, ak4y, ak4z)

    DO i_step = 1, N_S
       
       ! --- Step 1: k1 ---
       ! 注意：這裡移除了 present()，因為外面已經包了 data present
       !$acc parallel loop gang vector_length(WARP_LENGTH) &
       !$acc private(fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
       DO n = 1, N_P
          fx=ax(n)/DX; fy=ay(n)/DY; fz=az(n)/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak1x(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak1y(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak1z(n)=w0*(1.-wz)+w1*wz
       END DO

       ! --- Step 2: k2 ---
       !$acc parallel loop gang vector_length(WARP_LENGTH) &
       !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
       DO n = 1, N_P
          tx=ax(n)+0.5*DT*ak1x(n); ty=ay(n)+0.5*DT*ak1y(n); tz=az(n)+0.5*DT*ak1z(n)
          fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak2x(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak2y(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak2z(n)=w0*(1.-wz)+w1*wz
       END DO

       ! --- Step 3: k3 ---
       !$acc parallel loop gang vector_length(WARP_LENGTH) &
       !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
       DO n = 1, N_P
          tx=ax(n)+0.5*DT*ak2x(n); ty=ay(n)+0.5*DT*ak2y(n); tz=az(n)+0.5*DT*ak2z(n)
          fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak3x(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak3y(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak3z(n)=w0*(1.-wz)+w1*wz
       END DO

       ! --- Step 4: k4 ---
       !$acc parallel loop gang vector_length(WARP_LENGTH) &
       !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
       DO n = 1, N_P
          tx=ax(n)+DT*ak3x(n); ty=ay(n)+DT*ak3y(n); tz=az(n)+DT*ak3z(n)
          fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak4x(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak4y(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=f1d_local(off8)*(1.-wx)+f1d_local(off8+1)*wx; w10=f1d_local(off8+GX)*(1.-wx)+f1d_local(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=f1d_local(off8+GXY)*(1.-wx)+f1d_local(off8+GXY+1)*wx; w10=f1d_local(off8+GXY+GX)*(1.-wx)+f1d_local(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          ak4z(n)=w0*(1.-wz)+w1*wz
       END DO

       ! --- Final Update ---
       !$acc parallel loop gang vector_length(WARP_LENGTH)
       DO n = 1, N_P
          ax(n) = ax(n) + (DT/6.0_4)*(ak1x(n) + 2.0_4*ak2x(n) + 2.0_4*ak3x(n) + ak4x(n))
          ay(n) = ay(n) + (DT/6.0_4)*(ak1y(n) + 2.0_4*ak2y(n) + 2.0_4*ak3y(n) + ak4y(n))
          az(n) = az(n) + (DT/6.0_4)*(ak1z(n) + 2.0_4*ak2z(n) + 2.0_4*ak3z(n) + ak4z(n))
       END DO
    END DO
    !$acc end data
    ! ★★★ 優化結束 ★★★

    CALL device_synchronize()
    
    ! 清理
    CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
    CALL k1x%acc_unmap(); CALL k2x%acc_unmap(); CALL k3x%acc_unmap(); CALL k4x%acc_unmap()
    CALL k1y%acc_unmap(); CALL k2y%acc_unmap(); CALL k3y%acc_unmap(); CALL k4y%acc_unmap()
    CALL k1z%acc_unmap(); CALL k2z%acc_unmap(); CALL k3z%acc_unmap(); CALL k4z%acc_unmap()

    CALL px%free(); CALL py%free(); CALL pz%free()
    CALL k1x%free(); CALL k2x%free(); CALL k3x%free(); CALL k4x%free()
    CALL k1y%free(); CALL k2y%free(); CALL k3y%free(); CALL k4y%free()
    CALL k1z%free(); CALL k2z%free(); CALL k3z%free(); CALL k4z%free()
    
    !$acc exit data delete(f1d_local)
    DEALLOCATE(f1d_local)
    CALL device_env_finalize()
  END SUBROUTINE run_device_vector_ver2

 SUBROUTINE run_device_vector()
  IMPLICIT NONE

  !ALLOCATE(f1d(GXYZ * 3)); f1d = 0.1_4
  !!$acc enter data copyin(f1d)

  CALL px%create_buffer(N_P);  CALL py%create_buffer(N_P);  CALL pz%create_buffer(N_P)
  CALL vx%create_buffer(N_P);  CALL vy%create_buffer(N_P);  CALL vz%create_buffer(N_P)
  CALL k1x%create_buffer(N_P); CALL k1y%create_buffer(N_P); CALL k1z%create_buffer(N_P)
  CALL k2x%create_buffer(N_P); CALL k2y%create_buffer(N_P); CALL k2z%create_buffer(N_P)
  CALL k3x%create_buffer(N_P); CALL k3y%create_buffer(N_P); CALL k3z%create_buffer(N_P)
  CALL k4x%create_buffer(N_P); CALL k4y%create_buffer(N_P); CALL k4z%create_buffer(N_P)

  CALL px%acc_map(ax);    CALL py%acc_map(ay);    CALL pz%acc_map(az)
  CALL vx%acc_map(aux);   CALL vy%acc_map(auy);   CALL vz%acc_map(auz)
  CALL k1x%acc_map(ak1x); CALL k1y%acc_map(ak1y); CALL k1z%acc_map(ak1z)
  CALL k2x%acc_map(ak2x); CALL k2y%acc_map(ak2y); CALL k2z%acc_map(ak2z)
  CALL k3x%acc_map(ak3x); CALL k3y%acc_map(ak3y); CALL k3z%acc_map(ak3z)
  CALL k4x%acc_map(ak4x); CALL k4y%acc_map(ak4y); CALL k4z%acc_map(ak4z)

  !$acc parallel loop present(ax, ay, az, aux, auy, auz)
  DO n = 1, N_P
     ax(n)=32.5; ay(n)=32.5; az(n)=32.5; aux(n)=1.0; auy(n)=0.0; auz(n)=0.0
  END DO

  PRINT *, "[Run] Physics RK4  on GPU..."
  CALL device_synchronize() 

   DO i_step = 1, N_S
     
     ! --- Step 1: k1 = f(P_n) ---
     !$acc parallel loop gang vector_length(WARP_LENGTH) present(f1d, ax, ay, az, ak1x, ak1y, ak1z) &
     !$acc private(fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
     DO n = 1, N_P
        fx=ax(n)/DX; fy=ay(n)/DY; fz=az(n)/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
        IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
        wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
        
        ! ★ 修正
        off8=kk*GXY+jj*GX+ii+1_8
        
        ! X-interp
        w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx
        w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx ! ★ +64 改 +GX
        w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx
        w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx ! ★ +64 改 +GX
        w1=w00*(1.-wy)+w10*wy
        ak1x(n)=w0*(1.-wz)+w1*wz
        
        ! Y-interp
        off8=off8+GXYZ
        w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx
        w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx ! ★ +64 改 +GX
        w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx
        w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx ! ★ +64 改 +GX
        w1=w00*(1.-wy)+w10*wy
        ak1y(n)=w0*(1.-wz)+w1*wz
        
        ! Z-interp
        off8=off8+GXYZ
        w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx
        w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx ! ★ +64 改 +GX
        w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx
        w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx ! ★ +64 改 +GX
        w1=w00*(1.-wy)+w10*wy
        ak1z(n)=w0*(1.-wz)+w1*wz
     END DO

     ! --- Step 2: k2 = f(P_n + 0.5*dt*k1) ---
     !$acc parallel loop gang vector_length(WARP_LENGTH) present(f1d, ax, ay, az, ak1x, ak1y, ak1z, ak2x, ak2y, ak2z) &
     !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
     DO n = 1, N_P
        tx=ax(n)+0.5*DT*ak1x(n); ty=ay(n)+0.5*DT*ak1y(n); tz=az(n)+0.5*DT*ak1z(n)
        fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
        IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
        wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
        
        off8=kk*GXY+jj*GX+ii+1_8 !
        
        ! X-interp
        w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
        ak2x(n)=w0*(1.-wz)+w1*wz
        ! Y-interp
        off8=off8+GXYZ; w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
        ak2y(n)=w0*(1.-wz)+w1*wz
        ! Z-interp
        off8=off8+GXYZ; w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
        ak2z(n)=w0*(1.-wz)+w1*wz
     END DO

     ! --- Step 3: k3 = f(P_n + 0.5*dt*k2) ---
     !$acc parallel loop gang vector_length(WARP_LENGTH) present(f1d, ax, ay, az, ak2x, ak2y, ak2z, ak3x, ak3y, ak3z) &
     !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
     DO n = 1, N_P
        tx=ax(n)+0.5*DT*ak2x(n); ty=ay(n)+0.5*DT*ak2y(n); tz=az(n)+0.5*DT*ak2z(n)
        fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
        IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
        wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
        
        off8=kk*GXY+jj*GX+ii+1_8 !
        
        ! X-interp
        w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
        ak3x(n)=w0*(1.-wz)+w1*wz
        ! Y-interp
        off8=off8+GXYZ; w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
        ak3y(n)=w0*(1.-wz)+w1*wz
        ! Z-interp
        off8=off8+GXYZ; w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
        ak3z(n)=w0*(1.-wz)+w1*wz
     END DO

     ! --- Step 4: k4 = f(P_n + dt*k3) ---
     !$acc parallel loop gang vector_length(WARP_LENGTH) present(f1d, ax, ay, az, ak3x, ak3y, ak3z, ak4x, ak4y, ak4z) &
     !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
     DO n = 1, N_P
        tx=ax(n)+DT*ak3x(n); ty=ay(n)+DT*ak3y(n); tz=az(n)+DT*ak3z(n)
        fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
        IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
        wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
        
        off8=kk*GXY+jj*GX+ii+1_8 !
        
        ! X-interp
        w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
        ak4x(n)=w0*(1.-wz)+w1*wz
        ! Y-interp
        off8=off8+GXYZ; w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
        ak4y(n)=w0*(1.-wz)+w1*wz
        ! Z-interp
        off8=off8+GXYZ; w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
        w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
        ak4z(n)=w0*(1.-wz)+w1*wz
     END DO

     ! --- Final Update ---
     !$acc parallel loop gang vector_length(WARP_LENGTH) &
     !$acc present(ax, ay, az, ak1x, ak1y, ak1z, ak2x, ak2y, ak2z, ak3x, ak3y, ak3z, ak4x, ak4y, ak4z)
     DO n = 1, N_P
        ax(n) = ax(n) + (DT/6.0_4)*(ak1x(n) + 2.0_4*ak2x(n) + 2.0_4*ak3x(n) + ak4x(n))
        ay(n) = ay(n) + (DT/6.0_4)*(ak1y(n) + 2.0_4*ak2y(n) + 2.0_4*ak3y(n) + ak4y(n))
        az(n) = az(n) + (DT/6.0_4)*(ak1z(n) + 2.0_4*ak2z(n) + 2.0_4*ak3z(n) + ak4z(n))
     END DO
  END DO

  CALL device_synchronize() 

  CALL px%acc_unmap();  CALL py%acc_unmap();  CALL pz%acc_unmap()
  CALL k1x%acc_unmap(); CALL k2x%acc_unmap(); CALL k3x%acc_unmap(); CALL k4x%acc_unmap()
  CALL k1y%acc_unmap(); CALL k2y%acc_unmap(); CALL k3y%acc_unmap(); CALL k4y%acc_unmap()
  CALL k1z%acc_unmap(); CALL k2z%acc_unmap(); CALL k3z%acc_unmap(); CALL k4z%acc_unmap()

  CALL px%free();  CALL py%free();  CALL pz%free()
  CALL k1x%free(); CALL k2x%free(); CALL k3x%free(); CALL k4x%free()
  CALL k1y%free(); CALL k2y%free(); CALL k3y%free(); CALL k4y%free()
  CALL k1z%free(); CALL k2z%free(); CALL k3z%free(); CALL k4z%free()
  !$acc exit data delete(f1d)
  
 END SUBROUTINE
 
  SUBROUTINE run_raw_openacc()
   IMPLICIT NONE
   
   REAL(4), ALLOCATABLE :: ax(:), ay(:), az(:)
   REAL(4), ALLOCATABLE :: k1x(:), k1y(:), k1z(:), k2x(:), k2y(:), k2z(:)
   REAL(4), ALLOCATABLE :: k3x(:), k3y(:), k3z(:), k4x(:), k4y(:), k4z(:)
   
   INTEGER(8) :: n, i_step, off8
   REAL(4)    :: tx, ty, tz, fx, fy, fz, wx, wy, wz, w00, w10, w0, w1
   INTEGER(8) :: ii, jj, kk

   PRINT *, "----------------------------------------------------------"
   PRINT *, "[Bench] Starting Raw OpenACC (Baseline - Full Physics)..."

   ALLOCATE(ax(N_P), ay(N_P), az(N_P))
   ALLOCATE(k1x(N_P), k1y(N_P), k1z(N_P), k2x(N_P), k2y(N_P), k2z(N_P))
   ALLOCATE(k3x(N_P), k3y(N_P), k3z(N_P), k4x(N_P), k4y(N_P), k4z(N_P))

   !$acc enter data copyin(ax, ay, az, f1d) &
   !$acc create(k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z)

   !$acc parallel loop present(ax, ay, az)
   DO n = 1, N_P
      ax(n)=32.5; ay(n)=32.5; az(n)=32.5
   END DO

   CALL device_synchronize()

   DO i_step = 1, N_S
      
      ! --- Step 1: k1 = f(P_n) ---
      !$acc parallel loop gang vector_length(WARP_LENGTH) present(f1d, ax, ay, az, k1x, k1y, k1z) &
      !$acc private(fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
      DO n = 1, N_P
         fx=ax(n)/DX; fy=ay(n)/DY; fz=az(n)/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
         IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
         wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
         
         off8=kk*GXY+jj*GX+ii+1_8
         
         ! X-interp
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k1x(n)=w0*(1.-wz)+w1*wz
         
         ! Y-interp
         off8=off8+GXYZ
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k1y(n)=w0*(1.-wz)+w1*wz
         
         ! Z-interp
         off8=off8+GXYZ
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k1z(n)=w0*(1.-wz)+w1*wz
      END DO

      ! --- Step 2: k2 = f(P_n + 0.5*dt*k1) ---
      !$acc parallel loop gang vector_length(WARP_LENGTH) present(f1d, ax, ay, az, k1x, k1y, k1z, k2x, k2y, k2z) &
      !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
      DO n = 1, N_P
         tx=ax(n)+0.5*DT*k1x(n); ty=ay(n)+0.5*DT*k1y(n); tz=az(n)+0.5*DT*k1z(n)
         fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
         IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
         wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
         
         off8=kk*GXY+jj*GX+ii+1_8
         
         ! X-interp
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k2x(n)=w0*(1.-wz)+w1*wz
         
         ! Y-interp
         off8=off8+GXYZ
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k2y(n)=w0*(1.-wz)+w1*wz
         
         ! Z-interp
         off8=off8+GXYZ
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k2z(n)=w0*(1.-wz)+w1*wz
      END DO

      ! --- Step 3: k3 = f(P_n + 0.5*dt*k2) ---
      !$acc parallel loop gang vector_length(WARP_LENGTH) present(f1d, ax, ay, az, k2x, k2y, k2z, k3x, k3y, k3z) &
      !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
      DO n = 1, N_P
         tx=ax(n)+0.5*DT*k2x(n); ty=ay(n)+0.5*DT*k2y(n); tz=az(n)+0.5*DT*k2z(n)
         fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
         IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
         wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
         
         off8=kk*GXY+jj*GX+ii+1_8
         
         ! X-interp
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k3x(n)=w0*(1.-wz)+w1*wz
         
         ! Y-interp
         off8=off8+GXYZ
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k3y(n)=w0*(1.-wz)+w1*wz
         
         ! Z-interp
         off8=off8+GXYZ
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k3z(n)=w0*(1.-wz)+w1*wz
      END DO

      ! --- Step 4: k4 = f(P_n + dt*k3) ---
      !$acc parallel loop gang vector_length(WARP_LENGTH) present(f1d, ax, ay, az, k3x, k3y, k3z, k4x, k4y, k4z) &
      !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
      DO n = 1, N_P
         tx=ax(n)+DT*k3x(n); ty=ay(n)+DT*k3y(n); tz=az(n)+DT*k3z(n)
         fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
         IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
         wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
         
         off8=kk*GXY+jj*GX+ii+1_8
         
         ! X-interp
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k4x(n)=w0*(1.-wz)+w1*wz
         
         ! Y-interp
         off8=off8+GXYZ
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k4y(n)=w0*(1.-wz)+w1*wz
         
         ! Z-interp
         off8=off8+GXY
         w00=f1d(off8)*(1.-wx)+f1d(off8+1)*wx; w10=f1d(off8+GX)*(1.-wx)+f1d(off8+GX+1)*wx
         w0=w00*(1.-wy)+w10*wy
         w00=f1d(off8+GXY)*(1.-wx)+f1d(off8+GXY+1)*wx; w10=f1d(off8+GXY+GX)*(1.-wx)+f1d(off8+GXY+GX+1)*wx
         w1=w00*(1.-wy)+w10*wy
         k4z(n)=w0*(1.-wz)+w1*wz
      END DO

      ! --- Final Update ---
      !$acc parallel loop gang vector_length(WARP_LENGTH) &
      !$acc present(ax, ay, az, k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z)
      DO n = 1, N_P
         ax(n) = ax(n) + (DT/6.0_4)*(k1x(n) + 2.0_4*k2x(n) + 2.0_4*k3x(n) + k4x(n))
         ay(n) = ay(n) + (DT/6.0_4)*(k1y(n) + 2.0_4*k2y(n) + 2.0_4*k3y(n) + k4y(n))
         az(n) = az(n) + (DT/6.0_4)*(k1z(n) + 2.0_4*k2z(n) + 2.0_4*k3z(n) + k4z(n))
      END DO
   END DO

   CALL device_synchronize()

   !$acc exit data delete(ax, ay, az, f1d, k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z)
   DEALLOCATE(ax, ay, az, f1d, k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z)
   END SUBROUTINE

END PROGRAM