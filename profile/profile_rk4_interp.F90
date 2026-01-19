PROGRAM profile_rk4_dv
  USE Device_Vector
  USE openacc
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

  ! Reference
  REAL(4), ALLOCATABLE :: ref_x(:), ref_y(:), ref_z(:)
  
  INTEGER(8) :: i_step, n, t1, t2, t_rate, off8
  REAL(4)    :: tx, ty, tz, fx, fy, fz, wx, wy, wz, w00, w10, w0, w1
  REAL(8)    :: t_acc, t_dv, t_dv2, t_dv3, t_dv4
  INTEGER(8) :: ii, jj, kk

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

  ALLOCATE(ref_x(N_P), ref_y(N_P), ref_z(N_P))

  CALL device_env_init(0, 1)

  CALL RANDOM_SEED()
  CALL RANDOM_NUMBER(f1d)
  !$acc enter data copyin(f1d)

  CALL SYSTEM_CLOCK(t1, t_rate)
  CALL run_raw_openacc(f1d, ref_x, ref_y, ref_z)
  CALL SYSTEM_CLOCK(t2, t_rate)
  CALL SYSTEM_CLOCK(t2)

  t_acc = REAL(t2-t1,8)/REAL(t_rate,8)
  PRINT *, " [openACC] Total Time (Full Physics): ", t_acc, " s"

  CALL SYSTEM_CLOCK(t1, t_rate)
  CALL run_device_vector_ver2(f1d, ref_x, ref_y, ref_z)
  CALL SYSTEM_CLOCK(t2, t_rate)
  CALL SYSTEM_CLOCK(t2)

  t_dv2 = REAL(t2-t1,8)/REAL(t_rate,8)
  PRINT *, " [DeviceVector] Total Time (Full Physics): ", t_dv2, " s"

  DEALLOCATE(f1d)
  CALL device_env_finalize()

  ! ==================================================================
  ! Report
  ! ==================================================================
  PRINT *, "=========================================================="
  PRINT *, "                 FINAL TRUE RESULTS                       "
  PRINT *, "=========================================================="
  PRINT '(A, F10.4, A)', " [1] OpenACC Time              : ", t_acc, " s"
  PRINT '(A, F10.4, A)', " [2] DeviceVector_Improve Time : ", t_dv2,  " s"
  PRINT *, "----------------------------------------------------------"
  PRINT '(A, F10.2, A)', " Speedup (DV_improve vs OpenACC):         ", t_acc / t_dv2, " x"
  PRINT *, "=========================================================="

CONTAINS

SUBROUTINE run_device_vector_ver2(host_data, check_x, check_y, check_z)
    USE Device_Vector
    IMPLICIT NONE
    REAL(4), INTENT(IN) :: host_data(:)
    REAL(4), INTENT(IN) :: check_x(:), check_y(:), check_z(:)
    
    TYPE(device_vector_r4_t) :: dv_f1d 

    REAL(4), POINTER :: p_f1d(:)

    TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
    REAL(4), POINTER, CONTIGUOUS :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
    
    REAL(4), ALLOCATABLE :: h_ax(:), h_ay(:), h_az(:)
    REAL(4) :: max_err
    INTEGER(8) :: i, i_step, n, ii, jj, kk, off8
    REAL(4) :: curr_x, curr_y, curr_z, start_x, start_y, start_z
    REAL(4) :: tk1x, tk1y, tk1z, tk2x, tk2y, tk2z, tk3x, tk3y, tk3z, tk4x, tk4y, tk4z
    REAL(4) :: fx, fy, fz, wx, wy, wz, w00, w10, w0, w1

    CALL dv_f1d%create_buffer(INT(SIZE(host_data), 8))
    
    dv_f1d%ptr(:) = host_data(:)
    
    
    CALL dv_f1d%upload() 
    
    ! 映射 Device Memory
    CALL dv_f1d%acc_map() 
    
    p_f1d => dv_f1d%ptr
    CALL px%create_buffer(N_P);  CALL py%create_buffer(N_P);  CALL pz%create_buffer(N_P)
    CALL vx%create_buffer(N_P);  CALL vy%create_buffer(N_P);  CALL vz%create_buffer(N_P)
    
    CALL px%acc_map(ax);    
    CALL py%acc_map(ay);    
    CALL pz%acc_map(az)
    CALL vx%acc_map(aux);   
    CALL vy%acc_map(auy);   
    CALL vz%acc_map(auz)

    !$acc parallel loop present(ax, ay, az, aux, auy, auz)
    DO n = 1, N_P
       ax(n)=32.5; ay(n)=32.5; az(n)=32.5; aux(n)=1.0; auy(n)=0.0; auz(n)=0.0
    END DO
    CALL device_synchronize() 

    ! p_f1d 指向 dv_f1d%ptr，該位址已被 acc_map 註冊
    !$acc data present(p_f1d, ax, ay, az)
    DO i_step = 1, N_S
       !$acc parallel loop gang vector_length(WARP_LENGTH) &
       !$acc private(curr_x, curr_y, curr_z, start_x, start_y, start_z) &
       !$acc private(fx, fy, fz, ii, jj, kk, wx, wy, wz, off8, w00, w10, w0, w1) &
       !$acc private(tk1x, tk1y, tk1z, tk2x, tk2y, tk2z) &
       !$acc private(tk3x, tk3y, tk3z, tk4x, tk4y, tk4z)
       DO n = 1, N_P
          start_x = ax(n); start_y = ay(n); start_z = az(n)

          ! Step 1
          curr_x = start_x; curr_y = start_y; curr_z = start_z
          fx=curr_x/DX; fy=curr_y/DY; fz=curr_z/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          
          ! 使用 p_f1d
          w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk1x=w0*(1.-wz)+w1*wz
          
          off8=off8+GXYZ; w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk1y=w0*(1.-wz)+w1*wz
          
          off8=off8+GXYZ; w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk1z=w0*(1.-wz)+w1*wz

          ! Step 2
          curr_x=start_x+0.5*DT*tk1x; curr_y=start_y+0.5*DT*tk1y; curr_z=start_z+0.5*DT*tk1z
          fx=curr_x/DX; fy=curr_y/DY; fz=curr_z/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          
          w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk2x=w0*(1.-wz)+w1*wz
          
          off8=off8+GXYZ; w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk2y=w0*(1.-wz)+w1*wz
          
          off8=off8+GXYZ; w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk2z=w0*(1.-wz)+w1*wz

          ! Step 3
          curr_x=start_x+0.5*DT*tk2x; curr_y=start_y+0.5*DT*tk2y; curr_z=start_z+0.5*DT*tk2z
          fx=curr_x/DX; fy=curr_y/DY; fz=curr_z/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          
          w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk3x=w0*(1.-wz)+w1*wz
          
          off8=off8+GXYZ; w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk3y=w0*(1.-wz)+w1*wz
          
          off8=off8+GXYZ; w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk3z=w0*(1.-wz)+w1*wz

          ! Step 4
          curr_x=start_x+DT*tk3x; curr_y=start_y+DT*tk3y; curr_z=start_z+DT*tk3z
          fx=curr_x/DX; fy=curr_y/DY; fz=curr_z/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          
          w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk4x=w0*(1.-wz)+w1*wz
          
          off8=off8+GXYZ; w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk4y=w0*(1.-wz)+w1*wz
          
          off8=off8+GXYZ; w00=p_f1d(off8)*(1.-wx)+p_f1d(off8+1)*wx; w10=p_f1d(off8+GX)*(1.-wx)+p_f1d(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=p_f1d(off8+GXY)*(1.-wx)+p_f1d(off8+GXY+1)*wx; w10=p_f1d(off8+GXY+GX)*(1.-wx)+p_f1d(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          tk4z=w0*(1.-wz)+w1*wz

          ! Final Update
          ax(n)=start_x+(DT/6.0_4)*(tk1x+2.*tk2x+2.*tk3x+tk4x)
          ay(n)=start_y+(DT/6.0_4)*(tk1y+2.*tk2y+2.*tk3y+tk4y)
          az(n)=start_z+(DT/6.0_4)*(tk1z+2.*tk2z+2.*tk3z+tk4z)
       END DO
    END DO
    !$acc end data

    CALL device_synchronize()
    
    ! 驗證
    ALLOCATE(h_ax(N_P), h_ay(N_P), h_az(N_P))
    !$acc update host(ax, ay, az)
    h_ax=ax(1:N_P); h_ay=ay(1:N_P); h_az=az(1:N_P)
    max_err=0.0
    DO i=1,N_P
       max_err=MAX(max_err,ABS(h_ax(i)-check_x(i)))
       max_err=MAX(max_err,ABS(h_ay(i)-check_y(i)))
       max_err=MAX(max_err,ABS(h_az(i)-check_z(i)))
    END DO
    PRINT *, "    [Validation] Max Error: ", max_err
    DEALLOCATE(h_ax, h_ay, h_az)

    CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
    CALL vx%acc_unmap(); CALL vy%acc_unmap(); CALL vz%acc_unmap()
    CALL dv_f1d%acc_unmap() 

    CALL px%free(); CALL py%free(); CALL pz%free()
    CALL vx%free(); CALL vy%free(); CALL vz%free()
    CALL dv_f1d%free() 

  END SUBROUTINE run_device_vector_ver2

  SUBROUTINE run_raw_openacc(host_data, out_x, out_y, out_z)
   IMPLICIT NONE
    REAL(4), INTENT(IN) :: host_data(:)
    REAL(4), INTENT(OUT) :: out_x(:), out_y(:), out_z(:) ! Output for verification   REAL(4), ALLOCATABLE :: ax(:), ay(:), az(:)
   REAL(4), ALLOCATABLE :: k1x(:), k1y(:), k1z(:), k2x(:), k2y(:), k2z(:)
   REAL(4), ALLOCATABLE :: k3x(:), k3y(:), k3z(:), k4x(:), k4y(:), k4z(:)
   
   INTEGER(8) :: n, i_step, off8
   REAL(4)    :: tx, ty, tz, fx, fy, fz, wx, wy, wz, w00, w10, w0, w1
   INTEGER(8) :: ii, jj, kk

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
      
! Step 1
       !$acc parallel loop gang vector_length(WARP_LENGTH) present(host_data, ax, ay, az, k1x, k1y, k1z) &
       !$acc private(fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
       DO n = 1, N_P
          fx=ax(n)/DX; fy=ay(n)/DY; fz=az(n)/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k1x(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k1y(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k1z(n)=w0*(1.-wz)+w1*wz
       END DO

       ! Step 2
       !$acc parallel loop gang vector_length(WARP_LENGTH) present(host_data, ax, ay, az, k1x, k1y, k1z, k2x, k2y, k2z) &
       !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
       DO n = 1, N_P
          tx=ax(n)+0.5*DT*k1x(n); ty=ay(n)+0.5*DT*k1y(n); tz=az(n)+0.5*DT*k1z(n)
          fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k2x(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k2y(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k2z(n)=w0*(1.-wz)+w1*wz
       END DO

       ! Step 3
       !$acc parallel loop gang vector_length(WARP_LENGTH) present(host_data, ax, ay, az, k2x, k2y, k2z, k3x, k3y, k3z) &
       !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
       DO n = 1, N_P
          tx=ax(n)+0.5*DT*k2x(n); ty=ay(n)+0.5*DT*k2y(n); tz=az(n)+0.5*DT*k2z(n)
          fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k3x(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k3y(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k3z(n)=w0*(1.-wz)+w1*wz
       END DO

       ! Step 4
       !$acc parallel loop gang vector_length(WARP_LENGTH) present(host_data, ax, ay, az, k3x, k3y, k3z, k4x, k4y, k4z) &
       !$acc private(tx,ty,tz,fx,fy,fz,ii,jj,kk,wx,wy,wz,off8,w00,w10,w0,w1)
       DO n = 1, N_P
          tx=ax(n)+DT*k3x(n); ty=ay(n)+DT*k3y(n); tz=az(n)+DT*k3z(n)
          fx=tx/DX; fy=ty/DY; fz=tz/DZ; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
          IF(ii<0)ii=0; IF(ii>GX-2)ii=GX-2; IF(jj<0)jj=0; IF(jj>GY-2)jj=GY-2; IF(kk<0)kk=0; IF(kk>GZ-2)kk=GZ-2
          wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
          off8=kk*GXY+jj*GX+ii+1_8
          w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k4x(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k4y(n)=w0*(1.-wz)+w1*wz
          off8=off8+GXYZ; w00=host_data(off8)*(1.-wx)+host_data(off8+1)*wx; w10=host_data(off8+GX)*(1.-wx)+host_data(off8+GX+1)*wx; w0=w00*(1.-wy)+w10*wy
          w00=host_data(off8+GXY)*(1.-wx)+host_data(off8+GXY+1)*wx; w10=host_data(off8+GXY+GX)*(1.-wx)+host_data(off8+GXY+GX+1)*wx; w1=w00*(1.-wy)+w10*wy
          k4z(n)=w0*(1.-wz)+w1*wz
       END DO

       ! Update
       !$acc parallel loop gang vector_length(128) present(ax, ay, az, k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z)
       DO n = 1, N_P
          ax(n) = ax(n) + (DT/6.0_4)*(k1x(n) + 2.0_4*k2x(n) + 2.0_4*k3x(n) + k4x(n))
          ay(n) = ay(n) + (DT/6.0_4)*(k1y(n) + 2.0_4*k2y(n) + 2.0_4*k3y(n) + k4y(n))
          az(n) = az(n) + (DT/6.0_4)*(k1z(n) + 2.0_4*k2z(n) + 2.0_4*k3z(n) + k4z(n))
       END DO
    END DO

    CALL device_synchronize()

    ! ★★★ 關鍵：把結果拉回 Host 陣列 ★★★
    !$acc update host(ax, ay, az)
    out_x = ax
    out_y = ay
    out_z = az

    !$acc exit data delete(ax, ay, az, host_data, k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z)
    DEALLOCATE(ax, ay, az, k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z)
  END SUBROUTINE run_raw_openacc

END PROGRAM