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

  !PRINT *, "[Run] Physics RK4  on GPU..."
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
