PROGRAM template_rk4_fusion
  USE Device_Vector
  USE openacc
  IMPLICIT NONE

  ! --- 1. 物理與網格參數 ---
  INTEGER(8) :: N_P, N_S
  REAL(4)    :: DT, DX, DY, DZ
  INTEGER(8) :: GX, GY, GZ, GXY, GXYZ
  INTEGER    :: ios

  ! --- 2. 粒子資料 (僅保留位置與速度) ---
  TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
  REAL(4), POINTER, CONTIGUOUS :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
  
  ! --- 3. 場資料 ---
  REAL(4), ALLOCATABLE, TARGET :: f1d(:)
  
  ! --- 4. 效能計時 ---
  INTEGER(8) :: i_step, n, t1, t2, t_rate
  
  ! --- 5. 硬體參數 ---
  INTEGER, PARAMETER :: WARP_LENGTH = 128

  NAMELIST /sim_config/ N_P, N_S, DT, DX, DY, DZ, GX, GY, GZ

  ! =================================================================
  ! [Step A] 初始化環境與讀取參數
  ! =================================================================
  PRINT *, "[Init] Loading parameters..."
  OPEN(UNIT=10, FILE='../configs/test_rk4.nml', STATUS='OLD', IOSTAT=ios)
  IF (ios == 0) THEN
     READ(10, NML=sim_config)
     GXY  = GX * GY    
     GXYZ = GXY * GZ   
     CLOSE(10)
  ELSE
     PRINT *, "[Error] nml not found!"; STOP
  END IF

  CALL device_env_init(0, 1)

  ! 分配場空間 (假設是一個隨機的向量場)
  ALLOCATE(f1d(GXYZ * 3))
  CALL RANDOM_SEED()
  CALL RANDOM_NUMBER(f1d)
  !$acc enter data copyin(f1d)

  ! =================================================================
  ! [Step B] 建立與映射 Device Vector
  ! =================================================================
  CALL px%create_buffer(N_P); CALL py%create_buffer(N_P); CALL pz%create_buffer(N_P)
  CALL vx%create_buffer(N_P); CALL vy%create_buffer(N_P); CALL vz%create_buffer(N_P)

  CALL px%acc_map(ax);  CALL py%acc_map(ay);  CALL pz%acc_map(az)
  CALL vx%acc_map(aux); CALL vy%acc_map(auy); CALL vz%acc_map(auz)

  ! 初始化粒子位置與速度
  !$acc parallel loop present(ax, ay, az, aux, auy, auz)
  DO n = 1, N_P
     ax(n)=32.5; ay(n)=32.5; az(n)=32.5
     aux(n)=1.0; auy(n)=0.0; auz(n)=0.0
  END DO

  ! =================================================================
  ! [Step C] 核心運算：RK4 Kernel Fusion
  ! =================================================================
  PRINT *, "[Run] Starting RK4 Fusion Loop..."
  CALL device_synchronize() 
  CALL SYSTEM_CLOCK(t1, t_rate)

  !$acc data present(f1d, ax, ay, az)
  DO i_step = 1, N_S
     
     !$acc parallel loop gang vector_length(WARP_LENGTH) &
     !$acc private(n)
     DO n = 1, N_P
        CALL rk4_step_core(n, ax, ay, az, f1d, DT, DX, DY, DZ, GX, GY, GZ, GXY, GXYZ)
     END DO

  END DO
  !$acc end data

  CALL device_synchronize() 
  CALL SYSTEM_CLOCK(t2)

  PRINT *, "[Done] Total Time: ", REAL(t2-t1,8)/REAL(t_rate,8), " s"

  ! =================================================================
  ! [Step D] 清理資源
  ! =================================================================
  CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
  CALL vx%acc_unmap(); CALL vy%acc_unmap(); CALL vz%acc_unmap()
  
  CALL px%free(); CALL py%free(); CALL pz%free()
  CALL vx%free(); CALL vy%free(); CALL vz%free()

  !$acc exit data delete(f1d)
  DEALLOCATE(f1d)
  CALL device_env_finalize()

CONTAINS

  ! ---------------------------------------------------------
  ! [Core Routine] RK4 單步計算核心 (已 Inline 處理以避開編譯 Bug)
  ! ---------------------------------------------------------
  SUBROUTINE rk4_step_core(pid, px_arr, py_arr, pz_arr, field, &
                           dt_val, dx_val, dy_val, dz_val, &
                           gx8, gy8, gz8, gxy8, gxyz8)
    !$acc routine seq
    INTEGER(8), INTENT(IN) :: pid
    REAL(4), INTENT(INOUT) :: px_arr(*), py_arr(*), pz_arr(*)
    REAL(4), INTENT(IN)    :: field(*)
    REAL(4), INTENT(IN)    :: dt_val, dx_val, dy_val, dz_val
    INTEGER(8), INTENT(IN) :: gx8, gy8, gz8, gxy8, gxyz8

    REAL(4) :: s_x, s_y, s_z, c_x, c_y, c_z
    REAL(4) :: k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z

    ! 紀錄原始起點
    s_x = px_arr(pid); s_y = py_arr(pid); s_z = pz_arr(pid)

    ! --- Step 1: K1 ---
    c_x = s_x; c_y = s_y; c_z = s_z
    CALL get_interp_field(c_x, c_y, c_z, field, gx8, gy8, gz8, gxy8, gxyz8, dx_val, dy_val, dz_val, k1x, k1y, k1z)

    ! --- Step 2: K2 ---
    c_x = s_x + 0.5*dt_val*k1x; c_y = s_y + 0.5*dt_val*k1y; c_z = s_z + 0.5*dt_val*k1z
    CALL get_interp_field(c_x, c_y, c_z, field, gx8, gy8, gz8, gxy8, gxyz8, dx_val, dy_val, dz_val, k2x, k2y, k2z)

    ! --- Step 3: K3 ---
    c_x = s_x + 0.5*dt_val*k2x; c_y = s_y + 0.5*dt_val*k2y; c_z = s_z + 0.5*dt_val*k2z
    CALL get_interp_field(c_x, c_y, c_z, field, gx8, gy8, gz8, gxy8, gxyz8, dx_val, dy_val, dz_val, k3x, k3y, k3z)

    ! --- Step 4: K4 ---
    c_x = s_x + dt_val*k3x; c_y = s_y + dt_val*k3y; c_z = s_z + dt_val*k3z
    CALL get_interp_field(c_x, c_y, c_z, field, gx8, gy8, gz8, gxy8, gxyz8, dx_val, dy_val, dz_val, k4x, k4y, k4z)

    ! --- Final Update ---
    px_arr(pid) = s_x + (dt_val/6.0_4) * (k1x + 2.0*k2x + 2.0*k3x + k4x)
    py_arr(pid) = s_y + (dt_val/6.0_4) * (k1y + 2.0*k2y + 2.0*k3y + k4y)
    pz_arr(pid) = s_z + (dt_val/6.0_4) * (k1z + 2.0*k2z + 2.0*k3z + k4z)

  END SUBROUTINE rk4_step_core

  ! ---------------------------------------------------------
  ! [Field Helper] 3D 插值計算
  ! ---------------------------------------------------------
  SUBROUTINE get_interp_field(x, y, z, f, gx, gy, gz, gxy, gxyz, dx, dy, dz, vx, vy, vz)
    !$acc routine seq
    REAL(4), INTENT(IN) :: x, y, z, f(*), dx, dy, dz
    INTEGER(8), INTENT(IN) :: gx, gy, gz, gxy, gxyz
    REAL(4), INTENT(OUT) :: vx, vy, vz
    INTEGER(8) :: ii, jj, kk, off8
    REAL(4) :: fx, fy, fz, wx, wy, wz, w00, w10, w0, w1

    fx=x/dx; fy=y/dy; fz=z/dz; ii=INT(fx,8); jj=INT(fy,8); kk=INT(fz,8)
    IF(ii<0)ii=0; IF(ii>gx-2)ii=gx-2; IF(jj<0)jj=0; IF(jj>gy-2)jj=gy-2; IF(kk<0)kk=0; IF(kk>gz-2)kk=gz-2
    wx=fx-REAL(ii,4); wy=fy-REAL(jj,4); wz=fz-REAL(kk,4)
    off8=kk*gxy+jj*gx+ii+1_8

    ! Component X
    w00=f(off8)*(1.-wx)+f(off8+1)*wx; w10=f(off8+gx)*(1.-wx)+f(off8+gx+1)*wx; w0=w00*(1.-wy)+w10*wy
    w00=f(off8+gxy)*(1.-wx)+f(off8+gxy+1)*wx; w10=f(off8+gxy+gx)*(1.-wx)+f(off8+gxy+gx+1)*wx; w1=w00*(1.-wy)+w10*wy
    vx = w0*(1.-wz)+w1*wz
    ! Component Y
    off8=off8+gxyz
    w00=f(off8)*(1.-wx)+f(off8+1)*wx; w10=f(off8+gx)*(1.-wx)+f(off8+gx+1)*wx; w0=w00*(1.-wy)+w10*wy
    w00=f(off8+gxy)*(1.-wx)+f(off8+gxy+1)*wx; w10=f(off8+gxy+gx)*(1.-wx)+f(off8+gxy+gx+1)*wx; w1=w00*(1.-wy)+w10*wy
    vy = w0*(1.-wz)+w1*wz
    ! Component Z
    off8=off8+gxyz
    w00=f(off8)*(1.-wx)+f(off8+1)*wx; w10=f(off8+gx)*(1.-wx)+f(off8+gx+1)*wx; w0=w00*(1.-wy)+w10*wy
    w00=f(off8+gxy)*(1.-wx)+f(off8+gxy+1)*wx; w10=f(off8+gxy+gx)*(1.-wx)+f(off8+gxy+gx+1)*wx; w1=w00*(1.-wy)+w10*wy
    vz = w0*(1.-wz)+w1*wz
  END SUBROUTINE get_interp_field

END PROGRAM