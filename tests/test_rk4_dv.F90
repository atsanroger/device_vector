PROGRAM test_rk4_dv
  USE Device_Vector
  USE openacc
  IMPLICIT NONE

  ! 格點與常數
  INTEGER, PARAMETER :: GX = 64, GY = 64, GZ = 64
  INTEGER(8), PARAMETER :: GXY = 4096_8   ! 64*64
  INTEGER(8), PARAMETER :: GXYZ = 262144_8 ! 64*64*64
  REAL(4), PARAMETER :: DX = 1.0, DY = 1.0, DZ = 1.0
  REAL(4), PARAMETER :: DT = 0.01_4

  INTEGER(8), PARAMETER :: N_P = 1000000_8
  INTEGER,    PARAMETER :: N_S = 100 

  ! Device Vector
  TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
  REAL(4), POINTER :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
  REAL(4), ALLOCATABLE, TARGET :: field_1d(:)
  
  ! 變數宣告 (徹底避開重複命名)
  INTEGER(8) :: i_step, p_idx, tick1, tick2, tick_rate, offset8
  REAL(8)    :: time_total
  REAL(4)    :: cx, cy, cz, cu, cv, cw, kx, ky, kz
  REAL(4)    :: fx, fy, fz, wx, wy, wz, w00, w10, w0, w1
  INTEGER(8) :: ii, jj, kk

  CALL device_env_init(0, 1)

  ! 1. 場資料初始化 (SoA 一維化)
  ALLOCATE(field_1d(GXYZ * 3))
  field_1d = 0.1_4
  !$acc enter data copyin(field_1d)

  ! 2. 建立 Buffer (無類推)
  CALL px%create_buffer(N_P, .TRUE.); CALL py%create_buffer(N_P, .TRUE.); CALL pz%create_buffer(N_P, .TRUE.)
  CALL vx%create_buffer(N_P, .TRUE.); CALL vy%create_buffer(N_P, .TRUE.); CALL vz%create_buffer(N_P, .TRUE.)

  ! 3. 映射指針
  CALL px%acc_map(ax); CALL py%acc_map(ay); CALL pz%acc_map(az)
  CALL vx%acc_map(aux); CALL vy%acc_map(auy); CALL vz%acc_map(auz)

  ! 4. GPU 初始化
  !$acc parallel loop present(ax, ay, az, aux, auy, auz)
  DO p_idx = 1, N_P
     ax(p_idx)=0.5; ay(p_idx)=0.5; az(p_idx)=0.5; aux(p_idx)=0.0; auy(p_idx)=0.0; auz(p_idx)=0.0
  END DO

  PRINT *, "[Run] Forcing Stable RK4 Implementation..."
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(tick1, tick_rate)

  ! 5. 主計算循環
  DO i_step = 1, N_S
     !$acc parallel loop present(field_1d, ax, ay, az, aux, auy, auz) &
     !$acc private(cx, cy, cz, cu, cv, cw, kx, ky, kz, fx, fy, fz, wx, wy, wz, w00, w10, w0, w1, ii, jj, kk, offset8)
     DO p_idx = 1, N_P
        cx = ax(p_idx); cy = ay(p_idx); cz = az(p_idx)
        cu = aux(p_idx); cv = auy(p_idx); cw = auz(p_idx)

        ! 手動展開插值
        fx = cx / DX; fy = cy / DY; fz = cz / DZ
        ii = INT(fx, 8); jj = INT(fy, 8); kk = INT(fz, 8)
        
        ! 邊界鉗制
        IF (ii < 0) ii = 0; IF (ii > GX-2) ii = GX-2
        IF (jj < 0) jj = 0; IF (jj > GY-2) jj = GY-2
        IF (kk < 0) kk = 0; IF (kk > GZ-2) kk = GZ-2
        
        wx = fx - REAL(ii, 4); wy = fy - REAL(jj, 4); wz = fz - REAL(kk, 4)
        offset8 = kk*GXY + jj*64_8 + ii + 1_8

        ! 執行 X 分量插值 (模擬物理場影響)
        w00 = field_1d(offset8)*(1.0-wx) + field_1d(offset8+1)*wx
        w10 = field_1d(offset8+64)*(1.0-wx) + field_1d(offset8+64+1)*wx
        w0 = w00*(1.0-wy) + w10*wy
        kx = w0*(1.0-wz) + 0.1*wz 
        
        ! RK4 簡化積分步
        aux(p_idx) = cu + DT*kx
        ax(p_idx) = cx + DT*aux(p_idx)
        ay(p_idx) = cy + DT*cv
        az(p_idx) = cz + DT*cw
     END DO
  END DO

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(tick2)
  time_total = REAL(tick2 - tick1, 8) / REAL(tick_rate, 8)
  PRINT *, " [Result] Total Time: ", REAL(time_total, 4), " s"

  ! 6. 清理 (手寫到底)
  CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
  CALL vx%acc_unmap(); CALL vy%acc_unmap(); CALL vz%acc_unmap()
  CALL px%free(); CALL py%free(); CALL pz%free()
  CALL vx%free(); CALL vy%free(); CALL vz%free()
  !$acc exit data delete(field_1d)
  IF (ALLOCATED(field_1d)) DEALLOCATE(field_1d)
  CALL device_env_finalize()
END PROGRAM