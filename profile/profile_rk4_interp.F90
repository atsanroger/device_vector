PROGRAM test_rk4_dv
  USE Device_Vector
  USE openacc
  IMPLICIT NONE

  ! 維度設定
  INTEGER, PARAMETER :: GX = 64, GY = 64, GZ = 64
  INTEGER(8), PARAMETER :: GXY = 64_8 * 64_8
  INTEGER(8), PARAMETER :: GXYZ = 64_8 * 64_8 * 64_8
  REAL(4), PARAMETER :: DX = 1.0, DY = 1.0, DZ = 1.0
  REAL(4), PARAMETER :: DT = 0.01_4

  INTEGER(8), PARAMETER :: N_P = 1000000_8
  INTEGER,    PARAMETER :: N_S = 100 

  ! Device Vector 容器
  TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
  REAL(4), POINTER :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
  
  ! ★ 關鍵：將場改為一維陣列，徹底消除維度計算壓力
  REAL(4), ALLOCATABLE, TARGET :: field_1d(:)
  INTEGER(8) :: i_step, n, c1, c2, cr, idx8
  REAL(8)    :: time_total

  ! 私有變數
  REAL(4) :: cx, cy, cz, cu, cv, cw, tx, ty, tz, kx, ky, kz
  REAL(4) :: fx, fy, fz, wx, wy, wz, c00, c10, c01, c11, c0, c1
  INTEGER(8) :: i, j, k

  CALL device_env_init(0, 1)

  ! 1. 分配一維場 (GX*GY*GZ*3)
  ALLOCATE(field_1d(GXYZ * 3))
  field_1d = 0.1_4
  !$acc enter data copyin(field_1d)

  ! 2. 建立 Buffer
  CALL px%create_buffer(N_P, .TRUE.); CALL py%create_buffer(N_P, .TRUE.); CALL pz%create_buffer(N_P, .TRUE.)
  CALL vx%create_buffer(N_P, .TRUE.); CALL vy%create_buffer(N_P, .TRUE.); CALL vz%create_buffer(N_P, .TRUE.)

  ! 3. 映射指針
  CALL px%acc_map(ax); CALL py%acc_map(ay); CALL pz%acc_map(az)
  CALL vx%acc_map(aux); CALL vy%acc_map(auy); CALL vz%acc_map(auz)

  ! 4. GPU 初始化
  !$acc parallel loop present(ax, ay, az, aux, auy, auz)
  DO n = 1, N_P
     ax(n)=0.5; ay(n)=0.5; az(n)=0.5; aux(n)=0.0; auy(n)=0.0; auz(n)=0.0
  END DO

  PRINT *, "[Run] Starting Manual-Offset RK4..."
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(c1, cr)

  DO i_step = 1, N_S
     !$acc parallel loop present(field_1d, ax, ay, az, aux, auy, auz) &
     !$acc private(cx, cy, cz, cu, cv, cw, tx, ty, tz, kx, ky, kz, &
     !$acc         fx, fy, fz, wx, wy, wz, c00, c10, c01, c11, c0, c1, i, j, k, idx8)
     DO n = 1, N_P
        cx = ax(n); cy = ay(n); cz = az(n)
        cu = aux(n); cv = auy(n); cw = auz(n)

        ! --- RK4 Step 1 (K1) ---
        tx = cx; ty = cy; tz = cz
        fx = tx/DX; fy = ty/DY; fz = tz/DZ
        i = MAX(0_8, MIN(INT(GX-2,8), INT(fx,8)))
        j = MAX(0_8, MIN(INT(GY-2,8), INT(fy,8)))
        k = MAX(0_8, MIN(INT(GZ-2,8), INT(fz,8)))
        wx = fx-REAL(i,4); wy = fy-REAL(j,4); wz = fz-REAL(k,4)

        ! 手動計算偏移 (i,j,k,1) -> k*GXY + j*GX + i + 1
        ! X-Dim (Offset 0)
        idx8 = k*GXY + j*GX + i + 1
        c00 = field_1d(idx8)*(1.-wx) + field_1d(idx8+1)*wx
        c10 = field_1d(idx8+GX)*(1.-wx) + field_1d(idx8+GX+1)*wx
        c0 = c00*(1.-wy) + c10*wy
        idx8 = idx8 + GXY
        c01 = field_1d(idx8)*(1.-wx) + field_1d(idx8+1)*wx
        c11 = field_1d(idx8+GX)*(1.-wx) + field_1d(idx8+GX+1)*wx
        c1 = c01*(1.-wy) + c11*wy
        kx = c0*(1.-wz) + c1*wz
        aux(n) = cu + (DT/6.0_4)*kx ! 這裡先存在 aux 做中間值

        ! Y-Dim (Offset GXYZ)
        idx8 = k*GXY + j*GX + i + 1 + GXYZ
        c00 = field_1d(idx8)*(1.-wx) + field_1d(idx8+1)*wx
        c10 = field_1d(idx8+GX)*(1.-wx) + field_1d(idx8+GX+1)*wx
        c0 = c00*(1.-wy) + c10*wy
        idx8 = idx8 + GXY
        c01 = field_1d(idx8)*(1.-wx) + field_1d(idx8+1)*wx
        c11 = field_1d(idx8+GX)*(1.-wx) + field_1d(idx8+GX+1)*wx
        c1 = c01*(1.-wy) + c11*wy
        ky = c0*(1.-wz) + c1*wz
        auy(n) = cv + (DT/6.0_4)*ky

        ! Z-Dim (Offset 2*GXYZ)
        idx8 = k*GXY + j*GX + i + 1 + 2*GXYZ
        c00 = field_1d(idx8)*(1.-wx) + field_1d(idx8+1)*wx
        c10 = field_1d(idx8+GX)*(1.-wx) + field_1d(idx8+GX+1)*wx
        c0 = c00*(1.-wy) + c10*wy
        idx8 = idx8 + GXY
        c01 = field_1d(idx8)*(1.-wx) + field_1d(idx8+1)*wx
        c11 = field_1d(idx8+GX)*(1.-wx) + field_1d(idx8+GX+1)*wx
        c1 = c01*(1.-wy) + c11*wy
        kz = c0*(1.-wz) + c1*wz
        auz(n) = cw + (DT/6.0_4)*kz

        ! 最終更新粒子位置 (這裡為了測試先簡化成 RK1)
        ax(n) = cx + DT*aux(n)
        ay(n) = cy + DT*auy(n)
        az(n) = cz + DT*auz(n)
     END DO
  END DO

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(c2)
  time_total = REAL(c2 - c1, 8) / REAL(cr, 8)
  PRINT '(A, F10.4, A)', " [Result] Total Time: ", time_total, " s"

  ! 清理
  CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
  CALL vx%acc_unmap(); CALL vy%acc_unmap(); CALL vz%acc_unmap()
  CALL px%free(); CALL py%free(); CALL pz%free(); CALL vx%free(); CALL vy%free(); CALL vz%free()
  !$acc exit data delete(field_1d)
  DEALLOCATE(field_1d)
  CALL device_env_finalize()
END PROGRAM