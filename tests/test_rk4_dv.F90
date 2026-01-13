MODULE rk4_interp_mod
  USE openacc
  IMPLICIT NONE
  ! 固定維度參數，協助編譯器優化
  INTEGER, PARAMETER :: GX = 64, GY = 64, GZ = 64
  REAL(4), PARAMETER :: DX = 1.0, DY = 1.0, DZ = 1.0

CONTAINS

  !$acc routine seq
  SUBROUTINE get_velocity_isolated(px, py, pz, f, vx, vy, vz)
    REAL(4), INTENT(IN) :: px, py, pz, f(GX, GY, GZ, 3)
    REAL(4), INTENT(OUT) :: vx, vy, vz
    REAL(4) :: fx, fy, fz, wx, wy, wz, c00, c01, c10, c11, c0, c1
    INTEGER :: i, j, k

    ! 座標轉換
    fx = px/DX; fy = py/DY; fz = pz/DZ
    i = MAX(1, MIN(GX-1, FLOOR(fx)+1))
    j = MAX(1, MIN(GY-1, FLOOR(fy)+1))
    k = MAX(1, MIN(GZ-1, FLOOR(fz)+1))
    wx = fx-FLOOR(fx); wy = fy-FLOOR(fy); wz = fz-FLOOR(fz)

    ! X 分量插值 (標量展開)
    c00 = f(i,j,k,1)*(1-wx) + f(i+1,j,k,1)*wx
    c10 = f(i,j+1,k,1)*(1-wx) + f(i+1,j+1,k,1)*wx
    c01 = f(i,j,k+1,1)*(1-wx) + f(i+1,j,k+1,1)*wx
    c11 = f(i,j+1,k+1,1)*(1-wx) + f(i+1,j+1,k+1,1)*wx
    vx = (c00*(1-wy) + c10*wy)*(1-wz) + (c01*(1-wy) + c11*wy)*wz

    ! Y 分量插值
    c00 = f(i,j,k,2)*(1-wx) + f(i+1,j,k,2)*wx
    c10 = f(i,j+1,k,2)*(1-wx) + f(i+1,j+1,k,2)*wx
    c01 = f(i,j,k+1,2)*(1-wx) + f(i+1,j,k+1,2)*wx
    c11 = f(i,j+1,k+1,2)*(1-wx) + f(i+1,j+1,k+1,2)*wx
    vy = (c00*(1-wy) + c10*wy)*(1-wz) + (c01*(1-wy) + c11*wy)*wz

    ! Z 分量插值
    c00 = f(i,j,k,3)*(1-wx) + f(i+1,j,k,3)*wx
    c10 = f(i,j+1,k,3)*(1-wx) + f(i+1,j+1,k,3)*wx
    c01 = f(i,j,k+1,3)*(1-wx) + f(i+1,j,k+1,3)*wx
    c11 = f(i,j+1,k+1,3)*(1-wx) + f(i+1,j+1,k+1,3)*wx
    vz = (c00*(1-wy) + c10*wy)*(1-wz) + (c01*(1-wy) + c11*wy)*wz
  END SUBROUTINE
END MODULE rk4_interp_mod

PROGRAM test_rk4_dv
  USE Device_Vector
  USE rk4_interp_mod
  USE openacc
  IMPLICIT NONE

  INTEGER(8), PARAMETER :: N_P = 1000000_8
  INTEGER,    PARAMETER :: N_S = 100 
  REAL(4),    PARAMETER :: DT  = 0.01_4

  ! SoA 結構設計
  TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
  REAL(4), POINTER :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
  
  REAL(4), ALLOCATABLE, TARGET :: field_data(:,:,:,:)
  INTEGER(8) :: i_step, n, c1, c2, cr
  REAL(8) :: time_total

  ! 暫存變數
  REAL(4) :: cx, cy, cz, cu, cv, cw, tx, ty, tz
  REAL(4) :: k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z

  CALL device_env_init(0, 1)

  PRINT *, "Initializing Field..."
  ALLOCATE(field_data(GX, GY, GZ, 3))
  field_data = 0.1_4
  !$acc enter data copyin(field_data)

  PRINT *, "Allocating Device Vectors..."
  CALL px%create_buffer(N_P, .TRUE.)
  CALL py%create_buffer(N_P, .TRUE.)
  CALL pz%create_buffer(N_P, .TRUE.)
  CALL vx%create_buffer(N_P, .TRUE.)
  CALL vy%create_buffer(N_P, .TRUE.)
  CALL vz%create_buffer(N_P, .TRUE.)

  CALL px%acc_map(ax); CALL py%acc_map(ay); CALL pz%acc_map(az)
  CALL vx%acc_map(aux); CALL vy%acc_map(auy); CALL vz%acc_map(auz)

  !$acc parallel loop present(ax, ay, az, aux, auy, auz)
  DO n = 1, N_P
     ax(n) = 0.5; ay(n) = 0.5; az(n) = 0.5
     aux(n) = 0.0; auy(n) = 0.0; auz(n) = 0.0
  END DO

  PRINT *, "Running RK4 Integration on GPU..."
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(c1, cr)

  DO i_step = 1, N_S
     !$acc parallel loop present(field_data, ax, ay, az, aux, auy, auz) &
     !$acc private(cx, cy, cz, cu, cv, cw, tx, ty, tz, &
     !$acc         k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z)
     DO n = 1, N_P
        cx = ax(n); cy = ay(n); cz = az(n)
        cu = aux(n); cv = auy(n); cw = auz(n)
        
        ! Step 1
        CALL get_velocity_isolated(cx, cy, cz, field_data, k1x, k1y, k1z)
        
        ! Step 2
        tx = cx + 0.5*DT*cu; ty = cy + 0.5*DT*cv; tz = cz + 0.5*DT*cw
        CALL get_velocity_isolated(tx, ty, tz, field_data, k2x, k2y, k2z)
        
        ! Step 3 (Midpoint again)
        tx = cx + 0.5*DT*k2x; ty = cy + 0.5*DT*k2y; tz = cz + 0.5*DT*k2z
        CALL get_velocity_isolated(tx, ty, tz, field_data, k3x, k3y, k3z)
        
        ! Step 4
        tx = cx + DT*k3x; ty = cy + DT*k3y; tz = cz + DT*k3z
        CALL get_velocity_isolated(tx, ty, tz, field_data, k4x, k4y, k4z)
        
        ! Update
        aux(n) = cu + (DT/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
        auy(n) = cv + (DT/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        auz(n) = cw + (DT/6.0)*(k1z + 2*k2z + 2*k3z + k4z)
        
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
  CALL px%free(); CALL py%free(); CALL pz%free()
  CALL vx%free(); CALL vy%free(); CALL vz%free()
  !$acc exit data delete(field_data)
  DEALLOCATE(field_data)
  CALL device_env_finalize()
END PROGRAM