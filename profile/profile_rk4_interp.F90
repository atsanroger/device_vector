! ====================================================================
! 1. 物理場與補間模組 (維持原樣，但確保 routine seq 正確)
! ====================================================================
MODULE field_module
  USE iso_c_binding
  USE openacc
  IMPLICIT NONE
  INTEGER, PARAMETER :: GX = 64, GY = 64, GZ = 64
  REAL(4), PARAMETER :: DX = 1.0, DY = 1.0, DZ = 1.0
  REAL(4), ALLOCATABLE, TARGET :: field_data(:,:,:,:) 
CONTAINS
  FUNCTION interp_3d(pos, f_array) RESULT(val)
    !$acc routine seq
    REAL(4), INTENT(IN) :: pos(3)
    REAL(4), INTENT(IN) :: f_array(GX, GY, GZ, 3)
    REAL(4) :: val(3)
    INTEGER :: i, j, k
    REAL(4) :: fx, fy, fz, wx, wy, wz, c00, c01, c10, c11, c0, c1
    INTEGER :: dim
    fx = pos(1) / DX; fy = pos(2) / DY; fz = pos(3) / DZ
    i = FLOOR(fx) + 1; j = FLOOR(fy) + 1; k = FLOOR(fz) + 1
    i = MAX(1, MIN(GX-1, i)); j = MAX(1, MIN(GY-1, j)); k = MAX(1, MIN(GZ-1, k))
    wx = fx - FLOOR(fx); wy = fy - FLOOR(fy); wz = fz - FLOOR(fz)
    DO dim = 1, 3
       c00 = f_array(i,j,k,dim)*(1-wx) + f_array(i+1,j,k,dim)*wx
       c10 = f_array(i,j+1,k,dim)*(1-wx) + f_array(i+1,j+1,k,dim)*wx
       c01 = f_array(i,j,k+1,dim)*(1-wx) + f_array(i+1,j,k+1,dim)*wx
       c11 = f_array(i,j+1,k+1,dim)*(1-wx) + f_array(i+1,j+1,k+1,dim)*wx
       c0 = c00*(1-wy) + c10*wy; c1 = c01*(1-wy) + c11*wy
       val(dim) = c0*(1-wz) + c1*wz
    END DO
  END FUNCTION interp_3d
END MODULE field_module

! ====================================================================
! 2. 主程式：RK4 積分測試
! ====================================================================
PROGRAM profile_rk4_interp
  USE Device_Vector
  USE field_module
  USE openacc
  IMPLICIT NONE

  INTEGER(8), PARAMETER :: N_PARTICLES = 1000000_8
  INTEGER,    PARAMETER :: N_STEPS     = 100 
  REAL(4),    PARAMETER :: DT          = 0.01

  TYPE(device_vector_r4_t) :: px, py, pz, vx, vy, vz
  
  ! ★ 關鍵修正：宣告本地指針來接數據，解決 present 查表問題
  REAL(4), POINTER :: ax(:), ay(:), az(:), aux(:), auy(:), auz(:)
  
  INTEGER(8) :: i_step, n
  INTEGER(8) :: c1, c2, cr
  REAL(8)    :: time_total
  INTEGER :: i, j, k

  CALL device_env_init(0, 1)

  PRINT *, "[Init] Allocating Field Data..."
  ALLOCATE(field_data(GX, GY, GZ, 3))
  !$acc enter data create(field_data)
  
  !$acc parallel loop present(field_data) collapse(3)
  DO k = 1, GZ
     DO j = 1, GY
        DO i = 1, GX
           field_data(i,j,k, 1) = -REAL(j)*0.1
           field_data(i,j,k, 2) =  REAL(i)*0.1
           field_data(i,j,k, 3) =  0.1
        END DO
     END DO
  END DO

  PRINT *, "[Init] Creating Particle Vectors..."
  CALL px%create_buffer(N_PARTICLES, pinned=.TRUE.)
  CALL py%create_buffer(N_PARTICLES, pinned=.TRUE.)
  CALL pz%create_buffer(N_PARTICLES, pinned=.TRUE.)
  CALL vx%create_buffer(N_PARTICLES, pinned=.TRUE.)
  CALL vy%create_buffer(N_PARTICLES, pinned=.TRUE.)
  CALL vz%create_buffer(N_PARTICLES, pinned=.TRUE.)

  !$acc parallel loop present(px%ptr, py%ptr, pz%ptr, vx%ptr, vy%ptr, vz%ptr)
  DO n = 1, N_PARTICLES
     px%ptr(n) = MOD(n, INT(GX,8)) + 0.5
     py%ptr(n) = MOD(n, INT(GY,8)) + 0.5
     pz%ptr(n) = MOD(n, INT(GZ,8)) + 0.5
     vx%ptr(n) = 0.0; vy%ptr(n) = 0.0; vz%ptr(n) = 0.0
  END DO

  CALL px%upload(); CALL py%upload(); CALL pz%upload()
  CALL vx%upload(); CALL vy%upload(); CALL vz%upload()

  PRINT *, "[Map] Mapping DeviceVectors to OpenACC..."
  ! ★ 使用帶參數的 acc_map 自動提取指針
  CALL px%acc_map(ax); CALL py%acc_map(ay); CALL pz%acc_map(az)
  CALL vx%acc_map(aux); CALL vy%acc_map(auy); CALL vz%acc_map(auz)

  PRINT *, "[Run] Starting RK4 Integration..."
  CALL device_synchronize()
  CALL SYSTEM_CLOCK(c1, cr)

  DO i_step = 1, N_STEPS
     !$acc parallel loop present(field_data, ax, ay, az, aux, auy, auz)
     DO n = 1, N_PARTICLES
        ! ★ 傳遞本地指針而非 v%ptr，保證 GPU 地址直接引用
        CALL rk4_kernel_step(n, ax, ay, az, aux, auy, auz, field_data)
     END DO
  END DO

  CALL device_synchronize()
  CALL SYSTEM_CLOCK(c2)
  time_total = REAL(c2 - c1) / REAL(cr)

  PRINT '(A, F10.4, A)', " [Result] Total Time: ", time_total, " s"
  PRINT '(A, F10.2, A)', " [Result] Throughput: ", (REAL(N_PARTICLES)*REAL(N_STEPS))/time_total/1.0e6, " M updates/s"

  ! ★ 解除映射
  CALL px%acc_unmap(); CALL py%acc_unmap(); CALL pz%acc_unmap()
  CALL vx%acc_unmap(); CALL vy%acc_unmap(); CALL vz%acc_unmap()

  CALL px%free(); CALL py%free(); CALL pz%free()
  CALL vx%free(); CALL vy%free(); CALL vz%free()
  !$acc exit data delete(field_data)
  DEALLOCATE(field_data)
  CALL device_env_finalize()

CONTAINS

  !$acc routine seq
  SUBROUTINE rk4_kernel_step(idx, x, y, z, u, v, w, f)
    INTEGER(8), INTENT(IN) :: idx
    REAL(4), INTENT(INOUT) :: x(:), y(:), z(:), u(:), v(:), w(:)
    REAL(4), INTENT(IN)    :: f(GX, GY, GZ, 3)
    REAL(4) :: p0(3), v0(3), k1(3), k2(3), k3(3), k4(3), p_tmp(3)
    
    p0 = [x(idx), y(idx), z(idx)]
    v0 = [u(idx), v(idx), w(idx)]
    
    k1 = interp_3d(p0, f)
    p_tmp = p0 + 0.5 * DT * v0 
    k2 = interp_3d(p_tmp, f)
    p_tmp = p0 + 0.5 * DT * v0
    k3 = interp_3d(p_tmp, f)
    p_tmp = p0 + DT * v0
    k4 = interp_3d(p_tmp, f)
    
    u(idx) = v0(1) + (DT / 6.0) * (k1(1) + 2.0*k2(1) + 2.0*k3(1) + k4(1))
    v(idx) = v0(2) + (DT / 6.0) * (k1(2) + 2.0*k2(2) + 2.0*k3(2) + k4(2))
    w(idx) = v0(3) + (DT / 6.0) * (k1(3) + 2.0*k2(3) + 2.0*k3(3) + k4(3))
    
    x(idx) = p0(1) + DT * u(idx); y(idx) = p0(2) + DT * v(idx); z(idx) = p0(3) + DT * w(idx)
  END SUBROUTINE rk4_kernel_step

END PROGRAM profile_rk4_interp