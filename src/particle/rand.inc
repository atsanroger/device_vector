! ============================================================================
! Random Number Generators (XORShift32 + Box-Muller)
! File: rand.inc
! ============================================================================

!$acc routine seq
FUNCTION urand(seed) RESULT(val)
    INTEGER, INTENT(INOUT) :: seed
    REAL(4) :: val
    
    ! 定義縮放常數 1/2^32
    REAL(4), PARAMETER :: S_SCALE = 2.3283064E-10_4
    
    ! 1. 安全檢查：XORShift 不能為 0
    IF (seed == 0) seed = 123456789

    ! 2. XORShift32 演算法
    seed = IEOR(seed, ISHFT(seed, 13))
    seed = IEOR(seed, ISHFT(seed, -17))
    seed = IEOR(seed, ISHFT(seed, 5))

    ! 3. 轉換為 0.0 ~ 1.0 的浮點數
    !    Fortran 整數是帶號的 (-2^31 ~ 2^31-1)
    !    REAL(seed) * S_SCALE 會得到 -0.5 ~ 0.5
    !    加上 0.5 後範圍變成 0.0 ~ 1.0
    val = 0.5_4 + REAL(seed, 4) * S_SCALE
END FUNCTION urand

!$acc routine seq
FUNCTION nrand(seed) RESULT(z)
    INTEGER, INTENT(INOUT) :: seed
    REAL(4) :: z
    REAL(4) :: u1, u2, r, theta
    
    ! 必須定義 PI 與極小值
    REAL(4), PARAMETER :: PI_4   = 3.14159265359_4
    REAL(4), PARAMETER :: TINY_4 = 1.0E-30_4

    ! 1. 取得兩個均勻分佈隨機數
    u1 = urand(seed)
    u2 = urand(seed)

    ! 2. 防止 LOG(0) 導致 NaN/Infinity
    IF (u1 < TINY_4) u1 = TINY_4

    ! 3. Box-Muller 轉換 (極座標法)
    r     = SQRT(-2.0_4 * LOG(u1))
    theta = 2.0_4 * PI_4 * u2
    
    ! 回傳常態分佈數值 (平均值 0, 標準差 1)
    z = r * COS(theta)
    
    ! 註：Box-Muller 其實還產生了另一個數 r * SIN(theta)，
    ! 但為了節省 GPU 上的狀態儲存空間，我們直接捨棄它。
END FUNCTION nrand