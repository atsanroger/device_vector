! ============================================================================
! Random Number Generators (XORShift32 + Box-Muller)
! File: rand.inc
! ============================================================================

!$acc routine seq
FUNCTION urand(seed) RESULT(val)
    INTEGER, INTENT(INOUT) :: seed
    REAL(4) :: val
    
    REAL(4), PARAMETER :: S_SCALE = 2.3283064E-10_4
    
    IF (seed == 0) seed = 123456789

    seed = IEOR(seed, ISHFT(seed, 13))
    seed = IEOR(seed, ISHFT(seed, -17))
    seed = IEOR(seed, ISHFT(seed, 5))

    val = 0.5_4 + REAL(seed, 4) * S_SCALE
END FUNCTION urand

!$acc routine seq
FUNCTION nrand(seed) RESULT(z)
    INTEGER, INTENT(INOUT) :: seed
    REAL(4) :: z
    REAL(4) :: u1, u2, r, theta
    
    REAL(4), PARAMETER :: PI_4   = 3.14159265359_4
    REAL(4), PARAMETER :: TINY_4 = 1.0E-30_4

    u1 = urand(seed)
    u2 = urand(seed)

    IF (u1 < TINY_4) u1 = TINY_4

    r     = SQRT(-2.0_4 * LOG(u1))
    theta = 2.0_4 * PI_4 * u2
    
    z = r * COS(theta)
    
END FUNCTION nrand