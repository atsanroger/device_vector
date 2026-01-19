MODULE Random_Mod
    USE openacc         
    IMPLICIT NONE

    PRIVATE             
    PUBLIC :: urand, nrand

    !$acc routine(urand) seq
    !$acc routine(nrand) seq

    REAL(4), PARAMETER :: S_SCALE = 2.3283064E-10
    REAL(4), PARAMETER :: PI_VAL  = 3.14159265359
    REAL(4), PARAMETER :: TINY_VAL= 1.0E-30

CONTAINS

    ! ----------------------------------------------------------------
    ! Uniform Random Generator (XORShift32)
    ! Range: [0, 1]
    ! ----------------------------------------------------------------
    FUNCTION urand(seed) RESULT(val)
        INTEGER, INTENT(INOUT) :: seed
        REAL(4) :: val
        
        IF (seed == 0) seed = 123456789

        ! XORShift32 Algorithm
        seed = IEOR(seed, ISHFT(seed, 13))
        seed = IEOR(seed, ISHFT(seed, -17))
        seed = IEOR(seed, ISHFT(seed, 5))

        val = 0.5 + REAL(seed, 4) * S_SCALE
    END FUNCTION urand

    ! ----------------------------------------------------------------
    ! Normal Distribution Generator (Box-Muller Transform)
    ! Mean: 0, StdDev: 1
    ! ----------------------------------------------------------------
    FUNCTION nrand(seed) RESULT(z)
        INTEGER, INTENT(INOUT) :: seed
        REAL(4) :: z
        REAL(4) :: u1, u2, r, theta
        
        u1 = urand(seed)
        u2 = urand(seed)

        IF (u1 < TINY_VAL) u1 = TINY_VAL

        r     = SQRT(-2.0 * LOG(u1))
        theta = 2.0 * PI_VAL * u2
        
        z = r * COS(theta)
        
    END FUNCTION nrand

END MODULE Random_Mod