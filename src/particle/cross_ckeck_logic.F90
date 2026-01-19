! 放在 Boundary_Logic_Mod.f90
MODULE Boundary_Logic_Mod
    USE openacc         

    IMPLICIT NONE

    PRIVATE             
    
    PUBLIC :: cross_x_branchless, cross_y_branchless, cross_z_branchless

    !$acc routine(cross_x_branchless) seq
    !$acc routine(cross_y_branchless) seq
    !$acc routine(cross_z_branchless) seq

CONTAINS

    ! ----------------------------------------------------------------
    ! X Direction
    ! ----------------------------------------------------------------
    SUBROUTINE cross_x_branchless(ipos, xpos, m_packed, dx_val, idx0_next, idx0_prev, reflc, repul)

        INTEGER,    INTENT(INOUT) :: ipos
        REAL(4),   INTENT(INOUT) :: xpos
        INTEGER(1), INTENT(IN)    :: m_packed
        REAL(4),   INTENT(IN)    :: dx_val, idx0_next, idx0_prev, reflc, repul
        
        REAL(4) :: x_next, x_refl
        INTEGER  :: side_bit
        LOGICAL  :: is_over, is_under, is_water

        ! Early Exit Check
        is_over  = (xpos >= 1.0_4)
        is_under = (xpos < 0.0_4)

        IF (is_over .OR. is_under) THEN
            side_bit = MERGE(0, 1, is_over)
            is_water = BTEST(m_packed, side_bit)

            x_next = MERGE((xpos - 1.0_4) * dx_val * idx0_next, &
                           1.0_4 + xpos * dx_val * idx0_prev, &
                           is_over)
            
            x_refl = MERGE(1.0_4 - (xpos - 1.0_4) * reflc, &
                           -xpos * reflc, &
                           is_over)

            xpos = MERGE(x_next, x_refl, is_water)
            ipos = ipos + MERGE(MERGE(1, -1, is_over), 0, is_water)
        END IF

        ! Repulsion (Bit 0: +X, Bit 1: -X)
        xpos = MERGE(min(xpos, 1.0_4 - repul), xpos, .NOT. BTEST(m_packed, 0))
        xpos = MERGE(max(xpos, repul),         xpos, .NOT. BTEST(m_packed, 1))
    END SUBROUTINE cross_x_branchless

    ! ----------------------------------------------------------------
    ! Y Direction
    ! ----------------------------------------------------------------
    SUBROUTINE cross_y_branchless(jpos, ypos, m_packed, dy_val, idy0_next, idy0_prev, reflc, repul)

        INTEGER,    INTENT(INOUT) :: jpos
        REAL(4),   INTENT(INOUT) :: ypos
        INTEGER(1), INTENT(IN)    :: m_packed
        REAL(4),   INTENT(IN)    :: dy_val, idy0_next, idy0_prev, reflc, repul
        
        REAL(4) :: y_next, y_refl
        INTEGER  :: side_bit
        LOGICAL  :: is_over, is_under, is_water

        is_over  = (ypos >= 1.0_4)
        is_under = (ypos < 0.0_4)

        IF (is_over .OR. is_under) THEN
            ! Bit 2: +Y, Bit 3: -Y
            side_bit = MERGE(2, 3, is_over)
            is_water = BTEST(m_packed, side_bit)

            y_next = MERGE((ypos - 1.0_4) * dy_val * idy0_next, &
                           1.0_4 + ypos * dy_val * idy0_prev, &
                           is_over)
            
            y_refl = MERGE(1.0_4 - (ypos - 1.0_4) * reflc, &
                           -ypos * reflc, &
                           is_over)

            ypos = MERGE(y_next, y_refl, is_water)
            jpos = jpos + MERGE(MERGE(1, -1, is_over), 0, is_water)
        END IF

        ! Repulsion (Bit 2: +Y, Bit 3: -Y)
        ypos = MERGE(min(ypos, 1.0_4 - repul), ypos, .NOT. BTEST(m_packed, 2))
        ypos = MERGE(max(ypos, repul),         ypos, .NOT. BTEST(m_packed, 3))
    END SUBROUTINE cross_y_branchless

    ! ----------------------------------------------------------------
    ! Z Direction
    ! ----------------------------------------------------------------
    SUBROUTINE cross_z_branchless(kpos, zpos, m_packed, dz_val, idz0_val, &
                                  idz0_next, idz0_prev, dz_ref_val, &
                                  reflc_sfc, reflc_btm, repul_sfc, repul_btm)

        INTEGER,    INTENT(INOUT) :: kpos
        REAL(4),   INTENT(INOUT) :: zpos
        INTEGER(1), INTENT(IN)    :: m_packed
        REAL(4),   INTENT(IN)    :: dz_val, idz0_val, idz0_next, idz0_prev, dz_ref_val
        REAL(4),   INTENT(IN)    :: reflc_sfc, reflc_btm, repul_sfc, repul_btm
        
        REAL(4) :: z_next, z_refl, r_coef, solid_bottom
        INTEGER  :: side_bit
        LOGICAL  :: is_over, is_under, is_water

        is_over  = (zpos >= 1.0_4)
        is_under = (zpos < 0.0_4)

        IF (is_over .OR. is_under) THEN
            ! Bit 4: +Z (Surface), Bit 5: -Z (Bottom)
            side_bit = MERGE(4, 5, is_over)
            is_water = BTEST(m_packed, side_bit)

            r_coef = MERGE(reflc_sfc, reflc_btm, is_over)

            z_next = MERGE((zpos - 1.0_4) * dz_val * idz0_next, &
                           1.0_4 + zpos * dz_val * idz0_prev, &
                           is_over)
            
            z_refl = MERGE(1.0_4 - (zpos - 1.0_4) * r_coef, &
                           -zpos * r_coef, &
                           is_over)

            zpos = MERGE(z_next, z_refl, is_water)
            kpos = kpos + MERGE(MERGE(1, -1, is_over), 0, is_water)
        END IF

        ! Partial Step Repulsion
        ! Surface (Bit 4): Standard repulsion
        zpos = MERGE(min(zpos, 1.0_4 - repul_sfc), zpos, .NOT. BTEST(m_packed, 4))
        
        ! Bottom (Bit 5): Partial step repulsion
        solid_bottom = (dz_val - dz_ref_val) * idz0_val
        zpos = MERGE(max(zpos, solid_bottom + repul_btm), zpos, .NOT. BTEST(m_packed, 5))

        ! Safety Clamp
        zpos = min(max(zpos, 0.0_4), 1.0_4 - 1.0E-5_4)
    END SUBROUTINE cross_z_branchless

END MODULE Boundary_Logic_Mod