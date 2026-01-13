MODULE Device_OpenACC_Interface
  USE iso_c_binding
  IMPLICIT NONE

  INTERFACE
    ! =========================================================
    ! INTEGER(4) - Int 4
    ! =========================================================
    SUBROUTINE vec_acc_map_i4(h) BIND(C, NAME='vec_acc_map_i4')
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_acc_unmap_i4(h) BIND(C, NAME='vec_acc_unmap_i4')
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: h
    END SUBROUTINE
    
    FUNCTION vec_acc_is_mapped_i4(h) BIND(C, NAME='vec_acc_is_mapped_i4')
      IMPORT :: C_PTR, C_INT
      TYPE(C_PTR), VALUE :: h
      INTEGER(C_INT) :: vec_acc_is_mapped_i4
    END FUNCTION

    ! =========================================================
    ! INTEGER(8) - Int 8
    ! =========================================================
    SUBROUTINE vec_acc_map_i8(h) BIND(C, NAME='vec_acc_map_i8')
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_acc_unmap_i8(h) BIND(C, NAME='vec_acc_unmap_i8')
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: h
    END SUBROUTINE
    
    FUNCTION vec_acc_is_mapped_i8(h) BIND(C, NAME='vec_acc_is_mapped_i8')
      IMPORT :: C_PTR, C_INT
      TYPE(C_PTR), VALUE :: h
      INTEGER(C_INT) :: vec_acc_is_mapped_i8
    END FUNCTION

    ! =========================================================
    ! REAL(4) - Real 4
    ! =========================================================
    SUBROUTINE vec_acc_map_r4(h) BIND(C, NAME='vec_acc_map_r4')
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_acc_unmap_r4(h) BIND(C, NAME='vec_acc_unmap_r4')
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: h
    END SUBROUTINE

    FUNCTION vec_acc_is_mapped_r4(h) BIND(C, NAME='vec_acc_is_mapped_r4')
      IMPORT :: C_PTR, C_INT
      TYPE(C_PTR), VALUE :: h
      INTEGER(C_INT) :: vec_acc_is_mapped_r4
    END FUNCTION

    ! =========================================================
    ! REAL(8) - Real 8
    ! =========================================================
    SUBROUTINE vec_acc_map_r8(h) BIND(C, NAME='vec_acc_map_r8')
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_acc_unmap_r8(h) BIND(C, NAME='vec_acc_unmap_r8')
      IMPORT :: C_PTR
      TYPE(C_PTR), VALUE :: h
    END SUBROUTINE

    FUNCTION vec_acc_is_mapped_r8(h) BIND(C, NAME='vec_acc_is_mapped_r8')
      IMPORT :: C_PTR, C_INT
      TYPE(C_PTR), VALUE :: h
      INTEGER(C_INT) :: vec_acc_is_mapped_r8
    END FUNCTION

  END INTERFACE

END MODULE Device_OpenACC_Interface