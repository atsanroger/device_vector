MODULE Device_Vector
  USE iso_c_binding
  IMPLICIT NONE

  ! ====================================================================
  ! 1. Whitelist Mode
  ! ====================================================================
  PRIVATE 

  ! ====================================================================
  ! 2. Public Interface
  ! ====================================================================
  
  ! (1) Smart Classes
  PUBLIC :: device_vector_i4_t
  PUBLIC :: device_vector_i8_t
  PUBLIC :: device_vector_r4_t
  PUBLIC :: device_vector_r8_t

  ! (2) Environment
  PUBLIC :: device_env_init
  PUBLIC :: device_env_finalize
  PUBLIC :: device_synchronize

  ! (3) Algorithms & Reductions
  PUBLIC :: vec_sort_i4
  
  PUBLIC :: vec_sum_i4, vec_min_i4, vec_max_i4
  PUBLIC :: vec_sum_i8, vec_min_i8, vec_max_i8
  PUBLIC :: vec_sum_r4, vec_min_r4, vec_max_r4
  PUBLIC :: vec_sum_r8, vec_min_r8, vec_max_r8

  ! ====================================================================
  ! 3. C Function Interfaces
  ! ====================================================================
  INTERFACE

    ! --- Environment ---
    SUBROUTINE device_env_init(rank, gpus_per_node) BIND(C, name="device_env_init")
      IMPORT
      INTEGER(c_int), VALUE :: rank, gpus_per_node
    END SUBROUTINE

    SUBROUTINE device_env_finalize() BIND(C, name="device_env_finalize")
    END SUBROUTINE

    SUBROUTINE device_synchronize() BIND(C, name="device_synchronize")
    END SUBROUTINE

    ! ------------------------------------------------------------------
    ! [i4] Integer (32-bit)
    ! ------------------------------------------------------------------
    FUNCTION vec_create_i4_c(n, mode) BIND(C, name="vec_create_i4")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int), VALUE    :: mode
      TYPE(c_ptr)              :: vec_create_i4_c
    END FUNCTION

    SUBROUTINE vec_delete_i4_c(h) BIND(C, name="vec_delete_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_resize_i4_c(h, n) BIND(C, name="vec_resize_i4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE

    SUBROUTINE vec_reserve_i4_c(h, n) BIND(C, name="vec_reserve_i4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE

    FUNCTION vec_host_i4_c(h) BIND(C, name="vec_host_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_host_i4_c
    END FUNCTION

    FUNCTION vec_dev_i4_c(h) BIND(C, name="vec_dev_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_dev_i4_c
    END FUNCTION
    
    FUNCTION vec_size_i4_c(h) BIND(C, name="vec_size_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: vec_size_i4_c
    END FUNCTION

    SUBROUTINE vec_upload_i4_c(h) BIND(C, name="vec_upload_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_download_i4_c(h) BIND(C, name="vec_download_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_upload_part_i4_c(h, offset, count) BIND(C, name="vec_upload_part_i4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: offset, count
    END SUBROUTINE

    SUBROUTINE vec_download_part_i4_c(h, offset, count) BIND(C, name="vec_download_part_i4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: offset, count
    END SUBROUTINE

    SUBROUTINE vec_fill_zero_i4_c(h) BIND(C, name="vec_fill_zero_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_set_value_i4_c(h, val) BIND(C, name="vec_set_value_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_int), VALUE :: val
    END SUBROUTINE

    SUBROUTINE vec_gather_i4_c(src, map, dst) BIND(C, name="vec_gather_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: src, map, dst
    END SUBROUTINE

    FUNCTION vec_clone_i4_c(h) BIND(C, name="vec_clone_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_clone_i4_c
    END FUNCTION

    ! --- Reductions (Full) ---
    FUNCTION vec_sum_i4_c(h) BIND(C, name="vec_sum_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_int)     :: vec_sum_i4_c
    END FUNCTION
    FUNCTION vec_min_i4_c(h) BIND(C, name="vec_min_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_int)     :: vec_min_i4_c
    END FUNCTION
    FUNCTION vec_max_i4_c(h) BIND(C, name="vec_max_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_int)     :: vec_max_i4_c
    END FUNCTION

    ! --- Reductions (Partial) ---
    FUNCTION vec_sum_partial_i4_c(h, n) BIND(C, name="vec_sum_partial_i4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int)           :: vec_sum_partial_i4_c
    END FUNCTION
    FUNCTION vec_min_partial_i4_c(h, n) BIND(C, name="vec_min_partial_i4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int)           :: vec_min_partial_i4_c
    END FUNCTION
    FUNCTION vec_max_partial_i4_c(h, n) BIND(C, name="vec_max_partial_i4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int)           :: vec_max_partial_i4_c
    END FUNCTION


    ! ------------------------------------------------------------------
    ! [i8] Integer (64-bit)
    ! ------------------------------------------------------------------
    FUNCTION vec_create_i8_c(n, mode) BIND(C, name="vec_create_i8")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int), VALUE    :: mode
      TYPE(c_ptr)              :: vec_create_i8_c
    END FUNCTION

    SUBROUTINE vec_delete_i8_c(h) BIND(C, name="vec_delete_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    
    SUBROUTINE vec_resize_i8_c(h, n) BIND(C, name="vec_resize_i8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE

    FUNCTION vec_host_i8_c(h) BIND(C, name="vec_host_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_host_i8_c
    END FUNCTION

    FUNCTION vec_dev_i8_c(h) BIND(C, name="vec_dev_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_dev_i8_c
    END FUNCTION
    
    FUNCTION vec_size_i8_c(h) BIND(C, name="vec_size_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: vec_size_i8_c
    END FUNCTION

    SUBROUTINE vec_upload_i8_c(h) BIND(C, name="vec_upload_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_download_i8_c(h) BIND(C, name="vec_download_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_fill_zero_i8_c(h) BIND(C, name="vec_fill_zero_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_set_value_i8_c(h, val) BIND(C, name="vec_set_value_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_long_long), VALUE :: val
    END SUBROUTINE

    SUBROUTINE vec_gather_i8_c(src, map, dst) BIND(C, name="vec_gather_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: src, map, dst
    END SUBROUTINE

    FUNCTION vec_clone_i8_c(h) BIND(C, name="vec_clone_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_clone_i8_c
    END FUNCTION

    ! --- Reductions (Full) ---
    FUNCTION vec_sum_i8_c(h) BIND(C, name="vec_sum_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_long_long) :: vec_sum_i8_c
    END FUNCTION
    FUNCTION vec_min_i8_c(h) BIND(C, name="vec_min_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_long_long) :: vec_min_i8_c
    END FUNCTION
    FUNCTION vec_max_i8_c(h) BIND(C, name="vec_max_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_long_long) :: vec_max_i8_c
    END FUNCTION

    ! --- Reductions (Partial) ---
    FUNCTION vec_sum_partial_i8_c(h, n) BIND(C, name="vec_sum_partial_i8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_long_long)     :: vec_sum_partial_i8_c
    END FUNCTION
    FUNCTION vec_min_partial_i8_c(h, n) BIND(C, name="vec_min_partial_i8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_long_long)     :: vec_min_partial_i8_c
    END FUNCTION
    FUNCTION vec_max_partial_i8_c(h, n) BIND(C, name="vec_max_partial_i8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_long_long)     :: vec_max_partial_i8_c
    END FUNCTION


    ! ------------------------------------------------------------------
    ! [r4] Real (32-bit Float)
    ! ------------------------------------------------------------------
    FUNCTION vec_create_r4_c(n, mode) BIND(C, name="vec_create_r4")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int), VALUE    :: mode
      TYPE(c_ptr)              :: vec_create_r4_c
    END FUNCTION

    SUBROUTINE vec_delete_r4_c(h) BIND(C, name="vec_delete_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_resize_r4_c(h, n) BIND(C, name="vec_resize_r4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE

    FUNCTION vec_host_r4_c(h) BIND(C, name="vec_host_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_host_r4_c
    END FUNCTION

    FUNCTION vec_dev_r4_c(h) BIND(C, name="vec_dev_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_dev_r4_c
    END FUNCTION
    
    FUNCTION vec_size_r4_c(h) BIND(C, name="vec_size_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: vec_size_r4_c
    END FUNCTION

    SUBROUTINE vec_upload_r4_c(h) BIND(C, name="vec_upload_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_download_r4_c(h) BIND(C, name="vec_download_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_fill_zero_r4_c(h) BIND(C, name="vec_fill_zero_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_set_value_r4_c(h, val) BIND(C, name="vec_set_value_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_float), VALUE :: val
    END SUBROUTINE

    SUBROUTINE vec_gather_r4_c(src, map, dst) BIND(C, name="vec_gather_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: src, map, dst
    END SUBROUTINE

    FUNCTION vec_clone_r4_c(h) BIND(C, name="vec_clone_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_clone_r4_c
    END FUNCTION

    ! --- Reductions (Full) ---
    FUNCTION vec_sum_r4_c(h) BIND(C, name="vec_sum_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_float)      :: vec_sum_r4_c
    END FUNCTION
    FUNCTION vec_min_r4_c(h) BIND(C, name="vec_min_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_float)      :: vec_min_r4_c
    END FUNCTION
    FUNCTION vec_max_r4_c(h) BIND(C, name="vec_max_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_float)      :: vec_max_r4_c
    END FUNCTION

    ! --- Reductions (Partial) ---
    FUNCTION vec_sum_partial_r4_c(h, n) BIND(C, name="vec_sum_partial_r4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_float)            :: vec_sum_partial_r4_c
    END FUNCTION
    FUNCTION vec_min_partial_r4_c(h, n) BIND(C, name="vec_min_partial_r4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_float)            :: vec_min_partial_r4_c
    END FUNCTION
    FUNCTION vec_max_partial_r4_c(h, n) BIND(C, name="vec_max_partial_r4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_float)            :: vec_max_partial_r4_c
    END FUNCTION


    ! ------------------------------------------------------------------
    ! [r8] Real (64-bit Double)
    ! ------------------------------------------------------------------
    FUNCTION vec_create_r8_c(n, mode) BIND(C, name="vec_create_r8")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int), VALUE    :: mode
      TYPE(c_ptr)              :: vec_create_r8_c
    END FUNCTION

    SUBROUTINE vec_delete_r8_c(h) BIND(C, name="vec_delete_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    
    SUBROUTINE vec_resize_r8_c(h, n) BIND(C, name="vec_resize_r8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE

    FUNCTION vec_host_r8_c(h) BIND(C, name="vec_host_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_host_r8_c
    END FUNCTION

    FUNCTION vec_dev_r8_c(h) BIND(C, name="vec_dev_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_dev_r8_c
    END FUNCTION
    
    FUNCTION vec_size_r8_c(h) BIND(C, name="vec_size_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: vec_size_r8_c
    END FUNCTION

    SUBROUTINE vec_upload_r8_c(h) BIND(C, name="vec_upload_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_download_r8_c(h) BIND(C, name="vec_download_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_fill_zero_r8_c(h) BIND(C, name="vec_fill_zero_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_set_value_r8_c(h, val) BIND(C, name="vec_set_value_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_double), VALUE :: val
    END SUBROUTINE

    SUBROUTINE vec_gather_r8_c(src, map, dst) BIND(C, name="vec_gather_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: src, map, dst
    END SUBROUTINE

    FUNCTION vec_clone_r8_c(h) BIND(C, name="vec_clone_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: vec_clone_r8_c
    END FUNCTION

    ! --- Reductions (Full) ---
    FUNCTION vec_sum_r8_c(h) BIND(C, name="vec_sum_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_double)     :: vec_sum_r8_c
    END FUNCTION
    FUNCTION vec_min_r8_c(h) BIND(C, name="vec_min_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_double)     :: vec_min_r8_c
    END FUNCTION
    FUNCTION vec_max_r8_c(h) BIND(C, name="vec_max_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_double)     :: vec_max_r8_c
    END FUNCTION

    ! --- Reductions (Partial) ---
    FUNCTION vec_sum_partial_r8_c(h, n) BIND(C, name="vec_sum_partial_r8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_double)           :: vec_sum_partial_r8_c
    END FUNCTION
    FUNCTION vec_min_partial_r8_c(h, n) BIND(C, name="vec_min_partial_r8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_double)           :: vec_min_partial_r8_c
    END FUNCTION
    FUNCTION vec_max_partial_r8_c(h, n) BIND(C, name="vec_max_partial_r8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_double)           :: vec_max_partial_r8_c
    END FUNCTION


    ! ==================================================================
    ! Sort Interface (Integer Only for Keys)
    ! ==================================================================
    SUBROUTINE vec_sort_pairs_i4_c(keys_in, keys_buf, vals_in, vals_buf, n) &
          BIND(C, NAME='vec_sort_pairs_i4_c')
      IMPORT
      TYPE(C_PTR), VALUE :: keys_in, keys_buf, vals_in, vals_buf
      INTEGER(C_SIZE_T), VALUE :: n
    END SUBROUTINE

  END INTERFACE

  ! ====================================================================
  ! Smart Wrapper Classes (device_vector_xx_t)
  ! ====================================================================

  ! --- device_vector_i4_t ---
  TYPE :: device_vector_i4_t
    TYPE(c_ptr), PRIVATE :: handle = C_NULL_PTR
    INTEGER(4), POINTER, PUBLIC :: data(:) => NULL()
  CONTAINS
    PROCEDURE :: create   => impl_create_i4
    PROCEDURE :: resize   => impl_resize_i4
    PROCEDURE :: free     => impl_free_i4
    PROCEDURE :: upload   => impl_upload_i4
    PROCEDURE :: download => impl_download_i4
    PROCEDURE :: fill_zero => impl_fill_zero_i4
    PROCEDURE :: get_handle => impl_get_handle_i4
    PROCEDURE :: get_device_ptr => impl_get_device_ptr_i4
  END TYPE device_vector_i4_t

  ! --- device_vector_i8_t ---
  TYPE :: device_vector_i8_t
    TYPE(c_ptr), PRIVATE :: handle = C_NULL_PTR
    INTEGER(8), POINTER, PUBLIC :: data(:) => NULL()
  CONTAINS
    PROCEDURE :: create   => impl_create_i8
    PROCEDURE :: resize   => impl_resize_i8
    PROCEDURE :: free     => impl_free_i8
    PROCEDURE :: upload   => impl_upload_i8
    PROCEDURE :: download => impl_download_i8
    PROCEDURE :: fill_zero => impl_fill_zero_i8
    PROCEDURE :: get_handle => impl_get_handle_i8
    PROCEDURE :: get_device_ptr => impl_get_device_ptr_i8
  END TYPE device_vector_i8_t

  ! --- device_vector_r4_t ---
  TYPE :: device_vector_r4_t
    TYPE(c_ptr), PRIVATE :: handle = C_NULL_PTR
    REAL(4), POINTER, PUBLIC :: data(:) => NULL()
  CONTAINS
    PROCEDURE :: create   => impl_create_r4
    PROCEDURE :: resize   => impl_resize_r4
    PROCEDURE :: free     => impl_free_r4
    PROCEDURE :: upload   => impl_upload_r4
    PROCEDURE :: download => impl_download_r4
    PROCEDURE :: fill_zero => impl_fill_zero_r4
    PROCEDURE :: get_handle => impl_get_handle_r4
    PROCEDURE :: get_device_ptr => impl_get_device_ptr_r4
  END TYPE device_vector_r4_t

  ! --- device_vector_r8_t ---
  TYPE :: device_vector_r8_t
    TYPE(c_ptr), PRIVATE :: handle = C_NULL_PTR
    REAL(8), POINTER, PUBLIC :: data(:) => NULL()
  CONTAINS
    PROCEDURE :: create   => impl_create_r8
    PROCEDURE :: resize   => impl_resize_r8
    PROCEDURE :: free     => impl_free_r8
    PROCEDURE :: upload   => impl_upload_r8
    PROCEDURE :: download => impl_download_r8
    PROCEDURE :: fill_zero => impl_fill_zero_r8
    PROCEDURE :: get_handle => impl_get_handle_r8
    PROCEDURE :: get_device_ptr => impl_get_device_ptr_r8
  END TYPE device_vector_r8_t


CONTAINS

  ! ====================================================================
  ! Class Implementations 
  ! ====================================================================

  ! --------------------------------------------------------------------
  ! Impl: i4
  ! --------------------------------------------------------------------
  SUBROUTINE impl_create_i4(this, n, mode)
    CLASS(device_vector_i4_t), INTENT(OUT) :: this
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN), OPTIONAL :: mode
    INTEGER :: m
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    m = 0; IF (PRESENT(mode)) m = mode
    this%handle = vec_create_i4_c(INT(n, c_size_t), INT(m, c_int))
    
    ! Sync Pointer
    raw_c_ptr = vec_host_i4_c(this%handle)
    sz        = vec_size_i4_c(this%handle)

    IF (sz > 0) THEN
      CALL C_F_POINTER(raw_c_ptr, this%data, [sz])
    ELSE
      NULLIFY(this%data)
    END IF

  END SUBROUTINE impl_create_i4

  SUBROUTINE impl_resize_i4(this, n)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    ! 1. Call C++ Resize
    CALL vec_resize_i4_c(this%handle, INT(n, c_size_t))

    ! 2. Auto-Sync Pointer
    raw_c_ptr = vec_host_i4_c(this%handle)
    sz        = vec_size_i4_c(this%handle)
    
    IF (sz > 0) THEN
      CALL C_F_POINTER(raw_c_ptr, this%data, [sz])
    ELSE
      NULLIFY(this%data)
    END IF
  END SUBROUTINE impl_resize_i4

  SUBROUTINE impl_free_i4(this)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    IF (C_ASSOCIATED(this%handle)) THEN
       CALL vec_delete_i4_c(this%handle)
       this%handle = C_NULL_PTR
    END IF
    NULLIFY(this%data)
  END SUBROUTINE impl_free_i4

  SUBROUTINE impl_upload_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this
    CALL vec_upload_i4_c(this%handle)
  END SUBROUTINE impl_upload_i4

  SUBROUTINE impl_download_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this
    CALL vec_download_i4_c(this%handle)
  END SUBROUTINE impl_download_i4

  SUBROUTINE impl_fill_zero_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this
    CALL vec_fill_zero_i4_c(this%handle)
  END SUBROUTINE impl_fill_zero_i4


  ! --------------------------------------------------------------------
  ! Impl: r8 (Double)
  ! --------------------------------------------------------------------
  SUBROUTINE impl_create_r8(this, n, mode)
    CLASS(device_vector_r8_t), INTENT(OUT) :: this
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN), OPTIONAL :: mode
    INTEGER :: m
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    m = 0; IF (PRESENT(mode)) m = mode
    this%handle = vec_create_r8_c(INT(n, c_size_t), INT(m, c_int))
    
    ! Sync Pointer
    raw_c_ptr = vec_host_r8_c(this%handle)
    sz = vec_size_r8_c(this%handle)
    IF (sz > 0) THEN
      CALL C_F_POINTER(raw_c_ptr, this%data, [sz])
    ELSE
      NULLIFY(this%data)
    END IF
  END SUBROUTINE impl_create_r8

  SUBROUTINE impl_resize_r8(this, n)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    CALL vec_resize_r8_c(this%handle, INT(n, c_size_t))

    ! Auto-Sync Pointer
    raw_c_ptr = vec_host_r8_c(this%handle)
    sz = vec_size_r8_c(this%handle)
    IF (sz > 0) THEN
      CALL C_F_POINTER(raw_c_ptr, this%data, [sz])
    ELSE
      NULLIFY(this%data)
    END IF
  END SUBROUTINE impl_resize_r8

  SUBROUTINE impl_free_r8(this)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    IF (C_ASSOCIATED(this%handle)) THEN
       CALL vec_delete_r8_c(this%handle)
       this%handle = C_NULL_PTR
    END IF
    NULLIFY(this%data)
  END SUBROUTINE impl_free_r8

  SUBROUTINE impl_upload_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this
    CALL vec_upload_r8_c(this%handle)
  END SUBROUTINE impl_upload_r8

  SUBROUTINE impl_download_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this
    CALL vec_download_r8_c(this%handle)
  END SUBROUTINE impl_download_r8

  SUBROUTINE impl_fill_zero_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this
    CALL vec_fill_zero_r8_c(this%handle)
  END SUBROUTINE impl_fill_zero_r8


  ! --------------------------------------------------------------------
  ! Impl: r4 (Float)
  ! --------------------------------------------------------------------
  SUBROUTINE impl_create_r4(this, n, mode)
    CLASS(device_vector_r4_t), INTENT(OUT) :: this
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN), OPTIONAL :: mode
    INTEGER :: m
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    m = 0; IF (PRESENT(mode)) m = mode
    this%handle = vec_create_r4_c(INT(n, c_size_t), INT(m, c_int))
    raw_c_ptr = vec_host_r4_c(this%handle)
    sz = vec_size_r4_c(this%handle)
    IF (sz > 0) THEN
      CALL C_F_POINTER(raw_c_ptr, this%data, [sz])
    ELSE
      NULLIFY(this%data)
    END IF
  END SUBROUTINE impl_create_r4

  SUBROUTINE impl_resize_r4(this, n)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    CALL vec_resize_r4_c(this%handle, INT(n, c_size_t))
    raw_c_ptr = vec_host_r4_c(this%handle)
    sz = vec_size_r4_c(this%handle)
    IF (sz > 0) THEN
      CALL C_F_POINTER(raw_c_ptr, this%data, [sz])
    ELSE
      NULLIFY(this%data)
    END IF
  END SUBROUTINE impl_resize_r4

  SUBROUTINE impl_free_r4(this)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    IF (C_ASSOCIATED(this%handle)) THEN
       CALL vec_delete_r4_c(this%handle)
       this%handle = C_NULL_PTR
    END IF
    NULLIFY(this%data)
  END SUBROUTINE impl_free_r4

  SUBROUTINE impl_upload_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this
    CALL vec_upload_r4_c(this%handle)
  END SUBROUTINE impl_upload_r4

  SUBROUTINE impl_download_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this
    CALL vec_download_r4_c(this%handle)
  END SUBROUTINE impl_download_r4

  SUBROUTINE impl_fill_zero_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this
    CALL vec_fill_zero_r4_c(this%handle)
  END SUBROUTINE impl_fill_zero_r4


  ! --------------------------------------------------------------------
  ! Impl: i8 (Long Long)
  ! --------------------------------------------------------------------
  SUBROUTINE impl_create_i8(this, n, mode)
    CLASS(device_vector_i8_t), INTENT(OUT) :: this
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN), OPTIONAL :: mode
    INTEGER :: m
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    m = 0; IF (PRESENT(mode)) m = mode
    this%handle = vec_create_i8_c(INT(n, c_size_t), INT(m, c_int))
    raw_c_ptr = vec_host_i8_c(this%handle)
    sz = vec_size_i8_c(this%handle)
    IF (sz > 0) THEN
      CALL C_F_POINTER(raw_c_ptr, this%data, [sz])
    ELSE
      NULLIFY(this%data)
    END IF
  END SUBROUTINE impl_create_i8

  SUBROUTINE impl_resize_i8(this, n)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    CALL vec_resize_i8_c(this%handle, INT(n, c_size_t))
    raw_c_ptr = vec_host_i8_c(this%handle)
    sz = vec_size_i8_c(this%handle)
    IF (sz > 0) THEN
      CALL C_F_POINTER(raw_c_ptr, this%data, [sz])
    ELSE
      NULLIFY(this%data)
    END IF
  END SUBROUTINE impl_resize_i8

  SUBROUTINE impl_free_i8(this)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    IF (C_ASSOCIATED(this%handle)) THEN
       CALL vec_delete_i8_c(this%handle)
       this%handle = C_NULL_PTR
    END IF
    NULLIFY(this%data)
  END SUBROUTINE impl_free_i8

  SUBROUTINE impl_upload_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this
    CALL vec_upload_i8_c(this%handle)
  END SUBROUTINE impl_upload_i8

  SUBROUTINE impl_download_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this
    CALL vec_download_i8_c(this%handle)
  END SUBROUTINE impl_download_i8

  SUBROUTINE impl_fill_zero_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this
    CALL vec_fill_zero_i8_c(this%handle)
  END SUBROUTINE impl_fill_zero_i8

  ! ====================================================================
  ! Original Procedural Wrappers (Partially Hidden/Replaced by Class)
  ! ====================================================================
  ! NOTE: These are kept PRIVATE because the Class replaces them, 
  ! but the implementation logic is here if you ever need to expose them.
  
  FUNCTION vec_create_i4(n, mode) RESULT(ptr)
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN)    :: mode
    TYPE(c_ptr) :: ptr
    ptr = vec_create_i4_c(INT(n, c_size_t), INT(mode, c_int))
  END FUNCTION vec_create_i4

  SUBROUTINE vec_delete_i4(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_delete_i4_c(ptr)
  END SUBROUTINE vec_delete_i4

  FUNCTION vec_host_i4(ptr) RESULT(f_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(4), POINTER :: f_ptr(:)
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    raw_c_ptr = vec_host_i4_c(ptr)
    sz = vec_size_i4_c(ptr) ! Dynamic Query

    IF (sz > 0) THEN
       CALL C_F_POINTER(raw_c_ptr, f_ptr, [sz]) 
    ELSE
       NULLIFY(f_ptr)
    END IF
  END FUNCTION vec_host_i4

  FUNCTION vec_dev_i4(ptr) RESULT(d_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    TYPE(c_ptr) :: d_ptr
    d_ptr = vec_dev_i4_c(ptr)
  END FUNCTION vec_dev_i4

  SUBROUTINE vec_upload_i4(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_upload_i4_c(ptr)
  END SUBROUTINE vec_upload_i4

  SUBROUTINE vec_download_i4(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_download_i4_c(ptr)
  END SUBROUTINE vec_download_i4

  SUBROUTINE vec_upload_part_i4(ptr, start_idx, count)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN)  :: start_idx, count
    CALL vec_upload_part_i4_c(ptr, INT(start_idx - 1, c_size_t), INT(count, c_size_t))
  END SUBROUTINE vec_upload_part_i4

  SUBROUTINE vec_download_part_i4(ptr, start_idx, count)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN)  :: start_idx, count
    CALL vec_download_part_i4_c(ptr, INT(start_idx - 1, c_size_t), INT(count, c_size_t))
  END SUBROUTINE vec_download_part_i4

  SUBROUTINE vec_fill_zero_i4(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_fill_zero_i4_c(ptr)
  END SUBROUTINE vec_fill_zero_i4

  SUBROUTINE vec_set_value_i4(ptr, val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER, INTENT(IN)     :: val
    CALL vec_set_value_i4_c(ptr, INT(val, c_int))
  END SUBROUTINE vec_set_value_i4

  SUBROUTINE vec_gather_i4(src, map, dst)
    TYPE(c_ptr), INTENT(IN) :: src, map, dst
    CALL vec_gather_i4_c(src, map, dst)
  END SUBROUTINE vec_gather_i4

  FUNCTION vec_clone_i4(ptr) RESULT(new_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    TYPE(c_ptr) :: new_ptr
    new_ptr = vec_clone_i4_c(ptr)
  END FUNCTION vec_clone_i4

  ! 🔥 Modified Reductions: Support Optional Real Size (n_opt)
  FUNCTION vec_sum_i4(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    INTEGER(4) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_sum_partial_i4_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_sum_i4_c(ptr)
    END IF
  END FUNCTION vec_sum_i4

  FUNCTION vec_min_i4(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    INTEGER(4) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_min_partial_i4_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_min_i4_c(ptr)
    END IF
  END FUNCTION vec_min_i4

  FUNCTION vec_max_i4(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    INTEGER(4) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_max_partial_i4_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_max_i4_c(ptr)
    END IF
  END FUNCTION vec_max_i4

  ! ====================================================================
  ! Fortran Wrappers: i8
  ! ====================================================================
  FUNCTION vec_create_i8(n, mode) RESULT(ptr)
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN)    :: mode
    TYPE(c_ptr) :: ptr
    ptr = vec_create_i8_c(INT(n, c_size_t), INT(mode, c_int))
  END FUNCTION vec_create_i8

  SUBROUTINE vec_delete_i8(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_delete_i8_c(ptr)
  END SUBROUTINE vec_delete_i8

  FUNCTION vec_host_i8(ptr) RESULT(f_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), POINTER :: f_ptr(:)
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz
    raw_c_ptr = vec_host_i8_c(ptr)
    sz = vec_size_i8_c(ptr)
    IF (sz > 0) THEN
       CALL C_F_POINTER(raw_c_ptr, f_ptr, [sz]) 
    ELSE
       NULLIFY(f_ptr)
    END IF
  END FUNCTION vec_host_i8

  FUNCTION vec_dev_i8(ptr) RESULT(d_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    TYPE(c_ptr) :: d_ptr
    d_ptr = vec_dev_i8_c(ptr)
  END FUNCTION vec_dev_i8

  SUBROUTINE vec_upload_i8(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_upload_i8_c(ptr)
  END SUBROUTINE vec_upload_i8

  SUBROUTINE vec_download_i8(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_download_i8_c(ptr)
  END SUBROUTINE vec_download_i8

  SUBROUTINE vec_fill_zero_i8(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_fill_zero_i8_c(ptr)
  END SUBROUTINE vec_fill_zero_i8

  SUBROUTINE vec_set_value_i8(ptr, val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN)  :: val
    CALL vec_set_value_i8_c(ptr, INT(val, c_long_long))
  END SUBROUTINE vec_set_value_i8

  SUBROUTINE vec_gather_i8(src, map, dst)
    TYPE(c_ptr), INTENT(IN) :: src, map, dst
    CALL vec_gather_i8_c(src, map, dst)
  END SUBROUTINE vec_gather_i8

  FUNCTION vec_clone_i8(ptr) RESULT(new_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    TYPE(c_ptr) :: new_ptr
    new_ptr = vec_clone_i8_c(ptr)
  END FUNCTION vec_clone_i8

  FUNCTION vec_sum_i8(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    INTEGER(8) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_sum_partial_i8_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_sum_i8_c(ptr)
    END IF
  END FUNCTION vec_sum_i8

  FUNCTION vec_min_i8(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    INTEGER(8) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_min_partial_i8_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_min_i8_c(ptr)
    END IF
  END FUNCTION vec_min_i8

  FUNCTION vec_max_i8(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    INTEGER(8) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_max_partial_i8_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_max_i8_c(ptr)
    END IF
  END FUNCTION vec_max_i8

  ! ====================================================================
  ! Fortran Wrappers: r4
  ! ====================================================================
  FUNCTION vec_create_r4(n, mode) RESULT(ptr)
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN)    :: mode
    TYPE(c_ptr) :: ptr
    ptr = vec_create_r4_c(INT(n, c_size_t), INT(mode, c_int))
  END FUNCTION vec_create_r4

  SUBROUTINE vec_delete_r4(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_delete_r4_c(ptr)
  END SUBROUTINE vec_delete_r4

  FUNCTION vec_host_r4(ptr) RESULT(f_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    REAL(4), POINTER :: f_ptr(:)
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz

    raw_c_ptr = vec_host_r4_c(ptr)
    sz = vec_size_r4_c(ptr)

    IF (sz > 0) THEN
       CALL C_F_POINTER(raw_c_ptr, f_ptr, [sz]) 
    ELSE
       NULLIFY(f_ptr)
    END IF

  END FUNCTION vec_host_r4

  FUNCTION vec_dev_r4(ptr) RESULT(d_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    TYPE(c_ptr) :: d_ptr
    d_ptr = vec_dev_r4_c(ptr)
  END FUNCTION vec_dev_r4

  SUBROUTINE vec_upload_r4(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_upload_r4_c(ptr)
  END SUBROUTINE vec_upload_r4

  SUBROUTINE vec_download_r4(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_download_r4_c(ptr)
  END SUBROUTINE vec_download_r4

  SUBROUTINE vec_fill_zero_r4(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_fill_zero_r4_c(ptr)
  END SUBROUTINE vec_fill_zero_r4

  SUBROUTINE vec_set_value_r4(ptr, val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    REAL(4), INTENT(IN)     :: val
    CALL vec_set_value_r4_c(ptr, REAL(val, c_float))
  END SUBROUTINE vec_set_value_r4

  SUBROUTINE vec_gather_r4(src, map, dst)
    TYPE(c_ptr), INTENT(IN) :: src, map, dst
    CALL vec_gather_r4_c(src, map, dst)
  END SUBROUTINE vec_gather_r4

  FUNCTION vec_clone_r4(ptr) RESULT(new_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    TYPE(c_ptr) :: new_ptr
    new_ptr = vec_clone_r4_c(ptr)
  END FUNCTION vec_clone_r4

  FUNCTION vec_sum_r4(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    REAL(4) :: val

    IF (PRESENT(n_opt)) THEN
       val = vec_sum_partial_r4_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_sum_r4_c(ptr)
    END IF

  END FUNCTION vec_sum_r4

  FUNCTION vec_min_r4(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    REAL(4) :: val

    IF (PRESENT(n_opt)) THEN
       val = vec_min_partial_r4_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_min_r4_c(ptr)
    END IF

  END FUNCTION vec_min_r4

  FUNCTION vec_max_r4(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    REAL(4) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_max_partial_r4_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_max_r4_c(ptr)
    END IF
  END FUNCTION vec_max_r4

  ! ====================================================================
  ! Fortran Wrappers: r8
  ! ====================================================================
  FUNCTION vec_create_r8(n, mode) RESULT(ptr)
    INTEGER(8), INTENT(IN) :: n
    INTEGER, INTENT(IN)    :: mode
    TYPE(c_ptr) :: ptr
    ptr = vec_create_r8_c(INT(n, c_size_t), INT(mode, c_int))
  END FUNCTION vec_create_r8

  SUBROUTINE vec_delete_r8(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_delete_r8_c(ptr)
  END SUBROUTINE vec_delete_r8

  FUNCTION vec_host_r8(ptr) RESULT(f_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    REAL(8), POINTER :: f_ptr(:)
    TYPE(c_ptr) :: raw_c_ptr
    INTEGER(c_size_t) :: sz
    raw_c_ptr = vec_host_r8_c(ptr)
    sz = vec_size_r8_c(ptr)
    IF (sz > 0) THEN
       CALL C_F_POINTER(raw_c_ptr, f_ptr, [sz]) 
    ELSE
       NULLIFY(f_ptr)
    END IF
  END FUNCTION vec_host_r8

  FUNCTION vec_dev_r8(ptr) RESULT(d_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    TYPE(c_ptr) :: d_ptr
    d_ptr = vec_dev_r8_c(ptr)
  END FUNCTION vec_dev_r8

  SUBROUTINE vec_upload_r8(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_upload_r8_c(ptr)
  END SUBROUTINE vec_upload_r8

  SUBROUTINE vec_download_r8(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_download_r8_c(ptr)
  END SUBROUTINE vec_download_r8

  SUBROUTINE vec_fill_zero_r8(ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    CALL vec_fill_zero_r8_c(ptr)
  END SUBROUTINE vec_fill_zero_r8

  SUBROUTINE vec_set_value_r8(ptr, val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    REAL(8), INTENT(IN)     :: val
    CALL vec_set_value_r8_c(ptr, REAL(val, c_double))
  END SUBROUTINE vec_set_value_r8

  SUBROUTINE vec_gather_r8(src, map, dst)
    TYPE(c_ptr), INTENT(IN) :: src, map, dst
    CALL vec_gather_r8_c(src, map, dst)
  END SUBROUTINE vec_gather_r8

  FUNCTION vec_clone_r8(ptr) RESULT(new_ptr)
    TYPE(c_ptr), INTENT(IN) :: ptr
    TYPE(c_ptr) :: new_ptr
    new_ptr = vec_clone_r8_c(ptr)
  END FUNCTION vec_clone_r8

  FUNCTION vec_sum_r8(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    REAL(8) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_sum_partial_r8_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_sum_r8_c(ptr)
    END IF
  END FUNCTION vec_sum_r8

  FUNCTION vec_min_r8(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    REAL(8) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_min_partial_r8_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_min_r8_c(ptr)
    END IF
  END FUNCTION vec_min_r8

  FUNCTION vec_max_r8(ptr, n_opt) RESULT(val)
    TYPE(c_ptr), INTENT(IN) :: ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    REAL(8) :: val
    IF (PRESENT(n_opt)) THEN
       val = vec_max_partial_r8_c(ptr, INT(n_opt, c_size_t))
    ELSE
       val = vec_max_r8_c(ptr)
    END IF
  END FUNCTION vec_max_r8

  ! ====================================================================
  ! Sort Wrapper (Corrected)
  ! ====================================================================
  SUBROUTINE vec_sort_i4(keys_ptr, keys_buf_ptr, vals_ptr, vals_buf_ptr, n_opt)
    TYPE(c_ptr), INTENT(IN) :: keys_ptr, keys_buf_ptr, vals_ptr, vals_buf_ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    INTEGER(c_size_t) :: n
    
    ! 1. Determine size (use n_opt if present, else fallback to full aligned size)
    IF (PRESENT(n_opt)) THEN
        n = INT(n_opt, c_size_t)
    ELSE
        n = vec_size_i4_c(keys_ptr)
    END IF
    
    ! 2. Call C++ Sort (Pass HANDLES directly, do NOT extract d_ptr here)
    CALL vec_sort_pairs_i4_c(keys_ptr, keys_buf_ptr, vals_ptr, vals_buf_ptr, n)
  END SUBROUTINE vec_sort_i4

! ====================================================================
  ! Getter 
  ! ====================================================================
  FUNCTION impl_get_handle_i4(this) RESULT(ptr)
    CLASS(device_vector_i4_t), INTENT(IN) :: this
    TYPE(c_ptr) :: ptr
    ptr = this%handle
  END FUNCTION impl_get_handle_i4

  FUNCTION impl_get_handle_i8(this) RESULT(ptr)
    CLASS(device_vector_i8_t), INTENT(IN) :: this
    TYPE(c_ptr) :: ptr
    ptr = this%handle
  END FUNCTION impl_get_handle_i8

  FUNCTION impl_get_handle_r4(this) RESULT(ptr)
    CLASS(device_vector_r4_t), INTENT(IN) :: this
    TYPE(c_ptr) :: ptr
    ptr = this%handle
  END FUNCTION impl_get_handle_r4

  FUNCTION impl_get_handle_r8(this) RESULT(ptr)
    CLASS(device_vector_r8_t), INTENT(IN) :: this
    TYPE(c_ptr) :: ptr
    ptr = this%handle
  END FUNCTION impl_get_handle_r8


  ! --------------------------------------------------------------------
  ! Getter: Raw Device Pointer (for OpenACC/OpenMP interop, etc.)
  ! NOTE:
  !   - Returned pointer is a raw CUDA device pointer (void* on C side).
  !   - If the vector is not created, returns C_NULL_PTR.
  !   - If the vector is resized, the returned pointer may change.
  ! --------------------------------------------------------------------
  FUNCTION impl_get_device_ptr_i4(this) RESULT(p)
    CLASS(device_vector_i4_t), INTENT(IN) :: this
    TYPE(c_ptr) :: p
    IF (c_associated(this%handle)) THEN
      p = vec_dev_i4_c(this%handle)
    ELSE
      p = C_NULL_PTR
    END IF
  END FUNCTION impl_get_device_ptr_i4

  FUNCTION impl_get_device_ptr_i8(this) RESULT(p)
    CLASS(device_vector_i8_t), INTENT(IN) :: this
    TYPE(c_ptr) :: p
    IF (c_associated(this%handle)) THEN
      p = vec_dev_i8_c(this%handle)
    ELSE
      p = C_NULL_PTR
    END IF
  END FUNCTION impl_get_device_ptr_i8

  FUNCTION impl_get_device_ptr_r4(this) RESULT(p)
    CLASS(device_vector_r4_t), INTENT(IN) :: this
    TYPE(c_ptr) :: p
    IF (c_associated(this%handle)) THEN
      p = vec_dev_r4_c(this%handle)
    ELSE
      p = C_NULL_PTR
    END IF
  END FUNCTION impl_get_device_ptr_r4

  FUNCTION impl_get_device_ptr_r8(this) RESULT(p)
    CLASS(device_vector_r8_t), INTENT(IN) :: this
    TYPE(c_ptr) :: p
    IF (c_associated(this%handle)) THEN
      p = vec_dev_r8_c(this%handle)
    ELSE
      p = C_NULL_PTR
    END IF
  END FUNCTION impl_get_device_ptr_r8


END MODULE Device_Vector