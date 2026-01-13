MODULE Device_Vector
  USE iso_c_binding
  USE Device_OpenACC_Interface
  IMPLICIT NONE

  PRIVATE 
  PUBLIC :: device_vector_i4_t, device_vector_i8_t
  PUBLIC :: device_vector_r4_t, device_vector_r8_t
  PUBLIC :: device_env_init, device_env_finalize, device_synchronize
  PUBLIC :: vec_sort_i4

  ! ====================================================================
  ! 1. C FUNCTION INTERFACES (完全展開，絕無省略)
  ! ====================================================================
  INTERFACE
    SUBROUTINE device_env_init(rank, gpus_per_node) BIND(C, name="device_env_init")
      IMPORT :: c_int; INTEGER(c_int), VALUE :: rank, gpus_per_node
    END SUBROUTINE
    SUBROUTINE device_env_finalize() BIND(C, name="device_env_finalize")
    END SUBROUTINE
    SUBROUTINE device_synchronize() BIND(C, name="device_synchronize")
    END SUBROUTINE

    ! --- INTEGER 4 ---
    FUNCTION vec_new_vector_i4_c(n) RESULT(res) BIND(C, name="vec_new_vector_i4")
      IMPORT :: c_size_t, c_ptr; INTEGER(c_size_t), VALUE :: n; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_new_buffer_i4_c(n, pinned) RESULT(res) BIND(C, name="vec_new_buffer_i4")
      IMPORT :: c_size_t, c_bool, c_ptr; INTEGER(c_size_t), VALUE :: n; LOGICAL(c_bool), VALUE :: pinned; TYPE(C_PTR) :: res
    END FUNCTION
    SUBROUTINE vec_delete_i4_c(h) BIND(C, name="vec_delete_i4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    SUBROUTINE vec_resize_i4_c(h, n) BIND(C, name="vec_resize_i4")
      IMPORT :: c_ptr, c_size_t; TYPE(c_ptr), VALUE :: h; INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE
    SUBROUTINE vec_copy_from_i4_c(dest, src) BIND(C, name="vec_copy_from_i4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: dest, src
    END SUBROUTINE
    FUNCTION vec_host_i4_c(h) RESULT(res) BIND(C, name="vec_host_i4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_dev_i4_c(h) RESULT(res) BIND(C, name="vec_dev_i4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_size_i4_c(h) RESULT(res) BIND(C, name="vec_size_i4")
      IMPORT :: c_ptr, c_size_t; TYPE(c_ptr), VALUE :: h; INTEGER(c_size_t) :: res
    END FUNCTION
    SUBROUTINE vec_upload_i4_c(h) BIND(C, name="vec_upload_i4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    SUBROUTINE vec_download_i4_c(h) BIND(C, name="vec_download_i4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    FUNCTION vec_sum_i4_c(h) RESULT(res) BIND(C, name="vec_sum_i4")
      IMPORT :: c_ptr, c_int; TYPE(c_ptr), VALUE :: h; INTEGER(c_int) :: res
    END FUNCTION
    FUNCTION vec_min_i4_c(h) RESULT(res) BIND(C, name="vec_min_i4")
      IMPORT :: c_ptr, c_int; TYPE(c_ptr), VALUE :: h; INTEGER(c_int) :: res
    END FUNCTION
    FUNCTION vec_max_i4_c(h) RESULT(res) BIND(C, name="vec_max_i4")
      IMPORT :: c_ptr, c_int; TYPE(c_ptr), VALUE :: h; INTEGER(c_int) :: res
    END FUNCTION

    ! --- INTEGER 8 ---
    FUNCTION vec_new_vector_i8_c(n) RESULT(res) BIND(C, name="vec_new_vector_i8")
      IMPORT :: c_size_t, c_ptr; INTEGER(c_size_t), VALUE :: n; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_new_buffer_i8_c(n, pinned) RESULT(res) BIND(C, name="vec_new_buffer_i8")
      IMPORT :: c_size_t, c_bool, c_ptr; INTEGER(c_size_t), VALUE :: n; LOGICAL(c_bool), VALUE :: pinned; TYPE(C_PTR) :: res
    END FUNCTION
    SUBROUTINE vec_delete_i8_c(h) BIND(C, name="vec_delete_i8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    SUBROUTINE vec_resize_i8_c(h, n) BIND(C, name="vec_resize_i8")
      IMPORT :: c_ptr, c_size_t; TYPE(c_ptr), VALUE :: h; INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE
    SUBROUTINE vec_copy_from_i8_c(dest, src) BIND(C, name="vec_copy_from_i8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: dest, src
    END SUBROUTINE
    FUNCTION vec_host_i8_c(h) RESULT(res) BIND(C, name="vec_host_i8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_dev_i8_c(h) RESULT(res) BIND(C, name="vec_dev_i8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_size_i8_c(h) RESULT(res) BIND(C, name="vec_size_i8")
      IMPORT :: c_ptr, c_size_t; TYPE(c_ptr), VALUE :: h; INTEGER(c_size_t) :: res
    END FUNCTION
    SUBROUTINE vec_upload_i8_c(h) BIND(C, name="vec_upload_i8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    SUBROUTINE vec_download_i8_c(h) BIND(C, name="vec_download_i8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    FUNCTION vec_sum_i8_c(h) RESULT(res) BIND(C, name="vec_sum_i8")
      IMPORT :: c_ptr, c_long_long; TYPE(c_ptr), VALUE :: h; INTEGER(c_long_long) :: res
    END FUNCTION
    FUNCTION vec_min_i8_c(h) RESULT(res) BIND(C, name="vec_min_i8")
      IMPORT :: c_ptr, c_long_long; TYPE(c_ptr), VALUE :: h; INTEGER(c_long_long) :: res
    END FUNCTION
    FUNCTION vec_max_i8_c(h) RESULT(res) BIND(C, name="vec_max_i8")
      IMPORT :: c_ptr, c_long_long; TYPE(c_ptr), VALUE :: h; INTEGER(c_long_long) :: res
    END FUNCTION

    ! --- REAL 4 ---
    FUNCTION vec_new_vector_r4_c(n) RESULT(res) BIND(C, name="vec_new_vector_r4")
      IMPORT :: c_size_t, c_ptr; INTEGER(c_size_t), VALUE :: n; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_new_buffer_r4_c(n, pinned) RESULT(res) BIND(C, name="vec_new_buffer_r4")
      IMPORT :: c_size_t, c_bool, c_ptr; INTEGER(c_size_t), VALUE :: n; LOGICAL(c_bool), VALUE :: pinned; TYPE(C_PTR) :: res
    END FUNCTION
    SUBROUTINE vec_delete_r4_c(h) BIND(C, name="vec_delete_r4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    SUBROUTINE vec_resize_r4_c(h, n) BIND(C, name="vec_resize_r4")
      IMPORT :: c_ptr, c_size_t; TYPE(c_ptr), VALUE :: h; INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE
    SUBROUTINE vec_copy_from_r4_c(dest, src) BIND(C, name="vec_copy_from_r4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: dest, src
    END SUBROUTINE
    FUNCTION vec_host_r4_c(h) RESULT(res) BIND(C, name="vec_host_r4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_dev_r4_c(h) RESULT(res) BIND(C, name="vec_dev_r4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_size_r4_c(h) RESULT(res) BIND(C, name="vec_size_r4")
      IMPORT :: c_ptr, c_size_t; TYPE(c_ptr), VALUE :: h; INTEGER(c_size_t) :: res
    END FUNCTION
    SUBROUTINE vec_upload_r4_c(h) BIND(C, name="vec_upload_r4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    SUBROUTINE vec_download_r4_c(h) BIND(C, name="vec_download_r4")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    FUNCTION vec_sum_r4_c(h) RESULT(res) BIND(C, name="vec_sum_r4")
      IMPORT :: c_ptr, c_float; TYPE(c_ptr), VALUE :: h; REAL(c_float) :: res
    END FUNCTION
    FUNCTION vec_min_r4_c(h) RESULT(res) BIND(C, name="vec_min_r4")
      IMPORT :: c_ptr, c_float; TYPE(c_ptr), VALUE :: h; REAL(c_float) :: res
    END FUNCTION
    FUNCTION vec_max_r4_c(h) RESULT(res) BIND(C, name="vec_max_r4")
      IMPORT :: c_ptr, c_float; TYPE(c_ptr), VALUE :: h; REAL(c_float) :: res
    END FUNCTION

    ! --- REAL 8 ---
    FUNCTION vec_new_vector_r8_c(n) RESULT(res) BIND(C, name="vec_new_vector_r8")
      IMPORT :: c_size_t, c_ptr; INTEGER(c_size_t), VALUE :: n; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_new_buffer_r8_c(n, pinned) RESULT(res) BIND(C, name="vec_new_buffer_r8")
      IMPORT :: c_size_t, c_bool, c_ptr; INTEGER(c_size_t), VALUE :: n; LOGICAL(c_bool), VALUE :: pinned; TYPE(C_PTR) :: res
    END FUNCTION
    SUBROUTINE vec_delete_r8_c(h) BIND(C, name="vec_delete_r8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    SUBROUTINE vec_resize_r8_c(h, n) BIND(C, name="vec_resize_r8")
      IMPORT :: c_ptr, c_size_t; TYPE(c_ptr), VALUE :: h; INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE
    SUBROUTINE vec_copy_from_r8_c(dest, src) BIND(C, name="vec_copy_from_r8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: dest, src
    END SUBROUTINE
    FUNCTION vec_host_r8_c(h) RESULT(res) BIND(C, name="vec_host_r8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_dev_r8_c(h) RESULT(res) BIND(C, name="vec_dev_r8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h; TYPE(C_PTR) :: res
    END FUNCTION
    FUNCTION vec_size_r8_c(h) RESULT(res) BIND(C, name="vec_size_r8")
      IMPORT :: c_ptr, c_size_t; TYPE(c_ptr), VALUE :: h; INTEGER(c_size_t) :: res
    END FUNCTION
    SUBROUTINE vec_upload_r8_c(h) BIND(C, name="vec_upload_r8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    SUBROUTINE vec_download_r8_c(h) BIND(C, name="vec_download_r8")
      IMPORT :: c_ptr; TYPE(c_ptr), VALUE :: h
    END SUBROUTINE
    FUNCTION vec_sum_r8_c(h) RESULT(res) BIND(C, name="vec_sum_r8")
      IMPORT :: c_ptr, c_double; TYPE(c_ptr), VALUE :: h; REAL(c_double) :: res
    END FUNCTION
    FUNCTION vec_min_r8_c(h) RESULT(res) BIND(C, name="vec_min_r8")
      IMPORT :: c_ptr, c_double; TYPE(c_ptr), VALUE :: h; REAL(c_double) :: res
    END FUNCTION
    FUNCTION vec_max_r8_c(h) RESULT(res) BIND(C, name="vec_max_r8")
      IMPORT :: c_ptr, c_double; TYPE(c_ptr), VALUE :: h; REAL(c_double) :: res
    END FUNCTION

    SUBROUTINE vec_sort_pairs_i4_c(kin, kbuf, vin, vbuf, n) BIND(C, name="vec_sort_pairs_i4_c")
      IMPORT :: c_ptr, c_size_t
      TYPE(C_PTR), VALUE :: kin, kbuf, vin, vbuf; INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE
  END INTERFACE

  ! ====================================================================
  ! 2. TYPE DEFINITIONS (完全展開)
  ! ====================================================================

  TYPE :: device_vector_i4_t
      TYPE(c_ptr) :: handle = C_NULL_PTR
      INTEGER(4), POINTER :: ptr(:) => NULL()
    CONTAINS
      PROCEDURE :: create_vector => impl_create_vector_i4
      PROCEDURE :: create_buffer => impl_create_buffer_i4
      PROCEDURE :: free          => impl_free_i4
      PROCEDURE :: resize        => impl_resize_i4
      PROCEDURE :: copy_from     => impl_copy_from_i4
      PROCEDURE :: get_handle    => impl_get_handle_i4
      PROCEDURE :: device_ptr    => impl_device_ptr_i4
      PROCEDURE :: upload        => impl_upload_i4
      PROCEDURE :: download      => impl_download_i4
      PROCEDURE :: size          => impl_size_i4
      PROCEDURE :: sum           => impl_sum_i4
      PROCEDURE :: min           => impl_min_i4
      PROCEDURE :: max           => impl_max_i4
      PROCEDURE :: acc_map       => impl_acc_map_i4
      PROCEDURE :: acc_unmap     => impl_acc_unmap_i4
  END TYPE

  TYPE :: device_vector_i8_t
      TYPE(c_ptr) :: handle = C_NULL_PTR
      INTEGER(8), POINTER :: ptr(:) => NULL()
    CONTAINS
      PROCEDURE :: create_vector => impl_create_vector_i8
      PROCEDURE :: create_buffer => impl_create_buffer_i8
      PROCEDURE :: free          => impl_free_i8
      PROCEDURE :: resize        => impl_resize_i8
      PROCEDURE :: copy_from     => impl_copy_from_i8
      PROCEDURE :: get_handle    => impl_get_handle_i8
      PROCEDURE :: device_ptr    => impl_device_ptr_i8
      PROCEDURE :: upload        => impl_upload_i8
      PROCEDURE :: download      => impl_download_i8
      PROCEDURE :: size          => impl_size_i8
      PROCEDURE :: sum           => impl_sum_i8
      PROCEDURE :: min           => impl_min_i8
      PROCEDURE :: max           => impl_max_i8
      PROCEDURE :: acc_map       => impl_acc_map_i8
      PROCEDURE :: acc_unmap     => impl_acc_unmap_i8
  END TYPE

  TYPE :: device_vector_r4_t
      TYPE(c_ptr) :: handle = C_NULL_PTR
      REAL(4), POINTER :: ptr(:) => NULL()
    CONTAINS
      PROCEDURE :: create_vector => impl_create_vector_r4
      PROCEDURE :: create_buffer => impl_create_buffer_r4
      PROCEDURE :: free          => impl_free_r4
      PROCEDURE :: resize        => impl_resize_r4
      PROCEDURE :: copy_from     => impl_copy_from_r4
      PROCEDURE :: get_handle    => impl_get_handle_r4
      PROCEDURE :: device_ptr    => impl_device_ptr_r4
      PROCEDURE :: upload        => impl_upload_r4
      PROCEDURE :: download      => impl_download_r4
      PROCEDURE :: size          => impl_size_r4
      PROCEDURE :: sum           => impl_sum_r4
      PROCEDURE :: min           => impl_min_r4
      PROCEDURE :: max           => impl_max_r4
      PROCEDURE :: acc_map       => impl_acc_map_r4
      PROCEDURE :: acc_unmap     => impl_acc_unmap_r4
  END TYPE

  TYPE :: device_vector_r8_t
      TYPE(c_ptr) :: handle = C_NULL_PTR
      REAL(8), POINTER :: ptr(:) => NULL()
    CONTAINS
      PROCEDURE :: create_vector => impl_create_vector_r8
      PROCEDURE :: create_buffer => impl_create_buffer_r8
      PROCEDURE :: free          => impl_free_r8
      PROCEDURE :: resize        => impl_resize_r8
      PROCEDURE :: copy_from     => impl_copy_from_r8
      PROCEDURE :: get_handle    => impl_get_handle_r8
      PROCEDURE :: device_ptr    => impl_device_ptr_r8
      PROCEDURE :: upload        => impl_upload_r8
      PROCEDURE :: download      => impl_download_r8
      PROCEDURE :: size          => impl_size_r8
      PROCEDURE :: sum           => impl_sum_r8
      PROCEDURE :: min           => impl_min_r8
      PROCEDURE :: max           => impl_max_r8
      PROCEDURE :: acc_map       => impl_acc_map_r8
      PROCEDURE :: acc_unmap     => impl_acc_unmap_r8
  END TYPE

CONTAINS

  ! ====================================================================
  ! 3. IMPLEMENTATIONS: INTEGER 4
  ! ====================================================================
  SUBROUTINE impl_create_vector_i4(this, n)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n
    this%handle = vec_new_vector_i4_c(INT(n, c_size_t)); CALL sync_ptr_i4(this)
  END SUBROUTINE
  SUBROUTINE impl_create_buffer_i4(this, n, pinned)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n; LOGICAL, INTENT(IN), OPTIONAL :: pinned
    LOGICAL(c_bool) :: is_p; is_p = .TRUE.; IF(PRESENT(pinned)) is_p = pinned
    this%handle = vec_new_buffer_i4_c(INT(n, c_size_t), is_p); CALL sync_ptr_i4(this)
  END SUBROUTINE
  SUBROUTINE impl_free_i4(this)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    IF(C_ASSOCIATED(this%handle)) CALL vec_delete_i4_c(this%handle); this%handle = C_NULL_PTR; NULLIFY(this%ptr)
  END SUBROUTINE
  SUBROUTINE impl_resize_i4(this, n)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n
    CALL vec_resize_i4_c(this%handle, INT(n, c_size_t)); CALL sync_ptr_i4(this)
  END SUBROUTINE
  SUBROUTINE impl_copy_from_i4(this, other)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this; CLASS(device_vector_i4_t), INTENT(IN) :: other
    CALL vec_copy_from_i4_c(this%handle, other%handle); CALL sync_ptr_i4(this)
  END SUBROUTINE
  FUNCTION impl_get_handle_i4(this) RESULT(res)
    CLASS(device_vector_i4_t), INTENT(IN) :: this; TYPE(C_PTR) :: res; res = this%handle
  END FUNCTION
  FUNCTION impl_device_ptr_i4(this) RESULT(res)
    CLASS(device_vector_i4_t), INTENT(IN) :: this; TYPE(C_PTR) :: res; res = vec_dev_i4_c(this%handle)
  END FUNCTION
  SUBROUTINE impl_upload_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this; CALL vec_upload_i4_c(this%handle)
  END SUBROUTINE
  SUBROUTINE impl_download_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this; CALL vec_download_i4_c(this%handle)
  END SUBROUTINE
  INTEGER(8) FUNCTION impl_size_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this; impl_size_i4 = INT(vec_size_i4_c(this%handle), 8)
  END FUNCTION
  INTEGER(4) FUNCTION impl_sum_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this; impl_sum_i4 = vec_sum_i4_c(this%handle)
  END FUNCTION
  INTEGER(4) FUNCTION impl_min_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this; impl_min_i4 = vec_min_i4_c(this%handle)
  END FUNCTION
  INTEGER(4) FUNCTION impl_max_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this; impl_max_i4 = vec_max_i4_c(this%handle)
  END FUNCTION
  SUBROUTINE impl_acc_map_i4(this, p_opt)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this; INTEGER(4), POINTER, OPTIONAL, INTENT(OUT) :: p_opt(:)
    CALL vec_acc_map_i4(this%handle); IF(PRESENT(p_opt)) p_opt => this%ptr
  END SUBROUTINE
  SUBROUTINE impl_acc_unmap_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this; CALL vec_acc_unmap_i4(this%handle)
  END SUBROUTINE
  SUBROUTINE sync_ptr_i4(this)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this; TYPE(C_PTR) :: h_ptr; INTEGER(8) :: n
    h_ptr = vec_host_i4_c(this%handle); n = impl_size_i4(this)
    IF(C_ASSOCIATED(h_ptr).AND.n>0) CALL C_F_POINTER(h_ptr, this%ptr, [n])
  END SUBROUTINE

  ! ====================================================================
  ! 4. IMPLEMENTATIONS: INTEGER 8
  ! ====================================================================
  SUBROUTINE impl_create_vector_i8(this, n)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n
    this%handle = vec_new_vector_i8_c(INT(n, c_size_t)); CALL sync_ptr_i8(this)
  END SUBROUTINE
  SUBROUTINE impl_create_buffer_i8(this, n, pinned)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n; LOGICAL, INTENT(IN), OPTIONAL :: pinned
    LOGICAL(c_bool) :: is_p; is_p = .TRUE.; IF(PRESENT(pinned)) is_p = pinned
    this%handle = vec_new_buffer_i8_c(INT(n, c_size_t), is_p); CALL sync_ptr_i8(this)
  END SUBROUTINE
  SUBROUTINE impl_free_i8(this)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    IF(C_ASSOCIATED(this%handle)) CALL vec_delete_i8_c(this%handle); this%handle = C_NULL_PTR; NULLIFY(this%ptr)
  END SUBROUTINE
  SUBROUTINE impl_resize_i8(this, n)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n
    CALL vec_resize_i8_c(this%handle, INT(n, c_size_t)); CALL sync_ptr_i8(this)
  END SUBROUTINE
  SUBROUTINE impl_copy_from_i8(this, other)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this; CLASS(device_vector_i8_t), INTENT(IN) :: other
    CALL vec_copy_from_i8_c(this%handle, other%handle); CALL sync_ptr_i8(this)
  END SUBROUTINE
  FUNCTION impl_get_handle_i8(this) RESULT(res)
    CLASS(device_vector_i8_t), INTENT(IN) :: this; TYPE(C_PTR) :: res; res = this%handle
  END FUNCTION
  FUNCTION impl_device_ptr_i8(this) RESULT(res)
    CLASS(device_vector_i8_t), INTENT(IN) :: this; TYPE(C_PTR) :: res; res = vec_dev_i8_c(this%handle)
  END FUNCTION
  SUBROUTINE impl_upload_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this; CALL vec_upload_i8_c(this%handle)
  END SUBROUTINE
  SUBROUTINE impl_download_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this; CALL vec_download_i8_c(this%handle)
  END SUBROUTINE
  INTEGER(8) FUNCTION impl_size_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this; impl_size_i8 = INT(vec_size_i8_c(this%handle), 8)
  END FUNCTION
  INTEGER(8) FUNCTION impl_sum_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this; impl_sum_i8 = vec_sum_i8_c(this%handle)
  END FUNCTION
  INTEGER(8) FUNCTION impl_min_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this; impl_min_i8 = vec_min_i8_c(this%handle)
  END FUNCTION
  INTEGER(8) FUNCTION impl_max_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this; impl_max_i8 = vec_max_i8_c(this%handle)
  END FUNCTION
  SUBROUTINE impl_acc_map_i8(this, p_opt)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this; INTEGER(8), POINTER, OPTIONAL, INTENT(OUT) :: p_opt(:)
    CALL vec_acc_map_i8(this%handle); IF(PRESENT(p_opt)) p_opt => this%ptr
  END SUBROUTINE
  SUBROUTINE impl_acc_unmap_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this; CALL vec_acc_unmap_i8(this%handle)
  END SUBROUTINE
  SUBROUTINE sync_ptr_i8(this)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this; TYPE(C_PTR) :: h_ptr; INTEGER(8) :: n
    h_ptr = vec_host_i8_c(this%handle); n = impl_size_i8(this)
    IF(C_ASSOCIATED(h_ptr).AND.n>0) CALL C_F_POINTER(h_ptr, this%ptr, [n])
  END SUBROUTINE

  ! ====================================================================
  ! 5. IMPLEMENTATIONS: REAL 4
  ! ====================================================================
  SUBROUTINE impl_create_vector_r4(this, n)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n
    this%handle = vec_new_vector_r4_c(INT(n, c_size_t)); CALL sync_ptr_r4(this)
  END SUBROUTINE
  SUBROUTINE impl_create_buffer_r4(this, n, pinned)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n; LOGICAL, INTENT(IN), OPTIONAL :: pinned
    LOGICAL(c_bool) :: is_p; is_p = .TRUE.; IF(PRESENT(pinned)) is_p = pinned
    this%handle = vec_new_buffer_r4_c(INT(n, c_size_t), is_p); CALL sync_ptr_r4(this)
  END SUBROUTINE
  SUBROUTINE impl_free_r4(this)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    IF(C_ASSOCIATED(this%handle)) CALL vec_delete_r4_c(this%handle); this%handle = C_NULL_PTR; NULLIFY(this%ptr)
  END SUBROUTINE
  SUBROUTINE impl_resize_r4(this, n)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n
    CALL vec_resize_r4_c(this%handle, INT(n, c_size_t)); CALL sync_ptr_r4(this)
  END SUBROUTINE
  SUBROUTINE impl_copy_from_r4(this, other)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this; CLASS(device_vector_r4_t), INTENT(IN) :: other
    CALL vec_copy_from_r4_c(this%handle, other%handle); CALL sync_ptr_r4(this)
  END SUBROUTINE
  FUNCTION impl_get_handle_r4(this) RESULT(res)
    CLASS(device_vector_r4_t), INTENT(IN) :: this; TYPE(C_PTR) :: res; res = this%handle
  END FUNCTION
  FUNCTION impl_device_ptr_r4(this) RESULT(res)
    CLASS(device_vector_r4_t), INTENT(IN) :: this; TYPE(C_PTR) :: res; res = vec_dev_r4_c(this%handle)
  END FUNCTION
  SUBROUTINE impl_upload_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this; CALL vec_upload_r4_c(this%handle)
  END SUBROUTINE
  SUBROUTINE impl_download_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this; CALL vec_download_r4_c(this%handle)
  END SUBROUTINE
  INTEGER(8) FUNCTION impl_size_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this; impl_size_r4 = INT(vec_size_r4_c(this%handle), 8)
  END FUNCTION
  REAL(4) FUNCTION impl_sum_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this; impl_sum_r4 = vec_sum_r4_c(this%handle)
  END FUNCTION
  REAL(4) FUNCTION impl_min_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this; impl_min_r4 = vec_min_r4_c(this%handle)
  END FUNCTION
  REAL(4) FUNCTION impl_max_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this; impl_max_r4 = vec_max_r4_c(this%handle)
  END FUNCTION
  SUBROUTINE impl_acc_map_r4(this, p_opt)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this; REAL(4), POINTER, OPTIONAL, INTENT(OUT) :: p_opt(:)
    CALL vec_acc_map_r4(this%handle); IF(PRESENT(p_opt)) p_opt => this%ptr
  END SUBROUTINE
  SUBROUTINE impl_acc_unmap_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this; CALL vec_acc_unmap_r4(this%handle)
  END SUBROUTINE
  SUBROUTINE sync_ptr_r4(this)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this; TYPE(C_PTR) :: h_ptr; INTEGER(8) :: n
    h_ptr = vec_host_r4_c(this%handle); n = impl_size_r4(this)
    IF(C_ASSOCIATED(h_ptr).AND.n>0) CALL C_F_POINTER(h_ptr, this%ptr, [n])
  END SUBROUTINE

  ! ====================================================================
  ! 6. IMPLEMENTATIONS: REAL 8
  ! ====================================================================
  SUBROUTINE impl_create_vector_r8(this, n)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n
    this%handle = vec_new_vector_r8_c(INT(n, c_size_t)); CALL sync_ptr_r8(this)
  END SUBROUTINE
  SUBROUTINE impl_create_buffer_r8(this, n, pinned)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n; LOGICAL, INTENT(IN), OPTIONAL :: pinned
    LOGICAL(c_bool) :: is_p; is_p = .TRUE.; IF(PRESENT(pinned)) is_p = pinned
    this%handle = vec_new_buffer_r8_c(INT(n, c_size_t), is_p); CALL sync_ptr_r8(this)
  END SUBROUTINE
  SUBROUTINE impl_free_r8(this)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    IF(C_ASSOCIATED(this%handle)) CALL vec_delete_r8_c(this%handle); this%handle = C_NULL_PTR; NULLIFY(this%ptr)
  END SUBROUTINE
  SUBROUTINE impl_resize_r8(this, n)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this; INTEGER(8), INTENT(IN) :: n
    CALL vec_resize_r8_c(this%handle, INT(n, c_size_t)); CALL sync_ptr_r8(this)
  END SUBROUTINE
  SUBROUTINE impl_copy_from_r8(this, other)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this; CLASS(device_vector_r8_t), INTENT(IN) :: other
    CALL vec_copy_from_r8_c(this%handle, other%handle); CALL sync_ptr_r8(this)
  END SUBROUTINE
  FUNCTION impl_get_handle_r8(this) RESULT(res)
    CLASS(device_vector_r8_t), INTENT(IN) :: this; TYPE(C_PTR) :: res; res = this%handle
  END FUNCTION
  FUNCTION impl_device_ptr_r8(this) RESULT(res)
    CLASS(device_vector_r8_t), INTENT(IN) :: this; TYPE(C_PTR) :: res; res = vec_dev_r8_c(this%handle)
  END FUNCTION
  SUBROUTINE impl_upload_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this; CALL vec_upload_r8_c(this%handle)
  END SUBROUTINE
  SUBROUTINE impl_download_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this; CALL vec_download_r8_c(this%handle)
  END SUBROUTINE
  INTEGER(8) FUNCTION impl_size_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this; impl_size_r8 = INT(vec_size_r8_c(this%handle), 8)
  END FUNCTION
  REAL(8) FUNCTION impl_sum_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this; impl_sum_r8 = vec_sum_r8_c(this%handle)
  END FUNCTION
  REAL(8) FUNCTION impl_min_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this; impl_min_r8 = vec_min_r8_c(this%handle)
  END FUNCTION
  REAL(8) FUNCTION impl_max_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this; impl_max_r8 = vec_max_r8_c(this%handle)
  END FUNCTION
  SUBROUTINE impl_acc_map_r8(this, p_opt)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this; REAL(8), POINTER, OPTIONAL, INTENT(OUT) :: p_opt(:)
    CALL vec_acc_map_r8(this%handle); IF(PRESENT(p_opt)) p_opt => this%ptr
  END SUBROUTINE
  SUBROUTINE impl_acc_unmap_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this; CALL vec_acc_unmap_r8(this%handle)
  END SUBROUTINE
  SUBROUTINE sync_ptr_r8(this)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this; TYPE(C_PTR) :: h_ptr; INTEGER(8) :: n
    h_ptr = vec_host_r8_c(this%handle); n = impl_size_r8(this)
    IF(C_ASSOCIATED(h_ptr).AND.n>0) CALL C_F_POINTER(h_ptr, this%ptr, [n])
  END SUBROUTINE

  ! ====================================================================
  ! 7. ALGORITHMS
  ! ====================================================================
  SUBROUTINE vec_sort_i4(keys_ptr, keys_buf_ptr, vals_ptr, vals_buf_ptr, n_opt)
    TYPE(c_ptr), INTENT(IN) :: keys_ptr, keys_buf_ptr, vals_ptr, vals_buf_ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt; INTEGER(c_size_t) :: n
    n = vec_size_i4_c(keys_ptr); IF(PRESENT(n_opt)) n = INT(n_opt, c_size_t)
    CALL vec_sort_pairs_i4_c(keys_ptr, keys_buf_ptr, vals_ptr, vals_buf_ptr, n)
  END SUBROUTINE

END MODULE Device_Vector