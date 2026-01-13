MODULE Device_Vector
  USE iso_c_binding
  IMPLICIT NONE

  ! ====================================================================
  ! 1. Visibility
  ! ====================================================================
  PRIVATE 
  
  ! Public Types
  PUBLIC :: device_vector_i4_t
  PUBLIC :: device_vector_i8_t
  PUBLIC :: device_vector_r4_t
  PUBLIC :: device_vector_r8_t
  
  ! Public Env Functions
  PUBLIC :: device_env_init
  PUBLIC :: device_env_finalize
  PUBLIC :: device_synchronize
  
  ! Public Algo
  PUBLIC :: vec_sort_i4

  ! ====================================================================
  ! 2. C Function Interfaces
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

    ! ==================================================================
    ! [I4] Integer (32-bit)
    ! ==================================================================
    ! 1. Semantic Constructors
    FUNCTION vec_new_vector_i4_c(n) RESULT(res) BIND(C, name="vec_new_vector_i4")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      TYPE(C_PTR) :: res
    END FUNCTION
    
    FUNCTION vec_new_buffer_i4_c(n, pinned) RESULT(res) BIND(C, name="vec_new_buffer_i4")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      LOGICAL(c_bool), VALUE :: pinned
      TYPE(C_PTR) :: res
    END FUNCTION

    ! 2. Common Ops
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

    SUBROUTINE vec_copy_from_i4_c(dst, src) BIND(C, name="vec_copy_from_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: dst, src
    END SUBROUTINE

    ! 3. Getters
    FUNCTION vec_host_i4_c(h) RESULT(res) BIND(C, name="vec_host_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION

    FUNCTION vec_dev_i4_c(h) RESULT(res) BIND(C, name="vec_dev_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION
    
    FUNCTION vec_size_i4_c(h) RESULT(res) BIND(C, name="vec_size_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: res
    END FUNCTION
    
    FUNCTION vec_capacity_i4_c(h) RESULT(res) BIND(C, name="vec_capacity_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: res
    END FUNCTION

    ! 4. Transfers
    SUBROUTINE vec_upload_i4_c(h) BIND(C, name="vec_upload_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_download_i4_c(h) BIND(C, name="vec_download_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
    END SUBROUTINE

    SUBROUTINE vec_upload_part_i4_c(h, off, cnt) BIND(C, name="vec_upload_part_i4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: off, cnt
    END SUBROUTINE

    SUBROUTINE vec_download_part_i4_c(h, off, cnt) BIND(C, name="vec_download_part_i4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: off, cnt
    END SUBROUTINE

    ! 5. Utils
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

    FUNCTION vec_clone_i4_c(h) RESULT(res) BIND(C, name="vec_clone_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION

    ! 6. Reductions
    FUNCTION vec_sum_i4_c(h) RESULT(res) BIND(C, name="vec_sum_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_int)     :: res
    END FUNCTION
    
    FUNCTION vec_sum_partial_i4_c(h, n) RESULT(res) BIND(C, name="vec_sum_partial_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int)     :: res
    END FUNCTION
    
    ! (Min/Max omitted for brevity in partial, usually sum is most critical, but adding placeholders)
    FUNCTION vec_min_i4_c(h) RESULT(res) BIND(C, name="vec_min_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_int)     :: res
    END FUNCTION
    FUNCTION vec_min_partial_i4_c(h, n) RESULT(res) BIND(C, name="vec_min_partial_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int)     :: res
    END FUNCTION

    FUNCTION vec_max_i4_c(h) RESULT(res) BIND(C, name="vec_max_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_int)     :: res
    END FUNCTION
    FUNCTION vec_max_partial_i4_c(h, n) RESULT(res) BIND(C, name="vec_max_partial_i4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_int)     :: res
    END FUNCTION


    ! ==================================================================
    ! [I8] Integer (64-bit)
    ! ==================================================================
    FUNCTION vec_new_vector_i8_c(n) RESULT(res) BIND(C, name="vec_new_vector_i8")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      TYPE(C_PTR) :: res
    END FUNCTION
    
    FUNCTION vec_new_buffer_i8_c(n, pinned) RESULT(res) BIND(C, name="vec_new_buffer_i8")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      LOGICAL(c_bool), VALUE :: pinned
      TYPE(C_PTR) :: res
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

    SUBROUTINE vec_reserve_i8_c(h, n) BIND(C, name="vec_reserve_i8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE

    SUBROUTINE vec_copy_from_i8_c(dst, src) BIND(C, name="vec_copy_from_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: dst, src
    END SUBROUTINE

    FUNCTION vec_host_i8_c(h) RESULT(res) BIND(C, name="vec_host_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION

    FUNCTION vec_dev_i8_c(h) RESULT(res) BIND(C, name="vec_dev_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION
    
    FUNCTION vec_size_i8_c(h) RESULT(res) BIND(C, name="vec_size_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: res
    END FUNCTION
    
    FUNCTION vec_capacity_i8_c(h) RESULT(res) BIND(C, name="vec_capacity_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: res
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

    FUNCTION vec_clone_i8_c(h) RESULT(res) BIND(C, name="vec_clone_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION

    FUNCTION vec_sum_i8_c(h) RESULT(res) BIND(C, name="vec_sum_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_long_long) :: res
    END FUNCTION
    
    FUNCTION vec_sum_partial_i8_c(h, n) RESULT(res) BIND(C, name="vec_sum_partial_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_long_long) :: res
    END FUNCTION

    FUNCTION vec_min_i8_c(h) RESULT(res) BIND(C, name="vec_min_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_long_long) :: res
    END FUNCTION
    FUNCTION vec_min_partial_i8_c(h, n) RESULT(res) BIND(C, name="vec_min_partial_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_long_long) :: res
    END FUNCTION

    FUNCTION vec_max_i8_c(h) RESULT(res) BIND(C, name="vec_max_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_long_long) :: res
    END FUNCTION
    FUNCTION vec_max_partial_i8_c(h, n) RESULT(res) BIND(C, name="vec_max_partial_i8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      INTEGER(c_long_long) :: res
    END FUNCTION


    ! ==================================================================
    ! [R4] Real (32-bit)
    ! ==================================================================
    FUNCTION vec_new_vector_r4_c(n) RESULT(res) BIND(C, name="vec_new_vector_r4")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      TYPE(C_PTR) :: res
    END FUNCTION
    
    FUNCTION vec_new_buffer_r4_c(n, pinned) RESULT(res) BIND(C, name="vec_new_buffer_r4")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      LOGICAL(c_bool), VALUE :: pinned
      TYPE(C_PTR) :: res
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

    SUBROUTINE vec_reserve_r4_c(h, n) BIND(C, name="vec_reserve_r4")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE

    SUBROUTINE vec_copy_from_r4_c(dst, src) BIND(C, name="vec_copy_from_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: dst, src
    END SUBROUTINE

    FUNCTION vec_host_r4_c(h) RESULT(res) BIND(C, name="vec_host_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION

    FUNCTION vec_dev_r4_c(h) RESULT(res) BIND(C, name="vec_dev_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION
    
    FUNCTION vec_size_r4_c(h) RESULT(res) BIND(C, name="vec_size_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: res
    END FUNCTION
    
    FUNCTION vec_capacity_r4_c(h) RESULT(res) BIND(C, name="vec_capacity_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: res
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

    FUNCTION vec_clone_r4_c(h) RESULT(res) BIND(C, name="vec_clone_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION

    FUNCTION vec_sum_r4_c(h) RESULT(res) BIND(C, name="vec_sum_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_float)      :: res
    END FUNCTION
    
    FUNCTION vec_sum_partial_r4_c(h, n) RESULT(res) BIND(C, name="vec_sum_partial_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_float)      :: res
    END FUNCTION

    FUNCTION vec_min_r4_c(h) RESULT(res) BIND(C, name="vec_min_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_float)      :: res
    END FUNCTION
    FUNCTION vec_min_partial_r4_c(h, n) RESULT(res) BIND(C, name="vec_min_partial_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_float)      :: res
    END FUNCTION

    FUNCTION vec_max_r4_c(h) RESULT(res) BIND(C, name="vec_max_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_float)      :: res
    END FUNCTION
    FUNCTION vec_max_partial_r4_c(h, n) RESULT(res) BIND(C, name="vec_max_partial_r4")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_float)      :: res
    END FUNCTION


    ! ==================================================================
    ! [R8] Real (64-bit)
    ! ==================================================================
    FUNCTION vec_new_vector_r8_c(n) RESULT(res) BIND(C, name="vec_new_vector_r8")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      TYPE(C_PTR) :: res
    END FUNCTION
    
    FUNCTION vec_new_buffer_r8_c(n, pinned) RESULT(res) BIND(C, name="vec_new_buffer_r8")
      IMPORT
      INTEGER(c_size_t), VALUE :: n
      LOGICAL(c_bool), VALUE :: pinned
      TYPE(C_PTR) :: res
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

    SUBROUTINE vec_reserve_r8_c(h, n) BIND(C, name="vec_reserve_r8")
      IMPORT
      TYPE(c_ptr), VALUE       :: h
      INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE

    SUBROUTINE vec_copy_from_r8_c(dst, src) BIND(C, name="vec_copy_from_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: dst, src
    END SUBROUTINE

    FUNCTION vec_host_r8_c(h) RESULT(res) BIND(C, name="vec_host_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION

    FUNCTION vec_dev_r8_c(h) RESULT(res) BIND(C, name="vec_dev_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION
    
    FUNCTION vec_size_r8_c(h) RESULT(res) BIND(C, name="vec_size_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: res
    END FUNCTION
    
    FUNCTION vec_capacity_r8_c(h) RESULT(res) BIND(C, name="vec_capacity_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t)  :: res
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

    FUNCTION vec_clone_r8_c(h) RESULT(res) BIND(C, name="vec_clone_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      TYPE(c_ptr)        :: res
    END FUNCTION

    FUNCTION vec_sum_r8_c(h) RESULT(res) BIND(C, name="vec_sum_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_double)     :: res
    END FUNCTION
    
    FUNCTION vec_sum_partial_r8_c(h, n) RESULT(res) BIND(C, name="vec_sum_partial_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_double)     :: res
    END FUNCTION

    FUNCTION vec_min_r8_c(h) RESULT(res) BIND(C, name="vec_min_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_double)     :: res
    END FUNCTION
    FUNCTION vec_min_partial_r8_c(h, n) RESULT(res) BIND(C, name="vec_min_partial_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_double)     :: res
    END FUNCTION

    FUNCTION vec_max_r8_c(h) RESULT(res) BIND(C, name="vec_max_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      REAL(c_double)     :: res
    END FUNCTION
    FUNCTION vec_max_partial_r8_c(h, n) RESULT(res) BIND(C, name="vec_max_partial_r8")
      IMPORT
      TYPE(c_ptr), VALUE :: h
      INTEGER(c_size_t), VALUE :: n
      REAL(c_double)     :: res
    END FUNCTION

    ! --- Sort ---
    SUBROUTINE vec_sort_pairs_i4_c(kin, kbuf, vin, vbuf, n) BIND(C, name="vec_sort_pairs_i4_c")
      IMPORT
      TYPE(C_PTR), VALUE :: kin, kbuf, vin, vbuf
      INTEGER(c_size_t), VALUE :: n
    END SUBROUTINE

  END INTERFACE

  ! ====================================================================
  ! 3. Abstract Base Class
  ! ====================================================================
  TYPE, ABSTRACT :: device_vector_base_t
     TYPE(c_ptr) :: handle = C_NULL_PTR
     TYPE(c_ptr) :: data = C_NULL_PTR  ! Host Mirror Pointer (Managed by C++)
   CONTAINS
     ! -- Lifecycle --
     PROCEDURE(impl_create_vector_base), DEFERRED, PASS :: create_vector
     PROCEDURE(impl_create_buffer_base), DEFERRED, PASS :: create_buffer
     PROCEDURE(impl_free_base),          DEFERRED, PASS :: free
     
     ! -- Memory Ops --
     PROCEDURE(impl_resize_base),        DEFERRED, PASS :: resize
     PROCEDURE(impl_reserve_base),       DEFERRED, PASS :: reserve
     PROCEDURE(impl_copy_from_base),     DEFERRED, PASS :: copy_from
     
     ! -- Transfers --
     PROCEDURE(impl_upload_base),        DEFERRED, PASS :: upload
     PROCEDURE(impl_download_base),      DEFERRED, PASS :: download
     
     ! -- Utils --
     PROCEDURE(impl_fill_zero_base),     DEFERRED, PASS :: fill_zero
     PROCEDURE(impl_get_handle_base),    DEFERRED, PASS :: get_handle
     PROCEDURE(impl_device_ptr_base),    DEFERRED, PASS :: device_ptr
     
     ! -- Properties --
     PROCEDURE(impl_size_base),          DEFERRED, PASS :: size
     PROCEDURE(impl_capacity_base),      DEFERRED, PASS :: capacity
     
     ! -- Internal Helper --
     PROCEDURE(impl_sync_host_ptr_base), DEFERRED, PASS :: sync_host_ptr
  END TYPE device_vector_base_t

  ! Interfaces for Deferred Procedures
  INTERFACE
     SUBROUTINE impl_create_vector_base(this, n)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
       INTEGER(8), INTENT(IN) :: n
     END SUBROUTINE impl_create_vector_base

     SUBROUTINE impl_create_buffer_base(this, n, pinned)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
       INTEGER(8), INTENT(IN) :: n
       LOGICAL, INTENT(IN), OPTIONAL :: pinned
     END SUBROUTINE impl_create_buffer_base

     SUBROUTINE impl_free_base(this)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
     END SUBROUTINE impl_free_base

     SUBROUTINE impl_resize_base(this, n)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
       INTEGER(8), INTENT(IN) :: n
     END SUBROUTINE impl_resize_base

     SUBROUTINE impl_reserve_base(this, n)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
       INTEGER(8), INTENT(IN) :: n
     END SUBROUTINE impl_reserve_base

     SUBROUTINE impl_copy_from_base(this, other)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
       CLASS(device_vector_base_t), INTENT(IN)    :: other
     END SUBROUTINE impl_copy_from_base

     SUBROUTINE impl_upload_base(this)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
     END SUBROUTINE impl_upload_base

     SUBROUTINE impl_download_base(this)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
     END SUBROUTINE impl_download_base

     SUBROUTINE impl_fill_zero_base(this)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
     END SUBROUTINE impl_fill_zero_base

     TYPE(C_PTR) FUNCTION impl_get_handle_base(this)
       USE iso_c_binding
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(IN) :: this
     END FUNCTION impl_get_handle_base

     TYPE(C_PTR) FUNCTION impl_device_ptr_base(this)
       USE iso_c_binding
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(IN) :: this
     END FUNCTION impl_device_ptr_base

     INTEGER(8) FUNCTION impl_size_base(this)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(IN) :: this
     END FUNCTION impl_size_base

     INTEGER(8) FUNCTION impl_capacity_base(this)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(IN) :: this
     END FUNCTION impl_capacity_base

     SUBROUTINE impl_sync_host_ptr_base(this)
       IMPORT :: device_vector_base_t
       CLASS(device_vector_base_t), INTENT(INOUT) :: this
     END SUBROUTINE impl_sync_host_ptr_base
  END INTERFACE

  ! ====================================================================
  ! 4. Concrete Types
  ! ====================================================================

  ! --- I4 ---
  TYPE, EXTENDS(device_vector_base_t) :: device_vector_i4_t
     INTEGER(4), POINTER :: ptr(:) => NULL() ! Accessor for Host Data
   CONTAINS
     PROCEDURE :: create_vector => impl_create_vector_i4
     PROCEDURE :: create_buffer => impl_create_buffer_i4
     PROCEDURE :: free          => impl_free_i4
     PROCEDURE :: resize        => impl_resize_i4
     PROCEDURE :: reserve       => impl_reserve_i4
     PROCEDURE :: copy_from     => impl_copy_from_i4
     PROCEDURE :: upload        => impl_upload_i4
     PROCEDURE :: download      => impl_download_i4
     PROCEDURE :: fill_zero     => impl_fill_zero_i4
     PROCEDURE :: get_handle    => impl_get_handle_i4
     PROCEDURE :: device_ptr    => impl_device_ptr_i4
     PROCEDURE :: size          => impl_size_i4
     PROCEDURE :: capacity      => impl_capacity_i4
     PROCEDURE :: sync_host_ptr => impl_sync_host_ptr_i4
     PROCEDURE :: host_data     => impl_host_data_i4
     PROCEDURE :: sum           => impl_sum_i4
  END TYPE device_vector_i4_t

  ! --- I8 ---
  TYPE, EXTENDS(device_vector_base_t) :: device_vector_i8_t
     INTEGER(8), POINTER :: ptr(:) => NULL()
   CONTAINS
     PROCEDURE :: create_vector => impl_create_vector_i8
     PROCEDURE :: create_buffer => impl_create_buffer_i8
     PROCEDURE :: free          => impl_free_i8
     PROCEDURE :: resize        => impl_resize_i8
     PROCEDURE :: reserve       => impl_reserve_i8
     PROCEDURE :: copy_from     => impl_copy_from_i8
     PROCEDURE :: upload        => impl_upload_i8
     PROCEDURE :: download      => impl_download_i8
     PROCEDURE :: fill_zero     => impl_fill_zero_i8
     PROCEDURE :: get_handle    => impl_get_handle_i8
     PROCEDURE :: device_ptr    => impl_device_ptr_i8
     PROCEDURE :: size          => impl_size_i8
     PROCEDURE :: capacity      => impl_capacity_i8
     PROCEDURE :: sync_host_ptr => impl_sync_host_ptr_i8
     PROCEDURE :: host_data     => impl_host_data_i8
     PROCEDURE :: sum           => impl_sum_i8
  END TYPE device_vector_i8_t

  ! --- R4 ---
  TYPE, EXTENDS(device_vector_base_t) :: device_vector_r4_t
     REAL(4), POINTER :: ptr(:) => NULL()
   CONTAINS
     PROCEDURE :: create_vector => impl_create_vector_r4
     PROCEDURE :: create_buffer => impl_create_buffer_r4
     PROCEDURE :: free          => impl_free_r4
     PROCEDURE :: resize        => impl_resize_r4
     PROCEDURE :: reserve       => impl_reserve_r4
     PROCEDURE :: copy_from     => impl_copy_from_r4
     PROCEDURE :: upload        => impl_upload_r4
     PROCEDURE :: download      => impl_download_r4
     PROCEDURE :: fill_zero     => impl_fill_zero_r4
     PROCEDURE :: get_handle    => impl_get_handle_r4
     PROCEDURE :: device_ptr    => impl_device_ptr_r4
     PROCEDURE :: size          => impl_size_r4
     PROCEDURE :: capacity      => impl_capacity_r4
     PROCEDURE :: sync_host_ptr => impl_sync_host_ptr_r4
     PROCEDURE :: host_data     => impl_host_data_r4
     PROCEDURE :: sum           => impl_sum_r4
  END TYPE device_vector_r4_t

  ! --- R8 ---
  TYPE, EXTENDS(device_vector_base_t) :: device_vector_r8_t
     REAL(8), POINTER :: ptr(:) => NULL()
   CONTAINS
     PROCEDURE :: create_vector => impl_create_vector_r8
     PROCEDURE :: create_buffer => impl_create_buffer_r8
     PROCEDURE :: free          => impl_free_r8
     PROCEDURE :: resize        => impl_resize_r8
     PROCEDURE :: reserve       => impl_reserve_r8
     PROCEDURE :: copy_from     => impl_copy_from_r8
     PROCEDURE :: upload        => impl_upload_r8
     PROCEDURE :: download      => impl_download_r8
     PROCEDURE :: fill_zero     => impl_fill_zero_r8
     PROCEDURE :: get_handle    => impl_get_handle_r8
     PROCEDURE :: device_ptr    => impl_device_ptr_r8
     PROCEDURE :: size          => impl_size_r8
     PROCEDURE :: capacity      => impl_capacity_r8
     PROCEDURE :: sync_host_ptr => impl_sync_host_ptr_r8
     PROCEDURE :: host_data     => impl_host_data_r8
     PROCEDURE :: sum           => impl_sum_r8
  END TYPE device_vector_r8_t

CONTAINS

  ! ====================================================================
  ! IMPLEMENTATIONS: I4
  ! ====================================================================
  SUBROUTINE impl_create_vector_i4(this, n)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    ! Create Compute Vector (Mode 2)
    this%handle = vec_new_vector_i4_c(INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_create_vector_i4

  SUBROUTINE impl_create_buffer_i4(this, n, pinned)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    LOGICAL, INTENT(IN), OPTIONAL :: pinned
    LOGICAL(c_bool) :: is_pinned
    
    ! Default to Pinned (Mode 0) for Buffers
    is_pinned = .TRUE.
    IF (PRESENT(pinned)) THEN
       IF (.NOT. pinned) is_pinned = .FALSE.
    END IF
    
    this%handle = vec_new_buffer_i4_c(INT(n, c_size_t), is_pinned)
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_create_buffer_i4

  SUBROUTINE impl_copy_from_i4(this, other)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    CLASS(device_vector_base_t), INTENT(IN)  :: other
    CALL vec_copy_from_i4_c(this%handle, other%handle)
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_copy_from_i4

  SUBROUTINE impl_free_i4(this)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    IF (C_ASSOCIATED(this%handle)) THEN
       CALL vec_delete_i4_c(this%handle)
       this%handle = C_NULL_PTR
    END IF
    NULLIFY(this%ptr)
  END SUBROUTINE impl_free_i4

  SUBROUTINE impl_resize_i4(this, n)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    CALL vec_resize_i4_c(this%handle, INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_resize_i4

  SUBROUTINE impl_reserve_i4(this, n)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    CALL vec_reserve_i4_c(this%handle, INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_reserve_i4

  SUBROUTINE impl_upload_i4(this)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    CALL vec_upload_i4_c(this%handle)
  END SUBROUTINE impl_upload_i4

  SUBROUTINE impl_download_i4(this)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    CALL vec_download_i4_c(this%handle)
  END SUBROUTINE impl_download_i4

  SUBROUTINE impl_fill_zero_i4(this)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    CALL vec_fill_zero_i4_c(this%handle)
  END SUBROUTINE impl_fill_zero_i4

  TYPE(C_PTR) FUNCTION impl_get_handle_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this
    impl_get_handle_i4 = this%handle
  END FUNCTION impl_get_handle_i4

  TYPE(C_PTR) FUNCTION impl_device_ptr_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this
    impl_device_ptr_i4 = vec_dev_i4_c(this%handle)
  END FUNCTION impl_device_ptr_i4

  INTEGER(8) FUNCTION impl_size_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this
    impl_size_i4 = INT(vec_size_i4_c(this%handle), 8)
  END FUNCTION impl_size_i4

  INTEGER(8) FUNCTION impl_capacity_i4(this)
    CLASS(device_vector_i4_t), INTENT(IN) :: this
    impl_capacity_i4 = INT(vec_capacity_i4_c(this%handle), 8)
  END FUNCTION impl_capacity_i4

  SUBROUTINE impl_sync_host_ptr_i4(this)
    CLASS(device_vector_i4_t), INTENT(INOUT) :: this
    TYPE(C_PTR) :: raw
    INTEGER(8) :: n
    raw = vec_host_i4_c(this%handle)
    n = INT(vec_size_i4_c(this%handle), 8)
    IF (C_ASSOCIATED(raw) .AND. n > 0) THEN
       CALL C_F_POINTER(raw, this%ptr, [n])
    ELSE
       NULLIFY(this%ptr)
    END IF
  END SUBROUTINE impl_sync_host_ptr_i4

  SUBROUTINE impl_host_data_i4(this, p)
     CLASS(device_vector_i4_t), INTENT(IN) :: this
     INTEGER(4), POINTER, INTENT(OUT) :: p(:)
     p => this%ptr
  END SUBROUTINE impl_host_data_i4

  INTEGER(4) FUNCTION impl_sum_i4(this)
     CLASS(device_vector_i4_t), INTENT(INOUT) :: this
     impl_sum_i4 = vec_sum_i4_c(this%handle)
  END FUNCTION impl_sum_i4


  ! ====================================================================
  ! IMPLEMENTATIONS: I8
  ! ====================================================================
  SUBROUTINE impl_create_vector_i8(this, n)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    this%handle = vec_new_vector_i8_c(INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_create_vector_i8

  SUBROUTINE impl_create_buffer_i8(this, n, pinned)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    LOGICAL, INTENT(IN), OPTIONAL :: pinned
    LOGICAL(c_bool) :: is_pinned
    is_pinned = .TRUE.
    IF (PRESENT(pinned)) THEN
       IF (.NOT. pinned) is_pinned = .FALSE.
    END IF
    this%handle = vec_new_buffer_i8_c(INT(n, c_size_t), is_pinned)
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_create_buffer_i8

  SUBROUTINE impl_copy_from_i8(this, other)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    CLASS(device_vector_base_t), INTENT(IN)  :: other
    CALL vec_copy_from_i8_c(this%handle, other%handle)
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_copy_from_i8

  SUBROUTINE impl_free_i8(this)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    IF (C_ASSOCIATED(this%handle)) THEN
       CALL vec_delete_i8_c(this%handle)
       this%handle = C_NULL_PTR
    END IF
    NULLIFY(this%ptr)
  END SUBROUTINE impl_free_i8

  SUBROUTINE impl_resize_i8(this, n)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    CALL vec_resize_i8_c(this%handle, INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_resize_i8

  SUBROUTINE impl_reserve_i8(this, n)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    CALL vec_reserve_i8_c(this%handle, INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_reserve_i8

  SUBROUTINE impl_upload_i8(this)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    CALL vec_upload_i8_c(this%handle)
  END SUBROUTINE impl_upload_i8

  SUBROUTINE impl_download_i8(this)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    CALL vec_download_i8_c(this%handle)
  END SUBROUTINE impl_download_i8

  SUBROUTINE impl_fill_zero_i8(this)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    CALL vec_fill_zero_i8_c(this%handle)
  END SUBROUTINE impl_fill_zero_i8

  TYPE(C_PTR) FUNCTION impl_get_handle_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this
    impl_get_handle_i8 = this%handle
  END FUNCTION impl_get_handle_i8

  TYPE(C_PTR) FUNCTION impl_device_ptr_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this
    impl_device_ptr_i8 = vec_dev_i8_c(this%handle)
  END FUNCTION impl_device_ptr_i8

  INTEGER(8) FUNCTION impl_size_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this
    impl_size_i8 = INT(vec_size_i8_c(this%handle), 8)
  END FUNCTION impl_size_i8

  INTEGER(8) FUNCTION impl_capacity_i8(this)
    CLASS(device_vector_i8_t), INTENT(IN) :: this
    impl_capacity_i8 = INT(vec_capacity_i8_c(this%handle), 8)
  END FUNCTION impl_capacity_i8

  SUBROUTINE impl_sync_host_ptr_i8(this)
    CLASS(device_vector_i8_t), INTENT(INOUT) :: this
    TYPE(C_PTR) :: raw
    INTEGER(8) :: n
    raw = vec_host_i8_c(this%handle)
    n = INT(vec_size_i8_c(this%handle), 8)
    IF (C_ASSOCIATED(raw) .AND. n > 0) THEN
       CALL C_F_POINTER(raw, this%ptr, [n])
    ELSE
       NULLIFY(this%ptr)
    END IF
  END SUBROUTINE impl_sync_host_ptr_i8

  SUBROUTINE impl_host_data_i8(this, p)
     CLASS(device_vector_i8_t), INTENT(IN) :: this
     INTEGER(8), POINTER, INTENT(OUT) :: p(:)
     p => this%ptr
  END SUBROUTINE impl_host_data_i8

  INTEGER(8) FUNCTION impl_sum_i8(this)
     CLASS(device_vector_i8_t), INTENT(INOUT) :: this
     impl_sum_i8 = vec_sum_i8_c(this%handle)
  END FUNCTION impl_sum_i8


  ! ====================================================================
  ! IMPLEMENTATIONS: R4
  ! ====================================================================
  SUBROUTINE impl_create_vector_r4(this, n)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    this%handle = vec_new_vector_r4_c(INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_create_vector_r4

  SUBROUTINE impl_create_buffer_r4(this, n, pinned)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    LOGICAL, INTENT(IN), OPTIONAL :: pinned
    LOGICAL(c_bool) :: is_pinned
    is_pinned = .TRUE.
    IF (PRESENT(pinned)) THEN
       IF (.NOT. pinned) is_pinned = .FALSE.
    END IF
    this%handle = vec_new_buffer_r4_c(INT(n, c_size_t), is_pinned)
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_create_buffer_r4

  SUBROUTINE impl_copy_from_r4(this, other)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    CLASS(device_vector_base_t), INTENT(IN)  :: other
    CALL vec_copy_from_r4_c(this%handle, other%handle)
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_copy_from_r4

  SUBROUTINE impl_free_r4(this)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    IF (C_ASSOCIATED(this%handle)) THEN
       CALL vec_delete_r4_c(this%handle)
       this%handle = C_NULL_PTR
    END IF
    NULLIFY(this%ptr)
  END SUBROUTINE impl_free_r4

  SUBROUTINE impl_resize_r4(this, n)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    CALL vec_resize_r4_c(this%handle, INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_resize_r4

  SUBROUTINE impl_reserve_r4(this, n)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    CALL vec_reserve_r4_c(this%handle, INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_reserve_r4

  SUBROUTINE impl_upload_r4(this)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    CALL vec_upload_r4_c(this%handle)
  END SUBROUTINE impl_upload_r4

  SUBROUTINE impl_download_r4(this)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    CALL vec_download_r4_c(this%handle)
  END SUBROUTINE impl_download_r4

  SUBROUTINE impl_fill_zero_r4(this)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    CALL vec_fill_zero_r4_c(this%handle)
  END SUBROUTINE impl_fill_zero_r4

  TYPE(C_PTR) FUNCTION impl_get_handle_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this
    impl_get_handle_r4 = this%handle
  END FUNCTION impl_get_handle_r4

  TYPE(C_PTR) FUNCTION impl_device_ptr_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this
    impl_device_ptr_r4 = vec_dev_r4_c(this%handle)
  END FUNCTION impl_device_ptr_r4

  INTEGER(8) FUNCTION impl_size_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this
    impl_size_r4 = INT(vec_size_r4_c(this%handle), 8)
  END FUNCTION impl_size_r4

  INTEGER(8) FUNCTION impl_capacity_r4(this)
    CLASS(device_vector_r4_t), INTENT(IN) :: this
    impl_capacity_r4 = INT(vec_capacity_r4_c(this%handle), 8)
  END FUNCTION impl_capacity_r4

  SUBROUTINE impl_sync_host_ptr_r4(this)
    CLASS(device_vector_r4_t), INTENT(INOUT) :: this
    TYPE(C_PTR) :: raw
    INTEGER(8) :: n
    raw = vec_host_r4_c(this%handle)
    n = INT(vec_size_r4_c(this%handle), 8)
    IF (C_ASSOCIATED(raw) .AND. n > 0) THEN
       CALL C_F_POINTER(raw, this%ptr, [n])
    ELSE
       NULLIFY(this%ptr)
    END IF
  END SUBROUTINE impl_sync_host_ptr_r4

  SUBROUTINE impl_host_data_r4(this, p)
     CLASS(device_vector_r4_t), INTENT(IN) :: this
     REAL(4), POINTER, INTENT(OUT) :: p(:)
     p => this%ptr
  END SUBROUTINE impl_host_data_r4

  REAL(4) FUNCTION impl_sum_r4(this)
     CLASS(device_vector_r4_t), INTENT(INOUT) :: this
     impl_sum_r4 = vec_sum_r4_c(this%handle)
  END FUNCTION impl_sum_r4


  ! ====================================================================
  ! IMPLEMENTATIONS: R8
  ! ====================================================================
  SUBROUTINE impl_create_vector_r8(this, n)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    this%handle = vec_new_vector_r8_c(INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_create_vector_r8

  SUBROUTINE impl_create_buffer_r8(this, n, pinned)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    LOGICAL, INTENT(IN), OPTIONAL :: pinned
    LOGICAL(c_bool) :: is_pinned
    is_pinned = .TRUE.
    IF (PRESENT(pinned)) THEN
       IF (.NOT. pinned) is_pinned = .FALSE.
    END IF
    this%handle = vec_new_buffer_r8_c(INT(n, c_size_t), is_pinned)
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_create_buffer_r8

  SUBROUTINE impl_copy_from_r8(this, other)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    CLASS(device_vector_base_t), INTENT(IN)  :: other
    CALL vec_copy_from_r8_c(this%handle, other%handle)
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_copy_from_r8

  SUBROUTINE impl_free_r8(this)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    IF (C_ASSOCIATED(this%handle)) THEN
       CALL vec_delete_r8_c(this%handle)
       this%handle = C_NULL_PTR
    END IF
    NULLIFY(this%ptr)
  END SUBROUTINE impl_free_r8

  SUBROUTINE impl_resize_r8(this, n)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    CALL vec_resize_r8_c(this%handle, INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_resize_r8

  SUBROUTINE impl_reserve_r8(this, n)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    INTEGER(8), INTENT(IN) :: n
    CALL vec_reserve_r8_c(this%handle, INT(n, c_size_t))
    CALL this%sync_host_ptr()
  END SUBROUTINE impl_reserve_r8

  SUBROUTINE impl_upload_r8(this)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    CALL vec_upload_r8_c(this%handle)
  END SUBROUTINE impl_upload_r8

  SUBROUTINE impl_download_r8(this)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    CALL vec_download_r8_c(this%handle)
  END SUBROUTINE impl_download_r8

  SUBROUTINE impl_fill_zero_r8(this)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    CALL vec_fill_zero_r8_c(this%handle)
  END SUBROUTINE impl_fill_zero_r8

  TYPE(C_PTR) FUNCTION impl_get_handle_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this
    impl_get_handle_r8 = this%handle
  END FUNCTION impl_get_handle_r8

  TYPE(C_PTR) FUNCTION impl_device_ptr_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this
    impl_device_ptr_r8 = vec_dev_r8_c(this%handle)
  END FUNCTION impl_device_ptr_r8

  INTEGER(8) FUNCTION impl_size_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this
    impl_size_r8 = INT(vec_size_r8_c(this%handle), 8)
  END FUNCTION impl_size_r8

  INTEGER(8) FUNCTION impl_capacity_r8(this)
    CLASS(device_vector_r8_t), INTENT(IN) :: this
    impl_capacity_r8 = INT(vec_capacity_r8_c(this%handle), 8)
  END FUNCTION impl_capacity_r8

  SUBROUTINE impl_sync_host_ptr_r8(this)
    CLASS(device_vector_r8_t), INTENT(INOUT) :: this
    TYPE(C_PTR) :: raw
    INTEGER(8) :: n
    raw = vec_host_r8_c(this%handle)
    n = INT(vec_size_r8_c(this%handle), 8)
    IF (C_ASSOCIATED(raw) .AND. n > 0) THEN
       CALL C_F_POINTER(raw, this%ptr, [n])
    ELSE
       NULLIFY(this%ptr)
    END IF
  END SUBROUTINE impl_sync_host_ptr_r8

  SUBROUTINE impl_host_data_r8(this, p)
     CLASS(device_vector_r8_t), INTENT(IN) :: this
     REAL(8), POINTER, INTENT(OUT) :: p(:)
     p => this%ptr
  END SUBROUTINE impl_host_data_r8

  REAL(8) FUNCTION impl_sum_r8(this)
     CLASS(device_vector_r8_t), INTENT(INOUT) :: this
     impl_sum_r8 = vec_sum_r8_c(this%handle)
  END FUNCTION impl_sum_r8

  ! ====================================================================
  ! Procedural sort wrapper
  ! ====================================================================
  SUBROUTINE vec_sort_i4(keys_ptr, keys_buf_ptr, vals_ptr, vals_buf_ptr, n_opt)
    TYPE(c_ptr), INTENT(IN) :: keys_ptr, keys_buf_ptr, vals_ptr, vals_buf_ptr
    INTEGER(8), INTENT(IN), OPTIONAL :: n_opt
    INTEGER(c_size_t) :: n
    
    IF (PRESENT(n_opt)) THEN
        n = INT(n_opt, c_size_t)
    ELSE
        n = vec_size_i4_c(keys_ptr)
    END IF
    
    CALL vec_sort_pairs_i4_c(keys_ptr, keys_buf_ptr, vals_ptr, vals_buf_ptr, n)
  END SUBROUTINE vec_sort_i4

END MODULE Device_Vector