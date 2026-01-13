program test_openacc_multivector
  use Device_Vector
  use iso_c_binding
  implicit none

  type(device_vector_i4_t) :: va, vb, v_akeyb, v_bvalb
  integer :: i, n
  integer, pointer :: a(:), b(:)
  real :: rand_val

  n = 10
  call device_env_init(0, 1)

  ! 1. 建立所有需要的物件 (包含排序緩衝區)
  call va%create_buffer(int(n,8))
  call vb%create_buffer(int(n,8))
  call v_akeyb%create_buffer(int(n,8))
  call v_bvalb%create_buffer(int(n,8))

  ! 2. 隨機生成數據：a 是 Key (倒序), b 是隨機值
  do i = 1, n
     va%ptr(i) = n - i + 1  ! 10, 9, 8... 1
     call random_number(rand_val)
     vb%ptr(i) = int(rand_val * 100)
  end do

  print *, "--- 排序前 (Host Side) ---"
  print *, "A (Key): ", va%ptr
  print *, "B (隨機):", vb%ptr

  ! 3. 上傳到 GPU 並進行 OpenACC 映射
  call va%upload()
  call vb%upload()
  call va%acc_map(a)
  call vb%acc_map(b)

  ! 4. GPU 計算：先把 b 加上 1000
  !$acc parallel loop present(a, b)
  do i = 1, n
      b(i) = b(i) + 1000
  end do
  !$acc end parallel loop
  call device_synchronize()

  ! 5. 必須先解除 OpenACC 映射，才能叫 C++/Thrust 排序 (避免內存衝突)
  call va%acc_unmap()
  call vb%acc_unmap()

  print *, "--- 執行 GPU 排序 (A 是 Key, B 是 Value) ---"
  ! 這裡我們傳入 handles 給 C++ 端的排序函數
  call vec_sort_i4(va%get_handle(), v_akeyb%get_handle(), &
                   vb%get_handle(), v_bvalb%get_handle())

  ! 6. 下載排序後的結果
  call va%download()
  call vb%download()

  print *, "--- 排序後 (Host Side) ---"
  print *, "A (Key 應該變成 1..10): ", va%ptr
  print *, "B (隨機值應該跟著 A 排列):", vb%ptr

  ! 7. 驗證
  if (va%ptr(1) == 1 .and. va%ptr(10) == 10) then
     print *, "✅ PASS: 隨機陣列 B 根據 A 排列成功！"
  else
     print *, "❌ FAIL: 排序邏輯錯誤。"
  end if

  ! 8. 清理
  call va%free(); call vb%free()
  call v_akeyb%free(); call v_bvalb%free()
  call device_env_finalize()

end program test_openacc_multivector