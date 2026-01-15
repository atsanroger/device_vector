SUBROUTINE pack_grid_masks
    INTEGER :: i, j, k
    INTEGER(1) :: val

    ! 配置空間，範圍涵蓋計算域與 Halo
    IF (ALLOCATED(mask_packed)) DEALLOCATE(mask_packed)
    ALLOCATE(mask_packed(-1:isize+2, -1:jsize+2, -1:ksize+2))
    mask_packed = 0_1

    !$OMP PARALLEL DO COLLAPSE(3) PRIVATE(val)
    DO k = 1, ksize
       DO j = 1, jsize
          DO i = 1, isize
             ! 如果 mask3d 為 0 (陸地)，該格點完全無效
             IF (mask3d(i, j, k) == 0) THEN
                mask_packed(i, j, k) = 0_1
                CYCLE
             END IF

             val = 0_1
             
             ! --- Bit 7: Active (mask3d == 1) ---
             ! 這是最基礎的門檻，代表「這是一個可以計算的水體格點」
             val = IBSET(val, 7)

             ! --- Bit 6: Surface (k == surface_k) ---
             ! 觸發海面速度外推邏輯 [cite: 524]
             IF (surface_flag(i, j) .AND. k == surface_k(i, j)) val = IBSET(val, 6)

             ! --- Bit 0-5: Faces (umask, vmask, wmask) [cite: 71-76] ---
             ! 處理 cross_x/y/z 的邊界碰撞 [cite: 643-651]
             IF (umask(i, j, k)   == 1) val = IBSET(val, 0) ! +X
             IF (umask(i-1, j, k) == 1) val = IBSET(val, 1) ! -X
             IF (vmask(i, j, k)   == 1) val = IBSET(val, 2) ! +Y
             IF (vmask(i, j-1, k) == 1) val = IBSET(val, 3) ! -Y
             IF (wmask(i, j, k)   == 1) val = IBSET(val, 4) ! +Z
             IF (wmask(i, j, k-1) == 1) val = IBSET(val, 5) ! -Z
             
             mask_packed(i, j, k) = val
          END DO
       END DO
    END DO

    !$acc update device(mask_packed)
  END SUBROUTINE pack_grid_masks

END MODULE particles_acc

SUBROUTINE finalize_pack_grid
    ! 檢查陣列是否已配置 [cite: 132, 410]
    IF (ALLOCATED(mask_packed)) THEN
       
       ! 1. 告知 GPU 刪除此陣列空間
       !$acc exit data delete(mask_packed)
       
       ! 2. 釋放 CPU 端的系統記憶體 [cite: 132, 360]
       DEALLOCATE(mask_packed)
       
    END IF
  END SUBROUTINE finalize_pack_grid