#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

	int vec_acc_is_mapped_i4(void *h);
	int vec_acc_is_mapped_i8(void *h);
	int vec_acc_is_mapped_r4(void *h);
	int vec_acc_is_mapped_r8(void *h);

	void vec_acc_map_i4(void *h);
	void vec_acc_map_i8(void *h);
	void vec_acc_map_r4(void *h);
	void vec_acc_map_r8(void *h);

	void vec_acc_unmap_i4(void *h);
	void vec_acc_unmap_i8(void *h);
	void vec_acc_unmap_r4(void *h);
	void vec_acc_unmap_r8(void *h);

#ifdef __cplusplus
}
#endif
