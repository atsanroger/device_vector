# lib_accel/CUDA.mk
NVCC = nvcc
CUDA_ARCH = -gencode=arch=compute_80,code=sm_80 \
            -gencode=arch=compute_90,code=sm_90
NVCCFLAGS = -O3 $(CUDA_ARCH) -Xcompiler -fPIC

CUDA_SRCS = $(wildcard lib_accel/*.cu)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
CUDA_FORTRAN_OBJ = lib_accel/Device_Interface.o

FFLAGS     += -Mnoinfo -Ilib_accel
EXTRA_OBJS += $(CUDA_OBJS) $(CUDA_FORTRAN_OBJ)

lib_accel/Device_Interface.o: lib_accel/Device_Interface.F90
	$(FC) $(FFLAGS) -c $< -o $@

lib_accel/%.o: lib_accel/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

lib_accel/%.o: lib_accel/%.F90
	$(FC) $(FFLAGS) -c $< -o $@