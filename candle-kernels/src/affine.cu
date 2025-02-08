#include "cuda_utils.cuh"
#include<stdint.h>

#define AFFINE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const TYPENAME mul, \
    const TYPENAME add \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = inp ? inp[i] : out[i]; \
            out[i] = x * mul + add; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp ? inp[strided_i] : out[i]; \
            out[i] = x * mul + add; \
        } \
    } \
} \

#if __CUDA_ARCH__ >= 800 || (__CUDA_ARCH__ >= 530 && __CUDA_ARCH__ < 800)
AFFINE_OP(__nv_bfloat16, affine_bf16)
#endif

#if __CUDA_ARCH__ >= 530
AFFINE_OP(__half, affine_f16)
#endif

AFFINE_OP(float, affine_f32)
AFFINE_OP(double, affine_f64)
AFFINE_OP(uint8_t, affine_u8)
AFFINE_OP(uint32_t, affine_u32)
AFFINE_OP(int64_t, affine_i64)
