#include "cuda_utils.cuh"
#include<stdint.h>

#define AFFINE_OP(TYPENAME, FN_NAME, AFFINE) \
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
            out[i] = AFFINE; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp ? inp[strided_i] : out[i]; \
            out[i] = AFFINE; \
        } \
    } \
} \

#if __CUDA_ARCH__ >= 800
AFFINE_OP(__nv_bfloat16, affine_bf16, x * mul + add)

#define F8E4M3_TO_FLOAT(x) __half2float(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3))

AFFINE_OP(__nv_fp8_e4m3, affine_f8_e4m3, __nv_fp8_e4m3(F8E4M3_TO_FLOAT(x) * F8E4M3_TO_FLOAT(mul) + F8E4M3_TO_FLOAT(add)))
#endif

#if __CUDA_ARCH__ >= 530
AFFINE_OP(__half, affine_f16, x * mul + add)
#endif

AFFINE_OP(float, affine_f32, x * mul + add)
AFFINE_OP(double, affine_f64, x * mul + add)
AFFINE_OP(uint8_t, affine_u8, x * mul + add)
AFFINE_OP(uint32_t, affine_u32, x * mul + add)
AFFINE_OP(int16_t, affine_i16, x * mul + add)
AFFINE_OP(int32_t, affine_i32, x * mul + add)
AFFINE_OP(int64_t, affine_i64, x * mul + add)
