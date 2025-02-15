#include "cuda_utils.cuh"
#include<stdint.h>

#define WHERE_OP(TYPENAME, ID_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const ID_TYPENAME *ids, \
    const TYPENAME *t, \
    const TYPENAME *f, \
    TYPENAME *out \
) {  \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    const size_t *strides_t = info + 2*num_dims; \
    const size_t *strides_f = info + 3*num_dims; \
    if (is_contiguous(num_dims, dims, strides) \
        && is_contiguous(num_dims, dims, strides_f) \
        && is_contiguous(num_dims, dims, strides_t)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = ids[i] ? t[i] : f[i]; \
        } \
    } \
    else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            unsigned strided_i_t = get_strided_index(i, num_dims, dims, strides_t); \
            unsigned strided_i_f = get_strided_index(i, num_dims, dims, strides_f); \
            out[i] = ids[strided_i] ? t[strided_i_t] : f[strided_i_f]; \
        } \
    } \
} \

#if __CUDA_ARCH__ >= 800
WHERE_OP(__nv_bfloat16, int64_t, where_i64_bf16)
WHERE_OP(__nv_bfloat16, uint32_t, where_u32_bf16)
WHERE_OP(__nv_bfloat16, uint8_t, where_u8_bf16)

WHERE_OP(__nv_fp8_e4m3, int16_t, where_i16_fp8_e4m3)
WHERE_OP(__nv_fp8_e4m3, int32_t, where_i32_fp8_e4m3)
WHERE_OP(__nv_fp8_e4m3, int64_t, where_i64_fp8_e4m3)
WHERE_OP(__nv_fp8_e4m3, uint32_t, where_u32_fp8_e4m3)
WHERE_OP(__nv_fp8_e4m3, uint8_t, where_u8_fp8_e4m3)
#endif

#if __CUDA_ARCH__ >= 530
WHERE_OP(__half, int64_t, where_i64_f16)
WHERE_OP(__half, uint32_t, where_u32_f16)
WHERE_OP(__half, uint8_t, where_u8_f16)
#endif

WHERE_OP(float, int64_t, where_i64_f32)
WHERE_OP(double, int64_t, where_i64_f64)
WHERE_OP(uint8_t, int64_t, where_i64_u8)
WHERE_OP(uint32_t, int64_t, where_i64_u32)
WHERE_OP(int64_t, int64_t, where_i64_i64)

WHERE_OP(float, uint32_t, where_u32_f32)
WHERE_OP(double, uint32_t, where_u32_f64)
WHERE_OP(uint8_t, uint32_t, where_u32_u8)
WHERE_OP(uint32_t, uint32_t, where_u32_u32)
WHERE_OP(int64_t, uint32_t, where_u32_i64)

WHERE_OP(float, uint8_t, where_u8_f32)
WHERE_OP(double, uint8_t, where_u8_f64)
WHERE_OP(uint8_t, uint8_t, where_u8_u8)
WHERE_OP(uint32_t, uint8_t, where_u8_u32)
WHERE_OP(int64_t, uint8_t, where_u8_i64)
