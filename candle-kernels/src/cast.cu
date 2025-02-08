#include "cuda_utils.cuh"
#include<stdint.h>

template <typename S, typename T>
__device__ void cast_(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const S *inp,
    T *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = inp[i];
        }
    }
    else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = inp[strided_i];
        }
    }
}

#define F8E4M3_TO_FLOAT(x) __half2float(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3))

template <typename T>
__device__ void cast_fp8_(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const __nv_fp8_e4m3 *inp,
    T *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = F8E4M3_TO_FLOAT(inp[i]);
        }
    }
    else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = F8E4M3_TO_FLOAT(inp[strided_i]);
        }
    }
}
template <typename S>
__device__ void cast_fp8_into_(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const S *inp,
    __nv_fp8_e4m3 *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = __nv_fp8_e4m3((float)inp[i]);
        }
    }
    else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = __nv_fp8_e4m3((float)inp[strided_i]);
        }
    }
}

template <typename S, typename T, typename I>
__device__ void cast_through(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const S *inp,
    T *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            out[i] = static_cast<T>(static_cast<I>(inp[i]));
        }
    }
    else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            out[i] = static_cast<T>(static_cast<I>(inp[strided_i]));
        }
    }
}


#define CAST_OP(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    cast_<SRC_TYPENAME, DST_TYPENAME>(numel, num_dims, info, inp, out); \
} \


#define CAST_OP_FP8(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    cast_fp8_<DST_TYPENAME>(numel, num_dims, info, inp, out); \
} \


#define CAST_OP_FP8_INTO(SRC_TYPENAME, DST_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    cast_fp8_into_<SRC_TYPENAME>(numel, num_dims, info, inp, out); \
} \

#define CAST_THROUGH_OP(SRC_TYPENAME, DST_TYPENAME, INT_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const SRC_TYPENAME *inp, \
    DST_TYPENAME *out \
) { \
    cast_through<SRC_TYPENAME, DST_TYPENAME, INT_TYPENAME>(numel, num_dims, info, inp, out); \
} \

#if __CUDA_ARCH__ >= 800
CAST_OP(__nv_bfloat16, __nv_bfloat16, cast_bf16_bf16)
CAST_OP(__nv_fp8_e4m3, __nv_fp8_e4m3, cast_f8_e4m3_f8_e4m3)

CAST_OP(__nv_bfloat16, uint32_t, cast_bf16_u32)
CAST_OP(__nv_bfloat16, float,    cast_bf16_f32)
CAST_OP(__nv_bfloat16, double,   cast_bf16_f64)
CAST_OP(uint8_t, __nv_bfloat16, cast_u8_bf16)
CAST_OP(uint32_t, __nv_bfloat16, cast_u32_bf16)
CAST_OP(float,    __nv_bfloat16, cast_f32_bf16)
CAST_OP(double,   __nv_bfloat16, cast_f64_bf16)
CAST_THROUGH_OP(__nv_bfloat16, uint8_t, float, cast_bf16_u8)
CAST_THROUGH_OP(__nv_bfloat16, __half,   float, cast_bf16_f16)
CAST_THROUGH_OP(__half,   __nv_bfloat16, float, cast_f16_bf16)

CAST_OP_FP8(__nv_fp8_e4m3, float,    cast_f8_e4m3_f32)
CAST_OP_FP8_INTO(float,    __nv_fp8_e4m3, cast_f32_f8_e4m3)
CAST_OP_FP8(__nv_fp8_e4m3, uint8_t, cast_f8_e4m3_u8)
CAST_OP_FP8(__nv_fp8_e4m3, __half, cast_f8_e4m3_f16)
CAST_OP_FP8(__nv_fp8_e4m3, double,  cast_f8_e4m3_f64)
CAST_OP_FP8_INTO(__half,   __nv_fp8_e4m3, cast_f16_f8_e4m3)
CAST_OP_FP8_INTO(double,   __nv_fp8_e4m3, cast_f64_f8_e4m3)
CAST_OP_FP8_INTO(uint8_t,   __nv_fp8_e4m3, cast_u8_f8_e4m3)
CAST_OP_FP8_INTO(int32_t,   __nv_fp8_e4m3, cast_i32_f8_e4m3)
CAST_OP_FP8(__nv_fp8_e4m3, int32_t, cast_f8_e4m3_i32)
CAST_OP_FP8(__nv_fp8_e4m3, __nv_bfloat16, cast_f8_e4m3_bf16)
CAST_OP_FP8_INTO(__nv_bfloat16, __nv_fp8_e4m3, cast_bf16_f8_e4m3)
#else
#include <cuda.h>
#if CUDA_VERSION >= 11000
CAST_OP(__nv_bfloat16, float,    cast_bf16_f32)
CAST_OP(float,    __nv_bfloat16, cast_f32_bf16)
CAST_THROUGH_OP(__nv_bfloat16, uint8_t, float, cast_bf16_u8)
CAST_THROUGH_OP(__nv_bfloat16, __half,  float, cast_bf16_f16)
CAST_THROUGH_OP(__nv_bfloat16, double,  float, cast_bf16_f64)
CAST_THROUGH_OP(__half,   __nv_bfloat16, float, cast_f16_bf16)
CAST_THROUGH_OP(double,   __nv_bfloat16, float, cast_f64_bf16)
CAST_THROUGH_OP(uint8_t,   __nv_bfloat16, float, cast_u8_bf16)
CAST_THROUGH_OP(__nv_bfloat16, __nv_fp8_e4m3, float, cast_bf16_f8_e4m3)
#endif
#endif

#if __CUDA_ARCH__ >= 530
CAST_OP(__half, __half, cast_f16_f16)

CAST_THROUGH_OP(__half, uint8_t,  float, cast_f16_u8)
CAST_OP(__half, uint32_t, cast_f16_u32)
CAST_OP(__half, float,    cast_f16_f32)
CAST_OP(__half, double,   cast_f16_f64)
CAST_OP(uint8_t,  __half, cast_u8_f16 )
CAST_OP(uint32_t, __half, cast_u32_f16)
CAST_OP(float,    __half, cast_f32_f16)
CAST_OP(double,   __half, cast_f64_f16)
#endif

CAST_OP(uint32_t, uint32_t, cast_u32_u32)
CAST_OP(uint32_t, uint8_t,  cast_u32_u8 )
CAST_OP(uint32_t, int64_t,  cast_u32_i64 )
CAST_OP(uint32_t, float,    cast_u32_f32)
CAST_OP(uint32_t, double,   cast_u32_f64)

CAST_OP(uint8_t, uint32_t, cast_u8_u32)
CAST_OP(uint8_t, uint8_t,  cast_u8_u8 )
CAST_OP(uint8_t, int64_t,  cast_u8_i64 )
CAST_OP(uint8_t, float,    cast_u8_f32)
CAST_OP(uint8_t, double,   cast_u8_f64)

CAST_OP(int64_t, uint32_t, cast_i64_u32)
CAST_OP(int64_t, uint8_t,  cast_i64_u8 )
CAST_OP(int64_t, int64_t,  cast_i64_i64 )
CAST_OP(int64_t, float,    cast_i64_f32)
CAST_OP(int64_t, double,   cast_i64_f64)

CAST_OP(float, uint8_t,  cast_f32_u8 )
CAST_OP(float, uint32_t, cast_f32_u32)
CAST_OP(float, int64_t,  cast_f32_i64 )
CAST_OP(float, float,    cast_f32_f32)
CAST_OP(float, double,   cast_f32_f64)

CAST_OP(double, uint8_t,  cast_f64_u8 )
CAST_OP(double, uint32_t, cast_f64_u32)
CAST_OP(double, int64_t,  cast_f64_i64 )
CAST_OP(double, float,    cast_f64_f32)
CAST_OP(double, double,   cast_f64_f64)
