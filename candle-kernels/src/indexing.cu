// WARNING: THIS IS ONLY VALID ASSUMING THAT inp IS CONTIGUOUS!
// TODO: proper error reporting when ids are larger than v_size.
#include "cuda_utils.cuh"
#include<stdint.h>

template<typename T, typename I>
__device__ void index_select(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    bool b = is_contiguous(num_dims, dims, strides);
    for (unsigned int dst_i = blockIdx.x * blockDim.x + threadIdx.x; dst_i < numel; dst_i += blockDim.x * gridDim.x) {
          unsigned int left_i = dst_i / (ids_dim_size * right_size);
          unsigned int id_i = dst_i / right_size % ids_dim_size;
          unsigned int right_i = dst_i % right_size;
          unsigned int src_i = left_i * (src_dim_size * right_size) + ids[id_i] * right_size + right_i;
          unsigned strided_i = b ? src_i : get_strided_index(src_i, num_dims, dims, strides);
          out[dst_i] = inp[strided_i];
    }
}

#define IS_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t ids_dim_size, \
    const size_t right_size \
) { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size); } \

template<typename T, typename I>
__device__ void gather(
    const size_t numel,
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        size_t post = i % right_size;
        size_t idx = ids[i];
        size_t pre = i / (right_size * ids_dim_size);
        size_t src_i = (pre * src_dim_size + idx) * right_size + post;
        out[i] = inp[src_i];
    }
}

#define GATHER_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t ids_dim_size, \
    const size_t right_size \
) { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size); } \

template<typename T, typename I>
__device__ void index_add(
    const I *ids,
    const size_t ids_dim_size,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size
) {
      const size_t numel = left_size * right_size;
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
          const size_t pre = i / right_size;
          const size_t post = i % right_size;
          for (unsigned int j = 0; j < ids_dim_size; ++j) {
              const size_t idx = ids[j];
              const size_t src_i = (pre * ids_dim_size + j) * right_size + post;
              const size_t dst_i = (pre * dst_dim_size + idx) * right_size + post;
              out[dst_i] += inp[src_i];
          }
      }
}

#if __CUDA_ARCH__ >= 800
#define F8E4M3_TO_FLOAT(x) __half2float(__nv_cvt_fp8_to_halfraw(x.__x, __NV_E4M3))

template<typename I>
__device__ void scatter_add_f8(
    const I *ids,
    const __nv_fp8_e4m3 *inp,
    __nv_fp8_e4m3 *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size
) {
      const size_t numel = left_size * right_size;
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
          const size_t pre = i / right_size;
          const size_t post = i % right_size;
          for (unsigned int j = 0; j < src_dim_size; ++j) {
              const size_t src_i = (pre * src_dim_size + j) * right_size + post;
              const size_t idx = ids[src_i];
              const size_t dst_i = (pre * dst_dim_size + idx) * right_size + post;
              out[dst_i] = __nv_fp8_e4m3(F8E4M3_TO_FLOAT(out[dst_i]) + F8E4M3_TO_FLOAT(inp[src_i]));
          }
      }
}

template<typename I>
__device__ void index_add_f8(
    const I *ids,
    const size_t ids_dim_size,
    const __nv_fp8_e4m3 *inp,
    __nv_fp8_e4m3 *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size
) {
      const size_t numel = left_size * right_size;
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
          const size_t pre = i / right_size;
          const size_t post = i % right_size;
          for (unsigned int j = 0; j < ids_dim_size; ++j) {
              const size_t idx = ids[j];
              const size_t src_i = (pre * ids_dim_size + j) * right_size + post;
              const size_t dst_i = (pre * dst_dim_size + idx) * right_size + post;
              out[dst_i] = __nv_fp8_e4m3(F8E4M3_TO_FLOAT(out[dst_i]) + F8E4M3_TO_FLOAT(inp[src_i]));
          }
      }
}
#endif

#define IA_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const INDEX_TYPENAME *ids, \
    const size_t ids_dim_size, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t dst_dim_size, \
    const size_t right_size \
) { index_add(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size); } \

#define IA_OP_F8(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const INDEX_TYPENAME *ids, \
    const size_t ids_dim_size, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t dst_dim_size, \
    const size_t right_size \
) { index_add_f8(ids, ids_dim_size, inp, out, left_size, src_dim_size, dst_dim_size, right_size); } \

template<typename T, typename I>
__device__ void scatter_add(
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t dst_dim_size,
    const size_t right_size
) {
      const size_t numel = left_size * right_size;
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
          const size_t pre = i / right_size;
          const size_t post = i % right_size;
          for (unsigned int j = 0; j < src_dim_size; ++j) {
              const size_t src_i = (pre * src_dim_size + j) * right_size + post;
              const size_t idx = ids[src_i];
              const size_t dst_i = (pre * dst_dim_size + idx) * right_size + post;
              out[dst_i] += inp[src_i];
          }
      }
}

#define SA_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t dst_dim_size, \
    const size_t right_size \
) { scatter_add(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); } \

#define SA_OP_F8(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t dst_dim_size, \
    const size_t right_size \
) { scatter_add_f8(ids, inp, out, left_size, src_dim_size, dst_dim_size, right_size); } \


#if __CUDA_ARCH__ >= 800
IS_OP(__nv_bfloat16, int64_t, is_i64_bf16)
IS_OP(__nv_bfloat16, uint32_t, is_u32_bf16)
IS_OP(__nv_bfloat16, uint8_t, is_u8_bf16)
GATHER_OP(__nv_bfloat16, int64_t, gather_i64_bf16)
GATHER_OP(__nv_bfloat16, uint32_t, gather_u32_bf16)
GATHER_OP(__nv_bfloat16, uint8_t, gather_u8_bf16)
IA_OP(__nv_bfloat16, int64_t, ia_i64_bf16)
IA_OP(__nv_bfloat16, uint32_t, ia_u32_bf16)
IA_OP(__nv_bfloat16, uint8_t, ia_u8_bf16)
SA_OP(__nv_bfloat16, int64_t, sa_i64_bf16)
SA_OP(__nv_bfloat16, uint32_t, sa_u32_bf16)
SA_OP(__nv_bfloat16, uint8_t, sa_u8_bf16)

IS_OP(__nv_fp8_e4m3, int16_t, is_i16_f8_e4m3)
IS_OP(__nv_fp8_e4m3, int32_t, is_i32_f8_e4m3)
IS_OP(__nv_fp8_e4m3, int64_t, is_i64_f8_e4m3)
IS_OP(__nv_fp8_e4m3, uint32_t, is_u32_f8_e4m3)
IS_OP(__nv_fp8_e4m3, uint8_t, is_u8_f8_e4m3)
GATHER_OP(__nv_fp8_e4m3, int16_t, gather_i16_f8_e4m3)
GATHER_OP(__nv_fp8_e4m3, int32_t, gather_i32_f8_e4m3)
GATHER_OP(__nv_fp8_e4m3, int64_t, gather_i64_f8_e4m3)
GATHER_OP(__nv_fp8_e4m3, uint32_t, gather_u32_f8_e4m3)
GATHER_OP(__nv_fp8_e4m3, uint8_t, gather_u8_f8_e4m3)
IA_OP_F8(__nv_fp8_e4m3, int16_t, ia_i16_f8_e4m3)
IA_OP_F8(__nv_fp8_e4m3, int32_t, ia_i32_f8_e4m3)
IA_OP_F8(__nv_fp8_e4m3, int64_t, ia_i64_f8_e4m3)
IA_OP_F8(__nv_fp8_e4m3, uint32_t, ia_u32_f8_e4m3)
IA_OP_F8(__nv_fp8_e4m3, uint8_t, ia_u8_f8_e4m3)
SA_OP_F8(__nv_fp8_e4m3, int16_t, sa_i16_f8_e4m3)
SA_OP_F8(__nv_fp8_e4m3, int32_t, sa_i32_f8_e4m3)
SA_OP_F8(__nv_fp8_e4m3, int64_t, sa_i64_f8_e4m3)
SA_OP_F8(__nv_fp8_e4m3, uint32_t, sa_u32_f8_e4m3)
SA_OP_F8(__nv_fp8_e4m3, uint8_t, sa_u8_f8_e4m3)
#endif

#if __CUDA_ARCH__ >= 530
IS_OP(__half, int64_t, is_i64_f16)
IS_OP(__half, uint32_t, is_u32_f16)
IS_OP(__half, uint8_t, is_u8_f16)
GATHER_OP(__half, int64_t, gather_i64_f16)
GATHER_OP(__half, uint32_t, gather_u32_f16)
GATHER_OP(__half, uint8_t, gather_u8_f16)
IA_OP(__half, int64_t, ia_i64_f16)
IA_OP(__half, uint32_t, ia_u32_f16)
IA_OP(__half, uint8_t, ia_u8_f16)
SA_OP(__half, int64_t, sa_i64_f16)
SA_OP(__half, uint32_t, sa_u32_f16)
SA_OP(__half, uint8_t, sa_u8_f16)
#endif

IS_OP(float, int64_t, is_i64_f32)
IS_OP(double, int64_t, is_i64_f64)
IS_OP(uint8_t, int64_t, is_i64_u8)
IS_OP(uint32_t, int64_t, is_i64_u32)
IS_OP(int64_t, int64_t, is_i64_i64)

IS_OP(float, uint32_t, is_u32_f32)
IS_OP(double, uint32_t, is_u32_f64)
IS_OP(uint8_t, uint32_t, is_u32_u8)
IS_OP(int64_t, uint32_t, is_u32_i64)
IS_OP(uint32_t, uint32_t, is_u32_u32)

IS_OP(float, uint8_t, is_u8_f32)
IS_OP(double, uint8_t, is_u8_f64)
IS_OP(uint8_t, uint8_t, is_u8_u8)
IS_OP(uint32_t, uint8_t, is_u8_u32)
IS_OP(int64_t, uint8_t, is_u8_i64)

GATHER_OP(float, int64_t, gather_i64_f32)
GATHER_OP(double, int64_t, gather_i64_f64)
GATHER_OP(uint8_t, int64_t, gather_i64_u8)
GATHER_OP(uint32_t, int64_t, gather_i64_u32)
GATHER_OP(int64_t, int64_t, gather_i64_i64)

GATHER_OP(float, uint32_t, gather_u32_f32)
GATHER_OP(double, uint32_t, gather_u32_f64)
GATHER_OP(uint8_t, uint32_t, gather_u32_u8)
GATHER_OP(int64_t, uint32_t, gather_u32_i64)
GATHER_OP(uint32_t, uint32_t, gather_u32_u32)

GATHER_OP(float, uint8_t, gather_u8_f32)
GATHER_OP(double, uint8_t, gather_u8_f64)
GATHER_OP(uint8_t, uint8_t, gather_u8_u8)
GATHER_OP(uint32_t, uint8_t, gather_u8_u32)
GATHER_OP(int64_t, uint8_t, gather_u8_i64)

IA_OP(float, int64_t, ia_i64_f32)
IA_OP(double, int64_t, ia_i64_f64)
IA_OP(uint8_t, int64_t, ia_i64_u8)
IA_OP(int64_t, int64_t, ia_i64_i64)
IA_OP(uint32_t, int64_t, ia_i64_u32)

IA_OP(float, uint32_t, ia_u32_f32)
IA_OP(double, uint32_t, ia_u32_f64)
IA_OP(uint8_t, uint32_t, ia_u32_u8)
IA_OP(int64_t, uint32_t, ia_u32_i64)
IA_OP(uint32_t, uint32_t, ia_u32_u32)

IA_OP(float, uint8_t, ia_u8_f32)
IA_OP(double, uint8_t, ia_u8_f64)
IA_OP(uint8_t, uint8_t, ia_u8_u8)
IA_OP(uint32_t, uint8_t, ia_u8_u32)
IA_OP(int64_t, uint8_t, ia_u8_i64)

SA_OP(float, int64_t, sa_i64_f32)
SA_OP(double, int64_t, sa_i64_f64)
SA_OP(uint8_t, int64_t, sa_i64_u8)
SA_OP(int64_t, int64_t, sa_i64_i64)
SA_OP(uint32_t, int64_t, sa_i64_u32)

SA_OP(float, uint32_t, sa_u32_f32)
SA_OP(double, uint32_t, sa_u32_f64)
SA_OP(uint8_t, uint32_t, sa_u32_u8)
SA_OP(int64_t, uint32_t, sa_u32_i64)
SA_OP(uint32_t, uint32_t, sa_u32_u32)

SA_OP(float, uint8_t, sa_u8_f32)
SA_OP(double, uint8_t, sa_u8_f64)
SA_OP(uint8_t, uint8_t, sa_u8_u8)
SA_OP(uint32_t, uint8_t, sa_u8_u32)
SA_OP(int64_t, uint8_t, sa_u8_i64)
