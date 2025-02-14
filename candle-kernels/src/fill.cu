#include<stdint.h>
#include "cuda_fp16.h"

template<typename T>
__device__ void fill_with(T *buf, T value, const size_t numel) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        buf[i] = value;
    }
}
extern "C" __global__ void fill_u8(uint8_t *buf, uint8_t value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_u32(uint32_t *buf, uint32_t value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_i64(int64_t *buf, int64_t value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_f32(float *buf, float value, const size_t numel) { fill_with(buf, value, numel); }
extern "C" __global__ void fill_f64(double *buf, double value, const size_t numel) { fill_with(buf, value, numel); }

template<typename T>
__device__ void copy2d(const T *src, T *dst, uint32_t d1, uint32_t d2, uint32_t src_s, uint32_t dst_s) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d1 * d2) {
    return;
  }
  uint32_t idx1 = idx / d2;
  uint32_t idx2 = idx - d2 * idx1;
  dst[idx1 * dst_s + idx2] = src[idx1 * src_s + idx2];
}

#define COPY2D_OP(TYPENAME, FNNAME) \
extern "C" __global__ \
void FNNAME(const TYPENAME *src, TYPENAME *dst, uint32_t d1, uint32_t d2, uint32_t src_s, uint32_t dst_s) { \
  copy2d(src, dst, d1, d2, src_s, dst_s); \
} \

COPY2D_OP(float, copy2d_f32)
COPY2D_OP(double, copy2d_f64)
COPY2D_OP(uint8_t, copy2d_u8)
COPY2D_OP(uint32_t, copy2d_u32)
COPY2D_OP(int64_t, copy2d_i64)

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void fill_f16(__half *buf, __half value, const size_t numel) { fill_with(buf, value, numel); }
COPY2D_OP(__half, copy2d_f16)
#endif

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#include <cuda_fp8.h>

extern "C" __global__ void fill_bf16(__nv_bfloat16 *buf, __nv_bfloat16 value, const size_t numel) { fill_with(buf, value, numel); }
COPY2D_OP(__nv_bfloat16, copy2d_bf16)

extern "C" __global__ void fill_f8_e4m3(__nv_fp8_e4m3 *buf, __nv_fp8_e4m3 value, const size_t numel) { fill_with(buf, value, numel); }
COPY2D_OP(__nv_fp8_e4m3, copy2d_f8_e4m3)
#endif
