// Based on https://github.com/NVIDIA/apex/blob/master/apex/contrib/csrc/multihead_attn/layer_norm.cuh#L243
// Modified Eric Buehler 2024

#include "cuda_fp16.h"
#include <cuda.h>
#include <cuda_runtime.h>

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

template <typename U>
__device__ void cuWelfordOnlineSum(const U curr, U &mu, U &sigma2, U &count) {
  count = count + U(1);
  U delta = curr - mu;
  U lmean = mu + delta / count;
  mu = lmean;
  U delta2 = curr - lmean;
  sigma2 = sigma2 + delta * delta2;
}

template <typename U>
__device__ void cuChanOnlineSum(const U muB, const U sigma2B, const U countB,
                                U &mu, U &sigma2, U &count) {
  U delta = muB - mu;
  U nA = count;
  U nB = countB;
  count = count + countB;
  U nX = count;
  if (nX > U(0)) {
    nA = nA / nX;
    nB = nB / nX;
    mu = nA * mu + nB * muB;
    sigma2 = sigma2 + sigma2B + delta * delta * nA * nB * nX;
  } else {
    mu = U(0);
    sigma2 = U(0);
  }
}

// https://github.com/pytorch/pytorch/blob/7fe0cc53e903e515e86b4a350614011c66e3b32d/aten/src/ATen/cuda/DeviceUtils.cuh#L50
template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if !defined(USE_ROCM)
    return __shfl_sync(mask, value, srcLane, width);
#else
    return __shfl(value, srcLane, width);
#endif
}

template <typename T, typename U>
__device__ void cuWelfordMuSigma2(const T *__restrict__ vals, const int n1,
                                  const int n2, const int i1, U &mu, U &sigma2,
                                  U *buf) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  mu = U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T *lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) {
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      U curr = static_cast<U>(lvals[l]);
      cuWelfordOnlineSum<U>(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      U muB = WARP_SHFL(mu, srcLaneB);
      U countB = WARP_SHFL(count, srcLaneB);
      U sigma2B = WARP_SHFL(sigma2, srcLaneB);
      cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U *ubuf = (U *)buf;
      U *ibuf = (U *)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U muB = ubuf[2 * threadIdx.y];
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          U countB = ibuf[threadIdx.y];
          cuChanOnlineSum<U>(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / U(n2), 0);
    }
  }
}

template <>
__device__ void cuWelfordMuSigma2(const __half *__restrict__ vals,
                                  const int n1, const int n2, const int i1,
                                  float &mu, float &sigma2, float *buf) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  float count = 0.0f;
  mu = float(0);
  sigma2 = float(0);

  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const __half *lvals = vals + i1 * n2;
    int l = 8 * thrx;
    if ((((size_t)lvals) & 3) != 0) {
      // 16 bit alignment
      // first thread consumes first point
      if (thrx == 0) {
        float curr = static_cast<float>(lvals[0]);
        cuWelfordOnlineSum(curr, mu, sigma2, count);
      }
      ++l;
    }
    // at this point, lvals[l] are 32 bit aligned for all threads.
    for (; l + 7 < n2; l += 8 * numx) {
      for (int k = 0; k < 8; k += 2) {
        float2 curr = __half22float2(*((__half2 *)(lvals + l + k)));
        cuWelfordOnlineSum(curr.x, mu, sigma2, count);
        cuWelfordOnlineSum(curr.y, mu, sigma2, count);
      }
    }
    for (; l < n2; ++l) {
      float curr = static_cast<float>(lvals[l]);
      cuWelfordOnlineSum(curr, mu, sigma2, count);
    }
    // intra-warp reductions
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & 31;
      float muB = WARP_SHFL(mu, srcLaneB);
      float countB = WARP_SHFL(count, srcLaneB);
      float sigma2B = WARP_SHFL(sigma2, srcLaneB);
      cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      float *ubuf = (float *)buf;
      float *ibuf = (float *)(ubuf + blockDim.y);
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y] = mu;
          ubuf[2 * wrt_y + 1] = sigma2;
          ibuf[wrt_y] = count;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          float muB = ubuf[2 * threadIdx.y];
          float sigma2B = ubuf[2 * threadIdx.y + 1];
          float countB = ibuf[threadIdx.y];
          cuChanOnlineSum(muB, sigma2B, countB, mu, sigma2, count);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[0] = mu;
        ubuf[1] = sigma2;
      }
      __syncthreads();
      mu = ubuf[0];
      sigma2 = ubuf[1] / float(n2);
      // don't care about final value of count, we know count == n2
    } else {
      mu = WARP_SHFL(mu, 0);
      sigma2 = WARP_SHFL(sigma2 / float(n2), 0);
    }
  }
}

template <typename U> __device__ U rsqrt(U v) { return U(1) / sqrt(v); }
template <> __device__ float rsqrt(float v) { return rsqrtf(v); }
template <> __device__ double rsqrt(double v) { return rsqrt(v); }
template <> __device__ __half rsqrt(__half v) { return rsqrt(v); }
#if __CUDA_ARCH__ >= 800
template <> __device__ __nv_bfloat16 rsqrt(__nv_bfloat16 v) { return rsqrt(v); }
#endif

// This is the un-specialized struct.  Note that we prevent instantiation of
// this struct by putting an undefined symbol in the function body so it won't
// compile.
//  template <typename T>
//  struct SharedMemory
//  {
//      // Ensure that we won't compile any un-specialized types
//      __device__ T *getPointer()
//      {
//          extern __device__ void error(void);
//          error();
//          return NULL;
//      }
//  };
// https://github.com/NVIDIA/apex/issues/246
template <typename T> struct SharedMemory;
template <> struct SharedMemory<float> {
  __device__ float *getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <> struct SharedMemory<__half> {
  __device__ __half *getPointer() {
    extern __shared__ __half s_half[];
    return s_half;
  }
};

#if __CUDA_ARCH__ >= 800
template <> struct SharedMemory<__nv_bfloat16> {
  __device__ __nv_bfloat16 *getPointer() {
    extern __shared__ __nv_bfloat16 s_bf[];
    return s_bf;
  }
};
#endif

// Assumptions:
// 1) blockDim.x == warpSize
// 2) Tensors are contiguous
//
#define LAYERNORM(FNNAME, TYPENAME) __global__ void \
extern "C" FNNAME(TYPENAME *__restrict__ output_vals, TYPENAME *__restrict__ mean, \
                 TYPENAME *__restrict__ invvar, const TYPENAME *__restrict__ vals, \
                 const int n1, const int n2, const TYPENAME epsilon, \
                 const TYPENAME *__restrict__ gamma, const TYPENAME *__restrict__ beta) { \
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) { \
    SharedMemory<TYPENAME> shared; \
    TYPENAME *buf = shared.getPointer(); \
    TYPENAME mu, sigma2; \
    cuWelfordMuSigma2(vals, n1, n2, i1, mu, sigma2, buf); \
    const TYPENAME *lvals = vals + i1 * n2; \
    TYPENAME *ovals = output_vals + i1 * n2; \
    TYPENAME c_invvar = rsqrt(sigma2 + epsilon); \
    const int numx = blockDim.x * blockDim.y; \
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x; \
    if (gamma != NULL && beta != NULL) { \
      for (int i = thrx; i < n2; i += numx) { \
        TYPENAME curr = static_cast<TYPENAME>(lvals[i]); \
        ovals[i] = gamma[i] * static_cast<TYPENAME>(c_invvar * (curr - mu)) + beta[i]; \
      } \
    } else { \
      for (int i = thrx; i < n2; i += numx) { \
        TYPENAME curr = static_cast<TYPENAME>(lvals[i]); \
        ovals[i] = static_cast<TYPENAME>(c_invvar * (curr - mu)); \
      } \
    } \
    if (threadIdx.x == 0 && threadIdx.y == 0) { \
      mean[i1] = mu; \
      invvar[i1] = c_invvar; \
    } \
  } \
}

LAYERNORM(layernorm_f16, __half)
LAYERNORM(layernorm_f32, float)

#if __CUDA_ARCH__ >= 800
LAYERNORM(layernorm_bf16, __nv_bfloat16)
#endif
