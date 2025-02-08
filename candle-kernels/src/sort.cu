// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/argsort.cu
#define SORT_ORDER_ASC 1
#define SORT_ORDER_DESC 0
#include "cuda_utils.cuh"
#include<stdint.h>

template<typename T>
static inline __device__ void ggml_cuda_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template<int order, typename T>
static __device__ void k_argsort(const T * x, uint32_t * dst, const int ncols, int ncols_pad) {
    // bitonic sort
    int col = threadIdx.x;
    int row = blockIdx.y;

    if (col >= ncols_pad) {
        return;
    }

    const T * x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    // initialize indices
    dst_row[col] = col;

    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (dst_row[col] >= ncols ||
                        (dst_row[ixj] < ncols && (order == SORT_ORDER_ASC ?
                            x_row[dst_row[col]] > x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] < x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                } else {
                    if (dst_row[ixj] >= ncols ||
                        (dst_row[col] < ncols && (order == SORT_ORDER_ASC ?
                            x_row[dst_row[col]] < x_row[dst_row[ixj]] :
                            x_row[dst_row[col]] > x_row[dst_row[ixj]]))
                    ) {
                        ggml_cuda_swap(dst_row[col], dst_row[ixj]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // copy the result to dst without the padding
    if (col < ncols) {
        dst[row * ncols + col] = dst_row[col];
    }
}

#define ASORT_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void asort_asc_##RUST_NAME(  \
    const TYPENAME * x, uint32_t * dst, const int ncols, int ncols_pad \
) { \
    k_argsort<SORT_ORDER_ASC>(x, dst, ncols, ncols_pad); \
} \
extern "C" __global__ void asort_desc_##RUST_NAME(  \
    const TYPENAME * x, uint32_t * dst, const int ncols, int ncols_pad \
) { \
    k_argsort<SORT_ORDER_DESC>(x, dst, ncols, ncols_pad); \
} \
 
#if __CUDA_ARCH__ >= 800
ASORT_OP(__nv_bfloat16, bf16)

// NOTE: No sort ops for f8
// ASORT_OP(__nv_fp8_e4m3, fp8_e4m3)
#endif

#if __CUDA_ARCH__ >= 530
ASORT_OP(__half, f16)
#endif

ASORT_OP(float, f32)
ASORT_OP(double, f64)
ASORT_OP(uint8_t, u8)
ASORT_OP(uint32_t, u32)
ASORT_OP(int64_t, i64)
