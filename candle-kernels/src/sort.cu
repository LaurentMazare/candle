// Adapted from https://github.com/ggerganov/llama.cpp/blob/master/ggml-cuda/argsort.cu
#define SORT_ORDER_ASC 1
#define SORT_ORDER_DESC 0
#include "cuda_utils.cuh"
#include<stdint.h>

template<typename T>
inline __device__ void swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <typename T>
__device__ void bitonicSortGPU(T* arr, uint32_t * dst, int j, int k, bool ascending) {
    unsigned int i, ij;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    ij = i ^ j;

    if (ij > i) {
        if ((i & k) == 0) {
            // Sort in ascending order
            if (ascending) {
                if (arr[i] > arr[ij]) {
                    swap(arr[i], arr[ij]);
                    swap(dst[i], dst[ij]);
                }
            }
            // Sort in descending order
            else {
                if (arr[i] < arr[ij]) {
                    swap(arr[i], arr[ij]);
                    swap(dst[i], dst[ij]);
                }
            }
        } else {
            // Sort in ascending order
            if (ascending) {
                if (arr[i] < arr[ij]) {
                    swap(arr[i], arr[ij]);
                    swap(dst[i], dst[ij]);
                }
            }
            // Sort in descending order
            else {
                if (arr[i] > arr[ij]) {
                    swap(arr[i], arr[ij]);
                    swap(dst[i], dst[ij]);
                }
            }
        }
    }
}

#define ASORT_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void asort_asc_##RUST_NAME(  \
    TYPENAME * x, uint32_t * dst, const int j, const int k \
) { \
    bitonicSortGPU(x, dst, j, k, true);\
} \
extern "C" __global__ void asort_desc_##RUST_NAME(  \
    TYPENAME * x, uint32_t * dst, const int j, const int k \
) { \
    bitonicSortGPU(x, dst, j, k, false);\
} \
 
#if __CUDA_ARCH__ >= 800
ASORT_OP(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
ASORT_OP(__half, f16)
#endif

ASORT_OP(float, f32)
ASORT_OP(double, f64)
ASORT_OP(uint8_t, u8)
ASORT_OP(uint32_t, u32)
ASORT_OP(int64_t, i64)
