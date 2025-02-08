#include "binary_op_macros.cuh"
#include<stdint.h>

#if __CUDA_ARCH__ >= 800 || (__CUDA_ARCH__ >= 530 && __CUDA_ARCH__ < 800)
BINARY_OP(__nv_bfloat16, badd_bf16, x + y)
BINARY_OP(__nv_bfloat16, bdiv_bf16, x / y)
BINARY_OP(__nv_bfloat16, bmul_bf16, x * y)
BINARY_OP(__nv_bfloat16, bsub_bf16, x - y)
BINARY_OP(__nv_bfloat16, bmaximum_bf16, maxg(x, y))
BINARY_OP(__nv_bfloat16, bminimum_bf16, ming(x, y))
BINARY_OP_OUT(__nv_bfloat16, uint8_t, eq_bf16, x == y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, ne_bf16, x != y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, lt_bf16, x < y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, le_bf16, x <= y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, gt_bf16, x > y)
BINARY_OP_OUT(__nv_bfloat16, uint8_t, ge_bf16, x >= y)
#endif

#if __CUDA_ARCH__ >= 530
BINARY_OP(__half, badd_f16, x + y)
BINARY_OP(__half, bdiv_f16, x / y)
BINARY_OP(__half, bmul_f16, x * y)
BINARY_OP(__half, bsub_f16, x - y)
BINARY_OP(__half, bmaximum_f16, maxg(x, y))
BINARY_OP(__half, bminimum_f16, ming(x, y))
BINARY_OP_OUT(__half, uint8_t, eq_f16, x == y)
BINARY_OP_OUT(__half, uint8_t, ne_f16, x != y)
BINARY_OP_OUT(__half, uint8_t, lt_f16, x < y)
BINARY_OP_OUT(__half, uint8_t, le_f16, x <= y)
BINARY_OP_OUT(__half, uint8_t, gt_f16, x > y)
BINARY_OP_OUT(__half, uint8_t, ge_f16, x >= y)
#endif

BINARY_OP(float, badd_f32, x + y)
BINARY_OP(double, badd_f64, x + y);
BINARY_OP(uint8_t, badd_u8, x + y);
BINARY_OP(uint32_t, badd_u32, x + y);
BINARY_OP(int64_t, badd_i64, x + y);
BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(double, bdiv_f64, x / y);
BINARY_OP(uint8_t, bdiv_u8, x / y);
BINARY_OP(uint32_t, bdiv_u32, x / y);
BINARY_OP(int64_t, bdiv_i64, x / y);
BINARY_OP(float, bmul_f32, x * y)
BINARY_OP(double, bmul_f64, x * y);
BINARY_OP(uint8_t, bmul_u8, x * y);
BINARY_OP(uint32_t, bmul_u32, x * y);
BINARY_OP(int64_t, bmul_i64, x * y);
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y);
BINARY_OP(uint8_t, bsub_u8, x - y);
BINARY_OP(uint32_t, bsub_u32, x - y);
BINARY_OP(int64_t, bsub_i64, x - y);
BINARY_OP(float, bminimum_f32, ming(x, y));
BINARY_OP(double, bminimum_f64, ming(x, y));
BINARY_OP(uint8_t, bminimum_u8, ming(x, y));
BINARY_OP(uint32_t, bminimum_u32, ming(x, y));
BINARY_OP(int64_t, bminimum_i64, ming(x, y));
BINARY_OP(float, bmaximum_f32, maxg(x, y));
BINARY_OP(double, bmaximum_f64, maxg(x, y));
BINARY_OP(uint8_t, bmaximum_u8, maxg(x, y));
BINARY_OP(uint32_t, bmaximum_u32, maxg(x, y));
BINARY_OP(int64_t, bmaximum_i64, maxg(x, y));

BINARY_OP_OUT(float, uint8_t, eq_f32, x == y)
BINARY_OP_OUT(double, uint8_t, eq_f64, x == y)
BINARY_OP_OUT(uint8_t, uint8_t, eq_u8, x == y)
BINARY_OP_OUT(uint32_t, uint8_t, eq_u32, x == y)
BINARY_OP_OUT(int64_t, uint8_t, eq_i64, x == y)

BINARY_OP_OUT(float, uint8_t, ne_f32, x != y)
BINARY_OP_OUT(double, uint8_t, ne_f64, x != y)
BINARY_OP_OUT(uint8_t, uint8_t, ne_u8, x != y)
BINARY_OP_OUT(uint32_t, uint8_t, ne_u32, x != y)
BINARY_OP_OUT(int64_t, uint8_t, ne_i64, x != y)

BINARY_OP_OUT(float, uint8_t, lt_f32, x < y)
BINARY_OP_OUT(double, uint8_t, lt_f64, x < y)
BINARY_OP_OUT(uint8_t, uint8_t, lt_u8, x < y)
BINARY_OP_OUT(uint32_t, uint8_t, lt_u32, x < y)
BINARY_OP_OUT(int64_t, uint8_t, lt_i64, x < y)

BINARY_OP_OUT(float, uint8_t, le_f32, x <= y)
BINARY_OP_OUT(double, uint8_t, le_f64, x <= y)
BINARY_OP_OUT(uint8_t, uint8_t, le_u8, x <= y)
BINARY_OP_OUT(uint32_t, uint8_t, le_u32, x <= y)
BINARY_OP_OUT(int64_t, uint8_t, le_i64, x <= y)

BINARY_OP_OUT(float, uint8_t, gt_f32, x > y)
BINARY_OP_OUT(double, uint8_t, gt_f64, x > y)
BINARY_OP_OUT(uint8_t, uint8_t, gt_u8, x > y)
BINARY_OP_OUT(uint32_t, uint8_t, gt_u32, x > y)
BINARY_OP_OUT(int64_t, uint8_t, gt_i64, x > y)

BINARY_OP_OUT(float, uint8_t, ge_f32, x >= y)
BINARY_OP_OUT(double, uint8_t, ge_f64, x >= y)
BINARY_OP_OUT(uint8_t, uint8_t, ge_u8, x >= y)
BINARY_OP_OUT(uint32_t, uint8_t, ge_u32, x >= y)
BINARY_OP_OUT(int64_t, uint8_t, ge_i64, x >= y)
