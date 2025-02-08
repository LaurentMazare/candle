#include "cuda_utils.cuh"
#include<stdint.h>

// Naive implementation of conv1d.
template <typename T, typename A>
__device__ void conv1d(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  // src: (b_size, c_in, l_in)
  // k: (c_out, c_in, k_size)
  const size_t *src_dims = info;
  const size_t *src_s = info + 3;
  const size_t *k_dims = info + 6;
  const size_t *k_s = info + 9;
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t k_size = k_dims[2];
  const size_t c_out = k_dims[0];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];
  if (dst_i >= src_dims[0] * c_out * l_out) {
    return;
  }

  // TODO
  const size_t b_idx = dst_i / (l_out * c_out);
  const size_t dst_c_idx = (dst_i / l_out) % c_out;
  const size_t dst_l = dst_i % l_out;

  const size_t src_idx0 = b_idx * src_s[0];
  A d = 0;
  for (size_t offset = 0; offset < k_size; ++offset) {
    size_t src_l = (stride * dst_l + offset) * dilation;
    if (src_l < padding || src_l >= padding + l_in) {
      continue;
    }
    src_l -= padding;
    for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
      const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_l * src_s[2];
      const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + offset * k_s[2];
      d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
    }
  }
  dst[dst_i] = static_cast<T>(d);
}

template <typename T>
__device__ void im2col1d(
    const size_t dst_numel,
    const size_t l_out,
    const size_t l_k,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t *info,
    const T *src,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // dst: (b_size, l_out, c_in, l_k)
  // src: (b_size, c_in, l_in)
  if (dst_i >= dst_numel) {
    return;
  }
  const size_t *src_dims = info;
  const size_t *src_s = info + 3;
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];

  const size_t dst_s2 = l_k;
  const size_t dst_s1 = c_in * dst_s2;
  const size_t dst_s0 = l_out * dst_s1;

  size_t tmp_dst_i = dst_i;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t l_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= l_idx * dst_s1;
  const size_t c_idx = tmp_dst_i / dst_s2;
  tmp_dst_i -= c_idx * dst_s2;
  const size_t l_k_idx = tmp_dst_i;
  size_t src_l_idx = l_idx * stride + l_k_idx * dilation;
  if (src_l_idx < padding || src_l_idx >= l_in + padding) {
    dst[dst_i] = static_cast<T>(0);
  }
  else {
    src_l_idx -= padding;
    const size_t src_i = b_idx * src_s[0] + c_idx * src_s[1] + src_l_idx * src_s[2];
    dst[dst_i] = src[src_i];
  }
}

template <typename T>
__device__ void col2im1d(
    const size_t dst_el,
    const size_t l_out,
    const size_t l_in,
    const size_t c_out,
    const size_t k_size,
    const size_t stride,
    const T *src,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, l_in, c_out, l_k)
  // dst: (b_size, c_out, l_out)
  if (dst_i >= dst_el) {
    return;
  }

  const size_t dst_s0 = c_out * l_out;
  const size_t dst_s1 = l_out;
  const size_t src_s0 = c_out * k_size * l_in;
  const size_t src_s1 = c_out * k_size;
  const size_t src_s2 = k_size;

  size_t tmp_dst_i = dst_i;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t c_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= c_idx * dst_s1;
  const int l_out_idx = tmp_dst_i;

  dst[dst_i] = static_cast<T>(0);

  int l_in_idx = l_out_idx / stride;
  int k0 = l_out_idx - l_in_idx * stride;
  // l_out_idx = l_in_idx * stride + k0
  for (; k0 < k_size && l_in_idx >= 0; k0 += stride, --l_in_idx) {
    if (l_in_idx < l_in) {
      const size_t src_i = b_idx * src_s0 + l_in_idx * src_s1 + c_idx * src_s2 + k0;
      dst[dst_i] += src[src_i];
    }
  }
}

template <typename T>
__device__ void im2col(
    const size_t dst_numel,
    const size_t h_out,
    const size_t w_out,
    const size_t h_k,
    const size_t w_k,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t *info,
    const T *src,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // dst: (b_size, h_out, w_out, c_in, h_k, w_k)
  // src: (b_size, c_in, h_in, w_in)
  if (dst_i >= dst_numel) {
    return;
  }
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;
  const size_t c_in = src_dims[1];
  const size_t h_in = src_dims[2];
  const size_t w_in = src_dims[3];

  const size_t dst_s4 = w_k;
  const size_t dst_s3 = h_k * dst_s4;
  const size_t dst_s2 = c_in * dst_s3;
  const size_t dst_s1 = w_out * dst_s2;
  const size_t dst_s0 = h_out * dst_s1;

  size_t tmp_dst_i = dst_i;
  const size_t b_idx = tmp_dst_i / dst_s0;
  tmp_dst_i -= b_idx * dst_s0;
  const size_t h_idx = tmp_dst_i / dst_s1;
  tmp_dst_i -= h_idx * dst_s1;
  const size_t w_idx = tmp_dst_i / dst_s2;
  tmp_dst_i -= w_idx * dst_s2;
  const size_t c_idx = tmp_dst_i / dst_s3;
  tmp_dst_i -= c_idx * dst_s3;
  const size_t h_k_idx = tmp_dst_i / dst_s4;
  tmp_dst_i -= h_k_idx * dst_s4;
  const size_t w_k_idx = tmp_dst_i;
  size_t src_h_idx = h_idx * stride + h_k_idx * dilation;
  size_t src_w_idx = w_idx * stride + w_k_idx * dilation;
  if (src_h_idx < padding || src_h_idx >= h_in + padding) {
    dst[dst_i] = static_cast<T>(0);
  }
  else if (src_w_idx < padding || src_w_idx >= w_in + padding) {
    dst[dst_i] = static_cast<T>(0);
  }
  else {
    src_h_idx -= padding;
    src_w_idx -= padding;
    const size_t src_i =
      b_idx * src_s[0]
      + c_idx * src_s[1]
      + src_h_idx * src_s[2]
      + src_w_idx * src_s[3];
    dst[dst_i] = src[src_i];
  }
}

// Naive implementation of conv2d.
template <typename T, typename A>
__device__ void conv2d(
    const size_t src_numel,
    const size_t w_out,
    const size_t h_out,
    const size_t stride,
    const size_t padding,
    const size_t dilation,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, c_in, h_in, w_in)
  // k: (c_out, c_in, h_k, w_k)
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;
  const size_t *k_dims = info + 8;
  const size_t *k_s = info + 12;
  const size_t h_k = k_dims[2];
  const size_t w_k = k_dims[3];
  const size_t c_out = k_dims[0];
  const size_t c_in = src_dims[1];
  const size_t h_in = src_dims[2];
  const size_t w_in = src_dims[3];
  if (dst_i >= src_dims[0] * c_out * w_out * h_out) {
    return;
  }

  // TODO
  const size_t b_idx = dst_i / (w_out * h_out * c_out);
  const size_t dst_c_idx = (dst_i / (w_out * h_out)) % c_out;
  // NCHW layout.
  const size_t dst_h = (dst_i / w_out) % h_out;
  const size_t dst_w = dst_i % w_out;

  const size_t src_idx0 = b_idx * src_s[0];
  A d = 0;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = stride * dst_w + w_offset * dilation;
    if (src_w < padding || src_w >= w_in + padding) {
      continue;
    }
    src_w -= padding;
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = stride * dst_h + h_offset * dilation;
      if (src_h < padding || src_h >= h_in + padding) {
        continue;
      }
      src_h -= padding;
      for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
        const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + src_h * src_s[2] + src_w * src_s[3];
        const size_t k_idx = dst_c_idx * k_s[0] + src_c_idx * k_s[1] + h_offset * k_s[2] + w_offset * k_s[3];
        d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
      }
    }
  }
  dst[dst_i] = static_cast<T>(d);
}

// Naive implementation of conv_transpose1d.
template <typename T, typename A>
__device__ void conv_transpose1d(
    const size_t src_numel,
    const size_t l_out,
    const size_t stride,
    const size_t padding,
    const size_t out_padding,
    const size_t dilation,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, c_in, l_in)
  // k: (c_in, c_out, l_k)
  const size_t *src_dims = info;
  const size_t *src_s = info + 3;
  const size_t *k_dims = info + 6;
  const size_t *k_s = info + 9;
  const size_t l_k = k_dims[2];
  const size_t c_out = k_dims[1];
  const size_t c_in = src_dims[1];
  const size_t l_in = src_dims[2];
  if (dst_i >= src_dims[0] * c_out * l_out) {
    return;
  }

  // TODO
  const size_t b_idx = dst_i / (l_out * c_out);
  const size_t dst_c_idx = (dst_i / l_out) % c_out;
  // NCL layout.
  const size_t out_x = dst_i % l_out;

  const size_t src_idx0 = b_idx * src_s[0];
  A d = 0;
  for (int k_x = 0; k_x < (int)l_k; ++k_x) {
      // let out_x = inp_x * p.stride + k_x * p.dilation - p.padding;
      int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) {
          continue;
      }
      int inp_x = inp_x_stride / stride;
      if (inp_x >= l_in) continue;
      for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
          const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + inp_x * src_s[2];
          const size_t k_idx = src_c_idx * k_s[0] + dst_c_idx * k_s[1] + k_x * k_s[2];
          d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
      }
  }
  dst[dst_i] = static_cast<T>(d);
}

// Naive implementation of conv_transpose2d.
template <typename T, typename A>
__device__ void conv_transpose2d(
    const size_t src_numel,
    const size_t w_out,
    const size_t h_out,
    const size_t stride,
    const size_t padding,
    const size_t out_padding,
    const size_t dilation,
    const size_t *info,
    const T *src,
    const T *kernel,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, c_in, h_in, w_in)
  // k: (c_in, c_out, h_k, w_k)
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;
  const size_t *k_dims = info + 8;
  const size_t *k_s = info + 12;
  const size_t h_k = k_dims[2];
  const size_t w_k = k_dims[3];
  const size_t c_out = k_dims[1];
  const size_t c_in = src_dims[1];
  const size_t h_in = src_dims[2];
  const size_t w_in = src_dims[3];
  if (dst_i >= src_dims[0] * c_out * w_out * h_out) {
    return;
  }

  // TODO
  const size_t b_idx = dst_i / (w_out * h_out * c_out);
  const size_t dst_c_idx = (dst_i / (w_out * h_out)) % c_out;
  // NCHW layout.
  const size_t out_y = (dst_i / w_out) % h_out;
  const size_t out_x = dst_i % w_out;

  const size_t src_idx0 = b_idx * src_s[0];
  A d = 0;
  for (int k_x = 0; k_x < (int)w_k; ++k_x) {
      // let out_x = inp_x * p.stride + k_x * p.dilation - p.padding;
      int inp_x_stride = (int)(out_x + padding) - k_x * dilation;
      if (inp_x_stride < 0 || inp_x_stride % stride) {
          continue;
      }
      int inp_x = inp_x_stride / stride;
      if (inp_x >= w_in) continue;
      for (int k_y = 0; k_y < (int)h_k; ++k_y) {
          int inp_y_stride = (int)(out_y + padding) - k_y * dilation;
          if (inp_y_stride < 0 || inp_y_stride % stride) {
              continue;
          }
          int inp_y = inp_y_stride / stride;
          if (inp_y >= h_in) continue;
          for (size_t src_c_idx = 0; src_c_idx < c_in; ++src_c_idx) {
              const size_t src_idx = src_idx0 + src_c_idx * src_s[1] + inp_y * src_s[2] + inp_x * src_s[3];
              const size_t k_idx = src_c_idx * k_s[0] + dst_c_idx * k_s[1] + k_y * k_s[2] + k_x * k_s[3];
              d += static_cast<A>(src[src_idx]) * static_cast<A>(kernel[k_idx]);
          }
      }
  }
  dst[dst_i] = static_cast<T>(d);
}

template <typename T, typename A>
__device__ void avg_pool2d(
    const size_t src_numel,
    const size_t w_k,
    const size_t h_k,
    const size_t w_stride,
    const size_t h_stride,
    const size_t *info,
    const T *src,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, c_in, w_in, h_in)
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (dst_i >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  // TODO: Improve this.
  const size_t b_idx = dst_i / (w_out * h_out * c);
  const size_t c_idx = (dst_i / (w_out * h_out)) % c;
  const size_t dst_w = (dst_i / h_out) % w_out;
  const size_t dst_h = dst_i % h_out;

  const size_t src_idx0 = b_idx * src_s[0];
  const float scale = 1.0 / (w_k * h_k);
  A d = 0;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in) {
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
      d += static_cast<A>(src[src_idx]);
    }
  }
  dst[dst_i] = static_cast<T>(d * scale);
}

template <typename T>
__device__ void max_pool2d(
    const size_t src_numel,
    const size_t w_k,
    const size_t h_k,
    const size_t w_stride,
    const size_t h_stride,
    const size_t *info,
    const T *src,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, c_in, w_in, h_in)
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  const size_t w_out = (w_in - w_k) / w_stride + 1;
  const size_t h_out = (h_in - h_k) / h_stride + 1;
  if (dst_i >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  // TODO: Improve this.
  const size_t b_idx = dst_i / (w_out * h_out * c);
  const size_t c_idx = (dst_i / (w_out * h_out)) % c;
  const size_t dst_w = (dst_i / h_out) % w_out;
  const size_t dst_h = dst_i % h_out;

  const size_t src_idx0 = b_idx * src_s[0];
  T d = 0;
  bool set = false;
  for (size_t w_offset = 0; w_offset < w_k; ++w_offset) {
    size_t src_w = w_stride * dst_w + w_offset;
    if (src_w >= w_in) {
      continue;
    }
    for (size_t h_offset = 0; h_offset < h_k; ++h_offset) {
      size_t src_h = h_stride * dst_h + h_offset;
      if (src_h >= h_in) {
        continue;
      }
      const size_t src_idx = src_idx0 + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
      if (set) {
        d = maxg(d, src[src_idx]);
      }
      else {
        d = src[src_idx];
        set = true;
      }
    }
  }
  dst[dst_i] = d;
}

template <typename T>
__device__ void upsample_nearest2d(
    const size_t w_out,
    const size_t h_out,
    const double w_scale,
    const double h_scale,
    const size_t *info,
    const T *src,
    T *dst
) {
  const size_t dst_i = blockIdx.x * blockDim.x + threadIdx.x;
  // src: (b_size, c_in, w_in, h_in)
  const size_t *src_dims = info;
  const size_t *src_s = info + 4;

  const size_t c = src_dims[1];
  const size_t w_in = src_dims[2];
  const size_t h_in = src_dims[3];

  if (dst_i >= src_dims[0] * c * w_out * h_out) {
    return;
  }

  // TODO: Improve this.
  const size_t b_idx = dst_i / (w_out * h_out * c);
  const size_t c_idx = (dst_i / (w_out * h_out)) % c;
  const size_t dst_w = (dst_i / h_out) % w_out;
  const size_t dst_h = dst_i % h_out;

  size_t src_w = static_cast<size_t>(dst_w * w_scale);
  size_t src_h = static_cast<size_t>(dst_h * h_scale);
  if (src_w >= w_in) {
    src_w = w_in - 1;
  }
  if (src_h >= h_in) {
    src_h = h_in - 1;
  }

  const size_t src_i = b_idx * src_s[0] + c_idx * src_s[1] + src_w * src_s[2] + src_h * src_s[3];
  dst[dst_i] = src[src_i];
}


#define CONV1D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t num_dims, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv1d<TYPENAME, TYPEACC>(src_numel, num_dims, stride, padding, dilation, info, src, kernel, dst); \
} \

#define CONV2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t w_out, \
    const size_t h_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv2d<TYPENAME, TYPEACC>(src_numel, w_out, h_out, stride, padding, dilation, info, src, kernel, dst); \
} \

#define IM2COL1D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t dst_numel, \
    const size_t l_out, \
    const size_t l_k, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) {  \
  im2col1d<TYPENAME>(dst_numel, l_out, l_k, stride, padding, dilation, info, src, dst); \
} \

#define COL2IM1D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t dst_el, \
    const size_t l_out, \
    const size_t l_in, \
    const size_t c_out, \
    const size_t k_size, \
    const size_t stride, \
    const TYPENAME *src, \
    TYPENAME *dst \
) {  \
  col2im1d<TYPENAME>(dst_el, l_out, l_in, c_out, k_size, stride, src, dst); \
} \

#define IM2COL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t dst_numel, \
    const size_t h_out, \
    const size_t w_out, \
    const size_t h_k, \
    const size_t w_k, \
    const size_t stride, \
    const size_t padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) {  \
  im2col<TYPENAME>(dst_numel, h_out, w_out, h_k, w_k, stride, padding, dilation, info, src, dst); \
} \

#define CONVT1D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t l_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t out_padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv_transpose1d<TYPENAME, TYPEACC>(src_numel, l_out, stride, padding, out_padding, dilation, info, src, kernel, dst); \
} \

#define CONVT2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t w_out, \
    const size_t h_out, \
    const size_t stride, \
    const size_t padding, \
    const size_t out_padding, \
    const size_t dilation, \
    const size_t *info, \
    const TYPENAME *src, \
    const TYPENAME *kernel, \
    TYPENAME *dst \
) {  \
  conv_transpose2d<TYPENAME, TYPEACC>(src_numel, w_out, h_out, stride, padding, out_padding, dilation, info, src, kernel, dst); \
} \

#define AVG_POOL2D_OP(TYPENAME, TYPEACC, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t w_k, \
    const size_t h_k, \
    const size_t w_stride, \
    const size_t h_stride, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) {  \
  avg_pool2d<TYPENAME, TYPEACC>(src_numel, w_k, h_k, w_stride, h_stride, info, src, dst); \
} \

#define MAX_POOL2D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t src_numel, \
    const size_t w_k, \
    const size_t h_k, \
    const size_t w_stride, \
    const size_t h_stride, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) {  \
  max_pool2d<TYPENAME>(src_numel, w_k, h_k, w_stride, h_stride, info, src, dst); \
} \

#define UPSAMPLE_NEAREST2D_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t w_out, \
    const size_t h_out, \
    const double w_scale, \
    const double h_scale, \
    const size_t *info, \
    const TYPENAME *src, \
    TYPENAME *dst \
) {  \
  upsample_nearest2d<TYPENAME>(w_out, h_out, w_scale, h_scale, info, src, dst); \
} \

#if __CUDA_ARCH__ >= 800 || (__CUDA_ARCH__ >= 530 && __CUDA_ARCH__ < 800)
CONV1D_OP(__nv_bfloat16, float, conv1d_bf16)
CONV2D_OP(__nv_bfloat16, float, conv2d_bf16)
CONVT1D_OP(__nv_bfloat16, float, conv_transpose1d_bf16)
CONVT2D_OP(__nv_bfloat16, float, conv_transpose2d_bf16)
AVG_POOL2D_OP(__nv_bfloat16, float, avg_pool2d_bf16)
MAX_POOL2D_OP(__nv_bfloat16, max_pool2d_bf16)
UPSAMPLE_NEAREST2D_OP(__nv_bfloat16, upsample_nearest2d_bf16)
IM2COL_OP(__nv_bfloat16, im2col_bf16)
IM2COL1D_OP(__nv_bfloat16, im2col1d_bf16)
COL2IM1D_OP(__nv_bfloat16, col2im1d_bf16)
#endif

#if __CUDA_ARCH__ >= 530
CONV1D_OP(__half, float, conv1d_f16)
CONV2D_OP(__half, float, conv2d_f16)
CONVT1D_OP(__half, float, conv_transpose1d_f16)
CONVT2D_OP(__half, float, conv_transpose2d_f16)
AVG_POOL2D_OP(__half, float, avg_pool2d_f16)
MAX_POOL2D_OP(__half, max_pool2d_f16)
UPSAMPLE_NEAREST2D_OP(__half, upsample_nearest2d_f16)
IM2COL_OP(__half, im2col_f16)
IM2COL1D_OP(__half, im2col1d_f16)
COL2IM1D_OP(__half, col2im1d_f16)
#endif

CONV1D_OP(float, float, conv1d_f32)
CONV1D_OP(double, double, conv1d_f64)
CONV1D_OP(uint8_t, uint8_t, conv1d_u8)
CONV1D_OP(uint32_t, uint32_t, conv1d_u32)

CONV2D_OP(float, float, conv2d_f32)
CONV2D_OP(double, double, conv2d_f64)
CONV2D_OP(uint8_t, uint8_t, conv2d_u8)
CONV2D_OP(uint32_t, uint32_t, conv2d_u32)

CONVT1D_OP(float, float, conv_transpose1d_f32)
CONVT1D_OP(double, double, conv_transpose1d_f64)
CONVT1D_OP(uint8_t, uint8_t, conv_transpose1d_u8)
CONVT1D_OP(uint32_t, uint32_t, conv_transpose1d_u32)

CONVT2D_OP(float, float, conv_transpose2d_f32)
CONVT2D_OP(double, double, conv_transpose2d_f64)
CONVT2D_OP(uint8_t, uint8_t, conv_transpose2d_u8)
CONVT2D_OP(uint32_t, uint32_t, conv_transpose2d_u32)

AVG_POOL2D_OP(float, float, avg_pool2d_f32)
AVG_POOL2D_OP(double, double, avg_pool2d_f64)
AVG_POOL2D_OP(uint8_t, uint8_t, avg_pool2d_u8)
AVG_POOL2D_OP(uint32_t, uint32_t, avg_pool2d_u32)

MAX_POOL2D_OP(float, max_pool2d_f32)
MAX_POOL2D_OP(double, max_pool2d_f64)
MAX_POOL2D_OP(uint8_t, max_pool2d_u8)
MAX_POOL2D_OP(uint32_t, max_pool2d_u32)

UPSAMPLE_NEAREST2D_OP(float, upsample_nearest2d_f32)
UPSAMPLE_NEAREST2D_OP(double, upsample_nearest2d_f64)
UPSAMPLE_NEAREST2D_OP(uint8_t, upsample_nearest2d_u8)
UPSAMPLE_NEAREST2D_OP(uint32_t, upsample_nearest2d_u32)

IM2COL_OP(float, im2col_f32)
IM2COL_OP(double, im2col_f64)
IM2COL_OP(uint8_t, im2col_u8)
IM2COL_OP(uint32_t, im2col_u32)

IM2COL1D_OP(float, im2col1d_f32)
IM2COL1D_OP(double, im2col1d_f64)
IM2COL1D_OP(uint8_t, im2col1d_u8)
IM2COL1D_OP(uint32_t, im2col1d_u32)

COL2IM1D_OP(float, col2im1d_f32)
COL2IM1D_OP(double, col2im1d_f64)
COL2IM1D_OP(uint8_t, col2im1d_u8)
COL2IM1D_OP(uint32_t, col2im1d_u32)
