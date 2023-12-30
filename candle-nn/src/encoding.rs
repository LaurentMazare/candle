//! Encoding Utilities. (e.g., one-hot/cold encoding)
//!
//! # Examples
//!
//! ## One-hot encoding
//!
//! ```rust
//! use candle::{Shape, Tensor, Device};
//! use candle_nn::encoding::one_hot;
//!
//! let device = candle::Device::Cpu;
//!
//! let indices = Tensor::new(vec![vec![0f32, 2.], vec![1., -1.]], &device)?;
//! let depth = 4;
//!
//! let allow_f64 = true;
//! let on_value = 1.0;
//! let off_value = 0.0;
//!
//! let one_hot = one_hot::<f32>(indices, depth, allow_f64, on_value, off_value)?;
//!
//! let expected_matrix = [
//!     [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
//!     [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
//! ];
//!
//! assert_eq!(one_hot.shape(), &Shape::from((2, 2, depth)));
//!
//! let matrix = one_hot.to_vec3::<f32>()?;
//!
//! assert_eq!(matrix, expected_matrix);
//!```
//! ## One-cold Encoding
//!
//! ```rust
//! use candle::{Shape, Tensor, Device};
//! use candle_nn::encoding::one_hot;
//!
//!
//! let device = candle::Device::Cpu;
//! let depth = 4;
//! let indices = Tensor::new(vec![vec![0i64, 2], vec![1, 3]], &device)?;
//!
//! let allow_f64 = false;
//! let on_value = 0;
//! let off_value = 1;
//!
//! let one_cold = one_hot::<u8>(indices, depth, allow_f64, on_value, off_value)?;
//!
//! let expected_matrix = [[[0, 1, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 0]]];
//!
//! assert_eq!(one_cold.shape(), &Shape::from((2, 2, depth)));
//!
//! let matrix = one_cold.to_vec3::<u8>()?;
//!
//! assert_eq!(matrix, expected_matrix);
//! ```
//!

use candle::{bail, DType, Result, Tensor, WithDType};

/// One-hot/cold encoding.
///
/// Given an input tensor of indices, this function returns a tensor of the same shape as the input
/// tensor with an additional dimension of the given depth size. The values in the returned tensor are
/// all set to the `off_value` except for the positions represented by the indices, which are set to the `on_value`.
///
/// This method returns a tensor with a rank that is one rank larger than the input tensor.
///
/// As an example, the following tensor will be encoded to a one-hot matrix:
///
/// `[[0i64, 2], [1, -1]]`
///
/// with a depth of 4 will be encoded to:
///
/// `[[[1, 0, 0, 0], [0, 0, 1, 0]], [[0, 1, 0, 0], [0, 0, 0, 0]]]`
///
/// When the input tensor index has a value of -1, the corresponding one-hot vector will be ignored,
/// resulting in a vector of values set to the `off_value`.
///
///
/// This method supports one-cold encoding by setting `on_value` to `0` and `off_value` to `1`.
/// By default `on_value` is `1` and `off_value` is `0`.
///
/// Other encoding values can be used by setting `on_value` and `off_value` to the desired values.
///
/// # Examples
///
/// ## One-hot encoding
///
/// ```rust
/// use candle::{Shape, Tensor, Device};
/// use candle_nn::encoding::one_hot;
///
/// let device = candle::Device::Cpu;
///
/// let indices = Tensor::new(vec![vec![0i64, 2], vec![1, -1]], &device).unwrap();
/// let depth = 4;
///
/// let on_value = 1.0;
/// let off_value = 0.0;
///
/// let one_hot = one_hot::<f32>(indices, depth, on_value, off_value).unwrap();
///
/// let expected_matrix = [
///     [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
///     [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
/// ];
///
/// assert_eq!(one_hot.shape(), &Shape::from((2, 2, depth)));
///
/// let matrix = one_hot.to_vec3::<f32>().unwrap();
///
/// assert_eq!(matrix, expected_matrix);
///```
/// ## One-cold Encoding
///
/// ```rust
/// use candle::{Shape, Tensor, Device};
/// use candle_nn::encoding::one_hot;
///
///
/// let device = candle::Device::Cpu;
/// let depth = 4;
/// let indices = Tensor::new(vec![vec![0u8, 2], vec![1, 3]], &device).unwrap();
///
/// let on_value: u8 = 0;
/// let off_value = 1;
///
/// // Note: the method does not require the turbofish operator, as the type is inferred from the `on_value` and `off_value`.
/// let one_cold = one_hot(indices, depth, on_value, off_value).unwrap();
///
/// let expected_matrix = [[[0, 1, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 0]]];
///
/// assert_eq!(one_cold.shape(), &Shape::from((2, 2, depth)));
///
/// let matrix = one_cold.to_vec3::<u8>().unwrap();
///
/// assert_eq!(matrix, expected_matrix);
/// ```
///
///
/// # Bails
///
/// This method will bail on tensors with a rank greater than 3.
///
/// This method will bail if the index value is less than -1.
///
/// This method will bail if the index value is greater than or equal to the depth value.
///
/// This method will bail if the input data type is not one of `U8`, `U32` or `I64`.
///
/// # API Design
///
/// The api design for this method is loosely based on the [TensorFlow One-Hot](https://www.tensorflow.org/api_docs/python/tf/one_hot) method.
pub fn one_hot<D: WithDType>(
    indices: Tensor,
    depth: usize,
    on_value: D,
    off_value: D,
) -> Result<Tensor> {
    let dtype = indices.dtype();
    let rank = indices.rank();

    match rank {
        0 => {
            let mut v = vec![off_value; depth];
            let idx = 0;
            match dtype {
                DType::U8 => {
                    let vi = indices.to_vec0::<u8>()?;
                    set_usize_value(vi as usize, idx, depth, &mut v, on_value)?;
                }
                DType::U32 => {
                    let vi = indices.to_vec0::<u32>()?;
                    set_usize_value(vi as usize, idx, depth, &mut v, on_value)?;
                }
                DType::I64 => {
                    let vi = indices.to_vec0::<i64>()?;
                    set_int64_value(vi, idx, depth, &mut v, on_value)?;
                }
                d => unsupported_dtype(d)?,
            };

            Tensor::new(v, indices.device())
        }
        1 => {
            let dim1 = indices.dims1()?;
            let mut v = vec![off_value; depth * dim1];

            match dtype {
                DType::U8 => {
                    let index = indices.to_vec1::<u8>()?;
                    for i in 0..dim1 {
                        set_usize_value(index[i] as usize, i * depth, depth, &mut v, on_value)?;
                    }
                }
                DType::U32 => {
                    let index = indices.to_vec1::<u32>()?;
                    for i in 0..dim1 {
                        set_usize_value(index[i] as usize, i * depth, depth, &mut v, on_value)?;
                    }
                }
                DType::I64 => {
                    let index = indices.to_vec1::<i64>()?;
                    for i in 0..dim1 {
                        set_int64_value(index[i], i * depth, depth, &mut v, on_value)?;
                    }
                }
                d => unsupported_dtype(d)?,
            };

            Tensor::new(v, indices.device())?.reshape(&[dim1, depth])
        }
        2 => {
            let (dim1, dim2) = indices.dims2()?;
            let mut v = vec![off_value; depth * dim1 * dim2];

            let idx = |i: usize, j: usize, depth: usize, dim2: usize| -> usize {
                i * depth * dim2 + j * depth
            };

            let iter = (0..dim1).flat_map(|i| (0..dim2).map(move |j| (i, j)));

            match dtype {
                DType::U8 => {
                    let index = indices.to_vec2::<u8>()?;
                    for (i, j) in iter {
                        set_usize_value(
                            index[i][j] as usize,
                            idx(i, j, depth, dim2),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                DType::U32 => {
                    let index = indices.to_vec2::<u32>()?;
                    for (i, j) in iter {
                        set_usize_value(
                            index[i][j] as usize,
                            idx(i, j, depth, dim2),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                DType::I64 => {
                    let index = indices.to_vec2::<i64>()?;
                    for (i, j) in iter {
                        set_int64_value(
                            index[i][j],
                            idx(i, j, depth, dim2),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                d => unsupported_dtype(d)?,
            };

            Tensor::new(v, indices.device())?.reshape(&[dim1, dim2, depth])
        }
        3 => {
            let (dim1, dim2, dim3) = indices.dims3()?;
            let mut v = vec![off_value; depth * dim1 * dim2 * dim3];

            let idx =
                |i: usize, j: usize, k: usize, depth: usize, dim2: usize, dim3: usize| -> usize {
                    i * depth * dim2 * dim3 + j * depth * dim3 + k * depth
                };

            let iter = (0..dim1)
                .flat_map(|i| (0..dim2).flat_map(move |j| (0..dim3).map(move |k| (i, j, k))));

            match dtype {
                DType::U8 => {
                    let index = indices.to_vec3::<u8>()?;
                    for (i, j, k) in iter {
                        set_usize_value(
                            index[i][j][k] as usize,
                            idx(i, j, k, depth, dim2, dim3),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                DType::U32 => {
                    let index = indices.to_vec3::<u32>()?;
                    for (i, j, k) in iter {
                        set_usize_value(
                            index[i][j][k] as usize,
                            idx(i, j, k, depth, dim2, dim3),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                DType::I64 => {
                    let index = indices.to_vec3::<i64>()?;
                    for (i, j, k) in iter {
                        set_int64_value(
                            index[i][j][k],
                            idx(i, j, k, depth, dim2, dim3),
                            depth,
                            &mut v,
                            on_value,
                        )?;
                    }
                }
                d => unsupported_dtype(d)?,
            };

            Tensor::new(v, indices.device())?.reshape(&[dim1, dim2, dim3, depth])
        }
        _ => {
            bail!("one_hot: rank {} is not supported.", rank)
        }
    }
}

// Bail if the value is less than -1.
fn check_negative(value: i64) -> Result<()> {
    if value < -1 {
        bail!("one_hot: Invalid negative index value. Expected a positive index value or `-1` value to ignore. Received {}", value);
    }
    Ok(())
}

// Bail if the value is greater than the depth.
fn check_value_bounds(value: usize, depth: usize) -> Result<()> {
    if value >= depth {
        bail!(
            "one_hot: Index value exceeds the depth value. Index value, {}, must be less than the depth value, {}.",
            value,
            depth
        )
    }

    Ok(())
}

// Bail if the index is out of bounds.
fn check_idx(idx: usize, len: usize) -> Result<()> {
    if idx >= len {
        bail!(
            "one_hot: Index out of bounds. Expected index value, {}, to be less than {}.",
            idx,
            len
        );
    }

    Ok(())
}

// Bail if the data type is not supported.
fn unsupported_dtype(dtype: DType) -> Result<()> {
    bail!(
        "one_hot: Unsupported data type. Expected U8, U32, or I64. Received {:?}",
        dtype
    );
}

// Set unsigned usize index values to the given value.
fn set_usize_value<D: WithDType>(
    value: usize,
    idx: usize,
    depth: usize,
    v: &mut Vec<D>,
    on_value: D,
) -> Result<()> {
    check_value_bounds(value, depth)?;

    let idx = idx + value;

    check_idx(idx, v.len())?;
    v[idx] = on_value;

    Ok(())
}

// Set signed integer index values to the given value.
// Signed integer values are only permitted for `-1` values.
// Otherwise, the value must be positive for unsigned usize values.
// This method will only case i64 values to usize values if the value is positive,
// otherwise the method will bail.
fn set_int64_value<D: WithDType>(
    value: i64,
    idx: usize,
    depth: usize,
    v: &mut Vec<D>,
    on_value: D,
) -> Result<()> {
    // Skip for an entire row of off_values
    if value == -1 {
        return Ok(());
    }

    // Bail if the index value is less than -1
    check_negative(value)?;

    // Since i64 cannot be negative after the previous check,
    // we can safely cast to usize.
    set_usize_value(value as usize, idx, depth, v, on_value)
}

#[cfg(test)]
mod tests {
    use candle::Shape;

    use super::*;

    #[test]
    fn test_i64_one_hot() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = candle::Device::Cpu;

        let indices = Tensor::new(vec![vec![0i64, 2], vec![1, -1]], &device)?;
        let depth = 4;

        let on_value = 1.0;
        let off_value = 0.0;

        let one_hot = one_hot::<f32>(indices, depth, on_value, off_value)?;

        let expected_matrix = [
            [[1., 0., 0., 0.], [0., 0., 1., 0.]],
            [[0., 1., 0., 0.], [0., 0., 0., 0.]],
        ];

        assert_eq!(one_hot.shape(), &Shape::from((2, 2, depth)));

        let matrix = one_hot.to_vec3::<f32>()?;

        assert_eq!(matrix, expected_matrix);

        Ok(())
    }

    #[test]
    fn test_rank_3_one_hot() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = candle::Device::Cpu;

        let indices = Tensor::new(
            vec![
                vec![vec![0i64, 1], vec![2, 3]],
                vec![vec![3, 1], vec![1, -1]],
            ],
            &device,
        )?;
        let depth = 4;

        let on_value = 1.0;
        let off_value = 0.0;

        let one_hot = one_hot::<f32>(indices, depth, on_value, off_value)?;

        let expected_matrix = Tensor::new(
            vec![
                vec![
                    vec![vec![1f32, 0., 0., 0.], vec![0., 1., 0., 0.]],
                    vec![vec![0., 0., 1., 0.], vec![0., 0., 0., 1.]],
                ],
                vec![
                    vec![vec![0., 0., 0., 1.], vec![0., 1., 0., 0.]],
                    vec![vec![0., 1., 0., 0.], vec![0., 0., 0., 0.]],
                ],
            ],
            &device,
        )?;

        assert_eq!(one_hot.shape(), expected_matrix.shape());
        assert_eq!(one_hot.dims(), expected_matrix.dims());

        let matrix = one_hot.get(1)?.to_vec3::<f32>()?;
        let expected_matrix = expected_matrix.get(1)?.to_vec3::<f32>()?;

        assert_eq!(matrix, expected_matrix);

        Ok(())
    }

    #[test]
    fn test_u8_one_cold() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = candle::Device::Cpu;
        let depth = 4;
        let indices = Tensor::new(vec![vec![0i64, 2], vec![1, -1]], &device)?;

        let on_value = 0u8;
        let off_value = 1;

        // Note that the method does not require the turbofish operator, as the type is inferred from the on_value.
        let one_cold = one_hot(indices, depth, on_value, off_value)?;

        let expected_matrix = [[[0, 1, 1, 1], [1, 1, 0, 1]], [[1, 0, 1, 1], [1, 1, 1, 1]]];

        assert_eq!(one_cold.shape(), &Shape::from((2, 2, depth)));

        let matrix = one_cold.to_vec3::<u8>()?;

        assert_eq!(matrix, expected_matrix);

        Ok(())
    }

    #[test]
    fn test_iter() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let device = candle::Device::Cpu;
        let depth = 4;
        let indices = Tensor::new(vec![vec![0i64, 2], vec![1, -1]], &device)?;
        let matrix = indices.to_vec2::<i64>()?;
        let (dim1, dim2) = indices.dims2()?;

        let iter = (0..dim1).flat_map(|i| (0..dim2).map(move |j| (i, j)));

        let mut v = vec![0; depth * dim1 * dim2];

        for (i, j) in iter {
            let idx = i * depth * dim2 + j * depth;
            v[idx] = matrix[i][j];
        }

        for i in 0..dim1 {
            for j in 0..dim2 {
                let idx = i * depth * dim2 + j * depth;
                assert_eq!(v[idx], matrix[i][j]);
            }
        }

        Ok(())
    }
}
