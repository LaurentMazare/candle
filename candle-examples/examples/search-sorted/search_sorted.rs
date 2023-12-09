use candle::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape};
use half::{bf16, f16};
use rayon::prelude::*;
use std::fmt::Debug;
use std::marker::{Send, Sync};

use crate::cuda_kernels::SEARCH_SORTED_KERNEL;

pub struct SearchSorted {
    pub right: bool,
}

pub trait Sortable<T: PartialOrd + Debug + Sync + Send> {
    fn search_sorted(
        &self,
        innerdim_bd: usize,
        values: &[T],
        innerdim_val: usize,
        is_1d_bd: bool,
        is_1d_vals: bool,
        right: bool,
    ) -> Vec<i64>;
}
macro_rules! match_cpu_storage {
    ($s1:expr, $s2:expr, $code:expr) => {
        match $s1 {
            CpuStorage::U8(vs) => match $s2 {
                CpuStorage::U8(values) => $code(vs, values),
                _ => candle::bail!(
                    "Sorted sequence and values must be of the same type: got {:?} and {:?}",
                    $s1,
                    $s2
                ),
            },
            CpuStorage::U32(vs) => match $s2 {
                CpuStorage::U32(values) => $code(vs, values),
                _ => candle::bail!(
                    "Sorted sequence and values must be of the same type: got {:?} and {:?}",
                    $s1,
                    $s2
                ),
            },
            CpuStorage::I64(vs) => match $s2 {
                CpuStorage::I64(values) => $code(vs, values),
                _ => candle::bail!(
                    "Sorted sequence and values must be of the same type: got {:?} and {:?}",
                    $s1,
                    $s2
                ),
            },
            CpuStorage::BF16(vs) => match $s2 {
                CpuStorage::BF16(values) => $code(vs, values),
                _ => candle::bail!(
                    "Sorted sequence and values must be of the same type: got {:?} and {:?}",
                    $s1,
                    $s2
                ),
            },
            CpuStorage::F16(vs) => match $s2 {
                CpuStorage::F16(values) => $code(vs, values),
                _ => candle::bail!(
                    "Sorted sequence and values must be of the same type: got {:?} and {:?}",
                    $s1,
                    $s2
                ),
            },
            CpuStorage::F32(vs) => match $s2 {
                CpuStorage::F32(values) => $code(vs, values),
                _ => candle::bail!(
                    "Sorted sequence and values must be of the same type: got {:?} and {:?}",
                    $s1,
                    $s2
                ),
            },
            CpuStorage::F64(vs) => match $s2 {
                CpuStorage::F64(values) => $code(vs, values),
                _ => candle::bail!(
                    "Sorted sequence and values must be of the same type: got {:?} and {:?}",
                    $s1,
                    $s2
                ),
            },
        }
    };
}

fn binary_search<T: PartialOrd>(slice: &[T], value: &T, right: bool) -> i64 {
    let mut start: usize = 0;
    let mut end: usize = slice.len();
    while start < end {
        let mid = start + ((end - start) >> 1);
        let mid_val = &slice[mid];
        let pred = if right {
            !(mid_val > value)
        } else {
            !(mid_val >= value)
        };
        if pred {
            start = mid + 1;
        } else {
            end = mid;
        }
    }
    start as i64
}
impl<T: PartialOrd + Debug + Sync + Send> Sortable<T> for Vec<T> {
    fn search_sorted(
        &self,
        innerdim_bd: usize,
        values: &[T],
        innerdim_val: usize,
        is_1d_bd: bool,
        is_1d_vals: bool,
        right: bool,
    ) -> Vec<i64> {
        let indices: Vec<i64> = match (is_1d_bd, is_1d_vals) {
            // 1-d sorted seq, n-d vals --> apply each "row" of vals to the sorted seq
            (true, false) => {
                let num_val_its = values.len() / innerdim_val;
                (0..num_val_its)
                    .into_par_iter()
                    .map(|i| {
                        let slice = &self[..];
                        let vals = &values[i * innerdim_val..(i + 1) * innerdim_val];
                        let mut inner_vec: Vec<i64> = Vec::new();
                        for v in vals {
                            let found = binary_search(slice, v, right);
                            inner_vec.push(found as i64);
                        }
                        inner_vec
                    })
                    .flatten()
                    .collect()
            }
            // n-d sorted seq, 1-d vals --> search for vals in each row of sorted seq
            (false, true) => {
                let num_it = self.len() / innerdim_bd;
                let matches: Vec<i64> = (0..num_it)
                    .into_par_iter()
                    // .step_by(innerdim_bd)
                    .map(|i| {
                        let slice = &self[i * innerdim_bd..(i + 1) * innerdim_bd];
                        let vals = &values[..];
                        let mut inner_vec: Vec<i64> = Vec::new();
                        for v in vals {
                            let found = binary_search(slice, v, right);
                            inner_vec.push(found as i64);
                        }
                        inner_vec
                    })
                    .flatten()
                    .collect();
                matches
            }
            // N-d sorted seq, N-d vals --> num "rows" of vals must be equal to the num "rows" of sorted seq
            // each row of vals is applied to the corresponding row of sorted seq
            _ => {
                assert!(self.len() / innerdim_bd == values.len() / innerdim_val);

                let num_it = self.len() / innerdim_bd;
                let matches: Vec<i64> = (0..num_it)
                    .into_par_iter()
                    // .step_by(innerdim_bd)
                    .map(|i| {
                        let mut inner_vec: Vec<i64> = Vec::new();
                        let slice = &self[i * innerdim_bd..(i + 1) * innerdim_bd];
                        let vals = &values[i * innerdim_val..(i + 1) * innerdim_val];
                        for v in vals {
                            let found = binary_search(slice, v, right);
                            inner_vec.push(found as i64);
                        }
                        inner_vec
                    })
                    .flatten()
                    .collect();
                matches
            }
        };
        indices
    }
}

impl CustomOp2 for SearchSorted {
    fn name(&self) -> &'static str {
        "search-sorted"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let rank_bd = l1.shape().rank();
        let l1_dims = l1.shape().dims().to_vec();
        let l2_dims = l2.shape().dims().to_vec();
        let (innerdim_bd, leadingdims_bd) = l1_dims.split_last().unwrap();
        let (innerdim_val, leadingdims_val) = l2_dims.split_last().unwrap();

        // let innerdim_bd = l1.shape().dims()[rank_bd - 1];
        let numels_bd = l1.shape().elem_count();
        assert!(numels_bd % innerdim_bd == 0);
        // let num_rows_bd = numels_bd / innerdim_bd;

        let rank_val = l2.shape().rank();
        let numels_val = l2.shape().elem_count();
        assert!(numels_val % innerdim_val == 0);
        // let num_rows_val = numels_val / innerdim_val;

        if rank_bd != 1 && rank_val != 1 {
            //Check that leading dims are the same
            assert!(leadingdims_bd == leadingdims_val);
        }

        //Check that sorted seq is sorted
        //Check contiguity
        if l1.contiguous_offsets().is_none() | l2.contiguous_offsets().is_none() {
            candle::bail!("input has to be contiguous");
        }

        let is_1d_bd = l1.shape().rank() == 1;
        let is_1d_vals = l2.shape().rank() == 1;

        let indices = match_cpu_storage!(s1, s2, |vs: &Vec<_>, values: &Vec<_>| {
            let indices = vs.search_sorted(
                *innerdim_bd,
                values,
                *innerdim_val,
                is_1d_bd,
                is_1d_vals,
                self.right,
            );
            CpuStorage::I64(indices)
        });
        let output_dims = match is_1d_bd {
            true => [leadingdims_val, &[*innerdim_val]].concat(),
            false => [leadingdims_bd, &[*innerdim_val]].concat(),
        };
        let output_shape = Shape::from_dims(&output_dims);

        Ok((indices, output_shape))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::{LaunchAsync, LaunchConfig};
        use candle::cuda_backend::WrapErr;

        let rank_bd = l1.shape().rank();
        let l1_dims = l1.shape().dims().to_vec();
        let l2_dims = l2.shape().dims().to_vec();
        let (innerdim_bd, leadingdims_bd) = l1_dims.split_last().unwrap();
        let (innerdim_val, leadingdims_val) = l2_dims.split_last().unwrap();

        // let innerdim_bd = l1.shape().dims()[rank_bd - 1];
        let numels_bd = l1.shape().elem_count();
        assert!(numels_bd % innerdim_bd == 0);
        // let num_rows_bd = numels_bd / innerdim_bd;

        let rank_val = l2.shape().rank();
        let numels_val = l2.shape().elem_count();
        assert!(numels_val % innerdim_val == 0);
        // let num_rows_val = numels_val / innerdim_val;

        if rank_bd != 1 && rank_val != 1 {
            //Check that leading dims are the same
            assert!(leadingdims_bd == leadingdims_val);
        }
        if s1.dtype() != s2.dtype() {
            candle::bail!(
                "Sorted sequence and values must be of the same type: got {:?} and {:?}",
                s1.dtype(),
                s2.dtype()
            );
        }

        //Check that sorted seq is sorted
        //Check contiguity
        if l1.contiguous_offsets().is_none() | l2.contiguous_offsets().is_none() {
            candle::bail!("input has to be contiguous");
        }

        let is_1d_bd = l1.shape().rank() == 1;
        let is_1d_vals = l2.shape().rank() == 1;

        let dev = s1.device().clone();

        let output_dims = match is_1d_bd {
            true => [leadingdims_val, &[*innerdim_val]].concat(),
            false => [leadingdims_bd, &[*innerdim_val]].concat(),
        };
        let output_shape = Shape::from_dims(&output_dims);
        let output_slice = unsafe { dev.alloc::<i64>(output_shape.elem_count()) }.w()?;

        let idim_in = *innerdim_val as u32;
        let idim_bd = *innerdim_bd as u32;
        let numel_in = output_shape.elem_count() as u32;

        let threads_per_block = std::cmp::min(1024, output_shape.elem_count() as u32);
        let num_blocks = (numel_in + threads_per_block - 1) / threads_per_block;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        macro_rules! dispatch_cuda {
            ($t:ty, $kernel_name:expr) => {{
                let slice_ss = s1.as_cuda_slice::<$t>()?;
                let slice_ss = match l1.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => slice_ss.slice(o1..o2),
                };
                let slice_vals = s2.as_cuda_slice::<$t>()?;
                let slice_vals = match l2.contiguous_offsets() {
                    None => candle::bail!("input has to be contiguous"),
                    Some((o1, o2)) => slice_vals.slice(o1..o2),
                };
                let params = (
                    &output_slice,
                    &slice_vals,
                    &slice_ss,
                    idim_in,
                    idim_bd,
                    numel_in,
                    self.right,
                    is_1d_bd,
                    is_1d_vals,
                );
                let func = dev.get_or_load_func($kernel_name, SEARCH_SORTED_KERNEL)?;

                unsafe { func.launch(cfg, params) }.w()?;
            }};
        }

        //Dispatch based on dtype
        match s1.dtype() {
            DType::U32 => dispatch_cuda!(u32, "search_sorted_u32"),
            DType::F32 => dispatch_cuda!(f32, "search_sorted_f32"),
            DType::U8 => dispatch_cuda!(u8, "search_sorted_u8"),
            DType::F64 => dispatch_cuda!(f64, "search_sorted_f64"),
            DType::I64 => dispatch_cuda!(i64, "search_sorted_i64"),
            DType::BF16 => dispatch_cuda!(bf16, "search_sorted_bf16"),
            DType::F16 => dispatch_cuda!(f16, "search_sorted_f16"),
        }
        let output = CudaStorage::wrap_cuda_slice(output_slice, dev.clone());

        // let dst = candle::CudaStorage::wrap_cuda_slice(&[1, 2, 3], dev);
        Ok((output, output_shape))
    }
}
#[cfg(test)]
mod tests {

    use super::*;
    use candle::{Device, Tensor};

    macro_rules! test_cuda_dispatch {
        ($t:ty, $ss:expr, $vals:expr, $ss_shape:expr, $vals_shape:expr, $right:expr, $expected:expr) => {
            let device = Device::new_cuda(0).unwrap();
            let ss: Vec<$t> = $ss;
            let ss_shape = Shape::from_dims(&$ss_shape[..]);

            let vals: Vec<$t> = $vals;
            let vals_shape = Shape::from_dims(&$vals_shape[..]);
            let t1 = Tensor::from_vec(ss, &ss_shape, &device).unwrap();
            let t2 = Tensor::from_vec(vals, &vals_shape, &device).unwrap();
            //Test left
            let t3 = t1.apply_op2(&t2, SearchSorted { right: $right }).unwrap();
            assert!(
                t3.flatten_all().unwrap().to_vec1::<i64>().unwrap() == $expected,
                "Expected {:?}, got {:?}",
                $expected,
                t3.flatten_all().unwrap().to_vec1::<i64>().unwrap()
            );
        };
    }
    macro_rules! test_cuda_dispatch_half {
        ($t:ty, $ss:expr, $vals:expr, $ss_shape:expr, $vals_shape:expr, $right:expr, $expected:expr) => {
            let device = Device::new_cuda(0).unwrap();
            let ss: Vec<$t> = $ss.into_iter().map(|x| <$t>::from_f32(x)).collect();
            let ss_shape = Shape::from_dims(&$ss_shape[..]);

            let vals: Vec<$t> = $vals.into_iter().map(|x| <$t>::from_f32(x)).collect();
            let vals_shape = Shape::from_dims(&$vals_shape[..]);
            let t1 = Tensor::from_vec(ss, &ss_shape, &device).unwrap();
            let t2 = Tensor::from_vec(vals, &vals_shape, &device).unwrap();
            //Test left
            let t3 = t1.apply_op2(&t2, SearchSorted { right: $right }).unwrap();
            assert!(
                t3.flatten_all().unwrap().to_vec1::<i64>().unwrap() == $expected,
                "Expected {:?}, got {:?}",
                $expected,
                t3.flatten_all().unwrap().to_vec1::<i64>().unwrap()
            );
        };
    }
    #[test]
    fn test_cuda_ss1d_vals1d_u8() {
        test_cuda_dispatch!(
            u8,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }

    #[test]
    fn test_cuda_ss1d_vals1d_u32() {
        test_cuda_dispatch!(
            u32,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals1d_u32_right() {
        test_cuda_dispatch!(
            u32,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            true,
            vec![2, 3, 5]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals1d_i64() {
        test_cuda_dispatch!(
            i64,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals1d_f32() {
        test_cuda_dispatch!(
            f32,
            vec![1., 3., 5., 7., 9.],
            vec![3., 6., 9.],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }
    #[test]
    fn test_cuda_ss2d_vals1d_f32() {
        test_cuda_dispatch!(
            f32,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9.],
            vec![2, 5],
            vec![3],
            false,
            vec![1, 3, 4, 1, 2, 4]
        );
    }
    #[test]
    fn test_cuda_ss2d_vals1d_f32_right() {
        test_cuda_dispatch!(
            f32,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9.],
            vec![2, 5],
            vec![3],
            true,
            vec![2, 3, 5, 1, 3, 4]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals2d_f32() {
        test_cuda_dispatch!(
            f32,
            vec![1., 3., 5., 7., 9.],
            vec![3., 6., 9., 3., 6., 9.],
            vec![5],
            vec![2, 3],
            false,
            vec![1, 3, 4, 1, 3, 4]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals2d_f32_right() {
        test_cuda_dispatch!(
            f32,
            vec![1., 3., 5., 7., 9.],
            vec![3., 6., 9., 3., 6., 9.],
            vec![5],
            vec![2, 3],
            true,
            vec![2, 3, 5, 2, 3, 5]
        );
    }
    #[test]
    fn test_cuda_ss2d_vals2d_f32() {
        test_cuda_dispatch!(
            f32,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9., 1., 2., 3.],
            vec![2, 5],
            vec![2, 3],
            false,
            vec![1, 3, 4, 0, 0, 1]
        );
    }
    #[test]
    fn test_cuda_ss2d_vals2d_f32_right() {
        test_cuda_dispatch!(
            f32,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9., 1., 2., 3.],
            vec![2, 5],
            vec![2, 3],
            true,
            vec![2, 3, 5, 0, 1, 1]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals1d_f64() {
        test_cuda_dispatch!(
            f32,
            vec![1., 3., 5., 7., 9.],
            vec![3., 6., 9.],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals1d_f16() {
        test_cuda_dispatch_half!(
            f16,
            vec![1., 3., 5., 7., 9.],
            vec![3., 6., 9.],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals1d_f16_right() {
        test_cuda_dispatch_half!(
            f16,
            vec![1., 3., 5., 7., 9.],
            vec![3., 6., 9.],
            vec![5],
            vec![3],
            true,
            vec![2, 3, 5]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals1d_bf16() {
        test_cuda_dispatch_half!(
            bf16,
            vec![1., 3., 5., 7., 9.],
            vec![3., 6., 9.],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }
    #[test]
    fn test_cuda_ss1d_vals1d_bf16_right() {
        test_cuda_dispatch_half!(
            bf16,
            vec![1., 3., 5., 7., 9.],
            vec![3., 6., 9.],
            vec![5],
            vec![3],
            true,
            vec![2, 3, 5]
        );
    }
    macro_rules! test_cpu_shapes {
        ($t:ty, $ss:expr, $vals:expr, $ss_shape:expr, $vals_shape:expr, $right:expr, $expected:expr) => {
            let device = Device::Cpu;
            let ss: Vec<$t> = $ss;
            let ss_shape = Shape::from_dims(&$ss_shape[..]);

            let vals: Vec<$t> = $vals;
            let vals_shape = Shape::from_dims(&$vals_shape[..]);
            let t1 = Tensor::from_vec(ss, &ss_shape, &device).unwrap();
            let t2 = Tensor::from_vec(vals, &vals_shape, &device).unwrap();
            //Test left
            let t3 = t1.apply_op2(&t2, SearchSorted { right: $right }).unwrap();
            assert!(
                t3.flatten_all().unwrap().to_vec1::<i64>().unwrap() == $expected,
                "Expected {:?}, got {:?}",
                $expected,
                t3.flatten_all().unwrap().to_vec1::<i64>().unwrap()
            );
        };
    }
    #[test]
    fn test_cpu_ss2d_vals1d_u8() {
        test_cpu_shapes!(
            u8,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9],
            vec![2, 5],
            vec![3],
            false,
            vec![1, 3, 4, 1, 2, 4]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals1d_u8_right() {
        test_cpu_shapes!(
            u8,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9],
            vec![2, 5],
            vec![3],
            true,
            vec![2, 3, 5, 1, 3, 4]
        );
    }
    #[test]
    fn test_cpu_ss1d_vals2d_u8() {
        test_cpu_shapes!(
            u8,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9, 3, 6, 9],
            vec![5],
            vec![2, 3],
            false,
            vec![1, 3, 4, 1, 3, 4]
        );
    }
    #[test]
    fn test_cpu_ss1d_vals2d_u8_right() {
        test_cpu_shapes!(
            u8,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9, 3, 6, 9],
            vec![5],
            vec![2, 3],
            true,
            vec![2, 3, 5, 2, 3, 5]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals2d_u8() {
        test_cpu_shapes!(
            u8,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9, 1, 2, 3],
            vec![2, 5],
            vec![2, 3],
            false,
            vec![1, 3, 4, 0, 0, 1]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals2d_u8_right() {
        test_cpu_shapes!(
            u8,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9, 1, 2, 3],
            vec![2, 5],
            vec![2, 3],
            true,
            vec![2, 3, 5, 0, 1, 1]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals1d_u32() {
        test_cpu_shapes!(
            u32,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9],
            vec![2, 5],
            vec![3],
            false,
            vec![1, 3, 4, 1, 2, 4]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals1d_u32_right() {
        test_cpu_shapes!(
            u32,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9],
            vec![2, 5],
            vec![3],
            true,
            vec![2, 3, 5, 1, 3, 4]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals1d_i64() {
        test_cpu_shapes!(
            i64,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9],
            vec![2, 5],
            vec![3],
            false,
            vec![1, 3, 4, 1, 2, 4]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals1d_i64_right() {
        test_cpu_shapes!(
            i64,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9],
            vec![2, 5],
            vec![3],
            true,
            vec![2, 3, 5, 1, 3, 4]
        );
    }
    #[test]
    fn test_cpu_ss1d_vals1d_u8() {
        test_cpu_shapes!(
            u8,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }
    #[test]
    fn test_cpu_ss1d_vals1d_u8_right() {
        test_cpu_shapes!(
            u8,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            true,
            vec![2, 3, 5]
        );
    }
    #[test]
    fn test_cpu_ss1d_vals1d_u32() {
        test_cpu_shapes!(
            u32,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }
    #[test]
    fn test_cpu_ss1d_vals1d_u32_right() {
        test_cpu_shapes!(
            u32,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            true,
            vec![2, 3, 5]
        );
    }
    #[test]
    fn test_cpu_ss1d_vals1d_i64() {
        test_cpu_shapes!(
            u32,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            false,
            vec![1, 3, 4]
        );
    }
    #[test]
    fn test_cpu_ss1d_vals1d_i64_right() {
        test_cpu_shapes!(
            u32,
            vec![1, 3, 5, 7, 9],
            vec![3, 6, 9],
            vec![5],
            vec![3],
            true,
            vec![2, 3, 5]
        );
    }

    //Macro for testing remaining types (CPU)
    macro_rules! test_cpu_ss2d_vals2d {
        ($t:ty, $ss:expr, $vals:expr) => {
            let device = Device::Cpu;
            let ss: Vec<$t> = $ss;
            let ss_shape = Shape::from_dims(&[2, 5]);
            let vals: Vec<$t> = $vals;
            let vals_shape = Shape::from_dims(&[2, 3]);

            // Test left
            let t1 = Tensor::from_vec::<_, $t>(ss, &ss_shape, &device).unwrap();
            let t2 = Tensor::from_vec::<_, $t>(vals, &vals_shape, &device).unwrap();
            let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();

            let expected_indices: Vec<i64> = vec![1, 3, 4, 0, 0, 1];
            let expected_shape = Shape::from_dims(&[2, 3]);
            let actual_shape = t3.shape();
            let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                actual_indices == expected_indices,
                "Expected {:?}, got {:?}",
                expected_indices,
                actual_indices
            );
            assert!(
                actual_shape.dims() == expected_shape.dims(),
                "Expected shape {:?}, got {:?}",
                expected_shape,
                actual_shape
            );

            let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
            let expected_indices: Vec<i64> = vec![2, 3, 5, 0, 1, 1];
            let expected_shape = Shape::from_dims(&[2, 3]);
            let actual_shape = t3.shape();
            let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                actual_indices == expected_indices,
                "Expected {:?}, got {:?}",
                expected_indices,
                actual_indices
            );
            assert!(
                actual_shape.dims() == expected_shape.dims(),
                "Expected shape {:?}, got {:?}",
                expected_shape,
                actual_shape
            );
        };
    }

    macro_rules! test_cpu_ss2d_vals2d_half {
        ($t:ty, $ss:expr, $vals:expr) => {
            let device = Device::Cpu;
            let ss: Vec<$t> = $ss.iter().map(|x| <$t>::from_f32(*x)).collect();
            let ss_shape = Shape::from_dims(&[2, 5]);
            let vals: Vec<$t> = $vals.iter().map(|x| <$t>::from_f32(*x)).collect();
            let vals_shape = Shape::from_dims(&[2, 3]);

            // Test left
            let t1 = Tensor::from_vec::<_, $t>(ss, &ss_shape, &device).unwrap();
            let t2 = Tensor::from_vec::<_, $t>(vals, &vals_shape, &device).unwrap();
            let t3 = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();

            let expected_indices: Vec<i64> = vec![1, 3, 4, 0, 0, 1];
            let expected_shape = Shape::from_dims(&[2, 3]);
            let actual_shape = t3.shape();
            let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                actual_indices == expected_indices,
                "Expected {:?}, got {:?}",
                expected_indices,
                actual_indices
            );
            assert!(
                actual_shape.dims() == expected_shape.dims(),
                "Expected shape {:?}, got {:?}",
                expected_shape,
                actual_shape
            );

            let t3 = t1.apply_op2(&t2, SearchSorted { right: true }).unwrap();
            let expected_indices: Vec<i64> = vec![2, 3, 5, 0, 1, 1];
            let expected_shape = Shape::from_dims(&[2, 3]);
            let actual_shape = t3.shape();
            let actual_indices: Vec<i64> = t3.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                actual_indices == expected_indices,
                "Expected {:?}, got {:?}",
                expected_indices,
                actual_indices
            );
            assert!(
                actual_shape.dims() == expected_shape.dims(),
                "Expected shape {:?}, got {:?}",
                expected_shape,
                actual_shape
            );
        };
    }
    #[test]
    fn test_cpu_ss2d_vals2d_u32() {
        test_cpu_ss2d_vals2d!(
            u32,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9, 1, 2, 3]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals2d_i64() {
        test_cpu_ss2d_vals2d!(
            i64,
            vec![1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            vec![3, 6, 9, 1, 2, 3]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals2d_f64() {
        test_cpu_ss2d_vals2d!(
            f64,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9., 1., 2., 3.]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals2d_f16() {
        test_cpu_ss2d_vals2d_half!(
            f16,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9., 1., 2., 3.]
        );
    }
    #[test]
    fn test_cpu_ss2d_vals2d_bf16() {
        test_cpu_ss2d_vals2d_half!(
            bf16,
            vec![1., 3., 5., 7., 9., 2., 4., 6., 8., 10.],
            vec![3., 6., 9., 1., 2., 3.]
        );
    }

    // #[test]
    // #[should_panic(expected = "Incompatible dtypes")]
    // fn test_different_dtypes() {
    //     let device = Device::Cpu;
    //     let ss: Vec<u32> = vec![1, 2, 3, 4, 5];
    //     let ss_shape = Shape::from_dims(&[5]);
    //     let vals: Vec<i64> = vec![1, 2, 3];
    //     let vals_shape = Shape::from_dims(&[3]);

    //     // Test left
    //     let t1 = Tensor::from_vec(ss, &ss_shape, &device).unwrap();
    //     let t2 = Tensor::from_vec(vals, &vals_shape, &device).unwrap();

    //     //Should panic
    //     _ = t1.apply_op2(&t2, SearchSorted { right: false }).unwrap();
    // }
}
