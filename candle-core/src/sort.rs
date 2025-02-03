use crate::{DType, Result, Shape, Storage, Tensor};
use rayon::prelude::*;
#[derive(Debug, Clone)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
    dtype: DType,
    #[cfg(feature = "cuda")]
    indices: Tensor,
}

impl ArgSort {
    fn asort<T: crate::WithDType>(&self, vs: &[T], layout: &crate::Layout) -> Vec<u32> {
        #[allow(clippy::uninit_vec)]
        // Safety: indexes are set later in the parallelized section.
        let mut sort_indexes = unsafe {
            let el_count = layout.shape().elem_count();
            let mut v = Vec::with_capacity(el_count);
            v.set_len(el_count);
            v
        };
        if self.asc {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&i, &j| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        } else {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&j, &i| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        }
        sort_indexes
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::cuda_backend::cudarc::driver::{
        result::memcpy_dtod_sync, CudaSlice, DevicePtr, DeviceRepr, LaunchAsync, LaunchConfig,
        ValidAsZeroBits,
    };
    use crate::cuda_backend::{kernel_name, kernels, CudaStorageSlice as S, WrapErr};
    use crate::{CudaDevice, WithDType};

    impl crate::cuda_backend::Map1Any for ArgSort {
        fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
            &self,
            src: &CudaSlice<T>,
            dev: &CudaDevice,
            layout: &crate::Layout,
            _wrap: W,
        ) -> Result<S> {
            let slice = match layout.contiguous_offsets() {
                None => crate::bail!("input has to be contiguous"),
                Some((o1, o2)) => src.slice(o1..o2),
            };
            let elem_count = layout.shape().elem_count();
            let func = if self.asc {
                dev.get_or_load_func(&kernel_name::<T>("asort_asc"), kernels::SORT)?
            } else {
                dev.get_or_load_func(&kernel_name::<T>("asort_desc"), kernels::SORT)?
            };
            let ncols = self.last_dim;
            let nrows = elem_count / ncols;
            let (indices, _) = self.indices.storage_and_layout();

            let indices = match &*indices {
                Storage::Cuda(k) => k.as_cuda_slice::<u32>()?.to_owned(),
                _ => crate::bail!("indices must be a cuda tensor"),
            };
            let dst = unsafe { dev.alloc::<u32>(elem_count) }.w()?;

            //size of each row must be log2-base for bitonic sort
            let ncols_pad = next_power_of_2(ncols);
            //alloc temp buffer for paddings
            let tmp_rows = dev.const_impl(
                if self.asc {
                    std::f64::MAX
                } else {
                    std::f64::MIN
                },
                &Shape::from((nrows, ncols_pad)),
                self.dtype,
            )?;
            let tmp_indices = unsafe { dev.alloc::<u32>(ncols_pad) }.w()?;
            // Determine the number of threads per block and blocks per row
            let max_threads_per_block = 1024;
            let threads_per_block = max_threads_per_block.min(ncols_pad);
            let blocks_per_row = (ncols_pad + threads_per_block - 1) / threads_per_block;

            let cfg = LaunchConfig {
                grid_dim: (blocks_per_row as u32, 1, 1),
                block_dim: (threads_per_block as u32, 1, 1),
                shared_mem_bytes: (threads_per_block * std::mem::size_of::<u32>()) as u32,
            };

            unsafe {
                for row in 0..nrows {
                    let start_o = row * ncols;
                    let slice_row = slice.slice(start_o..);
                    let dst_row = dst.slice(start_o..);
                    let tmp_row_ptr = match &tmp_rows.slice {
                        S::U8(inp) => *inp.slice(start_o..).device_ptr(),
                        S::U32(inp) => *inp.slice(start_o..).device_ptr(),
                        S::I64(inp) => *inp.slice(start_o..).device_ptr(),
                        S::BF16(inp) => *inp.slice(start_o..).device_ptr(),
                        S::F16(inp) => *inp.slice(start_o..).device_ptr(),
                        S::F32(inp) => *inp.slice(start_o..).device_ptr(),
                        S::F64(inp) => *inp.slice(start_o..).device_ptr(),
                    };

                    memcpy_dtod_sync(
                        tmp_row_ptr,
                        *slice_row.device_ptr(),
                        ncols * std::mem::size_of::<T>(),
                    )
                    .w()?;
                    memcpy_dtod_sync(
                        *tmp_indices.device_ptr(),
                        *indices.device_ptr(),
                        ncols * std::mem::size_of::<u32>(),
                    )
                    .w()?;

                    let mut k = 2;
                    while k <= ncols_pad {
                        // Minor step
                        let mut j = k >> 1;
                        while j > 0 {
                            let params = (tmp_row_ptr, &tmp_indices, j as i32, k as i32);
                            func.clone().launch(cfg, params).w()?;
                            j = j >> 1;
                        }
                        k <<= 1;
                    }

                    //copy back valid elements
                    memcpy_dtod_sync(
                        *slice_row.device_ptr(),
                        tmp_row_ptr,
                        ncols * std::mem::size_of::<T>(),
                    )
                    .w()?;
                    memcpy_dtod_sync(
                        *dst_row.device_ptr(),
                        *tmp_indices.device_ptr(),
                        ncols * std::mem::size_of::<u32>(),
                    )
                    .w()?;
                }
            }
            Ok(S::U32(dst))
        }
    }
}

impl crate::CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, crate::Shape)> {
        let sort_indexes = match storage {
            crate::CpuStorage::U8(vs) => self.asort(vs, layout),
            crate::CpuStorage::U32(vs) => self.asort(vs, layout),
            crate::CpuStorage::I64(vs) => self.asort(vs, layout),
            crate::CpuStorage::BF16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F16(vs) => self.asort(vs, layout),
            crate::CpuStorage::F32(vs) => self.asort(vs, layout),
            crate::CpuStorage::F64(vs) => self.asort(vs, layout),
        };
        let sort_indexes = crate::CpuStorage::U32(sort_indexes);
        Ok((sort_indexes, layout.shape().into()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::cuda_backend::Map1Any;
        let dev = storage.device();
        let slice = self.map(&storage.slice, dev, layout)?;
        let dst = crate::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        use crate::DType;

        let name = {
            if self.asc {
                match storage.dtype() {
                    DType::BF16 => "asort_asc_bf16",
                    DType::F16 => "asort_asc_f16",
                    DType::F32 => "asort_asc_f32",
                    DType::F64 => "asort_asc_f64",
                    DType::U8 => "asort_asc_u8",
                    DType::U32 => "asort_asc_u32",
                    DType::I64 => "asort_asc_i64",
                }
            } else {
                match storage.dtype() {
                    DType::BF16 => "asort_desc_bf16",
                    DType::F16 => "asort_desc_f16",
                    DType::F32 => "asort_desc_f32",
                    DType::F64 => "asort_desc_f64",
                    DType::U8 => "asort_desc_u8",
                    DType::U32 => "asort_desc_u32",
                    DType::I64 => "asort_desc_i64",
                }
            }
        };
        let device = storage.device();
        let kernels = device.kernels();
        let command_buffer = device.command_buffer()?;
        let el = layout.shape().elem_count();
        let ncols = self.last_dim;
        let nrows = el / ncols;
        let src = crate::metal_backend::buffer_o(storage.buffer(), layout, storage.dtype());
        let dst = device.new_buffer(el, DType::U32, "asort")?;
        let mut ncols_pad = 1;
        while ncols_pad < ncols {
            ncols_pad *= 2;
        }
        candle_metal_kernels::call_arg_sort(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            nrows,
            ncols,
            ncols_pad,
            src,
            &dst,
        )
        .map_err(crate::Error::wrap)?;
        let dst = crate::MetalStorage::new(dst, device.clone(), el, DType::U32);
        Ok((dst, layout.shape().clone()))
    }
}

#[allow(unused)]
fn next_power_of_2(x: usize) -> usize {
    let mut n = 1;
    while n < x {
        n *= 2
    }
    n
}

impl Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "arg_sort_last_dim",
            });
        }
        let last_dim = match self.dims().last() {
            None => crate::bail!("empty last-dim in arg-sort"),
            Some(last_dim) => *last_dim,
        };
        #[cfg(feature = "cuda")]
        let indices_cpu = (0..last_dim).into_iter().map(|a| a as u32).collect();
        #[cfg(feature = "cuda")]
        let indices = Tensor::from_vec(indices_cpu, (1, last_dim), self.device())?;
        // No need for a backward pass for arg sort.
        self.apply_op1_no_bwd(&ArgSort {
            asc,
            last_dim,
            dtype: self.dtype(),
            #[cfg(feature = "cuda")]
            indices,
        })
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "sort_last_dim",
            });
        }
        let sorted = self.copy()?;
        let asort = sorted.arg_sort_last_dim(asc)?;
        Ok((sorted, asort))
    }
}
