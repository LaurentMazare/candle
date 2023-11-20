use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, Shape};
use candle_metal_kernels;
use candle_metal_kernels::Kernels;
use half::f16;
use metal;
use metal::mps::matrix::{Matrix, MatrixDescriptor, MatrixMultiplication};
use metal::{Buffer, CommandBuffer, CommandQueue, MTLResourceOptions, NSUInteger};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Metal related errors
#[derive(thiserror::Error, Debug)]
pub enum MetalError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    KernelError(#[from] candle_metal_kernels::MetalKernelError),

    #[error("matmul is only supported for contiguous tensors lstride: {lhs_stride:?} rstride: {rhs_stride:?} mnk: {mnk:?}")]
    MatMulNonContiguous {
        lhs_stride: Vec<usize>,
        rhs_stride: Vec<usize>,
        mnk: (usize, usize, usize),
    },
}

impl From<String> for MetalError {
    fn from(e: String) -> Self {
        MetalError::Message(e)
    }
}

#[derive(Clone)]
pub struct MetalDevice {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    command_buffer: Arc<RwLock<metal::CommandBuffer>>,
    kernels: Arc<candle_metal_kernels::Kernels>,
    buffers: Arc<RwLock<HashMap<(NSUInteger, MTLResourceOptions), Vec<Arc<Buffer>>>>>,
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalDevice({:?})", self.device.registry_id())
    }
}

impl std::ops::Deref for MetalDevice {
    type Target = metal::DeviceRef;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl MetalDevice {
    pub fn id(&self) -> NSUInteger {
        self.registry_id()
    }

    pub fn metal_device(&self) -> &metal::Device {
        &self.device
    }

    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    pub fn command_buffer(&self) -> std::sync::RwLockReadGuard<CommandBuffer> {
        self.command_buffer.try_read().unwrap()
    }

    pub fn commit(&self) {
        let mut old = self.command_buffer.try_write().unwrap();
        match old.status() {
            metal::MTLCommandBufferStatus::NotEnqueued
            | metal::MTLCommandBufferStatus::Enqueued => {
                old.commit();
                let command_buffer = self.command_queue.new_command_buffer().to_owned();
                *old = command_buffer;
            }
            _ => {}
        }
        // self.command_buffer.replace_with(|_| command_buffer)
    }

    pub fn wait_until_completed(&self) {
        let mut old = self.command_buffer.try_write().unwrap();
        match old.status() {
            metal::MTLCommandBufferStatus::NotEnqueued
            | metal::MTLCommandBufferStatus::Enqueued => {
                old.commit();
                old.wait_until_completed();
            }
            _ => {}
        }
        let command_buffer = self.command_queue.new_command_buffer().to_owned();
        *old = command_buffer;
        // self.command_buffer.replace_with(|_| command_buffer)
    }

    pub fn kernels(&self) -> &Kernels {
        &self.kernels
    }

    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    pub fn new_buffer(&self, element_count: usize, dtype: DType) -> Arc<Buffer> {
        let size = (element_count * dtype.size_in_bytes()) as NSUInteger;
        self._new_buffer(size, MTLResourceOptions::StorageModePrivate)
    }

    fn _new_buffer(&self, size: NSUInteger, option: MTLResourceOptions) -> Arc<Buffer> {
        let mut buffers = self.buffers.try_write().unwrap();
        let subbuffers = buffers.entry((size, option)).or_insert(vec![]);

        for sub in &mut *subbuffers {
            if Arc::strong_count(sub) == 1 {
                return sub.clone();
            }
        }
        let new_buffer = self.device.new_buffer(size as NSUInteger, option);
        let new_buffer = Arc::new(new_buffer);
        subbuffers.push(new_buffer.clone());
        new_buffer
    }

    pub fn new_buffer_managed(&self, size: NSUInteger) -> Arc<Buffer> {
        self._new_buffer(size, MTLResourceOptions::StorageModeManaged)
    }

    pub fn new_buffer_with_data<T>(&self, data: &[T]) -> Arc<Buffer> {
        let tmp = self.device.new_buffer_with_data(
            data.as_ptr() as *const core::ffi::c_void,
            core::mem::size_of_val(data) as NSUInteger,
            metal::MTLResourceOptions::StorageModeManaged,
        );
        let real = self._new_buffer(
            core::mem::size_of_val(data) as NSUInteger,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        {
            let command = self.command_buffer();
            let blit = command.new_blit_command_encoder();
            blit.copy_from_buffer(&tmp, 0, &real, 0, tmp.length());
            blit.end_encoding();
        }
        real
    }

    pub fn new_matrix(
        &self,
        (b, m, n): (NSUInteger, NSUInteger, NSUInteger),
        size: NSUInteger,
        type_id: u32,
        dtype: DType,
    ) -> Result<(Matrix, Arc<Buffer>)> {
        let elem_count = (b * m * n) as usize;
        let out_buffer = self.new_buffer(elem_count, dtype);

        let result_descriptor =
            MatrixDescriptor::init_multiple(m, n, b, n * size, m * n * size, type_id);
        let result_matrix = Matrix::init_with_buffer_descriptor(&out_buffer, 0, &result_descriptor)
            .ok_or_else(|| {
                MetalError::from("Failed to create matrix multiplication kernel".to_string())
            })?;
        Ok((result_matrix, out_buffer))
    }
}

#[derive(Debug, Clone)]
pub struct MetalStorage {
    buffer: Arc<metal::Buffer>,
    matrices: Arc<
        RwLock<
            HashMap<
                (
                    NSUInteger,
                    NSUInteger,
                    NSUInteger,
                    bool,
                    NSUInteger,
                    NSUInteger,
                    u32,
                ),
                Matrix,
            >,
        >,
    >,
    device: MetalDevice,
    dtype: DType,
}

impl BackendStorage for MetalStorage {
    type Device = MetalDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let length = self.buffer.length() as usize;
        let size = self.dtype.size_in_bytes();
        if length % size != 0 {
            crate::bail!(
                "The Metal buffer length is not aligned with dtype {:?}",
                self.dtype
            );
        }

        let buffer = self.device.new_buffer_managed(self.buffer.length());
        let command_buffer = self.device.command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_buffer(&self.buffer, 0, &buffer, 0, self.buffer.length());
        blit.end_encoding();
        drop(command_buffer);
        self.device.wait_until_completed();

        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.buffer.read_to_vec(length / size))),
            DType::U32 => Ok(CpuStorage::U32(self.buffer.read_to_vec(length / size))),
            DType::I64 => Ok(CpuStorage::I64(self.buffer.read_to_vec(length / size))),
            DType::F16 => Ok(CpuStorage::F16(self.buffer.read_to_vec(length / size))),
            DType::BF16 => Ok(CpuStorage::BF16(self.buffer.read_to_vec(length / size))),
            DType::F32 => Ok(CpuStorage::F32(self.buffer.read_to_vec(length / size))),
            DType::F64 => Ok(CpuStorage::F64(self.buffer.read_to_vec(length / size))),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();

        let shape = layout.shape();
        let el = shape.elem_count();
        let dtype = self.dtype;

        let buffer = device.new_buffer(el, self.dtype);
        let command_buffer = self.device.command_buffer();
        if layout.is_contiguous() && layout.start_offset() == 0 {
            let name = match self.dtype {
                DType::F32 => "affine_float",
                DType::F16 => "affine_half",
                dtype => crate::bail!("Affine {dtype:?}"),
            };
            candle_metal_kernels::call_affine(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                el,
                &self.buffer,
                &buffer,
                mul as f32,
                add as f32,
            )
            .map_err(MetalError::from)?;
        } else {
            let name = match self.dtype {
                DType::F32 => "affine_float_strided",
                DType::F16 => "affine_half_strided",
                dtype => crate::bail!("Affine {dtype:?}"),
            };
            candle_metal_kernels::call_affine_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * dtype.size_in_bytes(),
                &buffer,
                mul as f32,
                add as f32,
            )
            .map_err(MetalError::from)?;
        }
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        crate::bail!("powf metal")
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        crate::bail!("elu metal")
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        if !(sum_dims.len() == 1
            && sum_dims[0] == layout.shape().rank() - 1
            && layout.is_contiguous()
            && layout.start_offset() == 0
            && layout.stride()[sum_dims[0]] == 1)
        {
            crate::bail!("Non last dim reduce op not supported yet");
        }

        let device = self.device.clone();
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !sum_dims.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in sum_dims.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }

        // The reduction loop requires the shared array to be properly initialized and for
        // this we want the number of threads to be a power of two.
        let (name, check_empty, return_index) = match (op, self.dtype) {
            (ReduceOp::Sum, DType::F32) => ("fast_sum_float", false, false),
            (ReduceOp::Min, DType::F32) => ("fast_min_float", true, false),
            (ReduceOp::Max, DType::F32) => ("fast_max_float", true, false),
            (ReduceOp::ArgMin, DType::F32) => ("fast_argmin_float", true, true),
            (ReduceOp::ArgMax, DType::F32) => ("fast_argmax_float", true, true),
            _ => crate::bail!("Reduce op for non float"),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        let dtype = if return_index { DType::U32 } else { self.dtype };
        if dtype == DType::U32 {
            crate::bail!("Implement return index reduce op");
        }
        let buffer = device.new_buffer(dst_el, dtype);
        let command_buffer = self.device.command_buffer();
        candle_metal_kernels::call_reduce_contiguous(
            &device.device,
            &command_buffer,
            &device.kernels,
            name,
            src_el,
            dst_el,
            &self.buffer,
            layout.start_offset() * self.dtype.size_in_bytes(),
            &buffer,
        )
        .map_err(MetalError::from)?;

        Ok(Self::new(buffer, device, dtype))
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        crate::bail!("cmp metal")
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let device = self.device();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype);
        let command_buffer = device.command_buffer();
        if layout.is_contiguous() {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::F32) => "cast_u32_f32",
                (DType::F32, DType::F16) => "cast_f32_f16",
                (DType::F16, DType::F32) => "cast_f16_f32",
                (left, right) => crate::bail!("to dtype {left:?} - {right:?}"),
            };
            candle_metal_kernels::call_cast_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            let kernel_name = match (self.dtype, dtype) {
                (DType::U32, DType::F32) => "cast_u32_f32_strided",
                (DType::F32, DType::F16) => "cast_f32_f16_strided",
                (DType::F16, DType::F32) => "cast_f16_f32_strided",
                (left, right) => crate::bail!("to dtype {left:?} - {right:?}"),
            };
            candle_metal_kernels::call_cast_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * self.dtype.size_in_bytes(),
                &buffer,
            )
            .map_err(MetalError::from)?;
        }

        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device();
        let dtype = self.dtype;
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype);
        let command_buffer = device.command_buffer();
        if layout.is_contiguous() && layout.start_offset() == 0 {
            use candle_metal_kernels::unary::contiguous;

            let kernel_name = match (B::KERNEL, dtype) {
                ("ucos", DType::F32) => contiguous::cos::FLOAT,
                ("usin", DType::F32) => contiguous::sin::FLOAT,
                ("usqr", DType::F32) => contiguous::sqr::FLOAT,
                ("usqrt", DType::F32) => contiguous::sqrt::FLOAT,
                ("uneg", DType::F32) => contiguous::neg::FLOAT,
                ("uexp", DType::F32) => contiguous::exp::FLOAT,
                ("ulog", DType::F32) => contiguous::log::FLOAT,
                ("ugelu", DType::F32) => contiguous::gelu::FLOAT,
                ("ugelu_erf", DType::F32) => contiguous::gelu_erf::FLOAT,
                ("uerf", DType::F32) => contiguous::erf::FLOAT,
                ("uceil", DType::F32) => contiguous::ceil::FLOAT,
                ("ufloor", DType::F32) => contiguous::floor::FLOAT,
                ("uround", DType::F32) => contiguous::round::FLOAT,
                ("ucos", DType::F16) => contiguous::cos::HALF,
                ("usin", DType::F16) => contiguous::sin::HALF,
                ("usqr", DType::F16) => contiguous::sqr::HALF,
                ("usqrt", DType::F16) => contiguous::sqrt::HALF,
                ("uneg", DType::F16) => contiguous::neg::HALF,
                ("uexp", DType::F16) => contiguous::exp::HALF,
                ("ulog", DType::F16) => contiguous::log::HALF,
                ("ugelu", DType::F16) => contiguous::gelu::HALF,
                ("ugelu_erf", DType::F16) => contiguous::gelu_erf::HALF,
                ("uerf", DType::F16) => contiguous::erf::HALF,
                ("uceil", DType::F16) => contiguous::ceil::HALF,
                ("ufloor", DType::F16) => contiguous::floor::HALF,
                ("uround", DType::F16) => contiguous::round::HALF,
                (name, dtype) => crate::bail!("Match {name} - {dtype:?}"),
            };
            candle_metal_kernels::call_unary_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            use candle_metal_kernels::unary::strided;
            let kernel_name = match (B::KERNEL, dtype) {
                ("ucos", DType::F32) => strided::cos::FLOAT,
                ("usin", DType::F32) => strided::sin::FLOAT,
                ("usqr", DType::F32) => strided::sqr::FLOAT,
                ("usqrt", DType::F32) => strided::sqrt::FLOAT,
                ("uneg", DType::F32) => strided::neg::FLOAT,
                ("uexp", DType::F32) => strided::exp::FLOAT,
                ("ulog", DType::F32) => strided::log::FLOAT,
                ("ugelu", DType::F32) => strided::gelu::FLOAT,
                ("ugelu_erf", DType::F32) => strided::gelu_erf::FLOAT,
                ("uerf", DType::F32) => strided::erf::FLOAT,
                ("uceil", DType::F32) => strided::ceil::FLOAT,
                ("ufloor", DType::F32) => strided::floor::FLOAT,
                ("uround", DType::F32) => strided::round::FLOAT,
                ("ucos", DType::F16) => strided::cos::HALF,
                ("usin", DType::F16) => strided::sin::HALF,
                ("usqr", DType::F16) => strided::sqr::HALF,
                ("usqrt", DType::F16) => strided::sqrt::HALF,
                ("uneg", DType::F16) => strided::neg::HALF,
                ("uexp", DType::F16) => strided::exp::HALF,
                ("ulog", DType::F16) => strided::log::HALF,
                ("ugelu", DType::F16) => strided::gelu::HALF,
                ("ugelu_erf", DType::F16) => strided::gelu_erf::HALF,
                ("uerf", DType::F16) => strided::erf::HALF,
                ("uceil", DType::F16) => strided::ceil::HALF,
                ("ufloor", DType::F16) => strided::floor::HALF,
                ("uround", DType::F16) => strided::round::HALF,
                (name, dtype) => crate::bail!("Match {name} - {dtype:?}"),
            };
            candle_metal_kernels::call_unary_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                layout.dims(),
                &self.buffer,
                layout.stride(),
                layout.start_offset() * self.dtype.size_in_bytes(),
                &buffer,
                0,
            )
            .map_err(MetalError::from)?;
        }
        command_buffer.set_label("unary");
        drop(command_buffer);
        self.device.commit();
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device();
        let dtype = self.dtype;
        let shape = lhs_l.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype);
        let command_buffer = device.command_buffer();
        if (lhs_l.is_contiguous() && lhs_l.start_offset() == 0)
            && (rhs_l.is_contiguous() && rhs_l.start_offset() == 0)
        {
            use candle_metal_kernels::binary::contiguous;

            let kernel_name = match (B::KERNEL, dtype) {
                ("add", DType::F32) => contiguous::add::FLOAT,
                ("badd", DType::F32) => contiguous::add::FLOAT,
                ("sub", DType::F32) => contiguous::sub::FLOAT,
                ("bsub", DType::F32) => contiguous::sub::FLOAT,
                ("mul", DType::F32) => contiguous::mul::FLOAT,
                ("bmul", DType::F32) => contiguous::mul::FLOAT,
                ("div", DType::F32) => contiguous::div::FLOAT,
                ("bdiv", DType::F32) => contiguous::div::FLOAT,
                ("add", DType::F16) => contiguous::add::HALF,
                ("badd", DType::F16) => contiguous::add::HALF,
                ("sub", DType::F16) => contiguous::sub::HALF,
                ("bsub", DType::F16) => contiguous::sub::HALF,
                ("mul", DType::F16) => contiguous::mul::HALF,
                ("bmul", DType::F16) => contiguous::mul::HALF,
                ("div", DType::F16) => contiguous::div::HALF,
                ("bdiv", DType::F16) => contiguous::div::HALF,
                (name, dtype) => crate::bail!("Match {name} - {dtype:?}"),
            };
            candle_metal_kernels::call_binary_contiguous(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                el_count,
                &self.buffer,
                &rhs.buffer,
                &buffer,
            )
            .map_err(MetalError::from)?;
        } else {
            use candle_metal_kernels::binary::strided;

            let kernel_name = match (B::KERNEL, dtype) {
                ("badd", DType::F32) => strided::add::FLOAT,
                ("bsub", DType::F32) => strided::sub::FLOAT,
                ("bmul", DType::F32) => strided::mul::FLOAT,
                ("bdiv", DType::F32) => strided::div::FLOAT,
                ("badd", DType::F16) => strided::add::HALF,
                ("bsub", DType::F16) => strided::sub::HALF,
                ("bmul", DType::F16) => strided::mul::HALF,
                ("bdiv", DType::F16) => strided::div::HALF,
                (name, dtype) => crate::bail!("Match {name} - {dtype:?}"),
            };
            candle_metal_kernels::call_binary_strided(
                &device.device,
                &command_buffer,
                &device.kernels,
                kernel_name,
                lhs_l.dims(),
                &self.buffer,
                lhs_l.stride(),
                lhs_l.start_offset() * self.dtype.size_in_bytes(),
                &rhs.buffer,
                rhs_l.stride(),
                rhs_l.start_offset() * rhs.dtype.size_in_bytes(),
                &buffer,
            )
            .map_err(MetalError::from)?;
        }
        command_buffer.set_label("binary");
        drop(command_buffer);
        self.device.commit();
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device.clone();
        let shape = t_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let dtype = t.dtype;
        let buffer = self.device.new_buffer(el, dtype);
        let command_buffer = self.device.command_buffer();
        candle_metal_kernels::call_where_cond_strided(
            &device.device,
            &command_buffer,
            &device.kernels,
            "where_u8_f32",
            dims,
            &self.buffer,
            (
                layout.stride(),
                layout.start_offset() * self.dtype.size_in_bytes(),
            ),
            &t.buffer,
            (&t_l.stride(), t_l.start_offset() * t.dtype.size_in_bytes()),
            &f.buffer,
            (&f_l.stride(), f_l.start_offset() * f.dtype.size_in_bytes()),
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device, dtype))
    }

    fn conv1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv1D,
    ) -> Result<Self> {
        crate::bail!("conv1d metal")
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        crate::bail!("conv_transpose1d metal")
    }

    fn conv2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConv2D,
    ) -> Result<Self> {
        crate::bail!("conv2d metal")
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        crate::bail!("conv_tranpose2d metal")
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("avg_pool2d metal")
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        crate::bail!("max_pool2d metal")
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("upsample_nearest1d metal")
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        crate::bail!("upsample_nearest2d metal")
    }

    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        crate::bail!("gather metal")
    }

    fn scatter_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        crate::bail!("scatter_add metal")
    }

    fn index_select(&self, ids: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        if !(src_l.is_contiguous()
            && src_l.start_offset() == 0
            && ids_l.is_contiguous()
            && ids_l.start_offset() == 0)
        {
            crate::bail!("Non contiguous index select not implemented");
        }
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let ids_el = ids_l.shape().elem_count();
        let dst_el = ids_el * left_size * right_size;
        let dtype = self.dtype;
        let device = self.device();
        let buffer = device.new_buffer(dst_el, dtype);
        let name = match (ids.dtype, self.dtype) {
            (DType::U32, DType::F32) => "is_u32_f32",
            (DType::U32, DType::F16) => "is_u32_f16",
            (left, right) => crate::bail!("index select metal {left:?} {right:?}"),
        };
        let command_buffer = self.device.command_buffer();
        candle_metal_kernels::call_index_select(
            &device.device,
            &command_buffer,
            &self.device.kernels,
            name,
            src_l.dims(),
            ids_el,
            dim,
            &self.buffer,
            &ids.buffer,
            &buffer,
        )
        .map_err(MetalError::from)?;
        Ok(Self::new(buffer, device.clone(), dtype))
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        crate::bail!("index_add metal")
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        // Create descriptors

        // let start = std::time::Instant::now();

        let (type_id, size) = match self.dtype {
            DType::F32 => (
                metal::mps::MPS_FLOATBIT_ENCODING | 32,
                core::mem::size_of::<f32>() as NSUInteger,
            ),
            DType::F16 => (
                metal::mps::MPS_FLOATBIT_ENCODING | 16,
                core::mem::size_of::<f16>() as NSUInteger,
            ),
            dtype => todo!("Dtype for matmul {dtype:?} is not supported"),
        };

        let lhs_stride = lhs_l.stride();
        let rhs_stride = rhs_l.stride();
        let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
        let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
        let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
        let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
        // The a tensor has dims batching, k, n (rhs)
        let transpose_left = if lhs_m1 == 1 && lhs_m2 == k {
            false
        } else if lhs_m1 == m && lhs_m2 == 1 {
            true
        } else {
            Err(MetalError::MatMulNonContiguous {
                lhs_stride: lhs_stride.to_vec(),
                rhs_stride: rhs_stride.to_vec(),
                mnk: (m, n, k),
            })?
        };
        let transpose_right = if rhs_m1 == 1 && rhs_m2 == n {
            false
        } else if rhs_m1 == k && rhs_m2 == 1 {
            true
        } else {
            Err(MetalError::MatMulNonContiguous {
                lhs_stride: lhs_stride.to_vec(),
                rhs_stride: rhs_stride.to_vec(),
                mnk: (m, n, k),
            })?
        };
        let b = b as NSUInteger;
        let m = m as NSUInteger;
        let n = n as NSUInteger;
        let k = k as NSUInteger;

        let left_matrix = self.matrix(
            (b, m, k),
            transpose_left,
            size,
            lhs_l.start_offset() as NSUInteger * size,
            type_id,
        )?;
        let right_matrix = rhs.matrix(
            (b, k, n),
            transpose_right,
            size,
            rhs_l.start_offset() as NSUInteger * size,
            type_id,
        )?;
        let (result_matrix, out_buffer) =
            self.device
                .new_matrix((b, m, n), size, type_id, self.dtype)?;

        let command_buffer = self.device.command_buffer();

        let alpha = 1.0f64;
        let beta = 0.0f64;
        // Create kernel
        let matrix_multiplication = MatrixMultiplication::init(
            &self.device,
            transpose_left,
            transpose_right,
            m,
            n,
            k,
            alpha,
            beta,
        )
        .ok_or_else(|| {
            MetalError::from("Failed to create matrix multiplication kernel".to_string())
        })?;

        // matrix_multiplication.set_batch_size(b);

        // Encode kernel to command buffer
        matrix_multiplication.encode_to_command_buffer(
            &command_buffer,
            &left_matrix,
            &right_matrix,
            &result_matrix,
        );
        command_buffer.set_label("matmul");
        drop(command_buffer);
        self.device.commit();

        Ok(Self::new(out_buffer, self.device.clone(), self.dtype()))
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let command_buffer = self.device.command_buffer();
        if src_l.is_contiguous() {
            command_buffer.set_label("copy_contiguous");
            let blit = command_buffer.new_blit_command_encoder();
            let src_offset = (src_l.start_offset() * self.dtype.size_in_bytes()) as NSUInteger;
            let dst_offset = (dst_offset * dst.dtype().size_in_bytes()) as NSUInteger;
            blit.copy_from_buffer(
                &self.buffer,
                src_offset,
                dst.buffer(),
                dst_offset,
                self.buffer.length() - src_offset,
            );
            blit.end_encoding();
        } else {
            let src_shape = src_l.shape();
            let el_count = src_shape.elem_count();
            if el_count == 0 {
                return Ok(());
            }
            let kernel_name = match self.dtype {
                DType::F32 => candle_metal_kernels::unary::strided::copy::FLOAT,
                DType::F16 => candle_metal_kernels::unary::strided::copy::HALF,
                DType::BF16 => candle_metal_kernels::unary::strided::copy::BFLOAT,
                DType::U32 => candle_metal_kernels::unary::strided::copy::U32,
                dtype => crate::bail!("copy_strided not implemented for {dtype:?}"),
            };
            candle_metal_kernels::call_unary_strided(
                &self.device.device,
                &command_buffer,
                &self.device.kernels,
                kernel_name,
                src_l.dims(),
                &self.buffer,
                src_l.stride(),
                src_l.start_offset() * self.dtype.size_in_bytes(),
                &dst.buffer,
                dst_offset * dst.dtype.size_in_bytes(),
            )
            .map_err(MetalError::from)?;
            command_buffer.set_label("copy_strided");
        }
        drop(command_buffer);
        self.device.commit();
        Ok(())
    }
}

impl MetalStorage {
    pub fn new(buffer: Arc<Buffer>, device: MetalDevice, dtype: DType) -> Self {
        let matrices = Arc::new(RwLock::new(HashMap::new()));
        Self {
            buffer,
            device,
            dtype,
            matrices,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    fn matrix(
        &self,
        (b, m, n): (NSUInteger, NSUInteger, NSUInteger),
        transpose: bool,
        size: NSUInteger,
        offset: NSUInteger,
        type_id: u32,
    ) -> Result<Matrix> {
        let key = (b, m, n, transpose, size, offset, type_id);

        let mut matrices = self.matrices.try_write().unwrap();
        if let Some(matrix) = matrices.get(&key) {
            Ok(matrix.clone())
        } else {
            let descriptor = if transpose {
                MatrixDescriptor::init_multiple(n, m, b, m * size, m * n * size, type_id)
            } else {
                MatrixDescriptor::init_multiple(m, n, b, n * size, m * n * size, type_id)
            };
            let matrix = Matrix::init_with_buffer_descriptor(&self.buffer, offset, &descriptor)
                .ok_or_else(|| {
                    MetalError::from("Failed to create matrix multiplication kernel".to_string())
                })?;
            matrices.insert(key, matrix.clone());
            Ok(matrix)
        }
    }
}

impl BackendDevice for MetalDevice {
    type Storage = MetalStorage;

    fn new(ordinal: usize) -> Result<Self> {
        let device = metal::Device::all().swap_remove(ordinal);

        // let capture = metal::CaptureManager::shared();
        // let descriptor = metal::CaptureDescriptor::new();
        // descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        // descriptor.set_capture_device(&device);
        // let mut dir = std::env::current_dir()?;
        // dir.push("out.gputrace");
        // descriptor.set_output_url(dir);

        // capture
        //     .start_capture(&descriptor)
        //     .map_err(MetalError::from)?;

        let command_queue = device.new_command_queue();
        let command_buffer = Arc::new(RwLock::new(command_queue.new_command_buffer().to_owned()));
        let kernels = Arc::new(Kernels::new());
        let buffers = Arc::new(RwLock::new(HashMap::new()));
        Ok(Self {
            device,
            command_queue,
            command_buffer,
            buffers,
            kernels,
        })
    }

    fn set_seed(&self, _seed: u64) -> Result<()> {
        crate::bail!("set_seed")
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Metal {
            gpu_id: self.registry_id() as usize,
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.device.registry_id() == rhs.device.registry_id()
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<MetalStorage> {
        let buffer = self.new_buffer(shape.elem_count(), dtype);
        Ok(MetalStorage::new(buffer, self.clone(), dtype))
    }

    fn ones_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        // TODO Is there a faster way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.ones_impl(shape, dtype)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let buffer = match storage {
            CpuStorage::U8(storage) => self.new_buffer_with_data(storage),
            CpuStorage::U32(storage) => self.new_buffer_with_data(storage),
            CpuStorage::I64(storage) => self.new_buffer_with_data(storage),
            CpuStorage::BF16(storage) => self.new_buffer_with_data(storage),
            CpuStorage::F16(storage) => self.new_buffer_with_data(storage),
            CpuStorage::F32(storage) => self.new_buffer_with_data(storage),
            CpuStorage::F64(storage) => self.new_buffer_with_data(storage),
        };
        Ok(Self::Storage::new(
            buffer.into(),
            self.clone(),
            storage.dtype(),
        ))
    }

    fn rand_uniform(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        // TODO is there a better way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.rand_uniform(shape, dtype, mean, stddev)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn rand_normal(
        &self,
        shape: &Shape,
        dtype: DType,
        mean: f64,
        stddev: f64,
    ) -> Result<Self::Storage> {
        // TODO is there a better way ?
        let cpu_storage = crate::cpu_backend::CpuDevice.rand_normal(shape, dtype, mean, stddev)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }
}
