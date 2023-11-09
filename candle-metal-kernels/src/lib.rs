use metal::{
    Buffer, CommandBufferRef, CompileOptions, ComputePipelineDescriptor, Device, Function, Library,
    MTLSize,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::RwLock;

const AFFINE: &str = include_str!("affine.metal");
const INDEXING: &str = include_str!("indexing.metal");
const UNARY: &str = include_str!("unary.metal");
const BINARY: &str = include_str!("binary.metal");
const CAST: &str = include_str!("cast.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Affine,
    Indexing,
    Unary,
    Binary,
    Cast,
}

macro_rules! ops{
    ($($name:ident),+) => {

        pub mod contiguous {
        pub struct Kernel(pub(crate) &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_float"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_half"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bfloat"));
        }
        )+
        }

        pub mod strided {
        pub struct Kernel(pub(crate) &'static str);
        $(
        pub mod $name {
            use super::Kernel;
            pub const FLOAT: Kernel = Kernel(concat!(stringify!($name), "_float_strided"));
            pub const HALF: Kernel = Kernel(concat!(stringify!($name), "_half_strided"));
            pub const BFLOAT: Kernel = Kernel(concat!(stringify!($name), "_bfloat_strided"));
        }
        )+
        }
    };
}

pub mod unary {
    ops!(cos, sin, exp, sqr, sqrt, neg, copy);
}
pub mod binary {
    ops!(add, sub, mul, div);
}

// static LIBRARY_SOURCES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
//     let mut l = HashMap::new();
//     l.insert("affine", AFFINE);
//     l.insert("indexing", INDEXING);
//     l.insert("unary", UNARY);
//     l
// });
//
#[derive(thiserror::Error, Debug)]
pub enum MetalKernelError {
    #[error("Could not lock kernel map: {0}")]
    LockError(String),
    #[error("Error while loading library: {0}")]
    LoadLibraryError(String),
    #[error("Error while loading function: {0}")]
    LoadFunctionError(String),
}

impl<T> From<std::sync::PoisonError<T>> for MetalKernelError {
    fn from(e: std::sync::PoisonError<T>) -> Self {
        Self::LockError(e.to_string())
    }
}

type KernelMap<T> = HashMap<&'static str, T>;
type Libraries = HashMap<Source, Library>;
type Functions = KernelMap<Function>;

#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    funcs: RwLock<Functions>,
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let funcs = RwLock::new(Functions::new());
        Self { libraries, funcs }
    }

    // pub fn init(device: &Device) -> Result<Self, MetalKernelError> {
    //     let kernels = Self::new();
    //     kernels.load_libraries(device)?;
    //     Ok(kernels)
    // }

    // fn load_libraries(&self, device: &Device) -> Result<(), MetalKernelError> {
    //     for name in LIBRARY_SOURCES.keys() {
    //         self.load_library(device, name)?;
    //     }
    //     Ok(())
    // }

    fn get_library_source(&self, source: Source) -> &'static str {
        // LIBRARY_SOURCES.get(name).cloned()
        match source {
            Source::Affine => AFFINE,
            Source::Unary => UNARY,
            Source::Binary => BINARY,
            Source::Indexing => INDEXING,
            Source::Cast => CAST,
        }
    }

    pub fn load_library(
        &self,
        device: &Device,
        source: Source,
    ) -> Result<Library, MetalKernelError> {
        let mut libraries = self.libraries.write()?;
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            let source_content = self.get_library_source(source);
            let lib = device
                .new_library_with_source(source_content, &CompileOptions::new())
                .map_err(|e| MetalKernelError::LoadLibraryError(e.to_string()))?;
            libraries.insert(source, lib.clone());
            Ok(lib)
        }
    }

    pub fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
    ) -> Result<Function, MetalKernelError> {
        let mut funcs = self.funcs.write()?;
        if let Some(func) = funcs.get(name) {
            Ok(func.clone())
        } else {
            let func = self
                .load_library(device, source)?
                .get_function(name, None)
                .map_err(|e| MetalKernelError::LoadFunctionError(e.to_string()))?;
            funcs.insert(name, func.clone());
            Ok(func)
        }
    }
}

pub fn call_unary_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: unary::contiguous::Kernel,
    length: usize,
    input: &Buffer,
    output: &mut Buffer,
) -> Result<(), MetalKernelError> {
    // println!("Kernel {:?}", kernel_name.0);
    // assert_eq!(input.length(), output.length());
    let func = kernels.load_function(device, Source::Unary, kernel_name.0)?;
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&func));

    let pipeline = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_bytes(0, 4, void_ptr(&length));
    encoder.set_buffer(1, Some(&input), 0);
    encoder.set_buffer(2, Some(&output), 0);

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), length as u64);
    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}
pub fn call_unary_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: unary::strided::Kernel,
    shape: &[usize],
    input: &Buffer,
    strides: &[usize],
    offset: usize,
    output: &mut Buffer,
    output_offset: usize,
) -> Result<(), MetalKernelError> {
    let func = kernels.load_function(device, Source::Unary, name.0)?;
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&func));

    let pipeline = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    let num_dims: usize = shape.len() as usize;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();
    encoder.set_bytes(0, std::mem::size_of::<usize>() as u64, void_ptr(&length));
    encoder.set_bytes(1, std::mem::size_of::<usize>() as u64, void_ptr(&num_dims));
    encoder.set_bytes(
        2,
        (shape.len() * std::mem::size_of::<usize>()) as u64,
        shape.as_ptr() as *const c_void,
    );
    encoder.set_bytes(
        3,
        (strides.len() * std::mem::size_of::<usize>()) as u64,
        strides.as_ptr() as *const c_void,
    );

    encoder.set_buffer(4, Some(&input), offset as u64);
    encoder.set_buffer(5, Some(&output), output_offset as u64);

    let width = output.length();

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width: std::cmp::min(pipeline.max_total_threads_per_threadgroup(), width),
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_binary_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: binary::contiguous::Kernel,
    length: usize,
    left: &Buffer,
    right: &Buffer,
    output: &mut Buffer,
) -> Result<(), MetalKernelError> {
    // println!("Kernel {:?}", kernel_name.0);
    // assert_eq!(input.length(), output.length());
    let func = kernels.load_function(device, Source::Binary, kernel_name.0)?;
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&func));

    let pipeline = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_bytes(0, 4, void_ptr(&length));
    encoder.set_buffer(1, Some(&left), 0);
    encoder.set_buffer(2, Some(&right), 0);
    encoder.set_buffer(3, Some(&output), 0);

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), length as u64);
    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_binary_strided(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    name: binary::strided::Kernel,
    shape: &[usize],
    left_input: &Buffer,
    left_strides: &[usize],
    left_offset: usize,
    right_input: &Buffer,
    right_strides: &[usize],
    right_offset: usize,
    output: &mut Buffer,
) -> Result<(), MetalKernelError> {
    let func = kernels.load_function(device, Source::Binary, name.0)?;
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&func));

    let pipeline = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    let num_dims: usize = shape.len() as usize;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();
    encoder.set_bytes(0, std::mem::size_of::<usize>() as u64, void_ptr(&length));
    encoder.set_bytes(1, std::mem::size_of::<usize>() as u64, void_ptr(&num_dims));
    encoder.set_bytes(
        2,
        (shape.len() * std::mem::size_of::<usize>()) as u64,
        shape.as_ptr() as *const c_void,
    );
    encoder.set_bytes(
        3,
        (left_strides.len() * std::mem::size_of::<usize>()) as u64,
        left_strides.as_ptr() as *const c_void,
    );
    encoder.set_bytes(
        4,
        (right_strides.len() * std::mem::size_of::<usize>()) as u64,
        right_strides.as_ptr() as *const c_void,
    );

    encoder.set_buffer(5, Some(&left_input), left_offset as u64);
    encoder.set_buffer(6, Some(&right_input), right_offset as u64);
    encoder.set_buffer(7, Some(&output), 0);

    let width = output.length();

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width: std::cmp::min(pipeline.max_total_threads_per_threadgroup(), width),
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn call_cast_contiguous(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    input: &Buffer,
    output: &mut Buffer,
) -> Result<(), MetalKernelError> {
    // println!("Kernel {:?}", kernel_name.0);
    // assert_eq!(input.length(), output.length());
    let func = kernels.load_function(device, Source::Cast, kernel_name)?;
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&func));

    let pipeline = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_bytes(0, 4, void_ptr(&length));
    encoder.set_buffer(1, Some(&input), 0);
    encoder.set_buffer(2, Some(&output), 0);

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), length as u64);
    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

pub fn void_ptr<T>(v: &T) -> *const c_void {
    (v as *const T).cast()
}

pub fn call_affine(
    device: &Device,
    command_buffer: &CommandBufferRef,
    kernels: &Kernels,
    size: usize,
    input: &Buffer,
    output: &mut Buffer,
    mul: f32,
    add: f32,
) -> Result<(), MetalKernelError> {
    let func = kernels.load_function(device, Source::Affine, "affine_float")?;
    let pipeline_state_descriptor = ComputePipelineDescriptor::new();
    pipeline_state_descriptor.set_compute_function(Some(&func));

    let pipeline = device
        .new_compute_pipeline_state_with_function(
            pipeline_state_descriptor.compute_function().unwrap(),
        )
        .unwrap();

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_bytes(0, core::mem::size_of::<usize>() as u64, void_ptr(&size));
    encoder.set_bytes(1, core::mem::size_of::<f32>() as u64, void_ptr(&mul));
    encoder.set_bytes(2, core::mem::size_of::<f32>() as u64, void_ptr(&add));
    encoder.set_buffer(3, Some(&input), 0);
    encoder.set_buffer(4, Some(&output), 0);

    let thread_group_count = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size as u64);
    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    encoder.end_encoding();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize, NSUInteger};
    use std::mem;

    fn device() -> Device {
        Device::system_default().unwrap()
    }

    fn approx(v: Vec<f32>, digits: i32) -> Vec<f32> {
        let b = 10f32.powi(digits);
        v.iter().map(|t| f32::round(t * b) / b).collect()
    }

    fn approx_f16(v: Vec<f16>, digits: i32) -> Vec<f32> {
        let b = 10f32.powi(digits);
        v.iter().map(|t| f32::round(t.to_f32() * b) / b).collect()
    }

    fn run<T: Clone>(v: &[T], name: unary::contiguous::Kernel) -> Vec<T> {
        let device = device();
        let kernels = Kernels::new();
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let options = MTLResourceOptions::StorageModeManaged;
        let input = device.new_buffer_with_data(
            v.as_ptr() as *const core::ffi::c_void,
            (v.len() * core::mem::size_of::<T>()) as u64,
            options,
        );
        let mut output = device.new_buffer((v.len() * core::mem::size_of::<T>()) as u64, options);
        call_unary_contiguous(
            &device,
            &command_buffer,
            &kernels,
            name,
            v.len(),
            &input,
            &mut output,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        output.read_to_vec::<T>(v.len())
    }

    fn run_binary<T: Clone>(x: &[T], y: &[T], name: binary::contiguous::Kernel) -> Vec<T> {
        let device = device();
        let kernels = Kernels::new();
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let options = MTLResourceOptions::StorageModeManaged;
        let left = device.new_buffer_with_data(
            x.as_ptr() as *const core::ffi::c_void,
            (x.len() * core::mem::size_of::<T>()) as u64,
            options,
        );
        let right = device.new_buffer_with_data(
            y.as_ptr() as *const core::ffi::c_void,
            (y.len() * core::mem::size_of::<T>()) as u64,
            options,
        );
        let mut output = device.new_buffer((x.len() * core::mem::size_of::<T>()) as u64, options);
        call_binary_contiguous(
            &device,
            &command_buffer,
            &kernels,
            name,
            x.len(),
            &left,
            &right,
            &mut output,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        output.read_to_vec::<T>(x.len())
    }

    fn run_strided<T: Clone>(
        v: &[T],
        kernel: unary::strided::Kernel,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Vec<T> {
        let device = device();
        let options = MTLResourceOptions::StorageModeManaged;
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let input = device.new_buffer_with_data(
            v.as_ptr() as *const core::ffi::c_void,
            (v.len() * core::mem::size_of::<T>()) as u64,
            options,
        );
        let mut output = device.new_buffer((v.len() * core::mem::size_of::<T>()) as u64, options);
        let kernels = Kernels::new();
        call_unary_strided(
            &device,
            &command_buffer,
            &kernels,
            kernel,
            shape,
            &input,
            strides,
            offset,
            &mut output,
            0,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        output.read_to_vec::<T>(v.len())
    }

    #[test]
    fn cos_f32() {
        let v = vec![1.0f32, 2.0, 3.0];
        let results = run(&v, unary::contiguous::cos::FLOAT);
        let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
        assert_eq!(approx(results, 4), vec![0.5403, -0.4161, -0.99]);
        assert_eq!(approx(expected, 4), vec![0.5403, -0.4161, -0.99]);

        let v = vec![1.0f32; 10_000];
        let results = run(&v, unary::contiguous::cos::FLOAT);
        let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
        assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
        assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
    }

    #[test]
    fn cos_f32_strided() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Shape = [6], strides = [1];
        let shape = vec![6];
        let strides = vec![1];
        let offset = 0;
        let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
        let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
        assert_eq!(
            approx(results, 4),
            vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
        );
        assert_eq!(
            approx(expected, 4),
            vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
        );

        // Contiguous
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![3, 2];
        let strides = vec![2, 1];
        let offset = 0;
        let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
        let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
        assert_eq!(
            approx(results, 4),
            vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
        );
        assert_eq!(
            approx(expected, 4),
            vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
        );

        // Transposed
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![3, 2];
        let strides = vec![1, 3];
        let offset = 0;
        let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
        let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
        assert_eq!(
            approx(results, 4),
            vec![0.5403, -0.6536, -0.4161, 0.2837, -0.99, 0.9602]
        );
        assert_eq!(
            approx(expected, 4),
            vec![0.5403, -0.4161, -0.99, -0.6536, 0.2837, 0.9602]
        );

        // Very large
        let v = vec![1.0f32; 10_000];
        let shape = vec![2, 5_000];
        let strides = vec![2, 1];
        let offset = 0;
        let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
        let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
        assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
        assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
    }

    #[test]
    fn binary_add_f32() {
        let left = vec![1.0f32, 2.0, 3.0];
        let right = vec![2.0f32, 3.1, 4.2];
        let results = run_binary(&left, &right, binary::contiguous::add::FLOAT);
        let expected: Vec<_> = left
            .iter()
            .zip(right.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        assert_eq!(approx(results, 4), vec![3.0f32, 5.1, 7.2]);
        assert_eq!(approx(expected, 4), vec![3.0f32, 5.1, 7.2]);
    }

    fn cast<T: Clone, U: Clone>(v: &[T], name: &'static str) -> Vec<U> {
        let device = device();
        let kernels = Kernels::new();
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let options = MTLResourceOptions::StorageModeManaged;
        let input = device.new_buffer_with_data(
            v.as_ptr() as *const core::ffi::c_void,
            (v.len() * core::mem::size_of::<T>()) as u64,
            options,
        );
        let mut output = device.new_buffer((v.len() * core::mem::size_of::<U>()) as u64, options);
        call_cast_contiguous(
            &device,
            &command_buffer,
            &kernels,
            name,
            v.len(),
            &input,
            &mut output,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        output.read_to_vec::<U>(v.len())
    }

    #[test]
    fn cast_u32_f32() {
        let v = vec![1u32, 2, 3];
        let results = cast(&v, "cast_u32_f32");
        let expected: Vec<_> = v.iter().map(|&v| v as f32).collect();
        assert_eq!(approx(results, 4), vec![1.0f32, 2.0, 3.0]);
        assert_eq!(approx(expected, 4), vec![1.0f32, 2.0, 3.0]);

        let v = vec![1.0f32; 10_000];
        let results = run(&v, unary::contiguous::cos::FLOAT);
        let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
        assert_eq!(approx(results, 4), vec![0.5403; 10_000]);
        assert_eq!(approx(expected, 4), vec![0.5403; 10_000]);
    }

    fn run_affine<T: Clone>(v: &[T], mul: f64, add: f64) -> Vec<T> {
        let device = device();
        let kernels = Kernels::new();
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let options = MTLResourceOptions::StorageModeManaged;

        let input = device.new_buffer_with_data(
            v.as_ptr() as *const core::ffi::c_void,
            (v.len() * core::mem::size_of::<T>()) as u64,
            options,
        );
        let mut output = device.new_buffer((v.len() * core::mem::size_of::<T>()) as u64, options);

        let size = v.len();

        call_affine(
            &device,
            &command_buffer,
            &kernels,
            size,
            &input,
            &mut output,
            mul as f32,
            add as f32,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        output.read_to_vec::<T>(v.len())
    }

    #[test]
    fn affine() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mul = 1.5;
        let add = 1.1;
        let result = run_affine(&input, mul, add);
        assert_eq!(result, vec![2.6, 4.1, 5.6, 7.1, 8.6, 10.1, 11.6, 13.1]);

        let input = [1.0f32; 40_000];
        let mul = 1.5;
        let add = 1.1;
        let result = run_affine(&input, mul, add);
        assert_eq!(result, vec![2.6; 40_000]);
    }

    #[test]
    fn index_add() {
        let device = Device::system_default().expect("no device found");

        let options = CompileOptions::new();
        let library = device.new_library_with_source(INDEXING, &options).unwrap();

        let left = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let right = [1.0f32; 15];
        let index = [0u32, 4, 2];
        let ids_dim_size = index.len() as u32;
        let dst_dim_size: u32 = 15;
        let left_size: u32 = 3;
        let right_size: u32 = 3;

        let function = library.get_function("ia_u32_f32", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();
        let options = MTLResourceOptions::StorageModeManaged;

        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let ids_size = (index.len() * mem::size_of::<u32>()) as NSUInteger;
        let input_size = (left.len() * mem::size_of::<f32>()) as NSUInteger;
        let output_size = (right.len() * mem::size_of::<f32>()) as NSUInteger;

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_threadgroup_memory_length(0, output_size as NSUInteger);

        let index_buffer = device.new_buffer_with_data(void_ptr(&index), ids_size, options);
        let inputs_buffer = device.new_buffer_with_data(void_ptr(&left), input_size, options);
        let outputs_buffer = device.new_buffer_with_data(void_ptr(&right), output_size, options);

        encoder.set_buffer(0, Some(&index_buffer), 0);
        encoder.set_buffer(1, Some(&inputs_buffer), 0);
        encoder.set_buffer(2, Some(&outputs_buffer), 0);

        encoder.set_bytes(3, 4, void_ptr(&ids_dim_size));
        encoder.set_bytes(4, 4, void_ptr(&left_size));
        encoder.set_bytes(5, 4, void_ptr(&dst_dim_size));
        encoder.set_bytes(6, 4, void_ptr(&right_size));

        let grid_size = MTLSize {
            width: right.len() as NSUInteger,
            height: 1,
            depth: 1,
        };

        let thread_group_size = MTLSize {
            width: pipeline.max_total_threads_per_threadgroup(),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(grid_size, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let expected = vec![
            2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 8.0, 9.0, 10.0, 1.0, 1.0, 1.0, 5.0, 6.0, 7.0,
        ];
        let result = outputs_buffer.read_to_vec::<f32>(right.len());
        assert_eq!(result, expected);
    }

    #[test]
    fn cos_f16() {
        let v: Vec<f16> = [1.0f32, 2.0, 3.0]
            .iter()
            .map(|v| f16::from_f32(*v))
            .collect();
        let results = run(&v, unary::contiguous::cos::HALF);
        let expected: Vec<f16> = v.iter().map(|v| f16::from_f32(v.to_f32().cos())).collect();
        assert_eq!(approx_f16(results, 4), vec![0.54, -0.4165, -0.9902]);
        assert_eq!(approx_f16(expected, 4), vec![0.5405, -0.4163, -0.9902]);
    }
}
