use super::*;
use half::f16;
use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize, NSUInteger};

fn new_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let options = MTLResourceOptions::StorageModeManaged;
    let ptr = data.as_ptr() as *const core::ffi::c_void;
    let size = (data.len() * std::mem::size_of::<T>()) as u64;
    device.new_buffer_with_data(ptr, size, options)
}

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
    let input = new_buffer(&device, v);
    let mut output = new_buffer(&device, v);
    call_unary_contiguous(
        &device,
        command_buffer,
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
    let left = new_buffer(&device, x);
    let right = new_buffer(&device, y);
    let mut output = device.new_buffer(std::mem::size_of_val(x) as u64, options);
    call_binary_contiguous(
        &device,
        command_buffer,
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
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let mut output = new_buffer(&device, v);
    let kernels = Kernels::new();
    call_unary_strided(
        &device,
        command_buffer,
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
fn cos_strided_random() {
    let v: Vec<_> = (0..10_000).map(|_| rand::random::<f32>()).collect();
    let shape = vec![5_000, 2];
    let strides = vec![1, 5_000];
    let offset = 0;
    let results = run_strided(&v, unary::strided::cos::FLOAT, &shape, &strides, offset);
    let expected: Vec<_> = v.iter().map(|v| v.cos()).collect();
    assert_eq!(approx(vec![results[0]], 4), approx(vec![expected[0]], 4));
    assert_eq!(
        approx(vec![results[1]], 4),
        approx(vec![expected[5_000]], 4)
    );
    assert_eq!(approx(vec![results[2]], 4), approx(vec![expected[1]], 4));
    assert_eq!(
        approx(vec![results[3]], 4),
        approx(vec![expected[5_001]], 4)
    );
    assert_eq!(
        approx(vec![results[5_000]], 4),
        approx(vec![expected[2_500]], 4)
    );
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
    let input = new_buffer(&device, v);
    let mut output = new_buffer(&device, v);

    call_cast_contiguous(
        &device,
        command_buffer,
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

    let input = new_buffer(&device, v);
    let mut output = new_buffer(&device, v);

    let size = v.len();

    call_affine(
        &device,
        command_buffer,
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
fn index_select() {
    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let ids = [0u32, 4, 2];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &ids, dim);
    assert_eq!(result, vec![1.0f32, 2.0, 9.0, 10.0, 5.0, 6.0]);

    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [2, 5];
    let ids = [0u32, 1, 0];
    let dim = 0;
    let result = run_index_select(&embedding, &shape, &ids, dim);
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0f32, 2.0, 3.0, 4.0, 5.0]
    );

    let embedding = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let shape = [5, 2];
    let ids = [0u32, 1, 0];
    let dim = 1;
    let result = run_index_select(&embedding, &shape, &ids, dim);
    assert_eq!(
        result,
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0f32, 2.0, 3.0, 4.0, 5.0]
    );
}

fn run_index_select<T: Clone, I: Clone + std::fmt::Debug>(
    embeddings: &[T],
    shape: &[usize],
    ids: &[I],
    dim: usize,
) -> Vec<T> {
    let device = Device::system_default().expect("no device found");

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let embeddings_buffer = new_buffer(&device, &embeddings);
    let ids_buffer = new_buffer(&device, &ids);

    let left_size: usize = shape[..dim].iter().product();
    let right_size: usize = shape[dim + 1..].iter().product();
    let dst_el = ids.len() * left_size * right_size;
    let mut dst_buffer = new_buffer(&device, &vec![0.0f32; dst_el]);

    let kernels = Kernels::new();
    call_index_select(
        &device,
        &command_buffer,
        &kernels,
        "is_u32_f32",
        shape,
        ids.len(),
        dim,
        &embeddings_buffer,
        &ids_buffer,
        &mut dst_buffer,
    )
    .unwrap();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    dst_buffer.read_to_vec::<T>(dst_el)
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

    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    let index_buffer = new_buffer(&device, &index);
    let inputs_buffer = new_buffer(&device, &left);
    let outputs_buffer = new_buffer(&device, &right);

    set_params!(
        encoder,
        (
            &index_buffer,
            &inputs_buffer,
            &outputs_buffer,
            ids_dim_size,
            left_size,
            dst_dim_size,
            right_size
        )
    );

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
    assert_eq!(approx_f16(results, 4), vec![0.5405, -0.4163, -0.9902]);
    assert_eq!(approx_f16(expected, 4), vec![0.5405, -0.4163, -0.9902]);
}

fn run_reduce<T: Clone>(v: &[T], out_length: usize, name: &'static str) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);

    let options = MTLResourceOptions::StorageModeManaged;
    let mut output = device.new_buffer((out_length * core::mem::size_of::<T>()) as u64, options);
    call_reduce_contiguous(
        &device,
        command_buffer,
        &kernels,
        name,
        v.len(),
        out_length,
        &input,
        &mut output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    output.read_to_vec::<T>(out_length)
}

fn run_softmax<T: Clone + std::fmt::Debug>(v: &[T], last_dim: usize, name: &'static str) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let input = new_buffer(&device, v);
    let mut output = new_buffer(&device, v);
    call_last_softmax(
        &device,
        command_buffer,
        &kernels,
        name,
        v.len(),
        last_dim,
        &input,
        &mut output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    output.read_to_vec::<T>(v.len())
}

#[test]
fn reduce_sum() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out_length = 1;

    let results = run_reduce(&v, out_length, "fast_sum_float");
    assert_eq!(approx(results, 4), vec![21.0]);
}

#[test]
fn reduce_sum2() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let out_length = 2;

    let results = run_reduce(&v, out_length, "fast_sum_float");
    assert_eq!(approx(results, 4), vec![6.0, 15.0]);
}

#[test]
fn softmax() {
    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_float");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let v = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let last_dim = 6;
    let results = run_softmax(&v, last_dim, "softmax_float");
    assert_eq!(
        approx(results, 4),
        vec![0.0043, 0.0116, 0.0315, 0.0858, 0.2331, 0.6337]
    );

    let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let last_dim = 3;
    let results = run_softmax(&v, last_dim, "softmax_float");
    assert_eq!(
        approx(results, 4),
        vec![0.0900, 0.2447, 0.6652, 0.0900, 0.2447, 0.6652]
    );
}

fn run_where_cond<I: Clone, T: Clone>(
    shape: &[usize],
    cond: &[I],
    (cond_stride, cond_offset): (Vec<usize>, usize),
    left_true: &[T],
    (left_stride, left_offset): (Vec<usize>, usize),
    right_false: &[T],
    (_right_stride, _right_offset): (Vec<usize>, usize),
    name: &'static str,
) -> Vec<T> {
    let device = device();
    let kernels = Kernels::new();
    let command_queue = device.new_command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let options = MTLResourceOptions::StorageModeManaged;

    let length = cond.len();
    let cond = device.new_buffer_with_data(
        cond.as_ptr() as *const core::ffi::c_void,
        std::mem::size_of_val(cond) as u64,
        options,
    );
    let left = device.new_buffer_with_data(
        left_true.as_ptr() as *const core::ffi::c_void,
        (length * core::mem::size_of::<T>()) as u64,
        options,
    );
    let right = device.new_buffer_with_data(
        right_false.as_ptr() as *const core::ffi::c_void,
        (length * core::mem::size_of::<T>()) as u64,
        options,
    );

    let mut output = device.new_buffer((length * core::mem::size_of::<T>()) as u64, options);
    call_where_cond_strided(
        &device,
        command_buffer,
        &kernels,
        name,
        shape,
        &cond,
        (&cond_stride, cond_offset),
        &left,
        (&left_stride, left_offset),
        &right,
        (&cond_stride, cond_offset),
        &mut output,
    )
    .unwrap();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    output.read_to_vec::<T>(length)
}

#[test]
fn where_cond() {
    let shape = vec![6];
    let cond = vec![0u8, 1, 0, 0, 1, 1];
    let cond_l = (vec![1], 0);
    let left_true = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let left_l = (vec![1], 0);
    let right_false = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let right_l = (vec![1], 0);
    let results = run_where_cond(
        &shape,
        &cond,
        cond_l,
        &left_true,
        left_l,
        &right_false,
        right_l,
        "where_u8_f32",
    );
    assert_eq!(approx(results, 4), vec![-1.0f32, 2.0, -3.0, -4.0, 5.0, 6.0]);
}
