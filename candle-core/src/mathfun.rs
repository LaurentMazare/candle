


#[inline]
pub fn vs_tanh_inplace(y: &mut [f32]) {
    unsafe { mathfun::vs_tanh(y.len(), y.as_ptr(), y.as_mut_ptr()) }
}

#[inline]
pub fn vs_gelu(vs: &[f32], ys: &mut [f32]) {
    for (&v, y) in vs.iter().zip(ys.iter_mut()) {
        *y = (2.0f32 / std::f32::consts::PI).sqrt() * v * (1.0 + 0.044715 * v * v)
    }
    vs_tanh_inplace(ys);
    for (&v, y) in vs.iter().zip(ys.iter_mut()) {
        *y = 0.5 * v * (1.0 + *y)
    }
}


#[inline]
pub fn vs_exp_inplace(y: &mut [f32]) {
    unsafe { mathfun::vs_exp(y.len(), y.as_ptr(), y.as_mut_ptr()) }
}

#[inline]
pub fn vs_silu(vs: &[f32], ys: &mut [f32]) {
    for (&v, y) in vs.iter().zip(ys.iter_mut()) {
        *y = -v
    }
    vs_exp_inplace(ys);
    for (&v, y) in vs.iter().zip(ys.iter_mut()) {
        *y = v / (1.0 + *y)
    }
}

#[inline]
pub fn vs_tanh(a: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let y_len = y.len();
    if a_len != y_len {
        panic!("a and y have different lengths {a_len} <> {y_len}")
    }
    unsafe { mathfun::vs_tanh(a_len, a.as_ptr(), y.as_mut_ptr()) }
}


#[inline]
pub fn vs_sin(a: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let y_len = y.len();
    if a_len != y_len {
        panic!("a and y have different lengths {a_len} <> {y_len}")
    }
    unsafe { mathfun::vs_sin(a_len, a.as_ptr(), y.as_mut_ptr()) }
}

#[inline]
pub fn vs_exp(a: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let y_len = y.len();
    if a_len != y_len {
        panic!("a and y have different lengths {a_len} <> {y_len}")
    }
    unsafe { mathfun::vs_exp(a_len, a.as_ptr(), y.as_mut_ptr()) }
}

#[inline]
pub fn vs_cos(a: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let y_len = y.len();
    if a_len != y_len {
        panic!("a and y have different lengths {a_len} <> {y_len}")
    }
    unsafe { mathfun::vs_cos(a_len, a.as_ptr(), y.as_mut_ptr()) }
}


#[inline]
pub fn vs_ln(a: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let y_len = y.len();
    if a_len != y_len {
        panic!("a and y have different lengths {a_len} <> {y_len}")
    }
    unsafe { mathfun::vs_ln(a_len, a.as_ptr(), y.as_mut_ptr()) }
}

#[inline]
pub fn vs_sqrt(a: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let y_len = y.len();
    if a_len != y_len {
        panic!("a and y have different lengths {a_len} <> {y_len}")
    }
    unsafe { mathfun::vs_sqrt(a_len, a.as_ptr(), y.as_mut_ptr()) }
}

#[inline]
pub fn vs_add(a: &[f32], b: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let y_len = y.len();
    if a_len != b_len || a_len != y_len {
        panic!("a, b, and y have different lengths {a_len} <> {b_len} <> {y_len}")
    }
    for i in 0..a_len {
        y[i] = a[i] + b[i]
    }
}

#[inline]
pub fn vs_div(a: &[f32], b: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let y_len = y.len();
    if a_len != b_len || a_len != y_len {
        panic!("a, b, and y have different lengths {a_len} <> {b_len} <> {y_len}")
    }
    for i in 0..a_len {
        y[i] = a[i] / b[i]
    }
}

#[inline]
pub fn vs_sub(a: &[f32], b: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let y_len = y.len();
    if a_len != b_len || a_len != y_len {
        panic!("a, b, and y have different lengths {a_len} <> {b_len} <> {y_len}")
    }
    for i in 0..a_len {
        y[i] = a[i] - b[i]
    }
}

#[inline]
pub fn vs_mul(a: &[f32], b: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let y_len = y.len();
    if a_len != b_len || a_len != y_len {
        panic!("a, b, and y have different lengths {a_len} <> {b_len} <> {y_len}")
    }
    for i in 0..a_len {
        y[i] = a[i] * b[i]
    }
}

#[inline]
pub fn vs_min(a: &[f32], b: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let y_len = y.len();
    if a_len != b_len || a_len != y_len {
        panic!("a, b, and y have different lengths {a_len} <> {b_len} <> {y_len}")
    }
    for i in 0..a_len {
        y[i] = a[i].min(b[i])
    }
}

#[inline]
pub fn vs_max(a: &[f32], b: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let b_len = b.len();
    let y_len = y.len();
    if a_len != b_len || a_len != y_len {
        panic!("a, b, and y have different lengths {a_len} <> {b_len} <> {y_len}")
    }
    for i in 0..a_len {
        y[i] = a[i].max(b[i])
    }
}

#[inline]
pub fn vs_sqr(a: &[f32], y: &mut [f32]) {
    let a_len = a.len();
    let y_len = y.len();
    if a_len != y_len {
        panic!("a and y have different lengths {a_len} <> {y_len}")
    }
    for i in 0..a_len {
        y[i] = a[i] * a[i]
    }
}