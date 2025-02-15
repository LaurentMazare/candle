//! Types for elements that can be stored and manipulated using tensors.
#![allow(clippy::redundant_closure_call)]
use crate::backend::BackendStorage;
use crate::{CpuStorage, CpuStorageRef, Error, Result};

/// The different types of elements allowed in tensors.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    // Unsigned 8 bits integer.
    U8,
    // Unsigned 32 bits integer.
    U32,
    // Signed 64 bits integer.
    I64,
    // Brain floating-point using half precision (16 bits).
    BF16,
    // Floating-point using half precision (16 bits).
    F16,
    // Floating-point using single precision (32 bits).
    F32,
    // Floating-point using double precision (64 bits).
    F64,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DTypeParseError(String);

impl std::fmt::Display for DTypeParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot parse '{}' as a dtype", self.0)
    }
}

impl std::error::Error for DTypeParseError {}

impl std::str::FromStr for DType {
    type Err = DTypeParseError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "u8" => Ok(Self::U8),
            "u32" => Ok(Self::U32),
            "i64" => Ok(Self::I64),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "f64" => Ok(Self::F64),
            _ => Err(DTypeParseError(s.to_string())),
        }
    }
}

impl DType {
    /// String representation for dtypes.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::BF16 => "bf16",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// The size used by each element in bytes, i.e. 1 for `U8`, 4 for `F32`.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    pub fn is_int(&self) -> bool {
        match self {
            Self::U8 | Self::U32 | Self::I64 => true,
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match self {
            Self::U8 | Self::U32 | Self::I64 => false,
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 => true,
        }
    }
}

pub trait WithDType:
    Sized
    + Copy
    + num_traits::NumAssign
    + std::cmp::PartialOrd
    + std::fmt::Display
    + 'static
    + Send
    + Sync
    + std::any::Any
    + crate::cpu::kernels::VecOps
{
    const DTYPE: DType;

    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
    fn cpu_storage_ref(data: &[Self]) -> CpuStorageRef<'_>;
    fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage;

    fn to_cpu_storage(data: &[Self]) -> CpuStorage {
        Self::to_cpu_storage_owned(data.to_vec())
    }

    fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]>;
    fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>>;
    fn cpu_storage_as(s: &CpuStorage, layout: &crate::Layout, dtype: DType) -> Result<CpuStorage>;
}

macro_rules! as_ {
    (U8,   U8,   $v:expr) => {$v};
    (U32,  U32,  $v:expr) => {$v};
    (I64,  I64,  $v:expr) => {$v};
    (F32,  F32,  $v:expr) => {$v};
    (F64,  F64,  $v:expr) => {$v};
    (BF16, BF16, $v:expr) => {$v};
    (F16,  F16,  $v:expr) => {$v};
    ($in:expr, U8,   $v:expr) => { num_traits::AsPrimitive::<u8>::as_($v)};
    ($in:expr, U32,  $v:expr) => { num_traits::AsPrimitive::<u32>::as_($v)};
    ($in:expr, I64,  $v:expr) => { num_traits::AsPrimitive::<i64>::as_($v)};
    ($in:expr, F32,  $v:expr) => { num_traits::AsPrimitive::<f32>::as_($v)};
    ($in:expr, F64,  $v:expr) => { num_traits::AsPrimitive::<f64>::as_($v)};
    ($in:expr, BF16, $v:expr) => { num_traits::AsPrimitive::<bf16>::as_($v)};
    ($in:expr, F16,  $v:expr) => { num_traits::AsPrimitive::<f16>::as_($v)};
}

macro_rules! cpu_storage_as {
    (match:($cpu_storage: expr, $match_dtype: ident), $layout: ident, $with_dtype: ident, ($($dtype: ident),+)) => {{
        match ($cpu_storage, $match_dtype) {
            $((CpuStorage::$with_dtype(storage), DType::$dtype) => {
                Ok({ let data = crate::cpu_backend::unary_map(&storage, $layout,
                    |v| as_!($with_dtype, $dtype, v));
                CpuStorage::$dtype(data)
            })}),+,
            _ => Err(Error::UnexpectedDType {
                expected: DType::$with_dtype,
                got: $cpu_storage.dtype(),
                msg: "unexpected dtype",
            }
            .bt()),
        }
    }};
}

macro_rules! with_dtype {
    ($ty:ty, $dtype:ident, $from_f64:expr, $to_f64:expr) => {
        impl WithDType for $ty {
            const DTYPE: DType = DType::$dtype;

            fn from_f64(v: f64) -> Self {
                $from_f64(v)
            }

            fn to_f64(self) -> f64 {
                $to_f64(self)
            }

            fn cpu_storage_ref(data: &[Self]) -> CpuStorageRef<'_> {
                CpuStorageRef::$dtype(data)
            }

            fn to_cpu_storage_owned(data: Vec<Self>) -> CpuStorage {
                CpuStorage::$dtype(data)
            }

            fn cpu_storage_data(s: CpuStorage) -> Result<Vec<Self>> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            fn cpu_storage_as_slice(s: &CpuStorage) -> Result<&[Self]> {
                match s {
                    CpuStorage::$dtype(data) => Ok(data),
                    _ => Err(Error::UnexpectedDType {
                        expected: DType::$dtype,
                        got: s.dtype(),
                        msg: "unexpected dtype",
                    }
                    .bt()),
                }
            }

            #[inline]
            fn cpu_storage_as(s: &CpuStorage, layout: &crate::Layout, dtype: DType) -> Result<CpuStorage> {
                cpu_storage_as!(match:(s, dtype), layout, $dtype, (U8, U32, I64, F16, BF16, F32, F64))
            }
        }
    };
}
use half::{bf16, f16};

with_dtype!(u8, U8, |v: f64| v as u8, |v: u8| v as f64);
with_dtype!(u32, U32, |v: f64| v as u32, |v: u32| v as f64);
with_dtype!(i64, I64, |v: f64| v as i64, |v: i64| v as f64);
with_dtype!(f16, F16, f16::from_f64, f16::to_f64);
with_dtype!(bf16, BF16, bf16::from_f64, bf16::to_f64);
with_dtype!(f32, F32, |v: f64| v as f32, |v: f32| v as f64);
with_dtype!(f64, F64, |v: f64| v, |v: f64| v);

pub trait IntDType: WithDType {
    fn is_true(&self) -> bool;
    fn as_usize(&self) -> usize;
}

impl IntDType for i64 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for u32 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

impl IntDType for u8 {
    fn is_true(&self) -> bool {
        *self != 0
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}

pub trait FloatDType: WithDType {}

impl FloatDType for f16 {}
impl FloatDType for bf16 {}
impl FloatDType for f32 {}
impl FloatDType for f64 {}
