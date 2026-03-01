//! MLX integration for Magpie — host-side array computation and ML API.
//!
//! Runtime behavior:
//! - `mlx_init()` attempts to load `libmlx.dylib` via `dlopen`.
//! - A dispatch table is populated via `dlsym`.
//! - All host APIs fail gracefully with `MlxError` when MLX is unavailable.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, OnceLock};

#[cfg(target_os = "macos")]
mod ffi {
    use std::ffi::{c_char, c_void};

    unsafe extern "C" {
        pub fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        pub fn dlclose(handle: *mut c_void) -> i32;
    }

    pub const RTLD_LAZY: i32 = 0x1;
    pub const RTLD_LOCAL: i32 = 0x4;
    pub const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;
}

const GPU_ERROR_BACKEND_UNAVAILABLE: i32 = 4;
const GPU_ERROR_UNSUPPORTED: i32 = 7;
const GPU_ERROR_DRIVER_ERROR: i32 = 9;
const GPU_ERROR_VALIDATION_FAILED: i32 = 11;

/// Runtime pointer type used by codegen/runtime C ABI.
pub type MpRtHeader = c_void;

type GpuErrorNewFn = unsafe extern "C" fn(
    kind: i32,
    backend: *const c_char,
    message: *const c_char,
    code: i32,
) -> *mut MpRtHeader;

/// Core MLX dispatch table (required symbols + optional extensions).
#[derive(Clone, Copy)]
pub struct MlxDispatch {
    // Array lifecycle
    pub array_zeros:
        unsafe extern "C" fn(dtype: u32, shape: *const u64, ndim: u32) -> *mut std::ffi::c_void,
    pub array_ones:
        unsafe extern "C" fn(dtype: u32, shape: *const u64, ndim: u32) -> *mut std::ffi::c_void,
    pub array_full: unsafe extern "C" fn(
        dtype: u32,
        shape: *const u64,
        ndim: u32,
        value: *const std::ffi::c_void,
    ) -> *mut std::ffi::c_void,
    pub array_free: unsafe extern "C" fn(*mut std::ffi::c_void),
    // Element-wise ops
    pub array_add:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> *mut std::ffi::c_void,
    pub array_sub:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> *mut std::ffi::c_void,
    pub array_mul:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> *mut std::ffi::c_void,
    pub array_div:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> *mut std::ffi::c_void,
    pub array_matmul:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> *mut std::ffi::c_void,
    // Shape
    pub array_shape: unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u32),
    pub array_reshape:
        unsafe extern "C" fn(*mut std::ffi::c_void, *const u64, u32) -> *mut std::ffi::c_void,
    // Reduction
    pub array_sum: unsafe extern "C" fn(*mut std::ffi::c_void, i32) -> *mut std::ffi::c_void,
    pub array_mean: unsafe extern "C" fn(*mut std::ffi::c_void, i32) -> *mut std::ffi::c_void,
    // Eval
    pub array_eval: unsafe extern "C" fn(*mut std::ffi::c_void),
    // NN
    pub nn_linear: unsafe extern "C" fn(u64, u64) -> *mut std::ffi::c_void,
    pub nn_forward:
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void) -> *mut std::ffi::c_void,
    pub nn_free: unsafe extern "C" fn(*mut std::ffi::c_void),
    // Random
    pub random_normal: unsafe extern "C" fn(u32, *const u64, u32) -> *mut std::ffi::c_void,
    pub random_uniform:
        unsafe extern "C" fn(u32, *const u64, u32, f64, f64) -> *mut std::ffi::c_void,
    // Optim
    pub optim_adam: unsafe extern "C" fn(f32) -> *mut std::ffi::c_void,
    pub optim_step: unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
    ) -> i32,
    pub optim_free: unsafe extern "C" fn(*mut std::ffi::c_void),
    // Grad
    pub grad: unsafe extern "C" fn(*mut std::ffi::c_void) -> *mut std::ffi::c_void,

    // Extended optional symbols for full §8 coverage.
    pub array_from_buffer:
        Option<unsafe extern "C" fn(*const c_void, u32, *const u64, u32) -> *mut c_void>,
    pub array_arange: Option<
        unsafe extern "C" fn(u32, *const c_void, *const c_void, *const c_void) -> *mut c_void,
    >,
    pub array_transpose: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub array_expand_dims: Option<unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void>,
    pub array_squeeze: Option<unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void>,
    pub array_dtype: Option<unsafe extern "C" fn(*mut c_void) -> u32>,
    pub array_neg: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub array_abs: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub array_exp: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub array_log: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub array_sqrt: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub array_pow: Option<unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void>,
    pub array_max: Option<unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void>,
    pub array_min: Option<unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void>,
    pub array_argmax: Option<unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void>,
    pub array_argmin: Option<unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void>,
    pub linalg_norm: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub linalg_inv: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub nn_relu: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub nn_gelu: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub nn_sigmoid: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub nn_tanh: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub nn_softmax: Option<unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void>,
    pub optim_sgd: Option<unsafe extern "C" fn(f32, f32) -> *mut c_void>,
    pub optim_adamw: Option<unsafe extern "C" fn(f32, f32) -> *mut c_void>,
    pub value_and_grad: Option<unsafe extern "C" fn(*mut c_void) -> *mut c_void>,
    pub random_seed: Option<unsafe extern "C" fn(u64)>,
}

static MLX_INIT: Once = Once::new();
static MLX_AVAILABLE: AtomicBool = AtomicBool::new(false);
static MLX_LIBRARY_HANDLE: OnceLock<Option<usize>> = OnceLock::new();
static MLX_DISPATCH: OnceLock<Option<MlxDispatch>> = OnceLock::new();
static GPU_ERROR_NEW: OnceLock<Option<GpuErrorNewFn>> = OnceLock::new();

/// Opaque handle to an MLX array (`mlx::core::array`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlxArrayHandle {
    pub ptr: *mut c_void,
}

impl MlxArrayHandle {
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }
}

/// Opaque handle to an MLX NN layer.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlxLayerHandle {
    pub ptr: *mut c_void,
}

/// Opaque handle to an MLX optimizer.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlxOptimizerHandle {
    pub ptr: *mut c_void,
}

/// Opaque handle to an MLX grad/value_and_grad callable.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlxGradHandle {
    pub ptr: *mut c_void,
}

/// MLX element data type (mirrors `mlx::core::Dtype`).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlxDtype {
    Bool = 0,
    U8 = 1,
    U16 = 2,
    U32 = 3,
    U64 = 4,
    I8 = 5,
    I16 = 6,
    I32 = 7,
    I64 = 8,
    F16 = 9,
    F32 = 10,
    Bf16 = 11,
    F64 = 12,
}

impl MlxDtype {
    /// Convert from Magpie fixed TypeId to MLX dtype.
    pub fn from_type_id(type_id: u32) -> Option<Self> {
        match type_id {
            1 => Some(MlxDtype::Bool),  // BOOL
            7 => Some(MlxDtype::U8),    // U8
            8 => Some(MlxDtype::U16),   // U16
            9 => Some(MlxDtype::U32),   // U32
            10 => Some(MlxDtype::U64),  // U64
            2 => Some(MlxDtype::I8),    // I8
            3 => Some(MlxDtype::I16),   // I16
            4 => Some(MlxDtype::I32),   // I32
            5 => Some(MlxDtype::I64),   // I64
            13 => Some(MlxDtype::F16),  // F16
            14 => Some(MlxDtype::F32),  // F32
            16 => Some(MlxDtype::Bf16), // BF16
            15 => Some(MlxDtype::F64),  // F64
            _ => None,
        }
    }

    pub fn from_raw(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(MlxDtype::Bool),
            1 => Some(MlxDtype::U8),
            2 => Some(MlxDtype::U16),
            3 => Some(MlxDtype::U32),
            4 => Some(MlxDtype::U64),
            5 => Some(MlxDtype::I8),
            6 => Some(MlxDtype::I16),
            7 => Some(MlxDtype::I32),
            8 => Some(MlxDtype::I64),
            9 => Some(MlxDtype::F16),
            10 => Some(MlxDtype::F32),
            11 => Some(MlxDtype::Bf16),
            12 => Some(MlxDtype::F64),
            _ => None,
        }
    }

    pub fn as_u32(self) -> u32 {
        self as u32
    }
}

/// MLX error type.
#[derive(Debug, Clone)]
pub struct MlxError {
    pub kind: i32,
    pub message: String,
    pub code: i32,
}

impl std::fmt::Display for MlxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MLX error (kind={}, code={}): {}",
            self.kind, self.code, self.message
        )
    }
}

impl std::error::Error for MlxError {}

impl MlxError {
    fn backend_unavailable(message: impl Into<String>) -> Self {
        Self {
            kind: GPU_ERROR_BACKEND_UNAVAILABLE,
            message: message.into(),
            code: -1,
        }
    }

    fn validation(message: impl Into<String>) -> Self {
        Self {
            kind: GPU_ERROR_VALIDATION_FAILED,
            message: message.into(),
            code: -1,
        }
    }

    fn driver(message: impl Into<String>) -> Self {
        Self {
            kind: GPU_ERROR_DRIVER_ERROR,
            message: message.into(),
            code: -1,
        }
    }

    fn null_result(op: &str) -> Self {
        Self {
            kind: GPU_ERROR_DRIVER_ERROR,
            message: format!("MLX operation `{op}` returned null handle"),
            code: -1,
        }
    }

    fn missing_symbol(symbol: &str) -> Self {
        Self {
            kind: GPU_ERROR_UNSUPPORTED,
            message: format!("MLX runtime symbol `{symbol}` is not available"),
            code: -1,
        }
    }
}

#[cfg(target_os = "macos")]
unsafe fn load_dispatch_from_handle(lib: *mut c_void) -> Option<MlxDispatch> {
    macro_rules! load_required {
        ($name:literal, $ty:ty) => {{
            let sym =
                unsafe { CStr::from_bytes_with_nul_unchecked(concat!($name, "\0").as_bytes()) };
            let ptr = unsafe { ffi::dlsym(lib, sym.as_ptr()) };
            if ptr.is_null() {
                return None;
            }
            unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) }
        }};
    }

    macro_rules! load_optional {
        ($name:literal, $ty:ty) => {{
            let sym =
                unsafe { CStr::from_bytes_with_nul_unchecked(concat!($name, "\0").as_bytes()) };
            let ptr = unsafe { ffi::dlsym(lib, sym.as_ptr()) };
            if ptr.is_null() {
                None
            } else {
                Some(unsafe { std::mem::transmute::<*mut c_void, $ty>(ptr) })
            }
        }};
    }

    Some(MlxDispatch {
        array_zeros: load_required!(
            "mlx_array_zeros",
            unsafe extern "C" fn(u32, *const u64, u32) -> *mut c_void
        ),
        array_ones: load_required!(
            "mlx_array_ones",
            unsafe extern "C" fn(u32, *const u64, u32) -> *mut c_void
        ),
        array_full: load_required!(
            "mlx_array_full",
            unsafe extern "C" fn(u32, *const u64, u32, *const c_void) -> *mut c_void
        ),
        array_free: load_required!("mlx_array_free", unsafe extern "C" fn(*mut c_void)),
        array_add: load_required!(
            "mlx_array_add",
            unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void
        ),
        array_sub: load_required!(
            "mlx_array_sub",
            unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void
        ),
        array_mul: load_required!(
            "mlx_array_mul",
            unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void
        ),
        array_div: load_required!(
            "mlx_array_div",
            unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void
        ),
        array_matmul: load_required!(
            "mlx_array_matmul",
            unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void
        ),
        array_shape: load_required!(
            "mlx_array_shape",
            unsafe extern "C" fn(*mut c_void, *mut u64, *mut u32)
        ),
        array_reshape: load_required!(
            "mlx_array_reshape",
            unsafe extern "C" fn(*mut c_void, *const u64, u32) -> *mut c_void
        ),
        array_sum: load_required!(
            "mlx_array_sum",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        ),
        array_mean: load_required!(
            "mlx_array_mean",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        ),
        array_eval: load_required!("mlx_array_eval", unsafe extern "C" fn(*mut c_void)),
        nn_linear: load_required!(
            "mlx_nn_linear",
            unsafe extern "C" fn(u64, u64) -> *mut c_void
        ),
        nn_forward: load_required!(
            "mlx_nn_forward",
            unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void
        ),
        nn_free: load_required!("mlx_nn_free", unsafe extern "C" fn(*mut c_void)),
        random_normal: load_required!(
            "mlx_random_normal",
            unsafe extern "C" fn(u32, *const u64, u32) -> *mut c_void
        ),
        random_uniform: load_required!(
            "mlx_random_uniform",
            unsafe extern "C" fn(u32, *const u64, u32, f64, f64) -> *mut c_void
        ),
        optim_adam: load_required!("mlx_optim_adam", unsafe extern "C" fn(f32) -> *mut c_void),
        optim_step: load_required!(
            "mlx_optim_step",
            unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> i32
        ),
        optim_free: load_required!("mlx_optim_free", unsafe extern "C" fn(*mut c_void)),
        grad: load_required!("mlx_grad", unsafe extern "C" fn(*mut c_void) -> *mut c_void),

        array_from_buffer: load_optional!(
            "mlx_array_from_buffer",
            unsafe extern "C" fn(*const c_void, u32, *const u64, u32) -> *mut c_void
        ),
        array_arange: load_optional!(
            "mlx_array_arange",
            unsafe extern "C" fn(u32, *const c_void, *const c_void, *const c_void) -> *mut c_void
        ),
        array_transpose: load_optional!(
            "mlx_array_transpose",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        array_expand_dims: load_optional!(
            "mlx_array_expand_dims",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        ),
        array_squeeze: load_optional!(
            "mlx_array_squeeze",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        ),
        array_dtype: load_optional!("mlx_array_dtype", unsafe extern "C" fn(*mut c_void) -> u32),
        array_neg: load_optional!(
            "mlx_array_neg",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        array_abs: load_optional!(
            "mlx_array_abs",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        array_exp: load_optional!(
            "mlx_array_exp",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        array_log: load_optional!(
            "mlx_array_log",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        array_sqrt: load_optional!(
            "mlx_array_sqrt",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        array_pow: load_optional!(
            "mlx_array_pow",
            unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void
        ),
        array_max: load_optional!(
            "mlx_array_max",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        ),
        array_min: load_optional!(
            "mlx_array_min",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        ),
        array_argmax: load_optional!(
            "mlx_array_argmax",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        ),
        array_argmin: load_optional!(
            "mlx_array_argmin",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        ),
        linalg_norm: load_optional!(
            "mlx_linalg_norm",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        linalg_inv: load_optional!(
            "mlx_linalg_inv",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        nn_relu: load_optional!(
            "mlx_nn_relu",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        nn_gelu: load_optional!(
            "mlx_nn_gelu",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        nn_sigmoid: load_optional!(
            "mlx_nn_sigmoid",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        nn_tanh: load_optional!(
            "mlx_nn_tanh",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        nn_softmax: load_optional!(
            "mlx_nn_softmax",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void
        ),
        optim_sgd: load_optional!(
            "mlx_optim_sgd",
            unsafe extern "C" fn(f32, f32) -> *mut c_void
        ),
        optim_adamw: load_optional!(
            "mlx_optim_adamw",
            unsafe extern "C" fn(f32, f32) -> *mut c_void
        ),
        value_and_grad: load_optional!(
            "mlx_value_and_grad",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        ),
        random_seed: load_optional!("mlx_random_seed", unsafe extern "C" fn(u64)),
    })
}

#[cfg(target_os = "macos")]
unsafe fn load_dispatch() -> Option<(usize, MlxDispatch)> {
    let mut candidates = Vec::new();
    if let Ok(custom) = std::env::var("MAGPIE_MLX_LIB") {
        let trimmed = custom.trim();
        if !trimmed.is_empty() {
            candidates.push(trimmed.to_string());
        }
    }
    candidates.push("libmlx.dylib".to_string());
    candidates.push("@rpath/libmlx.dylib".to_string());
    candidates.push("/opt/homebrew/lib/libmlx.dylib".to_string());
    candidates.push("/usr/local/lib/libmlx.dylib".to_string());

    for candidate in candidates {
        let cpath = match CString::new(candidate) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let handle = unsafe { ffi::dlopen(cpath.as_ptr(), ffi::RTLD_LAZY | ffi::RTLD_LOCAL) };
        if handle.is_null() {
            continue;
        }
        if let Some(dispatch) = unsafe { load_dispatch_from_handle(handle) } {
            return Some((handle as usize, dispatch));
        }
        let _ = unsafe { ffi::dlclose(handle) };
    }
    None
}

#[cfg(not(target_os = "macos"))]
unsafe fn load_dispatch() -> Option<(usize, MlxDispatch)> {
    None
}

#[cfg(target_os = "macos")]
unsafe fn load_gpu_error_new() -> Option<GpuErrorNewFn> {
    let symbol = unsafe { CStr::from_bytes_with_nul_unchecked(b"mp_rt_gpu_error_new\0") };
    let ptr = unsafe { ffi::dlsym(ffi::RTLD_DEFAULT, symbol.as_ptr()) };
    if ptr.is_null() {
        None
    } else {
        Some(unsafe { std::mem::transmute::<*mut c_void, GpuErrorNewFn>(ptr) })
    }
}

#[cfg(not(target_os = "macos"))]
unsafe fn load_gpu_error_new() -> Option<GpuErrorNewFn> {
    None
}

fn mlx_unavailable_error() -> MlxError {
    MlxError::backend_unavailable(
        "MLX backend unavailable (requires macOS + libmlx.dylib with full required symbols)",
    )
}

/// Initialize MLX runtime (loads `libmlx.dylib` + dispatch via `dlsym`).
pub fn mlx_init() -> bool {
    MLX_INIT.call_once(|| {
        let dispatch = unsafe { load_dispatch() };
        let (lib_handle, table) = match dispatch {
            Some((h, d)) => (Some(h), Some(d)),
            None => (None, None),
        };

        let _ = MLX_LIBRARY_HANDLE.set(lib_handle);
        let _ = MLX_DISPATCH.set(table);
        MLX_AVAILABLE.store(
            MLX_DISPATCH.get().and_then(|d| d.as_ref()).is_some(),
            Ordering::Relaxed,
        );

        let _ = GPU_ERROR_NEW.set(unsafe { load_gpu_error_new() });
    });

    MLX_AVAILABLE.load(Ordering::Relaxed)
}

/// Check whether MLX is available in the current process.
pub fn mlx_is_available() -> bool {
    mlx_init();
    MLX_AVAILABLE.load(Ordering::Relaxed)
}

fn require_dispatch() -> Result<&'static MlxDispatch, MlxError> {
    if !mlx_is_available() {
        return Err(mlx_unavailable_error());
    }
    MLX_DISPATCH
        .get()
        .and_then(|d| d.as_ref())
        .ok_or_else(mlx_unavailable_error)
}

fn require_non_null_ptr(ptr: *mut c_void, what: &str) -> Result<*mut c_void, MlxError> {
    if ptr.is_null() {
        Err(MlxError::validation(format!("`{what}` must not be null")))
    } else {
        Ok(ptr)
    }
}

fn require_non_null_const_ptr(ptr: *const c_void, what: &str) -> Result<*const c_void, MlxError> {
    if ptr.is_null() {
        Err(MlxError::validation(format!("`{what}` must not be null")))
    } else {
        Ok(ptr)
    }
}

fn wrap_array_result(op: &str, ptr: *mut c_void) -> Result<MlxArrayHandle, MlxError> {
    if ptr.is_null() {
        Err(MlxError::null_result(op))
    } else {
        Ok(MlxArrayHandle { ptr })
    }
}

fn wrap_layer_result(op: &str, ptr: *mut c_void) -> Result<MlxLayerHandle, MlxError> {
    if ptr.is_null() {
        Err(MlxError::null_result(op))
    } else {
        Ok(MlxLayerHandle { ptr })
    }
}

fn wrap_optimizer_result(op: &str, ptr: *mut c_void) -> Result<MlxOptimizerHandle, MlxError> {
    if ptr.is_null() {
        Err(MlxError::null_result(op))
    } else {
        Ok(MlxOptimizerHandle { ptr })
    }
}

fn wrap_grad_result(op: &str, ptr: *mut c_void) -> Result<MlxGradHandle, MlxError> {
    if ptr.is_null() {
        Err(MlxError::null_result(op))
    } else {
        Ok(MlxGradHandle { ptr })
    }
}

fn require_optional<T>(name: &str, f: Option<T>) -> Result<T, MlxError> {
    f.ok_or_else(|| MlxError::missing_symbol(name))
}

fn validate_shape(shape: &[u64]) -> Result<(), MlxError> {
    if shape.len() > u32::MAX as usize {
        return Err(MlxError::validation("shape rank exceeds u32::MAX"));
    }
    Ok(())
}

fn unary_array_op(
    op_name: &'static str,
    a: &MlxArrayHandle,
    f: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
) -> Result<MlxArrayHandle, MlxError> {
    let _ = require_dispatch()?;
    let a_ptr = require_non_null_ptr(a.ptr, "a")?;
    let out = unsafe { f(a_ptr) };
    wrap_array_result(op_name, out)
}

fn binary_array_op(
    op_name: &'static str,
    a: &MlxArrayHandle,
    b: &MlxArrayHandle,
    f: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void,
) -> Result<MlxArrayHandle, MlxError> {
    let _ = require_dispatch()?;
    let a_ptr = require_non_null_ptr(a.ptr, "a")?;
    let b_ptr = require_non_null_ptr(b.ptr, "b")?;
    let out = unsafe { f(a_ptr, b_ptr) };
    wrap_array_result(op_name, out)
}

fn axis_reduce_op(
    op_name: &'static str,
    a: &MlxArrayHandle,
    axis: i32,
    f: unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void,
) -> Result<MlxArrayHandle, MlxError> {
    let _ = require_dispatch()?;
    let a_ptr = require_non_null_ptr(a.ptr, "a")?;
    let out = unsafe { f(a_ptr, axis) };
    wrap_array_result(op_name, out)
}

// §8.3 Array operations ----------------------------------------------------

pub fn mlx_array_zeros(dtype: MlxDtype, shape: &[u64]) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    validate_shape(shape)?;
    let out = unsafe { (dispatch.array_zeros)(dtype.as_u32(), shape.as_ptr(), shape.len() as u32) };
    wrap_array_result("mlx_array_zeros", out)
}

pub fn mlx_array_ones(dtype: MlxDtype, shape: &[u64]) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    validate_shape(shape)?;
    let out = unsafe { (dispatch.array_ones)(dtype.as_u32(), shape.as_ptr(), shape.len() as u32) };
    wrap_array_result("mlx_array_ones", out)
}

pub fn mlx_array_full(
    dtype: MlxDtype,
    shape: &[u64],
    value: *const c_void,
) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    validate_shape(shape)?;
    let value_ptr = require_non_null_const_ptr(value, "value")?;
    let out = unsafe {
        (dispatch.array_full)(
            dtype.as_u32(),
            shape.as_ptr(),
            shape.len() as u32,
            value_ptr,
        )
    };
    wrap_array_result("mlx_array_full", out)
}

pub fn mlx_array_from_buffer(
    buffer: *const c_void,
    dtype: MlxDtype,
    shape: &[u64],
) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    validate_shape(shape)?;
    let f = require_optional("mlx_array_from_buffer", dispatch.array_from_buffer)?;
    let buffer_ptr = require_non_null_const_ptr(buffer, "buffer")?;
    let out = unsafe {
        f(
            buffer_ptr,
            dtype.as_u32(),
            shape.as_ptr(),
            shape.len() as u32,
        )
    };
    wrap_array_result("mlx_array_from_buffer", out)
}

pub fn mlx_array_arange(
    dtype: MlxDtype,
    start: *const c_void,
    stop: *const c_void,
    step: *const c_void,
) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_arange", dispatch.array_arange)?;
    let start_ptr = require_non_null_const_ptr(start, "start")?;
    let stop_ptr = require_non_null_const_ptr(stop, "stop")?;
    let step_ptr = require_non_null_const_ptr(step, "step")?;
    let out = unsafe { f(dtype.as_u32(), start_ptr, stop_ptr, step_ptr) };
    wrap_array_result("mlx_array_arange", out)
}

pub fn mlx_array_reshape(a: &MlxArrayHandle, shape: &[u64]) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    validate_shape(shape)?;
    let a_ptr = require_non_null_ptr(a.ptr, "a")?;
    let out = unsafe { (dispatch.array_reshape)(a_ptr, shape.as_ptr(), shape.len() as u32) };
    wrap_array_result("mlx_array_reshape", out)
}

pub fn mlx_array_transpose(a: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_transpose", dispatch.array_transpose)?;
    unary_array_op("mlx_array_transpose", a, f)
}

pub fn mlx_array_expand_dims(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_expand_dims", dispatch.array_expand_dims)?;
    let a_ptr = require_non_null_ptr(a.ptr, "a")?;
    let out = unsafe { f(a_ptr, axis) };
    wrap_array_result("mlx_array_expand_dims", out)
}

pub fn mlx_array_squeeze(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_squeeze", dispatch.array_squeeze)?;
    let a_ptr = require_non_null_ptr(a.ptr, "a")?;
    let out = unsafe { f(a_ptr, axis) };
    wrap_array_result("mlx_array_squeeze", out)
}

pub fn mlx_array_shape(a: &MlxArrayHandle) -> Result<Vec<u64>, MlxError> {
    let dispatch = require_dispatch()?;
    let a_ptr = require_non_null_ptr(a.ptr, "a")?;

    // Portable two-pass shape query without null shape pointers.
    let mut ndim = 8u32;
    let mut small = [0u64; 8];
    unsafe { (dispatch.array_shape)(a_ptr, small.as_mut_ptr(), &mut ndim as *mut u32) };

    if ndim <= small.len() as u32 {
        return Ok(small[..ndim as usize].to_vec());
    }

    let mut out = vec![0u64; ndim as usize];
    let mut ndim2 = ndim;
    unsafe { (dispatch.array_shape)(a_ptr, out.as_mut_ptr(), &mut ndim2 as *mut u32) };
    out.truncate(ndim2 as usize);
    Ok(out)
}

pub fn mlx_array_ndim(a: &MlxArrayHandle) -> Result<u32, MlxError> {
    Ok(mlx_array_shape(a)?.len() as u32)
}

pub fn mlx_array_size(a: &MlxArrayHandle) -> Result<u64, MlxError> {
    let shape = mlx_array_shape(a)?;
    let mut acc = 1u64;
    for d in shape {
        acc = acc.checked_mul(d).ok_or_else(|| {
            MlxError::driver("array size overflow while computing product of shape dims")
        })?;
    }
    Ok(acc)
}

pub fn mlx_array_dtype(a: &MlxArrayHandle) -> Result<u32, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_dtype", dispatch.array_dtype)?;
    let a_ptr = require_non_null_ptr(a.ptr, "a")?;
    Ok(unsafe { f(a_ptr) })
}

pub fn mlx_array_add(a: &MlxArrayHandle, b: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    binary_array_op("mlx_array_add", a, b, dispatch.array_add)
}

pub fn mlx_array_sub(a: &MlxArrayHandle, b: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    binary_array_op("mlx_array_sub", a, b, dispatch.array_sub)
}

pub fn mlx_array_mul(a: &MlxArrayHandle, b: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    binary_array_op("mlx_array_mul", a, b, dispatch.array_mul)
}

pub fn mlx_array_div(a: &MlxArrayHandle, b: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    binary_array_op("mlx_array_div", a, b, dispatch.array_div)
}

pub fn mlx_array_neg(a: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_neg", dispatch.array_neg)?;
    unary_array_op("mlx_array_neg", a, f)
}

pub fn mlx_array_abs(a: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_abs", dispatch.array_abs)?;
    unary_array_op("mlx_array_abs", a, f)
}

pub fn mlx_array_exp(a: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_exp", dispatch.array_exp)?;
    unary_array_op("mlx_array_exp", a, f)
}

pub fn mlx_array_log(a: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_log", dispatch.array_log)?;
    unary_array_op("mlx_array_log", a, f)
}

pub fn mlx_array_sqrt(a: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_sqrt", dispatch.array_sqrt)?;
    unary_array_op("mlx_array_sqrt", a, f)
}

pub fn mlx_array_pow(a: &MlxArrayHandle, b: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_pow", dispatch.array_pow)?;
    binary_array_op("mlx_array_pow", a, b, f)
}

pub fn mlx_array_sum(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    axis_reduce_op("mlx_array_sum", a, axis, dispatch.array_sum)
}

pub fn mlx_array_mean(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    axis_reduce_op("mlx_array_mean", a, axis, dispatch.array_mean)
}

pub fn mlx_array_max(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_max", dispatch.array_max)?;
    axis_reduce_op("mlx_array_max", a, axis, f)
}

pub fn mlx_array_min(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_min", dispatch.array_min)?;
    axis_reduce_op("mlx_array_min", a, axis, f)
}

pub fn mlx_array_argmax(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_argmax", dispatch.array_argmax)?;
    axis_reduce_op("mlx_array_argmax", a, axis, f)
}

pub fn mlx_array_argmin(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_array_argmin", dispatch.array_argmin)?;
    axis_reduce_op("mlx_array_argmin", a, axis, f)
}

pub fn mlx_array_eval(a: &MlxArrayHandle) -> Result<(), MlxError> {
    let dispatch = require_dispatch()?;
    let a_ptr = require_non_null_ptr(a.ptr, "a")?;
    unsafe { (dispatch.array_eval)(a_ptr) };
    Ok(())
}

pub fn mlx_array_free(handle: MlxArrayHandle) -> Result<(), MlxError> {
    let dispatch = require_dispatch()?;
    let ptr = require_non_null_ptr(handle.ptr, "handle")?;
    unsafe { (dispatch.array_free)(ptr) };
    Ok(())
}

// §8.4 Linear algebra ------------------------------------------------------

pub fn mlx_linalg_matmul(
    a: &MlxArrayHandle,
    b: &MlxArrayHandle,
) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    binary_array_op("mlx_linalg_matmul", a, b, dispatch.array_matmul)
}

pub fn mlx_array_matmul(
    a: &MlxArrayHandle,
    b: &MlxArrayHandle,
) -> Result<MlxArrayHandle, MlxError> {
    mlx_linalg_matmul(a, b)
}

pub fn mlx_linalg_norm(a: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_linalg_norm", dispatch.linalg_norm)?;
    unary_array_op("mlx_linalg_norm", a, f)
}

pub fn mlx_linalg_inv(a: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_linalg_inv", dispatch.linalg_inv)?;
    unary_array_op("mlx_linalg_inv", a, f)
}

// §8.5 Neural network layers ----------------------------------------------

pub fn mlx_nn_linear(in_features: u64, out_features: u64) -> Result<MlxLayerHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let out = unsafe { (dispatch.nn_linear)(in_features, out_features) };
    wrap_layer_result("mlx_nn_linear", out)
}

pub fn mlx_nn_forward(
    layer: &MlxLayerHandle,
    input: &MlxArrayHandle,
) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let layer_ptr = require_non_null_ptr(layer.ptr, "layer")?;
    let input_ptr = require_non_null_ptr(input.ptr, "input")?;
    let out = unsafe { (dispatch.nn_forward)(layer_ptr, input_ptr) };
    wrap_array_result("mlx_nn_forward", out)
}

pub fn mlx_nn_relu(x: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_nn_relu", dispatch.nn_relu)?;
    unary_array_op("mlx_nn_relu", x, f)
}

pub fn mlx_nn_gelu(x: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_nn_gelu", dispatch.nn_gelu)?;
    unary_array_op("mlx_nn_gelu", x, f)
}

pub fn mlx_nn_sigmoid(x: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_nn_sigmoid", dispatch.nn_sigmoid)?;
    unary_array_op("mlx_nn_sigmoid", x, f)
}

pub fn mlx_nn_tanh(x: &MlxArrayHandle) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_nn_tanh", dispatch.nn_tanh)?;
    unary_array_op("mlx_nn_tanh", x, f)
}

pub fn mlx_nn_softmax(x: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_nn_softmax", dispatch.nn_softmax)?;
    let x_ptr = require_non_null_ptr(x.ptr, "x")?;
    let out = unsafe { f(x_ptr, axis) };
    wrap_array_result("mlx_nn_softmax", out)
}

pub fn mlx_nn_free(layer: MlxLayerHandle) -> Result<(), MlxError> {
    let dispatch = require_dispatch()?;
    let ptr = require_non_null_ptr(layer.ptr, "layer")?;
    unsafe { (dispatch.nn_free)(ptr) };
    Ok(())
}

// §8.7 Optimizers ----------------------------------------------------------

pub fn mlx_optim_adam(lr: f32) -> Result<MlxOptimizerHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let out = unsafe { (dispatch.optim_adam)(lr) };
    wrap_optimizer_result("mlx_optim_adam", out)
}

pub fn mlx_optim_sgd(lr: f32, momentum: f32) -> Result<MlxOptimizerHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_optim_sgd", dispatch.optim_sgd)?;
    let out = unsafe { f(lr, momentum) };
    wrap_optimizer_result("mlx_optim_sgd", out)
}

pub fn mlx_optim_adamw(lr: f32, wd: f32) -> Result<MlxOptimizerHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_optim_adamw", dispatch.optim_adamw)?;
    let out = unsafe { f(lr, wd) };
    wrap_optimizer_result("mlx_optim_adamw", out)
}

pub fn mlx_optim_step(
    optim: &MlxOptimizerHandle,
    param: &MlxArrayHandle,
    grad: &MlxArrayHandle,
) -> Result<(), MlxError> {
    let dispatch = require_dispatch()?;
    let optim_ptr = require_non_null_ptr(optim.ptr, "optim")?;
    let param_ptr = require_non_null_ptr(param.ptr, "param")?;
    let grad_ptr = require_non_null_ptr(grad.ptr, "grad")?;
    let rc = unsafe { (dispatch.optim_step)(optim_ptr, param_ptr, grad_ptr) };
    if rc == 0 {
        Ok(())
    } else {
        Err(MlxError {
            kind: GPU_ERROR_DRIVER_ERROR,
            message: format!("mlx_optim_step failed with code {rc}"),
            code: rc,
        })
    }
}

pub fn mlx_optim_free(optim: MlxOptimizerHandle) -> Result<(), MlxError> {
    let dispatch = require_dispatch()?;
    let ptr = require_non_null_ptr(optim.ptr, "optim")?;
    unsafe { (dispatch.optim_free)(ptr) };
    Ok(())
}

// §8.6 Autograd ------------------------------------------------------------

pub fn mlx_grad(callable: *mut c_void) -> Result<MlxGradHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let callable_ptr = require_non_null_ptr(callable, "callable")?;
    let out = unsafe { (dispatch.grad)(callable_ptr) };
    wrap_grad_result("mlx_grad", out)
}

pub fn mlx_value_and_grad(callable: *mut c_void) -> Result<MlxGradHandle, MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_value_and_grad", dispatch.value_and_grad)?;
    let callable_ptr = require_non_null_ptr(callable, "callable")?;
    let out = unsafe { f(callable_ptr) };
    wrap_grad_result("mlx_value_and_grad", out)
}

// §8.8 Random --------------------------------------------------------------

pub fn mlx_random_normal(dtype: MlxDtype, shape: &[u64]) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    validate_shape(shape)?;
    let out =
        unsafe { (dispatch.random_normal)(dtype.as_u32(), shape.as_ptr(), shape.len() as u32) };
    wrap_array_result("mlx_random_normal", out)
}

pub fn mlx_random_uniform(
    dtype: MlxDtype,
    shape: &[u64],
    low: f64,
    high: f64,
) -> Result<MlxArrayHandle, MlxError> {
    let dispatch = require_dispatch()?;
    validate_shape(shape)?;
    let out = unsafe {
        (dispatch.random_uniform)(
            dtype.as_u32(),
            shape.as_ptr(),
            shape.len() as u32,
            low,
            high,
        )
    };
    wrap_array_result("mlx_random_uniform", out)
}

pub fn mlx_random_seed(seed: u64) -> Result<(), MlxError> {
    let dispatch = require_dispatch()?;
    let f = require_optional("mlx_random_seed", dispatch.random_seed)?;
    unsafe { f(seed) };
    Ok(())
}

// C ABI runtime entrypoints ------------------------------------------------

unsafe fn clear_out_handle(out: *mut *mut MpRtHeader) {
    if !out.is_null() {
        *out = std::ptr::null_mut();
    }
}

unsafe fn set_out_error(out_err: *mut *mut MpRtHeader, err: &MlxError) {
    clear_out_handle(out_err);
    if out_err.is_null() {
        return;
    }

    let ctor = GPU_ERROR_NEW.get_or_init(|| unsafe { load_gpu_error_new() });
    if let Some(make_error) = *ctor {
        let backend = b"mlx\0";
        let message = CString::new(err.message.as_str())
            .unwrap_or_else(|_| CString::new("mlx error").expect("static CString"));
        let obj = unsafe {
            make_error(
                err.kind,
                backend.as_ptr() as *const c_char,
                message.as_ptr(),
                err.code,
            )
        };
        *out_err = obj;
    }
}

unsafe fn set_success(out_err: *mut *mut MpRtHeader) {
    clear_out_handle(out_err);
}

unsafe fn write_array_result(
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
    result: Result<MlxArrayHandle, MlxError>,
) -> i32 {
    clear_out_handle(out);
    clear_out_handle(out_err);
    if out.is_null() {
        set_out_error(out_err, &MlxError::validation("`out` must not be null"));
        return -1;
    }
    match result {
        Ok(v) => {
            *out = v.ptr as *mut MpRtHeader;
            set_success(out_err);
            0
        }
        Err(e) => {
            set_out_error(out_err, &e);
            -1
        }
    }
}

unsafe fn write_layer_result(
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
    result: Result<MlxLayerHandle, MlxError>,
) -> i32 {
    clear_out_handle(out);
    clear_out_handle(out_err);
    if out.is_null() {
        set_out_error(out_err, &MlxError::validation("`out` must not be null"));
        return -1;
    }
    match result {
        Ok(v) => {
            *out = v.ptr as *mut MpRtHeader;
            set_success(out_err);
            0
        }
        Err(e) => {
            set_out_error(out_err, &e);
            -1
        }
    }
}

unsafe fn write_optim_result(
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
    result: Result<MlxOptimizerHandle, MlxError>,
) -> i32 {
    clear_out_handle(out);
    clear_out_handle(out_err);
    if out.is_null() {
        set_out_error(out_err, &MlxError::validation("`out` must not be null"));
        return -1;
    }
    match result {
        Ok(v) => {
            *out = v.ptr as *mut MpRtHeader;
            set_success(out_err);
            0
        }
        Err(e) => {
            set_out_error(out_err, &e);
            -1
        }
    }
}

unsafe fn write_grad_result(
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
    result: Result<MlxGradHandle, MlxError>,
) -> i32 {
    clear_out_handle(out);
    clear_out_handle(out_err);
    if out.is_null() {
        set_out_error(out_err, &MlxError::validation("`out` must not be null"));
        return -1;
    }
    match result {
        Ok(v) => {
            *out = v.ptr as *mut MpRtHeader;
            set_success(out_err);
            0
        }
        Err(e) => {
            set_out_error(out_err, &e);
            -1
        }
    }
}

unsafe fn write_unit_result(out_err: *mut *mut MpRtHeader, result: Result<(), MlxError>) -> i32 {
    clear_out_handle(out_err);
    match result {
        Ok(()) => {
            set_success(out_err);
            0
        }
        Err(e) => {
            set_out_error(out_err, &e);
            -1
        }
    }
}

unsafe fn write_u32_result(
    out: *mut u32,
    out_err: *mut *mut MpRtHeader,
    result: Result<u32, MlxError>,
) -> i32 {
    clear_out_handle(out_err);
    if out.is_null() {
        set_out_error(out_err, &MlxError::validation("`out` must not be null"));
        return -1;
    }
    match result {
        Ok(v) => {
            *out = v;
            set_success(out_err);
            0
        }
        Err(e) => {
            set_out_error(out_err, &e);
            -1
        }
    }
}

unsafe fn write_u64_result(
    out: *mut u64,
    out_err: *mut *mut MpRtHeader,
    result: Result<u64, MlxError>,
) -> i32 {
    clear_out_handle(out_err);
    if out.is_null() {
        set_out_error(out_err, &MlxError::validation("`out` must not be null"));
        return -1;
    }
    match result {
        Ok(v) => {
            *out = v;
            set_success(out_err);
            0
        }
        Err(e) => {
            set_out_error(out_err, &e);
            -1
        }
    }
}

unsafe fn read_shape_arg<'a>(shape: *const u64, ndim: u32) -> Result<&'a [u64], MlxError> {
    if ndim == 0 {
        return Ok(&[]);
    }
    if shape.is_null() {
        return Err(MlxError::validation(
            "`shape` must not be null when ndim > 0",
        ));
    }
    Ok(std::slice::from_raw_parts(shape, ndim as usize))
}

unsafe fn read_array_arg(ptr: *mut MpRtHeader, name: &str) -> Result<MlxArrayHandle, MlxError> {
    if ptr.is_null() {
        Err(MlxError::validation(format!("`{name}` must not be null")))
    } else {
        Ok(MlxArrayHandle {
            ptr: ptr as *mut c_void,
        })
    }
}

unsafe fn read_layer_arg(ptr: *mut MpRtHeader, name: &str) -> Result<MlxLayerHandle, MlxError> {
    if ptr.is_null() {
        Err(MlxError::validation(format!("`{name}` must not be null")))
    } else {
        Ok(MlxLayerHandle {
            ptr: ptr as *mut c_void,
        })
    }
}

unsafe fn read_optim_arg(ptr: *mut MpRtHeader, name: &str) -> Result<MlxOptimizerHandle, MlxError> {
    if ptr.is_null() {
        Err(MlxError::validation(format!("`{name}` must not be null")))
    } else {
        Ok(MlxOptimizerHandle {
            ptr: ptr as *mut c_void,
        })
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_init() -> i32 {
    if mlx_init() {
        0
    } else {
        -1
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_is_available() -> i32 {
    if mlx_is_available() {
        1
    } else {
        0
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_zeros(
    dtype: u32,
    shape: *const u64,
    ndim: u32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let dtype = MlxDtype::from_raw(dtype)
            .ok_or_else(|| MlxError::validation(format!("invalid MLX dtype value `{dtype}`")))?;
        let shape = read_shape_arg(shape, ndim)?;
        mlx_array_zeros(dtype, shape)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_ones(
    dtype: u32,
    shape: *const u64,
    ndim: u32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let dtype = MlxDtype::from_raw(dtype)
            .ok_or_else(|| MlxError::validation(format!("invalid MLX dtype value `{dtype}`")))?;
        let shape = read_shape_arg(shape, ndim)?;
        mlx_array_ones(dtype, shape)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_full(
    dtype: u32,
    shape: *const u64,
    ndim: u32,
    value: *const c_void,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let dtype = MlxDtype::from_raw(dtype)
            .ok_or_else(|| MlxError::validation(format!("invalid MLX dtype value `{dtype}`")))?;
        let shape = read_shape_arg(shape, ndim)?;
        mlx_array_full(dtype, shape, value)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_from_buffer(
    buffer: *const c_void,
    dtype: u32,
    shape: *const u64,
    ndim: u32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let dtype = MlxDtype::from_raw(dtype)
            .ok_or_else(|| MlxError::validation(format!("invalid MLX dtype value `{dtype}`")))?;
        let shape = read_shape_arg(shape, ndim)?;
        mlx_array_from_buffer(buffer, dtype, shape)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_arange(
    dtype: u32,
    start: *const c_void,
    stop: *const c_void,
    step: *const c_void,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let dtype = MlxDtype::from_raw(dtype)
            .ok_or_else(|| MlxError::validation(format!("invalid MLX dtype value `{dtype}`")))?;
        mlx_array_arange(dtype, start, stop, step)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_reshape(
    a: *mut MpRtHeader,
    shape: *const u64,
    ndim: u32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let a = read_array_arg(a, "a")?;
        let shape = read_shape_arg(shape, ndim)?;
        mlx_array_reshape(&a, shape)
    })();
    write_array_result(out, out_err, result)
}

macro_rules! c_api_array_unary {
    ($fn_name:ident, $api_fn:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $fn_name(
            a: *mut MpRtHeader,
            out: *mut *mut MpRtHeader,
            out_err: *mut *mut MpRtHeader,
        ) -> i32 {
            let result = (|| {
                let a = read_array_arg(a, "a")?;
                $api_fn(&a)
            })();
            write_array_result(out, out_err, result)
        }
    };
}

macro_rules! c_api_array_binary {
    ($fn_name:ident, $api_fn:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $fn_name(
            a: *mut MpRtHeader,
            b: *mut MpRtHeader,
            out: *mut *mut MpRtHeader,
            out_err: *mut *mut MpRtHeader,
        ) -> i32 {
            let result = (|| {
                let a = read_array_arg(a, "a")?;
                let b = read_array_arg(b, "b")?;
                $api_fn(&a, &b)
            })();
            write_array_result(out, out_err, result)
        }
    };
}

macro_rules! c_api_array_reduce {
    ($fn_name:ident, $api_fn:ident) => {
        #[no_mangle]
        pub unsafe extern "C" fn $fn_name(
            a: *mut MpRtHeader,
            axis: i32,
            out: *mut *mut MpRtHeader,
            out_err: *mut *mut MpRtHeader,
        ) -> i32 {
            let result = (|| {
                let a = read_array_arg(a, "a")?;
                $api_fn(&a, axis)
            })();
            write_array_result(out, out_err, result)
        }
    };
}

c_api_array_unary!(mp_rt_mlx_array_transpose, mlx_array_transpose);

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_expand_dims(
    a: *mut MpRtHeader,
    axis: i32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let a = read_array_arg(a, "a")?;
        mlx_array_expand_dims(&a, axis)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_squeeze(
    a: *mut MpRtHeader,
    axis: i32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let a = read_array_arg(a, "a")?;
        mlx_array_squeeze(&a, axis)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_shape(
    a: *mut MpRtHeader,
    out_shape: *mut u64,
    out_ndim: *mut u32,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    clear_out_handle(out_err);
    if out_ndim.is_null() {
        set_out_error(
            out_err,
            &MlxError::validation("`out_ndim` must not be null"),
        );
        return -1;
    }

    let result = (|| {
        let a = read_array_arg(a, "a")?;
        mlx_array_shape(&a)
    })();

    match result {
        Ok(shape) => {
            let required = shape.len() as u32;
            let capacity = *out_ndim;
            *out_ndim = required;

            if out_shape.is_null() {
                set_success(out_err);
                return 0;
            }

            if capacity < required {
                set_out_error(
                    out_err,
                    &MlxError::validation(format!(
                        "`out_shape` capacity too small: need {required}, got {capacity}"
                    )),
                );
                return -1;
            }

            std::ptr::copy_nonoverlapping(shape.as_ptr(), out_shape, required as usize);
            set_success(out_err);
            0
        }
        Err(e) => {
            set_out_error(out_err, &e);
            -1
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_ndim(
    a: *mut MpRtHeader,
    out: *mut u32,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let a = read_array_arg(a, "a")?;
        mlx_array_ndim(&a)
    })();
    write_u32_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_size(
    a: *mut MpRtHeader,
    out: *mut u64,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let a = read_array_arg(a, "a")?;
        mlx_array_size(&a)
    })();
    write_u64_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_dtype(
    a: *mut MpRtHeader,
    out: *mut u32,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let a = read_array_arg(a, "a")?;
        mlx_array_dtype(&a)
    })();
    write_u32_result(out, out_err, result)
}

c_api_array_binary!(mp_rt_mlx_array_add, mlx_array_add);
c_api_array_binary!(mp_rt_mlx_array_sub, mlx_array_sub);
c_api_array_binary!(mp_rt_mlx_array_mul, mlx_array_mul);
c_api_array_binary!(mp_rt_mlx_array_div, mlx_array_div);
c_api_array_unary!(mp_rt_mlx_array_neg, mlx_array_neg);
c_api_array_unary!(mp_rt_mlx_array_abs, mlx_array_abs);
c_api_array_unary!(mp_rt_mlx_array_exp, mlx_array_exp);
c_api_array_unary!(mp_rt_mlx_array_log, mlx_array_log);
c_api_array_unary!(mp_rt_mlx_array_sqrt, mlx_array_sqrt);
c_api_array_binary!(mp_rt_mlx_array_pow, mlx_array_pow);
c_api_array_reduce!(mp_rt_mlx_array_sum, mlx_array_sum);
c_api_array_reduce!(mp_rt_mlx_array_mean, mlx_array_mean);
c_api_array_reduce!(mp_rt_mlx_array_max, mlx_array_max);
c_api_array_reduce!(mp_rt_mlx_array_min, mlx_array_min);
c_api_array_reduce!(mp_rt_mlx_array_argmax, mlx_array_argmax);
c_api_array_reduce!(mp_rt_mlx_array_argmin, mlx_array_argmin);

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_eval(
    a: *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let a = read_array_arg(a, "a")?;
        mlx_array_eval(&a)
    })();
    write_unit_result(out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_free(
    a: *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let a = read_array_arg(a, "a")?;
        mlx_array_free(a)
    })();
    write_unit_result(out_err, result)
}

c_api_array_binary!(mp_rt_mlx_linalg_matmul, mlx_linalg_matmul);
c_api_array_unary!(mp_rt_mlx_linalg_norm, mlx_linalg_norm);
c_api_array_unary!(mp_rt_mlx_linalg_inv, mlx_linalg_inv);

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_array_matmul(
    a: *mut MpRtHeader,
    b: *mut MpRtHeader,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    mp_rt_mlx_linalg_matmul(a, b, out, out_err)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_nn_linear(
    in_features: u64,
    out_features: u64,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    write_layer_result(out, out_err, mlx_nn_linear(in_features, out_features))
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_nn_forward(
    layer: *mut MpRtHeader,
    input: *mut MpRtHeader,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let layer = read_layer_arg(layer, "layer")?;
        let input = read_array_arg(input, "input")?;
        mlx_nn_forward(&layer, &input)
    })();
    write_array_result(out, out_err, result)
}

c_api_array_unary!(mp_rt_mlx_nn_relu, mlx_nn_relu);
c_api_array_unary!(mp_rt_mlx_nn_gelu, mlx_nn_gelu);
c_api_array_unary!(mp_rt_mlx_nn_sigmoid, mlx_nn_sigmoid);
c_api_array_unary!(mp_rt_mlx_nn_tanh, mlx_nn_tanh);

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_nn_softmax(
    x: *mut MpRtHeader,
    axis: i32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let x = read_array_arg(x, "x")?;
        mlx_nn_softmax(&x, axis)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_nn_free(
    layer: *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let layer = read_layer_arg(layer, "layer")?;
        mlx_nn_free(layer)
    })();
    write_unit_result(out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_optim_adam(
    lr: f32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    write_optim_result(out, out_err, mlx_optim_adam(lr))
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_optim_sgd(
    lr: f32,
    momentum: f32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    write_optim_result(out, out_err, mlx_optim_sgd(lr, momentum))
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_optim_adamw(
    lr: f32,
    wd: f32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    write_optim_result(out, out_err, mlx_optim_adamw(lr, wd))
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_optim_step(
    optim: *mut MpRtHeader,
    param: *mut MpRtHeader,
    grad: *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let optim = read_optim_arg(optim, "optim")?;
        let param = read_array_arg(param, "param")?;
        let grad = read_array_arg(grad, "grad")?;
        mlx_optim_step(&optim, &param, &grad)
    })();
    write_unit_result(out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_optim_free(
    optim: *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let optim = read_optim_arg(optim, "optim")?;
        mlx_optim_free(optim)
    })();
    write_unit_result(out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_grad(
    callable: *mut MpRtHeader,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    write_grad_result(out, out_err, mlx_grad(callable as *mut c_void))
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_value_and_grad(
    callable: *mut MpRtHeader,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    write_grad_result(out, out_err, mlx_value_and_grad(callable as *mut c_void))
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_random_normal(
    dtype: u32,
    shape: *const u64,
    ndim: u32,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let dtype = MlxDtype::from_raw(dtype)
            .ok_or_else(|| MlxError::validation(format!("invalid MLX dtype value `{dtype}`")))?;
        let shape = read_shape_arg(shape, ndim)?;
        mlx_random_normal(dtype, shape)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_random_uniform(
    dtype: u32,
    shape: *const u64,
    ndim: u32,
    low: f64,
    high: f64,
    out: *mut *mut MpRtHeader,
    out_err: *mut *mut MpRtHeader,
) -> i32 {
    let result = (|| {
        let dtype = MlxDtype::from_raw(dtype)
            .ok_or_else(|| MlxError::validation(format!("invalid MLX dtype value `{dtype}`")))?;
        let shape = read_shape_arg(shape, ndim)?;
        mlx_random_uniform(dtype, shape, low, high)
    })();
    write_array_result(out, out_err, result)
}

#[no_mangle]
pub unsafe extern "C" fn mp_rt_mlx_random_seed(seed: u64, out_err: *mut *mut MpRtHeader) -> i32 {
    write_unit_result(out_err, mlx_random_seed(seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    #[test]
    fn mlx_dtype_from_type_id() {
        assert_eq!(MlxDtype::from_type_id(14), Some(MlxDtype::F32));
        assert_eq!(MlxDtype::from_type_id(16), Some(MlxDtype::Bf16));
        assert_eq!(MlxDtype::from_type_id(999), None);
    }

    #[test]
    fn mlx_dtype_from_raw() {
        assert_eq!(MlxDtype::from_raw(10), Some(MlxDtype::F32));
        assert_eq!(MlxDtype::from_raw(11), Some(MlxDtype::Bf16));
        assert_eq!(MlxDtype::from_raw(42), None);
    }

    #[test]
    fn mlx_unavailable_returns_error() {
        if !mlx_is_available() {
            let result = mlx_array_zeros(MlxDtype::F32, &[2, 3]);
            assert!(result.is_err());
        }
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    fn require_mlx_runtime() -> bool {
        if !mlx_init() {
            eprintln!("skipping MLX execution test: mlx_init() failed");
            return false;
        }
        if !mlx_is_available() {
            eprintln!("skipping MLX execution test: MLX runtime is unavailable");
            return false;
        }
        true
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_exec_array_linalg_random_and_optimizer_paths() {
        if !require_mlx_runtime() {
            return;
        }

        let one = mlx_array_ones(MlxDtype::F32, &[2, 3]).expect("ones should succeed");
        let fill: f32 = 2.0;
        let full = mlx_array_full(MlxDtype::F32, &[2, 3], &fill as *const f32 as *const c_void)
            .expect("full should succeed");
        let add = mlx_array_add(&one, &full).expect("add should succeed");
        assert_eq!(
            mlx_array_shape(&add).expect("shape should succeed"),
            vec![2, 3]
        );
        assert_eq!(mlx_array_size(&add).expect("size should succeed"), 6);

        let reshaped = mlx_array_reshape(&add, &[3, 2]).expect("reshape should succeed");
        assert_eq!(
            mlx_array_shape(&reshaped).expect("shape should succeed"),
            vec![3, 2]
        );

        let rhs = mlx_array_ones(MlxDtype::F32, &[2, 4]).expect("rhs should succeed");
        let mat = mlx_array_matmul(&reshaped, &rhs).expect("matmul should succeed");
        assert_eq!(
            mlx_array_shape(&mat).expect("shape should succeed"),
            vec![3, 4]
        );

        let sum = mlx_array_sum(&mat, 1).expect("sum should succeed");
        let mean = mlx_array_mean(&mat, 0).expect("mean should succeed");
        mlx_array_eval(&sum).expect("sum eval should succeed");
        mlx_array_eval(&mean).expect("mean eval should succeed");

        let optim = mlx_optim_adam(1e-3).expect("adam should succeed");
        let grad = mlx_array_ones(MlxDtype::F32, &[3, 4]).expect("grad should succeed");
        mlx_optim_step(&optim, &mat, &grad).expect("optimizer step should succeed");

        let normal =
            mlx_random_normal(MlxDtype::F32, &[2, 2]).expect("random_normal should succeed");
        let uniform = mlx_random_uniform(MlxDtype::F32, &[2, 2], -1.0, 1.0)
            .expect("random_uniform should succeed");
        assert_eq!(
            mlx_array_shape(&normal).expect("shape should succeed"),
            vec![2, 2]
        );
        assert_eq!(
            mlx_array_shape(&uniform).expect("shape should succeed"),
            vec![2, 2]
        );
        mlx_array_eval(&normal).expect("normal eval should succeed");
        mlx_array_eval(&uniform).expect("uniform eval should succeed");

        if let Err(err) = mlx_random_seed(42) {
            assert_eq!(
                err.kind, 7,
                "random_seed should either succeed or report unsupported optional symbol"
            );
        }

        mlx_array_free(uniform).expect("free uniform");
        mlx_array_free(normal).expect("free normal");
        mlx_array_free(grad).expect("free grad");
        mlx_optim_free(optim).expect("free optimizer");
        mlx_array_free(mean).expect("free mean");
        mlx_array_free(sum).expect("free sum");
        mlx_array_free(mat).expect("free mat");
        mlx_array_free(rhs).expect("free rhs");
        mlx_array_free(reshaped).expect("free reshaped");
        mlx_array_free(add).expect("free add");
        mlx_array_free(full).expect("free full");
        mlx_array_free(one).expect("free one");
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn mlx_exec_nn_forward_path() {
        if !require_mlx_runtime() {
            return;
        }

        let input = mlx_array_ones(MlxDtype::F32, &[5, 4]).expect("input should succeed");
        let layer = mlx_nn_linear(4, 3).expect("nn linear should succeed");
        let output = mlx_nn_forward(&layer, &input).expect("nn forward should succeed");
        assert_eq!(
            mlx_array_shape(&output).expect("shape should succeed"),
            vec![5, 3]
        );
        mlx_array_eval(&output).expect("nn output eval should succeed");

        match mlx_nn_softmax(&output, 1) {
            Ok(softmax) => {
                assert_eq!(
                    mlx_array_shape(&softmax).expect("shape should succeed"),
                    vec![5, 3]
                );
                mlx_array_eval(&softmax).expect("softmax eval should succeed");
                mlx_array_free(softmax).expect("free softmax");
            }
            Err(err) => {
                assert_eq!(
                    err.kind, 7,
                    "nn_softmax should either succeed or report unsupported optional symbol"
                );
            }
        }

        mlx_array_free(output).expect("free output");
        mlx_nn_free(layer).expect("free layer");
        mlx_array_free(input).expect("free input");
    }
}
