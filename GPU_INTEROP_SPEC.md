# GPU Upgrade Interoperability Specification

**Purpose:** Defines ALL shared names, data structures, function signatures, TypeIds, diagnostic codes, and crate boundaries that every GPU-related crate must agree on. This is the single source of truth for cross-crate contracts.

---

## 1. Crate Naming & Dependencies

### 1.1 New Crates

| Crate | Path | Type | Dependencies |
|-------|------|------|-------------|
| `magpie_gpu` | `crates/magpie_gpu/` | lib (EXISTING, refactored) | `magpie_mpir`, `magpie_types`, `magpie_diag` |
| `magpie_gpu_spirv` | `crates/magpie_gpu_spirv/` | lib (NEW) | `magpie_gpu`, `magpie_mpir`, `magpie_types`, `magpie_diag` |
| `magpie_gpu_msl` | `crates/magpie_gpu_msl/` | lib (NEW) | `magpie_gpu`, `magpie_mpir`, `magpie_types`, `magpie_diag` |
| `magpie_gpu_ptx` | `crates/magpie_gpu_ptx/` | lib (NEW) | `magpie_gpu`, `magpie_mpir`, `magpie_types`, `magpie_diag` |
| `magpie_gpu_hip` | `crates/magpie_gpu_hip/` | lib (NEW) | `magpie_gpu`, `magpie_mpir`, `magpie_types`, `magpie_diag` |
| `magpie_gpu_wgsl` | `crates/magpie_gpu_wgsl/` | lib (NEW) | `magpie_gpu`, `magpie_mpir`, `magpie_types`, `magpie_diag` |
| `magpie_mlx` | `crates/magpie_mlx/` | lib (NEW) | `magpie_types`, `magpie_diag` |

### 1.2 Modified Crates

| Crate | Changes |
|-------|---------|
| `magpie_types` | Add `PrimType::Bf16`, `fixed_type_ids::BF16(16)`, TypeIds 33-40,50 |
| `magpie_ast` | Extend `AstGpuFnDecl` with `is_unsafe`, `workgroup`, `requires` |
| `magpie_parse` | Parse `unsafe gpu fn`, `workgroup()`, `requires()`, backend ids |
| `magpie_hir` | Add `dim: u8` to GPU intrinsic ops |
| `magpie_sema` | Propagate GPU metadata through lowering |
| `magpie_mpir` | Add `dim: u8` to GPU intrinsics, `gpu_meta: Option<MpirGpuMeta>` to `MpirFn`, 3D `GpuLaunch` |
| `magpie_codegen_llvm` | Use `dim` field, declare new runtime functions |
| `magpie_driver` | Route to backend emitters, propagate GPU metadata |
| `magpie_rt` | Unify TypeIds, add `gpu.TError`, new runtime functions, backend dispatch |
| `magpie_own` | Update pattern matches for new GPU op fields |
| `magpie_arc` | Update pattern matches for new GPU op fields |
| `magpie_mono` | Update pattern matches for new GPU op fields |
| `magpie_diag` | Support `MPG_XXX_NNNN` diagnostic code format |

---

## 2. TypeId Allocation (Canonical)

All crates MUST use these exact TypeId values. No crate may allocate its own.

```
TypeId(0)   = unit
TypeId(1)   = bool
TypeId(2)   = i8
TypeId(3)   = i16
TypeId(4)   = i32
TypeId(5)   = i64
TypeId(6)   = i128
TypeId(7)   = u8
TypeId(8)   = u16
TypeId(9)   = u32
TypeId(10)  = u64
TypeId(11)  = u128
TypeId(12)  = u1
TypeId(13)  = f16
TypeId(14)  = f32
TypeId(15)  = f64
TypeId(16)  = bf16              ← NEW
TypeId(17-19) = reserved
TypeId(20)  = Str
TypeId(21)  = StrBuilder
TypeId(22)  = Array<?>  (base)
TypeId(23)  = Map<?,?>  (base)
TypeId(24)  = TOption<?> (base)
TypeId(25)  = TResult<?,?> (base)
TypeId(26)  = TCallable<?> (base)
TypeId(27-29) = reserved
TypeId(30)  = gpu.TDevice
TypeId(31)  = gpu.TBuffer<?> (base)
TypeId(32)  = gpu.TFence
TypeId(33)  = gpu.TError        ← NEW
TypeId(34)  = gpu.ErrorKind     ← NEW
TypeId(35)  = gpu.TProfileSession ← NEW
TypeId(36)  = gpu.TProfileEvent   ← NEW
TypeId(37)  = gpu.TMemoryStats    ← NEW
TypeId(38)  = mlx.TArray<?> (base) ← NEW
TypeId(39)  = mlx.nn.TLayerHandle  ← NEW
TypeId(40)  = mlx.optim.TOptimizerHandle ← NEW
TypeId(41-49) = reserved
TypeId(50)  = gpu.TKernel (runtime-internal) ← MOVED from 9004
TypeId(1000+) = user types
```

### 2.1 Constants in `magpie_types/src/lib.rs` (`fixed_type_ids` module)

```rust
pub const BF16: TypeId = TypeId(16);
pub const GPU_ERROR: TypeId = TypeId(33);
pub const GPU_ERROR_KIND: TypeId = TypeId(34);
pub const GPU_PROFILE_SESSION: TypeId = TypeId(35);
pub const GPU_PROFILE_EVENT: TypeId = TypeId(36);
pub const GPU_MEMORY_STATS: TypeId = TypeId(37);
pub const MLX_ARRAY_BASE: TypeId = TypeId(38);
pub const MLX_LAYER_HANDLE: TypeId = TypeId(39);
pub const MLX_OPTIMIZER_HANDLE: TypeId = TypeId(40);
pub const GPU_KERNEL_INTERNAL: TypeId = TypeId(50);
```

### 2.2 Constants in `magpie_rt/src/lib.rs`

```rust
// DEPRECATED — remove these:
// const TYPE_ID_GPU_DEVICE_RT: u32 = 9001;  → use 30
// const TYPE_ID_GPU_BUFFER_RT: u32 = 9002;  → use 31
// const TYPE_ID_GPU_FENCE_RT: u32 = 9003;   → use 32
// const TYPE_ID_GPU_KERNEL_RT: u32 = 9004;  → use 50

// NEW canonical constants:
const TYPE_ID_GPU_DEVICE: u32 = 30;
const TYPE_ID_GPU_BUFFER: u32 = 31;
const TYPE_ID_GPU_FENCE: u32 = 32;
const TYPE_ID_GPU_ERROR: u32 = 33;
const TYPE_ID_GPU_ERROR_KIND: u32 = 34;
const TYPE_ID_GPU_PROFILE_SESSION: u32 = 35;
const TYPE_ID_GPU_PROFILE_EVENT: u32 = 36;
const TYPE_ID_GPU_MEMORY_STATS: u32 = 37;
const TYPE_ID_MLX_ARRAY: u32 = 38;
const TYPE_ID_MLX_LAYER_HANDLE: u32 = 39;
const TYPE_ID_MLX_OPTIMIZER_HANDLE: u32 = 40;
const TYPE_ID_GPU_KERNEL: u32 = 50;
```

---

## 3. GpuBackend Enum (Shared)

Defined in `magpie_gpu/src/lib.rs`, re-exported for all backend crates.

```rust
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    Spv  = 1,
    Msl  = 2,
    Ptx  = 3,
    Hip  = 4,
    Wgsl = 5,
}

impl GpuBackend {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "spv" => Some(Self::Spv),
            "msl" => Some(Self::Msl),
            "ptx" => Some(Self::Ptx),
            "hip" => Some(Self::Hip),
            "wgsl" => Some(Self::Wgsl),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Spv => "spv",
            Self::Msl => "msl",
            Self::Ptx => "ptx",
            Self::Hip => "hip",
            Self::Wgsl => "wgsl",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Spv => "Vulkan/SPIR-V",
            Self::Msl => "Metal/MSL",
            Self::Ptx => "CUDA/PTX",
            Self::Hip => "HIP/HSACO",
            Self::Wgsl => "WebGPU/WGSL",
        }
    }
}
```

---

## 4. BackendEmitter Trait (Shared)

Defined in `magpie_gpu/src/lib.rs`.

```rust
use magpie_mpir::{MpirFn, MpirTypeTable};
use magpie_diag::Diagnostic;

pub struct KernelLayout {
    pub params: Vec<KernelParam>,
    pub num_buffers: u32,
    pub push_const_size: u32,
}

pub trait BackendEmitter {
    fn backend_id(&self) -> GpuBackend;
    fn validate_kernel(&self, kernel: &MpirFn, types: &MpirTypeTable) -> Result<(), Vec<Diagnostic>>;
    fn compute_layout(&self, kernel: &MpirFn, types: &MpirTypeTable) -> Result<KernelLayout, String>;
    fn emit_kernel(&self, kernel: &MpirFn, types: &MpirTypeTable, layout: &KernelLayout) -> Result<Vec<u8>, String>;
    fn artifact_extension(&self) -> &str;
}
```

---

## 5. MpirGpuMeta Struct

Defined in `magpie_mpir/src/lib.rs`.

```rust
use magpie_gpu::GpuBackend;  // or inline the enum

#[derive(Debug, Clone)]
pub struct MpirGpuMeta {
    pub target: GpuBackend,
    pub workgroup: [u32; 3],
    pub is_unsafe: bool,
    pub requires: Vec<String>,
}
```

Added to `MpirFn`:
```rust
pub struct MpirFn {
    pub sid: Sid,
    pub name: String,
    pub params: Vec<(LocalId, TypeId)>,
    pub ret_ty: TypeId,
    pub blocks: Vec<MpirBlock>,
    pub locals: Vec<MpirLocalDecl>,
    pub is_async: bool,
    pub gpu_meta: Option<MpirGpuMeta>,  // NEW
}
```

---

## 6. MPIR GPU Op Changes

### 6.1 Value-returning ops (`MpirOp`)

```rust
// CHANGED — add dim: u8 field
GpuThreadId { dim: u8 },
GpuWorkgroupId { dim: u8 },
GpuWorkgroupSize { dim: u8 },
GpuGlobalId { dim: u8 },

// UNCHANGED
GpuBufferLoad { buf: MpirValue, idx: MpirValue },
GpuBufferLen { buf: MpirValue },
GpuShared { ty: TypeId, size: MpirValue },

// CHANGED — 3D grid/block
GpuLaunch {
    device: MpirValue,
    kernel: Sid,
    grid: [MpirValue; 3],
    block: [MpirValue; 3],
    args: Vec<MpirValue>,
},
GpuLaunchAsync {
    device: MpirValue,
    kernel: Sid,
    grid: [MpirValue; 3],
    block: [MpirValue; 3],
    args: Vec<MpirValue>,
},
```

### 6.2 Void ops (`MpirOpVoid`) — UNCHANGED
```rust
GpuBarrier,
GpuBufferStore { buf: MpirValue, idx: MpirValue, val: MpirValue },
```

### 6.3 Corresponding HIR changes (`HirOp`)

Same changes as MPIR:
```rust
GpuThreadId { dim: u8 },
GpuWorkgroupId { dim: u8 },
GpuWorkgroupSize { dim: u8 },
GpuGlobalId { dim: u8 },
GpuLaunch { device, kernel, grid: [HirValue; 3], block: [HirValue; 3], args },
GpuLaunchAsync { device, kernel, grid: [HirValue; 3], block: [HirValue; 3], args },
```

---

## 7. AST Changes

### 7.1 `AstGpuFnDecl` (in `magpie_ast/src/lib.rs`)

```rust
pub struct AstGpuFnDecl {
    pub inner: AstFnDecl,
    pub target: String,                  // "spv", "msl", "ptx", "hip", "wgsl"
    pub is_unsafe: bool,                 // NEW
    pub workgroup: Option<[u32; 3]>,     // NEW
    pub requires: Vec<String>,           // NEW
}
```

### 7.2 Parser tokens (in `magpie_parse/src/lib.rs`)

New keywords to recognize:
- `unsafe` (already exists for `unsafe fn`)
- `workgroup` (NEW keyword)
- `requires` (NEW keyword)
- Backend identifiers: `spv`, `msl`, `ptx`, `hip`, `wgsl` (parsed as idents)

Parse flow for `unsafe gpu fn`:
```
"unsafe" → "gpu" → "fn" → name → "(" params ")" → "->" → type
         → "target" → "(" ident ")"
         → ["workgroup" → "(" int "," int "," int ")"]
         → "requires" → "(" ident {"," ident} ")"
         → block
```

Parse flow for `gpu fn`:
```
"gpu" → "fn" → name → "(" params ")" → "->" → type
      → "target" → "(" ident ")"
      → ["workgroup" → "(" int "," int "," int ")"]
      → block
```

---

## 8. Runtime ABI — New C Functions

### 8.1 Naming Convention

All runtime GPU functions: `mp_rt_gpu_*`
All runtime MLX functions: `mp_rt_mlx_*`
All runtime profile functions: `mp_rt_gpu_profile_*`

### 8.2 Error Type ABI

```c
// MpRtGpuError — heap object with TypeId 33
typedef struct MpRtGpuErrorPayload {
    int32_t    kind;         // gpu.ErrorKind discriminant (0-11)
    MpRtHeader* backend;     // Str handle
    MpRtHeader* message;     // Str handle
    int32_t    code;         // vendor error code
} MpRtGpuErrorPayload;
```

### 8.3 New Runtime Functions (exact signatures)

```c
// Error construction/access
MpRtHeader* mp_rt_gpu_error_new(int32_t kind, const char* backend, const char* message, int32_t code);
void        mp_rt_gpu_error_drop(MpRtHeader* err);
int32_t     mp_rt_gpu_error_kind(MpRtHeader* err);
MpRtHeader* mp_rt_gpu_error_message(MpRtHeader* err);
MpRtHeader* mp_rt_gpu_error_backend(MpRtHeader* err);
int32_t     mp_rt_gpu_error_code(MpRtHeader* err);

// Unified device discovery
uint32_t    mp_rt_gpu_device_count_unified(void);
int32_t     mp_rt_gpu_device_default_unified(MpRtHeader** out_dev, MpRtHeader** out_err);
int32_t     mp_rt_gpu_device_by_index(uint32_t idx, MpRtHeader** out_dev, MpRtHeader** out_err);
MpRtHeader* mp_rt_gpu_device_name(MpRtHeader* dev);  // EXISTING — keep
int32_t     mp_rt_gpu_device_backends(MpRtHeader* dev, MpRtHeader** out_arr, MpRtHeader** out_err);

// Device capabilities
uint32_t    mp_rt_gpu_device_max_workgroup_size(MpRtHeader* dev);
uint32_t    mp_rt_gpu_device_max_shared_bytes(MpRtHeader* dev);
uint32_t    mp_rt_gpu_device_max_buffers(MpRtHeader* dev);
uint32_t    mp_rt_gpu_device_warp_size(MpRtHeader* dev);
uint64_t    mp_rt_gpu_device_memory_total(MpRtHeader* dev);
uint64_t    mp_rt_gpu_device_memory_available(MpRtHeader* dev);

// Extended buffer with hint
int32_t     mp_rt_gpu_buffer_new_hinted(
                MpRtHeader* dev, uint32_t elem_type_id, uint32_t elem_size,
                uint64_t len, uint32_t usage_flags, uint32_t hint,
                MpRtHeader** out_buf, MpRtHeader** out_err);

// EXISTING functions — signature change (out_errmsg → out_err, type changes)
int32_t     mp_rt_gpu_buffer_new(MpRtHeader* dev, uint32_t elem_type_id, uint64_t elem_size, uint64_t len, uint32_t usage_flags, MpRtHeader** out_buf, MpRtHeader** out_err);
int32_t     mp_rt_gpu_buffer_from_array(MpRtHeader* dev, MpRtHeader* host_arr, uint32_t usage_flags, MpRtHeader** out_buf, MpRtHeader** out_err);
int32_t     mp_rt_gpu_buffer_to_array(MpRtHeader* buf, MpRtHeader** out_arr, MpRtHeader** out_err);
int32_t     mp_rt_gpu_buffer_copy(MpRtHeader* src, MpRtHeader* dst, MpRtHeader** out_err);
int32_t     mp_rt_gpu_launch_sync(MpRtHeader* dev, uint64_t sid_hash, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz, const uint8_t* args_blob, uint64_t args_blob_len, MpRtHeader** out_err);
int32_t     mp_rt_gpu_launch_async(MpRtHeader* dev, uint64_t sid_hash, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz, const uint8_t* args_blob, uint64_t args_blob_len, MpRtHeader** out_fence, MpRtHeader** out_err);

// Profiling
int32_t     mp_rt_gpu_profile_begin(MpRtHeader* dev, MpRtHeader** out_session, MpRtHeader** out_err);
int32_t     mp_rt_gpu_profile_end(MpRtHeader* session, MpRtHeader** out_events, MpRtHeader** out_err);
int32_t     mp_rt_gpu_profile_mark_begin(MpRtHeader* session, MpRtHeader* name, MpRtHeader** out_err);
int32_t     mp_rt_gpu_profile_mark_end(MpRtHeader* session, MpRtHeader* name, MpRtHeader** out_err);
int32_t     mp_rt_gpu_profile_export_chrome(MpRtHeader* events, MpRtHeader* path, MpRtHeader** out_err);
int32_t     mp_rt_gpu_profile_available_counters(MpRtHeader* dev, MpRtHeader** out_arr);
int32_t     mp_rt_gpu_profile_enable_counters(MpRtHeader* session, MpRtHeader* counters, MpRtHeader** out_err);
int32_t     mp_rt_gpu_profile_read_counters(MpRtHeader* session, MpRtHeader** out_map, MpRtHeader** out_err);
int32_t     mp_rt_gpu_profile_memory_stats(MpRtHeader* dev, MpRtHeader** out_stats, MpRtHeader** out_err);

// MLX
int32_t     mp_rt_mlx_init(void);
int32_t     mp_rt_mlx_array_zeros(uint32_t dtype, const uint64_t* shape, uint32_t ndim, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_ones(uint32_t dtype, const uint64_t* shape, uint32_t ndim, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_full(uint32_t dtype, const uint64_t* shape, uint32_t ndim, const void* value, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_from_buffer(MpRtHeader* buf, const uint64_t* shape, uint32_t ndim, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_to_buffer(MpRtHeader* arr, MpRtHeader* dev, MpRtHeader** out_buf, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_add(MpRtHeader* a, MpRtHeader* b, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_sub(MpRtHeader* a, MpRtHeader* b, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_mul(MpRtHeader* a, MpRtHeader* b, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_div(MpRtHeader* a, MpRtHeader* b, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_matmul(MpRtHeader* a, MpRtHeader* b, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_reshape(MpRtHeader* a, const uint64_t* shape, uint32_t ndim, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_sum(MpRtHeader* a, int32_t axis, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_eval(MpRtHeader* a, MpRtHeader** out_err);
void        mp_rt_mlx_array_drop(MpRtHeader* arr);
int32_t     mp_rt_mlx_nn_linear(uint64_t in_features, uint64_t out_features, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_nn_forward(MpRtHeader* layer, MpRtHeader* input, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_optim_adam(float lr, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_optim_step(MpRtHeader* opt, MpRtHeader* params, MpRtHeader* grads, MpRtHeader** out_err);
int32_t     mp_rt_mlx_grad(MpRtHeader* fn_callable, MpRtHeader** out_grad_fn, MpRtHeader** out_err);
```

---

## 9. Diagnostic Code Contracts

### 9.1 Format

New format: `MPG_<BACKEND>_<NUMBER>` where:
- `<BACKEND>` ∈ {`CORE`, `SPV`, `MSL`, `PTX`, `HIP`, `WGSL`, `MLX`, `PROF`}
- `<NUMBER>` is a 4-digit code

### 9.2 Code Ranges

| Prefix | Range | Owner Crate |
|--------|-------|-------------|
| `MPG_CORE_1100-1107` | Kernel validation (existing, renamed) | `magpie_gpu` |
| `MPG_CORE_1200-1202` | Capability validation | `magpie_gpu` |
| `MPG_CORE_1300-1302` | Backend/tool errors | `magpie_gpu`, `magpie_driver` |
| `MPG_SPV_2001-2003` | SPIR-V backend | `magpie_gpu_spirv` |
| `MPG_MSL_1001,2001-2003` | MSL backend | `magpie_gpu_msl` |
| `MPG_PTX_2001-2003` | PTX backend | `magpie_gpu_ptx` |
| `MPG_HIP_2001-2002` | HIP backend | `magpie_gpu_hip` |
| `MPG_WGSL_1001-1006` | WGSL backend | `magpie_gpu_wgsl` |
| `MPG_MLX_1001-1004` | MLX | `magpie_mlx` |
| `MPG_PROF_1001-1003` | Profiling | `magpie_rt` |

### 9.3 Migration

Old `MPG1100` → new `MPG_CORE_1100` (etc., see SPEC_GPU_UPGRADE.md §13.1).

`magpie_diag` must accept BOTH formats during transition:
- Legacy: `/^MPG\d{4}$/`
- New: `/^MPG_[A-Z]+_\d{4}$/`

---

## 10. Kernel Registry IR Format

Generated by `magpie_gpu::generate_kernel_registry_ir()`.

### 10.1 Multi-blob entry struct

```llvm
%MpRtGpuKernelBlob = type { i8, ptr, i32 }
; { backend: u8, data: ptr, data_len: u32 }

%MpRtGpuParam = type { i8, i32, i32, i32 }
; { kind: u8, type_id: u32, offset_or_binding: u32, size: u32 }

%MpRtGpuKernelEntry = type { i64, i32, ptr, i32, ptr, i32, i32 }
; { sid_hash: u64, num_blobs: u32, blobs: ptr, num_params: u32, params: ptr, num_buffers: u32, push_const_size: u32 }
```

### 10.2 Registration function

```llvm
define void @mp_gpu_register_all_kernels() {
  call void @mp_rt_gpu_register_kernels(ptr @mp_gpu_kernel_registry, i32 <count>)
  ret void
}
```

---

## 11. CFG Structurizer Interface

Defined in `magpie_gpu/src/structurize.rs`, consumed by `magpie_gpu_msl` and `magpie_gpu_wgsl`.

```rust
pub enum StructuredNode {
    Block { label: BlockId, instrs: Vec<MpirInstr> },
    IfElse { cond: MpirValue, then_branch: Vec<StructuredNode>, else_branch: Vec<StructuredNode> },
    Loop { body: Vec<StructuredNode> },
    Break { depth: u32 },
    Continue { depth: u32 },
    Return,
    Assign { local: LocalId, value: MpirValue },
}

#[derive(Debug)]
pub struct StructurizeError {
    pub message: String,
    pub block_id: Option<BlockId>,
}

pub fn structurize_cfg(func: &MpirFn) -> Result<Vec<StructuredNode>, StructurizeError>;
```

---

## 12. File Naming Conventions

| Artifact | Path Pattern |
|----------|-------------|
| SPIR-V blob | `<output_dir>/gpu/<sid>.spv` |
| MSL source | `<output_dir>/gpu/<sid>.metal` |
| PTX source | `<output_dir>/gpu/<sid>.ptx` |
| LLVM IR (PTX) | `<output_dir>/gpu/<sid>.nvptx.ll` |
| HSACO blob | `<output_dir>/gpu/<sid>.hsaco` |
| LLVM IR (HIP) | `<output_dir>/gpu/<sid>.amdgcn.ll` |
| WGSL source | `<output_dir>/gpu/<sid>.wgsl` |
| Registry IR | `<output_dir>/gpu_registry.ll` |

---

## 13. LLVM Codegen Declarations

New runtime function declarations to add in `magpie_codegen_llvm/src/lib.rs`:

```llvm
; Error
declare ptr @mp_rt_gpu_error_new(i32, ptr, ptr, i32)
declare i32 @mp_rt_gpu_error_kind(ptr)
declare ptr @mp_rt_gpu_error_message(ptr)
declare ptr @mp_rt_gpu_error_backend(ptr)
declare i32 @mp_rt_gpu_error_code(ptr)

; Device discovery
declare i32 @mp_rt_gpu_device_count_unified()
declare i32 @mp_rt_gpu_device_default_unified(ptr, ptr)
declare i32 @mp_rt_gpu_device_by_index(i32, ptr, ptr)
declare i32 @mp_rt_gpu_device_backends(ptr, ptr, ptr)

; Device capabilities
declare i32 @mp_rt_gpu_device_max_workgroup_size(ptr)
declare i32 @mp_rt_gpu_device_max_shared_bytes(ptr)
declare i32 @mp_rt_gpu_device_max_buffers(ptr)
declare i32 @mp_rt_gpu_device_warp_size(ptr)
declare i64 @mp_rt_gpu_device_memory_total(ptr)
declare i64 @mp_rt_gpu_device_memory_available(ptr)

; Buffer with hint
declare i32 @mp_rt_gpu_buffer_new_hinted(ptr, i32, i32, i64, i32, i32, ptr, ptr)

; Profiling
declare i32 @mp_rt_gpu_profile_begin(ptr, ptr, ptr)
declare i32 @mp_rt_gpu_profile_end(ptr, ptr, ptr)
declare i32 @mp_rt_gpu_profile_mark_begin(ptr, ptr, ptr)
declare i32 @mp_rt_gpu_profile_mark_end(ptr, ptr, ptr)
declare i32 @mp_rt_gpu_profile_export_chrome(ptr, ptr, ptr)
declare i32 @mp_rt_gpu_profile_memory_stats(ptr, ptr, ptr)
```

---

## 14. PrimType::Bf16 Integration Points

| Crate | Change |
|-------|--------|
| `magpie_types` | Add `PrimType::Bf16`, `fixed_type_ids::BF16 = TypeId(16)`, layout `{size:2, align:2}`, `is_float()`, `is_signed()`, `bit_width()` |
| `magpie_ast` | `AstType::Prim` already handles string → PrimType; add `"bf16" => Bf16` mapping |
| `magpie_lex` | Recognize `bf16` as keyword/type token |
| `magpie_parse` | Handle `bf16` in type parsing |
| `magpie_sema` | Lower bf16 type, add explicit conversion ops `bf16.to_f32`, `f32.to_bf16` |
| `magpie_hir` | Add `Bf16ToF32`, `F32ToBf16` ops |
| `magpie_mpir` | Add `Bf16ToF32`, `F32ToBf16` ops |
| `magpie_codegen_llvm` | Map `TypeId(16) → "bfloat"`, codegen for conversion ops |
| `magpie_rt` | No change needed (bf16 is a value type, no runtime representation) |

---

## 15. gpu.TError Struct Layout (Runtime)

```rust
#[repr(C)]
pub struct MpRtGpuErrorPayload {
    pub kind: i32,           // gpu.ErrorKind discriminant (0-11)
    pub backend: *mut MpRtHeader,  // Str handle ("spv", "msl", etc.)
    pub message: *mut MpRtHeader,  // Str handle
    pub code: i32,           // vendor error code
}
```

Registered as:
```rust
MpRtTypeInfo {
    type_id: TYPE_ID_GPU_ERROR,  // 33
    flags: FLAG_HEAP | FLAG_HAS_DROP,
    payload_size: size_of::<MpRtGpuErrorPayload>(),
    payload_align: align_of::<MpRtGpuErrorPayload>(),
    drop_fn: Some(mp_rt_gpu_error_drop),
    debug_fqn: c"gpu.TError",
}
```

---

## 16. gpu.ErrorKind Values

```rust
pub const GPU_ERROR_DEVICE_LOST: i32 = 0;
pub const GPU_ERROR_OUT_OF_MEMORY: i32 = 1;
pub const GPU_ERROR_LAUNCH_FAILED: i32 = 2;
pub const GPU_ERROR_INVALID_KERNEL: i32 = 3;
pub const GPU_ERROR_BACKEND_UNAVAILABLE: i32 = 4;
pub const GPU_ERROR_BUFFER_ERROR: i32 = 5;
pub const GPU_ERROR_TIMEOUT_EXPIRED: i32 = 6;
pub const GPU_ERROR_UNSUPPORTED: i32 = 7;
pub const GPU_ERROR_COMPILATION_FAILED: i32 = 8;
pub const GPU_ERROR_DRIVER_ERROR: i32 = 9;
pub const GPU_ERROR_RESOURCE_EXHAUSTED: i32 = 10;
pub const GPU_ERROR_VALIDATION_FAILED: i32 = 11;
```

---

*End of Interoperability Specification*
