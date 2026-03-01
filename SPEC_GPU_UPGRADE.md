# Magpie GPU Expansion Specification

**Version:** 0.1 (included in Magpie v0.1)
**Date:** 2026-03-01
**Status:** Draft (Revision 1 — post-consensus review)
**Supersedes:** SPEC.md §31 (GPU Compute — SPIR-V only)
**Review:** Planner + Architect + Critic consensus review completed 2026-03-01

---

## Table of Contents

1. [Overview](#1-overview)
2. [Backends](#2-backends)
3. [Architecture](#3-architecture)
4. [Type System Additions](#4-type-system-additions)
5. [Kernel Model](#5-kernel-model)
6. [Capability System](#6-capability-system)
7. [Backend Codegen Specifications](#7-backend-codegen-specifications)
8. [MLX Integration](#8-mlx-integration)
9. [Runtime Architecture](#9-runtime-architecture)
10. [Kernel Registry](#10-kernel-registry)
11. [Profiling System](#11-profiling-system)
12. [Manifest Configuration](#12-manifest-configuration)
13. [Diagnostic Codes](#13-diagnostic-codes)
14. [Tool Discovery](#14-tool-discovery)
15. [Testing Strategy](#15-testing-strategy)
16. [Implementation Priority](#16-implementation-priority)
17. [Code Examples](#17-code-examples)

---

## 1. Overview

This specification expands Magpie's GPU support from a single SPIR-V/Vulkan prototype to a comprehensive multi-backend GPU compute system with six backends and an ML framework integration. All features described herein are part of Magpie v0.1.

### 1.1 Design Decisions Summary

| Decision | Choice |
|----------|--------|
| Dispatch layer | Raw vendor APIs (no wgpu abstraction) |
| Driver loading | Runtime dynamic loading (dlopen/LoadLibrary) |
| Kernel restrictions | Portable core + `unsafe` escape with capability annotations |
| Workgroup config | Per-backend specialization |
| Memory model | Unified copy-based with memory hints |
| Multi-target | Build-time backend selection |
| Error handling | `TResult<T, gpu.TError>` with 12-category `gpu.ErrorKind` enum |
| Fallback | Opt-in CPU fallback (three testing tiers) |
| Launch API | Always 3D (current) |
| Profiling | Integrated event-based tracing + vendor counters |
| Device discovery | Unified device list with backend query |
| MLX model | Separate `mlx` package (not in `std`), host array API |
| Registry | Multi-blob with runtime backend selection |

### 1.2 Backends

| Backend | Target | Shader Format | API | Priority |
|---------|--------|---------------|-----|----------|
| Vulkan Compute | `target(spv)` | SPIR-V 1.6 | Vulkan C API | Exists (upgrade) |
| Metal | `target(msl)` | MSL (native emission) | Metal.framework | 1st |
| MLX | Host API only | N/A (uses Metal internally) | MLX C++ API | 2nd |
| CUDA | `target(ptx)` | PTX via `llc` | CUDA Driver API | 3rd |
| HIP/ROCm | `target(hip)` | HSACO via `llc` | HIP Driver API (`hipModule*`) | 4th |
| WebGPU | `target(wgsl)` | WGSL (native emission) | WebGPU C API | 5th (second-class) |

---

## 2. Backends

### 2.1 Vulkan Compute — `target(spv)` (EXISTING, UPGRADED)

**Status:** Prototype exists in `magpie_gpu` (~1735 lines). Upgrade to production quality.

- SPIR-V 1.6, `GLCompute` execution model
- Runtime dispatches via Vulkan C API (`vkCreateInstance`, `vkCreateComputePipeline`, `vkCmdDispatch`)
- Buffer params → `StorageBuffer` descriptors (set 0, sequential bindings)
- Scalar params → push constants (std430 alignment, rounded to 16 bytes)
- Workgroup size via SPIR-V specialization constants (override at pipeline creation)
- `GlobalInvocationId` builtin for `gpu.global_id`

### 2.2 Metal — `target(msl)` (NEW)

- Native MSL emission from MPIR (no SPIRV-Cross dependency)
- Runtime dispatches via `Metal.framework` (`MTLDevice`, `MTLComputePipelineState`, `MTLComputeCommandEncoder`)
- Buffer params → Metal argument buffers (sequential indices)
- Scalar params → Metal constant buffer or `setBytes:`
- Workgroup size via `dispatchThreadgroups:threadsPerThreadgroup:` at dispatch time
- Address spaces: `device` (buffers), `constant` (scalars), `threadgroup` (shared memory), `thread` (locals)
- `thread_position_in_grid` for `gpu.global_id`
- `threadgroup_position_in_grid` for `gpu.workgroup_id`
- `threads_per_threadgroup` for `gpu.workgroup_size`
- `thread_position_in_threadgroup` for `gpu.thread_id`

**Metal-specific features (via `unsafe gpu fn`):**
- `gpu.metal.simd_shuffle` — SIMD group data exchange
- `gpu.metal.simd_sum` — SIMD group reduction
- `gpu.metal.simd_prefix_sum` — SIMD group prefix scan
- `gpu.metal.imageblock` — imageblock memory access
- `gpu.metal.tile_memory` — tile memory for tile shaders

### 2.3 CUDA — `target(ptx)` (NEW)

- Generates LLVM IR with `nvptx64-nvidia-cuda` triple
- Shells out to `llc -march=nvptx64` to produce PTX text
- Runtime dispatches via CUDA Driver API (`cuModuleLoadData`, `cuLaunchKernel`)
- Buffer params → device pointers passed as kernel arguments
- Scalar params → passed directly as kernel arguments
- Workgroup size via `cuLaunchKernel` grid/block dimensions
- Address spaces: 1 (global/device), 3 (shared), 4 (constant), 5 (local)
- `threadIdx.x/y/z` for `gpu.thread_id`
- `blockIdx.x/y/z` for `gpu.workgroup_id`
- `blockDim.x/y/z` for `gpu.workgroup_size`
- `blockIdx * blockDim + threadIdx` for `gpu.global_id`

### 2.4 HIP/ROCm — `target(hip)` (NEW)

- Generates LLVM IR with `amdgcn-amd-amdhsa` triple
- Shells out to `llc -march=amdgcn` + `lld` to produce HSACO ELF
- Runtime dispatches via HIP Driver API (`hipModuleLoadData`, `hipModuleLaunchKernel`)
- Structurally identical to CUDA backend — same LLVM IR → `llc` → blob pattern
- Buffer params → device pointers as kernel arguments
- Address spaces: 1 (global), 3 (shared/LDS), 4 (constant)
- Same thread/block/grid intrinsic mapping as CUDA

### 2.5 WebGPU — `target(wgsl)` (NEW, SECOND-CLASS)

**Second-class target.** Only supports a documented subset of `gpu fn` features.

- Native WGSL text emission from MPIR
- Runtime dispatches via WebGPU C API (`wgpuDeviceCreateComputePipeline`, `wgpuComputePassEncoderDispatchWorkgroups`)
- Buffer params → `@group(0) @binding(N) var<storage, read_write> buf: array<T>`
- Scalar params → `@group(0) @binding(N) var<uniform> params: Params` struct
- Workgroup size via `@workgroup_size(x, y, z)` attribute (compile-time only)
- `global_invocation_id` builtin for `gpu.global_id`

**WGSL limitations (compile-time enforced):**
- No pointers or pointer arithmetic
- No structs-of-arrays
- No unions or tagged unions
- Limited type system (no arbitrary value structs in buffers)
- Max workgroup size: 256 per dimension
- Max storage buffers: 8 per pipeline
- Compiler MUST emit `MPG_WGSL_1001` error for unsupported patterns

### 2.6 MLX — Host API Only (NEW)

MLX is NOT a kernel target. It provides a high-level array computation and ML API.
See [Section 8: MLX Integration](#8-mlx-integration) for full specification.

---

## 3. Architecture

### 3.1 Crate Structure

The GPU subsystem is organized as one crate per backend plus a shared core:

```
magpie_gpu/           — shared validation, kernel layout, BackendEmitter trait, registry IR
magpie_gpu_spirv/     — SPIR-V 1.6 binary emission (existing code, extracted)
magpie_gpu_msl/       — Metal Shading Language text emission
magpie_gpu_ptx/       — LLVM IR generation (nvptx64 triple) + llc invocation
magpie_gpu_hip/       — LLVM IR generation (amdgcn triple) + llc invocation
magpie_gpu_wgsl/      — WGSL text emission
magpie_mlx/           — MLX runtime bindings, array ops, nn, optim, grad
```

**Dependency graph:**
```
magpie_gpu (magpie_mpir, magpie_types, magpie_diag)  — core crate, renamed from magpie_gpu_core
  ├── magpie_gpu_spirv
  ├── magpie_gpu_msl
  ├── magpie_gpu_ptx
  ├── magpie_gpu_hip
  └── magpie_gpu_wgsl

magpie_mlx (magpie_types, magpie_diag)  — standalone, links MLX C++ API

magpie_rt  — runtime backends (Vulkan, Metal, CUDA, HIP, WebGPU dispatch)
```

### 3.2 BackendEmitter Trait

`magpie_gpu` (core) defines the shared trait that all backend crates implement:

```rust
pub trait BackendEmitter {
    /// Backend identifier (used in registry and diagnostics)
    fn backend_id(&self) -> GpuBackend;

    /// Validate kernel is compatible with this backend
    fn validate_kernel(&self, kernel: &MpirFn, types: &MpirTypeTable) -> Result<(), Vec<Diagnostic>>;

    /// Compute parameter layout for this backend
    fn compute_layout(&self, kernel: &MpirFn, types: &MpirTypeTable) -> Result<KernelLayout, String>;

    /// Emit kernel blob (SPIR-V binary, MSL text, PTX text, HSACO ELF, WGSL text)
    fn emit_kernel(&self, kernel: &MpirFn, types: &MpirTypeTable, layout: &KernelLayout)
        -> Result<Vec<u8>, String>;

    /// File extension for emitted kernel artifacts
    fn artifact_extension(&self) -> &str;
}
```

### 3.3 GpuBackend Enum

```rust
#[repr(u8)]
pub enum GpuBackend {
    Spv  = 1,  // Vulkan SPIR-V
    Msl  = 2,  // Metal Shading Language
    Ptx  = 3,  // CUDA PTX
    Hip  = 4,  // HIP/ROCm HSACO
    Wgsl = 5,  // WebGPU WGSL
}
```

### 3.4 MPIR GPU Intrinsic Changes (BREAKING)

The existing MPIR GPU intrinsics (`GpuThreadId`, `GpuWorkgroupId`, `GpuWorkgroupSize`, `GpuGlobalId`) are zero-operand variants that implicitly mean dimension 0. This spec requires parameterized dimension selection for 3D dispatch. The following breaking change to `MpirOp` is required in Phase 0:

**Current (v0.1 prototype):**
```rust
// magpie_mpir/src/lib.rs — current zero-operand variants
GpuThreadId,       // no fields — always dim 0
GpuWorkgroupId,    // no fields — always dim 0
GpuWorkgroupSize,  // no fields — always dim 0
GpuGlobalId,       // no fields — always dim 0
```

**Required (this spec):**
```rust
// magpie_mpir/src/lib.rs — parameterized variants
GpuThreadId { dim: u8 },       // dim ∈ {0, 1, 2}
GpuWorkgroupId { dim: u8 },    // dim ∈ {0, 1, 2}
GpuWorkgroupSize { dim: u8 },  // dim ∈ {0, 1, 2}
GpuGlobalId { dim: u8 },       // dim ∈ {0, 1, 2}
```

**Cascade of changes required:**
1. `magpie_mpir/src/lib.rs` — Add `dim: u8` field to all four variants; update operand collector (`operands()` currently returns `vec![]` for these)
2. `magpie_hir/src/lib.rs` — Add `dim` field to corresponding `HirOp` variants
3. `magpie_ast/src/lib.rs` — Add `dim` field to `AstOp` GPU variants
4. `magpie_parse/src/lib.rs` — Parse `{ dim=const.u32 N }` syntax for GPU intrinsics
5. `magpie_sema/src/lib.rs` — Lower AST dim → HIR dim
6. `magpie_driver/src/lib.rs` — Update HIR → MPIR lowering for dim field
7. `magpie_codegen_llvm/src/lib.rs` — Use `dim` instead of hardcoded `"0"` (line 1816)
8. `magpie_gpu/src/lib.rs` — Use `dim` for SPIR-V `GlobalInvocationId` component extraction (currently hardcoded via `ids.const_int_0`)
9. `magpie_own/src/lib.rs`, `magpie_arc/src/lib.rs`, `magpie_mono/src/lib.rs` — Update pattern matches

**MPIR verifier rule:** `dim` MUST be 0, 1, or 2. Values outside this range emit `MPS0001` (invalid operand).

**MpirFn GPU metadata (BREAKING):**

The `MpirFn` struct (`magpie_mpir/src/lib.rs:631-639`) must carry GPU kernel metadata so that `BackendEmitter` can access the target backend, workgroup configuration, and capability requirements:

```rust
// Current MpirFn (lines 631-639):
pub struct MpirFn {
    pub sid: Sid,
    pub name: String,
    pub params: Vec<(LocalId, TypeId)>,
    pub ret_ty: TypeId,
    pub blocks: Vec<MpirBlock>,
    pub locals: Vec<MpirLocalDecl>,
    pub is_async: bool,
}

// Required MpirFn:
pub struct MpirFn {
    pub sid: Sid,
    pub name: String,
    pub params: Vec<(LocalId, TypeId)>,
    pub ret_ty: TypeId,
    pub blocks: Vec<MpirBlock>,
    pub locals: Vec<MpirLocalDecl>,
    pub is_async: bool,
    pub gpu_meta: Option<MpirGpuMeta>,  // NEW — None for non-GPU fns
}

pub struct MpirGpuMeta {
    pub target: GpuBackend,        // spv, msl, ptx, hip, wgsl
    pub workgroup: [u32; 3],       // default (64, 1, 1) if annotation omitted
    pub is_unsafe: bool,           // true for `unsafe gpu fn`
    pub requires: Vec<String>,     // capability annotations
}
```

The driver's HIR → MPIR lowering must propagate `AstGpuFnDecl.{target, workgroup, is_unsafe, requires}` through HIR and into `MpirFn.gpu_meta`.

**GpuLaunch/GpuLaunchAsync 3D grid/block (BREAKING):**

The existing `MpirOp::GpuLaunch` uses scalar `groups`/`threads` fields but this spec requires 3D dispatch. The MPIR must change:

```rust
// Current (scalar):
GpuLaunch {
    device: MpirValue,
    kernel: Sid,
    groups: MpirValue,     // single scalar — 1D only
    threads: MpirValue,    // single scalar — 1D only
    args: Vec<MpirValue>,
}

// Required (3D):
GpuLaunch {
    device: MpirValue,
    kernel: Sid,
    grid: [MpirValue; 3],   // grid=[gx, gy, gz]
    block: [MpirValue; 3],  // block=[bx, by, bz]
    args: Vec<MpirValue>,
}
```

Same change applies to `GpuLaunchAsync`. The LLVM codegen must pass all 6 dimensions to `mp_rt_gpu_launch_sync`/`mp_rt_gpu_launch_async` instead of hardcoding `i32 1` for y and z.

### 3.5 TypeId Unification (BREAKING)

The compiler and runtime currently use **different TypeId values** for GPU types. This MUST be unified in Phase 0 before any new GPU types are added:

**Current mismatch:**

| Type | Compiler (`magpie_types`) | Runtime (`magpie_rt`) |
|------|---------------------------|----------------------|
| `gpu.TDevice` | `TypeId(30)` | `TYPE_ID_GPU_DEVICE_RT = 9001` |
| `gpu.TBuffer<?>` | `TypeId(31)` | `TYPE_ID_GPU_BUFFER_RT = 9002` |
| `gpu.TFence` | `TypeId(32)` | `TYPE_ID_GPU_FENCE_RT = 9003` |
| `gpu.TKernel` | — | `TYPE_ID_GPU_KERNEL_RT = 9004` |

**Resolution:** The runtime MUST adopt the compiler's TypeId values. The runtime constants change to:

```rust
// magpie_rt/src/lib.rs — updated to match fixed_type_ids
const TYPE_ID_GPU_DEVICE_RT: u32 = 30;   // was 9001
const TYPE_ID_GPU_BUFFER_RT: u32 = 31;   // was 9002
const TYPE_ID_GPU_FENCE_RT: u32 = 32;    // was 9003
const TYPE_ID_GPU_KERNEL_RT: u32 = 33;   // was 9004 — NOTE: conflicts with gpu.TError
```

**Conflict resolution:** `gpu.TKernel` (runtime-only type, not exposed to language) moves to TypeId 50. New GPU types start at 33:

| TypeId | Type | Notes |
|--------|------|-------|
| 30 | `gpu.TDevice` | unchanged |
| 31 | `gpu.TBuffer<?>` (base) | unchanged |
| 32 | `gpu.TFence` | unchanged |
| 33 | `gpu.TError` | NEW |
| 34 | `gpu.ErrorKind` | NEW |
| 35 | `gpu.TProfileSession` | NEW |
| 36 | `gpu.TProfileEvent` | NEW |
| 37 | `gpu.TMemoryStats` | NEW |
| 38 | `mlx.TArray<?>` (base) | NEW |
| 39 | `mlx.nn.TLayerHandle` | NEW |
| 40 | `mlx.optim.TOptimizerHandle` | NEW |
| 50 | `gpu.TKernel` (runtime-internal) | moved from 9004 |

TypeIds 17-29 and 41-49 are reserved for future use.

### 3.6 Runtime Dynamic Loading

All backend dispatch is loaded at runtime via platform-specific dynamic loading:

| Backend | Library | Symbol Prefix | Platforms |
|---------|---------|---------------|-----------|
| Vulkan | `libvulkan.so` / `libvulkan.dylib` / `vulkan-1.dll` | `vk*` | Linux, macOS (MoltenVK), Windows |
| Metal | `Metal.framework` | `MTL*` (Obj-C) | macOS, iOS |
| CUDA | `libcuda.so` / `nvcuda.dll` | `cu*` | Linux, Windows |
| HIP | `libamdhip64.so` | `hip*` | Linux |
| WebGPU | `libwgpu_native.so` / `wgpu_native.dll` | `wgpu*` | All (via wgpu-native) |
| MLX | `libmlx.dylib` | `mlx_*` | macOS |

**Loading behavior:**
- At runtime, `mp_rt_gpu_init()` attempts to `dlopen` the library for the configured backend
- If the library is not found, GPU operations return `Err(gpu.TError { kind = BackendUnavailable, ... })`
- No compile-time feature flags — all backend stubs are always compiled into `magpie_rt`
- Single binary works everywhere; backends activate based on available drivers

---

## 4. Type System Additions

### 4.1 New Primitive: `bf16`

Add `bf16` (bfloat16) as a first-class Magpie primitive type:

| Property | Value |
|----------|-------|
| TypeId | 16 (next available after f64=15) |
| LLVM type | `bfloat` |
| Size | 2 bytes |
| Alignment | 2 bytes |
| Copy | yes |
| Send | yes |
| Sync | yes |

**Operations supported:**
- `bf16.add`, `bf16.sub`, `bf16.mul`, `bf16.div` — arithmetic
- `bf16.to_f32`, `f32.to_bf16` — conversion
- `fcmp.*` — comparisons
- `const.bf16 <value>` — literals (parsed as f32, truncated to bf16)

**Rationale:** bf16 is essential for ML workloads (MLX, CUDA tensor cores) and GPU compute in general. It is useful beyond just MLX.

**Required Rust integration (`magpie_types`):**

```rust
// PrimType enum — add variant
pub enum PrimType {
    // ... existing variants ...
    Bf16,  // NEW — bfloat16
}

// PrimType methods — required updates
impl PrimType {
    pub fn is_float(&self) -> bool {
        matches!(self, F16 | F32 | F64 | Bf16)  // add Bf16
    }
    pub fn is_signed(&self) -> bool {
        matches!(self, I8 | I16 | I32 | I64 | I128 | F16 | F32 | F64 | Bf16)  // add Bf16
    }
    pub fn bit_width(&self) -> u32 {
        match self {
            Bf16 => 16,  // NEW
            // ... existing ...
        }
    }
}

// prim_layout() — add entry
Bf16 => Layout { size: 2, align: 2 },

// lookup_by_prim() — add mapping
PrimType::Bf16 => fixed_type_ids::BF16,  // TypeId(16)

// fixed_type_ids — add constant
pub const BF16: TypeId = TypeId(16);
```

**Coercion rules:**
- `bf16` does NOT implicitly coerce to or from any other type
- Explicit conversion only: `bf16.to_f32` and `f32.to_bf16`
- No `bf16 → f64` or `f16 → bf16` — must go through f32
- Arithmetic on bf16 follows IEEE round-to-nearest-even after each operation

**LLVM codegen:**
```rust
// magpie_codegen_llvm — llvm_ty() method
TypeId(16) => "bfloat",  // LLVM bfloat type (supported since LLVM 11)
```

**Backend support:**
| Backend | bf16 Support | Notes |
|---------|-------------|-------|
| SPIR-V | Extension `SPV_KHR_bfloat16` | Requires Vulkan 1.4+ driver |
| Metal/MSL | Native `bfloat` type | Apple Silicon M1+ |
| CUDA/PTX | `sm_80+` (`nv_bfloat16`) | Ampere and later |
| HIP/HSACO | Native `bfloat16_t` | CDNA1+ / RDNA3+ |
| WebGPU/WGSL | NOT SUPPORTED | Emit `MPG_WGSL_1003` |

**Reserved TypeIds:** 17, 18, 19 are reserved for future numeric types (e.g., `f8e4m3`, `f8e5m2`, `complex64`).

### 4.2 GPU Error Type: `gpu.TError`

New heap struct type replacing `Str` in GPU error positions:

```mp
heap struct gpu.TError {
    field kind: gpu.ErrorKind
    field backend: Str
    field message: Str
    field code: i32
}
```

**TypeId:** 33 (next available after gpu.TFence=32)

### 4.3 GPU Error Kind Enum: `gpu.ErrorKind`

```mp
enum gpu.ErrorKind {
    DeviceLost          ; 0 — GPU device became unavailable
    OutOfMemory         ; 1 — GPU or host memory exhausted
    LaunchFailed        ; 2 — kernel dispatch failed
    InvalidKernel       ; 3 — kernel blob is corrupt or incompatible
    BackendUnavailable  ; 4 — requested backend not available at runtime
    BufferError         ; 5 — buffer allocation, read, write, or copy failed
    TimeoutExpired      ; 6 — fence wait or device sync timed out
    Unsupported         ; 7 — operation not supported on this backend/device
    CompilationFailed   ; 8 — shader/kernel compilation error at runtime (JIT)
    DriverError         ; 9 — driver-level error or crash
    ResourceExhausted   ; 10 — descriptor, binding, or pipeline limits exceeded
    ValidationFailed    ; 11 — GPU validation/debug layer error
}
```

**TypeId:** 34

All GPU host APIs now return `TResult<T, gpu.TError>` instead of `TResult<T, Str>`.

**ABI migration from `Str` errors:**

The current runtime GPU functions use `MpRtHeader** out_errmsg` (a `Str` handle) for error output. This spec changes the error type to `gpu.TError`. The migration proceeds as follows:

1. **Phase 0:** Add `MpRtGpuError` struct and `mp_rt_gpu_error_new()` constructor to `magpie_rt`
2. **Phase 0:** Update all existing GPU runtime functions to return `MpRtGpuError*` via `out_err` instead of `Str` via `out_errmsg`. The parameter name changes from `out_errmsg` to `out_err` to signal the type change.
3. **Phase 0:** Update `magpie_codegen_llvm` GPU op lowering to expect `gpu.TError` return types instead of `Str`
4. **Phase 0:** Register `gpu.TError` (TypeId 33) in `MpRtTypeInfo` with `drop_fn = mp_rt_gpu_error_drop` that releases the inner `backend: Str` and `message: Str` fields

**Affected runtime functions (all in `magpie_rt/src/lib.rs`):**
```c
// Old signature:
int32_t mp_rt_gpu_device_default(MpRtHeader** out_dev, MpRtHeader** out_errmsg);
// New signature:
int32_t mp_rt_gpu_device_default(MpRtHeader** out_dev, MpRtHeader** out_err);
```

All 15+ GPU runtime functions with `out_errmsg` parameters follow the same pattern. The `int32_t` return value (0=success, -1=error) is unchanged.

### 4.4 Updated Host-Visible GPU Types

| Type | TypeId | Description |
|------|--------|-------------|
| `gpu.TDevice` | 30 | GPU device handle (unchanged) |
| `gpu.TBuffer<T>` | 31 (base) | Typed device buffer (unchanged) |
| `gpu.TFence` | 32 | Async completion handle (unchanged) |
| `gpu.TError` | 33 | Structured GPU error (NEW) |
| `gpu.ErrorKind` | 34 | Error category enum (NEW) |
| `gpu.TProfileSession` | 35 | Profiling session handle (NEW) |
| `gpu.TProfileEvent` | 36 | Profiling event handle (NEW) |

---

## 5. Kernel Model

### 5.1 Portable Core (Default)

By default, all `gpu fn` declarations use the **portable core** restriction set. These restrictions guarantee the kernel compiles and runs correctly on ALL backends (including WGSL second-class with its documented subset).

**Portable core restrictions (MUST):**
- No heap allocation (`New`, `ArrNew`, `MapNew`, `StrBuilderNew`, `CallableCapture`) → `MPG_CORE_1100`
- No ARC/ownership ops (`ArcRetain/Release`, `Share`, `CloneShared`) → `MPG_CORE_1101`
- No `TCallable`/dynamic dispatch (`CallIndirect`, `ArrMap/Filter/Reduce/Foreach`) → `MPG_CORE_1102`
- No recursion → `MPG_CORE_1103`
- No `Str`, `Array`, `Map`, `TCallable` types → `MPG_CORE_1104`–`MPG_CORE_1107`
- Allowed types: primitives (including `bf16`), value structs of primitives, `gpu.TBuffer<T>` handles
- All OOB checks MUST be explicit unless inside `unsafe {}`

### 5.2 Unsafe Escape: `unsafe gpu fn`

Non-portable GPU features are accessible via `unsafe gpu fn` with mandatory `requires()` annotations:

```mp
unsafe gpu fn @device_alloc_kernel(
    %buf: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(ptx) requires(device_malloc) {
bb0:
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    ; ... use device-side malloc ...
    ret
}
```

**Rules for `unsafe gpu fn`:**
1. MUST have a `requires(cap1, cap2, ...)` annotation listing all non-portable capabilities used
2. Compiler checks that each listed capability is supported by the declared `target()`
3. Compiler error `MPG_CORE_1200` if a capability is listed but the target doesn't support it
4. Compiler error `MPG_CORE_1201` if a non-portable feature is used without the corresponding `requires()` annotation
5. Compiler warning `MPG_CORE_1202` for every `unsafe gpu fn` reminding that the kernel is non-portable

### 5.3 Shared Memory (First-Class Portable)

`gpu.shared<N,T>` and `gpu.barrier` are **portable across all backends**. All 5 kernel backends MUST implement them.

```mp
gpu fn @reduce_sum(
    %input: gpu.TBuffer<f32>,
    %output: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(spv) {
bb0:
    %tid: u32 = gpu.thread_id { dim=const.u32 0 }
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %shared: ptr = gpu.shared<256, f32>
    ; gpu.shared<N,T> returns a `ptr` to workgroup-local memory in the
    ; backend-appropriate address space (threadgroup for Metal, addrspace(3)
    ; for CUDA/HIP, workgroup variable for SPIR-V/WGSL).
    ; This pointer is valid only within the current workgroup invocation.
    ; gpu.buffer_load and gpu.buffer_store accept both gpu.TBuffer<T>
    ; handles (for device buffers) and ptr values (for shared memory).
    ; NOTE: gpu.shared does NOT allocate a heap object — it references
    ; statically-sized workgroup memory. This is compatible with the
    ; "no heap allocation in gpu fn" restriction (MPG_CORE_1100).
    gpu.barrier
    ret
}
```

**Per-backend shared memory limits (compile-time enforced):**

| Backend | Max Shared Memory | Diagnostic |
|---------|-------------------|-----------|
| Vulkan/SPIR-V | 48 KB (typical, device-dependent) | `MPG_SPV_2001` |
| Metal/MSL | 32 KB (Apple GPU) | `MPG_MSL_2001` |
| CUDA/PTX | 48 KB (default), 96 KB (opt-in) | `MPG_PTX_2001` |
| HIP/HSACO | 64 KB (typical) | `MPG_HIP_2001` |
| WebGPU/WGSL | 16 KB (spec minimum) | `MPG_WGSL_2001` |

### 5.4 Workgroup Size (Per-Backend Specialization)

Workgroup size is specified via an optional annotation on `gpu fn` and realized differently per backend:

```mp
gpu fn @kernel_add(
    %in: gpu.TBuffer<f32>,
    %out: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(spv) workgroup(256, 1, 1) {
    ; ...
}
```

If `workgroup()` is omitted, the default is `(64, 1, 1)`.

**Per-backend realization:**

| Backend | Mechanism |
|---------|-----------|
| Vulkan/SPIR-V | Specialization constants (`OpSpecConstant`) overridable at pipeline creation |
| Metal/MSL | `dispatchThreadgroups:threadsPerThreadgroup:` at dispatch time |
| CUDA/PTX | `cuLaunchKernel` block dimensions |
| HIP/HSACO | `hipModuleLaunchKernel` block dimensions |
| WebGPU/WGSL | `@workgroup_size(x, y, z)` attribute (compile-time only, not runtime overridable) |

**Grammar extension:**
```ebnf
GpuFnDecl       = "gpu" "fn" FnName "(" Params? ")" "->" Type
                  "target" "(" BackendId ")"
                  [ "workgroup" "(" Int "," Int "," Int ")" ]
                  Block

UnsafeGpuFnDecl = "unsafe" "gpu" "fn" FnName "(" Params? ")" "->" Type
                  "target" "(" BackendId ")"
                  [ "workgroup" "(" Int "," Int "," Int ")" ]
                  "requires" "(" CapList ")"
                  Block

BackendId       = "spv" | "msl" | "ptx" | "hip" | "wgsl"
CapList         = Capability { "," Capability }
Capability      = Ident [ "." Ident ]
```

Note: `FnName`, `Params`, and `Type` are existing EBNF non-terminals from SPEC.md §7.2. `UnsafeGpuFnDecl` MUST be added to the `Decl` production alongside `GpuFnDecl`.

**Required AST struct changes (`magpie_ast/src/lib.rs`):**

```rust
// Current struct (lines 171-174):
pub struct AstGpuFnDecl {
    pub inner: AstFnDecl,
    pub target: String,
}

// Required struct:
pub struct AstGpuFnDecl {
    pub inner: AstFnDecl,
    pub target: String,                      // "spv", "msl", "ptx", "hip", "wgsl"
    pub is_unsafe: bool,                     // NEW — true for `unsafe gpu fn`
    pub workgroup: Option<[u32; 3]>,         // NEW — workgroup(x, y, z) annotation
    pub requires: Vec<String>,               // NEW — requires(cap1, cap2, ...) list
}

// AstDecl enum — add variant for unsafe:
pub enum AstDecl {
    // ... existing ...
    GpuFn(AstGpuFnDecl),       // existing — now handles both safe and unsafe
    // No separate UnsafeGpuFn variant; is_unsafe field distinguishes them
}
```

### 5.5 Launch API

The launch API remains 3D with unchanged syntax:

```mp
%result: TResult<unit, gpu.TError> = gpu.launch {
    device=%dev,
    kernel=@kernel_add,
    grid=[%gx, %gy, %gz],
    block=[%bx, %by, %bz],
    args=[%buf_in, %buf_out, %n]
}
```

- `gpu.launch` (synchronous) → `TResult<unit, gpu.TError>` — REQUIRED on all backends
- `gpu.launch_async` → `TResult<gpu.TFence, gpu.TError>` — OPTIONAL

Unused grid/block dimensions MUST be set to `const.u32 1`.

---

## 6. Capability System

### 6.1 Capability Vocabulary

Capabilities are used in `requires()` annotations on `unsafe gpu fn`:

**Hardware capabilities:**

| Capability | Description | SPV | MSL | PTX | HIP | WGSL |
|------------|-------------|-----|-----|-----|-----|------|
| `device_malloc` | Device-side dynamic memory allocation | no | no | yes | yes | no |
| `recursion` | Recursive function calls in kernel | no | no | yes | yes | no |
| `dynamic_parallelism` | Launch child kernels from device | no | no | yes | yes | no |
| `atomics_f32` | 32-bit floating-point atomics | yes* | yes | yes | yes | no |
| `atomics_f64` | 64-bit floating-point atomics | no | no | yes | yes | no |
| `subgroups` | Subgroup/warp/SIMD-group operations | yes | yes | yes | yes | no |
| `cooperative_groups` | Cooperative group primitives | no | no | yes | yes | no |
| `int64` | 64-bit integer in kernels | yes | yes | yes | yes | no |
| `f16_arithmetic` | Native half-precision arithmetic | yes | yes | yes | yes | yes |

**Resource limit capabilities:**

| Capability | Description | Query |
|------------|-------------|-------|
| `max_workgroup_size` | Maximum threads per workgroup | `gpu.host.@device_max_workgroup_size(dev)` |
| `max_shared_bytes` | Maximum shared memory per workgroup | `gpu.host.@device_max_shared_bytes(dev)` |
| `max_buffers` | Maximum buffer bindings per kernel | `gpu.host.@device_max_buffers(dev)` |
| `warp_size` | Warp/wavefront/SIMD-group width | `gpu.host.@device_warp_size(dev)` |

**Backend-specific capabilities (accessible only via `unsafe gpu fn` with corresponding target):**

| Capability | Backend | Description |
|------------|---------|-------------|
| `metal.simd_shuffle` | MSL | SIMD group shuffle operations |
| `metal.simd_reduce` | MSL | SIMD group reduction (sum, min, max) |
| `metal.simd_prefix` | MSL | SIMD group prefix scan |
| `metal.imageblock` | MSL | Imageblock memory access |
| `metal.tile_memory` | MSL | Tile shader memory |
| `cuda.tensor_cores` | PTX | Tensor core matrix operations |
| `cuda.cooperative_launch` | PTX | Cooperative kernel launch |
| `hip.wave64` | HIP | AMD 64-wide wavefront operations |

### 6.2 Compile-Time Capability Validation

The compiler validates `requires()` annotations at kernel declaration:

1. Parse `requires(cap1, cap2, ...)` from the `unsafe gpu fn` declaration
2. Look up the declared `target()` backend
3. For each capability, check the backend's supported capability set
4. Emit `MPG_CORE_1200` if any capability is unsupported on the target
5. During kernel body validation, check that all non-portable ops have corresponding `requires()` entries

---

## 7. Backend Codegen Specifications

### 7.1 SPIR-V Codegen (Existing — Upgrade)

**Crate:** `magpie_gpu_spirv` (extracted from current `magpie_gpu`)

**Changes from current implementation:**
- Extract from monolithic `magpie_gpu` into dedicated crate
- Implement `BackendEmitter` trait
- Add specialization constants for workgroup size (currently hardcoded 64×1×1)
- Add shared memory support (`OpVariable` with `Workgroup` storage class)
- Add `bf16` type support (SPIR-V `OpTypeFloat 16` with `Bfloat16` capability)
- Upgrade error returns from `Str` to `gpu.TError`

**SPIR-V structure per kernel:**
```
; Capability declarations
OpCapability Shader
OpCapability StorageBuffer16BitAccess  ; if bf16 used
; Memory model
OpMemoryModel Logical GLSL450
; Entry point
OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationId
; Execution mode with specialization constants
OpExecutionMode %main LocalSize %spec_x %spec_y %spec_z
; Decorations for buffers
OpDecorate %buf0 DescriptorSet 0
OpDecorate %buf0 Binding 0
; ... function body ...
```

### 7.2 MSL Codegen (New — Native Emission)

**Crate:** `magpie_gpu_msl`

Emits Metal Shading Language source code directly from MPIR. No SPIRV-Cross dependency.

**MSL structure per kernel:**
```metal
#include <metal_stdlib>
using namespace metal;

kernel void kernel_add(
    device float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = in[gid] + 1.0f;
}
```

**MPIR → MSL mapping:**

| MPIR Construct | MSL Output |
|---------------|------------|
| `gpu.TBuffer<T>` param | `device T* name [[buffer(N)]]` |
| Scalar param | `constant T& name [[buffer(N)]]` |
| `gpu.global_id { dim=0 }` | `uint gid [[thread_position_in_grid]]` |
| `gpu.thread_id { dim=0 }` | `uint tid [[thread_position_in_threadgroup]]` |
| `gpu.workgroup_id { dim=0 }` | `uint wgid [[threadgroup_position_in_grid]]` |
| `gpu.workgroup_size { dim=0 }` | `uint wgsz [[threads_per_threadgroup]]` |
| `gpu.shared<N,T>` | `threadgroup T shared_mem[N]` |
| `gpu.barrier` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `gpu.buffer_load<T> { buf, idx }` | `buf[idx]` |
| `gpu.buffer_store<T> { buf, idx, v }` | `buf[idx] = v` |
| `i.add { a, b }` | `a + b` |
| `cbr %cond bb1 bb2` | `if (cond) { ... } else { ... }` (structured) |
| `phi` | Local variable with assignments in predecessor blocks |

**Control flow structuring:**

MSL requires structured control flow (no arbitrary `goto`). The WGSL backend has the same requirement. Both backends share a **common CFG structurizer** implemented in `magpie_gpu` (core crate), avoiding duplicated effort.

**Shared CFG Structurizer (`magpie_gpu::structurize`)**

The structurizer converts MPIR's unstructured CFG (basic blocks with `br`/`cbr`/`ret` terminators) into a structured intermediate representation consumed by both MSL and WGSL emitters.

**Algorithm:** Relooper-style (based on Emscripten's Relooper by Alon Zakai, 2011), with the following phases:

1. **Dominator tree construction** — compute immediate dominators via the Lengauer-Tarjan algorithm
2. **Loop detection** — identify natural loops via back-edge detection in the dominator tree
3. **Region formation** — group blocks into `StructuredRegion` nodes using dominance frontiers
4. **Pattern matching** — recognize `if-then`, `if-then-else`, `while`, `do-while`, `break`, `continue` patterns
5. **Phi elimination** — convert SSA phi nodes into local variable assignments at predecessor block exits

**Structured intermediate representation:**
```rust
// magpie_gpu/src/structurize.rs

pub enum StructuredNode {
    /// Straight-line code (one basic block, no control flow)
    Block {
        label: BlockId,
        instrs: Vec<MpirInstr>,
    },
    /// if (cond) { then } else { else_ }
    IfElse {
        cond: MpirValue,
        then_branch: Vec<StructuredNode>,
        else_branch: Vec<StructuredNode>,
    },
    /// loop { body } — natural loop with break/continue
    Loop {
        body: Vec<StructuredNode>,
    },
    /// break out of enclosing Loop
    Break { depth: u32 },
    /// continue to top of enclosing Loop
    Continue { depth: u32 },
    /// return from kernel
    Return,
    /// Local variable assignment (from phi elimination)
    Assign {
        local: LocalId,
        value: MpirValue,
    },
}

pub fn structurize_cfg(func: &MpirFn) -> Result<Vec<StructuredNode>, StructurizeError>;
```

**Error handling:**
- **Irreducible CFGs** (CFGs that cannot be expressed with structured control flow without code duplication): emit `MPG_MSL_1001` / `MPG_WGSL_1001` at compile time. The structurizer does NOT attempt node splitting to make irreducible CFGs reducible — this is deferred to a future version.
- In practice, Magpie's SSA form with `cbr`/`br` terminators produces reducible CFGs for all normal programs. Irreducible CFGs can only arise from computed gotos or very unusual loop nesting, neither of which Magpie's surface syntax can produce.

**Estimated size:** ~1,500-2,000 lines for the structurizer + ~500 lines per backend for structured → text emission.

**Backend consumption:**
- MSL emitter: walks `Vec<StructuredNode>`, emits C-like Metal syntax with `if`/`else`/`while`
- WGSL emitter: walks `Vec<StructuredNode>`, emits WGSL syntax with `if`/`else`/`loop`/`break`/`continuing`

### 7.3 PTX Codegen (New — LLVM IR + llc)

**Crate:** `magpie_gpu_ptx`

Generates LLVM IR text with `nvptx64-nvidia-cuda` target triple, writes to `.ll` file, invokes `llc -march=nvptx64 -mcpu=sm_70` to produce PTX.

**LLVM IR structure per kernel:**
```llvm
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @kernel_add(float addrspace(1)* %in, float addrspace(1)* %out, i32 %n) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %bdim = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %gid = add i32 %tid, ...
  ; ... kernel body ...
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{void (float addrspace(1)*, float addrspace(1)*, i32)* @kernel_add, !"kernel", i32 1}
```

**MPIR → LLVM IR (nvptx64) mapping:**

| MPIR Construct | LLVM IR Output |
|---------------|----------------|
| `gpu.TBuffer<T>` param | `T addrspace(1)* %name` (global address space) |
| Scalar param | `T %name` (by value) |
| `gpu.global_id { dim=0 }` | Compute from `tid.x + ctaid.x * ntid.x` |
| `gpu.thread_id { dim=0 }` | `@llvm.nvvm.read.ptx.sreg.tid.x()` |
| `gpu.workgroup_id { dim=0 }` | `@llvm.nvvm.read.ptx.sreg.ctaid.x()` |
| `gpu.workgroup_size { dim=0 }` | `@llvm.nvvm.read.ptx.sreg.ntid.x()` |
| `gpu.shared<N,T>` | `@shared_mem = addrspace(3) global [N x T] undef` |
| `gpu.barrier` | `call void @llvm.nvvm.barrier0()` |
| `gpu.buffer_load<T>` | `load T, T addrspace(1)* %ptr` |
| `gpu.buffer_store<T>` | `store T %val, T addrspace(1)* %ptr` |

**llc invocation:**
```sh
llc -march=nvptx64 -mcpu=sm_70 -O2 -o kernel.ptx kernel.ll
```

### 7.4 HIP/HSACO Codegen (New — LLVM IR + llc)

**Crate:** `magpie_gpu_hip`

Structurally identical to PTX backend, but targets `amdgcn-amd-amdhsa` triple.

**LLVM IR structure per kernel:**
```llvm
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-..."
target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @kernel_add(float addrspace(1)* %in, float addrspace(1)* %out, i32 %n) #0 {
entry:
  %gid = call i32 @llvm.amdgcn.workitem.id.x()
  ; ... kernel body ...
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="64,256" }
```

**MPIR → LLVM IR (amdgcn) mapping:**

| MPIR Construct | LLVM IR Output |
|---------------|----------------|
| `gpu.thread_id { dim=0 }` | `@llvm.amdgcn.workitem.id.x()` |
| `gpu.workgroup_id { dim=0 }` | `@llvm.amdgcn.workgroup.id.x()` |
| `gpu.barrier` | `call void @llvm.amdgcn.s.barrier()` |
| `gpu.shared<N,T>` | `@shared_mem = addrspace(3) global [N x T] undef` |

**Build pipeline:**
```sh
llc -march=amdgcn -mcpu=gfx1030 -O2 -filetype=obj -o kernel.o kernel.ll
ld.lld -shared -o kernel.hsaco kernel.o
```

### 7.5 WGSL Codegen (New — Native Emission, Second-Class)

**Crate:** `magpie_gpu_wgsl`

Emits WGSL source text directly from MPIR.

**WGSL structure per kernel:**
```wgsl
@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    n: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn kernel_add(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n) { return; }
    output[gid.x] = input[gid.x] + 1.0;
}
```

**Documented subset (what WGSL supports):**
- Buffer load/store via array indexing
- Scalar parameters via uniform struct
- Shared memory via `var<workgroup>`
- Barrier via `workgroupBarrier()`
- All integer and float arithmetic
- Comparison and branching
- `gpu.global_id`, `gpu.thread_id`, `gpu.workgroup_id`, `gpu.workgroup_size`

**Not supported on WGSL (compiler errors):**
- Value structs in buffers → `MPG_WGSL_1001`
- Pointer operations → `MPG_WGSL_1002`
- `bf16` type → `MPG_WGSL_1003` (WGSL has `f16` but no `bf16`)
- Any `requires()` capability → `MPG_WGSL_1004`
- More than 8 storage buffers → `MPG_WGSL_1005`

---

## 8. MLX Integration

### 8.1 Overview

MLX is exposed as a **separate package** (`mlx`, not `std.mlx`) providing a host-side array computation and ML API. MLX is NOT a kernel target — it does not use `gpu fn`. Instead, it provides high-level operations that internally dispatch to Metal compute shaders via Apple's MLX C++ library.

**Package structure:**
```
mlx.array       — Array creation, manipulation, element-wise ops
mlx.linalg      — Linear algebra (matmul, solve, inv, svd)
mlx.nn          — Neural network layers
mlx.optim       — Optimizers
mlx.grad        — Automatic differentiation
mlx.random      — Random number generation
mlx.fft         — Fast Fourier transforms
```

### 8.2 MLX Array Type

`mlx.TArray<T>` is an **opaque heap handle** backed by an `mlx::core::array` via C++ FFI. It does NOT wrap `gpu.TBuffer<T>`.

> **Design rationale:** An earlier draft proposed wrapping `gpu.TBuffer<T>`, but this is architecturally unsound: (1) the ARC pass would emit retain/release on a buffer field that doesn't exist at runtime, corrupting reference counts; (2) MLX arrays manage their own shape/strides internally, so Magpie-side fields would go stale after reshape operations; (3) users could apply `gpu.buffer_load` to the inner buffer, which would crash because no actual GPU buffer exists. The opaque handle pattern avoids all three issues.

```mp
; mlx.TArray<T> is an opaque heap handle to mlx::core::array
; Shape, strides, and rank are queried via runtime FFI calls
; ARC manages the handle lifetime; drop_fn calls mlx_array_free

heap struct mlx.TArray<T> {
    ; Opaque — internal layout managed by magpie_mlx runtime
}
```

**Runtime representation:**
```c
// magpie_mlx runtime — C FFI wrapper around mlx::core::array
typedef struct MpRtMlxArrayPayload {
    void* mlx_array_ptr;  // Pointer to mlx::core::array (C++ object)
} MpRtMlxArrayPayload;

// Registered with MpRtTypeInfo:
//   type_id = 38
//   drop_fn = mp_rt_mlx_array_drop (calls mlx::core::array destructor)
//   flags = FLAG_HEAP | FLAG_HAS_DROP
```

**Shape and metadata accessors (runtime FFI calls, not struct field access):**
```mp
fn mlx.array.@shape<T: type>(%a: borrow mlx.TArray<T>) -> Array<u64>
fn mlx.array.@strides<T: type>(%a: borrow mlx.TArray<T>) -> Array<u64>
fn mlx.array.@ndim<T: type>(%a: borrow mlx.TArray<T>) -> u32
fn mlx.array.@size<T: type>(%a: borrow mlx.TArray<T>) -> u64
fn mlx.array.@dtype<T: type>(%a: borrow mlx.TArray<T>) -> u32
```

**Conversion between gpu.TBuffer<T> and mlx.TArray<T>:**
```mp
; Create an MLX array from a GPU buffer (copies data)
fn mlx.array.@from_buffer<T: type>(%buf: borrow gpu.TBuffer<T>, %shape: borrow Array<u64>) -> TResult<mlx.TArray<T>, gpu.TError>

; Export an MLX array to a GPU buffer (copies data)
fn mlx.array.@to_buffer<T: type>(%a: borrow mlx.TArray<T>, %dev: borrow gpu.TDevice) -> TResult<gpu.TBuffer<T>, gpu.TError>
```

These are explicit copy operations — there is no zero-cost view between the two types.

**Supported element types:** `f32`, `f64`, `f16`, `bf16`, `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `bool`

### 8.3 MLX Array Operations (`mlx.array`)

```mp
; Creation
fn mlx.array.@zeros<T: type>(%shape: borrow Array<u64>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@ones<T: type>(%shape: borrow Array<u64>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@full<T: type>(%shape: borrow Array<u64>, %value: T) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@from_buffer<T: type>(%buf: borrow gpu.TBuffer<T>, %shape: borrow Array<u64>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@arange<T: type>(%start: T, %stop: T, %step: T) -> TResult<mlx.TArray<T>, gpu.TError>

; Shape manipulation
fn mlx.array.@reshape<T: type>(%a: borrow mlx.TArray<T>, %shape: borrow Array<u64>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@transpose<T: type>(%a: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@expand_dims<T: type>(%a: borrow mlx.TArray<T>, %axis: i32) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@squeeze<T: type>(%a: borrow mlx.TArray<T>, %axis: i32) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@shape<T: type>(%a: borrow mlx.TArray<T>) -> Array<u64>
fn mlx.array.@ndim<T: type>(%a: borrow mlx.TArray<T>) -> u32

; Element-wise operations
fn mlx.array.@add<T: type>(%a: borrow mlx.TArray<T>, %b: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@sub<T: type>(%a: borrow mlx.TArray<T>, %b: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@mul<T: type>(%a: borrow mlx.TArray<T>, %b: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@div<T: type>(%a: borrow mlx.TArray<T>, %b: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@neg<T: type>(%a: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@abs<T: type>(%a: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@exp<T: type>(%a: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@log<T: type>(%a: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@sqrt<T: type>(%a: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@pow<T: type>(%a: borrow mlx.TArray<T>, %b: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>

; Reduction
fn mlx.array.@sum<T: type>(%a: borrow mlx.TArray<T>, %axis: i32) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@mean<T: type>(%a: borrow mlx.TArray<T>, %axis: i32) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@max<T: type>(%a: borrow mlx.TArray<T>, %axis: i32) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@min<T: type>(%a: borrow mlx.TArray<T>, %axis: i32) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.array.@argmax<T: type>(%a: borrow mlx.TArray<T>, %axis: i32) -> TResult<mlx.TArray<i32>, gpu.TError>
fn mlx.array.@argmin<T: type>(%a: borrow mlx.TArray<T>, %axis: i32) -> TResult<mlx.TArray<i32>, gpu.TError>

; Evaluation (materialize lazy computation graph)
fn mlx.array.@eval<T: type>(%a: borrow mlx.TArray<T>) -> TResult<unit, gpu.TError>
```

### 8.4 MLX Linear Algebra (`mlx.linalg`)

```mp
fn mlx.linalg.@matmul<T: type>(%a: borrow mlx.TArray<T>, %b: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.linalg.@norm<T: type>(%a: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.linalg.@inv<T: type>(%a: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.linalg.@svd<T: type>(%a: borrow mlx.TArray<T>) -> TResult<mlx.TSvdResult<T>, gpu.TError>
fn mlx.linalg.@solve<T: type>(%a: borrow mlx.TArray<T>, %b: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
```

### 8.5 MLX Neural Network Layers (`mlx.nn`)

Neural network layers use a **layered architecture**: opaque runtime handles internally, wrapped in typed Magpie structs that expose parameters.

**Runtime layer (opaque):**
```mp
; Opaque handle for any nn layer
heap struct mlx.nn.TLayerHandle {
    ; ... opaque FFI pointer to mlx::nn module ...
}
```

**Magpie wrapper layer (typed):**
```mp
; Linear layer
heap struct mlx.nn.TLinear {
    field handle: mlx.nn.TLayerHandle
    field in_features: u64
    field out_features: u64
}

fn mlx.nn.@linear(%in_features: u64, %out_features: u64) -> TResult<mlx.nn.TLinear, gpu.TError>
fn mlx.nn.@linear_forward(%layer: borrow mlx.nn.TLinear, %input: borrow mlx.TArray<f32>) -> TResult<mlx.TArray<f32>, gpu.TError>
fn mlx.nn.@linear_weight(%layer: borrow mlx.nn.TLinear) -> mlx.TArray<f32>
fn mlx.nn.@linear_bias(%layer: borrow mlx.nn.TLinear) -> mlx.TArray<f32>
```

**Available layers:**

| Layer | Constructor | Parameters |
|-------|------------|------------|
| `mlx.nn.TLinear` | `mlx.nn.@linear(in, out)` | weight, bias |
| `mlx.nn.TConv1d` | `mlx.nn.@conv1d(in_ch, out_ch, kernel_size)` | weight, bias |
| `mlx.nn.TConv2d` | `mlx.nn.@conv2d(in_ch, out_ch, kernel_size)` | weight, bias |
| `mlx.nn.TLayerNorm` | `mlx.nn.@layer_norm(dims)` | weight, bias |
| `mlx.nn.TBatchNorm` | `mlx.nn.@batch_norm(num_features)` | weight, bias, running_mean, running_var |
| `mlx.nn.TRnn` | `mlx.nn.@rnn(input_size, hidden_size)` | weight_ih, weight_hh, bias |
| `mlx.nn.TLstm` | `mlx.nn.@lstm(input_size, hidden_size)` | weight_ih, weight_hh, bias_ih, bias_hh |
| `mlx.nn.TGru` | `mlx.nn.@gru(input_size, hidden_size)` | weight_ih, weight_hh, bias_ih, bias_hh |
| `mlx.nn.TTransformerEncoderLayer` | `mlx.nn.@transformer_encoder_layer(d_model, nhead)` | self_attn, ff weights |
| `mlx.nn.TEmbedding` | `mlx.nn.@embedding(num_embeddings, dim)` | weight |
| `mlx.nn.TMultiHeadAttention` | `mlx.nn.@multi_head_attention(d_model, nhead)` | q/k/v/out projections |

**Activation functions (stateless):**
```mp
fn mlx.nn.@relu<T: type>(%x: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.nn.@gelu<T: type>(%x: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.nn.@sigmoid<T: type>(%x: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.nn.@tanh<T: type>(%x: borrow mlx.TArray<T>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.nn.@softmax<T: type>(%x: borrow mlx.TArray<T>, %axis: i32) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.nn.@dropout<T: type>(%x: borrow mlx.TArray<T>, %p: f32) -> TResult<mlx.TArray<T>, gpu.TError>
```

### 8.6 MLX Automatic Differentiation (`mlx.grad`)

Autograd uses `mlx.grad` as a host operation that wraps a `TCallable` and returns a new `TCallable` computing gradients. This uses MLX's runtime tracing — no compiler-generated adjoints.

```mp
; Compute gradient of a scalar-valued function
fn mlx.grad.@grad(%fn: TCallable<TSigLoss>) -> TCallable<TSigLossGrad>

; Value-and-gradient (returns both the function value and its gradient)
fn mlx.grad.@value_and_grad(%fn: TCallable<TSigLoss>) -> TCallable<TSigValueAndGrad>

; Vector-Jacobian product
fn mlx.grad.@vjp<T: type>(%fn: TCallable<TSigFwd>, %primals: borrow mlx.TArray<T>, %cotangents: borrow mlx.TArray<T>)
    -> TResult<mlx.TGradResult<T>, gpu.TError>

; Jacobian-vector product
fn mlx.grad.@jvp<T: type>(%fn: TCallable<TSigFwd>, %primals: borrow mlx.TArray<T>, %tangents: borrow mlx.TArray<T>)
    -> TResult<mlx.TGradResult<T>, gpu.TError>
```

**How it works:**
1. User defines a loss function as a `TCallable` that takes `mlx.TArray<T>` and returns `mlx.TArray<T>` (scalar)
2. `mlx.grad.@grad(loss_fn)` returns a new `TCallable` that, when called, traces the computation graph and computes gradients via reverse-mode AD
3. All `mlx.array.*` and `mlx.nn.*` ops are traceable — they participate in the computation graph
4. `mlx.array.@eval()` materializes lazy computations

### 8.7 MLX Optimizers (`mlx.optim`)

```mp
; Optimizer types
heap struct mlx.optim.TAdam {
    field handle: mlx.optim.TOptimizerHandle
    field learning_rate: f32
}

heap struct mlx.optim.TSgd {
    field handle: mlx.optim.TOptimizerHandle
    field learning_rate: f32
    field momentum: f32
}

heap struct mlx.optim.TAdamW {
    field handle: mlx.optim.TOptimizerHandle
    field learning_rate: f32
    field weight_decay: f32
}

; Constructors
fn mlx.optim.@adam(%lr: f32) -> TResult<mlx.optim.TAdam, gpu.TError>
fn mlx.optim.@sgd(%lr: f32, %momentum: f32) -> TResult<mlx.optim.TSgd, gpu.TError>
fn mlx.optim.@adamw(%lr: f32, %wd: f32) -> TResult<mlx.optim.TAdamW, gpu.TError>

; Update step (generic over optimizer type)
fn mlx.optim.@step_adam(%opt: mutborrow mlx.optim.TAdam, %params: mutborrow Array<mlx.TArray<f32>>, %grads: borrow Array<mlx.TArray<f32>>) -> TResult<unit, gpu.TError>
fn mlx.optim.@step_sgd(%opt: mutborrow mlx.optim.TSgd, %params: mutborrow Array<mlx.TArray<f32>>, %grads: borrow Array<mlx.TArray<f32>>) -> TResult<unit, gpu.TError>
fn mlx.optim.@step_adamw(%opt: mutborrow mlx.optim.TAdamW, %params: mutborrow Array<mlx.TArray<f32>>, %grads: borrow Array<mlx.TArray<f32>>) -> TResult<unit, gpu.TError>
```

### 8.8 MLX Random (`mlx.random`)

```mp
fn mlx.random.@normal<T: type>(%shape: borrow Array<u64>) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.random.@uniform<T: type>(%shape: borrow Array<u64>, %low: T, %high: T) -> TResult<mlx.TArray<T>, gpu.TError>
fn mlx.random.@bernoulli(%shape: borrow Array<u64>, %p: f32) -> TResult<mlx.TArray<bool>, gpu.TError>
fn mlx.random.@seed(%s: u64) -> unit
```

### 8.9 MLX Platform Requirements

- **macOS only** — MLX requires Apple Silicon (M1 or later)
- Runtime: `libmlx.dylib` loaded via `dlopen`
- If MLX is unavailable, all `mlx.*` functions return `Err(gpu.TError { kind = BackendUnavailable, backend = "mlx", ... })`
- MLX operations use Apple's unified memory — no explicit host↔device transfers needed

---

## 9. Runtime Architecture

### 9.1 Runtime Initialization

```c
// Called at program startup (from generated main)
void mp_rt_gpu_init(void);
```

`mp_rt_gpu_init` performs:
1. Attempt to `dlopen` the configured backend library
2. Load all function pointers into a dispatch table
3. If loading fails, set backend status to `BackendUnavailable`
4. Register kernel blobs from the embedded registry

### 9.2 Unified Device Discovery

Devices are presented as a unified, deduplicated list across all available backends:

```mp
; Device discovery (unified across backends)
fn gpu.host.@device_count() -> u32
fn gpu.host.@device_default() -> TResult<gpu.TDevice, gpu.TError>
fn gpu.host.@device_by_index(%idx: u32) -> TResult<gpu.TDevice, gpu.TError>
fn gpu.host.@device_name(%dev: borrow gpu.TDevice) -> Str
fn gpu.host.@device_backends(%dev: borrow gpu.TDevice) -> Array<Str>

; Device capability queries
fn gpu.host.@device_max_workgroup_size(%dev: borrow gpu.TDevice) -> u32
fn gpu.host.@device_max_shared_bytes(%dev: borrow gpu.TDevice) -> u32
fn gpu.host.@device_max_buffers(%dev: borrow gpu.TDevice) -> u32
fn gpu.host.@device_warp_size(%dev: borrow gpu.TDevice) -> u32
fn gpu.host.@device_memory_total(%dev: borrow gpu.TDevice) -> u64
fn gpu.host.@device_memory_available(%dev: borrow gpu.TDevice) -> u64
```

**Deduplication rules:**
- Same physical GPU appearing in multiple backends (e.g., Apple GPU via both Metal and MoltenVK) is reported once
- `gpu.host.@device_backends(dev)` returns all backends that can target this device (e.g., `["msl", "spv"]`)
- `gpu.host.@device_default()` picks the best backend automatically:
  - macOS: Metal
  - Linux with NVIDIA GPU: CUDA
  - Linux with AMD GPU: HIP
  - Linux with no proprietary driver: Vulkan
  - WebAssembly: WebGPU
  - Fallback: CPU (if opt-in enabled)

### 9.3 Memory Model with Hints

The buffer API extends the current model with optional memory hints:

```mp
; Memory placement hints
enum gpu.MemoryHint {
    Auto            ; 0 — runtime decides (default)
    DeviceLocal     ; 1 — prefer GPU-only memory (fastest for compute)
    Unified         ; 2 — prefer unified/shared memory (zero-copy on Apple Silicon)
    Staging         ; 3 — prefer host-visible memory (optimal for transfers)
    Cached          ; 4 — prefer cached host memory (optimal for CPU readback)
}

; gpu.MemoryHint is a value enum with no heap allocation.
; It is lowered as a plain u32 discriminant — no TypeId needed.
; The runtime ABI passes it as `uint32_t hint`.

; Extended buffer creation with hint
fn gpu.host.@buffer_new<T: type>(
    %dev: borrow gpu.TDevice,
    %len: u64,
    %usage_flags: u32,
    %hint: gpu.MemoryHint
) -> TResult<gpu.TBuffer<T>, gpu.TError>

; Existing APIs (unchanged signatures, updated error type)
fn gpu.host.@buffer_from_array<T: type>(%dev: borrow gpu.TDevice, %src: borrow Array<T>, %usage_flags: u32) -> TResult<gpu.TBuffer<T>, gpu.TError>
fn gpu.host.@buffer_to_array<T: type>(%buf: borrow gpu.TBuffer<T>) -> TResult<Array<T>, gpu.TError>
fn gpu.host.@buffer_len<T: type>(%buf: borrow gpu.TBuffer<T>) -> u64
fn gpu.host.@buffer_copy<T: type>(%src: borrow gpu.TBuffer<T>, %dst: borrow gpu.TBuffer<T>) -> TResult<unit, gpu.TError>
```

**Hint behavior per backend:**

| Hint | Vulkan | Metal | CUDA | HIP | WebGPU |
|------|--------|-------|------|-----|--------|
| `Auto` | Device-local | Unified | Device | Device | Storage |
| `DeviceLocal` | `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` | `MTLStorageModePrivate` | `cuMemAlloc` | `hipMalloc` | Storage |
| `Unified` | Host-visible + device-local | `MTLStorageModeShared` (zero-copy) | `cuMemAllocManaged` | `hipMallocManaged` | N/A → Auto |
| `Staging` | Host-visible + coherent | `MTLStorageModeShared` | Pinned host | Pinned host | Map buffer |
| `Cached` | Host-cached | `MTLStorageModeShared` | `cuMemAllocHost` | `hipHostMalloc` | Map buffer |

### 9.4 Error Handling

All GPU runtime functions return `TResult<T, gpu.TError>`.

**Runtime ABI for gpu.TError:**
```c
typedef struct MpRtGpuError {
    MpRtHeader header;       // ARC-managed heap object
    int32_t    kind;         // gpu.ErrorKind discriminant (0-11)
    MpRtHeader* backend;     // Str: "spv", "msl", "ptx", "hip", "wgsl", "mlx"
    MpRtHeader* message;     // Str: human-readable error message
    int32_t    code;         // vendor-specific error code (VkResult, cudaError_t, etc.)
} MpRtGpuError;
```

### 9.5 Opt-In CPU Fallback

**Default behavior:** GPU operations return `Err(gpu.TError { kind = BackendUnavailable, ... })` when no GPU is available.

**Opt-in CPU fallback:**
```toml
# Magpie.toml
[gpu]
fallback = "cpu"  # Enable CPU simulation for gpu.launch
```

Or per-launch:
```mp
%result: TResult<unit, gpu.TError> = gpu.launch {
    device=%dev,
    kernel=@kernel_add,
    grid=[%gx, const.u32 1, const.u32 1],
    block=[const.u32 64, const.u32 1, const.u32 1],
    args=[%buf_in, %buf_out, %n],
    fallback=cpu
}
```

When CPU fallback is active, the runtime interprets the kernel on the CPU, simulating thread/workgroup IDs by iterating over the grid.

---

## 10. Kernel Registry

### 10.1 Multi-Blob Format

The kernel registry supports multiple blob entries per kernel for future flexibility, even though build-time selection produces only one blob type per build:

```c
typedef struct MpRtGpuKernelBlob {
    uint8_t   backend;      // GpuBackend discriminant
    const uint8_t* data;    // Pointer to blob data
    uint32_t  data_len;     // Blob size in bytes
} MpRtGpuKernelBlob;

typedef struct MpRtGpuKernelEntry {
    uint64_t  sid_hash;     // FNV-1a hash of kernel SID
    uint32_t  num_blobs;    // Number of blob entries (typically 1)
    const MpRtGpuKernelBlob* blobs;  // Array of blobs
    uint32_t  num_params;   // Parameter count
    const MpRtGpuParam* params;  // Parameter layout
    uint32_t  num_buffers;  // Buffer parameter count
    uint32_t  push_const_size;  // Push constant block size (bytes)
} MpRtGpuKernelEntry;
```

**Runtime blob selection:**
1. Iterate `blobs` array
2. Find entry matching the active backend
3. If no match, return `Err(gpu.TError { kind = InvalidKernel, ... })`

### 10.2 Registry IR Generation

The driver generates `<stem>.gpu_registry.ll` containing:
- Constant byte arrays for each kernel blob
- `MpRtGpuKernelEntry` array as a global constant
- `@mp_gpu_register_all_kernels` function that calls `mp_rt_gpu_register_kernels`

---

## 11. Profiling System

### 11.1 Overview

Magpie provides an integrated GPU profiling system with three layers:
1. **Event-based tracing** — Timeline of all GPU operations with timestamps
2. **Vendor counters** — Hardware performance counters (when available)
3. **Export** — Chrome trace format (JSON) for visualization

### 11.2 Profiling Types

```mp
heap struct gpu.TProfileSession {
    ; Opaque handle to profiling session
}

heap struct gpu.TProfileEvent {
    field name: Str
    field start_ns: u64
    field end_ns: u64
    field metadata: Map<Str, Str>
}
```

**TypeIds:** `gpu.TProfileSession = 35`, `gpu.TProfileEvent = 36`

### 11.3 Profiling API

```mp
; Session management
fn gpu.profile.@begin(%dev: borrow gpu.TDevice) -> TResult<gpu.TProfileSession, gpu.TError>
fn gpu.profile.@end(%session: gpu.TProfileSession) -> TResult<Array<gpu.TProfileEvent>, gpu.TError>
fn gpu.profile.@export_chrome_trace(%events: borrow Array<gpu.TProfileEvent>, %path: borrow Str) -> TResult<unit, gpu.TError>

; Manual event markers
fn gpu.profile.@mark_begin(%session: mutborrow gpu.TProfileSession, %name: borrow Str) -> TResult<unit, gpu.TError>
fn gpu.profile.@mark_end(%session: mutborrow gpu.TProfileSession, %name: borrow Str) -> TResult<unit, gpu.TError>

; Automatic profiling (wraps gpu.launch with timing)
fn gpu.profile.@launch_profiled(
    %session: mutborrow gpu.TProfileSession,
    %dev: borrow gpu.TDevice,
    %kernel: Sid,
    %grid: [u32; 3],
    %block: [u32; 3],
    %args: ...
) -> TResult<unit, gpu.TError>

; Memory statistics
fn gpu.profile.@memory_stats(%dev: borrow gpu.TDevice) -> TResult<gpu.TMemoryStats, gpu.TError>

heap struct gpu.TMemoryStats {
    field allocated_bytes: u64
    field peak_allocated_bytes: u64
    field num_allocations: u64
    field num_frees: u64
    field device_total_bytes: u64
    field device_available_bytes: u64
}
```

### 11.4 Vendor Counter Integration

When available, profiling sessions collect vendor-specific hardware counters:

| Backend | Counter API | Available Counters |
|---------|------------|-------------------|
| CUDA | CUPTI | SM occupancy, memory throughput, L1/L2 cache hit rate, warp efficiency |
| HIP | ROCm perf counters | CU occupancy, LDS utilization, VALU/SALU utilization |
| Metal | Metal GPU counters | Shader ALU utilization, memory bandwidth, occupancy |
| Vulkan | `VK_KHR_performance_query` | Vendor-dependent |
| WebGPU | None | Timing only |

**Counter query API:**
```mp
fn gpu.profile.@available_counters(%dev: borrow gpu.TDevice) -> Array<Str>
fn gpu.profile.@enable_counters(%session: mutborrow gpu.TProfileSession, %counters: borrow Array<Str>) -> TResult<unit, gpu.TError>
fn gpu.profile.@read_counters(%session: borrow gpu.TProfileSession) -> TResult<Map<Str, f64>, gpu.TError>
```

### 11.5 Chrome Trace Export

The `gpu.profile.@export_chrome_trace` function writes a JSON file compatible with `chrome://tracing` and Perfetto:

```json
{
  "traceEvents": [
    {"name": "kernel_add", "cat": "gpu.launch", "ph": "X", "ts": 1000, "dur": 500, "pid": 0, "tid": 0},
    {"name": "buffer_write", "cat": "gpu.transfer", "ph": "X", "ts": 200, "dur": 100, "pid": 0, "tid": 0}
  ]
}
```

---

## 12. Manifest Configuration

### 12.1 Base GPU Configuration

```toml
[gpu]
backend = "msl"           # Default backend: "spv", "msl", "ptx", "hip", "wgsl"
fallback = "none"          # "none" (default) or "cpu"
device_index = -1          # -1 = auto-select, 0+ = specific device
mock_in_tests = false      # Enable mock mode for `magpie test`
```

### 12.2 Profile Sections

```toml
[gpu.profiles.dev]
fallback = "cpu"           # CPU fallback in development
mock_in_tests = true       # Mock GPU in tests

[gpu.profiles.release]
fallback = "none"          # No fallback in release
mock_in_tests = false

[gpu.profiles.test]
fallback = "cpu"           # CPU fallback for CI
mock_in_tests = false      # But run actual (simulated) kernels
```

### 12.3 Tool Paths

```toml
[gpu.tools]
llc = "/usr/local/opt/llvm/bin/llc"    # Override llc path for PTX/HSACO
lld = "/usr/local/opt/llvm/bin/ld.lld" # Override lld path for HSACO
```

### 12.4 MLX Configuration

```toml
[mlx]
enabled = true             # Enable MLX integration
```

---

## 13. Diagnostic Codes

### 13.1 Code Format

GPU diagnostics use backend-prefixed codes:

```
MPG_<BACKEND>_<NUMBER>
```

Where `<BACKEND>` is one of: `CORE`, `SPV`, `MSL`, `PTX`, `HIP`, `WGSL`, `MLX`, `PROF`.

**Migration from existing codes:** The current codebase uses flat `MPG` codes (e.g., `MPG1100`). This spec introduces backend-prefixed codes. The following migration mapping MUST be applied:

| Old Code (current codebase) | New Code (this spec) | Location |
|-----------------------------|---------------------|----------|
| `MPG1100` | `MPG_CORE_1100` | `magpie_gpu/src/lib.rs`, `magpie_driver/src/lib.rs` |
| `MPG1101` | `MPG_CORE_1101` | `magpie_gpu/src/lib.rs` |
| `MPG1102` | `MPG_CORE_1102` | `magpie_gpu/src/lib.rs` |
| `MPG1103` | `MPG_CORE_1103` | `magpie_gpu/src/lib.rs` |
| `MPG1104` | `MPG_CORE_1104` | `magpie_gpu/src/lib.rs` |
| `MPG1105` | `MPG_CORE_1105` | `magpie_gpu/src/lib.rs` |
| `MPG1106` | `MPG_CORE_1106` | `magpie_gpu/src/lib.rs` |
| `MPG1107` | `MPG_CORE_1107` | `magpie_gpu/src/lib.rs` |
| `MPG1200` | `MPG_CORE_1300` | `magpie_driver/src/lib.rs` (backend unavailable) |
| `MPG1201` | `MPG_CORE_1201` | `magpie_gpu/src/lib.rs` (shared unsupported) |

The `MPG_` prefix with backend tag is the canonical format going forward. All new diagnostics MUST use the new format. The `magpie_diag` crate's diagnostic code parser must be updated to accept the `MPG_XXX_NNNN` format alongside the legacy `MPGNNNN` format during the transition period.

### 13.2 Core Diagnostics (`MPG_CORE_*`)

| Code | Severity | Title |
|------|----------|-------|
| `MPG_CORE_1100` | error | Heap allocation in gpu fn |
| `MPG_CORE_1101` | error | ARC operation in gpu fn |
| `MPG_CORE_1102` | error | Dynamic dispatch in gpu fn |
| `MPG_CORE_1103` | error | Recursion in gpu fn |
| `MPG_CORE_1104` | error | Str type in gpu fn |
| `MPG_CORE_1105` | error | Array type in gpu fn |
| `MPG_CORE_1106` | error | Map type in gpu fn |
| `MPG_CORE_1107` | error | TCallable type in gpu fn |
| `MPG_CORE_1200` | error | Unsupported capability for target |
| `MPG_CORE_1201` | error | Non-portable feature without requires() |
| `MPG_CORE_1202` | warning | unsafe gpu fn is non-portable |
| `MPG_CORE_1300` | error | Backend not available at compile time |
| `MPG_CORE_1301` | error | llc invocation failed |
| `MPG_CORE_1302` | error | Invalid workgroup size |

### 13.3 SPIR-V Diagnostics (`MPG_SPV_*`)

| Code | Severity | Title |
|------|----------|-------|
| `MPG_SPV_2001` | error | Shared memory exceeds device limit |
| `MPG_SPV_2002` | error | SPIR-V validation failed |
| `MPG_SPV_2003` | warning | Suboptimal buffer layout |

### 13.4 Metal Diagnostics (`MPG_MSL_*`)

| Code | Severity | Title |
|------|----------|-------|
| `MPG_MSL_1001` | error | Irreducible control flow (cannot structure for MSL) |
| `MPG_MSL_2001` | error | Shared memory exceeds 32KB limit |
| `MPG_MSL_2002` | error | MSL compilation failed |
| `MPG_MSL_2003` | error | Metal-specific op without requires() annotation |

### 13.5 CUDA/PTX Diagnostics (`MPG_PTX_*`)

| Code | Severity | Title |
|------|----------|-------|
| `MPG_PTX_2001` | error | Shared memory exceeds limit |
| `MPG_PTX_2002` | error | PTX compilation via llc failed |
| `MPG_PTX_2003` | warning | Compute capability too low for feature |

### 13.6 HIP Diagnostics (`MPG_HIP_*`)

| Code | Severity | Title |
|------|----------|-------|
| `MPG_HIP_2001` | error | Shared memory exceeds LDS limit |
| `MPG_HIP_2002` | error | HSACO generation via llc failed |

### 13.7 WGSL Diagnostics (`MPG_WGSL_*`)

| Code | Severity | Title |
|------|----------|-------|
| `MPG_WGSL_1001` | error | Value struct in buffer (unsupported in WGSL) |
| `MPG_WGSL_1002` | error | Pointer operation (unsupported in WGSL) |
| `MPG_WGSL_1003` | error | bf16 type (unsupported in WGSL) |
| `MPG_WGSL_1004` | error | requires() capability (unsupported in WGSL) |
| `MPG_WGSL_1005` | error | Too many storage buffers (max 8) |
| `MPG_WGSL_1006` | error | Workgroup size exceeds 256 per dimension |

### 13.8 MLX Diagnostics (`MPG_MLX_*`)

| Code | Severity | Title |
|------|----------|-------|
| `MPG_MLX_1001` | error | MLX not available (not macOS / no Apple Silicon) |
| `MPG_MLX_1002` | error | Unsupported element type for MLX array |
| `MPG_MLX_1003` | error | Shape mismatch in MLX operation |
| `MPG_MLX_1004` | warning | MLX operation on CPU (no GPU acceleration) |

### 13.9 Profiling Diagnostics (`MPG_PROF_*`)

| Code | Severity | Title |
|------|----------|-------|
| `MPG_PROF_1001` | warning | Vendor counters not available on this backend |
| `MPG_PROF_1002` | error | Profiling session already active |
| `MPG_PROF_1003` | warning | Counter not supported on this device |

---

## 14. Tool Discovery

### 14.1 llc Discovery

For PTX and HSACO backends, the compiler needs `llc` (and `ld.lld` for HSACO). Discovery order:

1. `[gpu.tools] llc = "..."` in `Magpie.toml` (explicit override)
2. `MAGPIE_LLC_PATH` environment variable
3. Same directory as the `clang` used for host linking (found via `which clang`)
4. `PATH` search for `llc`
5. Platform-specific defaults:
   - macOS: `/usr/local/opt/llvm/bin/llc`, `/opt/homebrew/opt/llvm/bin/llc`
   - Linux: `/usr/lib/llvm-*/bin/llc`
   - ROCm: `/opt/rocm/llvm/bin/llc` (for HIP target)

If `llc` is not found, emit `MPG_CORE_1301` error with installation instructions.

### 14.2 Required LLVM Targets

| Backend | Required `llc` Target |
|---------|----------------------|
| PTX | `nvptx64` (`llc --version` must list NVPTX) |
| HIP | `amdgcn` (`llc --version` must list AMDGPU) |

The compiler MUST verify that `llc` supports the required target before attempting codegen. Emit `MPG_CORE_1301` if the target is missing.

---

## 15. Testing Strategy

### 15.1 Three-Tier Testing

| Tier | Mode | What's Tested | When |
|------|------|---------------|------|
| **Tier 1: Mock** | `gpu.launch` returns `Ok(unit)` immediately | Host-side GPU orchestration logic, error handling, buffer management | Unit tests, CI |
| **Tier 2: CPU Fallback** | Kernels execute on CPU, simulating threads | Kernel correctness, numerical accuracy | Integration tests, CI |
| **Tier 3: Real GPU** | Full GPU dispatch via vendor API | Performance, real hardware behavior | Manual, GPU-enabled CI |

### 15.2 Configuration

```toml
# Magpie.toml for CI
[gpu.profiles.test]
fallback = "cpu"
mock_in_tests = false   # Tier 2: run kernels on CPU

# Magpie.toml for unit tests
[gpu.profiles.test]
mock_in_tests = true    # Tier 1: mock launches
```

**Programmatic control:**
```mp
; In test functions
fn @test_gpu_host_logic() -> unit {
bb0:
    ; Mock mode: gpu.launch returns Ok immediately
    ; Test that host logic handles results correctly
    %dev: gpu.TDevice = ... ; mock device
    %result: TResult<unit, gpu.TError> = gpu.launch { ... }
    ; assert result is Ok
    ret
}
```

### 15.3 Backend-Specific Tests

Each backend crate MUST include:
1. Unit tests for codegen output verification (check emitted SPIR-V/MSL/PTX/LLVM IR/WGSL text)
2. Integration test that compiles a vector-add kernel and verifies the blob is valid
3. Round-trip test: emit → load → validate (where tooling permits)

---

## 16. Implementation Priority

### 16.1 Phase Order

```
Phase 0: Shared Infrastructure (magpie_gpu refactor)
  ├── Extract BackendEmitter trait
  ├── Move SPIR-V code to magpie_gpu_spirv
  ├── Add gpu.TError + gpu.ErrorKind types
  ├── Add bf16 primitive type
  ├── Update kernel registry to multi-blob format
  ├── Implement unified device discovery API
  ├── Add workgroup() annotation to parser
  └── Add requires() annotation to parser

Phase 1: Metal / MSL (highest priority — testable on dev machine)
  ├── magpie_gpu_msl crate (native MSL emission)
  ├── Metal runtime backend (dlopen Metal.framework)
  ├── Metal-specific ops (gpu.metal.simd_*)
  └── Tests on Apple Silicon

Phase 2: MLX Integration
  ├── magpie_mlx crate
  ├── mlx.array, mlx.linalg operations
  ├── mlx.nn layers + wrappers
  ├── mlx.optim optimizers
  ├── mlx.grad autograd (host op via TCallable)
  └── End-to-end neural network training test

Phase 3: CUDA / PTX
  ├── magpie_gpu_ptx crate (LLVM IR → llc → PTX)
  ├── CUDA Driver API runtime backend (dlopen libcuda)
  ├── llc discovery + validation
  └── Tests on NVIDIA GPU

Phase 4: HIP / ROCm
  ├── magpie_gpu_hip crate (LLVM IR → llc → HSACO)
  ├── HIP Driver API runtime backend (dlopen libamdhip64)
  └── Tests on AMD GPU

Phase 5: WebGPU / WGSL (second-class)
  ├── magpie_gpu_wgsl crate (native WGSL emission)
  ├── WebGPU runtime backend (dlopen wgpu-native)
  ├── WGSL restriction enforcement
  └── Tests via wgpu-native

Phase 6: Profiling System
  ├── Event-based tracing infrastructure
  ├── Chrome trace export
  ├── Vendor counter integration (CUPTI, ROCm, Metal)
  └── Profiling API implementation

Phase 7: Polish
  ├── CPU fallback upgrade (three-tier testing)
  ├── Memory hint optimization per backend
  ├── Diagnostic message quality
  └── Documentation and examples
```

### 16.2 Estimated Crate Sizes

| Crate | Estimated Lines | Complexity |
|-------|----------------|------------|
| `magpie_gpu` (core, refactored) | ~800 | Medium |
| `magpie_gpu_spirv` (extracted) | ~1500 | Already exists |
| `magpie_gpu_msl` | ~2000 | High (control flow structuring) |
| `magpie_gpu_ptx` | ~1000 | Medium (LLVM IR + subprocess) |
| `magpie_gpu_hip` | ~800 | Medium (shares patterns with PTX) |
| `magpie_gpu_wgsl` | ~1200 | Medium (WGSL restrictions) |
| `magpie_mlx` | ~3000 | High (full ML stack FFI) |
| Runtime additions | ~4000 | High (5 backend dispatch implementations) |

---

## 17. Code Examples

### 17.1 Vector Add — Vulkan/SPIR-V

```mp
module gpu_example_spv
exports { @main }
imports { }
digest 0000000000000000000000000000000000000000000000000000000000000000

gpu fn @kernel_add(
    %in: gpu.TBuffer<f32>,
    %out: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(spv) workgroup(256, 1, 1) {
bb0:
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
    cbr %in_bounds bb1 bb2
bb1:
    %x: f32 = gpu.buffer_load<f32> { buf=%in, idx=%gid }
    %y: f32 = f.add { a=%x, b=const.f32 1.0 }
    gpu.buffer_store<f32> { buf=%out, idx=%gid, v=%y }
    br bb2
bb2:
    ret
}

fn @main() -> i64 {
bb0:
    %dev_r: TResult<gpu.TDevice, gpu.TError> = gpu.host.@device_default {}
    ; ... check result, create buffers, launch kernel ...
    %n: u32 = const.u32 1024
    %launch_r: TResult<unit, gpu.TError> = gpu.launch {
        device=%dev,
        kernel=@kernel_add,
        grid=[const.u32 4, const.u32 1, const.u32 1],
        block=[const.u32 256, const.u32 1, const.u32 1],
        args=[%buf_in, %buf_out, %n]
    }
    ret const.i64 0
}
```

### 17.2 Vector Add — Metal/MSL

```mp
module gpu_example_msl
exports { @main }
imports { }
digest 0000000000000000000000000000000000000000000000000000000000000000

gpu fn @kernel_add(
    %in: gpu.TBuffer<f32>,
    %out: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(msl) workgroup(256, 1, 1) {
bb0:
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
    cbr %in_bounds bb1 bb2
bb1:
    %x: f32 = gpu.buffer_load<f32> { buf=%in, idx=%gid }
    %y: f32 = f.add { a=%x, b=const.f32 1.0 }
    gpu.buffer_store<f32> { buf=%out, idx=%gid, v=%y }
    br bb2
bb2:
    ret
}

fn @main() -> i64 {
bb0:
    ; On macOS, device_default() selects Metal automatically
    %dev_r: TResult<gpu.TDevice, gpu.TError> = gpu.host.@device_default {}
    ; ... identical host code to SPIR-V example ...
    ret const.i64 0
}
```

### 17.3 Vector Add — CUDA/PTX

```mp
module gpu_example_ptx
exports { @main }
imports { }
digest 0000000000000000000000000000000000000000000000000000000000000000

gpu fn @kernel_add(
    %in: gpu.TBuffer<f32>,
    %out: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(ptx) workgroup(256, 1, 1) {
bb0:
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
    cbr %in_bounds bb1 bb2
bb1:
    %x: f32 = gpu.buffer_load<f32> { buf=%in, idx=%gid }
    %y: f32 = f.add { a=%x, b=const.f32 1.0 }
    gpu.buffer_store<f32> { buf=%out, idx=%gid, v=%y }
    br bb2
bb2:
    ret
}

fn @main() -> i64 {
bb0:
    ; On Linux with NVIDIA GPU, device_default() selects CUDA
    %dev_r: TResult<gpu.TDevice, gpu.TError> = gpu.host.@device_default {}
    ; ... identical host code ...
    ret const.i64 0
}
```

### 17.4 Vector Add — HIP/ROCm

```mp
module gpu_example_hip
exports { @main }
imports { }
digest 0000000000000000000000000000000000000000000000000000000000000000

gpu fn @kernel_add(
    %in: gpu.TBuffer<f32>,
    %out: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(hip) workgroup(256, 1, 1) {
bb0:
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
    cbr %in_bounds bb1 bb2
bb1:
    %x: f32 = gpu.buffer_load<f32> { buf=%in, idx=%gid }
    %y: f32 = f.add { a=%x, b=const.f32 1.0 }
    gpu.buffer_store<f32> { buf=%out, idx=%gid, v=%y }
    br bb2
bb2:
    ret
}

fn @main() -> i64 {
bb0:
    ; On Linux with AMD GPU, device_default() selects HIP
    %dev_r: TResult<gpu.TDevice, gpu.TError> = gpu.host.@device_default {}
    ; ... identical host code ...
    ret const.i64 0
}
```

### 17.5 Vector Add — WebGPU/WGSL (Second-Class)

```mp
module gpu_example_wgsl
exports { @main }
imports { }
digest 0000000000000000000000000000000000000000000000000000000000000000

; NOTE: WGSL is second-class. Only primitive buffer types supported.
gpu fn @kernel_add(
    %in: gpu.TBuffer<f32>,
    %out: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(wgsl) workgroup(64, 1, 1) {
bb0:
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
    cbr %in_bounds bb1 bb2
bb1:
    %x: f32 = gpu.buffer_load<f32> { buf=%in, idx=%gid }
    %y: f32 = f.add { a=%x, b=const.f32 1.0 }
    gpu.buffer_store<f32> { buf=%out, idx=%gid, v=%y }
    br bb2
bb2:
    ret
}

fn @main() -> i64 {
bb0:
    %dev_r: TResult<gpu.TDevice, gpu.TError> = gpu.host.@device_default {}
    ; ... identical host code, but workgroup max 256 per dim ...
    ret const.i64 0
}
```

### 17.6 CUDA Unsafe Kernel with Device Malloc

```mp
module gpu_example_unsafe_ptx
exports { @main }
imports { }
digest 0000000000000000000000000000000000000000000000000000000000000000

; Non-portable kernel: uses CUDA device-side malloc
unsafe gpu fn @dynamic_kernel(
    %out: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(ptx) workgroup(256, 1, 1) requires(device_malloc) {
bb0:
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
    cbr %in_bounds bb1 bb2
bb1:
    ; Use device malloc (CUDA-only)
    ; ... device-side dynamic allocation ...
    br bb2
bb2:
    ret
}
```

### 17.7 Metal Unsafe Kernel with SIMD Shuffle

```mp
module gpu_example_unsafe_msl
exports { @main }
imports { }
digest 0000000000000000000000000000000000000000000000000000000000000000

unsafe gpu fn @simd_reduce_kernel(
    %input: gpu.TBuffer<f32>,
    %output: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(msl) workgroup(256, 1, 1) requires(metal.simd_reduce) {
bb0:
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
    cbr %in_bounds bb1 bb2
bb1:
    %x: f32 = gpu.buffer_load<f32> { buf=%input, idx=%gid }
    ; Metal SIMD group reduction (non-portable)
    %sum: f32 = gpu.metal.simd_sum { v=%x }
    ; First thread in simdgroup writes result
    gpu.buffer_store<f32> { buf=%output, idx=%gid, v=%sum }
    br bb2
bb2:
    ret
}
```

### 17.8 Shared Memory Reduction (Portable)

```mp
module gpu_example_shared
exports { @main }
imports { }
digest 0000000000000000000000000000000000000000000000000000000000000000

; Portable: works on all backends (shared memory is first-class)
gpu fn @reduce_sum(
    %input: gpu.TBuffer<f32>,
    %output: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(spv) workgroup(256, 1, 1) {
bb0:
    %tid: u32 = gpu.thread_id { dim=const.u32 0 }
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %shared: ptr = gpu.shared<256, f32>

    ; Load input into shared memory
    %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
    cbr %in_bounds bb1 bb2
bb1:
    %val: f32 = gpu.buffer_load<f32> { buf=%input, idx=%gid }
    gpu.buffer_store<f32> { buf=%shared, idx=%tid, v=%val }
    br bb3
bb2:
    gpu.buffer_store<f32> { buf=%shared, idx=%tid, v=const.f32 0.0 }
    br bb3
bb3:
    gpu.barrier

    ; Tree reduction in shared memory
    ; (simplified — full reduction would loop with stride halving)
    %half: u32 = const.u32 128
    %should_reduce: bool = icmp.ult { lhs=%tid, rhs=%half }
    cbr %should_reduce bb4 bb5
bb4:
    %my_val: f32 = gpu.buffer_load<f32> { buf=%shared, idx=%tid }
    %partner_idx: u32 = i.add { a=%tid, b=%half }
    %partner_val: f32 = gpu.buffer_load<f32> { buf=%shared, idx=%partner_idx }
    %sum: f32 = f.add { a=%my_val, b=%partner_val }
    gpu.buffer_store<f32> { buf=%shared, idx=%tid, v=%sum }
    br bb5
bb5:
    gpu.barrier

    ; Thread 0 writes result
    %is_zero: bool = icmp.eq { lhs=%tid, rhs=const.u32 0 }
    cbr %is_zero bb6 bb7
bb6:
    %final: f32 = gpu.buffer_load<f32> { buf=%shared, idx=const.u32 0 }
    %wgid: u32 = gpu.workgroup_id { dim=const.u32 0 }
    gpu.buffer_store<f32> { buf=%output, idx=%wgid, v=%final }
    br bb7
bb7:
    ret
}
```

### 17.9 MLX Neural Network Training Loop

```mp
module mlx_training_example
exports { @main }
imports { mlx.array, mlx.nn, mlx.optim, mlx.grad, mlx.random }
digest 0000000000000000000000000000000000000000000000000000000000000000

; Define a simple 2-layer MLP for MNIST
; Input: 784 -> Hidden: 128 -> Output: 10

fn @create_model() -> TResult<Array<mlx.nn.TLinear>, gpu.TError> {
bb0:
    ; Create layers
    %l1_r: TResult<mlx.nn.TLinear, gpu.TError> = mlx.nn.@linear { in_features=const.u64 784, out_features=const.u64 128 }
    %l1: mlx.nn.TLinear = try %l1_r bb_err
    %l2_r: TResult<mlx.nn.TLinear, gpu.TError> = mlx.nn.@linear { in_features=const.u64 128, out_features=const.u64 10 }
    %l2: mlx.nn.TLinear = try %l2_r bb_err

    ; Pack into array
    %layers: Array<mlx.nn.TLinear> = arr.new { v0=%l1, v1=%l2 }
    %ok: TResult<Array<mlx.nn.TLinear>, gpu.TError> = enum.new<TResult, 0> { v=%layers }
    ret %ok
bb_err:
    %err: gpu.TError = phi [bb0: %l1_r.err] [bb0: %l2_r.err]
    %fail: TResult<Array<mlx.nn.TLinear>, gpu.TError> = enum.new<TResult, 1> { v=%err }
    ret %fail
}

fn @forward(
    %layers: borrow Array<mlx.nn.TLinear>,
    %x: borrow mlx.TArray<f32>
) -> TResult<mlx.TArray<f32>, gpu.TError> {
bb0:
    ; Layer 1 + ReLU
    %l1: borrow mlx.nn.TLinear = arr.get { arr=%layers, idx=const.u64 0 }
    %h1_r: TResult<mlx.TArray<f32>, gpu.TError> = mlx.nn.@linear_forward { layer=%l1, input=%x }
    %h1: mlx.TArray<f32> = try %h1_r bb_err
    %a1_r: TResult<mlx.TArray<f32>, gpu.TError> = mlx.nn.@relu<f32> { x=%h1 }
    %a1: mlx.TArray<f32> = try %a1_r bb_err

    ; Layer 2 + Softmax
    %l2: borrow mlx.nn.TLinear = arr.get { arr=%layers, idx=const.u64 1 }
    %h2_r: TResult<mlx.TArray<f32>, gpu.TError> = mlx.nn.@linear_forward { layer=%l2, input=%a1 }
    %h2: mlx.TArray<f32> = try %h2_r bb_err
    %out_r: TResult<mlx.TArray<f32>, gpu.TError> = mlx.nn.@softmax<f32> { x=%h2, axis=const.i32 1 }
    %out: mlx.TArray<f32> = try %out_r bb_err

    %ok: TResult<mlx.TArray<f32>, gpu.TError> = enum.new<TResult, 0> { v=%out }
    ret %ok
bb_err:
    ; ... error propagation ...
    ret %fail
}

fn @main() -> i64 {
bb0:
    ; Create model and optimizer
    %model_r: TResult<Array<mlx.nn.TLinear>, gpu.TError> = call @create_model {}
    ; ... unwrap model ...

    %opt_r: TResult<mlx.optim.TAdam, gpu.TError> = mlx.optim.@adam { lr=const.f32 0.001 }
    ; ... unwrap optimizer ...

    ; Training loop (10 epochs)
    ; For each batch:
    ;   1. Forward pass
    ;   2. Compute loss (cross-entropy)
    ;   3. Backward pass via mlx.grad
    ;   4. Optimizer step
    ;   5. mlx.array.@eval to materialize

    ; ... training loop implementation ...

    ret const.i64 0
}
```

### 17.10 Profiling Session

```mp
module gpu_profiling_example
exports { @main }
imports { }
digest 0000000000000000000000000000000000000000000000000000000000000000

gpu fn @kernel_work(
    %data: gpu.TBuffer<f32>,
    %n: u32
) -> unit target(spv) workgroup(256, 1, 1) {
bb0:
    %gid: u32 = gpu.global_id { dim=const.u32 0 }
    %in_bounds: bool = icmp.ult { lhs=%gid, rhs=%n }
    cbr %in_bounds bb1 bb2
bb1:
    %x: f32 = gpu.buffer_load<f32> { buf=%data, idx=%gid }
    %y: f32 = f.mul { a=%x, b=%x }
    gpu.buffer_store<f32> { buf=%data, idx=%gid, v=%y }
    br bb2
bb2:
    ret
}

fn @main() -> i64 {
bb0:
    %dev_r: TResult<gpu.TDevice, gpu.TError> = gpu.host.@device_default {}
    %dev: gpu.TDevice = try %dev_r bb_fail

    ; Start profiling session
    %session_r: TResult<gpu.TProfileSession, gpu.TError> = gpu.profile.@begin { dev=%dev }
    %session: gpu.TProfileSession = try %session_r bb_fail

    ; Enable vendor counters if available
    %counters: Array<Str> = gpu.profile.@available_counters { dev=%dev }
    %enable_r: TResult<unit, gpu.TError> = gpu.profile.@enable_counters { session=%session, counters=%counters }

    ; Mark a region
    gpu.profile.@mark_begin { session=%session, name="computation" }

    ; Run kernel multiple times
    %n: u32 = const.u32 1048576
    ; ... create buffer, launch kernel 10 times ...

    gpu.profile.@mark_end { session=%session, name="computation" }

    ; Read hardware counters (MUST be called before @end, which consumes the session)
    %hw_counters_r: TResult<Map<Str, f64>, gpu.TError> = gpu.profile.@read_counters { session=%session }

    ; End session (consumes %session — no further use after this point)
    %events_r: TResult<Array<gpu.TProfileEvent>, gpu.TError> = gpu.profile.@end { session=%session }
    %events: Array<gpu.TProfileEvent> = try %events_r bb_fail

    ; Export Chrome trace
    %export_r: TResult<unit, gpu.TError> = gpu.profile.@export_chrome_trace {
        events=%events,
        path="profile_trace.json"
    }

    ret const.i64 0
bb_fail:
    ret const.i64 1
}
```

---

## Appendix A: Grammar Extensions

### A.1 Updated GpuFnDecl

```ebnf
GpuFnDecl       = "gpu" "fn" FnName "(" Params? ")" "->" Type
                  "target" "(" BackendId ")"
                  [ "workgroup" "(" Int "," Int "," Int ")" ]
                  Block

UnsafeGpuFnDecl = "unsafe" "gpu" "fn" FnName "(" Params? ")" "->" Type
                  "target" "(" BackendId ")"
                  [ "workgroup" "(" Int "," Int "," Int ")" ]
                  "requires" "(" CapList ")"
                  Block

BackendId       = "spv" | "msl" | "ptx" | "hip" | "wgsl"

CapList         = Capability { "," Capability }
Capability      = Ident [ "." Ident ]    ; e.g., "device_malloc" or "metal.simd_shuffle"

; Decl production — add UnsafeGpuFnDecl:
Decl = FnDecl | AsyncFnDecl | UnsafeFnDecl | GpuFnDecl | UnsafeGpuFnDecl
     | TypeDecl | ExternModuleDecl | GlobalDecl | ImplDecl | SigDecl
```

Note: `FnName`, `Params`, and `Type` are existing non-terminals from SPEC.md §7.2.

### A.2 New GPU Ops

```ebnf
; Metal-specific ops (unsafe gpu fn only)
MetalOp       = "gpu.metal.simd_shuffle" "{" ... "}"
              | "gpu.metal.simd_sum" "{" ... "}"
              | "gpu.metal.simd_prefix_sum" "{" ... "}"
```

### A.3 Memory Hint in Buffer Creation

```ebnf
MemoryHint    = "Auto" | "DeviceLocal" | "Unified" | "Staging" | "Cached"
```

---

## Appendix B: Runtime ABI Additions

### B.1 New C Functions

```c
// Error construction
MpRtHeader* mp_rt_gpu_error_new(int32_t kind, const char* backend, const char* message, int32_t code);
int32_t     mp_rt_gpu_error_kind(MpRtHeader* err);
MpRtHeader* mp_rt_gpu_error_message(MpRtHeader* err);
MpRtHeader* mp_rt_gpu_error_backend(MpRtHeader* err);
int32_t     mp_rt_gpu_error_code(MpRtHeader* err);

// Unified device discovery
uint32_t    mp_rt_gpu_device_count_unified(void);
int32_t     mp_rt_gpu_device_default_unified(MpRtHeader** out_dev, MpRtHeader** out_err);
int32_t     mp_rt_gpu_device_backends(MpRtHeader* dev, MpRtHeader** out_arr, MpRtHeader** out_err);

// Device capabilities
uint32_t    mp_rt_gpu_device_max_workgroup_size(MpRtHeader* dev);
uint32_t    mp_rt_gpu_device_max_shared_bytes(MpRtHeader* dev);
uint32_t    mp_rt_gpu_device_max_buffers(MpRtHeader* dev);
uint32_t    mp_rt_gpu_device_warp_size(MpRtHeader* dev);
uint64_t    mp_rt_gpu_device_memory_total(MpRtHeader* dev);
uint64_t    mp_rt_gpu_device_memory_available(MpRtHeader* dev);

// Extended buffer with memory hint
int32_t     mp_rt_gpu_buffer_new_hinted(MpRtHeader* dev, uint32_t elem_type_id, uint32_t elem_size,
                uint64_t len, uint32_t usage_flags, uint32_t hint, MpRtHeader** out_buf, MpRtHeader** out_err);

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

// MLX (separate from GPU dispatch)
int32_t     mp_rt_mlx_init(void);
int32_t     mp_rt_mlx_array_zeros(uint32_t elem_type_id, const uint64_t* shape, uint32_t ndim, MpRtHeader** out, MpRtHeader** out_err);
int32_t     mp_rt_mlx_array_matmul(MpRtHeader* a, MpRtHeader* b, MpRtHeader** out, MpRtHeader** out_err);
// ... (full MLX ABI follows same pattern)
```

### B.2 New Fixed TypeIds

See Section 3.5 for the full TypeId unification table. Summary of new allocations:

| TypeId | Type | Notes |
|--------|------|-------|
| 16 | `bf16` | New primitive type |
| 17-19 | *reserved* | Future numeric types |
| 30-32 | `gpu.TDevice`, `gpu.TBuffer<?>`, `gpu.TFence` | Existing (runtime must adopt these IDs) |
| 33 | `gpu.TError` | NEW — replaces Str errors |
| 34 | `gpu.ErrorKind` | NEW — error category enum |
| 35 | `gpu.TProfileSession` | NEW |
| 36 | `gpu.TProfileEvent` | NEW |
| 37 | `gpu.TMemoryStats` | NEW |
| 38 | `mlx.TArray<?>` (base) | NEW |
| 39 | `mlx.nn.TLayerHandle` | NEW |
| 40 | `mlx.optim.TOptimizerHandle` | NEW |
| 41-49 | *reserved* | Future GPU/MLX types |
| 50 | `gpu.TKernel` (runtime-internal) | Moved from runtime's former 9004 |

**IMPORTANT:** The runtime's private TypeIds (9001-9004) are deprecated. See Section 3.5 for the migration path.

---

## Appendix C: Backend Capability Matrix

| Feature | SPV | MSL | PTX | HIP | WGSL |
|---------|-----|-----|-----|-----|------|
| Buffer load/store | yes | yes | yes | yes | yes |
| Shared memory | yes | yes | yes | yes | yes |
| Barrier | yes | yes | yes | yes | yes |
| Atomics (int) | yes | yes | yes | yes | yes |
| Atomics (f32) | ext | yes | yes | yes | no |
| Atomics (f64) | no | no | yes | yes | no |
| bf16 | ext | yes | yes | yes | no |
| f16 arithmetic | yes | yes | yes | yes | yes |
| Subgroups | yes | yes (SIMD) | yes (warp) | yes (wave) | no |
| Device malloc | no | no | yes | yes | no |
| Recursion | no | no | yes | yes | no |
| Dynamic parallelism | no | no | yes | yes | no |
| Cooperative groups | no | no | yes | yes | no |
| Value structs in buffers | yes | yes | yes | yes | no |
| Max workgroup size | 1024 | 1024 | 1024 | 1024 | 256 |
| Max shared memory | 48KB | 32KB | 48-96KB | 64KB | 16KB |

---

*End of GPU Expansion Specification*
