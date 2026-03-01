# Magpie

Magpie is an experimental programming language and compiler toolchain.

It is built for:
- deterministic source and machine output
- explicit ownership/borrowing rules with ARC-managed heap lifetimes
- strong diagnostics for fast fix loops
- CLI workflows that are friendly to automation and LLM agents

## Current status

This repository is a Rust workspace for **Magpie v0.1**. It includes:
- a `magpie` CLI
- lexer/parser/semantic analysis/type checking
- ownership checking
- MPIR lowering + verification
- monomorphization (BLAKE3-keyed specialization)
- ARC insertion/optimization passes
- LLVM-text, WASM, and multi-backend GPU codegen paths (SPIR-V, MSL, PTX, HIP, WGSL)
- MLX host API integration (Apple Silicon ML acceleration)
- GPU profiling system (Chrome trace export, allocation tracking)
- runtime with real GPU dispatch via `dlopen` (Metal > CUDA > HIP > Vulkan > WebGPU)
- `bf16` (bfloat16) primitive type
- runtime, package, memory-index, graph, web, and MCP tooling

## Repository layout

High-level crates (29 total):

- `crates/magpie_cli` — command-line entrypoint (`magpie`)
- `crates/magpie_driver` — compiler orchestration pipeline
- `crates/magpie_lex`, `magpie_parse`, `magpie_ast` — frontend
- `crates/magpie_sema`, `magpie_hir`, `magpie_types` — semantic + type layers
- `crates/magpie_own` — ownership/borrow checker
- `crates/magpie_mpir`, `magpie_mono`, `magpie_arc` — mid-level IR and lowering passes
- `crates/magpie_codegen_llvm`, `magpie_codegen_wasm` — CPU backend codegen
- `crates/magpie_gpu` — GPU codegen core (BackendEmitter trait, CFG structurizer, kernel registry)
- `crates/magpie_gpu_spirv` — SPIR-V backend (Vulkan)
- `crates/magpie_gpu_msl` — Metal Shading Language backend (Apple)
- `crates/magpie_gpu_ptx` — PTX/NVVM backend (NVIDIA CUDA)
- `crates/magpie_gpu_hip` — HIP/HSACO backend (AMD ROCm)
- `crates/magpie_gpu_wgsl` — WGSL backend (WebGPU)
- `crates/magpie_mlx` — MLX host API integration (Apple Silicon ML)
- `crates/magpie_rt` — runtime library (ARC, GPU dispatch, profiling)
- `crates/magpie_diag` — diagnostics + envelopes
- `crates/magpie_csnf` — canonical formatter/digest handling
- `crates/magpie_pkg`, `magpie_memory`, `magpie_ctx`, `magpie_web` — tooling and platform subsystems

Other important paths:
- `tests/fixtures/` — language fixture programs, including `feature_harness.mp` and `tresult_parse_json.mp`
- `std/` — standard library modules used by Magpie projects
- `DOCUMENTATION.md` — full technical documentation
- `DOCUMENTATION_QUICKSTART.md` — fast command reference
- `SKILL.md` — detailed coding/diagnostic guide for agent workflows

## Prerequisites

Required:
- Rust **1.80+**
- Cargo

Optional but recommended (for execution/link workflows):
- `lli` (run LLVM IR via `magpie run` in dev workflows)
- `llc` + `clang` + system linker (native executable emission/linking)

Optional (for GPU backend compilation):
- `llc` with NVPTX target (PTX backend) or AMDGPU target (HIP backend)
- `ld.lld` (HIP HSACO linking)
- Metal.framework (MSL backend, macOS only — auto-detected via `dlopen`)
- Vulkan SDK (SPIR-V backend)
- MLX framework (Apple Silicon ML — auto-detected via `dlopen`)

## Build the compiler

From repo root:

```bash
cargo build -p magpie_cli
```

Build the full workspace:

```bash
cargo build --workspace
```

Check CLI help:

```bash
cargo run -p magpie_cli -- --help
```

Optional: install local `magpie` binary:

```bash
cargo install --path crates/magpie_cli --force
magpie --help
```

If you do not install the binary, use:

```bash
cargo run -p magpie_cli -- <GLOBAL_FLAGS> <SUBCOMMAND> ...
```

## Important CLI usage detail

`magpie` uses **global flags**, so put them **before** the subcommand.

Correct:

```bash
magpie --entry src/main.mp --emit mpir,llvm-ir --output json build
```

Not correct:

```bash
magpie build --entry src/main.mp
```

## Quick start (new project)

```bash
magpie new demo
cd demo
magpie --output json --emit mpir,llvm-ir build
```

This generates artifacts like:
- `target/<triple>/<profile>/main.mpir`
- `target/<triple>/<profile>/main.ll`
- `.magpie/memory/main.mms_index.json`

## What Magpie source looks like

Magpie source files use a strict module header and explicit basic blocks:

```mp
module demo.main
exports { @main }
imports { }
digest "0000000000000000"

fn @main() -> i64 {
bb0:
  ret const.i64 0
}
```

This structure is intentionally regular so formatting, parsing, diagnostics, and automated edits stay predictable.

## Common workflows

Format source files:

```bash
magpie fmt --fix-meta
```

Parse only:

```bash
magpie --entry src/main.mp --output json parse
```

Build with debug artifacts:

```bash
magpie --entry src/main.mp --emit mpir,llvm-ir,mpdbg --output json build
```

Explain a diagnostic code:

```bash
magpie explain MPT2014
```

Run tests in this repository:

```bash
cargo test
```

## CLI commands at a glance

Top-level commands in `magpie`:

- `new`
- `build`
- `run`
- `repl`
- `fmt`
- `parse`
- `lint`
- `test`
- `doc`
- `mpir verify`
- `explain`
- `pkg` (`resolve`, `add`, `remove`, `why`)
- `web` (`dev`, `build`, `serve`)
- `mcp serve`
- `memory` (`build`, `query`)
- `ctx pack`
- `ffi import`
- `graph` (`symbols`, `deps`, `ownership`, `cfg`)

## GPU multi-backend support

Magpie supports 5 GPU compute backends via a unified `BackendEmitter` trait:

| Backend | Emit kind | Target | Crate |
|---------|-----------|--------|-------|
| SPIR-V  | `spv`     | Vulkan | `magpie_gpu_spirv` |
| MSL     | `msl`     | Metal (Apple) | `magpie_gpu_msl` |
| PTX     | `ptx`     | CUDA (NVIDIA) | `magpie_gpu_ptx` |
| HIP     | `hip`     | ROCm (AMD) | `magpie_gpu_hip` |
| WGSL    | `wgsl`    | WebGPU | `magpie_gpu_wgsl` |

The runtime probes backends at startup via `dlopen` in priority order: Metal > CUDA > HIP > Vulkan > WebGPU. Falls back to CPU simulation if no GPU is available.

Configure GPU behavior in `Magpie.toml`:

```toml
[gpu]
backend = "auto"       # auto | spirv | msl | ptx | hip | wgsl
fallback = "cpu"       # cpu | error
llc_path = "/usr/local/bin/llc"   # optional
lld_path = "/usr/local/bin/ld.lld" # optional (HIP)
```

### MLX integration (Apple Silicon)

The `magpie_mlx` crate provides full MLX host API integration via `dlopen`/`dlsym` dispatch tables (~40 function pointers). This enables ML workloads (array ops, neural network layers, optimizers, automatic differentiation) on Apple Silicon without requiring MLX to be a build-time dependency.

### Monomorphization

The `magpie_mono` crate implements generic specialization using BLAKE3-keyed instance hashing. Generic functions are duplicated and specialized per concrete type argument set, with deterministic SID (Symbol ID) assignment.

## Parse/JSON migration note

Parse and JSON runtime ABI now has dual APIs:
- Preferred: fallible `*_try_*` symbols (`mp_rt_str_try_parse_*`, `mp_rt_json_try_*`) that return status codes.
- Legacy: `mp_rt_str_parse_*`, `mp_rt_json_encode`, `mp_rt_json_decode` are deprecated compatibility wrappers.

At source level, compatibility value-style ops still exist, and `TResult` parse/json fixtures are included for end-to-end coverage.

## Feature harness program

A broad language feature harness is included at:

- `tests/fixtures/feature_harness.mp`
- `tests/fixtures/tresult_parse_json.mp`

Build it:

```bash
magpie --entry tests/fixtures/feature_harness.mp --emit mpir,llvm-ir --output json build
```

Try execution paths:

- LLVM IR path (requires `lli`):
  ```bash
  magpie --entry tests/fixtures/feature_harness.mp --emit llvm-ir run
  ```
- Native binary path (requires full native toolchain):
  ```bash
  magpie --entry tests/fixtures/feature_harness.mp --emit exe build
  ```

## Output modes

Global `--output` supports:
- `text`
- `json`
- `jsonl`

Use `--output json` for machine-readable automation.

## Where to go next

- Language and semantics: `DOCUMENTATION.md`
- Fast command cheatsheet: `DOCUMENTATION_QUICKSTART.md`
- Deep compiler/diagnostic examples: `SKILL.md`
- GPU expansion specification: `SPEC_GPU_UPGRADE.md`
- GPU interoperability contracts: `GPU_INTEROP_SPEC.md`
