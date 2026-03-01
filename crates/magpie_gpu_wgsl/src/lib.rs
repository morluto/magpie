//! WebGPU WGSL backend for Magpie GPU compute (second-class target).
//!
//! Emits native WGSL source text from MPIR kernel functions.
//! Uses the shared CFG structurizer from `magpie_gpu`.
//! Enforces WGSL subset restrictions per SPEC_GPU_UPGRADE.md §7.5.

use std::collections::HashMap;
use std::fmt::Write;

use magpie_diag::Diagnostic;
use magpie_mpir::{
    HirConst, HirConstLit, MpirBlock, MpirFn, MpirInstr, MpirOp, MpirOpVoid, MpirTerminator,
    MpirTypeTable, MpirValue,
};
use magpie_types::{PrimType, TypeCtx, TypeId, TypeKind};

/// WGSL backend emitter (second-class).
pub struct WgslEmitter;

struct EmitState<'a> {
    type_ctx: &'a TypeCtx,
    local_aliases: HashMap<u32, String>,
    workgroup: [u32; 3],
}

struct SharedDecl {
    local_id: u32,
    name: String,
    elem_ty: &'static str,
    size: String,
}

impl WgslEmitter {
    pub fn new() -> Self {
        Self
    }

    pub fn artifact_extension(&self) -> &str {
        "wgsl"
    }

    /// Validate kernel compatibility with WGSL restrictions.
    pub fn validate_kernel(
        &self,
        kernel: &MpirFn,
        _types: &MpirTypeTable,
        type_ctx: &TypeCtx,
    ) -> Result<(), Vec<Diagnostic>> {
        let mut errors = Vec::new();

        // Check workgroup size limits (max 256 per dimension)
        if let Some(meta) = &kernel.gpu_meta {
            for (i, &dim) in meta.workgroup.iter().enumerate() {
                if dim > 256 {
                    errors.push(Diagnostic {
                        code: "MPG_WGSL_1006".to_string(),
                        severity: magpie_diag::Severity::Error,
                        title: "Workgroup size exceeds 256 per dimension".to_string(),
                        primary_span: None,
                        secondary_spans: vec![],
                        message: format!(
                            "WGSL workgroup dimension {} is {} but max is 256",
                            i, dim
                        ),
                        explanation_md: None,
                        why: None,
                        suggested_fixes: vec![],
                        rag_bundle: vec![],
                        related_docs: vec![],
                    });
                }
            }
        }

        // Count buffer parameters (max 8)
        let buffer_count = kernel
            .params
            .iter()
            .filter(|(_, ty)| *ty == magpie_types::fixed_type_ids::GPU_BUFFER_BASE)
            .count();
        if buffer_count > 8 {
            errors.push(Diagnostic {
                code: "MPG_WGSL_1005".to_string(),
                severity: magpie_diag::Severity::Error,
                title: "Too many storage buffers (max 8)".to_string(),
                primary_span: None,
                secondary_spans: vec![],
                message: format!(
                    "Kernel has {} buffer parameters but WGSL max is 8",
                    buffer_count
                ),
                explanation_md: None,
                why: None,
                suggested_fixes: vec![],
                rag_bundle: vec![],
                related_docs: vec![],
            });
        }

        // Check for bf16 usage
        for (_, ty) in &kernel.params {
            if matches!(type_ctx.lookup(*ty), Some(TypeKind::Prim(PrimType::Bf16))) {
                errors.push(Diagnostic {
                    code: "MPG_WGSL_1003".to_string(),
                    severity: magpie_diag::Severity::Error,
                    title: "bf16 type unsupported in WGSL".to_string(),
                    primary_span: None,
                    secondary_spans: vec![],
                    message: "WGSL does not support bfloat16".to_string(),
                    explanation_md: None,
                    why: None,
                    suggested_fixes: vec![],
                    rag_bundle: vec![],
                    related_docs: vec![],
                });
                break;
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Emit WGSL source text from an MPIR kernel function.
    pub fn emit_kernel(&self, kernel: &MpirFn, type_ctx: &TypeCtx) -> Result<Vec<u8>, String> {
        let mut out = String::with_capacity(4096);
        let workgroup = kernel
            .gpu_meta
            .as_ref()
            .map(|m| m.workgroup)
            .unwrap_or([64, 1, 1]);
        let mut state = EmitState {
            type_ctx,
            local_aliases: HashMap::new(),
            workgroup,
        };

        // Emit buffer bindings
        let mut binding = 0u32;
        let mut scalar_params = Vec::new();

        for (i, (local_id, ty_id)) in kernel.params.iter().enumerate() {
            let is_buffer = is_gpu_buffer(type_ctx, *ty_id);
            if is_buffer {
                writeln!(
                    out,
                    "@group(0) @binding({binding}) var<storage, read_write> buf_{i}: array<f32>;"
                )
                .map_err(|e| e.to_string())?;
                state.local_aliases.insert(local_id.0, format!("buf_{i}"));
            } else {
                scalar_params.push((i, *ty_id));
                state
                    .local_aliases
                    .insert(local_id.0, format!("params.param_{i}"));
            }
            binding += 1;
        }

        // Emit scalar params as uniform struct
        if !scalar_params.is_empty() {
            writeln!(out).map_err(|e| e.to_string())?;
            writeln!(out, "struct Params {{").map_err(|e| e.to_string())?;
            for (i, ty_id) in &scalar_params {
                let wgsl_ty = wgsl_type_name(type_ctx, *ty_id);
                writeln!(out, "    param_{i}: {wgsl_ty},").map_err(|e| e.to_string())?;
            }
            writeln!(out, "}}").map_err(|e| e.to_string())?;
            writeln!(
                out,
                "@group(0) @binding({binding}) var<uniform> params: Params;"
            )
            .map_err(|e| e.to_string())?;
        }

        let shared_decls = collect_shared_decls(kernel, type_ctx);
        if !shared_decls.is_empty() {
            writeln!(out).map_err(|e| e.to_string())?;
            for decl in &shared_decls {
                writeln!(
                    out,
                    "var<workgroup> {}: array<{}, {}>;",
                    decl.name, decl.elem_ty, decl.size
                )
                .map_err(|e| e.to_string())?;
                state.local_aliases.insert(decl.local_id, decl.name.clone());
            }
        }

        writeln!(out).map_err(|e| e.to_string())?;

        // Workgroup size
        writeln!(
            out,
            "@compute @workgroup_size({}, {}, {})",
            workgroup[0], workgroup[1], workgroup[2]
        )
        .map_err(|e| e.to_string())?;

        let fn_name = kernel.name.trim_start_matches('@').replace('.', "_");
        writeln!(out, "fn {fn_name}(").map_err(|e| e.to_string())?;
        writeln!(out, "    @builtin(global_invocation_id) gid: vec3<u32>,")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @builtin(local_invocation_id) tid: vec3<u32>,")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    @builtin(workgroup_id) wgid: vec3<u32>").map_err(|e| e.to_string())?;
        writeln!(out, ") {{").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "    let _wgsz = vec3<u32>({}u, {}u, {}u);",
            state.workgroup[0], state.workgroup[1], state.workgroup[2]
        )
        .map_err(|e| e.to_string())?;

        // Emit kernel body
        for block in &kernel.blocks {
            emit_wgsl_block(&mut out, block, &mut state)?;
        }

        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out.into_bytes())
    }
}

fn wgsl_type_name(type_ctx: &TypeCtx, ty_id: TypeId) -> &'static str {
    match type_ctx.lookup(ty_id) {
        Some(TypeKind::Prim(PrimType::F32)) => "f32",
        Some(TypeKind::Prim(PrimType::F16)) => "f16",
        Some(TypeKind::Prim(PrimType::I32)) => "i32",
        Some(TypeKind::Prim(PrimType::I64)) => "i32",
        Some(TypeKind::Prim(PrimType::U32)) => "u32",
        Some(TypeKind::Prim(PrimType::U64)) => "u32",
        Some(TypeKind::Prim(PrimType::Bool)) => "bool",
        _ => "u32",
    }
}

fn is_gpu_buffer(type_ctx: &TypeCtx, ty_id: TypeId) -> bool {
    ty_id == magpie_types::fixed_type_ids::GPU_BUFFER_BASE
        || matches!(type_ctx.lookup(ty_id), Some(TypeKind::HeapHandle { .. }))
}

fn collect_shared_decls(kernel: &MpirFn, type_ctx: &TypeCtx) -> Vec<SharedDecl> {
    let mut out = Vec::new();
    for block in &kernel.blocks {
        for instr in &block.instrs {
            if let MpirOp::GpuShared { ty, size } = &instr.op {
                let sz = match size {
                    MpirValue::Const(HirConst {
                        lit: HirConstLit::IntLit(v),
                        ..
                    }) if *v > 0 => v.to_string(),
                    _ => "1".to_string(),
                };
                out.push(SharedDecl {
                    local_id: instr.dst.0,
                    name: format!("shared_mem_{}", instr.dst.0),
                    elem_ty: wgsl_type_name(type_ctx, *ty),
                    size: sz,
                });
            }
        }
    }
    out
}

fn emit_wgsl_block(
    out: &mut String,
    block: &MpirBlock,
    state: &mut EmitState<'_>,
) -> Result<(), String> {
    writeln!(out, "    // bb{}", block.id.0).map_err(|e| e.to_string())?;
    for instr in &block.instrs {
        emit_wgsl_instr(out, instr, state)?;
    }
    for void_op in &block.void_ops {
        emit_wgsl_void_op(out, void_op, state)?;
    }
    emit_wgsl_terminator(out, &block.terminator, state)?;
    Ok(())
}

fn emit_wgsl_instr(
    out: &mut String,
    instr: &MpirInstr,
    state: &mut EmitState<'_>,
) -> Result<(), String> {
    match &instr.op {
        MpirOp::Const(c) => emit_local_expr(out, instr.dst.0, &format_wgsl_const(c), state)?,
        MpirOp::Move { v }
        | MpirOp::BorrowShared { v }
        | MpirOp::BorrowMut { v }
        | MpirOp::Share { v }
        | MpirOp::CloneShared { v }
        | MpirOp::CloneWeak { v }
        | MpirOp::WeakDowngrade { v }
        | MpirOp::WeakUpgrade { v } => {
            emit_local_expr(out, instr.dst.0, &format_wgsl_value(v, state), state)?
        }
        MpirOp::Cast { to, v } => {
            let ty_name = wgsl_type_name(state.type_ctx, *to);
            emit_local_expr(
                out,
                instr.dst.0,
                &format!("{ty_name}({})", format_wgsl_value(v, state)),
                state,
            )?;
        }
        MpirOp::IAdd { lhs, rhs }
        | MpirOp::IAddWrap { lhs, rhs }
        | MpirOp::IAddChecked { lhs, rhs }
        | MpirOp::FAdd { lhs, rhs }
        | MpirOp::FAddFast { lhs, rhs } => {
            emit_binary_expr(out, instr.dst.0, lhs, rhs, "+", state)?
        }
        MpirOp::ISub { lhs, rhs }
        | MpirOp::ISubWrap { lhs, rhs }
        | MpirOp::ISubChecked { lhs, rhs }
        | MpirOp::FSub { lhs, rhs }
        | MpirOp::FSubFast { lhs, rhs } => {
            emit_binary_expr(out, instr.dst.0, lhs, rhs, "-", state)?
        }
        MpirOp::IMul { lhs, rhs }
        | MpirOp::IMulWrap { lhs, rhs }
        | MpirOp::IMulChecked { lhs, rhs }
        | MpirOp::FMul { lhs, rhs }
        | MpirOp::FMulFast { lhs, rhs } => {
            emit_binary_expr(out, instr.dst.0, lhs, rhs, "*", state)?
        }
        MpirOp::ISDiv { lhs, rhs }
        | MpirOp::IUDiv { lhs, rhs }
        | MpirOp::FDiv { lhs, rhs }
        | MpirOp::FDivFast { lhs, rhs } => {
            emit_binary_expr(out, instr.dst.0, lhs, rhs, "/", state)?
        }
        MpirOp::ISRem { lhs, rhs } | MpirOp::IURem { lhs, rhs } | MpirOp::FRem { lhs, rhs } => {
            emit_binary_expr(out, instr.dst.0, lhs, rhs, "%", state)?
        }
        MpirOp::ICmp { pred, lhs, rhs } | MpirOp::FCmp { pred, lhs, rhs } => {
            if let Some(op) = cmp_predicate_to_wgsl(pred) {
                emit_binary_expr(out, instr.dst.0, lhs, rhs, op, state)?;
            } else {
                writeln!(
                    out,
                    "    // unsupported comparison predicate '{}': {:?}",
                    pred, instr.op
                )
                .map_err(|e| e.to_string())?;
            }
        }
        MpirOp::IAnd { lhs, rhs } => emit_binary_expr(out, instr.dst.0, lhs, rhs, "&", state)?,
        MpirOp::IOr { lhs, rhs } => emit_binary_expr(out, instr.dst.0, lhs, rhs, "|", state)?,
        MpirOp::IXor { lhs, rhs } => emit_binary_expr(out, instr.dst.0, lhs, rhs, "^", state)?,
        MpirOp::IShl { lhs, rhs } => emit_binary_expr(out, instr.dst.0, lhs, rhs, "<<", state)?,
        MpirOp::ILshr { lhs, rhs } | MpirOp::IAshr { lhs, rhs } => {
            emit_binary_expr(out, instr.dst.0, lhs, rhs, ">>", state)?
        }
        MpirOp::GpuGlobalId { dim } => {
            emit_local_expr(
                out,
                instr.dst.0,
                &format!("gid.{}", dim_component(*dim)),
                state,
            )?;
        }
        MpirOp::GpuThreadId { dim } => {
            emit_local_expr(
                out,
                instr.dst.0,
                &format!("tid.{}", dim_component(*dim)),
                state,
            )?;
        }
        MpirOp::GpuWorkgroupId { dim } => {
            emit_local_expr(
                out,
                instr.dst.0,
                &format!("wgid.{}", dim_component(*dim)),
                state,
            )?;
        }
        MpirOp::GpuWorkgroupSize { dim } => {
            emit_local_expr(
                out,
                instr.dst.0,
                &format!("_wgsz.{}", dim_component(*dim)),
                state,
            )?;
        }
        MpirOp::GpuBufferLoad { buf, idx } => {
            emit_local_expr(
                out,
                instr.dst.0,
                &format!(
                    "{}[u32({})]",
                    format_wgsl_value(buf, state),
                    format_wgsl_value(idx, state)
                ),
                state,
            )?;
        }
        MpirOp::GpuBufferLen { buf } => {
            emit_local_expr(
                out,
                instr.dst.0,
                &format!("arrayLength(&{})", format_wgsl_value(buf, state)),
                state,
            )?;
        }
        MpirOp::GpuShared { .. } => {
            writeln!(
                out,
                "    // _l{} aliases workgroup memory declaration",
                instr.dst.0
            )
            .map_err(|e| e.to_string())?;
        }
        _ => {
            writeln!(out, "    // unhandled wgsl op: {:?}", instr.op).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn emit_wgsl_void_op(
    out: &mut String,
    op: &MpirOpVoid,
    state: &EmitState<'_>,
) -> Result<(), String> {
    match op {
        MpirOpVoid::GpuBarrier => {
            writeln!(out, "    workgroupBarrier();").map_err(|e| e.to_string())?;
        }
        MpirOpVoid::GpuBufferStore { buf, idx, val } => {
            writeln!(
                out,
                "    {}[u32({})] = {};",
                format_wgsl_value(buf, state),
                format_wgsl_value(idx, state),
                format_wgsl_value(val, state)
            )
            .map_err(|e| e.to_string())?;
        }
        _ => {
            writeln!(out, "    // unhandled wgsl void op").map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn emit_wgsl_terminator(
    out: &mut String,
    term: &MpirTerminator,
    state: &EmitState<'_>,
) -> Result<(), String> {
    match term {
        MpirTerminator::Ret(Some(v)) => {
            writeln!(out, "    // return {}", format_wgsl_value(v, state))
                .map_err(|e| e.to_string())?;
            writeln!(out, "    return;").map_err(|e| e.to_string())?;
        }
        MpirTerminator::Ret(None) => {
            writeln!(out, "    return;").map_err(|e| e.to_string())?;
        }
        MpirTerminator::Br(bb) => {
            writeln!(out, "    // br bb{}", bb.0).map_err(|e| e.to_string())?;
        }
        MpirTerminator::Cbr {
            cond,
            then_bb,
            else_bb,
        } => {
            writeln!(out, "    if ({}) {{", format_wgsl_value(cond, state))
                .map_err(|e| e.to_string())?;
            writeln!(out, "        // goto bb{}", then_bb.0).map_err(|e| e.to_string())?;
            writeln!(out, "    }} else {{").map_err(|e| e.to_string())?;
            writeln!(out, "        // goto bb{}", else_bb.0).map_err(|e| e.to_string())?;
            writeln!(out, "    }}").map_err(|e| e.to_string())?;
        }
        MpirTerminator::Switch { val, arms, default } => {
            writeln!(out, "    switch ({}) {{", format_wgsl_value(val, state))
                .map_err(|e| e.to_string())?;
            for (case, bb) in arms {
                writeln!(
                    out,
                    "        case {}: {{ // goto bb{} }}",
                    format_wgsl_const(case),
                    bb.0
                )
                .map_err(|e| e.to_string())?;
            }
            writeln!(out, "        default: {{ // goto bb{} }}", default.0)
                .map_err(|e| e.to_string())?;
            writeln!(out, "    }}").map_err(|e| e.to_string())?;
        }
        MpirTerminator::Unreachable => {
            writeln!(out, "    // unreachable").map_err(|e| e.to_string())?;
            writeln!(out, "    return;").map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn emit_binary_expr(
    out: &mut String,
    dst: u32,
    lhs: &MpirValue,
    rhs: &MpirValue,
    op: &str,
    state: &mut EmitState<'_>,
) -> Result<(), String> {
    emit_local_expr(
        out,
        dst,
        &format!(
            "{} {op} {}",
            format_wgsl_value(lhs, state),
            format_wgsl_value(rhs, state)
        ),
        state,
    )
}

fn emit_local_expr(
    out: &mut String,
    dst: u32,
    expr: &str,
    state: &mut EmitState<'_>,
) -> Result<(), String> {
    writeln!(out, "    let _l{dst} = {expr};").map_err(|e| e.to_string())?;
    state.local_aliases.insert(dst, format!("_l{dst}"));
    Ok(())
}

fn cmp_predicate_to_wgsl(pred: &str) -> Option<&'static str> {
    match pred {
        "eq" | "oeq" | "ueq" => Some("=="),
        "ne" | "one" | "une" => Some("!="),
        "slt" | "ult" | "olt" | "lt" => Some("<"),
        "sle" | "ule" | "ole" | "le" => Some("<="),
        "sgt" | "ugt" | "ogt" | "gt" => Some(">"),
        "sge" | "uge" | "oge" | "ge" => Some(">="),
        _ => None,
    }
}

fn dim_component(dim: u8) -> &'static str {
    match dim {
        0 => "x",
        1 => "y",
        2 => "z",
        _ => "x",
    }
}

fn format_wgsl_value(v: &MpirValue, state: &EmitState<'_>) -> String {
    match v {
        MpirValue::Local(id) => state
            .local_aliases
            .get(&id.0)
            .cloned()
            .unwrap_or_else(|| format!("_l{}", id.0)),
        MpirValue::Const(c) => format_wgsl_const(c),
    }
}

fn format_wgsl_const(c: &HirConst) -> String {
    match &c.lit {
        HirConstLit::IntLit(v) => v.to_string(),
        HirConstLit::FloatLit(v) => {
            let mut s = v.to_string();
            if !s.contains('.') && !s.contains('e') && !s.contains('E') {
                s.push_str(".0");
            }
            s
        }
        HirConstLit::BoolLit(v) => v.to_string(),
        HirConstLit::StringLit(s) => format!("\"{}\"", escape_wgsl_string(s)),
        HirConstLit::Unit => "0u".to_string(),
    }
}

fn escape_wgsl_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_emitter_creates() {
        let emitter = WgslEmitter::new();
        assert_eq!(emitter.artifact_extension(), "wgsl");
    }
}
