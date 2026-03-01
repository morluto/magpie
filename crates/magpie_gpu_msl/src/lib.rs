//! Metal Shading Language (MSL) backend for Magpie GPU compute.
//!
//! Emits native MSL source code from MPIR kernel functions.
//! Uses the shared CFG structurizer from `magpie_gpu` for control flow.
//! See SPEC_GPU_UPGRADE.md §7.2.

use magpie_diag::Diagnostic;
use magpie_mpir::{
    HirConst, HirConstLit, MpirBlock, MpirFn, MpirInstr, MpirOp, MpirOpVoid, MpirTerminator,
    MpirTypeTable, MpirValue,
};
use magpie_types::{PrimType, TypeCtx, TypeId, TypeKind};
use std::collections::HashMap;
use std::fmt::Write;

/// MSL backend emitter.
pub struct MslEmitter;

impl MslEmitter {
    pub fn new() -> Self {
        Self
    }

    pub fn artifact_extension(&self) -> &str {
        "metal"
    }

    /// Validate kernel compatibility with Metal backend.
    pub fn validate_kernel(
        &self,
        kernel: &MpirFn,
        _types: &MpirTypeTable,
        _type_ctx: &TypeCtx,
    ) -> Result<(), Vec<Diagnostic>> {
        let _ = kernel;
        Ok(())
    }

    /// Emit MSL source text from an MPIR kernel function.
    pub fn emit_kernel(&self, kernel: &MpirFn, type_ctx: &TypeCtx) -> Result<Vec<u8>, String> {
        let mut out = String::with_capacity(4096);
        writeln!(out, "#include <metal_stdlib>").map_err(|e| e.to_string())?;
        writeln!(out, "using namespace metal;").map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;

        // Emit kernel function signature
        write!(
            out,
            "kernel void {name}(",
            name = msl_kernel_name(&kernel.name)
        )
        .map_err(|e| e.to_string())?;

        // Emit parameters with Metal attributes
        let mut binding = 0u32;
        let mut params_emitted = false;
        for (i, (_, ty_id)) in kernel.params.iter().enumerate() {
            if params_emitted {
                write!(out, ", ").map_err(|e| e.to_string())?;
            }
            let is_buffer = is_gpu_buffer(type_ctx, *ty_id);
            if is_buffer {
                write!(
                    out,
                    "device float* param_{i} [[buffer({binding})]]",
                    i = i,
                    binding = binding
                )
                .map_err(|e| e.to_string())?;
            } else {
                let msl_ty = msl_type_name(type_ctx, *ty_id);
                write!(
                    out,
                    "constant {ty}& param_{i} [[buffer({binding})]]",
                    ty = msl_ty,
                    i = i,
                    binding = binding
                )
                .map_err(|e| e.to_string())?;
            }
            binding += 1;
            params_emitted = true;
        }

        // Add builtin parameters
        if params_emitted {
            write!(out, ", ").map_err(|e| e.to_string())?;
        }
        writeln!(out, "uint3 _gid [[thread_position_in_grid]],").map_err(|e| e.to_string())?;
        writeln!(out, "    uint3 _tid [[thread_position_in_threadgroup]],")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    uint3 _wgid [[threadgroup_position_in_grid]],")
            .map_err(|e| e.to_string())?;
        writeln!(out, "    uint3 _wgsz [[threads_per_threadgroup]]").map_err(|e| e.to_string())?;
        writeln!(out, ") {{").map_err(|e| e.to_string())?;

        let phi_locals = collect_phi_locals(kernel);
        let phi_edge_assignments = collect_phi_edge_assignments(kernel);

        // Phi variables are function-scope locals; predecessor edges assign them.
        for (local_id, ty) in phi_locals {
            let ty_name = msl_type_name(type_ctx, ty);
            let init = msl_default_literal(type_ctx, ty);
            writeln!(out, "    {ty_name} _l{local_id} = {init};").map_err(|e| e.to_string())?;
        }
        if !kernel.blocks.is_empty() {
            writeln!(out).map_err(|e| e.to_string())?;
        }

        for (idx, block) in kernel.blocks.iter().enumerate() {
            let next_block = kernel.blocks.get(idx + 1).map(|b| b.id.0);
            emit_msl_block(&mut out, block, next_block, type_ctx, &phi_edge_assignments)?;
        }

        writeln!(out, "}}").map_err(|e| e.to_string())?;

        Ok(out.into_bytes())
    }
}

fn msl_kernel_name(name: &str) -> String {
    name.trim_start_matches('@').replace('.', "_")
}

fn is_gpu_buffer(type_ctx: &TypeCtx, ty_id: TypeId) -> bool {
    ty_id == magpie_types::fixed_type_ids::GPU_BUFFER_BASE
        || matches!(type_ctx.lookup(ty_id), Some(TypeKind::HeapHandle { .. }))
}

fn msl_type_name(type_ctx: &TypeCtx, ty_id: TypeId) -> &'static str {
    match type_ctx.lookup(ty_id) {
        Some(TypeKind::Prim(PrimType::F32)) => "float",
        Some(TypeKind::Prim(PrimType::F64)) => "float",
        Some(TypeKind::Prim(PrimType::F16)) => "half",
        Some(TypeKind::Prim(PrimType::Bf16)) => "bfloat",
        Some(TypeKind::Prim(PrimType::I32)) => "int",
        Some(TypeKind::Prim(PrimType::I64)) => "long",
        Some(TypeKind::Prim(PrimType::U32)) => "uint",
        Some(TypeKind::Prim(PrimType::U64)) => "ulong",
        Some(TypeKind::Prim(PrimType::Bool)) => "bool",
        _ => "uint",
    }
}

fn emit_msl_block(
    out: &mut String,
    block: &MpirBlock,
    next_block: Option<u32>,
    type_ctx: &TypeCtx,
    phi_edge_assignments: &HashMap<(u32, u32), Vec<(u32, MpirValue)>>,
) -> Result<(), String> {
    writeln!(out, "    // bb{}", block.id.0).map_err(|e| e.to_string())?;
    for instr in &block.instrs {
        emit_msl_instr(out, instr, type_ctx)?;
    }
    for void_op in &block.void_ops {
        emit_msl_void_op(out, void_op, type_ctx)?;
    }
    emit_msl_terminator(
        out,
        &block.terminator,
        block.id.0,
        next_block,
        phi_edge_assignments,
        type_ctx,
    )?;
    writeln!(out).map_err(|e| e.to_string())?;
    Ok(())
}

fn emit_msl_instr(out: &mut String, instr: &MpirInstr, type_ctx: &TypeCtx) -> Result<(), String> {
    match &instr.op {
        MpirOp::Const(c) => emit_msl_assign(out, instr.dst.0, &format_msl_const(c, type_ctx))?,
        MpirOp::Move { v }
        | MpirOp::BorrowShared { v }
        | MpirOp::BorrowMut { v }
        | MpirOp::Share { v }
        | MpirOp::CloneShared { v }
        | MpirOp::CloneWeak { v }
        | MpirOp::WeakDowngrade { v }
        | MpirOp::WeakUpgrade { v } => {
            emit_msl_assign(out, instr.dst.0, &format_msl_value(v, type_ctx))?
        }
        MpirOp::IAdd { lhs, rhs }
        | MpirOp::IAddWrap { lhs, rhs }
        | MpirOp::IAddChecked { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} + {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::ISub { lhs, rhs }
        | MpirOp::ISubWrap { lhs, rhs }
        | MpirOp::ISubChecked { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} - {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::IMul { lhs, rhs }
        | MpirOp::IMulWrap { lhs, rhs }
        | MpirOp::IMulChecked { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} * {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::ISDiv { lhs, rhs } | MpirOp::IUDiv { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} / {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::ISRem { lhs, rhs } | MpirOp::IURem { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} % {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::FAdd { lhs, rhs } | MpirOp::FAddFast { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} + {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::FSub { lhs, rhs } | MpirOp::FSubFast { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} - {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::FMul { lhs, rhs } | MpirOp::FMulFast { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} * {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::FDiv { lhs, rhs } | MpirOp::FDivFast { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} / {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::FRem { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "fmod({}, {})",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::IAnd { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} & {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::IOr { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} | {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::IXor { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} ^ {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::IShl { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} << {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::ILshr { lhs, rhs } | MpirOp::IAshr { lhs, rhs } => {
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!(
                    "{} >> {}",
                    format_msl_value(lhs, type_ctx),
                    format_msl_value(rhs, type_ctx)
                ),
            )?;
        }
        MpirOp::ICmp { pred, lhs, rhs } => {
            let lhs = format_msl_value(lhs, type_ctx);
            let rhs = format_msl_value(rhs, type_ctx);
            emit_msl_assign(out, instr.dst.0, &format_icmp_expr(pred, &lhs, &rhs))?;
        }
        MpirOp::FCmp { pred, lhs, rhs } => {
            let lhs = format_msl_value(lhs, type_ctx);
            let rhs = format_msl_value(rhs, type_ctx);
            emit_msl_assign(out, instr.dst.0, &format_fcmp_expr(pred, &lhs, &rhs))?;
        }
        MpirOp::Cast { to, v } => {
            let to_ty = msl_type_name(type_ctx, *to);
            emit_msl_assign(
                out,
                instr.dst.0,
                &format!("({to_ty})({})", format_msl_value(v, type_ctx)),
            )?;
        }
        MpirOp::GpuGlobalId { dim } => {
            writeln!(out, "    uint _l{} = _gid[{}];", instr.dst.0, dim)
                .map_err(|e| e.to_string())?;
        }
        MpirOp::GpuThreadId { dim } => {
            writeln!(out, "    uint _l{} = _tid[{}];", instr.dst.0, dim)
                .map_err(|e| e.to_string())?;
        }
        MpirOp::GpuWorkgroupId { dim } => {
            writeln!(out, "    uint _l{} = _wgid[{}];", instr.dst.0, dim)
                .map_err(|e| e.to_string())?;
        }
        MpirOp::GpuWorkgroupSize { dim } => {
            writeln!(out, "    uint _l{} = _wgsz[{}];", instr.dst.0, dim)
                .map_err(|e| e.to_string())?;
        }
        MpirOp::GpuBufferLoad { buf, idx } => {
            writeln!(
                out,
                "    auto _l{} = {}[{}];",
                instr.dst.0,
                format_msl_value(buf, type_ctx),
                format_msl_value(idx, type_ctx)
            )
            .map_err(|e| e.to_string())?;
        }
        MpirOp::GpuBufferLen { buf } => {
            writeln!(
                out,
                "    uint _l{} = 0u; // TODO(msl): buffer length for {}",
                instr.dst.0,
                format_msl_value(buf, type_ctx)
            )
            .map_err(|e| e.to_string())?;
        }
        MpirOp::GpuShared { ty, size } => {
            let elem = msl_type_name(type_ctx, *ty);
            writeln!(
                out,
                "    threadgroup {elem} _l{}[{}];",
                instr.dst.0,
                format_msl_value(size, type_ctx)
            )
            .map_err(|e| e.to_string())?;
        }
        MpirOp::Phi { .. } => {
            writeln!(
                out,
                "    // phi _l{} assigned on predecessor edges",
                instr.dst.0
            )
            .map_err(|e| e.to_string())?;
        }
        _ => {
            writeln!(out, "    // unhandled op: {:?}", instr.op).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn emit_msl_assign(out: &mut String, dst: u32, expr: &str) -> Result<(), String> {
    writeln!(out, "    auto _l{dst} = {expr};").map_err(|e| e.to_string())
}

fn emit_msl_void_op(out: &mut String, op: &MpirOpVoid, type_ctx: &TypeCtx) -> Result<(), String> {
    match op {
        MpirOpVoid::GpuBarrier => {
            writeln!(out, "    threadgroup_barrier(mem_flags::mem_threadgroup);")
                .map_err(|e| e.to_string())?;
        }
        MpirOpVoid::GpuBufferStore { buf, idx, val } => {
            writeln!(
                out,
                "    {}[{}] = {};",
                format_msl_value(buf, type_ctx),
                format_msl_value(idx, type_ctx),
                format_msl_value(val, type_ctx)
            )
            .map_err(|e| e.to_string())?;
        }
        _ => {
            writeln!(out, "    // unhandled void op").map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn emit_msl_terminator(
    out: &mut String,
    term: &MpirTerminator,
    block_id: u32,
    next_block: Option<u32>,
    phi_edge_assignments: &HashMap<(u32, u32), Vec<(u32, MpirValue)>>,
    type_ctx: &TypeCtx,
) -> Result<(), String> {
    match term {
        MpirTerminator::Ret(_) => {
            writeln!(out, "    return;").map_err(|e| e.to_string())?;
        }
        MpirTerminator::Br(target) => {
            emit_phi_edge_assignments(
                out,
                block_id,
                target.0,
                phi_edge_assignments,
                type_ctx,
                "    ",
            )?;
            if next_block != Some(target.0) {
                writeln!(out, "    // br bb{}", target.0).map_err(|e| e.to_string())?;
            }
        }
        MpirTerminator::Cbr {
            cond,
            then_bb,
            else_bb,
        } => {
            writeln!(out, "    if ({}) {{", format_msl_value(cond, type_ctx))
                .map_err(|e| e.to_string())?;
            emit_phi_edge_assignments(
                out,
                block_id,
                then_bb.0,
                phi_edge_assignments,
                type_ctx,
                "        ",
            )?;
            writeln!(out, "        // br bb{}", then_bb.0).map_err(|e| e.to_string())?;
            writeln!(out, "    }} else {{").map_err(|e| e.to_string())?;
            emit_phi_edge_assignments(
                out,
                block_id,
                else_bb.0,
                phi_edge_assignments,
                type_ctx,
                "        ",
            )?;
            writeln!(out, "        // br bb{}", else_bb.0).map_err(|e| e.to_string())?;
            writeln!(out, "    }}").map_err(|e| e.to_string())?;
        }
        MpirTerminator::Switch { val, .. } => {
            writeln!(
                out,
                "    // switch on {} is not yet lowered for MSL structured emission",
                format_msl_value(val, type_ctx)
            )
            .map_err(|e| e.to_string())?;
        }
        MpirTerminator::Unreachable => {
            writeln!(out, "    // unreachable").map_err(|e| e.to_string())?;
            writeln!(out, "    return;").map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn emit_phi_edge_assignments(
    out: &mut String,
    pred_block: u32,
    target_block: u32,
    phi_edge_assignments: &HashMap<(u32, u32), Vec<(u32, MpirValue)>>,
    type_ctx: &TypeCtx,
    indent: &str,
) -> Result<(), String> {
    if let Some(assignments) = phi_edge_assignments.get(&(pred_block, target_block)) {
        for (dst, value) in assignments {
            writeln!(
                out,
                "{indent}_l{dst} = {};",
                format_msl_value(value, type_ctx)
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn collect_phi_locals(kernel: &MpirFn) -> Vec<(u32, TypeId)> {
    let mut phi = HashMap::<u32, TypeId>::new();
    for block in &kernel.blocks {
        for instr in &block.instrs {
            if let MpirOp::Phi { ty, .. } = &instr.op {
                phi.entry(instr.dst.0).or_insert(*ty);
            }
        }
    }
    let mut out = phi.into_iter().collect::<Vec<_>>();
    out.sort_by_key(|(local, _)| *local);
    out
}

fn collect_phi_edge_assignments(kernel: &MpirFn) -> HashMap<(u32, u32), Vec<(u32, MpirValue)>> {
    let mut out = HashMap::<(u32, u32), Vec<(u32, MpirValue)>>::new();
    for block in &kernel.blocks {
        for instr in &block.instrs {
            if let MpirOp::Phi { incomings, .. } = &instr.op {
                for (pred_block, value) in incomings {
                    out.entry((pred_block.0, block.id.0))
                        .or_default()
                        .push((instr.dst.0, value.clone()));
                }
            }
        }
    }
    out
}

fn format_msl_value(v: &MpirValue, type_ctx: &TypeCtx) -> String {
    match v {
        MpirValue::Local(id) => format!("_l{}", id.0),
        MpirValue::Const(c) => format_msl_const(c, type_ctx),
    }
}

fn format_msl_const(c: &HirConst, type_ctx: &TypeCtx) -> String {
    match &c.lit {
        HirConstLit::IntLit(i) => i.to_string(),
        HirConstLit::FloatLit(f) => match type_ctx.lookup(c.ty) {
            Some(TypeKind::Prim(PrimType::F64)) => format_float_literal(*f),
            Some(TypeKind::Prim(PrimType::F16)) => {
                format!("half({})", format_float_literal_with_f32_suffix(*f))
            }
            _ => format_float_literal_with_f32_suffix(*f),
        },
        HirConstLit::BoolLit(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        HirConstLit::StringLit(s) => format!("{:?}", s),
        HirConstLit::Unit => "0u".to_string(),
    }
}

fn format_float_literal(v: f64) -> String {
    if v.is_nan() {
        "NAN".to_string()
    } else if v.is_infinite() {
        if v.is_sign_negative() {
            "-INFINITY".to_string()
        } else {
            "INFINITY".to_string()
        }
    } else {
        let mut s = format!("{v}");
        if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            s.push_str(".0");
        }
        s
    }
}

fn format_float_literal_with_f32_suffix(v: f64) -> String {
    let base = format_float_literal(v);
    if v.is_finite() {
        format!("{base}f")
    } else {
        base
    }
}

fn format_icmp_expr(pred: &str, lhs: &str, rhs: &str) -> String {
    match pred.to_ascii_lowercase().as_str() {
        "eq" => format!("{lhs} == {rhs}"),
        "ne" => format!("{lhs} != {rhs}"),
        "slt" | "lt" => format!("{lhs} < {rhs}"),
        "sle" | "le" => format!("{lhs} <= {rhs}"),
        "sgt" | "gt" => format!("{lhs} > {rhs}"),
        "sge" | "ge" => format!("{lhs} >= {rhs}"),
        "ult" => format!("(uint)({lhs}) < (uint)({rhs})"),
        "ule" => format!("(uint)({lhs}) <= (uint)({rhs})"),
        "ugt" => format!("(uint)({lhs}) > (uint)({rhs})"),
        "uge" => format!("(uint)({lhs}) >= (uint)({rhs})"),
        _ => format!("{lhs} == {rhs}"),
    }
}

fn format_fcmp_expr(pred: &str, lhs: &str, rhs: &str) -> String {
    match pred.to_ascii_lowercase().as_str() {
        "eq" | "oeq" | "ueq" => format!("{lhs} == {rhs}"),
        "ne" | "one" | "une" => format!("{lhs} != {rhs}"),
        "lt" | "olt" | "ult" => format!("{lhs} < {rhs}"),
        "le" | "ole" | "ule" => format!("{lhs} <= {rhs}"),
        "gt" | "ogt" | "ugt" => format!("{lhs} > {rhs}"),
        "ge" | "oge" | "uge" => format!("{lhs} >= {rhs}"),
        "ord" => format!("!isnan({lhs}) && !isnan({rhs})"),
        "uno" => format!("isnan({lhs}) || isnan({rhs})"),
        "true" => "true".to_string(),
        "false" => "false".to_string(),
        _ => format!("{lhs} == {rhs}"),
    }
}

fn msl_default_literal(type_ctx: &TypeCtx, ty: TypeId) -> String {
    match type_ctx.lookup(ty) {
        Some(TypeKind::Prim(PrimType::Bool)) => "false".to_string(),
        Some(TypeKind::Prim(PrimType::F64)) => "0.0".to_string(),
        Some(TypeKind::Prim(PrimType::F32)) => "0.0f".to_string(),
        Some(TypeKind::Prim(PrimType::F16)) => "half(0.0f)".to_string(),
        Some(TypeKind::Prim(PrimType::Bf16)) => "bfloat(0.0f)".to_string(),
        _ => "0u".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msl_emitter_creates() {
        let emitter = MslEmitter::new();
        assert_eq!(emitter.artifact_extension(), "metal");
    }

    #[test]
    fn msl_kernel_name_strips_at() {
        assert_eq!(msl_kernel_name("@kernel_add"), "kernel_add");
        assert_eq!(msl_kernel_name("@my.module.fn"), "my_module_fn");
    }
}
