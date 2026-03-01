//! HIP/ROCm HSACO backend for Magpie GPU compute.
//!
//! Generates LLVM IR with amdgcn-amd-amdhsa triple, then shells out
//! to `llc -march=amdgcn` + `ld.lld` to produce HSACO ELF.
//! See SPEC_GPU_UPGRADE.md §7.4.

use magpie_diag::Diagnostic;
use magpie_mpir::{
    HirConst, HirConstLit, MpirFn, MpirInstr, MpirOp, MpirOpVoid, MpirTypeTable, MpirValue,
};
use magpie_types::{fixed_type_ids, PrimType, TypeCtx, TypeId, TypeKind};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::process::Command;

#[derive(Clone)]
struct Operand {
    ty: String,
    ty_id: TypeId,
    repr: String,
}

#[derive(Clone)]
struct SharedDecl {
    symbol: String,
    elem_ty: String,
    count: u64,
}

/// HIP/HSACO backend emitter.
pub struct HipEmitter {
    llc_path: Option<String>,
    lld_path: Option<String>,
    gfx_arch: String,
}

impl HipEmitter {
    pub fn new() -> Self {
        Self {
            llc_path: None,
            lld_path: None,
            gfx_arch: "gfx1030".to_string(),
        }
    }

    pub fn with_llc_path(mut self, path: String) -> Self {
        self.llc_path = Some(path);
        self
    }

    pub fn with_lld_path(mut self, path: String) -> Self {
        self.lld_path = Some(path);
        self
    }

    pub fn artifact_extension(&self) -> &str {
        "hsaco"
    }

    pub fn validate_kernel(
        &self,
        kernel: &MpirFn,
        _types: &MpirTypeTable,
        _type_ctx: &TypeCtx,
    ) -> Result<(), Vec<Diagnostic>> {
        let _ = kernel;
        Ok(())
    }

    /// Emit LLVM IR for the amdgcn target.
    pub fn emit_llvm_ir(&self, kernel: &MpirFn, type_ctx: &TypeCtx) -> Result<String, String> {
        let mut out = String::with_capacity(8192);

        let local_tys = collect_local_types(kernel);
        let shared = collect_shared_decls(kernel, type_ctx);

        writeln!(
            out,
            "target datalayout = \"e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32\""
        )
        .map_err(|e| e.to_string())?;
        writeln!(out, "target triple = \"amdgcn-amd-amdhsa\"").map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;

        // Declare AMDGCN intrinsics
        writeln!(out, "declare i32 @llvm.amdgcn.workitem.id.x()").map_err(|e| e.to_string())?;
        writeln!(out, "declare i32 @llvm.amdgcn.workitem.id.y()").map_err(|e| e.to_string())?;
        writeln!(out, "declare i32 @llvm.amdgcn.workitem.id.z()").map_err(|e| e.to_string())?;
        writeln!(out, "declare i32 @llvm.amdgcn.workgroup.id.x()").map_err(|e| e.to_string())?;
        writeln!(out, "declare i32 @llvm.amdgcn.workgroup.id.y()").map_err(|e| e.to_string())?;
        writeln!(out, "declare i32 @llvm.amdgcn.workgroup.id.z()").map_err(|e| e.to_string())?;
        writeln!(out, "declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()")
            .map_err(|e| e.to_string())?;
        writeln!(out, "declare void @llvm.amdgcn.s.barrier()").map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;

        // Shared-memory globals (addrspace(3)).
        for decl in shared.values() {
            writeln!(
                out,
                "@{} = addrspace(3) global [{} x {}] undef",
                decl.symbol, decl.count, decl.elem_ty
            )
            .map_err(|e| e.to_string())?;
        }
        if !shared.is_empty() {
            writeln!(out).map_err(|e| e.to_string())?;
        }

        let fn_name = kernel.name.trim_start_matches('@');
        write!(out, "define amdgpu_kernel void @{fn_name}(").map_err(|e| e.to_string())?;

        for (i, (local_id, ty_id)) in kernel.params.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").map_err(|e| e.to_string())?;
            }
            let llvm_ty = llvm_type_for_hip(type_ctx, *ty_id);
            write!(out, "{llvm_ty} %_l{}", local_id.0).map_err(|e| e.to_string())?;
        }

        writeln!(out, ") #0 {{").map_err(|e| e.to_string())?;
        writeln!(out, "entry:").map_err(|e| e.to_string())?;

        let mut tmp_counter = 0_u32;
        for block in &kernel.blocks {
            for instr in &block.instrs {
                emit_hip_instr(
                    &mut out,
                    instr,
                    type_ctx,
                    &local_tys,
                    &shared,
                    &mut tmp_counter,
                )?;
            }
            for void_op in &block.void_ops {
                emit_hip_void_op(&mut out, void_op, type_ctx, &local_tys, &mut tmp_counter)?;
            }
        }

        writeln!(out, "  ret void").map_err(|e| e.to_string())?;
        writeln!(out, "}}").map_err(|e| e.to_string())?;
        writeln!(out).map_err(|e| e.to_string())?;
        writeln!(
            out,
            "attributes #0 = {{ \"amdgpu-flat-work-group-size\"=\"64,256\" }}"
        )
        .map_err(|e| e.to_string())?;

        Ok(out)
    }

    pub fn emit_kernel(&self, kernel: &MpirFn, type_ctx: &TypeCtx) -> Result<Vec<u8>, String> {
        let llvm_ir = self.emit_llvm_ir(kernel, type_ctx)?;

        let tmp_dir = std::env::temp_dir();
        let ir_path = tmp_dir.join(format!("{}.amdgcn.ll", kernel.name.trim_start_matches('@')));
        let obj_path = tmp_dir.join(format!("{}.o", kernel.name.trim_start_matches('@')));
        let hsaco_path = tmp_dir.join(format!("{}.hsaco", kernel.name.trim_start_matches('@')));

        std::fs::write(&ir_path, &llvm_ir).map_err(|e| format!("Failed to write LLVM IR: {e}"))?;

        let llc = self.llc_path.as_deref().unwrap_or("llc");
        let output = Command::new(llc)
            .args([
                "-march=amdgcn",
                &format!("-mcpu={}", self.gfx_arch),
                "-O2",
                "-filetype=obj",
                "-o",
                obj_path.to_str().unwrap(),
                ir_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("Failed to invoke llc: {e}"))?;

        if !output.status.success() {
            return Err(format!(
                "llc failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        let lld = self.lld_path.as_deref().unwrap_or("ld.lld");
        let output = Command::new(lld)
            .args([
                "-shared",
                "-o",
                hsaco_path.to_str().unwrap(),
                obj_path.to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("Failed to invoke ld.lld: {e}"))?;

        if !output.status.success() {
            return Err(format!(
                "ld.lld failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        std::fs::read(&hsaco_path).map_err(|e| format!("Failed to read HSACO: {e}"))
    }
}

fn emit_hip_instr(
    out: &mut String,
    instr: &MpirInstr,
    type_ctx: &TypeCtx,
    local_tys: &HashMap<u32, TypeId>,
    shared: &BTreeMap<u32, SharedDecl>,
    tmp_counter: &mut u32,
) -> Result<(), String> {
    match &instr.op {
        MpirOp::Const(c) => {
            let op = const_operand(c, type_ctx);
            emit_assign_from_operand(out, instr.dst.0, instr.ty, &op, type_ctx, tmp_counter)?;
        }
        MpirOp::Move { v }
        | MpirOp::BorrowShared { v }
        | MpirOp::BorrowMut { v }
        | MpirOp::Share { v }
        | MpirOp::CloneShared { v }
        | MpirOp::CloneWeak { v }
        | MpirOp::WeakDowngrade { v }
        | MpirOp::WeakUpgrade { v } => {
            let op = operand_for_value(v, type_ctx, local_tys)?;
            emit_assign_from_operand(out, instr.dst.0, instr.ty, &op, type_ctx, tmp_counter)?;
        }
        MpirOp::IAdd { lhs, rhs } | MpirOp::IAddWrap { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "add",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::ISub { lhs, rhs } | MpirOp::ISubWrap { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "sub",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::IMul { lhs, rhs } | MpirOp::IMulWrap { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "mul",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::ISDiv { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "sdiv",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::IUDiv { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "udiv",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::ISRem { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "srem",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::IURem { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "urem",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::IAnd { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "and",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::IOr { lhs, rhs } => {
            emit_binary_op(out, instr, "or", lhs, rhs, type_ctx, local_tys, tmp_counter)?
        }
        MpirOp::IXor { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "xor",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::IShl { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "shl",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::ILshr { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "lshr",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::IAshr { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "ashr",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::FAdd { lhs, rhs } | MpirOp::FAddFast { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "fadd",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::FSub { lhs, rhs } | MpirOp::FSubFast { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "fsub",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::FMul { lhs, rhs } | MpirOp::FMulFast { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "fmul",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::FDiv { lhs, rhs } | MpirOp::FDivFast { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "fdiv",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::FRem { lhs, rhs } => emit_binary_op(
            out,
            instr,
            "frem",
            lhs,
            rhs,
            type_ctx,
            local_tys,
            tmp_counter,
        )?,
        MpirOp::ICmp { pred, lhs, rhs } => {
            let lhs_op = operand_for_value(lhs, type_ctx, local_tys)?;
            let rhs_op = operand_for_value(rhs, type_ctx, local_tys)?;
            let cmp_ty = lhs_op.ty.clone();
            let rhs_repr =
                coerce_operand(out, &rhs_op, &cmp_ty, lhs_op.ty_id, type_ctx, tmp_counter)?;
            let cmp_tmp = next_tmp(tmp_counter);
            writeln!(
                out,
                "  {cmp_tmp} = icmp {} {} {}, {}",
                normalize_icmp_pred(pred),
                cmp_ty,
                lhs_op.repr,
                rhs_repr
            )
            .map_err(|e| e.to_string())?;
            let cmp_op = Operand {
                ty: "i1".to_string(),
                ty_id: fixed_type_ids::BOOL,
                repr: cmp_tmp,
            };
            emit_assign_from_operand(out, instr.dst.0, instr.ty, &cmp_op, type_ctx, tmp_counter)?;
        }
        MpirOp::FCmp { pred, lhs, rhs } => {
            let lhs_op = operand_for_value(lhs, type_ctx, local_tys)?;
            let rhs_op = operand_for_value(rhs, type_ctx, local_tys)?;
            let cmp_ty = lhs_op.ty.clone();
            let rhs_repr =
                coerce_operand(out, &rhs_op, &cmp_ty, lhs_op.ty_id, type_ctx, tmp_counter)?;
            let cmp_tmp = next_tmp(tmp_counter);
            writeln!(
                out,
                "  {cmp_tmp} = fcmp {} {} {}, {}",
                normalize_fcmp_pred(pred),
                cmp_ty,
                lhs_op.repr,
                rhs_repr
            )
            .map_err(|e| e.to_string())?;
            let cmp_op = Operand {
                ty: "i1".to_string(),
                ty_id: fixed_type_ids::BOOL,
                repr: cmp_tmp,
            };
            emit_assign_from_operand(out, instr.dst.0, instr.ty, &cmp_op, type_ctx, tmp_counter)?;
        }
        MpirOp::Cast { to, v } => {
            let src = operand_for_value(v, type_ctx, local_tys)?;
            let dst_name = format!("%_l{}", instr.dst.0);
            let dst_ty = llvm_type_for_hip(type_ctx, *to);
            emit_cast_assign(
                out,
                &dst_name,
                &dst_ty,
                instr.ty,
                &src,
                type_ctx,
                tmp_counter,
            )?;
        }
        MpirOp::GpuThreadId { dim } => {
            let suffix = dim_suffix(*dim)?;
            let tmp = next_tmp(tmp_counter);
            writeln!(
                out,
                "  {tmp} = call i32 @llvm.amdgcn.workitem.id.{suffix}()"
            )
            .map_err(|e| e.to_string())?;
            let op = Operand {
                ty: "i32".to_string(),
                ty_id: fixed_type_ids::U32,
                repr: tmp,
            };
            emit_assign_from_operand(out, instr.dst.0, instr.ty, &op, type_ctx, tmp_counter)?;
        }
        MpirOp::GpuWorkgroupId { dim } => {
            let suffix = dim_suffix(*dim)?;
            let tmp = next_tmp(tmp_counter);
            writeln!(
                out,
                "  {tmp} = call i32 @llvm.amdgcn.workgroup.id.{suffix}()"
            )
            .map_err(|e| e.to_string())?;
            let op = Operand {
                ty: "i32".to_string(),
                ty_id: fixed_type_ids::U32,
                repr: tmp,
            };
            emit_assign_from_operand(out, instr.dst.0, instr.ty, &op, type_ctx, tmp_counter)?;
        }
        MpirOp::GpuWorkgroupSize { dim } => {
            let size_i32 = emit_hip_workgroup_size_i32(out, *dim, tmp_counter)?;
            let op = Operand {
                ty: "i32".to_string(),
                ty_id: fixed_type_ids::U32,
                repr: size_i32,
            };
            emit_assign_from_operand(out, instr.dst.0, instr.ty, &op, type_ctx, tmp_counter)?;
        }
        MpirOp::GpuGlobalId { dim } => {
            let suffix = dim_suffix(*dim)?;
            let tid = next_tmp(tmp_counter);
            let bid = next_tmp(tmp_counter);
            let bdim = emit_hip_workgroup_size_i32(out, *dim, tmp_counter)?;
            let tmp = next_tmp(tmp_counter);
            writeln!(
                out,
                "  {tid} = call i32 @llvm.amdgcn.workitem.id.{suffix}()"
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                out,
                "  {bid} = call i32 @llvm.amdgcn.workgroup.id.{suffix}()"
            )
            .map_err(|e| e.to_string())?;
            writeln!(out, "  {tmp} = mul i32 {bid}, {bdim}").map_err(|e| e.to_string())?;
            let gid = next_tmp(tmp_counter);
            writeln!(out, "  {gid} = add i32 {tid}, {tmp}").map_err(|e| e.to_string())?;
            let op = Operand {
                ty: "i32".to_string(),
                ty_id: fixed_type_ids::U32,
                repr: gid,
            };
            emit_assign_from_operand(out, instr.dst.0, instr.ty, &op, type_ctx, tmp_counter)?;
        }
        MpirOp::GpuBufferLoad { buf, idx } => {
            emit_gpu_buffer_load(out, instr, buf, idx, type_ctx, local_tys, tmp_counter)?;
        }
        MpirOp::GpuShared { .. } => {
            emit_gpu_shared(out, instr, shared, type_ctx, tmp_counter)?;
        }
        MpirOp::Phi { incomings, .. } => {
            if let Some((_, incoming)) = incomings.first() {
                let op = operand_for_value(incoming, type_ctx, local_tys)?;
                emit_assign_from_operand(out, instr.dst.0, instr.ty, &op, type_ctx, tmp_counter)?;
            } else {
                emit_zero_assign(out, instr.dst.0, instr.ty, type_ctx, tmp_counter)?;
            }
        }
        MpirOp::GpuBufferLen { .. } => {
            emit_zero_assign(out, instr.dst.0, instr.ty, type_ctx, tmp_counter)?;
        }
        _ => {
            writeln!(out, "  ; unhandled hip op: {:?}", instr.op).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn emit_hip_void_op(
    out: &mut String,
    op: &MpirOpVoid,
    type_ctx: &TypeCtx,
    local_tys: &HashMap<u32, TypeId>,
    tmp_counter: &mut u32,
) -> Result<(), String> {
    match op {
        MpirOpVoid::GpuBarrier => {
            writeln!(out, "  call void @llvm.amdgcn.s.barrier()").map_err(|e| e.to_string())?;
        }
        MpirOpVoid::GpuBufferStore { buf, idx, val } => {
            emit_gpu_buffer_store(out, buf, idx, val, type_ctx, local_tys, tmp_counter)?;
        }
        _ => {
            writeln!(out, "  ; unhandled hip void op: {:?}", op).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn emit_binary_op(
    out: &mut String,
    instr: &MpirInstr,
    op: &str,
    lhs: &MpirValue,
    rhs: &MpirValue,
    type_ctx: &TypeCtx,
    local_tys: &HashMap<u32, TypeId>,
    tmp_counter: &mut u32,
) -> Result<(), String> {
    let ty = llvm_type_for_hip(type_ctx, instr.ty);
    let lhs_op = operand_for_value(lhs, type_ctx, local_tys)?;
    let rhs_op = operand_for_value(rhs, type_ctx, local_tys)?;
    let lhs_repr = coerce_operand(out, &lhs_op, &ty, instr.ty, type_ctx, tmp_counter)?;
    let rhs_repr = coerce_operand(out, &rhs_op, &ty, instr.ty, type_ctx, tmp_counter)?;
    writeln!(
        out,
        "  %_l{} = {op} {ty} {lhs_repr}, {rhs_repr}",
        instr.dst.0
    )
    .map_err(|e| e.to_string())
}

fn emit_hip_workgroup_size_i32(
    out: &mut String,
    dim: u8,
    tmp_counter: &mut u32,
) -> Result<String, String> {
    let offset = match dim {
        0 => 4_u32,
        1 => 6_u32,
        2 => 8_u32,
        _ => return Err(format!("invalid GPU dimension {dim}, expected 0..=2")),
    };
    let dispatch = next_tmp(tmp_counter);
    let size_ptr = next_tmp(tmp_counter);
    let size_i16 = next_tmp(tmp_counter);
    let size_i32 = next_tmp(tmp_counter);

    writeln!(
        out,
        "  {dispatch} = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()"
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        out,
        "  {size_ptr} = getelementptr inbounds i8, ptr addrspace(4) {dispatch}, i32 {offset}"
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        out,
        "  {size_i16} = load i16, ptr addrspace(4) {size_ptr}, align 2"
    )
    .map_err(|e| e.to_string())?;
    writeln!(out, "  {size_i32} = zext i16 {size_i16} to i32").map_err(|e| e.to_string())?;

    Ok(size_i32)
}

fn emit_gpu_buffer_load(
    out: &mut String,
    instr: &MpirInstr,
    buf: &MpirValue,
    idx: &MpirValue,
    type_ctx: &TypeCtx,
    local_tys: &HashMap<u32, TypeId>,
    tmp_counter: &mut u32,
) -> Result<(), String> {
    let elem_ty = llvm_type_for_hip(type_ctx, instr.ty);
    if elem_ty == "void" {
        return Ok(());
    }
    let elem_ptr_ty = format!("{elem_ty} addrspace(1)*");

    let buf_op = operand_for_value(buf, type_ctx, local_tys)?;
    let idx_op = operand_for_value(idx, type_ctx, local_tys)?;
    let buf_repr = coerce_operand(
        out,
        &buf_op,
        &elem_ptr_ty,
        fixed_type_ids::GPU_BUFFER_BASE,
        type_ctx,
        tmp_counter,
    )?;
    let idx_repr = coerce_operand(
        out,
        &idx_op,
        "i32",
        fixed_type_ids::U32,
        type_ctx,
        tmp_counter,
    )?;

    let gep = next_tmp(tmp_counter);
    writeln!(
        out,
        "  {gep} = getelementptr inbounds {elem_ty}, {elem_ptr_ty} {buf_repr}, i32 {idx_repr}"
    )
    .map_err(|e| e.to_string())?;

    let loaded = next_tmp(tmp_counter);
    writeln!(out, "  {loaded} = load {elem_ty}, {elem_ptr_ty} {gep}").map_err(|e| e.to_string())?;
    let op = Operand {
        ty: elem_ty,
        ty_id: instr.ty,
        repr: loaded,
    };
    emit_assign_from_operand(out, instr.dst.0, instr.ty, &op, type_ctx, tmp_counter)
}

fn emit_gpu_buffer_store(
    out: &mut String,
    buf: &MpirValue,
    idx: &MpirValue,
    val: &MpirValue,
    type_ctx: &TypeCtx,
    local_tys: &HashMap<u32, TypeId>,
    tmp_counter: &mut u32,
) -> Result<(), String> {
    let val_op = operand_for_value(val, type_ctx, local_tys)?;
    if val_op.ty == "void" {
        return Ok(());
    }

    let elem_ptr_ty = format!("{} addrspace(1)*", val_op.ty);
    let buf_op = operand_for_value(buf, type_ctx, local_tys)?;
    let idx_op = operand_for_value(idx, type_ctx, local_tys)?;
    let buf_repr = coerce_operand(
        out,
        &buf_op,
        &elem_ptr_ty,
        fixed_type_ids::GPU_BUFFER_BASE,
        type_ctx,
        tmp_counter,
    )?;
    let idx_repr = coerce_operand(
        out,
        &idx_op,
        "i32",
        fixed_type_ids::U32,
        type_ctx,
        tmp_counter,
    )?;
    let val_repr = coerce_operand(
        out,
        &val_op,
        &val_op.ty,
        val_op.ty_id,
        type_ctx,
        tmp_counter,
    )?;

    let gep = next_tmp(tmp_counter);
    writeln!(
        out,
        "  {gep} = getelementptr inbounds {}, {} {buf_repr}, i32 {idx_repr}",
        val_op.ty, elem_ptr_ty
    )
    .map_err(|e| e.to_string())?;
    writeln!(
        out,
        "  store {} {}, {} {gep}",
        val_op.ty, val_repr, elem_ptr_ty
    )
    .map_err(|e| e.to_string())
}

fn emit_gpu_shared(
    out: &mut String,
    instr: &MpirInstr,
    shared: &BTreeMap<u32, SharedDecl>,
    type_ctx: &TypeCtx,
    tmp_counter: &mut u32,
) -> Result<(), String> {
    let Some(decl) = shared.get(&instr.dst.0) else {
        return emit_zero_assign(out, instr.dst.0, instr.ty, type_ctx, tmp_counter);
    };

    let arr_ty = format!("[{} x {}]", decl.count, decl.elem_ty);
    let ptr_ty = format!("{} addrspace(3)*", decl.elem_ty);
    let gep = next_tmp(tmp_counter);
    writeln!(
        out,
        "  {gep} = getelementptr inbounds {arr_ty}, {arr_ty} addrspace(3)* @{}, i32 0, i32 0",
        decl.symbol
    )
    .map_err(|e| e.to_string())?;

    let src = Operand {
        ty: ptr_ty,
        ty_id: instr.ty,
        repr: gep,
    };
    emit_assign_from_operand(out, instr.dst.0, instr.ty, &src, type_ctx, tmp_counter)
}

fn emit_assign_from_operand(
    out: &mut String,
    dst_local: u32,
    dst_ty_id: TypeId,
    src: &Operand,
    type_ctx: &TypeCtx,
    tmp_counter: &mut u32,
) -> Result<(), String> {
    let dst_ty = llvm_type_for_hip(type_ctx, dst_ty_id);
    let dst_name = format!("%_l{dst_local}");
    emit_cast_assign(
        out,
        &dst_name,
        &dst_ty,
        dst_ty_id,
        src,
        type_ctx,
        tmp_counter,
    )
}

fn emit_zero_assign(
    out: &mut String,
    dst_local: u32,
    dst_ty_id: TypeId,
    type_ctx: &TypeCtx,
    tmp_counter: &mut u32,
) -> Result<(), String> {
    let dst_ty = llvm_type_for_hip(type_ctx, dst_ty_id);
    let z = zero_lit(&dst_ty);
    let src = Operand {
        ty: dst_ty.clone(),
        ty_id: dst_ty_id,
        repr: z,
    };
    emit_assign_from_operand(out, dst_local, dst_ty_id, &src, type_ctx, tmp_counter)
}

fn emit_cast_assign(
    out: &mut String,
    dst_name: &str,
    dst_ty: &str,
    dst_ty_id: TypeId,
    src: &Operand,
    type_ctx: &TypeCtx,
    _tmp_counter: &mut u32,
) -> Result<(), String> {
    if dst_ty == "void" {
        return Ok(());
    }

    if src.ty == dst_ty {
        return emit_identity_assign(out, dst_name, dst_ty, &src.repr);
    }

    if is_int_ty(&src.ty) && is_int_ty(dst_ty) {
        let src_bits = int_bits(&src.ty).unwrap_or(32);
        let dst_bits = int_bits(dst_ty).unwrap_or(32);
        if src_bits == dst_bits {
            return emit_identity_assign(out, dst_name, dst_ty, &src.repr);
        }
        let op = if src_bits < dst_bits {
            if is_signed_int(type_ctx, src.ty_id) {
                "sext"
            } else {
                "zext"
            }
        } else {
            "trunc"
        };
        writeln!(
            out,
            "  {dst_name} = {op} {} {} to {dst_ty}",
            src.ty, src.repr
        )
        .map_err(|e| e.to_string())?;
        return Ok(());
    }

    if is_float_ty(&src.ty) && is_float_ty(dst_ty) {
        let src_bits = float_bits(&src.ty).unwrap_or(32);
        let dst_bits = float_bits(dst_ty).unwrap_or(32);
        if src_bits == dst_bits {
            return emit_identity_assign(out, dst_name, dst_ty, &src.repr);
        }
        let op = if src_bits < dst_bits {
            "fpext"
        } else {
            "fptrunc"
        };
        writeln!(
            out,
            "  {dst_name} = {op} {} {} to {dst_ty}",
            src.ty, src.repr
        )
        .map_err(|e| e.to_string())?;
        return Ok(());
    }

    if is_int_ty(&src.ty) && is_float_ty(dst_ty) {
        let op = if is_signed_int(type_ctx, src.ty_id) {
            "sitofp"
        } else {
            "uitofp"
        };
        writeln!(
            out,
            "  {dst_name} = {op} {} {} to {dst_ty}",
            src.ty, src.repr
        )
        .map_err(|e| e.to_string())?;
        return Ok(());
    }

    if is_float_ty(&src.ty) && is_int_ty(dst_ty) {
        let op = if is_signed_int(type_ctx, dst_ty_id) {
            "fptosi"
        } else {
            "fptoui"
        };
        writeln!(
            out,
            "  {dst_name} = {op} {} {} to {dst_ty}",
            src.ty, src.repr
        )
        .map_err(|e| e.to_string())?;
        return Ok(());
    }

    if is_ptr_ty(&src.ty) && is_int_ty(dst_ty) {
        writeln!(
            out,
            "  {dst_name} = ptrtoint {} {} to {dst_ty}",
            src.ty, src.repr
        )
        .map_err(|e| e.to_string())?;
        return Ok(());
    }

    if is_int_ty(&src.ty) && is_ptr_ty(dst_ty) {
        writeln!(
            out,
            "  {dst_name} = inttoptr {} {} to {dst_ty}",
            src.ty, src.repr
        )
        .map_err(|e| e.to_string())?;
        return Ok(());
    }

    if is_ptr_ty(&src.ty) && is_ptr_ty(dst_ty) {
        let op = if addrspace_of_ptr(&src.ty) != addrspace_of_ptr(dst_ty) {
            "addrspacecast"
        } else {
            "bitcast"
        };
        writeln!(
            out,
            "  {dst_name} = {op} {} {} to {dst_ty}",
            src.ty, src.repr
        )
        .map_err(|e| e.to_string())?;
        return Ok(());
    }

    writeln!(
        out,
        "  {dst_name} = bitcast {} {} to {dst_ty}",
        src.ty, src.repr
    )
    .map_err(|e| e.to_string())
}

fn emit_identity_assign(
    out: &mut String,
    dst_name: &str,
    ty: &str,
    src_repr: &str,
) -> Result<(), String> {
    if is_int_ty(ty) {
        writeln!(out, "  {dst_name} = add {ty} {src_repr}, 0").map_err(|e| e.to_string())
    } else if is_float_ty(ty) {
        writeln!(out, "  {dst_name} = fadd {ty} {src_repr}, 0.0").map_err(|e| e.to_string())
    } else {
        writeln!(
            out,
            "  {dst_name} = select i1 true, {ty} {src_repr}, {ty} {src_repr}"
        )
        .map_err(|e| e.to_string())
    }
}

fn coerce_operand(
    out: &mut String,
    op: &Operand,
    want_ty: &str,
    want_ty_id: TypeId,
    type_ctx: &TypeCtx,
    tmp_counter: &mut u32,
) -> Result<String, String> {
    if op.ty == want_ty {
        return Ok(op.repr.clone());
    }
    let tmp = next_tmp(tmp_counter);
    emit_cast_assign(out, &tmp, want_ty, want_ty_id, op, type_ctx, tmp_counter)?;
    Ok(tmp)
}

fn operand_for_value(
    v: &MpirValue,
    type_ctx: &TypeCtx,
    local_tys: &HashMap<u32, TypeId>,
) -> Result<Operand, String> {
    match v {
        MpirValue::Local(local) => {
            let ty_id = *local_tys
                .get(&local.0)
                .ok_or_else(|| format!("missing local type for %{}", local.0))?;
            Ok(Operand {
                ty: llvm_type_for_hip(type_ctx, ty_id),
                ty_id,
                repr: format!("%_l{}", local.0),
            })
        }
        MpirValue::Const(c) => Ok(const_operand(c, type_ctx)),
    }
}

fn const_operand(c: &HirConst, type_ctx: &TypeCtx) -> Operand {
    let ty = llvm_type_for_hip(type_ctx, c.ty);
    Operand {
        ty: ty.clone(),
        ty_id: c.ty,
        repr: const_lit(c, &ty),
    }
}

fn const_lit(c: &HirConst, llvm_ty: &str) -> String {
    match &c.lit {
        HirConstLit::IntLit(v) => v.to_string(),
        HirConstLit::FloatLit(v) => float_lit(*v),
        HirConstLit::BoolLit(v) => {
            if *v {
                "1".to_string()
            } else {
                "0".to_string()
            }
        }
        HirConstLit::StringLit(_) => zero_lit(llvm_ty),
        HirConstLit::Unit => zero_lit(llvm_ty),
    }
}

fn collect_local_types(kernel: &MpirFn) -> HashMap<u32, TypeId> {
    let mut map = HashMap::new();
    for (id, ty) in &kernel.params {
        map.insert(id.0, *ty);
    }
    for local in &kernel.locals {
        map.insert(local.id.0, local.ty);
    }
    for block in &kernel.blocks {
        for instr in &block.instrs {
            map.insert(instr.dst.0, instr.ty);
        }
    }
    map
}

fn collect_shared_decls(kernel: &MpirFn, type_ctx: &TypeCtx) -> BTreeMap<u32, SharedDecl> {
    let mut out = BTreeMap::new();
    for block in &kernel.blocks {
        for instr in &block.instrs {
            if let MpirOp::GpuShared { ty, size } = &instr.op {
                let count = shared_size_value(size);
                out.insert(
                    instr.dst.0,
                    SharedDecl {
                        symbol: format!("mp_gpu_shared_l{}", instr.dst.0),
                        elem_ty: llvm_type_for_hip(type_ctx, *ty),
                        count,
                    },
                );
            }
        }
    }
    out
}

fn shared_size_value(v: &MpirValue) -> u64 {
    match v {
        MpirValue::Const(c) => match c.lit {
            HirConstLit::IntLit(i) if i > 0 => i as u64,
            HirConstLit::BoolLit(true) => 1,
            _ => 1,
        },
        _ => 1,
    }
}

fn llvm_type_for_hip(type_ctx: &TypeCtx, ty_id: TypeId) -> String {
    if ty_id == fixed_type_ids::GPU_BUFFER_BASE {
        return "float addrspace(1)*".to_string();
    }
    match type_ctx.lookup(ty_id) {
        Some(TypeKind::Prim(PrimType::I1 | PrimType::U1 | PrimType::Bool)) => "i1".to_string(),
        Some(TypeKind::Prim(PrimType::I8 | PrimType::U8)) => "i8".to_string(),
        Some(TypeKind::Prim(PrimType::I16 | PrimType::U16)) => "i16".to_string(),
        Some(TypeKind::Prim(PrimType::I32 | PrimType::U32)) => "i32".to_string(),
        Some(TypeKind::Prim(PrimType::I64 | PrimType::U64)) => "i64".to_string(),
        Some(TypeKind::Prim(PrimType::I128 | PrimType::U128)) => "i128".to_string(),
        Some(TypeKind::Prim(PrimType::F16)) => "half".to_string(),
        Some(TypeKind::Prim(PrimType::Bf16)) => "bfloat".to_string(),
        Some(TypeKind::Prim(PrimType::F32)) => "float".to_string(),
        Some(TypeKind::Prim(PrimType::F64)) => "double".to_string(),
        Some(TypeKind::Prim(PrimType::Unit)) => "void".to_string(),
        Some(TypeKind::RawPtr { .. }) | Some(TypeKind::HeapHandle { .. }) => "ptr".to_string(),
        _ => "i32".to_string(),
    }
}

fn is_signed_int(type_ctx: &TypeCtx, ty_id: TypeId) -> bool {
    match type_ctx.lookup(ty_id) {
        Some(TypeKind::Prim(
            PrimType::I1
            | PrimType::I8
            | PrimType::I16
            | PrimType::I32
            | PrimType::I64
            | PrimType::I128
            | PrimType::Bool,
        )) => true,
        Some(TypeKind::Prim(
            PrimType::U1
            | PrimType::U8
            | PrimType::U16
            | PrimType::U32
            | PrimType::U64
            | PrimType::U128,
        )) => false,
        _ => true,
    }
}

fn dim_suffix(dim: u8) -> Result<&'static str, String> {
    match dim {
        0 => Ok("x"),
        1 => Ok("y"),
        2 => Ok("z"),
        _ => Err(format!("invalid GPU dimension {dim}, expected 0..=2")),
    }
}

fn next_tmp(tmp_counter: &mut u32) -> String {
    let out = format!("%_t{}", *tmp_counter);
    *tmp_counter += 1;
    out
}

fn normalize_icmp_pred(pred: &str) -> &str {
    match pred {
        "eq" | "ne" | "ugt" | "uge" | "ult" | "ule" | "sgt" | "sge" | "slt" | "sle" => pred,
        "gt" => "sgt",
        "ge" => "sge",
        "lt" => "slt",
        "le" => "sle",
        _ => "eq",
    }
}

fn normalize_fcmp_pred(pred: &str) -> &str {
    match pred {
        "false" | "oeq" | "ogt" | "oge" | "olt" | "ole" | "one" | "ord" | "uno" | "ueq" | "ugt"
        | "uge" | "ult" | "ule" | "une" | "true" => pred,
        "eq" => "oeq",
        "ne" => "one",
        "gt" => "ogt",
        "ge" => "oge",
        "lt" => "olt",
        "le" => "ole",
        _ => "oeq",
    }
}

fn int_bits(ty: &str) -> Option<u32> {
    ty.strip_prefix('i')?.parse::<u32>().ok()
}

fn float_bits(ty: &str) -> Option<u32> {
    match ty {
        "half" | "bfloat" => Some(16),
        "float" => Some(32),
        "double" => Some(64),
        _ => None,
    }
}

fn is_int_ty(ty: &str) -> bool {
    int_bits(ty).is_some()
}

fn is_float_ty(ty: &str) -> bool {
    matches!(ty, "half" | "bfloat" | "float" | "double")
}

fn is_ptr_ty(ty: &str) -> bool {
    ty == "ptr" || ty.starts_with("ptr addrspace(") || ty.contains('*')
}

fn addrspace_of_ptr(ty: &str) -> Option<u32> {
    if ty == "ptr" || ty.ends_with('*') {
        if let Some(start) = ty.find("addrspace(") {
            let rest = &ty[start + "addrspace(".len()..];
            let end = rest.find(')')?;
            return rest[..end].parse::<u32>().ok();
        }
        return Some(0);
    }
    if let Some(rest) = ty.strip_prefix("ptr addrspace(") {
        let end = rest.find(')')?;
        return rest[..end].parse::<u32>().ok();
    }
    None
}

fn zero_lit(ty: &str) -> String {
    if ty == "void" {
        "0".to_string()
    } else if is_int_ty(ty) {
        "0".to_string()
    } else if is_float_ty(ty) {
        "0.0".to_string()
    } else if is_ptr_ty(ty) {
        "null".to_string()
    } else {
        "zeroinitializer".to_string()
    }
}

fn float_lit(v: f64) -> String {
    if v.is_nan() {
        "0x7ff8000000000000".to_string()
    } else if v.is_infinite() {
        if v.is_sign_negative() {
            "-0x7ff0000000000000".to_string()
        } else {
            "0x7ff0000000000000".to_string()
        }
    } else {
        let mut s = format!("{v}");
        if !s.contains('.') && !s.contains('e') && !s.contains('E') {
            s.push_str(".0");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hip_emitter_creates() {
        let emitter = HipEmitter::new();
        assert_eq!(emitter.artifact_extension(), "hsaco");
    }
}
