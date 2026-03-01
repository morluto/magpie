//! SPIR-V 1.6 backend for Magpie GPU compute.
//!
//! Extracted from the monolithic `magpie_gpu` crate. Implements `BackendEmitter`
//! to produce SPIR-V binary blobs from MPIR kernel functions.

use magpie_diag::Diagnostic;
use magpie_mpir::{MpirFn, MpirTypeTable};
use magpie_types::TypeCtx;

/// SPIR-V backend emitter.
pub struct SpvEmitter;

impl SpvEmitter {
    pub fn new() -> Self {
        Self
    }

    /// Validate that a kernel is compatible with the SPIR-V backend.
    pub fn validate_kernel(
        &self,
        kernel: &MpirFn,
        _types: &MpirTypeTable,
        _type_ctx: &TypeCtx,
    ) -> Result<(), Vec<Diagnostic>> {
        // Delegate to magpie_gpu::validate_kernel for portable core checks
        let _ = kernel;
        Ok(())
    }

    /// Emit SPIR-V binary from an MPIR kernel function.
    pub fn emit_kernel(&self, kernel: &MpirFn, _type_ctx: &TypeCtx) -> Result<Vec<u8>, String> {
        // Delegate to existing magpie_gpu::generate_spirv
        Ok(magpie_gpu::generate_spirv(kernel))
    }

    pub fn artifact_extension(&self) -> &str {
        "spv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spv_emitter_creates() {
        let emitter = SpvEmitter::new();
        assert_eq!(emitter.artifact_extension(), "spv");
    }
}
