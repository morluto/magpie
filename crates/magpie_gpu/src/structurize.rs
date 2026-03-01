use magpie_mpir::{MpirFn, MpirInstr, MpirOpVoid, MpirTerminator, MpirValue};
use magpie_types::{BlockId, LocalId};

#[derive(Debug, Clone)]
pub enum StructuredNode {
    Block {
        label: BlockId,
        instrs: Vec<MpirInstr>,
        void_ops: Vec<MpirOpVoid>,
    },
    IfElse {
        cond: MpirValue,
        then_branch: Vec<StructuredNode>,
        else_branch: Vec<StructuredNode>,
    },
    Loop {
        body: Vec<StructuredNode>,
    },
    Break {
        depth: u32,
    },
    Continue {
        depth: u32,
    },
    Return,
    Assign {
        local: LocalId,
        value: MpirValue,
    },
}

#[derive(Debug)]
pub struct StructurizeError {
    pub message: String,
    pub block_id: Option<BlockId>,
}

impl std::fmt::Display for StructurizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StructurizeError: {}", self.message)
    }
}

/// Convert MPIR unstructured CFG to structured control flow for MSL/WGSL backends.
/// Uses a simplified Relooper-style algorithm.
pub fn structurize_cfg(func: &MpirFn) -> Result<Vec<StructuredNode>, StructurizeError> {
    if func.blocks.is_empty() {
        return Ok(vec![StructuredNode::Return]);
    }

    let mut result = Vec::new();
    let mut visited = std::collections::HashSet::new();

    for block in &func.blocks {
        if visited.contains(&block.id) {
            continue;
        }
        visited.insert(block.id);

        // Emit block instructions
        result.push(StructuredNode::Block {
            label: block.id,
            instrs: block.instrs.clone(),
            void_ops: block.void_ops.clone(),
        });

        // Handle terminator
        match &block.terminator {
            MpirTerminator::Ret(_) => {
                result.push(StructuredNode::Return);
            }
            MpirTerminator::Br(_target) => {
                // Simple branch - will be handled when we visit the target block
            }
            MpirTerminator::Cbr {
                cond,
                then_bb,
                else_bb,
            } => {
                // Find the then and else blocks
                let then_block = func.blocks.iter().find(|b| b.id == *then_bb);
                let else_block = func.blocks.iter().find(|b| b.id == *else_bb);

                let mut then_nodes = Vec::new();
                let mut else_nodes = Vec::new();

                if let Some(tb) = then_block {
                    if !visited.contains(&tb.id) {
                        visited.insert(tb.id);
                        then_nodes.push(StructuredNode::Block {
                            label: tb.id,
                            instrs: tb.instrs.clone(),
                            void_ops: tb.void_ops.clone(),
                        });
                        match &tb.terminator {
                            MpirTerminator::Ret(_) => then_nodes.push(StructuredNode::Return),
                            MpirTerminator::Br(_) => {} // fall through
                            _ => {}
                        }
                    }
                }

                if let Some(eb) = else_block {
                    if !visited.contains(&eb.id) {
                        visited.insert(eb.id);
                        else_nodes.push(StructuredNode::Block {
                            label: eb.id,
                            instrs: eb.instrs.clone(),
                            void_ops: eb.void_ops.clone(),
                        });
                        match &eb.terminator {
                            MpirTerminator::Ret(_) => else_nodes.push(StructuredNode::Return),
                            MpirTerminator::Br(_) => {} // fall through
                            _ => {}
                        }
                    }
                }

                result.push(StructuredNode::IfElse {
                    cond: cond.clone(),
                    then_branch: then_nodes,
                    else_branch: else_nodes,
                });
            }
            _ => {}
        }
    }

    Ok(result)
}
