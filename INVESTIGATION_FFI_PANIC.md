# Investigation: FFI Panic Risk in Magpie Runtime

## Summary
Confirmed: multiple production `extern "C"` runtime APIs panic on malformed input or invariant violations, and several of those APIs are invoked directly by generated LLVM. This creates process-level fragility at the ABI boundary and diverges from the fallible parse model described in parts of the spec.

## Symptoms
- Concern that malformed inputs at FFI boundaries can trigger `panic!`.
- Concern that panic behavior may abort process instead of returning normal error values.
- Concern that this creates fragility for production embeddings/integrations.

## Investigation Log

### Phase 0 - Workspace Verification
**Hypothesis:** The correct codebase must be loaded in RepoPrompt before deep investigation.
**Findings:** `magpie` workspace was not initially loaded; created and bound workspace/window for `/Users/will/Desktop/magpie`.
**Evidence:** RepoPrompt `list_windows`, `manage_workspaces list/create`, `select_window`.
**Conclusion:** Confirmed; workspace context is valid.

### Phase 2 - Context Builder Attempt
**Hypothesis:** Use `context_builder` as required protocol step.
**Findings:** `context_builder` call failed with API 402 membership verification error; no context selection returned.
**Evidence:** RepoPrompt `context_builder` response: `API Error: 402 ... unable to verify membership benefits`.
**Conclusion:** Required step attempted but blocked by tool-side external failure; proceeding with manual evidence gathering.

### Phase 3 - Runtime FFI Panic Inventory
**Hypothesis:** Panics are only in tests or internal helpers, not in production `extern "C"` APIs.
**Findings:** Production FFI exports contain `expect!`, `panic!`, and `assert!` in non-test code.
**Evidence:**  
- Parse/JSON panic paths: `mp_rt_str_parse_*`, `mp_rt_json_encode`, `mp_rt_json_decode` in [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1357](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1357), [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1425](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1425), [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1455](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1455).  
- UTF-8 helper used by FFI parse/json also panics: [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1353](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1353).  
- Additional assert-driven panic in FFI functions (mutex/future/array bounds/etc.): [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1221](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1221), [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1337](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1337), [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:4178](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:4178).  
**Conclusion:** Confirmed. Panic-capable behavior exists in production ABI paths.

### Phase 3 - Call-Path Reachability
**Hypothesis:** Even if panic-capable functions exist, generated code may not call them.
**Findings:** LLVM codegen declares and emits direct calls to the risky parse/json runtime symbols.
**Evidence:**  
- Declared runtime symbols: [/Users/will/Desktop/magpie/crates/magpie_codegen_llvm/src/lib.rs:189](/Users/will/Desktop/magpie/crates/magpie_codegen_llvm/src/lib.rs:189), [/Users/will/Desktop/magpie/crates/magpie_codegen_llvm/src/lib.rs:191](/Users/will/Desktop/magpie/crates/magpie_codegen_llvm/src/lib.rs:191).  
- Emitted calls from MPIR ops: [/Users/will/Desktop/magpie/crates/magpie_codegen_llvm/src/lib.rs:1434](/Users/will/Desktop/magpie/crates/magpie_codegen_llvm/src/lib.rs:1434), [/Users/will/Desktop/magpie/crates/magpie_codegen_llvm/src/lib.rs:1482](/Users/will/Desktop/magpie/crates/magpie_codegen_llvm/src/lib.rs:1482).  
**Conclusion:** Confirmed. These panic-prone FFI functions are in active compiler-generated execution paths.

### Phase 4 - Error-Semantics Consistency Check
**Hypothesis:** Runtime ABI consistently uses return-code/null/error-out conventions.
**Findings:** Error semantics are mixed. GPU/web APIs commonly return status + out error string, while parse/json APIs hard-fail via panic semantics.
**Evidence:**  
- Status/out-err pattern (example): [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1986](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1986), [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:2036](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:2036), [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:2197](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:2197).  
- Parse/json primitive-return signatures in C header (no error channel): [/Users/will/Desktop/magpie/crates/magpie_rt/include/magpie_rt.h:137](/Users/will/Desktop/magpie/crates/magpie_rt/include/magpie_rt.h:137), [/Users/will/Desktop/magpie/crates/magpie_rt/include/magpie_rt.h:142](/Users/will/Desktop/magpie/crates/magpie_rt/include/magpie_rt.h:142).  
**Conclusion:** Confirmed inconsistency.

### Phase 4 - Spec/Implementation Alignment
**Hypothesis:** Current runtime panic behavior exactly matches current spec intent.
**Findings:** Spec is internally inconsistent with implementation around parse behavior.  
- Spec §34.2 defines `str.parse_*` as fallible `TResult<..., TParseError>`.  
- Runtime ABI + codegen implement primitive-return parse calls that panic on bad input.
**Evidence:**  
- Fallible spec signatures: [/Users/will/Desktop/magpie/SPEC.md:3796](/Users/will/Desktop/magpie/SPEC.md:3796), [/Users/will/Desktop/magpie/SPEC.md:3800](/Users/will/Desktop/magpie/SPEC.md:3800).  
- Web wrapper requirement on parse failure -> 400: [/Users/will/Desktop/magpie/SPEC.md:3177](/Users/will/Desktop/magpie/SPEC.md:3177), [/Users/will/Desktop/magpie/SPEC.md:3179](/Users/will/Desktop/magpie/SPEC.md:3179).  
- Current docs/examples show primitive parse usage: [/Users/will/Desktop/magpie/DOCUMENTATION_QUICKSTART.md:930](/Users/will/Desktop/magpie/DOCUMENTATION_QUICKSTART.md:930).  
- Runtime implementation panics on parse failure: [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1360](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:1360).  
**Conclusion:** Confirmed spec drift / unresolved migration state.

### Phase 4 - Test Coverage and History
**Hypothesis:** Negative-path tests and git history confirm this behavior is defended and intentional.
**Findings:**  
- Runtime tests include happy-path parse test but no invalid-parse safety tests.  
- Local repository history is unavailable (repo has no commits in local `.git` state), so recency/blame cannot be established here.
**Evidence:**  
- Happy-path parse test: [/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:3374](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs:3374).  
- No `#[should_panic]` or invalid parse coverage found for parse/json paths.  
- `git status` reports `No commits yet on master` in this local clone.
**Conclusion:** Coverage gap confirmed; change provenance unknown in this local checkout.

## Eliminated Hypotheses
- “Panics only exist in test code”: eliminated (production FFI functions panic).
- “Generated code does not hit panic-prone FFI”: eliminated (direct codegen calls exist).
- “FFI boundary has unwind containment (`catch_unwind`)”: eliminated (none found in runtime crate).
- “Error handling style is uniform across runtime”: eliminated (mixed panic vs status/out-err contracts).

## Root Cause
Primary root cause is ABI design mismatch: several FFI functions (`mp_rt_str_parse_*`, `mp_rt_json_decode/encode`) expose infallible signatures (primitive/pointer return with no error channel) but internally rely on Rust parsing and `expect!/panic!` for invalid input. Because these are exported `extern "C"` entrypoints invoked by generated LLVM IR, malformed runtime data can trigger process-terminating failure behavior rather than recoverable diagnostics.

Contributing factors:
- No standardized FFI error contract applied across all subsystems (GPU/web are more explicit with status/out error).
- No panic containment layer at ABI edges.
- Spec/docs drift on parse semantics (fallible TResult model vs infallible primitive-return implementation).
- Missing negative-path tests for malformed parse/json inputs at runtime ABI level.

## Recommendations
1. Introduce fallible FFI parse/json APIs with explicit status contract, e.g.:
   - `int32_t mp_rt_str_try_parse_i64(MpRtHeader* s, int64_t* out, MpRtHeader** out_errmsg)`
   - `int32_t mp_rt_json_try_decode(MpRtHeader* json, uint32_t type_id, uint8_t** out, MpRtHeader** out_errmsg)`
2. Update LLVM codegen to lower parse/json intrinsics to the fallible API and branch on status (aligning with `TResult` direction in spec).
3. Keep legacy panicing symbols temporarily as compatibility shims, but deprecate them and gate behind internal-only paths.
4. Normalize runtime error semantics: use `status + out_errmsg` (or equivalent) consistently for host-facing FFI calls.
5. Add runtime negative-path tests for invalid parse/json inputs and null/invalid pointer misuse where non-crashing behavior is expected.
6. Resolve spec drift: pick one model (trap/panic vs `TResult`) and align `SPEC.md`, docs, and implementation together.

## Preventive Measures
- Add CI checks that forbid `panic!/expect!/assert!` in production `extern "C"` bodies unless explicitly annotated/waived.
- Add ABI contract tests asserting stable error behavior (return codes + error strings) for malformed input cases.
- Add a design rule in runtime docs: FFI boundary functions must not panic for recoverable input errors.
- Track spec-implementation parity with a small conformance matrix for core intrinsics (parse/json/collections/web/gpu).

## Further Investigation (Second Pass)

### Expanded Runtime Inventory
- Parallel inventory found **57 production `extern "C"` functions** in panic-capable paths inside [magpie_rt/src/lib.rs](/Users/will/Desktop/magpie/crates/magpie_rt/src/lib.rs) (excluding `#[cfg(test)]` sections).
- Categories include:
  - input-validation panic (`assert!` on null/bounds/state)
  - invariant/overflow panic (`expect` on checked math/layout)
  - intentional abort APIs (`mp_rt_panic`, `mp_std_assert*`, `mp_std_fail`)

### Loss of Fallibility Through Compiler Pipeline
- `str.parse_*` and `json.encode/decode` are modeled as success-only ops from lexer → parser → AST → HIR → MPIR, then lowered into direct LLVM calls with infallible return shapes.
- Runtime ABI signatures for these calls are primitive/pointer returns without status/error out channels in [magpie_rt.h](/Users/will/Desktop/magpie/crates/magpie_rt/include/magpie_rt.h:137).
- Runtime implementation then maps failures to panic/abort behavior.

### Spec / Docs / Runtime Contradictions
- Spec claims fallible parse signatures (`TResult<..., TParseError>`), but implementation and docs currently reflect primitive parse returns.
- Spec requires typed route-parse failures to return 400, but current web dispatch/runtime path does not perform that typed parse/error branching end-to-end.
- Net effect: currently documented behavior mixes “target model” and “implemented model.”

### Tightened Next-Step Plan
1. Freeze one canonical contract for parse/json now (recommended: explicit fallible ABI for host/runtime boundary).
2. Choose migration mode:
   - non-breaking: add new `try` intrinsics/APIs + keep old panicking APIs as temporary wrappers
   - breaking: change existing parse/json signatures to fallible forms directly
3. Update codegen lowering to consume fallible API and branch into value-level error handling.
4. Add ABI negative tests (invalid UTF-8, malformed numbers/JSON, null pointers, unsupported `type_id`, error ownership checks).
5. Align `SPEC.md`, `DOCUMENTATION*.md`, and header/runtime signatures in one change set.
