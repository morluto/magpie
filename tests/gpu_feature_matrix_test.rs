use magpie_diag::Severity;
use magpie_driver::{build, BuildProfile, BuildResult, DriverConfig};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn nonce() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before unix epoch")
        .as_nanos()
}

fn unique_target(label: &str) -> String {
    format!("gpu-it-{label}-{}-{}", std::process::id(), nonce())
}

fn write_temp_source(label: &str, source: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "magpie_gpu_test_{label}_{}_{}",
        std::process::id(),
        nonce()
    ));
    fs::create_dir_all(&root).expect("temp source dir should be created");
    let entry = root.join("main.mp");
    fs::write(&entry, source).expect("temp source should be written");
    entry
}

fn build_source(entry: &Path, emit: &[&str], target_triple: String) -> (BuildResult, String) {
    let config = DriverConfig {
        entry_path: entry.to_string_lossy().to_string(),
        profile: BuildProfile::Dev,
        target_triple: target_triple.clone(),
        emit: emit.iter().map(|kind| (*kind).to_string()).collect(),
        ..DriverConfig::default()
    };
    (build(&config), target_triple)
}

fn has_diag_code(result: &BuildResult, code: &str) -> bool {
    result.diagnostics.iter().any(|diag| diag.code == code)
}

fn diag_dump(result: &BuildResult) -> Vec<String> {
    result
        .diagnostics
        .iter()
        .map(|diag| format!("{} {:?}: {}", diag.code, diag.severity, diag.message))
        .collect()
}

fn artifact_with_suffix(result: &BuildResult, suffix: &str) -> PathBuf {
    let artifact = result
        .artifacts
        .iter()
        .find(|path| path.ends_with(suffix))
        .unwrap_or_else(|| {
            panic!(
                "missing artifact with suffix {suffix}; got {:?}",
                result.artifacts
            )
        });
    PathBuf::from(artifact)
}

fn gpu_registry_path(entry: &Path, target_triple: &str) -> PathBuf {
    let stem = entry.file_stem().and_then(|s| s.to_str()).unwrap_or("main");
    Path::new("target")
        .join(target_triple)
        .join("dev")
        .join(format!("{stem}.gpu_registry.ll"))
}

fn cleanup(entry: &Path, target_triple: &str) {
    let _ = fs::remove_dir_all(entry.parent().expect("entry should have parent"));
    let _ = fs::remove_dir_all(Path::new("target").join(target_triple));
}

#[test]
fn gpu_emit_matrix_spv_msl_wgsl_and_registry_shape() {
    let entry = write_temp_source(
        "emit_matrix",
        r#"module gpu.emit_matrix
exports { @main }
imports { }
digest "0000000000000000"

gpu fn @k_spv() -> unit target(spv) {
bb0:
  ret
}

gpu fn @k_msl() -> unit target(msl) {
bb0:
  ret
}

gpu fn @k_wgsl() -> unit target(wgsl) {
bb0:
  ret
}

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
    );

    let target = unique_target("emit-matrix");
    let (result, target_triple) = build_source(&entry, &["spv", "msl", "wgsl"], target);

    assert!(
        result.success,
        "gpu emit matrix should succeed, diagnostics: {:?}",
        diag_dump(&result)
    );

    let spv = artifact_with_suffix(&result, ".spv");
    let msl = artifact_with_suffix(&result, ".metal");
    let wgsl = artifact_with_suffix(&result, ".wgsl");
    let registry = gpu_registry_path(&entry, &target_triple);

    for path in [&spv, &msl, &wgsl, &registry] {
        assert!(path.is_file(), "expected artifact at {}", path.display());
    }

    let registry_ir = fs::read_to_string(&registry).expect("registry IR should be readable");
    assert!(registry_ir.contains("%MpRtGpuKernelBlob = type"));
    assert!(registry_ir.contains("%MpRtGpuKernelEntry = type"));
    assert!(registry_ir.contains("@mp_gpu_kernel_registry"));
    assert!(registry_ir.contains("@mp_gpu_register_all_kernels"));

    cleanup(&entry, &target_triple);
}

#[test]
fn gpu_unsafe_requires_diagnostics() {
    let entry = write_temp_source(
        "unsafe_requires",
        r#"module gpu.unsafe_requires
exports { @main }
imports { }
digest "0000000000000000"

unsafe gpu fn @k_missing_requires() -> unit target(msl) {
bb0:
  ret
}

unsafe gpu fn @k_bad_capability() -> unit target(msl) requires(cuda.tensor) {
bb0:
  ret
}

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
    );

    let target = unique_target("unsafe-req");
    let (result, target_triple) = build_source(&entry, &["msl"], target);

    assert!(!result.success, "unsafe diagnostics test should fail build");
    assert!(
        has_diag_code(&result, "MPG_CORE_1201"),
        "expected MPG_CORE_1201, diagnostics: {:?}",
        diag_dump(&result)
    );
    assert!(
        has_diag_code(&result, "MPG_CORE_1200"),
        "expected MPG_CORE_1200, diagnostics: {:?}",
        diag_dump(&result)
    );

    let unsafe_warn_count = result
        .diagnostics
        .iter()
        .filter(|diag| diag.code == "MPG_CORE_1202" && matches!(diag.severity, Severity::Warning))
        .count();
    assert!(
        unsafe_warn_count >= 2,
        "expected unsafe warning MPG_CORE_1202 for both kernels, diagnostics: {:?}",
        diag_dump(&result)
    );

    cleanup(&entry, &target_triple);
}

#[test]
fn gpu_invalid_intrinsic_dim_reports_mps0001() {
    let entry = write_temp_source(
        "bad_dim",
        r#"module gpu.bad_dim
exports { @main }
imports { }
digest "0000000000000000"

gpu fn @k() -> unit target(spv) {
bb0:
  %gid: u32 = gpu.global_id { dim=const.u32 7 }
  ret
}

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
    );

    let target = unique_target("bad-dim");
    let (result, target_triple) = build_source(&entry, &["spv"], target);

    assert!(!result.success, "bad dim kernel should fail build");
    assert!(
        has_diag_code(&result, "MPS0001"),
        "expected MPS0001, diagnostics: {:?}",
        diag_dump(&result)
    );

    cleanup(&entry, &target_triple);
}

#[test]
fn gpu_core_restriction_diagnostics_cover_110x() {
    let entry = write_temp_source(
        "core_110x",
        r#"module gpu.restrictions
exports { @main }
imports { }
digest "0000000000000000"

heap struct TBox {
  field x: i32
}

sig TNoArgs() -> i32

gpu fn @k_alloc() -> unit target(spv) {
bb0:
  %p: TBox = new TBox { x=const.i32 1 }
  ret
}

gpu fn @k_arc(%buf: gpu.TBuffer<i32>) -> unit target(spv) {
bb0:
  ret
}

gpu fn @k_dyn(%cb: TCallable<TNoArgs>) -> unit target(spv) {
bb0:
  %v: i32 = call.indirect %cb { }
  ret
}

gpu fn @k_recur() -> unit target(spv) {
bb0:
  call_void @k_recur { }
  ret
}

gpu fn @k_str(%s: Str) -> unit target(spv) {
bb0:
  ret
}

gpu fn @k_arr(%a: Array<i32>) -> unit target(spv) {
bb0:
  ret
}

gpu fn @k_map(%m: Map<i32, i32>) -> unit target(spv) {
bb0:
  ret
}

gpu fn @k_callable(%cb: TCallable<TNoArgs>) -> unit target(spv) {
bb0:
  ret
}

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
    );

    let target = unique_target("core-110x");
    let (result, target_triple) = build_source(&entry, &["spv"], target);

    assert!(!result.success, "restriction fixture should fail build");
    for code in [
        "MPG_CORE_1100",
        "MPG_CORE_1101",
        "MPG_CORE_1102",
        "MPG_CORE_1103",
        "MPG_CORE_1104",
        "MPG_CORE_1105",
        "MPG_CORE_1106",
        "MPG_CORE_1107",
    ] {
        assert!(
            has_diag_code(&result, code),
            "missing {code}; diagnostics: {:?}",
            diag_dump(&result)
        );
    }

    cleanup(&entry, &target_triple);
}

#[test]
fn gpu_bf16_kernel_emits_msl_scalar_as_bfloat() {
    let entry = write_temp_source(
        "bf16_msl",
        r#"module gpu.bf16
exports { @main }
imports { }
digest "0000000000000000"

gpu fn @k_bf16(%scale: bf16) -> unit target(msl) {
bb0:
  ret
}

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
    );

    let target = unique_target("bf16-msl");
    let (result, target_triple) = build_source(&entry, &["msl"], target);

    assert!(
        result.success,
        "bf16 msl kernel should build, diagnostics: {:?}",
        diag_dump(&result)
    );

    let metal = artifact_with_suffix(&result, ".metal");
    let source = fs::read_to_string(&metal).expect("msl should be readable");
    assert!(
        source.contains("bfloat"),
        "expected bfloat scalar in emitted msl, got:\n{}",
        source
    );

    cleanup(&entry, &target_triple);
}

#[test]
fn gpu_ptx_and_hip_emit_or_report_missing_toolchain() {
    let entry = write_temp_source(
        "ptx_hip_tools",
        r#"module gpu.ptx_hip
exports { @main }
imports { }
digest "0000000000000000"

gpu fn @k_ptx() -> unit target(ptx) {
bb0:
  ret
}

gpu fn @k_hip() -> unit target(hip) {
bb0:
  ret
}

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
    );

    let target = unique_target("ptx-hip");
    let (result, target_triple) = build_source(&entry, &["ptx", "hip"], target);

    if result.success {
        assert!(
            result.artifacts.iter().any(|a| a.ends_with(".ptx")),
            "expected .ptx artifact, got {:?}",
            result.artifacts
        );
        assert!(
            result.artifacts.iter().any(|a| a.ends_with(".hip")),
            "expected .hip artifact, got {:?}",
            result.artifacts
        );
    } else {
        assert!(
            has_diag_code(&result, "MPG_CORE_1301"),
            "expected MPG_CORE_1301 when toolchain is missing, diagnostics: {:?}",
            diag_dump(&result)
        );
    }

    cleanup(&entry, &target_triple);
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
fn gpu_metal_executes_emitted_kernel_end_to_end() {
    let entry = write_temp_source(
        "metal_exec",
        r#"module gpu.metal_exec
exports { @main }
imports { }
digest "0000000000000000"

gpu fn @k(%a: rawptr<f32>, %b: rawptr<f32>, %out: rawptr<f32>) -> unit target(msl) workgroup(64, 1, 1) {
bb0:
  %gid: u32 = gpu.global_id { dim=const.u32 0 }
  %x: f32 = gpu.buffer_load<f32> { buf=%a, idx=%gid }
  %y: f32 = gpu.buffer_load<f32> { buf=%b, idx=%gid }
  %z: f32 = f.add { lhs=%x, rhs=%y }
  gpu.buffer_store<f32> { buf=%out, idx=%gid, v=%z }
  ret
}

fn @main() -> i32 {
bb0:
  ret const.i32 0
}
"#,
    );

    let target = unique_target("metal-exec");
    let (result, target_triple) = build_source(&entry, &["msl"], target);
    assert!(
        result.success,
        "metal kernel build should succeed, diagnostics: {:?}",
        diag_dump(&result)
    );

    let metal_src = artifact_with_suffix(&result, ".metal");
    let host_src = entry.parent().expect("entry parent").join("metal_runner.m");
    let host_bin = entry.parent().expect("entry parent").join("metal_runner");

    fs::write(
        &host_src,
        r#"#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <math.h>
#include <stdio.h>

static int failf(const char* msg) {
  fprintf(stderr, "%s\n", msg);
  return 1;
}

int main(int argc, const char** argv) {
  @autoreleasepool {
    if (argc < 2) {
      return failf("usage: metal_runner <kernel.metal>");
    }

    NSString* source_path = [NSString stringWithUTF8String:argv[1]];
    NSError* err = nil;
    NSString* source = [NSString stringWithContentsOfFile:source_path
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
    if (!source) {
      fprintf(stderr, "failed reading source: %s\n", [[err localizedDescription] UTF8String]);
      return 2;
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      return failf("no metal device");
    }

    id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&err];
    if (!lib) {
      fprintf(stderr, "failed compiling msl: %s\n", [[err localizedDescription] UTF8String]);
      return 3;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"k"];
    if (!fn) {
      return failf("kernel function 'k' missing");
    }

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
      fprintf(stderr, "failed creating pipeline: %s\n", [[err localizedDescription] UTF8String]);
      return 4;
    }

    const NSUInteger n = 256;
    const NSUInteger bytes = n * sizeof(float);

    id<MTLBuffer> a = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!a || !b || !out) {
      return failf("failed creating buffers");
    }

    float* pa = (float*)a.contents;
    float* pb = (float*)b.contents;
    float* po = (float*)out.contents;
    for (NSUInteger i = 0; i < n; ++i) {
      pa[i] = (float)i;
      pb[i] = (float)(i * 2);
      po[i] = 0.0f;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
      return failf("failed creating command queue");
    }

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:a offset:0 atIndex:0];
    [enc setBuffer:b offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];

    NSUInteger tptg = pso.maxTotalThreadsPerThreadgroup;
    if (tptg > 64) {
      tptg = 64;
    }
    if (tptg == 0) {
      tptg = 1;
    }

    MTLSize grid = MTLSizeMake(n, 1, 1);
    MTLSize tg = MTLSizeMake(tptg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      return failf("command buffer did not complete");
    }

    for (NSUInteger i = 0; i < n; ++i) {
      float expected = pa[i] + pb[i];
      if (fabsf(po[i] - expected) > 1e-5f) {
        fprintf(stderr, "mismatch at %lu: got=%f expected=%f\n", (unsigned long)i, po[i], expected);
        return 5;
      }
    }
  }

  return 0;
}
"#,
    )
    .expect("host source should be written");

    let compile = Command::new("xcrun")
        .arg("clang")
        .arg("-fobjc-arc")
        .arg("-framework")
        .arg("Foundation")
        .arg("-framework")
        .arg("Metal")
        .arg(&host_src)
        .arg("-o")
        .arg(&host_bin)
        .output()
        .expect("xcrun clang should run");

    assert!(
        compile.status.success(),
        "failed compiling metal host:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&compile.stdout),
        String::from_utf8_lossy(&compile.stderr)
    );

    let run = Command::new(&host_bin)
        .arg(&metal_src)
        .output()
        .expect("metal host binary should run");

    assert!(
        run.status.success(),
        "metal execution failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    cleanup(&entry, &target_triple);
}
