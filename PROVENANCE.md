# File provenance

Source baseline: TRP v0.6.0, `82c33af4eb9ce001afbb6a4be9a7381e42042281`.
Shape/license reference: dlssg-turing, `96ce50f8fe863b439b955ce9456184839129c5e5`.
Research CUDA and headers are Theo's behaviour-compatible reimplementations and
math tables. They contain no NVIDIA code text, PTX or model parameters, but for
bit-exact results they reproduce NVIDIA interface facts (kernel, parameter and
class names; parameter-block and weight-blob offsets) and a few approximation
constants, such as the cubic SiLU coefficients.

Research sources (byte-pinned) and files copied from TRP keep their inherited comments:
- Some name TRP-only files, environment variables, research scripts, notes or paths
  that are not part of this repository (for example `NeuralRenderingFeatureSession.cpp`
  and `TRP_NR_PASCAL_*` in `tools/build_pascal_kernels.py`, `build_rtx40.py` in
  `tools/ada_lowering.py`, the RTX 20/30/40 path in `src/NeuralRuntimeKernels.h`,
  and research notes or scripts such as `swin1h_spec.md` and `bench/layouts.py` in
  research headers).
- The swinmh `r11vectorcombo*` and `r13linearh2v*` sources hard-code the 19 SMs
  of the GTX 1070 Ti test card in their owner pools, despite a nearby comment
  saying no SM-count bound is required. `tools/prepare_native.py` replaces those
  pools with the SM-independent ticket-ring allocator at build time, so the pack
  has no SM-count dependency.

The last column relates each file to TRP v0.6.0: a verbatim copy (same Git blob),
adapted from a TRP file (same file with standalone changes), or new to this
repository. Research sources are new here; their research-lane origin is in
REGENERATION.md and `tools/selection.json`.
Original source line endings are normalized to LF where imported. In particular,
the historical emulation CUDA was verified against its original
`3E50FA25587ACDF82F767AA05B0048DC531CA994C94C7A3592624EA516DE4D04`
identity before normalization.

The following table has one row for every committed file, including this file.
GPL references mean this repository's GPL v3 and its modding/linking exceptions.
The shared host-admission patch-policy facts retain the openNR/ShortFuse MIT
credit and notice in THIRD-PARTY.md. This does not grant rights to NVIDIA material.

| Committed file | Classification / license | Origin / scope | TRP v0.6.0 |
| --- | --- | --- | --- |
| `.gitattributes` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `extern/dlssg-turing/.gitattributes` |
| `.gitignore` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new (layout follows dlssg-turing) |
| `CMakeLists.txt` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new (layout follows dlssg-turing) |
| `LICENSE` | Third-party license text + TRP exceptions | Verbatim dlssg-turing reference license; GPL v3 with modding/linking exceptions | verbatim copy of `LICENSE` |
| `PROVENANCE.md` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `README.md` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new (layout follows dlssg-turing) |
| `REGENERATION.md` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `THIRD-PARTY.md` | TRP-authored notice + third-party MIT | openNR / Carlos Lopez Jr. MIT notice; shared patch-policy facts | new (layout follows dlssg-turing) |
| `src/NeuralAdmission.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | adapted from `src/FrameGen/NeuralAdmission.cpp` |
| `src/NeuralAdmission.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `src/FrameGen/NeuralAdmission.h` |
| `src/NeuralFaultInjection.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `src/FrameGen/NeuralFaultInjection.h` |
| `src/NeuralLaunchRequirements.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `src/FrameGen/NeuralLaunchRequirements.h` |
| `src/NeuralRenderingRuntimeIdentity.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | adapted from `src/FrameGen/NeuralRenderingRuntimeIdentity.cpp` |
| `src/NeuralRenderingRuntimeIdentity.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | adapted from `src/FrameGen/NeuralRenderingRuntimeIdentity.h` |
| `src/NeuralRuntimeKernelPlan.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored host patch facts/validation; openNR MIT patch-policy credit retained | verbatim copy of `src/FrameGen/NeuralRuntimeKernelPlan.cpp` |
| `src/NeuralRuntimeKernelRegistry.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `src/FrameGen/NeuralRuntimeKernelRegistry.h` |
| `src/NeuralRuntimeKernels.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | adapted from `src/FrameGen/NeuralRuntimeKernels.cpp` |
| `src/NeuralRuntimeKernels.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `src/FrameGen/NeuralRuntimeKernels.h` |
| `src/NeuralRuntimeKernelsInternal.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored host patch facts/validation; openNR MIT patch-policy credit retained | verbatim copy of `src/FrameGen/NeuralRuntimeKernelsInternal.h` |
| `src/NvidiaDriverVersion.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `src/FrameGen/NvidiaDriverVersion.h` |
| `src/runtime.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `src/runtime.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `tests/NeuralAdmissionTests.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch | adapted from `tests/NeuralAdmissionTests.cpp` |
| `tests/test_inventory.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch | new |
| `tests/test_launch_bounds.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch | adapted from `tools/nr-runtime/test_launch_bounds.py` |
| `tests/test_launch_metadata.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch | adapted from `tools/nr-runtime/test_launch_metadata.py` |
| `tests/test_sm_general.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch | verbatim copy of `tools/nr-runtime/test_sm_general.py` |
| `tools/ada_lowering.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored lowering/stand-in generator; input/output kernel text excluded | verbatim copy of `tools/nr-runtime/ada_lowering.py` |
| `tools/build_pack.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `tools/build_pascal_kernels.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | adapted from `tools/nr-runtime/build_pascal_kernels.py` |
| `tools/build_rtx40_kernels.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored host patch facts/validation; openNR MIT patch-policy credit retained | verbatim copy of `tools/nr-runtime/build_rtx40_kernels.py` |
| `tools/compiles.json` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `tools/inventory.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `tools/launch-rules.txt` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `tools/launch_bounds.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `tools/nr-runtime/launch_bounds.py` |
| `tools/launch_metadata.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `tools/nr-runtime/launch_metadata.py` |
| `tools/pascal_groups.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored lowering/stand-in generator; input/output kernel text excluded | verbatim copy of `tools/nr-runtime/pascal_groups.py` |
| `tools/pascal_lowering.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored lowering/stand-in generator; input/output kernel text excluded | verbatim copy of `tools/nr-runtime/pascal_lowering.py` |
| `tools/prepare_native.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `tools/prepost-transform.json` | TRP-authored (GPL v3 + exceptions) | TRP-authored numeric references and wrapper insertions; NVIDIA text generated from supplied DLL at build time | new |
| `tools/ptx-lock.json` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `tools/regenerate.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `tools/selection.json` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation | new |
| `tools/sm_general_slots.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored; copied unchanged from TRP | verbatim copy of `tools/nr-runtime/sm_general_slots.cuh` |
| `tools/sources/E26/E26_qkv.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/E26/vitfuse_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/E27/E27_projection.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/E27/vitfuse_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/E29/E29_ffwd.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/E29/vitfuse_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/E29/vitfuse_r5_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | cubic-SiLU FP16→FP8 table computed from the three coefficients for every FP16 input; no model data | new |
| `tools/sources/E36/E36_ffwdproj.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/E36/vitfuse_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/bankpairs/bankpairs_core.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/bankpairs/bankpairs_device.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/bankpairs/postpair_scales.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/bankpairs/prepost_helpers_best.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/bankpairs/prepost_quant_exact.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/bankpairs/prepost_r6_silu.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/bankpairs/prepost_r6_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | cubic-SiLU FP16→FP8 table computed from the three coefficients for every FP16 input; no model data | new |
| `tools/sources/dual/dual_core.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/dual/dual_device.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/dual/dual_pool.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/dual/prepost_helpers_best.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/dual/prepost_quant_exact.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/dual/prepost_r6_silu.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/dual/prepost_r6_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | cubic-SiLU FP16→FP8 table computed from the three coefficients for every FP16 input; no model data | new |
| `tools/sources/emu/pascal_emulation.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | verbatim copy of `tools/nr-runtime/pascal_emulation.cu` |
| `tools/sources/pool/core-r7_pool.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/pool/prepost_core_r6_lut.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/pool/prepost_helpers_best.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/pool/prepost_quant_exact.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/pool/prepost_r6_silu.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/pool/prepost_r6_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | cubic-SiLU FP16→FP8 table computed from the three coefficients for every FP16 input; no model data | new |
| `tools/sources/pool/prepost_r7_pool.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/pool/prepost_r7_scales.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/r9/r9_attention_ready.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/r9/s16v_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/splitconv/native_splitconv.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/splitconv/s16v_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/splitconv_tile/r9_splitconv_tile.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/splitconv_tile/s16v_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/ffn_r11_acache.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r11vectorcomboh4v0.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r11vectorcomboh4v1.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r11vectorcomboh4v2.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r11vectorcomboh4v3.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r11vectorcomboh8v0.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r11vectorcomboh8v1.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r11vectorcomboh8v2.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r11vectorcomboh8v3.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r13linearh2v0.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r13linearh2v1.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r13linearh2v2.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r13linearh2v3.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r13pair_ffn_r5_cache.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/r13pair_swinmh_r6_qkv.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/swinmh_quant_exact.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tools/sources/swinmh/swinmh_r5_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | cubic-SiLU FP16→FP8 table computed from the three coefficients for every FP16 input; no model data | new |
| `tools/sources/swinmh/swinmh_r7_qkv.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored behaviour-compatible research CUDA/header (NVIDIA interface facts and approximation constants, see above) | new |
| `tests/test_regeneration.py` | TRP-authored (GPL v3 + exceptions) | CPU synthetic shared-alias, nested dispatch and hash-bound transform refusal fixtures; no NVIDIA kernel text | new |
| `tools/historical_ptx.py` | TRP-authored (GPL v3 + exceptions) | CPU-only adaptation of half-family round4/round8 and split16vit/integer round4/round5/round7 generators; wrapper/selection code only, no stored kernel bodies | new |
| `tools/sources/half_r4/halfpair_half.cu` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round4/halfpair_half.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r4/halfpair_nf_core.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round4/halfpair_nf_core.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r4/halfpair_nf_views.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round4/halfpair_nf_views.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r4/halfpair_of_core.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round4/halfpair_of_core.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r4/halfpair_of_views.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round4/halfpair_of_views.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r4/native_swin1h_r6_nf_helpers.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round4/native_swin1h_r6_nf_helpers.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r4/native_swin1h_r6_of_helpers.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round4/native_swin1h_r6_of_helpers.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r4/r2_quant.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round4/r2_quant.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r4/r4_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | cubic-SiLU FP16→FP8 table computed from the three coefficients for every FP16 input; no model data; research origin `swin1hfam/round4/r4_silu_lut.cuh` | new |
| `tools/sources/half_r8/control_all.cu` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/control_all.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/halfbank_all.cu` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/halfbank_all.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/halfbank_nf_core.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/halfbank_nf_core.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/halfbank_nf_views.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/halfbank_nf_views.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/halfbank_of_core.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/halfbank_of_core.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/halfbank_of_views.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/halfbank_of_views.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/native_swin1h_r6_nf_helpers.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/native_swin1h_r6_nf_helpers.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/native_swin1h_r6_of_helpers.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/native_swin1h_r6_of_helpers.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/r2_quant.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/r2_quant.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/r4_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | cubic-SiLU FP16→FP8 table computed from the three coefficients for every FP16 input; no model data; research origin `swin1hfam/round8/r4_silu_lut.cuh` | new |
| `tools/sources/half_r8/upinline_nf_core.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/upinline_nf_core.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/upinline_nf_views.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/upinline_nf_views.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/upinline_of_core.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/upinline_of_core.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/half_r8/upinline_of_views.cuh` | TRP-authored (GPL v3 + exceptions) | swin1hfam/round8/upinline_of_views.cuh; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/int8/r3_fallback_w8.cu` | TRP-authored (GPL v3 + exceptions) | int8/r3_fallback_w8.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/int8/r4_c128.cu` | TRP-authored (GPL v3 + exceptions) | int8/r4_c128.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/int8/r4_ccompact.cu` | TRP-authored (GPL v3 + exceptions) | int8/r4_ccompact.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/int8/r5_efallback.cu` | TRP-authored (GPL v3 + exceptions) | int8/r5_efallback.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/int8/r5_efp96.cu` | TRP-authored (GPL v3 + exceptions) | int8/r5_efp96.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/int8/r5_qwhole.cu` | TRP-authored (GPL v3 + exceptions) | int8/r5_qwhole.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/vit_expand/r5_expand_fixed.cu` | TRP-authored (GPL v3 + exceptions) | split16vit/best/r5-final/r5_expand_fixed.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/vit_qkv/r4_vit_q8x12.cu` | TRP-authored (GPL v3 + exceptions) | split16vit/best/r4-q8x12/r4_vit_q8x12.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/vit_r3/r3_vit_banks.cu` | TRP-authored (GPL v3 + exceptions) | split16vit/best/r3-banks/r3_vit_banks.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |
| `tools/sources/vit_r4/r4_vit_all96.cu` | TRP-authored (GPL v3 + exceptions) | split16vit/best/r4-final/r4_vit_all96.cu; Frozen TRP-authored CUDA/header; original algorithm, data-layout/ABI compatibility facts, no mechanically converted NVIDIA source | new |

## Generated at build time; never committed

| Output | Classification |
| --- | --- |
| Extracted original bundle PTX/fatbins/cubins | Generated at build time from NVIDIA input; NVIDIA terms |
| Lowered network PTX and mixed native composites | Generated at build time from NVIDIA input plus TRP code; NVIDIA input terms still apply |
| Compiled PTX of TRP-authored CUDA | Generated at build time from TRP source using the user's CUDA toolkit; excluded as build output |
| Final `.trpk` and compiled kernel binaries | Generated at build time from NVIDIA input plus TRP code; no relicensing of NVIDIA parts |

Excluded from history: original/modified DLLs, weights, all PTX files, packs,
cubins/fatbins, raw evidence, local configuration, research logs, captures,
GPU/remote benchmark runners, historical absolute-path selection lists and all
historical PTX inputs. Third-party toolchains/SDK headers are not
vendored. Full upstream TRP renderer, player mod packaging, deployment, NVAPI
submission hooks and NGX inference are outside this extracted library.

All nine inputs regenerate exactly; their source chains are documented in
REGENERATION.md and their output identities in tools/ptx-lock.json. Recovered CUDA
and headers are unchanged apart from LF normalization. Compatibility facts in
comments, FP16/FP8 arithmetic and mathematical lookup tables are treated as
authored implementations, not vendor kernel text or weights. Independent CPU
enumeration of the cubic/rounding formula matched all 65,536 values in each newly
imported SiLU table. No target PTX body
was imported or reclassified to complete the recipe.
