# File provenance

Source baseline: TRP v0.6.0, `82c33af4eb9ce001afbb6a4be9a7381e42042281`.
Shape/license reference: dlssg-turing, `96ce50f8fe863b439b955ce9456184839129c5e5`.
Research CUDA and headers are Theo's implementations and generic math tables;
they are not mechanically converted NVIDIA kernel source or model parameters.
Original source line endings are normalized to LF where imported. In particular,
the historical emulation CUDA was verified against its original
`3E50FA25587ACDF82F767AA05B0048DC531CA994C94C7A3592624EA516DE4D04`
identity before normalization.

The following table has one row for every committed file, including this file.
GPL references mean this repository's GPL v3 and its modding/linking exceptions.
The shared host-admission patch-policy facts retain the openNR/ShortFuse MIT
credit and notice in THIRD-PARTY.md. This does not grant rights to NVIDIA material.

| Committed file | Classification / license | Origin / scope |
| --- | --- | --- |
| `.gitattributes` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `.gitignore` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `CMakeLists.txt` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `LICENSE` | Third-party license text + TRP exceptions | Verbatim dlssg-turing reference license; GPL v3 with modding/linking exceptions |
| `PROVENANCE.md` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `README.md` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `REGENERATION.md` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `THIRD-PARTY.md` | TRP-authored notice + third-party MIT | openNR / Carlos Lopez Jr. MIT notice; shared patch-policy facts |
| `src/NeuralAdmission.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/NeuralAdmission.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/NeuralFaultInjection.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/NeuralLaunchRequirements.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/NeuralRenderingRuntimeIdentity.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/NeuralRenderingRuntimeIdentity.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/NeuralRuntimeKernelPlan.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored host patch facts/validation; openNR MIT patch-policy credit retained |
| `src/NeuralRuntimeKernelRegistry.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/NeuralRuntimeKernels.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/NeuralRuntimeKernels.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/NeuralRuntimeKernelsInternal.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored host patch facts/validation; openNR MIT patch-policy credit retained |
| `src/NvidiaDriverVersion.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/runtime.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `src/runtime.h` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tests/NeuralAdmissionTests.cpp` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch |
| `tests/test_inventory.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch |
| `tests/test_launch_bounds.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch |
| `tests/test_launch_metadata.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch |
| `tests/test_sm_general.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored CPU fixtures/tests; synthetic PTX only, no GPU dispatch |
| `tools/ada_lowering.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored lowering/stand-in generator; input/output kernel text excluded |
| `tools/build_pack.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/build_pascal_kernels.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/build_rtx40_kernels.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored host patch facts/validation; openNR MIT patch-policy credit retained |
| `tools/compiles.json` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/inventory.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/launch-rules.txt` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/launch_bounds.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/launch_metadata.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/pascal_groups.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored lowering/stand-in generator; input/output kernel text excluded |
| `tools/pascal_lowering.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored lowering/stand-in generator; input/output kernel text excluded |
| `tools/prepare_native.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/prepost-transform.json` | TRP-authored (GPL v3 + exceptions) | TRP-authored numeric references and wrapper insertions; NVIDIA text generated from supplied DLL at build time |
| `tools/ptx-lock.json` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/regenerate.py` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/selection.json` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/sm_general_slots.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored; standalone extraction/adaptation |
| `tools/sources/E26/E26_qkv.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/E26/vitfuse_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/E27/E27_projection.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/E27/vitfuse_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/E29/E29_ffwd.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/E29/vitfuse_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/E29/vitfuse_r5_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/E36/E36_ffwdproj.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/E36/vitfuse_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/bankpairs/bankpairs_core.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/bankpairs/bankpairs_device.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/bankpairs/postpair_scales.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/bankpairs/prepost_helpers_best.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/bankpairs/prepost_quant_exact.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/bankpairs/prepost_r6_silu.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/bankpairs/prepost_r6_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/dual/dual_core.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/dual/dual_device.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/dual/dual_pool.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/dual/prepost_helpers_best.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/dual/prepost_quant_exact.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/dual/prepost_r6_silu.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/dual/prepost_r6_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/emu/pascal_emulation.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/pool/core-r7_pool.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/pool/prepost_core_r6_lut.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/pool/prepost_helpers_best.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/pool/prepost_quant_exact.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/pool/prepost_r6_silu.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/pool/prepost_r6_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/pool/prepost_r7_pool.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/pool/prepost_r7_scales.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/r9/r9_attention_ready.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/r9/s16v_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/splitconv/native_splitconv.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/splitconv/s16v_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/splitconv_tile/r9_splitconv_tile.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/splitconv_tile/s16v_common.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/ffn_r11_acache.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r11vectorcomboh4v0.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r11vectorcomboh4v1.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r11vectorcomboh4v2.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r11vectorcomboh4v3.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r11vectorcomboh8v0.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r11vectorcomboh8v1.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r11vectorcomboh8v2.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r11vectorcomboh8v3.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r13linearh2v0.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r13linearh2v1.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r13linearh2v2.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r13linearh2v3.cu` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r13pair_ffn_r5_cache.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/r13pair_swinmh_r6_qkv.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/swinmh_quant_exact.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/swinmh_r5_silu_lut.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |
| `tools/sources/swinmh/swinmh_r7_qkv.cuh` | TRP-authored (GPL v3 + exceptions) | TRP-authored research CUDA/header; generic mathematical tables where present |

## Generated at build time; never committed

| Output | Classification |
| --- | --- |
| Extracted original bundle PTX/fatbins/cubins | Generated at build time from NVIDIA input; NVIDIA terms |
| Lowered network PTX and mixed native composites | Generated at build time from NVIDIA input plus TRP code; NVIDIA input terms still apply |
| Compiled PTX of TRP-authored CUDA | Generated at build time from TRP source using the user's CUDA toolkit; excluded as build output |
| Final `.trpk` and compiled kernel binaries | Generated at build time from NVIDIA input plus TRP code; no relicensing of NVIDIA parts |

Excluded from history: original/modified DLLs, weights, all PTX files, packs,
cubins/fatbins, raw evidence, local configuration, research logs, captures,
GPU/remote benchmark runners, historical absolute-path selection lists and the
two unresolved historical PTX inputs. Third-party toolchains/SDK headers are not
vendored. Full upstream TRP renderer, player mod packaging, deployment, NVAPI
submission hooks and NGX inference are outside this extracted library.

Incomplete inputs are identified in REGENERATION.md and tools/ptx-lock.json.
No unresolved output was reclassified as authored source to make the recipe pass.
