# Regeneration status: incomplete

The exact shipped pack was reproduced with the corrected toolchains before source
extraction. Seven historical inputs have independent source/transform recipes.
The remaining two are deliberately excluded, so the complete original-DLL-only
acceptance criterion has not been met.

| Input | Method / remaining work |
| --- | --- |
| `lower/pascal_emulation.ptx` | Historical TRP CUDA source, originally SHA-256 `3E50FA25587ACDF82F767AA05B0048DC531CA994C94C7A3592624EA516DE4D04`; compile with CUDA 12.9 `-rdc=true -ptx -arch=sm_61 -O3`, no fast math. |
| round8 `combo_prepost` | Compile bankpairs, pool and dual TRP CUDA units; extract/lower bundle 0 from the supplied DLL. Apply a hash-bound numerical selection/transform. Only authored wrapper additions are literal; original instructions are copied by numeric span from generated input. |
| `E26_qkv` | Compile TRP E26 CUDA with C++17. |
| `E27_projection` | Compile TRP E27 CUDA with C++17. |
| `E29_ffwd` | Compile TRP E29 CUDA with its authored SiLU table. |
| `E36_selected` | Compile native_splitconv, r9_splitconv_tile, E36_ffwdproj; replace the two projection entries and the finite split FFWD projection entry. |
| baseline `r9_attention_ready` | Compile frozen TRP r9 attention CUDA. |
| round8 `combo_half` | **Unresolved.** Compiling current control_all reproduces the retained ds entry, but the retained inpview entry differs (register allocation and body). Combining the regenerated halfbank up/out entries with the external historical control reproduces the target, which still depends on that control. The candidate family r6 CUDA also fails with its current headers because `swin1hfam_capture` is undefined. A scratch-only compile with capture disabled succeeds but still does not reproduce the retained inpview. A reconstruction using only current control_all and halfbank sources produces `161A7FFE767A6939040A483D21AE325B1FAADAFB5E2B00DC72CA755A1BC5645F`, rather than locked `3D01262171474D040125A6E0D8894D62C26AE3EE9D3058373D5C7D51F51412C3`. Need the matching historical inpview source/header snapshot or an audited authored transform of a demonstrably regenerated base. |
| baseline `r5_selected` | **Unresolved.** Frozen r4_c128, r4_ccompact, r3_fallback_w8, r5_efallback, r5_qwhole and r4 q8x12 CUDA units compile. They do not yet establish the exact nested r4/r5 dispatch/fallback composition. r7_vit_q128 has no same-named CUDA source: r7_gen combines earlier r5_vit_fixed and q8x12 PTX. Need to recursively regenerate those earlier selections and the final manual T240 QKV/FP32 expand selection, or express verified authored edits on those generated bases. A line-coverage experiment still left 23,284 target lines unexplained by the compiled candidate set; those lines were not imported. |

All nine locked hashes are in `tools/ptx-lock.json`; successful builds compare
actual bytes with those hashes. Source line endings are normalized in Git. The
historical composed PTX uses CRLF; regeneration emits that explicitly where needed.
NVCC's path-dependent internal-linkage identifiers for TRP's own SiLU tables are
canonicalized before checking the generated-base hashes. No absolute path is
embedded in the committed recipes.

## Authored transform audit

`tools/prepost-transform.json` stores zero-based line/count references, four
whole-instruction character-span references, full generated-source SHA-256 pins,
and TRP-authored wrapper additions. It stores no deleted lines or NVIDIA context.
The authored pieces come from the prepost stitch_v24, r7_input, r7_pool, r8_clip
and swin1hfam pairing/group-offset transformations. The inherited instruction
prefixes of combined lines are references into the generated base. Compile output
of TRP-authored sources is generated at build time, including the lookup-table
initializer. The numerical recipe intentionally fails on a changed generated base.

## Manual acceptance

Run the README's partial-build command from this repository. Inspect
`regenerated/results.json` for seven exact per-input hashes and `acceptance.json`
for the exact whole-pack hash. This is a CPU rebuild check, not inference or game
acceptance. The `complete` field stays false while either external input is needed.
The source-only command refuses a full build without those inputs and can still
regenerate the seven solved inputs with `--generate-only`.

No unproven candidate output is substituted for a locked input. No research PTX,
pack, cubin, fatbin, raw capture, benchmark runner, remote command or GPU lock is
part of this checkout. The exact seven-versus-nine status must remain visible
until the two missing source chains have passed their own locked hashes and a
new full pack build without the partial-input option.

## Checks completed for this extraction

- Corrected upstream recipe: exact shipped Pascal pack hash.
- Standalone partial recipe: all seven regenerated input hashes match, and the
  final pack matches the shipped hash exactly. The two external inputs remain
  recorded in acceptance; this is not source-only completeness.
- Windows x64 Release static library and admission executable build successfully.
- CTest: 5/5 targets passed (16 Python unit tests plus the C++ admission suite).
  CUDA 12.9 ptxas ran the five compiled launch-bounds tests; none were skipped.
- Additional CPU run: real pinned pack accepted by the host manifest/hash parser;
  copied/tampered/truncated pack cases refused. No DLL was loaded for that check.
- Full-build invocation without the partial-input option refuses before creating
  its output directory. No GPU dispatch or game acceptance was performed.
