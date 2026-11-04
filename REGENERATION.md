# Byte-exact regeneration

All nine historical inputs independently regenerate from committed TRP sources
and the user's original DLL. A complete build without historical PTX inputs
produces the pinned 11,574,444-byte pack, SHA-256
`5AC8546C9E08AD8D0722DE54DA5E2BFDB1976FCCC2D227B3079E739BDA2C4B43`.
The original DLL is the only NVIDIA input; its full identity and toolchain
requirements are in [README.md](README.md).

| Input | Method |
| --- | --- |
| `lower/pascal_emulation.ptx` | Historical TRP CUDA, originally SHA-256 `3E50FA25587ACDF82F767AA05B0048DC531CA994C94C7A3592624EA516DE4D04`; CUDA 12.9 `-rdc=true -ptx -arch=sm_61 -O3`, no fast math. |
| round8 `combo_prepost` | Compile bankpairs, pool and dual TRP CUDA; extract/lower bundle 0 from the supplied DLL. Apply the hash-bound numeric selection/transform described below. |
| `E26_qkv` | Compile TRP E26 CUDA with C++17. |
| `E27_projection` | Compile TRP E27 CUDA with C++17. |
| `E29_ffwd` | Compile TRP E29 CUDA with its authored SiLU table. |
| `E36_selected` | Compile native_splitconv, r9_splitconv_tile and E36_ffwdproj; select the two projection entries and the finite split FFWD projection entry. |
| baseline `r9_attention_ready` | Compile frozen TRP r9 attention CUDA. |
| round8 `combo_half` | Compile the frozen round4 halfpair, round8 control and round8 halfbank CUDA/header sets. Retain control's module envelope and ds entry, select round4's paired inpview and halfbank's up/out, then reproduce the historical bounds and shared-symbol rename. |
| baseline `r5_selected` | Compile ten frozen TRP ViT/integer CUDA units and reproduce the full recursive dispatch, fallback, shared-alias and final selection chain in `tools/historical_ptx.py`. No PTX body is stored in that transform. |

All nine full locked hashes are in `tools/ptx-lock.json`. Regeneration requires
exact lock coverage and checks every actual file's bytes before pack assembly.
`tools/compiles.json` records every source and its per-unit flags; common flags are
in `tools/regenerate.py` and allocator-variant flags in `tools/build_pack.py`. Historical composed
PTX uses CRLF; regeneration emits that explicitly. Source line endings are LF.
No absolute workstation paths are embedded in the recipes.

## Recovered half-family source chain

The retained inpview is E22's round4 `halfpair_half.cu` with its own frozen
`halfpair_*` headers, not the later round8 `control_all.cu` version. It compiles to
the exact retained entry. Round8 control independently supplies the exact ds
entry, and round8 halfbank supplies the exact up/out entries. The original
selection kept 64-register bounds for the selected entries and renamed the
upsample shared symbol to `swin1hfam_r5_up_shared`, including its demotion comment.
`combo_half` emits these transformations on compiled output and matches
`3D01262171474D040125A6E0D8894D62C26AE3EE9D3058373D5C7D51F51412C3`.

The sources live in `tools/sources/half_r4` and `half_r8`. Their SiLU table is an
authored mathematical lookup table, not model data: an independent CPU enumeration
of the cubic formula, FP16 rounding and FP8 encoding matches all 65,536 entries in
both headers. No instruction body or missing
header was reconstructed by copying the target PTX into source.

## Recovered ViT dispatch chain

The ten compiled units are `r3_vit_banks`, `r4_vit_all96`, `r5_expand_fixed`,
`r4_vit_q8x12`, `r4_c128`, `r4_ccompact`, `r3_fallback_w8`, `r5_efallback`,
`r5_efp96` and `r5_qwhole`. The CPU transform reproduces these stages in order:

1. `r5_vit_dispatch`: round4's T288 bodies and round3's general fallback share
   aliased A/B storage; `r5_vit_fixed` replaces expand with its compiled fixed/general source.
2. `r7_vit_96`: also selects the fast contract/QKV bodies at T96;
   `r7_vit_q128` replaces QKV's general fallback with compiled q8x12 bodies.
3. `r4_conly128`: integer M128 contract at T96/T240/T288, with eight-warp FP32
   fallback; `r5_c96` adds compact M96 contract at T96.
4. `r5_ebase`: replaces expand with the FP32 general fallback source;
   `r5_pre_e96` adds direct FP32 M96 expand at T96; `r5_pre_ec` adds FP32 M128 at T240.
5. `r5_selected`: adds whole-K integer QKV at T240 only, retaining the nested
   earlier exact FP32 fallback. Other token counts keep the earlier selection.

Every intermediate independently matched its historical bytes during recovery.
The builder retains generated stages under `regenerated/historical/` for auditing.
Shared arrays are packed with the historical 16-byte alignment, label suffixes
retain nested branch scope, and dispatch wrappers retain the original thread
guards and global metadata declarations. Compiler-generated temporary symbol names
for these TRP sources disappear through the historical shared aliases, so source
directory changes do not require copying historical compiler identifiers.
The final bytes match
`1D5F56D834A7CC6A5768B9D33BA382ED4E8F67479BA5C0452EDF0E8A1FACB93A`.

## Authored transform audit

`tools/historical_ptx.py` stores TRP-authored selection, dispatch wrappers, shared
aliases and label renames. All kernel bodies are obtained from fresh compilation
of the committed CUDA sources. It contains no NVIDIA kernel context, deleted
instructions, mechanically converted source or encoded target body.

`tools/prepost-transform.json` remains the existing hash-bound numerical recipe:
zero-based line/count references, four instruction character-span references,
full generated-base hashes and authored wrapper additions. It stores no deleted
lines or NVIDIA context. Inherited instruction prefixes of combined lines are
references into the generated DLL base. SiLU initializer data comes from compiling
TRP source. NVCC's path-dependent internal-linkage names for TRP's SiLU tables are
canonicalized before checking those base hashes. A changed base refuses.

## CPU acceptance

Run the complete-build command in README with a new output directory. Inspect
`regenerated/results.json` for nine exact per-input hashes and `acceptance.json`
for `complete: true`, `externalInputs: []`, `matched: true` and the pinned pack hash.
No historical pack, PTX or research selection list is read. `--generate-only`
performs all nine input checks without assembling a pack.

Windows x64 Release library/admission builds and all six CTest targets pass,
including 22 Python tests. The launch-bounds tests use CUDA 12.9 ptxas with no
skips. Regeneration tests cover shared-region alignment, scalar shared extents,
nested fallback/thread guards, missing replacements, changed generated-base
refusal and out-of-bounds numeric spans. The actual pinned pack is also accepted
by the CPU admission/hash parser when its path is passed: run
`<build-root>/host/Release/admission_tests.exe <pack>`. The pinned pack and a copy
are accepted; tampered or truncated copies refuse.
No GPU dispatch, DLL loading or game acceptance is claimed.
