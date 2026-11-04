# dlssnr-pascal

Experimental developer component for building and loading TRP's GTX 10 (Pascal,
**sm_61 only**) compatibility pack for NVIDIA Neural Rendering **310.9.16**.
Tested only on a **GTX 1070 Ti**. Requires **driver 582.78+ on the R580 Pascal
branch**; the inherited admission rule is `58278 <= version < 59000`.

All nine historical PTX inputs regenerate byte-exactly from committed sources
and the original DLL. The complete pack rebuild matches the pinned shipped hash.
See [REGENERATION.md](REGENERATION.md). No NVIDIA or compiled PTX is vendored.

This is a developer component, not an end-user mod. NR reaches players through
Theo's Render Pipeline's **Universal edition on Nexus**.

## Inputs and build

The repository ships no NVIDIA files: no DLLs, weights, packs, cubins, fatbins, and
no NVIDIA code text or compiled PTX. The CUDA sources are behaviour-compatible
reimplementations. To reproduce NVIDIA's results bit-exactly they use NVIDIA's
kernel interfaces (kernel, parameter and class names; parameter-block and
weight-blob offsets) and a few of its approximation constants, such as the cubic
SiLU coefficients. Users supply the original NVIDIA DLL and both toolkits:

- Original NR 310.9.16 DLL, 266,424,944 bytes, SHA-256
  `2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F`.
- CUDA **12.9 V12.9.86**: nvcc, ptxas, fatbinary. CUDA 13 cannot target sm_61.
- CUDA **13.4 V13.4.92**: cuobjdump extraction only.
- MSVC **19.44.35215** and the Windows x64 SDK. Python 3.10+ and CMake 3.21+.

CPU-only library and regression checks (use a build directory outside the repo):

```powershell
cmake -S . -B <build-root>/host -A x64 -DDLSSNR_PASCAL_PTXAS=<cuda12-root>/bin/ptxas.exe
cmake --build <build-root>/host --config Release
ctest --test-dir <build-root>/host -C Release --output-on-failure
```

Build the complete pack from the original DLL and toolchains:

```powershell
python -I -B tools/build_pack.py --official <original-dll> --cuda12 <cuda12-root> --cuda13 <cuda13-root> --vcvars <vcvars64.bat> --output <new-build-root>
```

Add `--generate-only` to regenerate and verify all nine locked PTX inputs without
assembling the pack. No historical PTX, research directory, selection list or prior
pack is consumed. All PTX and the twelve H2/H4/H8 allocator variants are rebuilt
from committed sources or generated from the supplied original DLL.
`selection.json` uses portable relative paths. The script refuses mismatched
inputs, incomplete lock coverage and a differing final pack hash. It records nine
per-input hashes in `regenerated/results.json` and whole-pack acceptance in
`acceptance.json` (`complete: true`, `externalInputs: []`). Build output stays in
the supplied build directory; the removed `--partial-input-root` option is no
longer accepted.
The expected Pascal pack is 11,574,444 bytes, SHA-256
`5AC8546C9E08AD8D0722DE54DA5E2BFDB1976FCCC2D227B3079E739BDA2C4B43`.

## Host integration

Link `dlssnr_pascal::compat`; include `runtime.h`. Windows x64/MSVC is required.
The source is extracted/adapted from TRP v0.6.0, revision
`82c33af4eb9ce001afbb6a4be9a7381e42042281`. No Skyrim, SKSE or NGX SDK is needed to
compile this static library. The rendering host still supplies its NGX/NVAPI
integration and owns all feature lifetime and GPU submission.

Use `QueryNeuralAdmission` with the actual rendering adapter LUID, original DLL,
scene dimensions and pass count, opting into compatibility with `universal=true`.
`NeuralAdmissionCache` caches stable admission and polls the live DXGI budget every
three seconds; invalidate it on Apply, ownership retirement, or request changes.
Hardware probes use CUDA driver attributes/LUID matching and NVAPI; the display
driver version uses NVAPI with the rendering adapter's DXGI UMD version fallback.
Unknown driver/component facts refuse admission. Calling those live probes is a
host operation; the regression tests do not call them.

After successful sm_61 admission with a known extent, call
`dlssnr_pascal::Load(original, pack, admission)` **before any NGX initialization**,
under exclusive host loading ownership. Use the pinned pack basename
`nvngx_dlssnr_310.9.16_pascal_v1.trpk`. The original disk DLL remains unchanged.
The installer verifies hashes, loaded-image identity, manifest, patch bytes,
publication and rollback. Applied or unsafe installations retain their module
and storage for process lifetime. Never initialize or fall back from an unsafe
result. Follow the returned status; a resident image without exclusive ownership
is refused.

The host must consume `RuntimeLaunchRequirements(module)` in its NVAPI launch
submission path: enforce each block/grid multiplier, dynamic shared-memory and
compiled bound. The static library exposes metadata; it does not install TRP's
NVAPI submission hook or run inference. A pack-loading success alone is not
correct launch integration or game qualification.

The SM-independent ticket-ring scratch allocator **relies on in-order GPU
dispatch**. This is an accepted correctness assumption for this path, not a proof
for other hosts. The scheduler model also assumes finite producer delays and fair
resumption. Cross-CTA producer/slice waits remain; CPU tests do not prove physical
WDDM/CUDA scheduling liveness on another host.

There is **no fixed VRAM minimum**. Admission compares the live rendering adapter's
DXGI local budget/usage against a conservative estimate:

```text
62,811,622 bytes + 64 MiB + passes * (128 MiB + width * height * 1152 bytes)
```

This is approximately 62.8 MB of pack globals plus reserve and per-pass working
sets; it is an estimate, not measured residency. Dimensions must be 1..16384 and
passes 1..5. Retained allocations are accounted for; replacement admission must
wait for proven retirement and a fresh budget query.

## Credits and terms

Theo authored the Pascal stand-ins, native kernels, lowering, allocator and host
integration. Component structure and license follow `theosw/dlssg-turing`.
Carlos Lopez Jr. / ShortFuse's **openNR** is credited under MIT for shared
host-admission patch-policy facts. NVIDIA retains its own runtime/kernel/model
terms; openNR's MIT license does not cover NVIDIA kernels or weights.
See [LICENSE](LICENSE), [THIRD-PARTY.md](THIRD-PARTY.md) and
[PROVENANCE.md](PROVENANCE.md).
