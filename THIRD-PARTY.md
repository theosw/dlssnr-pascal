# Third-party notices

TRP's GPL v3 license with its modding/linking exceptions is in LICENSE, retained
verbatim from the reference component. The imported implementation and research
sources are TRP-authored except for the shared patch-policy provenance below.

## openNR / ShortFuse

Host-admission patch facts also correspond to Carlos Lopez Jr. / ShortFuse's
[openNR](https://github.com/clshortfuse/openNR/tree/57ca170cb836f549f0e64544f282d535666b0f7d)
at revision `57ca170cb836f549f0e64544f282d535666b0f7d`, under MIT. Credit is retained
for those shared facts in the host patch plan and builder checks. No community
runtime or SM75/SM86 kernel payload is included or used by this Pascal recipe.

Copyright (c) 2026 Carlos Lopez Jr.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## NVIDIA

NVIDIA's DLL, kernels, weights and CUDA toolchains are separately supplied inputs
and retain NVIDIA's own terms. No grant in this repository, including openNR's MIT
notice, relicenses those inputs or the derived PTX/binary build outputs. CUDA
headers are referenced from the user's toolkit and are not vendored.

The TRP CUDA sources are behaviour-compatible reimplementations. For bit-exact
results they reproduce NVIDIA interface facts (kernel, parameter and class names;
parameter-block and weight-blob offsets) and a few approximation constants that
also appear in NVIDIA's kernels, such as the cubic SiLU coefficients. The FP8/SiLU
lookup tables are computed from those coefficients for every FP16 input; they are
not model weights. No NVIDIA code text, PTX or weights are included.

## Projects named in LICENSE

LICENSE is TRP's license text, retained verbatim. Its preamble credits
Skyrim-Upscaler (PureDark) and ENBFrameGeneration / Community Shaders (doodlum and
contributors) for parts of TRP. This component contains none of their code.
