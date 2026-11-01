#pragma once
#include "NeuralAdmission.h"
#include "NeuralRuntimeKernels.h"
namespace dlssnr_pascal {
// Caller exclusively owns loading before any NGX entry. Successful and unsafe
// installations pin their modules/storage for process lifetime. Never initialize
// a module when result.applied is false. Apply the returned launch requirements
// in the host's NVAPI launch submission path; this library never dispatches GPU work.
struct LoadedRuntime { HMODULE module{}; TheosRenderPipeline::NeuralRendering::RuntimeKernelResult result; };
LoadedRuntime Load(const std::filesystem::path& original, const std::filesystem::path& pack,
    const TheosRenderPipeline::NeuralRendering::NeuralCapability& admitted);
}
