#include "runtime.h"
#include "NeuralRenderingRuntimeIdentity.h"
namespace dlssnr_pascal {
LoadedRuntime Load(const std::filesystem::path& original, const std::filesystem::path& pack,
    const TheosRenderPipeline::NeuralRendering::NeuralCapability& admitted) {
    using namespace TheosRenderPipeline::NeuralRendering;
    using TheosRenderPipeline::NeuralRenderingRuntimeIdentity::VerifyExpected;
    if (!admitted.Pascal() || admitted.extentPending || admitted.Refusal())
        return {nullptr, {false, false, "Pascal admission has not succeeded"}};
    const auto path = std::filesystem::absolute(original);
    if (!VerifyExpected(path, 266424944, kKernelRuntimeSha256, true).matched || !VerifyPascalPack(pack))
        return {nullptr, {false, false, "original runtime or Pascal pack identity differs"}};
    // Check residency without adding a reference. An existing image has no
    // provable pre-NGX ownership and ApplyRuntimeKernels refuses it.
    HMODULE resident{};
    const bool wasLoaded = GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, path.c_str(), &resident) != 0;
    const auto module = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) return {nullptr, {false, false, "original runtime could not be loaded"}};
    auto result = ApplyRuntimeKernels(module, pack, !wasLoaded);
    if (!result.applied && !result.unsafe) { FreeLibrary(module); return {nullptr, result}; }
    return {module, result};
}
}
