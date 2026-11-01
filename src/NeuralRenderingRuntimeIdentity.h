#pragma once

#include <string_view>

#include <cstdint>
#include <filesystem>
#include <string>

namespace TheosRenderPipeline::NeuralRenderingRuntimeIdentity
{
	inline constexpr std::uint64_t kExpectedRuntimeSize = 165840496ull;
	inline constexpr std::string_view kExpectedRuntimeSha256 =
		"CEB6432F6FBDF44D886014BCD47241932BF8B67439FEEF9BBDD0961436662650";

	struct Snapshot
	{
		bool present{ false };
		bool matched{ false };
		std::uint64_t size{ 0 };
		std::string version{ "not queried" };
		std::string sha256{ "not calculated" };
		std::filesystem::path path{};
	};

    inline std::filesystem::path Normalize(const std::filesystem::path& path) {
        std::error_code error;
        auto result = std::filesystem::weakly_canonical(path, error);
        return error ? std::filesystem::absolute(path).lexically_normal() : result;
    }
    Snapshot VerifyExpected(const std::filesystem::path&, std::uint64_t, std::string_view, bool fresh = false);
    std::uint64_t HashInvocations();
}
