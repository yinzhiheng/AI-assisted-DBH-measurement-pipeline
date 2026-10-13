#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace batch {
// The solution marks the package root. Locate it from the executable, so
// no-argument runs work from Visual Studio, Explorer, or any working directory.
inline std::filesystem::path ProjectRoot(
    const std::filesystem::path& explicitRoot = {})
{
    namespace fs = std::filesystem;
    fs::path root;
    if (!explicitRoot.empty()) {
        root = fs::absolute(explicitRoot).lexically_normal();
    }
    else {
        std::vector<wchar_t> buffer(1024);
        for (;;) {
            const DWORD count = GetModuleFileNameW(nullptr, buffer.data(),
                                                   static_cast<DWORD>(buffer.size()));
            if (count == 0) throw std::runtime_error("Cannot locate the batch executable");
            if (count < buffer.size()) {
                root = fs::path(std::wstring(buffer.data(), count)).parent_path();
                break;
            }
            if (buffer.size() >= 32768)
                throw std::runtime_error("Batch executable path is too long");
            buffer.resize(buffer.size() * 2);
        }
        while (!fs::is_regular_file(root / "codes/DBH/DBHCalculationBatch.sln")) {
            const fs::path parent = root.parent_path();
            if (parent == root || parent.empty())
                throw std::runtime_error("Cannot locate DBHCalculationBatch.sln above the executable");
            root = parent;
        }
    }
    if (!fs::is_regular_file(root / "codes/DBH/DBHCalculationBatch.sln") ||
        !fs::is_directory(root / "codes/DBH/src"))
        throw std::runtime_error("Invalid project_root: expected codes/DBH/DBHCalculationBatch.sln and codes/DBH/src");
    return root;
}

// All optional relative input/output paths are relative to the same package root.
inline std::filesystem::path ResolvePath(const std::filesystem::path& root,
                                         const std::filesystem::path& path)
{
    return (path.is_absolute() ? path : root / path).lexically_normal();
}
} // namespace batch
