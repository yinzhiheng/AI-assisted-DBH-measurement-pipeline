#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace batch
{
struct MobileSAMBatchOptions
{
    // Keep argv[0] so existing positional argument indices remain unchanged.
    std::vector<std::wstring> positional;
    bool saveMasks = false;
    bool help = false;
    std::filesystem::path debugDir;
};

inline MobileSAMBatchOptions ParseMobileSAMBatchOptions(int argc, wchar_t* argv[])
{
    MobileSAMBatchOptions options;
    options.positional.emplace_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::wstring argument(argv[i]);
        if (argument == L"--save-masks") options.saveMasks = true;
        else if (argument == L"--help") options.help = true;
        else if (argument == L"--debug-dir") {
            if (++i >= argc || std::wstring(argv[i]).rfind(L"--", 0) == 0)
                throw std::runtime_error("--debug-dir requires a directory path");
            options.debugDir = std::filesystem::path(argv[i]);
        }
        else if (argument.rfind(L"--", 0) == 0)
            throw std::runtime_error("Unknown option: " + std::filesystem::path(argument).u8string());
        else options.positional.push_back(argument);
    }
    if (!options.help && !options.saveMasks && !options.debugDir.empty())
        throw std::runtime_error("Use --save-masks with --debug-dir");
    return options;
}

// One final mask/overlay pair per case. No files or directories are created
// without --save-masks. Existing files are preserved, including on later runs.
class MobileSAMMaskExporter
{
public:
    MobileSAMMaskExporter(bool enabled, const std::filesystem::path& directory)
        : enabled_(enabled), directory_(directory)
    {
        if (enabled_) {
            std::filesystem::create_directories(directory_ / "masks");
            std::filesystem::create_directories(directory_ / "overlays");
        }
    }

    bool needsSave(const std::string& id) const
    {
        return enabled_ && (!std::filesystem::is_regular_file(maskPath(id)) ||
                            !std::filesystem::is_regular_file(overlayPath(id)));
    }

    void saveOnce(const std::string& id, const cv::Mat& bgr, const cv::Mat& mask,
                  const cv::Point2i& breastHeightRGB = cv::Point2i(-1, -1)) const
    {
        if (!needsSave(id)) return;
        if (bgr.empty() || bgr.type() != CV_8UC3 || mask.empty() ||
            mask.type() != CV_8UC1 || mask.size() != bgr.size())
            throw std::runtime_error("No RGB-aligned MobileSAM mask available for export");

        // Tint foreground red; keep the binary mask and background unchanged.
        cv::Mat overlay = bgr.clone(), tinted;
        cv::addWeighted(bgr, 0.6,
            cv::Mat(bgr.size(), CV_8UC3, cv::Scalar(0, 0, 255)), 0.4, 0.0, tinted);
        tinted.copyTo(overlay, mask);
        // Draw in original RGB coordinates, after tinting so the dot stays yellow.
        // Invalid localization has no marker: a fallback base point is not BH.
        if (breastHeightRGB.x >= 0 && breastHeightRGB.x < bgr.cols &&
            breastHeightRGB.y >= 0 && breastHeightRGB.y < bgr.rows) {
            const int radius = std::max(12, std::min(bgr.cols, bgr.rows) / 60);
            cv::circle(overlay, breastHeightRGB, radius + 2, cv::Scalar(0, 0, 0),
                       cv::FILLED, cv::LINE_AA);
            cv::circle(overlay, breastHeightRGB, radius, cv::Scalar(0, 255, 255),
                       cv::FILLED, cv::LINE_AA);
        }
        writePNGOnce(maskPath(id), mask);
        writePNGOnce(overlayPath(id), overlay);
    }

private:
    std::filesystem::path maskPath(const std::string& id) const
    { return directory_ / "masks" / (id + "_mask.png"); }
    std::filesystem::path overlayPath(const std::string& id) const
    { return directory_ / "overlays" / (id + "_overlay.png"); }

    static void writePNGOnce(const std::filesystem::path& path, const cv::Mat& image)
    {
        if (std::filesystem::is_regular_file(path)) return;
        std::vector<uchar> bytes;
        if (!cv::imencode(".png", image, bytes, { cv::IMWRITE_PNG_COMPRESSION, 3 }))
            throw std::runtime_error("PNG encoding failed: " + path.u8string());
        // Native filesystem paths support Unicode folders on Windows.
        const std::filesystem::path temporary = path.parent_path() / (path.filename().wstring() + L".tmp");
        std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
        if (!stream) throw std::runtime_error("Cannot create: " + temporary.u8string());
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        stream.close();
        if (!stream) throw std::runtime_error("Cannot save: " + temporary.u8string());
        std::filesystem::rename(temporary, path);
    }

    bool enabled_;
    std::filesystem::path directory_;
};
} // namespace batch
