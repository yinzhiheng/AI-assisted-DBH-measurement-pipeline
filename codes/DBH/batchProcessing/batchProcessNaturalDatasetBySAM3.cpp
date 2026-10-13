#include "load.h"
#include "CameraModel.h"
#include "DepthFrame.h"
#include "BreastHeightLocalizer.h"
#include "DBHMeasurer.h"
#include "configs.h"
#include "BatchInput.h"
#include "BatchPaths.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <unordered_map>

namespace fs = std::filesystem;

namespace
{
using batch::ReadCsvLine;
using batch::ParseCsvLine;

int ParseNonnegativeInt(const std::string& text)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected a nonnegative integer, got: " + text);
    return std::stoi(text);
}

std::unordered_map<std::string, cv::Point2i> LoadBasePoints(const fs::path& csvPath)
{
    std::ifstream input(csvPath, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open base-point CSV: " + csvPath.u8string());
    std::string line;
    if (!ReadCsvLine(input, line)) throw std::runtime_error("Base-point CSV is empty");
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
    const auto header = ParseCsvLine(line);
    const auto column = [&](const std::string& name) -> size_t {
        const auto it = std::find(header.begin(), header.end(), name);
        if (it == header.end()) throw std::runtime_error("Missing CSV column: " + name);
        return static_cast<size_t>(it - header.begin());
    };
    const size_t idCol = column("uniqueName");
    const size_t xCol = column("baseCol");
    const size_t yCol = column("baseRow");
    std::unordered_map<std::string, cv::Point2i> points;
    size_t lineNumber = 1;
    while (ReadCsvLine(input, line)) {
        ++lineNumber;
        if (Trim(line).empty()) continue;
        try {
            const auto fields = ParseCsvLine(line);
            if (fields.size() != header.size())
                throw std::runtime_error("CSV field count does not match header");
            const std::string& id = fields[idCol];
            if (id.empty()) throw std::runtime_error("Empty uniqueName");
            const cv::Point2i point(ParseNonnegativeInt(fields[xCol]),
                                    ParseNonnegativeInt(fields[yCol]));
            const auto inserted = points.emplace(id, point);
            if (!inserted.second) {
                const cv::Point2i previous = inserted.first->second;
                std::cerr << "Warning: base-point CSV line " << lineNumber
                          << " replaces an earlier entry for " << id
                          << ": (" << previous.x << ", " << previous.y << ") -> ("
                          << point.x << ", " << point.y << ")\n";
                inserted.first->second = point;
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Base-point CSV line " + std::to_string(lineNumber)
                                     + ": " + e.what());
        }
    }
    if (input.bad()) throw std::runtime_error("Error reading base-point CSV");
    // A header-only CSV is valid: every sample will be recorded as skipped.
    return points;
}

void WriteResult(std::ostream& output, const std::string& uniqueName, const DiameterResultCPP& diaRes,
                 const std::optional<double>& groundTruthM)
{
    // CSV escaping doubles quotes inside the identifier.
    output << std::quoted(uniqueName, '"', '"') << ',';
    if (diaRes.ok) {
        output << diaRes.growthDirXYZ[0] << ',' << diaRes.growthDirXYZ[1] << ','
               << diaRes.growthDirXYZ[2] << ',' << static_cast<double>(diaRes.diameter_m) * 100.0 << ','
               << diaRes.distanceToCamera_m;
    }
    else output << "0,0,0,0,0";
    output << ',';
    if (groundTruthM) output << *groundTruthM * 100.0;
    output << '\n';
    output.flush();
    if (!output) throw std::runtime_error("Failed to write results CSV");
}
} // namespace


int wmain(int argc, wchar_t* argv[])
{
    try {
        if (argc > 7 || (argc == 2 && std::wstring(argv[1]) == L"--help")) {
            std::cout << "Usage: batchProcessNaturalDatasetBySAM3.exe [project_root] [output_csv] [max_files] [data_dir] [segmentation_dir] [base_points_csv]\n"
                      << "Defaults: auto-detected package root, output/DBHResults/SAM3/results_SAM3_natural.csv, "
                      << "0 (all files), project_root/data, project_root/output/SAM3SegResults/targetTreeMasks.\n"
                      << "Mask subfolder: natural. Failed/skipped predictions are zero; ground truth is retained.\n";
            return argc > 7 ? 1 : 0;
        }
        const fs::path root = batch::ProjectRoot(argc > 1 ? fs::path(argv[1]) : fs::path{});
        fs::path outputPath = argc > 2 ? fs::path(argv[2]) : fs::path("output/DBHResults/SAM3/results_SAM3_natural.csv");
        if (outputPath.is_relative()) outputPath = root / outputPath;
        const int maxFiles = argc > 3 ? ParseNonnegativeInt(fs::path(argv[3]).u8string()) : 0;
        const fs::path dataDir = argc > 4 ? batch::ResolvePath(root, fs::path(argv[4])) : root / "data";
        const fs::path segmentationDir = argc > 5 ? batch::ResolvePath(root, fs::path(argv[5]))
            : root / "output/SAM3SegResults/targetTreeMasks";
        const fs::path maskDir = segmentationDir / "natural";
        const fs::path depthDir = dataDir / "rawdata/depth_naturalField";
        const fs::path rgbDir = dataDir / "rawdata/rgb_naturalField";
        const fs::path basePointsPath = argc > 6 ? batch::ResolvePath(root, fs::path(argv[6]))
            : dataDir / "basePoints_natural.csv";
        const fs::path labelsPath = dataDir / "labelsForNaturalField.csv";

        if (!fs::is_directory(depthDir) || !fs::is_directory(rgbDir))
            throw std::runtime_error("Depth or RGB directory is missing under: " + dataDir.u8string());
        if (!fs::is_directory(maskDir))
            throw std::runtime_error("Mask directory is missing: " + maskDir.u8string());
        // Keep the same base-point duplicate policy and label unit mode as MobileSAM.
        const auto basePoints = LoadBasePoints(basePointsPath);
        const auto groundTruth = batch::LoadGroundTruth(labelsPath);
        std::cout << "Project root: " << root.u8string() << '\n'
                  << "Data directory: " << dataDir.u8string()
                  << "\nBase points: " << basePointsPath.u8string()
                  << "\nLabels: " << labelsPath.u8string()
                  << "\nSAM3 masks: " << maskDir.u8string() << '\n';
        std::vector<fs::path> depthFiles;
        for (const auto& entry : fs::directory_iterator(depthDir)) {
            const std::string name = entry.path().filename().u8string();
            if (entry.is_regular_file() && name.rfind("combine-", 0) == 0
                && entry.path().extension() == ".png") depthFiles.push_back(entry.path());
        }
        std::sort(depthFiles.begin(), depthFiles.end());
        if (depthFiles.empty()) throw std::runtime_error("No combine-*.png files in depth directory");
        if (maxFiles > 0 && depthFiles.size() > static_cast<size_t>(maxFiles))
            depthFiles.resize(static_cast<size_t>(maxFiles));

        DBH_DEBUG = false;
        BreastHeightLocalizer bh;
        DBHMeasurer meas;
        // All startup inputs are checked before opening the requested output.
        fs::create_directories(outputPath.parent_path());
        std::ofstream output(outputPath);
        if (!output) throw std::runtime_error("Cannot open output CSV: " + outputPath.u8string());
        output.imbue(std::locale::classic());
        output << std::setprecision(std::numeric_limits<float>::max_digits10);
        output << "uniqueName,growthDirXYZ_x,growthDirXYZ_y,growthDirXYZ_z,dbhSAM_UV_cm,distanceToCamera_m,dbhCaliper_cm\n";

        size_t succeeded = 0, skipped = 0, missingMasks = 0, localizationFailures = 0;
        for (size_t i = 0; i < depthFiles.size(); ++i) {
            const fs::path& depthFile = depthFiles[i];
            const std::string id = depthFile.stem().u8string().substr(8);
            DiameterResultCPP diaRes;
            cv::Point2i bhFull(-1, -1);
            std::string status = "input_error", error;
            const auto label = groundTruth.find(id);
            const std::optional<double> truth = label == groundTruth.end() ? std::nullopt : label->second;
            if (!truth) std::cerr << "Warning: missing ground truth for " << id << '\n';
            try {
                const auto point = basePoints.find(id);
                if (point == basePoints.end()) {
                    ++skipped;
                    status = "skipped_missing_base_point";
                    throw std::runtime_error("No base point; measurement skipped");
                }
                const cv::Point2i base = point->second;
                // Use native filesystem paths for the Unicode Desktop folder.
                const fs::path maskFile = maskDir / (id + "_selected_mask.png");
                if (!fs::is_regular_file(maskFile)) {
                    ++missingMasks;
                    status = "missing_mask";
                    throw std::runtime_error("Missing tree mask: " + maskFile.u8string());
                }
                const cv::Mat treeMask = readImageFile(maskFile, cv::IMREAD_UNCHANGED);
                if (treeMask.empty()) throw std::runtime_error("Cannot read tree mask: " + maskFile.u8string());
                const fs::path rgbFile = rgbDir / ("rgb-" + id + ".jpeg");
                const cv::Mat bgr = readImageFile(rgbFile, cv::IMREAD_COLOR);
                if (bgr.empty()) throw std::runtime_error("Cannot read RGB: " + rgbFile.u8string());
                if (base.x >= bgr.cols || base.y >= bgr.rows)
                    throw std::runtime_error("Base point is outside the RGB image");
                const cv::Mat combined = readImageFile(depthFile, cv::IMREAD_UNCHANGED);
                const cv::Mat depth = readDepthMap(combined);
                const cv::Mat K_full = readIntrinsicMatrixFromString(
                    readIntrinsicStringFromCombinePNG(combined), 1.0f);
                Frame frame(depth, bgr, base);
                CameraModel cam(K_full, frame.scaleFromLidar2Rgb());
                status = "pipeline_error";
                const PixelLoc bhLoc = bh.localize_SAM3(frame, cam, treeMask, base);
                if (!bhLoc.isValid()) {
                    ++localizationFailures;
                    status = "localization_failed";
                    throw std::runtime_error("localize_SAM3 returned an invalid breast-height pixel");
                }
                bhFull = cv::Point2i(bhLoc.col_full, bhLoc.row_full);
                diaRes = meas.measure_SAM3(frame, cam, treeMask, bhFull);
                if (diaRes.ok && (!std::isfinite(diaRes.diameter_m) || diaRes.diameter_m <= 0 ||
                    !std::isfinite(diaRes.distanceToCamera_m) || diaRes.distanceToCamera_m <= 0 ||
                    !std::isfinite(diaRes.growthDirXYZ[0]) || !std::isfinite(diaRes.growthDirXYZ[1]) ||
                    !std::isfinite(diaRes.growthDirXYZ[2]))) diaRes.ok = false;
                if (diaRes.ok) {
                    ++succeeded;
                    status = "ok";
                }
                else {
                    status = "measurement_failed";
                    error = "DBHMeasurer::measure_SAM3 did not return a valid diameter";
                }
            }
            catch (const std::exception& e) {
                diaRes = DiameterResultCPP{};
                error = e.what();
            }
            WriteResult(output, id, diaRes, truth);
            std::cout << '[' << i + 1 << '/' << depthFiles.size() << "] " << id << ": " << status
                      << ", diameter_m=" << (diaRes.ok ? diaRes.diameter_m : 0.0f);
            if (bhFull.x >= 0) std::cout << ", BH=(" << bhFull.x << ',' << bhFull.y << ')';
            if (!error.empty()) std::cout << " (" << error << ')';
            std::cout << std::endl;
        }
        output.close();
        if (!output) throw std::runtime_error("Failed to close results CSV");
        std::cout << "Saved " << depthFiles.size() << " rows to " << outputPath.u8string()
                  << "\nSucceeded: " << succeeded << ", skipped (no base point): " << skipped
                  << ", failed: " << depthFiles.size() - succeeded - skipped
                  << ", missing masks: " << missingMasks
                  << ", localization failures: " << localizationFailures << '\n';
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
}
