#include "RuntimePipeline.hpp"
#include "BreastHeightLocalizer.h"
#include "load.h"
#include <fstream>
#include <iomanip>

float g_displayScale = 1.0f; // Unused desktop click callback in the copied loader.
bool g_hasTap = false;
cv::Point2i g_tap_full;

// Loaded by the Python driver after loading host OpenCV. The production
// localizer, Frame, CameraModel, file loader and RuntimePipeline are compiled
// without changes; only neural inference is replaced with an explicit probe.
extern "C" int inspectPrompt(const char* rgbPath, const char* depthPath,
                             int baseX, int baseY, const char* reportPath) {
    try {
        cv::Mat rgb = cv::imread(rgbPath, cv::IMREAD_COLOR);
        cv::Mat depth = readDepthMap(depthPath);
        cv::Mat K = readIntrinsicMatrixFromString(readIntrinsicStringFromCombinePNG(depthPath), 1.0f);
        const cv::Point base(baseX, baseY);
        Frame frame(depth, rgb, base);
        CameraModel camera(K, frame.scaleFromLidar2Rgb());
        BreastHeightLocalizer localizer;
        auto bh = localizer.localize(frame, camera, base);
        if (!bh.isValid()) { bh.col_full = base.x; bh.row_full = base.y; }
        Sam sam;
#ifdef RUNTIME_REAL_SAM_REGRESSION
        bool terminated = false;
        const auto modelDirectory = std::string(RUNTIME_MODELS_DIRECTORY);
        if (!sam.loadModel(modelDirectory + "/mobile_sam_preprocess.onnx",
                           modelDirectory + "/mobile_sam.onnx", 1, &terminated))
            throw std::runtime_error("Cannot load regression models");
#endif
        DBHMeasurer measurer;
        auto result = measurer.segmentForRuntime(frame, camera, sam, {bh.col_full, bh.row_full});
#ifdef RUNTIME_REAL_SAM_REGRESSION
        const bool reached = result.segmentationCalls > 0 && result.segmentationMs >= 0 &&
            !result.originalMask.empty() && cv::countNonZero(result.originalMask) > 0;
        if (reached) {
            cv::imwrite(std::string(reportPath) + ".mask.png", result.originalMask);
            cv::Mat tinted = rgb.clone();
            tinted.setTo(cv::Scalar(70, 200, 50), result.originalMask);
            cv::addWeighted(rgb, 0.60, tinted, 0.40, 0, rgb);
            std::vector<cv::Point2f> corners = {
                {(float)result.box.x, (float)result.box.y},
                {(float)result.box.br().x, (float)result.box.y},
                {(float)result.box.br().x, (float)result.box.br().y},
                {(float)result.box.x, (float)result.box.br().y}};
            std::vector<cv::Point2f> mapped;
            cv::transform(corners, mapped, result.inverseRotation);
            for (int i = 0; i < 4; ++i)
                cv::line(rgb, mapped[i], mapped[(i+1)%4], cv::Scalar(255,200,0), 4);
            cv::imwrite(std::string(reportPath) + ".preview.jpg", rgb);
        }
#else
        const bool reached = sam.reachedEncoder && result.segmentationCalls == 1 &&
            result.message == "TEST_ENCODER_BOUNDARY" && result.box.area() > 0 &&
            result.modelBox.area() > 0;
#endif
        std::ofstream report(reportPath, std::ios::app);
        report << std::quoted(rgbPath) << ' ' << reached << ' '
               << bh.col_full << ' ' << bh.row_full << ' '
               << result.box.x << ' ' << result.box.y << ' '
               << result.box.width << ' ' << result.box.height << ' '
               << result.rotatedSize.width << ' ' << result.rotatedSize.height << ' '
               << result.modelBox.x << ' ' << result.modelBox.y << ' '
               << result.modelBox.width << ' ' << result.modelBox.height << ' '
               << std::quoted(result.message) << '\n';
        if (!report) throw std::runtime_error("Cannot write prompt regression report");
        return reached ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << rgbPath << ": " << error.what() << '\n';
        return -1;
    }
}
