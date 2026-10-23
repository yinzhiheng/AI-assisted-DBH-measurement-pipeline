#include "RuntimePipeline.hpp"

// Segmentation-only entry point into the copied iForester DBHMeasurer class.
// All math, rotation, box construction and inference helpers below are the
// original methods. The diameter/plane fitting branch is not needed for a box.
RuntimeSegmentationResult DBHMeasurer::segmentForRuntime(
    const Frame& frame, const CameraModel& cam, Sam& sam,
    const cv::Point2i& breastHeightPtTappedRGB) {
    RuntimeSegmentationResult out;
    out.breastHeight = breastHeightPtTappedRGB;
    const cv::Mat& depthMapLidar = frame.depth();
    const cv::Mat Kd = cam.K_depth();
    const cv::Point2i breastHeightPtLidar = frame.rgbToDepth(breastHeightPtTappedRGB);
    cv::Mat gradientXY = ComputeGradientXY(depthMapLidar);
    auto edgeList = FindEdgeList(gradientXY, cv::Point2f(breastHeightPtLidar));
    if (edgeList.empty()) { out.message = "Cannot find depth edges"; return out; }
    auto cleaned = CleanEdgeListEdgeLengthCenter(edgeList);
    if (!cleaned.ok) { out.message = "Depth edge cleaning failed"; return out; }
    cv::Vec3f growthDirXYZ(0, 0, 0), centroidXYZ(0, 0, 0);
    cv::Vec2f growthDirUV(0, 0);
    if (!ComputeGrowDirection(cleaned.centerListCleaned, depthMapLidar, Kd,
                              growthDirXYZ, growthDirUV, centroidXYZ)) {
        out.message = "Cannot estimate growth direction"; return out;
    }
    const cv::Point2i centroiduv = XYZ2LidarPoint(centroidXYZ, Kd);
    const float chordLengthInPixel = ComputeChordLength(cleaned.edgeListCleaned);
    if (!(chordLengthInPixel > 0.0f)) { out.message = "Invalid pixel width"; return out; }
    const cv::Mat& rgbImage = frame.rgb();
    const int H = rgbImage.rows, W = rgbImage.cols;
    const cv::Point2d rotationCenter((double)W * 0.5, (double)H * 0.5);
    const float scaleFromLidar2RGB = (float)W / (float)depthMapLidar.cols;
    cv::Point2f breastHeightPtRGB(centroiduv.x * scaleFromLidar2RGB, centroiduv.y * scaleFromLidar2RGB);
    auto base = frame.depthBasePoint();
    cv::Point2f basePtRGB(base.x * scaleFromLidar2RGB, base.y * scaleFromLidar2RGB);
    RotationMatrices R = computeRotationMatrixFromGrowthDirection(
        cv::Vec2d(growthDirUV[0], growthDirUV[1]), rotationCenter, H, W);
    cv::Mat rotatedImage = rotateImage(rgbImage, R.M_forward, R.outSize);
    auto baseRotated = rotatePointAroundCenter(basePtRGB, R.M_forward);
    auto breastRotated = rotatePointAroundCenter(breastHeightPtRGB, R.M_forward);
    out.box = createSAMBoxRect(breastRotated, baseRotated,
        chordLengthInPixel * scaleFromLidar2RGB, rotatedImage.rows, rotatedImage.cols,
        0.15f, 300.0f, 150.0f); // Exact explicit arguments in iForester measure().
    out.initialBox = out.box;
    out.rotatedSize = rotatedImage.size();
    out.inverseRotation = R.M_backward;

    // iForester may correct the BH center after checking the first mask, then
    // issue one more box prompt. Keep that behavior, without diameter fitting.
    for (int pass = 0; pass < 2; ++pass) {
        out.promptBreastHeight = breastHeightPtRGB;
        out.modelBox = scaleBoxToSamInput(out.box, rotatedImage.size(), sam.getInputSize());
        ++out.segmentationCalls;
        cv::Mat mask;
        std::string inferenceError;
        // Timing lives inside Sam, around the encoder and decoder ONNX Run
        // calls only. Resizing, tensor preparation and mask selection stay out.
        // Always encode, as iForester does for a newly loaded image.
        sam.resetInferenceTiming();
        try {
            mask = SAMBoxSegmentation(sam, rotatedImage, out.box, nullptr, true);
        } catch (const std::exception& error) { inferenceError = error.what(); }
        const double elapsed = sam.getInferenceTiming().totalMs();
        (pass == 0 ? out.firstPassMs : out.retryPassMs) = elapsed;
        out.segmentationMs = out.firstPassMs + std::max(0.0, out.retryPassMs);
        if (!inferenceError.empty()) {
            out.status = "segmentation_failed"; out.message = inferenceError; return out;
        }
        try {
            // These original mask/edge checks and inverse warp are NOT timed.
            cv::Mat binary, mask255;
            cv::threshold(mask, binary, 0, 1, cv::THRESH_BINARY);
            binary.convertTo(mask255, CV_8U, 255.0);
            out.originalMask = rotateImage(mask255, R.M_backward, rgbImage.size());
            cv::Mat originalBinary;
            cv::threshold(out.originalMask, originalBinary, 0, 1, cv::THRESH_BINARY);
            auto edges = findRGBEdgePointListFromSAMMask(rotatedImage, originalBinary, breastHeightPtRGB, 50);
            auto trunk = findTreeTrunkFromSAMMask(originalBinary, 0.5f);
            auto trunkEdges = findEdgeListWithTrunkMask(edges, trunk);
            auto cleanRGB = CleanEdgeListEdgeLengthCenter(trunkEdges);
            auto check = CheckBHWithinFittedEdges(cleanRGB.edgeListCleaned, breastHeightPtRGB);
            if (check.ok) { out.status = "ok"; return out; }
            if (pass == 1) {
                out.status = cv::countNonZero(mask) == 0 ? "empty_mask" : "mask_validation_failed";
                out.message = "Breast height is not within fitted trunk edges after iForester's retry";
                return out;
            }
            breastHeightPtRGB = check.updatedBreastHeightPtRGB;
            R = computeRotationMatrixFromGrowthDirection(
                cv::Vec2d(growthDirUV[0], growthDirUV[1]), rotationCenter, H, W);
            rotatedImage = rotateImage(rgbImage, R.M_forward, R.outSize);
            baseRotated = rotatePointAroundCenter(basePtRGB, R.M_forward);
            breastRotated = rotatePointAroundCenter(breastHeightPtRGB, R.M_forward);
            out.box = createSAMBoxRect(breastRotated, baseRotated,
                chordLengthInPixel * scaleFromLidar2RGB, rotatedImage.rows, rotatedImage.cols,
                0.15f, 300.0f, 150.0f);
            out.inverseRotation = R.M_backward;
        } catch (const std::exception& error) {
            out.status = "mask_validation_failed"; out.message = error.what(); return out;
        }
    }
    return out;
}
