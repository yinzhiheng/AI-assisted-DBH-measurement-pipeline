#include "RuntimePipeline.hpp"
#include "BreastHeightLocalizer.h"
#include "load.h"
#include <opencv2/imgcodecs/ios.h>
#import "RuntimeBridge.h"

@implementation RuntimeBridge {
    Sam _sam;
    NSUInteger _inferenceCount;
}
- (NSString *)loadModels {
    NSString *encoder = [NSBundle.mainBundle pathForResource:@"mobile_sam_preprocess" ofType:@"onnx"];
    NSString *decoder = [NSBundle.mainBundle pathForResource:@"mobile_sam" ofType:@"onnx"];
    if (!encoder || !decoder) return @"Bundled MobileSAMv2 models are missing.";
    bool terminated = false;
    // Same CPU provider, graph optimization and one intra-op thread as iForester.
    return _sam.loadModel(encoder.fileSystemRepresentation, decoder.fileSystemRepresentation, 1, &terminated)
        ? nil : @"ONNX Runtime could not load the MobileSAMv2 model pair.";
}
- (NSDictionary<NSString *, id> *)runRGB:(NSString *)rgbPath depth:(NSString *)depthPath
                                 baseX:(NSInteger)baseX baseY:(NSInteger)baseY preview:(BOOL)preview {
    NSMutableDictionary *values = [@{@"status": @"input_failed", @"error": @""} mutableCopy];
    try {
        // imread produces the same BGR layout passed to Frame by the active iOS bridge.
        cv::Mat rgb = cv::imread(rgbPath.fileSystemRepresentation, cv::IMREAD_COLOR);
        if (rgb.empty()) throw std::runtime_error("Cannot decode RGB JPEG");
        values[@"image_width"] = @(rgb.cols); values[@"image_height"] = @(rgb.rows);
        if (baseX < 0 || baseY < 0 || baseX >= rgb.cols || baseY >= rgb.rows)
            throw std::runtime_error("Base annotation is outside the RGB image");
        cv::Mat depth = readDepthMap(depthPath.fileSystemRepresentation);
        // Existing iForester file loader decodes the four PNG metadata rows,
        // transposes the serialized ARKit K and swaps portrait principal point.
        cv::Mat K = readIntrinsicMatrixFromString(readIntrinsicStringFromCombinePNG(depthPath.fileSystemRepresentation), 1.0f);
        if (!cv::checkRange(K) || K.at<float>(0,0) <= 0 || K.at<float>(1,1) <= 0)
            throw std::runtime_error("Invalid camera intrinsics");
        if (rgb.cols * depth.rows != rgb.rows * depth.cols)
            throw std::runtime_error("RGB/depth aspect ratios do not match");
        CameraModel camera(K, (float)rgb.cols / depth.cols);
        const cv::Point2i base((int)baseX, (int)baseY);
        Frame frame(depth, rgb, base);
        values[@"status"] = @"breast_height_failed";
        BreastHeightLocalizer localizer;
        auto bh = localizer.localize(frame, camera, base);
        // Match the GitHub MobileSAM batch runner: an unavailable BH estimate
        // uses the annotated base as the prompt seed instead of skipping SAM.
        const bool usedBaseFallback = !bh.isValid();
        if (usedBaseFallback) { bh.col_full = base.x; bh.row_full = base.y; }
        values[@"bh_x"] = @(bh.col_full); values[@"bh_y"] = @(bh.row_full);
        values[@"status"] = @"prompt_failed";
        DBHMeasurer measurer;
        auto result = measurer.segmentForRuntime(frame, camera, _sam, {bh.col_full, bh.row_full});
        if (usedBaseFallback) {
            if (!result.message.empty()) result.message += "; ";
            result.message += "Breast-height localization unavailable; used annotated base as in GitHub pipeline";
        }
        values[@"status"] = [NSString stringWithUTF8String:result.status.c_str()];
        values[@"error"] = [NSString stringWithUTF8String:result.message.c_str()];
        if (result.segmentationMs >= 0) {
            values[@"segmentation_ms"] = @(result.segmentationMs);
            values[@"first_pass_ms"] = @(result.firstPassMs);
            if (result.retryPassMs >= 0) values[@"retry_pass_ms"] = @(result.retryPassMs);
            values[@"segmentation_calls"] = @(result.segmentationCalls);
            values[@"session_inference_index"] = @(_inferenceCount + 1);
            values[@"first_inference_in_session"] = @(_inferenceCount == 0);
            _inferenceCount += result.segmentationCalls;
            values[@"prompt_bh_x"] = @(result.promptBreastHeight.x);
            values[@"prompt_bh_y"] = @(result.promptBreastHeight.y);
            values[@"initial_box_x"] = @(result.initialBox.x); values[@"initial_box_y"] = @(result.initialBox.y);
            values[@"initial_box_width"] = @(result.initialBox.width); values[@"initial_box_height"] = @(result.initialBox.height);
            values[@"box_x"] = @(result.box.x); values[@"box_y"] = @(result.box.y);
            values[@"box_width"] = @(result.box.width); values[@"box_height"] = @(result.box.height);
            values[@"model_box_x"] = @(result.modelBox.x); values[@"model_box_y"] = @(result.modelBox.y);
            values[@"model_box_width"] = @(result.modelBox.width); values[@"model_box_height"] = @(result.modelBox.height);
            values[@"rotated_width"] = @(result.rotatedSize.width); values[@"rotated_height"] = @(result.rotatedSize.height);
        }
        if (preview && !result.originalMask.empty()) {
            cv::Mat tinted = rgb.clone();
            tinted.setTo(cv::Scalar(70, 200, 50), result.originalMask);
            cv::addWeighted(rgb, 0.60, tinted, 0.40, 0, rgb);
            std::vector<cv::Point2f> boxPoints = {
                {(float)result.box.x, (float)result.box.y},
                {(float)result.box.br().x, (float)result.box.y},
                {(float)result.box.br().x, (float)result.box.br().y},
                {(float)result.box.x, (float)result.box.br().y}};
            std::vector<cv::Point2f> transformed;
            cv::transform(boxPoints, transformed, result.inverseRotation);
            for (int i = 0; i < 4; ++i)
                cv::line(rgb, transformed[i], transformed[(i+1)%4], cv::Scalar(255,200,0), 4);
            cv::Mat rgba; cv::cvtColor(rgb, rgba, cv::COLOR_BGR2RGBA);
            values[@"preview"] = MatToUIImage(rgba);
        }
    } catch (const std::exception& error) {
        values[@"error"] = [NSString stringWithUTF8String:error.what()];
    }
    return values;
}
@end
