#pragma once
// Test-only model boundary. The dataset regression executes the production
// prompt pipeline and stops exactly where a real encoder would be invoked.
// This directory is never on the iPhone target's header search path.
#include <opencv2/opencv.hpp>
#include <list>
#include <stdexcept>
#include "sam_inference_timing.hpp"

class Sam {
    SamInferenceTiming timing;
public:
    bool reachedEncoder = false;
    void resetInferenceTiming() { timing = {}; }
    const SamInferenceTiming& getInferenceTiming() const { return timing; }
    cv::Size getInputSize() const { return {1024, 720}; }
    bool hasPreprocessedImage(const cv::Mat&) const { return false; }
    bool preprocessImage(const cv::Mat& image, bool*) {
        if (image.size() != getInputSize() || image.type() != CV_8UC3)
            throw std::runtime_error("Invalid encoder input");
        reachedEncoder = true;
        throw std::runtime_error("TEST_ENCODER_BOUNDARY");
    }
    cv::Mat getMask(const std::list<cv::Point>&, const std::list<cv::Point>&,
                    const cv::Rect&, int, bool) { return {}; }
};
