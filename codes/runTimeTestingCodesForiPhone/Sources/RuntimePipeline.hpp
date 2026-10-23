#pragma once
#include "DBHMeasurer.h"
#include <string>

struct RuntimeSegmentationResult {
    std::string status = "prompt_failed";
    std::string message;
    double segmentationMs = -1;
    double firstPassMs = -1;
    double retryPassMs = -1;
    int segmentationCalls = 0;
    cv::Point2i breastHeight{-1, -1};
    cv::Point2f promptBreastHeight{-1, -1};
    cv::Rect initialBox;
    cv::Rect box;
    cv::Rect modelBox;
    cv::Size rotatedSize;
    cv::Mat originalMask;
    cv::Mat inverseRotation;
};
