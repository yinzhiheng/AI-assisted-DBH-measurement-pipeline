#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <vector>
#include "../DepthFrame/DepthFrame.h"
#include "../CameraModel/CameraModel.h"

// Shared preparation for precomputed masks; independent of SAM inference.
namespace sam3
{
    inline bool validGeometry(const Frame& frame, const CameraModel& cam)
    {
        const cv::Mat& K = cam.K_full();
        return !frame.depth().empty() && frame.depth().type() == CV_32FC1 &&
            !frame.rgb().empty() && frame.depthRGB().type() == CV_32FC1 &&
            frame.depthRGB().size() == frame.rgb().size() &&
            std::isfinite(frame.scaleFromLidar2Rgb()) && frame.scaleFromLidar2Rgb() > 0 &&
            K.rows == 3 && K.cols == 3 && K.type() == CV_32FC1 && cv::checkRange(K) &&
            K.at<float>(0, 0) > 0 && K.at<float>(1, 1) > 0;
    }

    inline bool inside(const cv::Point& point, const cv::Size& size)
    {
        return point.x >= 0 && point.y >= 0 && point.x < size.width && point.y < size.height;
    }

    inline cv::Mat maskAtRgbSize(const cv::Mat& input, const cv::Size& rgbSize)
    {
        if (input.empty() || input.dims != 2 || rgbSize.width <= 0 || rgbSize.height <= 0 ||
            (input.channels() != 1 && input.channels() != 3 && input.channels() != 4) ||
            input.depth() == CV_16F || !cv::checkRange(input)) return {};

        // Reject a rotated/cropped mask; resizing is for the same field of view.
        const double scaledHeight = (double)input.rows * rgbSize.width / input.cols;
        if (std::abs(scaledHeight - rgbSize.height) > 1.0) return {};
        std::vector<cv::Mat> channels;
        cv::split(input, channels);
        cv::Mat binary = cv::Mat::zeros(input.size(), CV_8U);
        // Ignore alpha: opaque black pixels are still background.
        for (size_t channel = 0; channel < std::min<size_t>(3, channels.size()); ++channel) {
            cv::Mat foreground;
            cv::compare(channels[channel], 0, foreground, cv::CMP_GT);
            cv::bitwise_or(binary, foreground, binary);
        }
        if (binary.size() != rgbSize)
            cv::resize(binary, binary, rgbSize, 0, 0, cv::INTER_NEAREST);
        if (cv::countNonZero(binary) == 0) return {};
        return binary;
    }
}
