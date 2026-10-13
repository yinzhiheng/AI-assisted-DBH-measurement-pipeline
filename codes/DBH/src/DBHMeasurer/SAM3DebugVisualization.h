#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <string>

namespace sam3debug
{
    // Build a display image without changing the caller's RGB or mask.
    // Mask must be normalized CV_8UC1 at RGB resolution.
    inline cv::Mat makeMaskOverlay(const cv::Mat& rgb, const cv::Mat& mask,
        const cv::Point& bh, const cv::Point& endpointA = cv::Point(-1, -1),
        const cv::Point& endpointB = cv::Point(-1, -1))
    {
        if (rgb.empty() || rgb.depth() != CV_8U || mask.type() != CV_8UC1 ||
            mask.size() != rgb.size()) return {};
        cv::Mat overlay;
        if (rgb.channels() == 3) overlay = rgb.clone();
        else if (rgb.channels() == 1) cv::cvtColor(rgb, overlay, cv::COLOR_GRAY2BGR);
        else if (rgb.channels() == 4) cv::cvtColor(rgb, overlay, cv::COLOR_BGRA2BGR);
        else return {};
        cv::Mat tinted;
        cv::addWeighted(overlay, 0.6, cv::Mat(overlay.size(), CV_8UC3, cv::Scalar(0, 255, 0)),
            0.4, 0.0, tinted);
        tinted.copyTo(overlay, mask);

        const int stroke = std::max(1, std::min(rgb.cols, rgb.rows) / 400);
        const double fontScale = std::max(0.4, std::min(rgb.cols, rgb.rows) / 1200.0);
        const auto inside = [&](const cv::Point& p) {
            return p.x >= 0 && p.y >= 0 && p.x < rgb.cols && p.y < rgb.rows;
        };
        const auto label = [&](const std::string& text, const cv::Point& origin, const cv::Scalar& color) {
            cv::putText(overlay, text, origin, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                cv::Scalar(0, 0, 0), stroke + 2, cv::LINE_AA);
            cv::putText(overlay, text, origin, cv::FONT_HERSHEY_SIMPLEX, fontScale,
                color, stroke, cv::LINE_AA);
        };
        if (inside(endpointA) && inside(endpointB)) {
            const cv::Scalar orange(0, 128, 255);
            cv::line(overlay, endpointA, endpointB, orange, stroke + 1, cv::LINE_AA);
            for (const auto& p : { endpointA, endpointB })
                cv::circle(overlay, p, 3 * stroke, orange, -1, cv::LINE_AA);
            label("A", endpointA + cv::Point(5 * stroke, -5 * stroke), orange);
            label("B", endpointB + cv::Point(5 * stroke, -5 * stroke), orange);
        }
        if (inside(bh)) {
            cv::drawMarker(overlay, bh, cv::Scalar(0, 255, 255), cv::MARKER_CROSS,
                12 * stroke, stroke + 1, cv::LINE_AA);
            label("BH", bh + cv::Point(7 * stroke, -7 * stroke), cv::Scalar(0, 255, 255));
        }
        label("Green: mask | Yellow: BH | Orange: DBH", cv::Point(8 * stroke, 22 * stroke),
            cv::Scalar(255, 255, 255));
        return overlay;
    }
}
