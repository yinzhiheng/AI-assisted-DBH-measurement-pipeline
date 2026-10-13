#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cmath>

// Holds depth map (CV_32F, meters) and provides common mapping utilities.
class Frame
{
public:
    Frame() = default;

    // depth: CV_32F (H x W), meters
    Frame(const cv::Mat& depth, const cv::Mat& rgb, const cv::Point2i tappedBasePoint);

    int depthWidth()  const { return m_depth.cols; }
    int depthHeight() const { return m_depth.rows; }

    int rgbWidth()  const { return m_rgb.cols; }
    int rgbHeight() const { return m_rgb.rows; }

    //float scaleDepthToRGB() const { return m_scale; }

    float scaleFromLidar2Rgb() const { return m_scale; }
    float scaleFromRgb2Lidar() const { return 1 / m_scale; }

    const cv::Mat& depth() const { return m_depth; }
    cv::Mat depthRGB() const
    {
        return m_depth_RGBres;
    }

    const cv::Mat& rgb() const { return m_rgb; }

    const cv::Point2i depthBasePoint() const
    {
        return cv::Point2i(
            static_cast<int>(std::lround(m_basePoint.x / 7.5)),
            static_cast<int>(std::lround(m_basePoint.y / 7.5))
        );
    }
    const cv::Point2i rgbBasePoint() const { return m_basePoint; }

    // Full-res pixel -> depth-res pixel (clamped)
    cv::Point2i rgbToDepth(const cv::Point2i& uv_full) const;

    // Depth-res pixel -> full-res pixel (clamped)
    cv::Point2i depthToRgb(const cv::Point2i& uv_depth) const;

    // Depth lookup (clamped). Returns 0 if invalid/out-of-range.
    float depthAtDepthPx(int u_depth, int v_depth) const;

    bool isValidDepth(float d) const { return (d > m_minValidDepth) && (d < m_maxValidDepth); }

private:
    static int clampi(int x, int lo, int hi) { return (x < lo) ? lo : (x > hi ? hi : x); }

    cv::Mat m_depth;          // CV_32F, meters
    cv::Mat m_depth_RGBres;
    cv::Mat m_rgb;
    cv::Point2i m_basePoint;
    float   m_scale = 7.5f;   // full / depth
    float   m_minValidDepth = 0.01f;
    float   m_maxValidDepth = 6.0f;
};
