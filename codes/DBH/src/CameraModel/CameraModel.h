#pragma once
#include <opencv2/core.hpp>

// Minimal pinhole camera model used by BH/DBH algorithms.
// Stores 1) intrinsics at "full-res" 2) scaled intrinsics for depth-res via scaleDepthToRGB.
class CameraModel
{
public:
    CameraModel() = default;

    // K_full: 3x3 CV_32F
    CameraModel(const cv::Mat& K_full, float s);

    void setScaleDepthToRGB(float s);

    float scaleDepthToRGB() const { return m_scale; }

    // Full-res intrinsics (as provided)
    const cv::Mat& K_full() const { return m_Kfull; }
    cv::Mat K_full_inv() const;

    // Depth-res intrinsics (scaled by / scaleDepthToRGB)
    cv::Mat K_depth() const;
    cv::Mat K_depth_inv() const;

    // Back-project a depth pixel (u,v) with depth d (meters) into camera 3D
    static cv::Point3f pixelTo3D(const cv::Mat& Kinv, float u, float v, float d);

    // Project a camera 3D point into pixel using K
    static cv::Point2f projectToPixel(const cv::Mat& K, const cv::Point3f& P);

private:
    cv::Mat m_Kfull;   // 3x3 CV_32F
    float   m_scale = 1.0f;
};
