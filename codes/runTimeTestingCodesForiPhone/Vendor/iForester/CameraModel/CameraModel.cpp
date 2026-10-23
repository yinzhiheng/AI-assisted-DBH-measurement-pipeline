#include "CameraModel.h"
#include <stdexcept>
#include <cmath>

static cv::Mat ensureK33f(const cv::Mat& K)
{
    if (K.empty() || K.rows != 3 || K.cols != 3)
        throw std::invalid_argument("CameraModel: K must be 3x3.");
    cv::Mat K32f;
    if (K.type() == CV_32F) K32f = K.clone();
    else K.convertTo(K32f, CV_32F);
    return K32f;
}

CameraModel::CameraModel(const cv::Mat& K_full, float s)
    : m_Kfull(ensureK33f(K_full)), m_scale(s)
{

}


void CameraModel::setScaleDepthToRGB(float s)
{
    if (!(s > 0.0f)) throw std::invalid_argument("CameraModel: scaleDepthToRGB must be > 0.");
    m_scale = s;
}

cv::Mat CameraModel::K_full_inv() const
{
    return m_Kfull.inv();
}

cv::Mat CameraModel::K_depth() const
{
    // Scale fx, fy, cx, cy by / scaleDepthToRGB (same as your ObjC++ code)
    const float s = m_scale;
    cv::Mat Kd = (cv::Mat_<float>(3, 3) <<
        m_Kfull.at<float>(0, 0) / s, 0.0f, m_Kfull.at<float>(0, 2) / s,
        0.0f, m_Kfull.at<float>(1, 1) / s, m_Kfull.at<float>(1, 2) / s,
        0.0f, 0.0f, 1.0f
        );
    return Kd;
}

cv::Mat CameraModel::K_depth_inv() const
{
    return K_depth().inv();
}

cv::Point3f CameraModel::pixelTo3D(const cv::Mat& Kinv, float u, float v, float d)
{
    cv::Mat ph = (cv::Mat_<float>(3, 1) << u, v, 1.0f);
    cv::Mat r = Kinv * ph;
    return cv::Point3f(r.at<float>(0) * d,
        r.at<float>(1) * d,
        r.at<float>(2) * d);
}

cv::Point2f CameraModel::projectToPixel(const cv::Mat& K, const cv::Point3f& P)
{
    if (std::abs(P.z) < 1e-6f) return cv::Point2f(-1.0f, -1.0f);
    const float x = P.x / P.z;
    const float y = P.y / P.z;
    const float u = K.at<float>(0, 0) * x + K.at<float>(0, 2);
    const float v = K.at<float>(1, 1) * y + K.at<float>(1, 2);
    return cv::Point2f(u, v);
}
