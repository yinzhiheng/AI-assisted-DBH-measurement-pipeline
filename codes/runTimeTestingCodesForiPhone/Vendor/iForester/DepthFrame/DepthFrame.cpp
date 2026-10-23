#include "DepthFrame.h"

static cv::Mat ensureDepth32f(const cv::Mat& depth)
{
    if (depth.empty()) throw std::invalid_argument("DepthFrame: depth is empty.");
    if (depth.channels() != 1) throw std::invalid_argument("DepthFrame: depth must be single-channel.");
    cv::Mat d32f;
    if (depth.type() == CV_32F) d32f = depth;
    else depth.convertTo(d32f, CV_32F);
    return d32f;
}

Frame::Frame(const cv::Mat& depth,const cv::Mat& rgb, const cv::Point2i tappedBasePointRGB):
    m_depth(ensureDepth32f(depth)),
    m_rgb(rgb),
    m_basePoint(tappedBasePointRGB)
{
    m_scale = (float)rgbWidth() / (float)depthWidth();
    cv::resize(
        m_depth,
        m_depth_RGBres,
        m_rgb.size(),      // full RGB resolution
        0, 0,
        cv::INTER_NEAREST
    );
}

cv::Point2i Frame::rgbToDepth(const cv::Point2i& uv_full) const
{
    const int u = (int)std::lround((float)uv_full.x / m_scale);
    const int v = (int)std::lround((float)uv_full.y / m_scale);

    const int uc = clampi(u, 0, depthWidth() - 1);
    const int vc = clampi(v, 0, depthHeight() - 1);
    return cv::Point2i(uc, vc);
}

cv::Point2i Frame::depthToRgb(const cv::Point2i& uv_depth) const
{
    const int u = (int)std::lround((float)uv_depth.x * m_scale);
    const int v = (int)std::lround((float)uv_depth.y * m_scale);

    const int uc = clampi(u, 0, rgbWidth() - 1);
    const int vc = clampi(v, 0, rgbHeight() - 1);
    return cv::Point2i(uc, vc);
}

float Frame::depthAtDepthPx(int u_depth, int v_depth) const
{
    if (m_depth.empty()) return 0.0f;
    u_depth = clampi(u_depth, 0, depthWidth() - 1);
    v_depth = clampi(v_depth, 0, depthHeight() - 1);

    const float d = m_depth.at<float>(v_depth, u_depth);
    return isValidDepth(d) ? d : 0.0f;
}
