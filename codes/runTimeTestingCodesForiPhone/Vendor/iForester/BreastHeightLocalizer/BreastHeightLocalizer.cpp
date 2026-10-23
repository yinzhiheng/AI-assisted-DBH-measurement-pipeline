#include "BreastHeightLocalizer.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

PixelLoc_cpp BreastHeightLocalizer::localize(const Frame& frame, const CameraModel& cam, const cv::Point2i& baseTap_full)
{
    const PixelLoc_cpp original = localizeImpl(frame, cam, baseTap_full, false);
    if (!original.isValid() || frame.depth().empty() || frame.depth().type() != CV_32FC1)
        return original;

    // Match measure()'s normalized Sobel magnitude and configured search window.
    cv::Mat gx, gy, magnitude, gradient;
    cv::Sobel(frame.depth(), gx, CV_64F, 1, 0, 3);
    cv::Sobel(frame.depth(), gy, CV_64F, 0, 1, 3);
    cv::magnitude(gx, gy, magnitude);
    double minimum, maximum;
    cv::minMaxLoc(magnitude, &minimum, &maximum);
    cv::Mat normalized = (magnitude - minimum) / (maximum - minimum + 1e-8);
    normalized.convertTo(gradient, CV_32F);
    const auto hasEdges = [&](const PixelLoc_cpp& loc) {
        if (!loc.isValid()) return false;
        const cv::Point seed = frame.rgbToDepth({loc.col_full, loc.row_full});
        // FindEdgeList changes its center only AFTER accepting a row. Thus its
        // nonempty/empty outcome is decided by the first bracket at this seed.
        for (int row = std::max(0, seed.y - g_DBHConfig.halfWindowSize);
            row <= std::min(gradient.rows - 1, seed.y + g_DBHConfig.halfWindowSize); ++row) {
            int left = -1, right = -1;
            for (int col = seed.x; col > 0; --col)
                if (gradient.at<float>(row, col) > g_DBHConfig.threshold) { left = col; break; }
            for (int col = seed.x; col < gradient.cols; ++col)
                if (gradient.at<float>(row, col) > g_DBHConfig.threshold) { right = col; break; }
            if (left >= 0 && right - left > 2) return true;
        }
        return false;
    };
    // Preserve every seed that already gives measure() at least one edge row.
    if (hasEdges(original)) return original;
    const PixelLoc_cpp alternative = localizeImpl(frame, cam, baseTap_full, true);
    if (!hasEdges(alternative)) return original;
    if (DBH_DEBUG) std::cout << "[BreastHeightLocalizer] Empty depth edge list at (FULL)="
        << cv::Point(original.col_full, original.row_full) << "; recovered seed="
        << cv::Point(alternative.col_full, alternative.row_full) << '\n';
    return alternative;
}

PixelLoc_cpp BreastHeightLocalizer::localizeImpl(const Frame& frame, const CameraModel& cam,
    const cv::Point2i& baseTap_full, bool requireBoundedSegments)
{
    PixelLoc_cpp out; // (-1,-1) invalid by default

    // Depth intrinsics and inverse
    const cv::Mat Kd = cam.K_depth();
    const cv::Mat KdInv = cam.K_depth_inv();

    const int Wd = frame.depthWidth();
    const int Hd = frame.depthHeight();

    // Map base tap from full-res -> depth-res (clamped)
    const cv::Point2i baseTap_depth = frame.rgbToDepth(baseTap_full);
    const int ub = baseTap_depth.x;
    const int vb = baseTap_depth.y;

    // Read base depth
    const float db = frame.depthAtDepthPx(ub, vb);
    if (!(db > m_p.minValidDepth_m)) return out;

    // Back-project base point to 3D (camera coord)
    const cv::Point3f Pb = CameraModel::pixelTo3D(KdInv, (float)ub, (float)vb, db);

    // Initial guess: move breastHeight_m upward along -Y (image coord convention)
    const float BH = m_p.breastHeight_m;
    cv::Point3f Pguess(Pb.x, Pb.y - BH, Pb.z);

    // In case failed, use the initial guess as BH loc
    // Project refined 3D point back to depth pixels
    cv::Point2f s2_uv_default = CameraModel::projectToPixel(Kd, Pguess);

    int uf = clampi((int)std::lround(s2_uv_default.x), 0, Wd - 1);
    int vf = clampi((int)std::lround(s2_uv_default.y), 0, Hd - 1);

    // Convert depth pixel -> full-res pixel
    cv::Point2i full = frame.depthToRgb(cv::Point2i(uf, vf));
    out.col_full = full.x;
    out.row_full = full.y;

    // Project the initial guess into depth image to get a predicted row
    const cv::Point2f s_uv = CameraModel::projectToPixel(Kd, Pguess);
    if (s_uv.x < 0.0f || s_uv.y < 0.0f) return out;

    int vs = clampi((int)std::lround(s_uv.y), 0, Hd - 1);

    // Depth-range filter around base depth using theta0
    const float theta0 = m_p.theta0_deg * (float)CV_PI / 180.0f;
    const float zLow = Pb.z - BH * std::tan(theta0);
    const float zHigh = Pb.z + BH * std::tan(theta0);

    // Predicate for a given row r: is depth at (r,c) valid and within [zLow,zHigh]?
    auto makeRowPredicate = [&](int r) -> std::function<bool(int)> {
        return [&, r](int c) -> bool {
            float d = frame.depthAtDepthPx(c, r);
            if (!(d > m_p.minValidDepth_m)) return false;
            return (d >= zLow && d <= zHigh);
            };
        };

    // An image-border segment has no observed edge on that side. Its midpoint
    // can lie in background merged with the trunk by the broad depth interval.
    // Only complete segments can recenter the projected breast-height seed.
    const auto boundedSegments = [&](int row, const std::function<bool(int)>& predicate) {
        auto segments = rowSegmentsByPredicate(row, Wd, predicate);
        if (requireBoundedSegments) segments.erase(std::remove_if(segments.begin(), segments.end(),
            [&](const std::pair<int, int>& segment) {
                return segment.first == 0 || segment.second == Wd - 1;
            }), segments.end());
        return segments;
    };

    // Try predicted row first
    int um = -1;
    std::pair<int, int> segChosen;

    {
        auto isValid = makeRowPredicate(vs);
        auto segs = boundedSegments(vs, isValid);
        if (!chooseClosestSegment(segs, ub, m_p.minSegmentLen_px, um, &segChosen)) {
            um = -1;
        }
    }

    // If failed, search a small vertical window around vs: vs±1, vs±2, ...
    if (um < 0) {
        const int vSearch = m_p.rowSearchWindow_px;
        bool found = false;

        for (int dv = 1; dv <= vSearch && !found; ++dv) {
            for (int sign : {+1, -1}) {
                int r = vs + sign * dv;
                if (r < 0 || r >= Hd) continue;

                auto isValid = makeRowPredicate(r);
                auto segs = boundedSegments(r, isValid);

                int u2 = -1;
                std::pair<int, int> s2;
                if (chooseClosestSegment(segs, ub, m_p.minSegmentLen_px, u2, &s2)) {
                    vs = r;
                    um = u2;
                    segChosen = s2;
                    found = true;
                    break;
                }
            }
        }
        if (!found) return out;
    }

    // Rough BH pixel on depth map: (um, vs)
    const float dm = frame.depthAtDepthPx(um, vs);
    if (!(dm > m_p.minValidDepth_m)) return out;

    const cv::Point3f Pm = CameraModel::pixelTo3D(KdInv, (float)um, (float)vs, dm);

    // Estimate growth direction N by fitting a 3D line to center points from vertical neighborhood
    const int R = m_p.growthDirWindowR_px;
    std::vector<cv::Point3f> center3D;
    center3D.reserve((size_t)(2 * R + 1));

    for (int r = std::max(0, vs - R); r <= std::min(Hd - 1, vs + R); ++r) {
        auto isValid = makeRowPredicate(r);
        auto segs = boundedSegments(r, isValid);

        int uCenter = -1;
        std::pair<int, int> segR;
        if (!chooseClosestSegment(segs, ub, m_p.minSegmentLen_px, uCenter, &segR)) continue;

        float dCenter = frame.depthAtDepthPx(uCenter, r);
        if (!(dCenter > m_p.minValidDepth_m)) continue;

        center3D.push_back(CameraModel::pixelTo3D(KdInv, (float)uCenter, (float)r, dCenter));
    }

    // If too few points, we cant fit a stable direction
    if (center3D.size() < 2) return out;

    cv::Vec3f N = fitLineDirectionPCA(center3D);

    // Ensure N points from base -> rough BH direction
    cv::Vec3f vbm(Pm.x - Pb.x, Pm.y - Pb.y, Pm.z - Pb.z);
    if (N.dot(vbm) < 0.0f) N *= -1.0f;

    // Compute distance from base to rough BH point
    const float h = cv::norm(vbm);

    // Correct to make it exactly BH: S = Pm - (h - BH) * N
    const float dh = h - BH;
    const cv::Point3f S(Pm.x - dh * N[0],
        Pm.y - dh * N[1],
        Pm.z - dh * N[2]);

    // Project refined 3D point back to depth pixels
    const cv::Point2f s2_uv = CameraModel::projectToPixel(Kd, S);
    if (s2_uv.x < 0.0f || s2_uv.y < 0.0f) return out;

    uf = clampi((int)std::lround(s2_uv.x), 0, Wd - 1);
    vf = clampi((int)std::lround(s2_uv.y), 0, Hd - 1);

    // Convert depth pixel -> full-res pixel
    full = frame.depthToRgb(cv::Point2i(uf, vf));
    out.col_full = full.x;
    out.row_full = full.y;
    return out;
}

// --------------------------- helpers ---------------------------

std::vector<std::pair<int, int>> BreastHeightLocalizer::rowSegmentsByPredicate(
    int row, int cols, const std::function<bool(int c)>& isValid)
{
    std::vector<std::pair<int, int>> segs;
    int start = -1;

    for (int c = 0; c < cols; ++c) {
        const bool ok = isValid(c);
        if (ok && start < 0) start = c;

        if ((!ok || c == cols - 1) && start >= 0) {
            int end = ok ? c : (c - 1);
            segs.emplace_back(start, end);
            start = -1;
        }
    }
    return segs;
}

bool BreastHeightLocalizer::chooseClosestSegment(
    const std::vector<std::pair<int, int>>& segs,
    int targetU,
    int minLenPx,
    int& outCenterU,
    std::pair<int, int>* outSeg)
{
    bool found = false;
    int bestCenter = -1;
    int bestDu = std::numeric_limits<int>::max();
    std::pair<int, int> bestSeg{ 0, -1 };

    for (const auto& s : segs) {
        const int len = s.second - s.first + 1;
        if (len < minLenPx) continue;

        const int center = (s.first + s.second) / 2;
        const int du = std::abs(center - targetU);

        if (!found || du < bestDu) {
            found = true;
            bestDu = du;
            bestCenter = center;
            bestSeg = s;
        }
    }

    if (!found) return false;
    outCenterU = bestCenter;
    if (outSeg) *outSeg = bestSeg;
    return true;
}

cv::Vec3f BreastHeightLocalizer::fitLineDirectionPCA(const std::vector<cv::Point3f>& pts)
{
    if (pts.size() < 2) return cv::Vec3f(0, -1, 0);

    cv::Mat X((int)pts.size(), 3, CV_32F);
    for (int i = 0; i < (int)pts.size(); ++i) {
        X.at<float>(i, 0) = pts[i].x;
        X.at<float>(i, 1) = pts[i].y;
        X.at<float>(i, 2) = pts[i].z;
    }

    cv::PCA pca(X, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Vec3f dir(pca.eigenvectors.at<float>(0, 0),
        pca.eigenvectors.at<float>(0, 1),
        pca.eigenvectors.at<float>(0, 2));

    const float n = std::sqrt(dir.dot(dir));
    if (n > 1e-6f) dir *= (1.0f / n);
    return dir;
}
