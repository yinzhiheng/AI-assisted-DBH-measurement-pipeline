#include "BreastHeightLocalizer.h"
#include "../SegmentationModel/TreeMask.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

PixelLoc BreastHeightLocalizer::localize(const Frame& frame, const CameraModel& cam, const cv::Point2i& baseTap_full)
{
    const PixelLoc original = localizeImpl(frame, cam, baseTap_full, false);
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
    const auto hasEdges = [&](const PixelLoc& loc) {
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
    const PixelLoc alternative = localizeImpl(frame, cam, baseTap_full, true);
    if (!hasEdges(alternative)) return original;
    if (DBH_DEBUG) std::cout << "[BreastHeightLocalizer] Empty depth edge list at (FULL)="
        << cv::Point(original.col_full, original.row_full) << "; recovered seed="
        << cv::Point(alternative.col_full, alternative.row_full) << '\n';
    return alternative;
}

PixelLoc BreastHeightLocalizer::localizeImpl(const Frame& frame, const CameraModel& cam,
    const cv::Point2i& baseTap_full, bool requireBoundedSegments)
{
    PixelLoc out; // (-1,-1) invalid by default

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

    // If too few points, we can’t fit a stable direction
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

namespace
{
    struct SAM3BoundaryRow
    {
        int left;
        int row;
        int right;
    };

    bool finitePointSAM3(const cv::Point3f& point)
    {
        return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
    }

    // Step 00's compute_edge_map / find_trunk_boundaries / filter_valid_rows
    // are not defined in fusedMethod.ipynb. Use its later ComputeGradientXY,
    // findEdgeList and cleanEdgeListEdgeLengthCenter equivalents here.
    std::vector<cv::Point2f> trunkCentersSAM3(const cv::Mat& depth,
        const cv::Point& seed, int radius, float edgeThreshold,
        float minWidthStd, float sigmaKeep)
    {
        // Prevent isolated NaN/Inf samples from poisoning Sobel normalization.
        cv::Mat finiteDepth = depth.clone();
        for (int row = 0; row < depth.rows; ++row) {
            float* values = finiteDepth.ptr<float>(row);
            for (int col = 0; col < depth.cols; ++col) {
                if (!std::isfinite(values[col])) values[col] = 0.0f;
            }
        }

        cv::Mat dx, dy, edges;
        cv::Sobel(finiteDepth, dx, CV_64F, 1, 0, 3);
        cv::Sobel(finiteDepth, dy, CV_64F, 0, 1, 3);
        cv::magnitude(dx, dy, edges);
        double minEdge = 0.0, maxEdge = 0.0;
        cv::minMaxLoc(edges, &minEdge, &maxEdge);
        edges = (edges - minEdge) / (maxEdge - minEdge + 1e-8);

        std::vector<SAM3BoundaryRow> bounds;
        int center = seed.x;
        for (int row = std::max(0, seed.y - radius);
            row <= std::min(depth.rows - 1, seed.y + radius); ++row) {
            int left = -1, right = -1;
            // Match Python's left range, which excludes column zero.
            for (int col = center; col > 0; --col) {
                if (edges.at<double>(row, col) > edgeThreshold) {
                    left = col;
                    break;
                }
            }
            for (int col = center; col < depth.cols; ++col) {
                if (edges.at<double>(row, col) > edgeThreshold) {
                    right = col;
                    break;
                }
            }
            if (left >= 0 && right - left > 2) {
                bounds.push_back({ left, row, right });
                center = (left + right) / 2;
            }
        }
        if (bounds.size() < 2) return {};

        double meanWidth = 0.0, varianceWidth = 0.0;
        for (const auto& bound : bounds) meanWidth += bound.right - bound.left;
        meanWidth /= bounds.size();
        for (const auto& bound : bounds) {
            const double delta = bound.right - bound.left - meanWidth;
            varianceWidth += delta * delta;
        }
        const double stdWidth = std::sqrt(varianceWidth / bounds.size());
        std::vector<cv::Point2f> centers;
        for (const auto& bound : bounds) {
            const double width = bound.right - bound.left;
            if (stdWidth > minWidthStd &&
                !(width > meanWidth - sigmaKeep * stdWidth &&
                  width < meanWidth + sigmaKeep * stdWidth)) continue;
            centers.emplace_back(0.5f * (bound.left + bound.right), (float)bound.row);
        }
        if (centers.size() < 2) return {};

        // Least squares u = a*v+b, then discard perpendicular 2-sigma outliers.
        double meanU = 0.0, meanV = 0.0;
        for (const auto& point : centers) {
            meanU += point.x;
            meanV += point.y;
        }
        meanU /= centers.size();
        meanV /= centers.size();
        double covariance = 0.0, varianceV = 0.0;
        for (const auto& point : centers) {
            covariance += (point.x - meanU) * (point.y - meanV);
            varianceV += (point.y - meanV) * (point.y - meanV);
        }
        if (varianceV <= 1e-12) return {};
        const double a = covariance / varianceV;
        const double b = meanU - a * meanV;
        const double denominator = std::sqrt(1.0 + a * a);
        std::vector<double> residuals;
        double residualMean = 0.0;
        for (const auto& point : centers) {
            const double residual = (point.x - a * point.y - b) / denominator;
            residuals.push_back(residual);
            residualMean += residual;
        }
        residualMean /= residuals.size();
        double residualVariance = 0.0;
        for (double residual : residuals) {
            residualVariance += (residual - residualMean) * (residual - residualMean);
        }
        const double residualStd = std::sqrt(residualVariance / residuals.size());
        if (residualStd < 1e-6) return centers;

        std::vector<cv::Point2f> filtered;
        for (size_t i = 0; i < centers.size(); ++i) {
            if (std::abs(residuals[i]) <= sigmaKeep * residualStd)
                filtered.push_back(centers[i]);
        }
        return filtered;
    }
}

PixelLoc BreastHeightLocalizer::localize_SAM3(
    const Frame& frame, const CameraModel& cam, const cv::Point2i& baseTap_full)
{
    PixelLoc out;
    const cv::Mat& depth = frame.depth();
    const cv::Mat& Kfull = cam.K_full();
    if (depth.empty() || depth.type() != CV_32F ||
        frame.rgbWidth() <= 0 || frame.rgbHeight() <= 0 ||
        Kfull.rows != 3 || Kfull.cols != 3 || Kfull.type() != CV_32F ||
        !cv::checkRange(Kfull) ||
        !std::isfinite(cam.scaleDepthToRGB()) || cam.scaleDepthToRGB() <= 0.0f)
        return out;

    if (!std::isfinite(m_p.breastHeight_m) || m_p.breastHeight_m <= 0.0f ||
        !std::isfinite(m_p.minValidDepth_m) || m_p.minValidDepth_m < 0.0f ||
        !std::isfinite(m_p.sam3MaxDepth_m) || m_p.sam3MaxDepth_m <= m_p.minValidDepth_m ||
        !std::isfinite(m_p.sam3HeightTolerance_m) || m_p.sam3HeightTolerance_m < 0.0f ||
        !std::isfinite(m_p.sam3EdgeThreshold) || m_p.sam3EdgeThreshold < 0.0f ||
        !std::isfinite(m_p.sam3MinWidthStd_px) || m_p.sam3MinWidthStd_px < 0.0f ||
        !std::isfinite(m_p.sam3SigmaKeep) || m_p.sam3SigmaKeep <= 0.0f ||
        m_p.minSegmentLen_px < 1 || m_p.sam3RowSearchWindow_px < 0 ||
        m_p.growthDirWindowR_px < 1) return out;

    const cv::Mat Kd = cam.K_depth();
    if (!cv::checkRange(Kd) || Kd.at<float>(0, 0) <= 0.0f ||
        Kd.at<float>(1, 1) <= 0.0f) return out;
    const cv::Mat KdInv = cam.K_depth_inv();
    const int width = depth.cols, height = depth.rows;
    const cv::Point base = frame.rgbToDepth(baseTap_full);
    const float baseDepth = frame.depthAtDepthPx(base.x, base.y);
    if (!std::isfinite(baseDepth) || baseDepth <= m_p.minValidDepth_m) return out;
    const cv::Point3f Pb = CameraModel::pixelTo3D(KdInv,
        (float)base.x, (float)base.y, baseDepth);
    if (!finitePointSAM3(Pb)) return out;

    // Initial vertical estimate, followed by the notebook's negative-row fold.
    const float BH = m_p.breastHeight_m;
    const cv::Point3f guess(Pb.x, Pb.y - BH, Pb.z);
    const cv::Point2f projectedGuess = CameraModel::projectToPixel(Kd, guess);
    if (!std::isfinite(projectedGuess.x) || !std::isfinite(projectedGuess.y)) return out;
    double guessRow = projectedGuess.y;
    if (guessRow < 0.0) {
        guessRow = -guessRow;
        // The notebook uses 200 and 256 for its 256-row depth images.
        if (guessRow >= height * (200.0 / 256.0)) guessRow = height - guessRow;
    }
    const int predictedRow = (int)std::lround(std::max(0.0,
        std::min((double)(height - 1), guessRow)));
    const int predictedCol = (int)std::lround(std::max(0.0,
        std::min((double)(width - 1), (double)projectedGuess.x)));

    // The notebook deliberately disables z_range filtering: use only depth.
    auto validDepth = [&](float value) {
        return std::isfinite(value) && value > m_p.minValidDepth_m &&
            value < m_p.sam3MaxDepth_m;
    };
    auto chooseOnRow = [&](int row, int& col) {
        const auto segments = rowSegmentsByPredicate(row, width,
            [&](int u) { return validDepth(depth.at<float>(row, u)); });
        // Compare floating centers before truncation, as np.mean/int do.
        double bestDistance = std::numeric_limits<double>::max();
        col = -1;
        for (const auto& segment : segments) {
            if (segment.second - segment.first + 1 < m_p.minSegmentLen_px) continue;
            const double center = 0.5 * (segment.first + segment.second);
            const double distance = std::abs(center - base.x);
            if (distance < bestDistance) {
                bestDistance = distance;
                col = (int)center;
            }
        }
        return col >= 0;
    };

    int selectedRow = predictedRow, selectedCol = -1;
    bool found = chooseOnRow(selectedRow, selectedCol);
    const int searchRadius = std::min(m_p.sam3RowSearchWindow_px, height - 1);
    // Keep each candidate's column and row together; the Python loop can keep
    // a stale column from a previous row when row_window is nonzero.
    for (int offset = 1; offset <= searchRadius && !found; ++offset) {
        for (int sign : { -1, 1 }) {
            const int row = predictedRow + sign * offset;
            if (row < 0 || row >= height) continue;
            if (chooseOnRow(row, selectedCol)) {
                selectedRow = row;
                found = true;
                break;
            }
        }
    }
    if (!found) {
        selectedCol = predictedCol;
        selectedRow = predictedRow;
    }
    const float selectedDepth = depth.at<float>(selectedRow, selectedCol);
    if (!validDepth(selectedDepth)) return out;
    const cv::Point3f rough = CameraModel::pixelTo3D(KdInv,
        (float)selectedCol, (float)selectedRow, selectedDepth);
    if (!finitePointSAM3(rough)) return out;

    const auto centers = trunkCentersSAM3(depth, cv::Point(selectedCol, selectedRow),
        std::min(m_p.growthDirWindowR_px, height - 1), m_p.sam3EdgeThreshold,
        m_p.sam3MinWidthStd_px, m_p.sam3SigmaKeep);
    std::vector<cv::Point3f> centers3D;
    centers3D.reserve(centers.size());
    for (const auto& center : centers) {
        const int col = (int)center.x, row = (int)center.y;
        const float value = depth.at<float>(row, col);
        if (!validDepth(value)) continue;
        const cv::Point3f point = CameraModel::pixelTo3D(KdInv,
            (float)col, (float)row, value);
        if (finitePointSAM3(point)) centers3D.push_back(point);
    }
    if (centers3D.size() < 2) return out;

    cv::Vec3f direction = fitLineDirectionPCA(centers3D);
    const double directionNorm = cv::norm(direction);
    if (!std::isfinite(directionNorm) || directionNorm <= 1e-6) return out;
    direction *= (float)(1.0 / directionNorm);
    if (direction[1] > 0.0f) direction *= -1.0f;

    // Same single correction as Step 00; within 1.5 cm keep the rough point.
    const double difference = cv::norm(rough - Pb) - BH;
    if (!std::isfinite(difference)) return out;
    cv::Point3f refined = rough;
    if (std::abs(difference) > m_p.sam3HeightTolerance_m) {
        refined.x -= (float)(difference * direction[0]);
        refined.y -= (float)(difference * direction[1]);
        refined.z -= (float)(difference * direction[2]);
    }
    if (!finitePointSAM3(refined) || refined.z <= 1e-6f) return out;
    const cv::Point2f uv = CameraModel::projectToPixel(Kd, refined);
    if (!std::isfinite(uv.x) || !std::isfinite(uv.y) || uv.x < 0.0f || uv.y < 0.0f)
        return out;

    // Match localize(): round/clamp the depth pixel, then map to full resolution.
    const cv::Point pixel(
        (int)std::lround(std::min((double)(width - 1), (double)uv.x)),
        (int)std::lround(std::min((double)(height - 1), (double)uv.y)));
    const cv::Point full = frame.depthToRgb(pixel);
    out.col_full = full.x;
    out.row_full = full.y;
    return out;
}

PixelLoc BreastHeightLocalizer::localize_SAM3(const Frame& frame, const CameraModel& cam,
    const cv::Mat& treeMask, const cv::Point2i& baseTap_full)
{
    PixelLoc out;
    if (!sam3::validGeometry(frame, cam) || !sam3::inside(baseTap_full, frame.rgb().size()) ||
        !std::isfinite(m_p.breastHeight_m) || m_p.breastHeight_m <= 0.0f ||
        !std::isfinite(m_p.minValidDepth_m) || m_p.minValidDepth_m < 0.0f ||
        !std::isfinite(m_p.sam3MaxDepth_m) || m_p.sam3MaxDepth_m <= m_p.minValidDepth_m ||
        m_p.sam3MaskRowSearchRadius_px < 0) return out;
    const cv::Mat mask = sam3::maskAtRgbSize(treeMask, frame.rgb().size());
    if (mask.empty()) return out;
    const cv::Mat depth = frame.depthRGB();
    const cv::Mat& K = cam.K_full();
    const cv::Mat Kinv = cam.K_full_inv();
    auto validDepth = [&](float value) {
        return std::isfinite(value) && value > m_p.minValidDepth_m && value < m_p.sam3MaxDepth_m;
    };

    // DBHprocessViaSAM3.ipynb: row spans of the already selected tree mask,
    // followed by a 3D direction fit using valid center depths.
    std::vector<int> rowCenters(mask.rows, -1);
    std::vector<cv::Point3f> points3D;
    cv::Point firstCenter(-1, -1);
    cv::Point3f centroid(0, 0, 0);
    for (int row = 0; row < mask.rows; ++row) {
        const uchar* values = mask.ptr<uchar>(row);
        int left = 0, right = mask.cols - 1;
        while (left < mask.cols && values[left] == 0) ++left;
        while (right >= left && values[right] == 0) --right;
        if (right <= left) continue;
        const int center = (int)std::lround(0.5 * (left + right));
        if (!values[center] || !validDepth(depth.at<float>(row, center))) continue;
        const cv::Point3f point = CameraModel::pixelTo3D(Kinv, (float)center,
            (float)row, depth.at<float>(row, center));
        if (!finitePointSAM3(point)) continue;
        rowCenters[row] = center;
        if (firstCenter.x < 0) firstCenter = cv::Point(center, row);
        points3D.push_back(point);
        centroid += point;
    }
    if (points3D.size() < 2) return out;
    centroid *= 1.0f / (float)points3D.size();
    cv::Vec3f direction = fitLineDirectionPCA(points3D);
    const double norm = cv::norm(direction);
    if (!std::isfinite(norm) || norm <= 1e-6) return out;
    direction *= (float)(1.0 / norm);
    if (direction[1] > 0) direction *= -1.0f;

    // Project the 3D direction at the centroid. The perspective derivative gives
    // the same image-line slope as projecting two points, without rounding them.
    const double du = K.at<float>(0, 0) *
        ((double)direction[0] * centroid.z - (double)centroid.x * direction[2]);
    const double dv = K.at<float>(1, 1) *
        ((double)direction[1] * centroid.z - (double)centroid.y * direction[2]);
    if (!std::isfinite(du) || !std::isfinite(dv) || std::abs(dv) < 1e-6) return out;

    // Keep the tapped base row and move its column to the fitted trunk axis.
    const double fittedU = firstCenter.x + (baseTap_full.y - firstCenter.y) * du / dv;
    if (!std::isfinite(fittedU) || fittedU < 0 || fittedU > mask.cols - 1) return out;
    const cv::Point fittedBase((int)std::lround(fittedU), baseTap_full.y);
    const float baseDepth = depth.at<float>(fittedBase);
    if (!validDepth(baseDepth)) return out;
    const cv::Point3f base3D = CameraModel::pixelTo3D(Kinv,
        (float)fittedBase.x, (float)fittedBase.y, baseDepth);
    const cv::Point3f bh3D = base3D + cv::Point3f(direction[0], direction[1], direction[2]) * m_p.breastHeight_m;
    if (!finitePointSAM3(bh3D) || bh3D.z <= 1e-6f) return out;
    const cv::Point2f projected = CameraModel::projectToPixel(K, bh3D);
    if (!std::isfinite(projected.x) || !std::isfinite(projected.y) ||
        projected.x < 0 || projected.x > mask.cols - 1 ||
        projected.y < 0 || projected.y > mask.rows - 1) return out;
    const int projectedRow = (int)std::lround(projected.y);

    auto selectRow = [&](int row) {
        if (row < 0 || row >= mask.rows || rowCenters[row] < 0) return false;
        out.col_full = rowCenters[row];
        out.row_full = row;
        return true;
    };
    if (selectRow(projectedRow)) return out;
    const int radius = std::min(m_p.sam3MaskRowSearchRadius_px, mask.rows - 1);
    // Notebook order: projected row, all rows above, then rows below.
    for (int offset = 1; offset <= radius; ++offset)
        if (selectRow(projectedRow - offset)) return out;
    for (int offset = 1; offset <= radius; ++offset)
        if (selectRow(projectedRow + offset)) return out;
    return out;
}
