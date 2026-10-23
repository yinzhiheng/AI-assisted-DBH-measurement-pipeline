#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <functional>
#include "CameraModel.h"
#include "DepthFrame.h"
#include "configs.h"

// Output pixel location in full-res coordinates.
struct PixelLoc_cpp
{
    int col_full = -1;
    int row_full = -1;
    bool isValid() const { return col_full >= 0 && row_full >= 0; }
};

struct Params
{
    // Breast height in meters
    float breastHeight_m = g_BHConfig.breastHeight_m;

    // Valid depth threshold (meters)
    float minValidDepth_m = g_BHConfig.minValidDepth_m;

    // Depth-range filter angle around base depth
    float theta0_deg = g_BHConfig.theta0_deg;

    // Segment selection
    int minSegmentLen_px = g_BHConfig.minSegmentLen_px;

    // If predicted row fails, search +/- rows
    int rowSearchWindow_px = g_BHConfig.rowSearchWindow_px;

    // PCA window radius for growth direction
    int growthDirWindowR_px = g_BHConfig.growthDirWindowR_px;
};

class BreastHeightLocalizer
{
public:
    explicit BreastHeightLocalizer(Params p = Params()) : m_p(p) {}

    // Input tap is in FULL-res coordinates. Output is FULL-res.
    // Preserve the original seed when it has depth edges. Otherwise retry without
    // image-border depth regions, accepting the retry only if it has an edge pair.
    PixelLoc_cpp localize(const Frame& frame, const CameraModel& cam, const cv::Point2i& baseTap_full);

private:
    Params m_p;
    PixelLoc_cpp localizeImpl(const Frame& frame, const CameraModel& cam,
        const cv::Point2i& baseTap_full, bool requireBoundedSegments);

    static inline int clampi(int x, int lo, int hi)
    {
        return (x < lo) ? lo : (x > hi ? hi : x);
    }

    // Split one row into contiguous segments that satisfy predicate. Segments are inclusive [start,end].
    static std::vector<std::pair<int, int>> rowSegmentsByPredicate(int row, int cols, const std::function<bool(int c)>& isValid);

    // Choose segment whose center is closest to targetU, subject to min length.
    static bool chooseClosestSegment(const std::vector<std::pair<int, int>>& segs, int targetU, int minLenPx, int& outCenterU, std::pair<int, int>* outSeg = nullptr);

    // PCA direction fit for a 3D line direction
    static cv::Vec3f fitLineDirectionPCA(const std::vector<cv::Point3f>& pts);
};
