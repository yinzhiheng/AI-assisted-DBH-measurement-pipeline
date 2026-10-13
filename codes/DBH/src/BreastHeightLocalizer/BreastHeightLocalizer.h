#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <functional>
#include "CameraModel.h"
#include "DepthFrame.h"
#include "configs.h"

// Output pixel location in full-res coordinates.
struct PixelLoc
{
    int col_full = -1;
    int row_full = -1;
    bool isValid() const { return col_full >= 0 && row_full >= 0; }
};

class BreastHeightLocalizer
{
public:
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

        // SAM3 uses the notebook's single predicted row by default. A nonzero
        // window tries nearby rows if that row has no qualifying depth segment.
        int sam3RowSearchWindow_px = g_BHConfig.sam3RowSearchWindow_px;
        float sam3MaxDepth_m = g_BHConfig.sam3MaxDepth_m;
        float sam3EdgeThreshold = g_BHConfig.sam3EdgeThreshold;
        float sam3HeightTolerance_m = g_BHConfig.sam3HeightTolerance_m;
        float sam3MinWidthStd_px = g_BHConfig.sam3MinWidthStd_px;
        float sam3SigmaKeep = g_BHConfig.sam3SigmaKeep;

        // Mask-based overload: search in FULL-res rows, as in DBHprocessViaSAM3.
        int sam3MaskRowSearchRadius_px = g_BHConfig.sam3MaskRowSearchRadius_px;
    };

    explicit BreastHeightLocalizer(Params p = Params()) : m_p(p) {}

    // Input tap is in FULL-res coordinates. Output is FULL-res.
    // Preserve the original seed when it has depth edges. Otherwise retry without
    // image-border depth regions, accepting the retry only if it has an edge pair.
    PixelLoc localize(const Frame& frame, const CameraModel& cam, const cv::Point2i& baseTap_full);

    // Port of fusedMethod.ipynb, Step 00: find_breast_height_pixel_from_base.
    // Uses depth edges, filtered trunk centers and an upward 3D PCA direction;
    // no SAM model or mask is required. Input and output are FULL-res pixels.
    // Returns the default invalid PixelLoc (-1, -1) when localization fails.
    // Shares breastHeight_m, minValidDepth_m, minSegmentLen_px and
    // growthDirWindowR_px with localize(); theta0_deg is unused because the
    // notebook disables the tilt-depth filter. SAM3-specific settings are above.
    PixelLoc localize_SAM3(const Frame& frame, const CameraModel& cam, const cv::Point2i& baseTap_full);

    // Preloaded mask of one tree, aligned with the original RGB image. Accepts
    // binary/grayscale or color masks; matching-aspect masks are resized with
    // nearest-neighbor interpolation. Fits the mask centerline in 3D, moves
    // breastHeight_m from the fitted base, then selects a nearby mask center.
    PixelLoc localize_SAM3(const Frame& frame, const CameraModel& cam,
        const cv::Mat& treeMask, const cv::Point2i& baseTap_full);

private:
    Params m_p;
    PixelLoc localizeImpl(const Frame& frame, const CameraModel& cam,
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
