#pragma once
#include <string>
#include <sstream>
#include <iomanip>

// All algorithm parameters (no magic numbers in algorithm code)
struct BHLocalizerConfig
{
    float breastHeight_m = 1.4f;
    float minValidDepth_m = 0.01f;
    float theta0_deg = 30.0f;
    int minSegmentLen_px = 5;
    int rowSearchWindow_px = 10;
    int growthDirWindowR_px = 20;

    // Step 00 / localize_SAM3 defaults (depth-image pixels and meters).
    int sam3RowSearchWindow_px = 0;
    float sam3MaxDepth_m = 5.0f;
    float sam3EdgeThreshold = 0.1f;
    float sam3HeightTolerance_m = 0.015f;
    float sam3MinWidthStd_px = 0.5f;
    float sam3SigmaKeep = 2.0f;
    int sam3MaskRowSearchRadius_px = 30; // FULL-res mask-localization rows
};

struct DBHMeasurerConfig
{
    int halfWindowSize = 20;
    float threshold = 0.1f;
    float minStdRowLen_px = 0.5f;
    float sigmaKeepK = 2.0f;
    float depthThreshold_m = 5.0;
    float extension_rate = 0.1f;
    float top_lift = 150.0f;
    float base_lift = 150.0f;
    int verticalSearchHalfRange = 100;
    float widthRatio = 0.5f;
    int sam3VerticalSearchHalfRange = 50; // FULL-res; matches measure()'s mask band
    // DBH_DEBUG enables SAM3 step logs. Set this false for console-only debugging.
    // Debug windows refresh without pausing; the test caller handles its final wait.
    bool sam3DebugShowWindows = true;
};

extern bool DBH_DEBUG;
extern DBHMeasurerConfig g_DBHConfig;
extern BHLocalizerConfig g_BHConfig;

#pragma once
