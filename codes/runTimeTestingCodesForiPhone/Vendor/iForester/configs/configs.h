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
};

extern bool DBH_DEBUG;
extern DBHMeasurerConfig g_DBHConfig;
extern BHLocalizerConfig g_BHConfig;

#pragma once
