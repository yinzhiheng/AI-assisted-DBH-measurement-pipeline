#pragma once
#include <opencv2/core.hpp>
#include <vector>
#include <sstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include "CameraModel.h"
#include "DepthFrame.h"
#include "sam.hpp"

#include "configs.h"

struct DiameterResultCPP
{
    // endpoints in FULL-res pixel coordinates (for drawing)
    cv::Point2i A_full{ -1,-1 };
    cv::Point2i B_full{ -1,-1 };

    // endpoints in DEPTH-res  (for debugging)
    cv::Point2f A_depth{ -1.f,-1.f };
    cv::Point2f B_depth{ -1.f,-1.f };

    // trunk direction in DEPTH-res (unit vector)
    cv::Vec2f growthDirUV{ 0.f, 0.f };
    
    cv::Mat Mask;

    float distanceToCamera_m = 0.0f;
    float chordLength_m = 0.0f;
    float diameter_m = 0.0f;

    bool ok = false;
};

struct EdgeRow
{
    float l = 0.f;
    int   row = 0;
    float r = 0.f;
};

struct CleanEdgeResult
{
    std::vector<EdgeRow> edgeListCleaned;
    std::vector<cv::Point2f> centerListCleaned; // (u_center, v=row)
    bool ok = false;
};

struct RotationMatrices {
    cv::Mat M_forward;      // 2x3, CV_64F
    cv::Mat M_backward;     // 2x3, CV_64F  (inverse in 2x3 form)
    cv::Size outSize;       // (new_w, new_h)
};

struct BHCheckResult
{
    bool ok = false;
    cv::Point2f updatedBreastHeightPtRGB{ -1.f, -1.f };
};

struct RuntimeSegmentationResult;

class DBHMeasurer
{
public:
    RuntimeSegmentationResult segmentForRuntime(const Frame&, const CameraModel&, Sam&, const cv::Point2i&);

    // Important dependency: provide trunk mask extracted by TrunkMaskExtractor.
    DiameterResultCPP measure_depth(const Frame& frame, const CameraModel& cam, const cv::Point2i& breastHeightPtRGB);
    DiameterResultCPP measure(const Frame& frame, const CameraModel& cam, Sam& sam, bool needPreprocess, const cv::Point2i& breastHeightPtRGB);
    DiameterResultCPP measure_point(const Frame& frame, const CameraModel& cam, Sam& sam, bool needPreprocess, const cv::Point2i& breastHeightPtTappedRGB);

private:

    // ------------------------------
    static inline float Deg(float rad) { return rad * 180.0f / (float)CV_PI; }

    static inline float Mean(const std::vector<float>& v)
    {
        if (v.empty()) return 0.f;
        double s = 0.0;
        for (float x : v) s += x;
        return (float)(s / (double)v.size());
    }

    static inline float Std(const std::vector<float>& v, float mean)
    {
        if (v.empty()) return 0.f;
        double var = 0.0;
        for (float x : v) {
            double d = (double)x - (double)mean;
            var += d * d;
        }
        var /= (double)v.size(); // numpy default ddof=0
        return (float)std::sqrt(var);
    }

    // Functions for Depth-only methods
    cv::Point3f LidarPoint2XYZ(const cv::Point2i& pointuv, float depth, const cv::Mat& K);
    cv::Point3f RGBPoint2XYZ(const cv::Point2i& pointuv, float depth, const cv::Mat& K);
    cv::Point2i XYZ2LidarPoint(const cv::Point3f& pointXYZ, const cv::Mat& K);
    cv::Point2i XYZ2RGBPoint(const cv::Point3f& pointXYZ, const cv::Mat& K);

    bool FitLine2D_X_as_aY_plus_b(const std::vector<float>& y, const std::vector<float>& x, float& a_out, float& b_out);
    bool FitLine3D_SVD(const std::vector<cv::Point3f>& points, cv::Point3f& centroid, cv::Vec3f& direction_unit);

    cv::Mat EdgeListToMask(const std::vector<EdgeRow>& edgeList, int imgHeight, int imgWidth);

    cv::Mat ComputeGradientXY(const cv::Mat& depthMap);

    std::vector<EdgeRow> FindEdgeList(const cv::Mat& gradientXY, const cv::Point2f& startingCenterPtuv, int halfWindowSize = g_DBHConfig.halfWindowSize, float threshold = g_DBHConfig.threshold);

    CleanEdgeResult CleanEdgeListEdgeLengthCenter( const std::vector<EdgeRow>& edgeList, float minStdRowLen_px = g_DBHConfig.minStdRowLen_px, float sigmaKeepK = g_DBHConfig.sigmaKeepK);

    bool ComputeGrowDirection(const std::vector<cv::Point2f>& centerList, const cv::Mat& depthMap, const cv::Mat& K, 
        cv::Vec3f& growthDirXYZ_unit, cv::Vec2f& growthDirUV_unit, cv::Vec3f& centroidXYZ);

    bool FitUVPlane(const std::vector<cv::Point>& pointsUV, const cv::Mat& depthMap, const cv::Mat& cameraMatrix, cv::Matx<float, 9, 1>& planeFunction_out);

    cv::Point3f FindXYZFromFittedUVPlanes(const cv::Point2f& pointuv, const cv::Matx<float, 9, 1>& planeFunction);

    bool GetPointsForPlaneFitting(const cv::Mat& depthMap, const cv::Mat& mask, const cv::Mat& cameraMatrix, const std::vector<cv::Point2f>& centerPtList, 
        std::vector<cv::Point>& uvPointsList_out, std::vector<cv::Point3f>& xyzPointsList_out, float depthThreshold_m = g_DBHConfig.depthThreshold_m, float estimatedDBH_m = -1);

    float ComputeChordLength(const std::vector<EdgeRow>& edgeList);

    bool Compute3DChordLength(const cv::Point2f& centroiduv, const cv::Vec2f& growthDirUV, float chordLength_px, const cv::Mat& depthMap, const cv::Mat& cameraMatrix,
        const cv::Matx<float, 9, 1>& planeFuncUV, cv::Point& leftEdgePointUV_out, cv::Point& rightEdgePointUV_out, float& chordLengthInMeterUV_out);

    float ComputeCentroidDistance(const cv::Point3f& centroidxyz);

    float ComputeDBHInMeter(float distanceInMeter, float chordLengthInMeter);

    // Functions for Depth-guided methods
    RotationMatrices computeRotationMatrixFromGrowthDirection(const cv::Vec2d& growthDirUV, const cv::Point2d& rotationCenter, int imageHeight, int imageWidth);

    cv::Mat rotateImage(const cv::Mat& image, const cv::Mat& M, const cv::Size& outSize);
    
    //cv::Point2f rotatePointAroundCenter(const cv::Point2f& ptUV, const cv::Mat& M);

    cv::Rect createSAMBoxRect(const cv::Point2f& breastHeightPt, const cv::Point2f& basePt, float chordLengthInPixel, int imageHeight, int imageWidth,
        float extension_rate = g_DBHConfig.extension_rate, float top_lift = g_DBHConfig.top_lift, float base_lift = g_DBHConfig.base_lift);

    cv::Rect scaleBoxToSamInput(const cv::Rect& box_in, const cv::Size& srcSize, const cv::Size& dstSize);
    cv::Mat SAMBoxSegmentation(Sam& sam, const cv::Mat& image_bgr, const cv::Rect& box_xywh, bool* terminated /*=nullptr*/, bool needPreprocess);
    cv::Point scalePointToSamInput(const cv::Point& p_full, const cv::Size& fullSize, const cv::Size& samSize);
    cv::Mat SAMPointSegmentation(Sam& sam, const cv::Mat& image_bgr, const cv::Point& point_full, bool* terminated /*=nullptr*/, bool needPreprocess);

    int clampInt(int v, int lo, int hi);
    cv::Point2f applyAffine2x3(const cv::Point2f& p, const cv::Mat& M2x3);
    cv::Point2f applyAffine2x3_f(const cv::Point2f& p, const cv::Mat& M2x3);
    cv::Point2f rotatePointAroundCenter(const cv::Point2f& p, const cv::Mat& M_backward);

    std::vector<EdgeRow> findRGBEdgePointListFromSAMMask(const cv::Mat& rotatedImage, const cv::Mat& maskBySAMOriginal, const cv::Point2f& breastHeightPtOriginal, int verticalSearchHalfRange = g_DBHConfig.verticalSearchHalfRange);
    cv::Mat findTreeTrunkFromSAMMask(const cv::Mat& maskBySAMOriginal, float widthRatio= g_DBHConfig.widthRatio);
    std::vector<EdgeRow> findEdgeListWithTrunkMask(const std::vector<EdgeRow>& edgeList, const cv::Mat& trunkMask);

    BHCheckResult CheckBHWithinFittedEdges(const std::vector<EdgeRow>& rawRGBEdgeListWithinTrunk, const cv::Point2f& breastHeightPtRGB);

};
