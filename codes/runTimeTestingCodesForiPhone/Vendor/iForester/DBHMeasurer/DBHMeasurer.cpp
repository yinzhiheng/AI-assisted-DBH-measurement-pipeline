#include "DBHMeasurer.h"
#include "TrunkMaskRows.h"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <limits>

DiameterResultCPP DBHMeasurer::measure_depth(
    const Frame& frame,
    const CameraModel& cam,
    const cv::Point2i& breastHeightPtRGB)
{
    DiameterResultCPP out;

    const cv::Mat& depthMapLidar = frame.depth();
    if (depthMapLidar.empty() || depthMapLidar.type() != CV_32F)
    {
        std::cout << "[DBHMeasurer] depthMap invalid.\n";
        return out;
    }

    const cv::Mat Kd = cam.K_depth();

    // breastHeightPtLidar in depth-res
    const cv::Point2i breastHeightPtLidar = frame.rgbToDepth(breastHeightPtRGB);


    cv::Mat gradientXY = ComputeGradientXY(depthMapLidar);
    std::vector<EdgeRow> edgeList = FindEdgeList(gradientXY, cv::Point2f((float)breastHeightPtLidar.x, (float)breastHeightPtLidar.y));

    if (edgeList.empty())
    {
        std::cout << "[DBHMeasurer] Cannot find edgeList.\n";
        return out;
    }

    CleanEdgeResult cleanedEdgeInfo;
    cleanedEdgeInfo = CleanEdgeListEdgeLengthCenter(edgeList);

    std::vector<EdgeRow> edgeListcleaned = cleanedEdgeInfo.edgeListCleaned;
    std::vector<cv::Point2f> centerListCleaned = cleanedEdgeInfo.centerListCleaned;
    bool isSuccessfullyCleaned = cleanedEdgeInfo.ok;

    if (!isSuccessfullyCleaned)
    {
        std::cout << "[DBHMeasurer] Depth map too bad. Continue.\n";
        return out;
    }

    cv::Mat maskBydepthMethod = EdgeListToMask(edgeListcleaned, depthMapLidar.rows, depthMapLidar.cols);

    cv::Vec3f growthDirXYZ(0, 0, 0);
    cv::Vec2f growthDirUV(0, 0);
    cv::Vec3f centroidXYZ(0, 0, 0);

    if (!ComputeGrowDirection(centerListCleaned, depthMapLidar, Kd, growthDirXYZ, growthDirUV, centroidXYZ))
    {
        std::cout << "[DBHMeasurer] ComputeGrowDirection failed.\n";
        return out;
    }

    // store n2D (depth-res) for visualization/debug
    out.growthDirUV = growthDirUV;

    const cv::Point2i centroiduv = XYZ2LidarPoint(centroidXYZ, Kd);
    if (DBH_DEBUG) {
        // ---------------- DEBUG PRINT ----------------
        std::cout << "[Centroid Projection DEBUG]\n";
        std::cout << "  centroidXYZ = ("
            << centroidXYZ[0] << ", "
            << centroidXYZ[1] << ", "
            << centroidXYZ[2] << ")\n";
        std::cout << "  centroiduv  = (u=" << centroiduv.x
            << ", v=" << centroiduv.y << ")\n";
        std::cout << "--------------------------------\n";
    }
   

    const float chordLengthInPixel = ComputeChordLength(edgeListcleaned);
    if (!(chordLengthInPixel > 0.0f))
    {
        std::cout << "[DBHMeasurer] chordLengthInPixel invalid.\n";
        return out;
    }

    // Plane Fitting
    std::vector<cv::Point2i> uvPointsToFit;
    std::vector<cv::Point3f> xyzPointsToFit;

    GetPointsForPlaneFitting(depthMapLidar, maskBydepthMethod, Kd, centerListCleaned, uvPointsToFit, xyzPointsToFit);

    if (uvPointsToFit.size() < 3)
    {
        std::cout << "[DBHMeasurer] Not enough uvPointsToFit for plane.\n";
        return out;
    }

    cv::Matx<float, 9, 1> planeFuncUV;
    FitUVPlane(uvPointsToFit, depthMapLidar, Kd, planeFuncUV);

    // ChordLength calculation (in meter)
    cv::Point2i leftEdgePointUV(-1, -1);
    cv::Point2i rightEdgePointUV(-1, -1);
    float chordLengthInMeterXYZ = 0.0f;
    float chordLengthInMeterUV = 0.0f;

    if (!Compute3DChordLength(centroiduv, growthDirUV, chordLengthInPixel, depthMapLidar, Kd, planeFuncUV, leftEdgePointUV, rightEdgePointUV, chordLengthInMeterUV))
    {
        std::cout << "[DBHMeasurer] Compute3DChordLength failed.\n";
        return out;
    }

    float distanceToCameraInMeter = ComputeCentroidDistance(centroidXYZ);
    if (!(distanceToCameraInMeter > 0.0f))
    {
        std::cout << "[DBHMeasurer] distanceToCameraInMeter invalid.\n";
        return out;
    }

    float dbhInMeterUV = ComputeDBHInMeter(distanceToCameraInMeter, chordLengthInMeterUV);
    if (!(dbhInMeterUV > 0.0f))
    {
        std::cout << "[DBHMeasurer] DBH invalid.\n";
        return out;
    }

    // Update the fitted plane after the initial DBH estimation
    GetPointsForPlaneFitting(depthMapLidar, maskBydepthMethod, Kd, centerListCleaned, uvPointsToFit, xyzPointsToFit);

    if (uvPointsToFit.size() < 3)
    {
        std::cout << "[DBHMeasurer] Not enough uvPointsToFit for plane.\n";
        return out;
    }

    FitUVPlane(uvPointsToFit, depthMapLidar, Kd, planeFuncUV);

    if (!Compute3DChordLength(centroiduv, growthDirUV, chordLengthInPixel, depthMapLidar, Kd, planeFuncUV, leftEdgePointUV, rightEdgePointUV, chordLengthInMeterUV))
    {
        std::cout << "[DBHMeasurer] Compute3DChordLength failed.\n";
        return out;
    }

    distanceToCameraInMeter = ComputeCentroidDistance(centroidXYZ);
    if (!(distanceToCameraInMeter > 0.0f))
    {
        std::cout << "[DBHMeasurer] distanceToCameraInMeter invalid.\n";
        return out;
    }

    dbhInMeterUV = ComputeDBHInMeter(distanceToCameraInMeter, chordLengthInMeterUV);
    if (!(dbhInMeterUV > 0.0f))
    {
        std::cout << "[DBHMeasurer] DBH invalid.\n";
        return out;
    }


    // Write output variable
    out.A_depth = cv::Point2f((float)leftEdgePointUV.x, (float)leftEdgePointUV.y);
    out.B_depth = cv::Point2f((float)rightEdgePointUV.x, (float)rightEdgePointUV.y);

    out.distanceToCamera_m = distanceToCameraInMeter;
    out.chordLength_m = chordLengthInMeterUV;     // per your updated python (UV-based)
    out.diameter_m = dbhInMeterUV;
    out.ok = true;

    // convert to full-res endpoints for drawing
    out.A_full = frame.depthToRgb(leftEdgePointUV);
    out.B_full = frame.depthToRgb(rightEdgePointUV);

    if (DBH_DEBUG) {
        std::cout << "[DBHMeasurer] breastHeightPtLidar=" << breastHeightPtLidar << "\n";
        std::cout << "[DBHMeasurer] centroiduv=" << centroiduv << "\n";
        std::cout << "[DBHMeasurer] growthDirUV=" << growthDirUV << "\n";
        std::cout << "[DBHMeasurer] A_depth=" << out.A_depth << "  B_depth=" << out.B_depth << "\n";
        std::cout << "[DBHMeasurer] chord_px=" << chordLengthInPixel
            << " chord_m_UV=" << chordLengthInMeterUV
            << " dist_m=" << distanceToCameraInMeter
            << " dbh_m=" << dbhInMeterUV << "\n";
    }

    return out;
}

DiameterResultCPP DBHMeasurer::measure(
    const Frame& frame,
    const CameraModel& cam,
    Sam& sam,
    bool needPreprocess,
    const cv::Point2i& breastHeightPtTappedRGB)
{
    DiameterResultCPP out;

    const cv::Mat& depthMapLidar = frame.depth();
    if (depthMapLidar.empty() || depthMapLidar.type() != CV_32F)
    {
        std::cout << "[DBHMeasurer] depthMap invalid.\n";
        return out;
    }

    const cv::Mat Kd = cam.K_depth();

    // breastHeightPtLidar in depth-res
    const cv::Point2i breastHeightPtLidar = frame.rgbToDepth(breastHeightPtTappedRGB);


    cv::Mat gradientXY = ComputeGradientXY(depthMapLidar);
    std::vector<EdgeRow> edgeList = FindEdgeList(gradientXY, cv::Point2f((float)breastHeightPtLidar.x, (float)breastHeightPtLidar.y));

    if (edgeList.empty())
    {
        std::cout << "[DBHMeasurer] Cannot find edgeList.\n";
        return out;
    }

    CleanEdgeResult cleanedEdgeInfo;
    cleanedEdgeInfo = CleanEdgeListEdgeLengthCenter(edgeList);

    std::vector<EdgeRow> edgeListcleaned = cleanedEdgeInfo.edgeListCleaned;
    std::vector<cv::Point2f> centerListCleaned = cleanedEdgeInfo.centerListCleaned;
    bool isSuccessfullyCleaned = cleanedEdgeInfo.ok;

    if (!isSuccessfullyCleaned)
    {
        std::cout << "[DBHMeasurer] Depth map too bad. Continue.\n";
        return out;
    }

    cv::Mat maskBydepthMethod = EdgeListToMask(edgeListcleaned, depthMapLidar.rows, depthMapLidar.cols);

    cv::Vec3f growthDirXYZ(0, 0, 0);
    cv::Vec2f growthDirUV(0, 0);
    cv::Vec3f centroidXYZ(0, 0, 0);

    if (!ComputeGrowDirection(centerListCleaned, depthMapLidar, Kd, growthDirXYZ, growthDirUV, centroidXYZ))
    {
        std::cout << "[DBHMeasurer] ComputeGrowDirection failed.\n";
        return out;
    }

    // store n2D (depth-res) for visualization/debug
    out.growthDirUV = growthDirUV;

    const cv::Point2i centroiduv = XYZ2LidarPoint(centroidXYZ, Kd);
    if (DBH_DEBUG) {
        std::cout << "[Centroid Projection DEBUG]\n";
        std::cout << "  centroidXYZ = ("
            << centroidXYZ[0] << ", "
            << centroidXYZ[1] << ", "
            << centroidXYZ[2] << ")\n";
        std::cout << "  centroiduv  = (u=" << centroiduv.x
            << ", v=" << centroiduv.y << ")\n";
        std::cout << "--------------------------------\n";
    }
    

    const float chordLengthInPixel = ComputeChordLength(edgeListcleaned);
    if (!(chordLengthInPixel > 0.0f))
    {
        std::cout << "[DBHMeasurer] chordLengthInPixel invalid.\n";
        return out;
    }

    // Plane Fitting
    std::vector<cv::Point2i> uvPointsToFit;
    std::vector<cv::Point3f> xyzPointsToFit;

    GetPointsForPlaneFitting(depthMapLidar, maskBydepthMethod, Kd, centerListCleaned, uvPointsToFit, xyzPointsToFit);

    if (uvPointsToFit.size() < 3)
    {
        std::cout << "[DBHMeasurer] Not enough uvPointsToFit for plane.\n";
        return out;
    }

    cv::Matx<float, 9, 1> planeFuncUV;
    FitUVPlane(uvPointsToFit, depthMapLidar, Kd, planeFuncUV);

    // ChordLength calculation (in meter)
    cv::Point2i leftEdgePointUV(-1, -1);
    cv::Point2i rightEdgePointUV(-1, -1);
    float chordLengthInMeterXYZ = 0.0f;
    float chordLengthInMeterUV = 0.0f;

    if (!Compute3DChordLength(centroiduv, growthDirUV, chordLengthInPixel, depthMapLidar, Kd, planeFuncUV, leftEdgePointUV, rightEdgePointUV, chordLengthInMeterUV))
    {
        std::cout << "[DBHMeasurer] Compute3DChordLength failed.\n";
        return out;
    }

    float distanceToCameraInMeter = ComputeCentroidDistance(centroidXYZ);
    if (!(distanceToCameraInMeter > 0.0f))
    {
        std::cout << "[DBHMeasurer] distanceToCameraInMeter invalid.\n";
        return out;
    }

    float dbhInMeterUV = ComputeDBHInMeter(distanceToCameraInMeter, chordLengthInMeterUV);
    if (!(dbhInMeterUV > 0.0f))
    {
        std::cout << "[DBHMeasurer] DBH invalid.\n";
        return out;
    }

    // Update the fitted plane after the initial DBH estimation
    GetPointsForPlaneFitting(depthMapLidar, maskBydepthMethod, Kd, centerListCleaned, uvPointsToFit, xyzPointsToFit);

    if (uvPointsToFit.size() < 3)
    {
        std::cout << "[DBHMeasurer] Not enough uvPointsToFit for plane.\n";
        return out;
    }

    FitUVPlane(uvPointsToFit, depthMapLidar, Kd, planeFuncUV);

    if (!Compute3DChordLength(centroiduv, growthDirUV, chordLengthInPixel, depthMapLidar, Kd, planeFuncUV, leftEdgePointUV, rightEdgePointUV, chordLengthInMeterUV))
    {
        std::cout << "[DBHMeasurer] Compute3DChordLength failed.\n";
        return out;
    }

    distanceToCameraInMeter = ComputeCentroidDistance(centroidXYZ);
    if (!(distanceToCameraInMeter > 0.0f))
    {
        std::cout << "[DBHMeasurer] distanceToCameraInMeter invalid.\n";
        return out;
    }

    dbhInMeterUV = ComputeDBHInMeter(distanceToCameraInMeter, chordLengthInMeterUV);
    if (!(dbhInMeterUV > 0.0f))
    {
        std::cout << "[DBHMeasurer] DBH invalid.\n";
        return out;
    }

    // Start the SAM box processing below:
    // ---- 1) Prepare RGB image & aligned RGB-depth map ----
    const cv::Mat& rgbImage = frame.rgb();        
    const cv::Mat& depthMapRGB = frame.depthRGB();
    
//    if (rgbImage.empty() || rgbImage.type() != CV_8UC3)
//    {
//        std::cout << "[DBHMeasurer][SAM] rgbImage invalid.\n";
//        return out;
//    }
//    if (depthMapRGB.empty() || depthMapRGB.type() != CV_32F)
//    {
//        std::cout << "[DBHMeasurer][SAM] depthMapRGB invalid.\n";
//        return out;
//    }

    const int H = rgbImage.rows;
    const int W = rgbImage.cols;

    const cv::Size originalImageSize(W, H);
    const cv::Point2d rotationCenter((double)W * 0.5, (double)H * 0.5);

    // ---- 2) Scale depth-res points to RGB-res points ----
    const float scaleFromLidar2RGB = (float)W / (float)depthMapLidar.cols;

    // centroiduv is depth-res (cv::Point2i) from your depth branch
    const cv::Point2i centroiduv_depth = centroiduv;

    cv::Point2f breastHeightPtRGB(
        (float)centroiduv_depth.x * scaleFromLidar2RGB,
        (float)centroiduv_depth.y * scaleFromLidar2RGB
    );

    cv::Point2i basePt_depth = frame.depthBasePoint();
    cv::Point2f basePtRGB(
        (float)basePt_depth.x * scaleFromLidar2RGB,
        (float)basePt_depth.y * scaleFromLidar2RGB
    );

    // ---- 3) Compute rotation matrices from growthDirUV ----
    RotationMatrices R = computeRotationMatrixFromGrowthDirection(
        cv::Vec2d((double)growthDirUV[0], (double)growthDirUV[1]),
        rotationCenter, H, W
    );

    // ---- 4) Rotate RGB image ----
    cv::Mat rotatedImage = rotateImage(rgbImage, R.M_forward, R.outSize);

    // ---- 5) Rotate points ----
    cv::Point2f basePtRGBRotated = rotatePointAroundCenter(basePtRGB, R.M_forward);
    cv::Point2f breastHeightPtRGBRotated = rotatePointAroundCenter(breastHeightPtRGB, R.M_forward);

    // chordLengthInPixel is depth-res; scale to RGB pixels
    float chordLengthInPixelRGB = chordLengthInPixel * scaleFromLidar2RGB;

    // ---- 6) Create SAM box on rotated image ----
    cv::Rect samBoxOnRotatedImage = createSAMBoxRect(
        breastHeightPtRGBRotated, basePtRGBRotated, chordLengthInPixelRGB,
        rotatedImage.rows, rotatedImage.cols,
        /*extension_rate=*/0.15f, /*top_lift=*/300.0f, /*base_lift=*/150.0f
    );

    // ---- 7) SAM segmentation (box prompt) on rotated image ----
    cv::Mat maskBySAMRotated_u8 = SAMBoxSegmentation(sam, rotatedImage, samBoxOnRotatedImage, nullptr, needPreprocess);
    if (maskBySAMRotated_u8.empty())
    {
        std::cout << "[DBHMeasurer][SAM] SAMBoxSegmentation returned empty mask.\n";
        return out;
    }
    // normalize to {0,1} in CV_8U
    if (maskBySAMRotated_u8.type() != CV_8U)
        maskBySAMRotated_u8.convertTo(maskBySAMRotated_u8, CV_8U);
    cv::threshold(maskBySAMRotated_u8, maskBySAMRotated_u8, 0, 1, cv::THRESH_BINARY);

    // ---- 8) Rotate mask back to original image size ----
    // 1) Bring SAM mask from SAM input size -> rotatedImage canvas size
    cv::Mat maskOnRotatedCanvas_u8;
    cv::resize(maskBySAMRotated_u8, maskOnRotatedCanvas_u8, rotatedImage.size(),
        0, 0, cv::INTER_NEAREST);

    // 2) Convert to 0/255 for warp
    cv::Mat maskOnRotatedCanvas_255;
    maskOnRotatedCanvas_u8.convertTo(maskOnRotatedCanvas_255, CV_8U, 255.0);

    // 3) Rotate back to original RGB size using M_backward
    cv::Mat maskBySAMOriginal_255 = rotateImage(maskOnRotatedCanvas_255, R.M_backward, originalImageSize);
    if (DBH_DEBUG) {
        // ------------------ Debug: Rotated canvas visualization ------------------
        std::cout << "rotatedImage(canvas) size: "
            << rotatedImage.cols << " x "
            << rotatedImage.rows << "\n";

        std::cout << "maskOnRotatedCanvas_255 size: "
            << maskOnRotatedCanvas_255.cols << " x "
            << maskOnRotatedCanvas_255.rows << "\n";

        // Ensure mask matches rotated canvas
        cv::Mat maskCanvas_255;
        if (maskOnRotatedCanvas_255.size() != rotatedImage.size())
        {
            std::cout << "[INFO] Resizing mask to rotated canvas size\n";
            cv::resize(maskOnRotatedCanvas_255,
                maskCanvas_255,
                rotatedImage.size(),
                0, 0,
                cv::INTER_NEAREST);
        }
        else
        {
            maskCanvas_255 = maskOnRotatedCanvas_255;
        }

        // Overlay mask
        cv::Mat visRot = rotatedImage.clone();
        const float alpha = 0.4f;

        for (int y = 0; y < visRot.rows; ++y)
        {
            const uchar* m = maskCanvas_255.ptr<uchar>(y);
            cv::Vec3b* p = visRot.ptr<cv::Vec3b>(y);

            for (int x = 0; x < visRot.cols; ++x)
            {
                if (m[x] > 0)
                {
                    p[x][0] = (uchar)((1 - alpha) * p[x][0]);
                    p[x][1] = (uchar)((1 - alpha) * p[x][1] + alpha * 255);
                    p[x][2] = (uchar)((1 - alpha) * p[x][2]);
                }
            }
        }

        // Draw SAM box
        cv::rectangle(visRot, samBoxOnRotatedImage, cv::Scalar(0, 0, 255), 3);

        // ------------------ Show in fixed-size window ------------------
        cv::namedWindow("DEBUG: Rotated Canvas + Mask + Box", cv::WINDOW_NORMAL);
        cv::resizeWindow("DEBUG: Rotated Canvas + Mask + Box", 720, 960);
        cv::imshow("DEBUG: Rotated Canvas + Mask + Box", visRot);

        cv::waitKey(0);

        // ------------- Viz: original Mask on original RGBImage
        cv::Mat visOrig = rgbImage.clone();

        if (maskBySAMOriginal_255.size() == rgbImage.size())
        {
            for (int y = 0; y < visOrig.rows; ++y)
            {
                const uchar* m = maskBySAMOriginal_255.ptr<uchar>(y);
                cv::Vec3b* p = visOrig.ptr<cv::Vec3b>(y);

                for (int x = 0; x < visOrig.cols; ++x)
                {
                    if (m[x] > 0)
                    {
                        p[x][0] = (uchar)(0.6f * p[x][0]);
                        p[x][1] = (uchar)(0.6f * p[x][1] + 0.4f * 255.0f);
                        p[x][2] = (uchar)(0.6f * p[x][2]);
                    }
                }
            }
        }
        else
        {
            std::cout << "[WARN] Mask size does not match RGB size.\n";
        }

        cv::namedWindow("DEBUG: Original RGB + RotatedBack Mask", cv::WINDOW_NORMAL);
        cv::resizeWindow("DEBUG: Original RGB + RotatedBack Mask", 720, 960);
        cv::imshow("DEBUG: Original RGB + RotatedBack Mask", visOrig);
        cv::waitKey(0);
    }
   
    /// ----------------------------
    cv::Mat maskBySAMOriginal_u8;
    cv::threshold(maskBySAMOriginal_255, maskBySAMOriginal_u8, 0, 1, cv::THRESH_BINARY);
    CV_Assert(maskBySAMOriginal_u8.type() == CV_8U);

    // ---- 9) Extract raw RGB edges from SAM mask ----
    std::vector<EdgeRow> rawRGBEdgeList = findRGBEdgePointListFromSAMMask(
        rotatedImage,
        maskBySAMOriginal_u8,                        // CV_8U {0,1}
        breastHeightPtRGB,
        /*verticalSearchHalfRange=*/50
    );
    if (DBH_DEBUG) {
        // ------------------ Debug Print ------------------
        std::cout << "\n===== rawRGBEdgeList Debug =====\n";
        std::cout << "Number of edge rows: " << rawRGBEdgeList.size() << "\n";
        std::cout << "================================\n";
    }
    

    // ---- 11) Keep only trunk component, then filter edges within trunk ----
    cv::Mat maskBySAMTrunkOriginal_u8 = findTreeTrunkFromSAMMask(maskBySAMOriginal_u8, 0.5f); // CV_8U {0,1}
   
    std::vector<EdgeRow> rawRGBEdgeListWithinTrunk = findEdgeListWithTrunkMask(rawRGBEdgeList, maskBySAMTrunkOriginal_u8);
    if (DBH_DEBUG) {
        // ------------------- Debug -----------------------------
        std::cout << "\n===== rawRGBEdgeListWithinTrunk Debug =====\n";
        std::cout << "Original edge rows: " << rawRGBEdgeList.size() << "\n";
        std::cout << "Within trunk rows : " << rawRGBEdgeListWithinTrunk.size() << "\n";

        std::cout << "[DBHMeasurer][SAM] rawRGBEdgeListWithinTrunk has "
            << rawRGBEdgeListWithinTrunk.size() << " rows within trunk.\n";
    }
    

    // ---- 12) Clean RGB edge list ----
    CleanEdgeResult cleanedRGB = CleanEdgeListEdgeLengthCenter(rawRGBEdgeListWithinTrunk);
    if (DBH_DEBUG) {
        // ------------------- Debug -----------------------------
        std::cout << "\n===== CleanEdgeResult Debug =====\n";
        std::cout << "cleanedRGB.ok = " << cleanedRGB.ok << "\n";
        std::cout << "edgeListCleaned size   = " << cleanedRGB.edgeListCleaned.size() << "\n";
        std::cout << "centerListCleaned size = " << cleanedRGB.centerListCleaned.size() << "\n";
    }
   
    std::vector<EdgeRow>& edgeListcleanedRGB = cleanedRGB.edgeListCleaned;
    std::vector<cv::Point2f>& centerListCleanedRGB = cleanedRGB.centerListCleaned;

    BHCheckResult bhcheck = CheckBHWithinFittedEdges(edgeListcleanedRGB, breastHeightPtRGB);
    if (!bhcheck.ok) {
        breastHeightPtRGB = bhcheck.updatedBreastHeightPtRGB;

        const RotationMatrices R = computeRotationMatrixFromGrowthDirection(
            cv::Vec2d((double)growthDirUV[0], (double)growthDirUV[1]),
            rotationCenter, H, W
        );

        // ---- Rotate RGB image ----
        rotatedImage = rotateImage(rgbImage, R.M_forward, R.outSize);

        // ---- 5) Rotate points ----
        basePtRGBRotated = rotatePointAroundCenter(basePtRGB, R.M_forward);
        breastHeightPtRGBRotated = rotatePointAroundCenter(breastHeightPtRGB, R.M_forward);

        // chordLengthInPixel is depth-res; scale to RGB pixels
        chordLengthInPixelRGB = chordLengthInPixel * scaleFromLidar2RGB;

        // ---- 6) Create SAM box on rotated image ----
        // Python used extension_rate=0.15, top_lift=300.0; keep base_lift default unless you used it.
        samBoxOnRotatedImage = createSAMBoxRect(
            breastHeightPtRGBRotated, basePtRGBRotated, chordLengthInPixelRGB,
            rotatedImage.rows, rotatedImage.cols,
            /*extension_rate=*/0.15f, /*top_lift=*/300.0f, /*base_lift=*/150.0f
        );

        // ---- 7) SAM segmentation (box prompt) on rotated image ----
        maskBySAMRotated_u8 = SAMBoxSegmentation(sam, rotatedImage, samBoxOnRotatedImage, nullptr, needPreprocess);

        // normalize to {0,1} in CV_8U
        if (maskBySAMRotated_u8.type() != CV_8U) maskBySAMRotated_u8.convertTo(maskBySAMRotated_u8, CV_8U);
        cv::threshold(maskBySAMRotated_u8, maskBySAMRotated_u8, 0, 1, cv::THRESH_BINARY);

        cv::resize(maskBySAMRotated_u8, maskOnRotatedCanvas_u8, rotatedImage.size(),
            0, 0, cv::INTER_NEAREST);

        maskOnRotatedCanvas_u8.convertTo(maskOnRotatedCanvas_255, CV_8U, 255.0);

        maskBySAMOriginal_255 = rotateImage(maskOnRotatedCanvas_255, R.M_backward, originalImageSize);

        cv::threshold(maskBySAMOriginal_255, maskBySAMOriginal_u8, 0, 1, cv::THRESH_BINARY);
        CV_Assert(maskBySAMOriginal_u8.type() == CV_8U);

        // ---- Extract raw RGB edges from SAM mask ----
        std::vector<EdgeRow> rawRGBEdgeList = findRGBEdgePointListFromSAMMask(
            rotatedImage,
            maskBySAMOriginal_u8,                        // CV_8U {0,1}
            breastHeightPtRGB,
            /*verticalSearchHalfRange=*/50
        );

        // ---- Keep only trunk component, then filter edges within trunk ----
        maskBySAMTrunkOriginal_u8 = findTreeTrunkFromSAMMask(maskBySAMOriginal_u8, 0.5f); // CV_8U {0,1}
        rawRGBEdgeListWithinTrunk = findEdgeListWithTrunkMask(rawRGBEdgeList, maskBySAMTrunkOriginal_u8);

        // ---- Clean RGB edge list ----
        CleanEdgeResult cleanedRGB = CleanEdgeListEdgeLengthCenter(rawRGBEdgeListWithinTrunk);
        edgeListcleanedRGB = cleanedRGB.edgeListCleaned;
        centerListCleanedRGB = cleanedRGB.centerListCleaned;

        bhcheck = CheckBHWithinFittedEdges(edgeListcleanedRGB, breastHeightPtRGB);
        if (!bhcheck.ok) {
            std::cout << "The breast height is not located in the trunk! Segmentation failed. " << "\n";
            return out;
        }
    }

    // ---- 13) Build mask around BH from cleaned edges (RGB space) ----
    cv::Mat maskBySAMAroundBH = EdgeListToMask(edgeListcleanedRGB, depthMapRGB.rows, depthMapRGB.cols); // CV_8U {0,1}

    // ---- 14) Compute growth direction again in RGB space (using RGB depth + K_full) ----
    const cv::Mat K_rgb = cam.K_full();
    if (DBH_DEBUG) {
        std::cout << "\n===== K_rgb (Intrinsic Matrix) =====\n";
        std::cout << "Size: " << K_rgb.rows << " x " << K_rgb.cols << "\n";
        std::cout << "Type: " << K_rgb.type() << "\n";
        std::cout << K_rgb << "\n";
        if (K_rgb.rows == 3 && K_rgb.cols == 3)
        {
            std::cout << "\n===== K_rgb Components =====\n";
            std::cout << "fx = " << K_rgb.at<float>(0, 0) << "\n";
            std::cout << "fy = " << K_rgb.at<float>(1, 1) << "\n";
            std::cout << "cx = " << K_rgb.at<float>(0, 2) << "\n";
            std::cout << "cy = " << K_rgb.at<float>(1, 2) << "\n";
            std::cout << "============================\n";
        }
        std::cout << "====================================\n";
    }

    cv::Vec3f growthDirXYZRGB(0, 0, 0);
    cv::Vec2f growthDirUVRGB(0, 0);
    cv::Vec3f centroidXYZRGB(0, 0, 0);

    if (!ComputeGrowDirection(centerListCleanedRGB, depthMapRGB, K_rgb, growthDirXYZRGB, growthDirUVRGB, centroidXYZRGB))
    {
        std::cout << "[DBHMeasurer][SAM] ComputeGrowDirection (RGB) failed.\n";
        return out;
    }

    const cv::Point2i centroiduvRGB = XYZ2RGBPoint(cv::Point3f(centroidXYZRGB[0], centroidXYZRGB[1], centroidXYZRGB[2]), K_rgb);

    const float chordLengthInPixelRGB_clean = ComputeChordLength(edgeListcleanedRGB);
    if (!(chordLengthInPixelRGB_clean > 0.0f))
    {
        std::cout << "[DBHMeasurer][SAM] chordLengthInPixelRGB invalid.\n";
        return out;
    }

    // ---- 15) Plane fitting in RGB space ----
    std::vector<cv::Point>  uvPointsToFitRGB;
    std::vector<cv::Point3f> xyzPointsToFitRGB;

    if (!GetPointsForPlaneFitting(depthMapRGB, maskBySAMAroundBH, K_rgb, centerListCleanedRGB,
        uvPointsToFitRGB, xyzPointsToFitRGB,
        /*depthThreshold_m=*/5.0f,
        /*estimatedDBH_m=*/-1.0f))
    {
        std::cout << "[DBHMeasurer][SAM] GetPointsForPlaneFitting failed.\n";
        return out;
    }

    if (uvPointsToFitRGB.size() < 3)
    {
        std::cout << "[DBHMeasurer][SAM] Not enough uvPointsToFitRGB for plane.\n";
        return out;
    }

    cv::Matx<float, 9, 1> planeFuncUV_RGB;
    if (!FitUVPlane(uvPointsToFitRGB, depthMapRGB, K_rgb, planeFuncUV_RGB))
    {
        std::cout << "[DBHMeasurer][SAM] FitUVPlane failed.\n";
        return out;
    }

    // ---- 16) 3D chord length in meters (UV-plane) ----
    cv::Point leftEdgePointUV_RGB(-1, -1);
    cv::Point rightEdgePointUV_RGB(-1, -1);
    float chordLengthInMeterUV_RGB = 0.0f;

    if (!Compute3DChordLength(
        cv::Point2f((float)centroiduvRGB.x, (float)centroiduvRGB.y),
        growthDirUVRGB,
        chordLengthInPixelRGB_clean,
        depthMapRGB,
        K_rgb,
        planeFuncUV_RGB,
        leftEdgePointUV_RGB,
        rightEdgePointUV_RGB,
        chordLengthInMeterUV_RGB))
    {
        std::cout << "[DBHMeasurer][SAM] Compute3DChordLength failed.\n";
        return out;
    }

    // ---- 17) Distance and DBH ----
    const float distanceToCamera_m_SAM = ComputeCentroidDistance(
        cv::Point3f(centroidXYZRGB[0], centroidXYZRGB[1], centroidXYZRGB[2])
    );
    if (!(distanceToCamera_m_SAM > 0.0f))
    {
        std::cout << "[DBHMeasurer][SAM] distanceToCamera invalid.\n";
        return out;
    }

    const float dbh_m_SAM = ComputeDBHInMeter(distanceToCamera_m_SAM, chordLengthInMeterUV_RGB);
    if (!(dbh_m_SAM > 0.0f))
    {
        std::cout << "[DBHMeasurer][SAM] DBH invalid.\n";
        return out;
    }

//    // Write output variable
//    out.A_depth = cv::Point2f((float)leftEdgePointUV.x, (float)leftEdgePointUV.y);
//    out.B_depth = cv::Point2f((float)rightEdgePointUV.x, (float)rightEdgePointUV.y);
//
//    out.distanceToCamera_m = distanceToCameraInMeter;
//    out.chordLength_m = chordLengthInMeterUV;     // per your updated python (UV-based)
//    out.diameter_m = dbhInMeterUV;
//    out.ok = true;
//
//    // convert to full-res endpoints for drawing
//    out.A_full = frame.depthToRgb(leftEdgePointUV);
//    out.B_full = frame.depthToRgb(rightEdgePointUV);
//    
//    out.Mask = maskBySAMTrunkOriginal_u8;
//
//    if (DBH_DEBUG) {
//        std::cout << "[DBHMeasurer] breastHeightPtLidar=" << breastHeightPtLidar << "\n";
//        std::cout << "[DBHMeasurer] centroiduv=" << centroiduv << "\n";
//        std::cout << "[DBHMeasurer] growthDirUV=" << growthDirUV << "\n";
//        std::cout << "[DBHMeasurer] A_depth=" << out.A_depth << "  B_depth=" << out.B_depth << "\n";
//        std::cout << "[DBHMeasurer] chord_px=" << chordLengthInPixel
//            << " chord_m_UV=" << chordLengthInMeterUV
//            << " dist_m=" << distanceToCameraInMeter
//            << " dbh_m=" << dbhInMeterUV << "\n";
//    }
    
    // Write output variable
    out.A_full = cv::Point2f((float)leftEdgePointUV_RGB.x, (float)leftEdgePointUV_RGB.y);
    out.B_full = cv::Point2f((float)rightEdgePointUV_RGB.x, (float)rightEdgePointUV_RGB.y);

    out.distanceToCamera_m = distanceToCamera_m_SAM;
    out.chordLength_m = chordLengthInMeterUV_RGB;     // per your updated python (UV-based)
    out.diameter_m = dbh_m_SAM;
    out.ok = true;

    // convert to full-res endpoints for drawing
    out.A_depth = frame.rgbToDepth(leftEdgePointUV_RGB);
    out.B_depth = frame.rgbToDepth(rightEdgePointUV_RGB);
    out.Mask = maskBySAMTrunkOriginal_u8;

    if (DBH_DEBUG) {
        std::cout << "[DBHMeasurer] breastHeightPtLidar=" << breastHeightPtLidar << "\n";
        std::cout << "[DBHMeasurer] growthDirUV=" << growthDirUVRGB << "\n";
        std::cout << "[DBHMeasurer] A_depth=" << out.A_depth << "  B_depth=" << out.B_depth << "\n";
        std::cout << " chord_m_UV=" << chordLengthInMeterUV_RGB
            << " dist_m=" << distanceToCamera_m_SAM
            << " dbh_m=" << dbh_m_SAM << "\n";
    }

    return out;
}

DiameterResultCPP DBHMeasurer::measure_point(
    const Frame& frame,
    const CameraModel& cam,
    Sam& sam,
    bool needPreprocess,
    const cv::Point2i& breastHeightPtTappedRGB)
{
    DiameterResultCPP out;

    const cv::Mat& depthMapLidar = frame.depth();
    if (depthMapLidar.empty() || depthMapLidar.type() != CV_32F)
    {
        std::cout << "[DBHMeasurer] depthMap invalid.\n";
        return out;
    }

    const cv::Mat Kd = cam.K_depth();

    // breastHeightPtLidar in depth-res
    const cv::Point2i breastHeightPtLidar = frame.rgbToDepth(breastHeightPtTappedRGB);

    // Diameter mode: segment at the user-tapped point on the original image.
    // ---- 1) Prepare RGB image & aligned RGB-depth map ----
    const cv::Mat& rgbImage = frame.rgb();
    const cv::Mat& depthMapRGB = frame.depthRGB();

//    if (rgbImage.empty() || rgbImage.type() != CV_8UC3)
//    {
//        std::cout << "[DBHMeasurer][SAM] rgbImage invalid.\n";
//        return out;
//    }
//    if (depthMapRGB.empty() || depthMapRGB.type() != CV_32F)
//    {
//        std::cout << "[DBHMeasurer][SAM] depthMapRGB invalid.\n";
//        return out;
//    }

    const int H = rgbImage.rows;
    const int W = rgbImage.cols;

    const cv::Size originalImageSize(W, H);
    const cv::Point2d rotationCenter((double)W * 0.5, (double)H * 0.5);

    // ---- 2) Scale depth-res points to RGB-res points ----
    const float scaleFromLidar2RGB = (float)W / (float)depthMapLidar.cols;

    // centroiduv is depth-res (cv::Point2i) from your depth branch
    const cv::Point2i centroiduv_depth = breastHeightPtLidar;

    cv::Point2i breastHeightPtRGB = breastHeightPtTappedRGB;

    // ---- 7) SAM segmentation with a point prompt on the original image ----
    cv::Mat maskBySAMOriginal_u8 = SAMPointSegmentation(sam, rgbImage, breastHeightPtRGB, nullptr, needPreprocess);
    if (maskBySAMOriginal_u8.empty())
    {
        std::cout << "[DBHMeasurer][SAM] SAMPointSegmentation returned empty mask.\n";
        return out;
    }

    // IMPORTANT: make sure mask resolution matches rgbImage before any per-pixel access
    if (maskBySAMOriginal_u8.size() != rgbImage.size())
    {
        cv::resize(maskBySAMOriginal_u8, maskBySAMOriginal_u8,
            rgbImage.size(), 0, 0, cv::INTER_NEAREST);
    }

    // normalize to {0,1} in CV_8U
    if (maskBySAMOriginal_u8.type() != CV_8U)
        maskBySAMOriginal_u8.convertTo(maskBySAMOriginal_u8, CV_8U);
    cv::Mat maskBySAMOriginal_255;
    cv::threshold(maskBySAMOriginal_u8, maskBySAMOriginal_255, 0, 255, cv::THRESH_BINARY);
    cv::threshold(maskBySAMOriginal_u8, maskBySAMOriginal_u8, 0, 1, cv::THRESH_BINARY);
    // Segmentation can succeed even when the later depth/geometry fit fails.
    // Keep its overlay available instead of making that look like no detection.
    out.Mask = maskBySAMOriginal_u8;

    
    if (DBH_DEBUG)
    {
        const float alpha = 0.4f;

        cv::Mat vis = rgbImage.clone();

        // Create green overlay image
        cv::Mat greenOverlay(rgbImage.size(), CV_8UC3, cv::Scalar(0, 255, 0));

        // Blend only where mask is non-zero
        for (int y = 0; y < vis.rows; ++y)
        {
            const uchar* m = maskBySAMOriginal_255.ptr<uchar>(y);
            cv::Vec3b* p = vis.ptr<cv::Vec3b>(y);
            const cv::Vec3b* g = greenOverlay.ptr<cv::Vec3b>(y);

            for (int x = 0; x < vis.cols; ++x)
            {
                if (m[x] > 0)
                {
                    p[x][0] = (uchar)((1 - alpha) * p[x][0] + alpha * g[x][0]);
                    p[x][1] = (uchar)((1 - alpha) * p[x][1] + alpha * g[x][1]);
                    p[x][2] = (uchar)((1 - alpha) * p[x][2] + alpha * g[x][2]);
                }
            }
        }

        cv::namedWindow("DEBUG: SAM Mask Overlay", cv::WINDOW_NORMAL);
        cv::resizeWindow("DEBUG: SAM Mask Overlay", 720, 960);
        cv::imshow("DEBUG: SAM Mask Overlay", vis);

        cv::waitKey(0);
    }

    // ---- 9) Extract raw RGB edges from SAM mask ----
    std::vector<EdgeRow> rawRGBEdgeList = findRGBEdgePointListFromSAMMask(
        rgbImage,
        maskBySAMOriginal_u8,                        // CV_8U {0,1}
        breastHeightPtRGB,
        /*verticalSearchHalfRange=*/50
    );
    if (DBH_DEBUG) {
        // ------------------ Debug Print ------------------
        std::cout << "\n===== rawRGBEdgeList Debug =====\n";
        std::cout << "Number of edge rows: " << rawRGBEdgeList.size() << "\n";
        std::cout << "================================\n";
    }


    // ---- 11) Keep only trunk component, then filter edges within trunk ----
    // A close-up trunk can legitimately occupy more than half the image.
    // Diameter mode already has a foreground tap; do not erase wide mask rows.
    // Preserve the existing largest-component cleanup without the 50% width cap.
    cv::Mat maskBySAMTrunkOriginal_u8 = findTreeTrunkFromSAMMask(maskBySAMOriginal_u8, 1.0f); // CV_8U {0,1}

    std::vector<EdgeRow> rawRGBEdgeListWithinTrunk = findEdgeListWithTrunkMask(rawRGBEdgeList, maskBySAMTrunkOriginal_u8);
    // A mask touching a side border does not locate that physical trunk edge.
    // Keep the overlay, but use only rows with both edges visible for diameter.
    rawRGBEdgeListWithinTrunk.erase(
        std::remove_if(rawRGBEdgeListWithinTrunk.begin(), rawRGBEdgeListWithinTrunk.end(),
                       [W](const EdgeRow& e) { return !hasVisibleTrunkEdges(e.l, e.r, W); }),
        rawRGBEdgeListWithinTrunk.end());
    if (rawRGBEdgeListWithinTrunk.size() < 2) {
        std::cout << "[Diameter][SAM] Mask detected, but not enough rows with both trunk edges visible. Move farther back.\n";
        return out;
    }

    if (DBH_DEBUG) {
        // ------------------- Debug -----------------------------
        std::cout << "\n===== rawRGBEdgeListWithinTrunk Debug =====\n";
        std::cout << "Original edge rows: " << rawRGBEdgeList.size() << "\n";
        std::cout << "Within trunk rows : " << rawRGBEdgeListWithinTrunk.size() << "\n";

        std::cout << "[DBHMeasurer][SAM] rawRGBEdgeListWithinTrunk has "
            << rawRGBEdgeListWithinTrunk.size() << " rows within trunk.\n";
    }


    // ---- 12) Clean RGB edge list ----
    CleanEdgeResult cleanedRGB = CleanEdgeListEdgeLengthCenter(rawRGBEdgeListWithinTrunk);
    if (DBH_DEBUG) {
        // ------------------- Debug -----------------------------
        std::cout << "\n===== CleanEdgeResult Debug =====\n";
        std::cout << "cleanedRGB.ok = " << cleanedRGB.ok << "\n";
        std::cout << "edgeListCleaned size   = " << cleanedRGB.edgeListCleaned.size() << "\n";
        std::cout << "centerListCleaned size = " << cleanedRGB.centerListCleaned.size() << "\n";
    }

    std::vector<EdgeRow>& edgeListcleanedRGB = cleanedRGB.edgeListCleaned;
    std::vector<cv::Point2f>& centerListCleanedRGB = cleanedRGB.centerListCleaned;

    // ---- 13) Build mask around BH from cleaned edges (RGB space) ----
    cv::Mat maskBySAMAroundBH = EdgeListToMask(edgeListcleanedRGB, depthMapRGB.rows, depthMapRGB.cols); // CV_8U {0,1}

    // ---- 14) Compute growth direction again in RGB space (centers + depth + K must match) ----
    cv::Mat K_rgb = cam.K_full();
    if (K_rgb.type() != CV_32F) K_rgb.convertTo(K_rgb, CV_32F);

    // If depthMapRGB is not the same resolution as RGB image, scale center points (and K)
    std::vector<cv::Point2f> centers_for_depth = centerListCleanedRGB;

    if (depthMapRGB.size() != rgbImage.size())
    {
        const float sx = (float)depthMapRGB.cols / (float)rgbImage.cols;
        const float sy = (float)depthMapRGB.rows / (float)rgbImage.rows;

        for (auto& p : centers_for_depth)
        {
            p.x *= sx;
            p.y *= sy;
        }

        // Scale intrinsics to match the depthMapRGB resolution we are sampling
        // (Assuming K_full is for rgbImage resolution)
        K_rgb.at<float>(0, 0) *= sx; // fx
        K_rgb.at<float>(1, 1) *= sy; // fy
        K_rgb.at<float>(0, 2) *= sx; // cx
        K_rgb.at<float>(1, 2) *= sy; // cy
    }

    // Optional: quick early reject if too few centers
    if (centers_for_depth.size() < 2)
    {
        std::cout << "[DBHMeasurer][SAM] Not enough centers for growth direction.\n";
        return out;
    }

    cv::Vec3f growthDirXYZRGB(0, 0, 0);
    cv::Vec2f growthDirUVRGB(0, 0);
    cv::Vec3f centroidXYZRGB(0, 0, 0);

    if (!ComputeGrowDirection(centerListCleanedRGB, depthMapRGB, K_rgb, growthDirXYZRGB, growthDirUVRGB, centroidXYZRGB))
    {
        std::cout << "[DBHMeasurer][SAM] ComputeGrowDirection (RGB) failed.\n";
        std::cout << "[DBHMeasurer][SAM] ComputeGrowDirection (RGB/depthMapRGB space) failed.\n";
        std::cout << "  rgbImage size      = " << rgbImage.cols << " x " << rgbImage.rows << "\n";
        std::cout << "  depthMapRGB size   = " << depthMapRGB.cols << " x " << depthMapRGB.rows << "\n";
        std::cout << "  K_use type         = " << K_rgb.type() << "\n";
        std::cout << "  centers count      = " << centers_for_depth.size() << "\n";
        return out;
    }

    const cv::Point2i centroiduvRGB = XYZ2RGBPoint(cv::Point3f(centroidXYZRGB[0], centroidXYZRGB[1], centroidXYZRGB[2]), K_rgb);

    const float chordLengthInPixelRGB_clean = ComputeChordLength(edgeListcleanedRGB);
    if (!(chordLengthInPixelRGB_clean > 0.0f))
    {
        std::cout << "[DBHMeasurer][SAM] chordLengthInPixelRGB invalid.\n";
        return out;
    }

    // ---- 15) Plane fitting in RGB space ----
    std::vector<cv::Point>  uvPointsToFitRGB;
    std::vector<cv::Point3f> xyzPointsToFitRGB;

    if (!GetPointsForPlaneFitting(depthMapRGB, maskBySAMAroundBH, K_rgb, centerListCleanedRGB,
        uvPointsToFitRGB, xyzPointsToFitRGB,
        /*depthThreshold_m=*/5.0f,
        /*estimatedDBH_m=*/-1.0f))
    {
        std::cout << "[DBHMeasurer][SAM] GetPointsForPlaneFitting failed.\n";
        return out;
    }

    if (uvPointsToFitRGB.size() < 3)
    {
        std::cout << "[DBHMeasurer][SAM] Not enough uvPointsToFitRGB for plane.\n";
        return out;
    }

    cv::Matx<float, 9, 1> planeFuncUV_RGB;
    if (!FitUVPlane(uvPointsToFitRGB, depthMapRGB, K_rgb, planeFuncUV_RGB))
    {
        std::cout << "[DBHMeasurer][SAM] FitUVPlane failed.\n";
        return out;
    }

    // ---- 16) 3D chord length in meters (UV-plane) ----
    cv::Point leftEdgePointUV_RGB(-1, -1);
    cv::Point rightEdgePointUV_RGB(-1, -1);
    float chordLengthInMeterUV_RGB = 0.0f;

    if (!Compute3DChordLength(
        cv::Point2f((float)centroiduvRGB.x, (float)centroiduvRGB.y),
        growthDirUVRGB,
        chordLengthInPixelRGB_clean,
        depthMapRGB,
        K_rgb,
        planeFuncUV_RGB,
        leftEdgePointUV_RGB,
        rightEdgePointUV_RGB,
        chordLengthInMeterUV_RGB))
    {
        std::cout << "[DBHMeasurer][SAM] Compute3DChordLength failed.\n";
        return out;
    }

    // ---- 17) Distance and DBH ----
    const float distanceToCamera_m_SAM = ComputeCentroidDistance(
        cv::Point3f(centroidXYZRGB[0], centroidXYZRGB[1], centroidXYZRGB[2])
    );
    if (!(distanceToCamera_m_SAM > 0.0f))
    {
        std::cout << "[DBHMeasurer][SAM] distanceToCamera invalid.\n";
        return out;
    }

    const float dbh_m_SAM = ComputeDBHInMeter(distanceToCamera_m_SAM, chordLengthInMeterUV_RGB);
    if (!(dbh_m_SAM > 0.0f))
    {
        std::cout << "[DBHMeasurer][SAM] DBH invalid.\n";
        return out;
    }

    // Write output variable
    out.A_full = cv::Point2f((float)leftEdgePointUV_RGB.x, (float)leftEdgePointUV_RGB.y);
    out.B_full = cv::Point2f((float)rightEdgePointUV_RGB.x, (float)rightEdgePointUV_RGB.y);

    out.distanceToCamera_m = distanceToCamera_m_SAM;
    out.chordLength_m = chordLengthInMeterUV_RGB;     // per your updated python (UV-based)
    out.diameter_m = dbh_m_SAM;
    out.ok = true;

    // convert to full-res endpoints for drawing
    out.A_depth = frame.rgbToDepth(leftEdgePointUV_RGB);
    out.B_depth = frame.rgbToDepth(rightEdgePointUV_RGB);
//    out.Mask = maskBySAMTrunkOriginal_u8;
    out.Mask = maskBySAMOriginal_u8;

    if (DBH_DEBUG) {
        std::cout << "[DBHMeasurer] breastHeightPtLidar=" << breastHeightPtLidar << "\n";
        std::cout << "[DBHMeasurer] growthDirUV=" << growthDirUVRGB << "\n";
        std::cout << "[DBHMeasurer] A_depth=" << out.A_depth << "  B_depth=" << out.B_depth << "\n";
        std::cout << " chord_m_UV=" << chordLengthInMeterUV_RGB
            << " dist_m=" << distanceToCamera_m_SAM
            << " dbh_m=" << dbh_m_SAM << "\n";
    }

    return out;
}


//----------------------------------------------
cv::Point3f DBHMeasurer::LidarPoint2XYZ(const cv::Point2i& pointuv, float depth, const cv::Mat& K)
{
    if (!(depth > 0.0f))
        throw std::invalid_argument("depth must be > 0.");

    cv::Mat Kinv = K.inv();

    // uvHomogeneous = [u, v, 1]
    cv::Mat uvh = (cv::Mat_<float>(3, 1) << (float)pointuv.x, (float)pointuv.y, 1.0f);

    // pointXYZ = depth * (Kinv @ uvh)
    cv::Mat xyz = depth * (Kinv * uvh);

    return cv::Point3f(xyz.at<float>(0, 0),
        xyz.at<float>(1, 0),
        xyz.at<float>(2, 0));
}

cv::Point2i DBHMeasurer::XYZ2LidarPoint(const cv::Point3f& pointXYZ, const cv::Mat& K)
{
    // ([x,y,z]/z) = [x/z, y/z, 1]
    cv::Mat pn = (cv::Mat_<float>(3, 1) << pointXYZ.x / pointXYZ.z, pointXYZ.y / pointXYZ.z, 1.0f);

    // uvHomogeneous = K @ pn
    cv::Mat uvh = K * pn;

    const int u = (int)std::lround(uvh.at<float>(0, 0));
    const int v = (int)std::lround(uvh.at<float>(1, 0));
    return cv::Point2i(u, v);
}

cv::Point3f DBHMeasurer::RGBPoint2XYZ(const cv::Point2i& pointuv, float depth, const cv::Mat& K)
{
    return LidarPoint2XYZ(pointuv, depth, K);
}

cv::Point2i DBHMeasurer::XYZ2RGBPoint(const cv::Point3f& pointXYZ, const cv::Mat& K)
{
    return XYZ2LidarPoint(pointXYZ, K);
}


// Fit x = a*y + b using ordinary least squares
bool DBHMeasurer::FitLine2D_X_as_aY_plus_b(const std::vector<float>& y, const std::vector<float>& x, float& a_out, float& b_out)
{
    const size_t n = std::min(y.size(), x.size());
    if (n < 2) return false;

    double sumY = 0.0, sumX = 0.0, sumYY = 0.0, sumYX = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double yi = (double)y[i];
        const double xi = (double)x[i];
        sumY += yi;
        sumX += xi;
        sumYY += yi * yi;
        sumYX += yi * xi;
    }

    const double denom = (double)n * sumYY - sumY * sumY;
    if (std::abs(denom) < 1e-12) return false;

    const double a = ((double)n * sumYX - sumY * sumX) / denom;
    const double b = (sumX - a * sumY) / (double)n;

    a_out = (float)a;
    b_out = (float)b;
    return true;
}

cv::Mat DBHMeasurer::ComputeGradientXY(const cv::Mat& depthMap)
{
    CV_Assert(depthMap.channels() == 1);
    CV_Assert(depthMap.type() == CV_32F || depthMap.type() == CV_64F);

    cv::Mat sobelX, sobelY;

    // Sobel gradients (same as Python)
    cv::Sobel(depthMap, sobelX, CV_64F, 1, 0, 3);
    cv::Sobel(depthMap, sobelY, CV_64F, 0, 1, 3);

    // gradient magnitude: sqrt(dx^2 + dy^2)
    cv::Mat gradientXY;
    cv::magnitude(sobelX, sobelY, gradientXY); // CV_64F

    // Normalize to [0, 1]
    double minVal, maxVal;
    cv::minMaxLoc(gradientXY, &minVal, &maxVal);

    cv::Mat gradientNorm;
    const double eps = 1e-8;

    gradientNorm = (gradientXY - minVal) / (maxVal - minVal + eps);

    // Convert to CV_32F for downstream use (optional but recommended)
    gradientNorm.convertTo(gradientNorm, CV_32F);

    return gradientNorm;
}

std::vector<EdgeRow> DBHMeasurer::FindEdgeList(const cv::Mat& gradientXY, const cv::Point2f& startingCenterPtuv, int halfWindowSize, float threshold)
{
    CV_Assert(!gradientXY.empty());
    CV_Assert(gradientXY.channels() == 1);
    CV_Assert(gradientXY.type() == CV_32F || gradientXY.type() == CV_64F);

    const int h = gradientXY.rows;
    const int w = gradientXY.cols;

    const int Col0 = (int)std::lround(startingCenterPtuv.x);
    const int Row0 = (int)std::lround(startingCenterPtuv.y);

    std::vector<EdgeRow> edgeList;
    edgeList.reserve((size_t)(2 * halfWindowSize + 1));

    double prevCenterCol = (double)Col0;

    for (int dRow = -halfWindowSize; dRow <= halfWindowSize; ++dRow)
    {
        const int currentRow = Row0 + dRow;
        if (currentRow < 0 || currentRow >= h) continue;

        bool isLeftFound = false;
        bool isRightFound = false;

        int colLefttoCenter = -1;
        int colRighttoCenter = -1;

        // clamp search start
        int startCol = (int)std::lround(prevCenterCol);
        startCol = std::max(0, std::min(w - 1, startCol));

        // Search left from current center (Python: range(int(prevCenterCol), 0, -1))
        for (int c = startCol; c > 0; --c)
        {
            double g = (gradientXY.type() == CV_32F)
                ? (double)gradientXY.at<float>(currentRow, c)
                : gradientXY.at<double>(currentRow, c);

            if (g > (double)threshold)
            {
                isLeftFound = true;
                colLefttoCenter = c;
                break;
            }
        }

        // Search right from current center (Python: range(int(prevCenterCol), w))
        for (int c = startCol; c < w; ++c)
        {
            double g = (gradientXY.type() == CV_32F)
                ? (double)gradientXY.at<float>(currentRow, c)
                : gradientXY.at<double>(currentRow, c);

            if (g > (double)threshold)
            {
                isRightFound = true;
                colRighttoCenter = c;
                break;
            }
        }

        if (isLeftFound && isRightFound && (colRighttoCenter - colLefttoCenter > 2))
        {
            EdgeRow e;
            e.l = colLefttoCenter;
            e.row = currentRow;
            e.r = colRighttoCenter;
            edgeList.push_back(e);

            // Update center for next row (same as Python)
            prevCenterCol = 0.5 * ((double)colLefttoCenter + (double)colRighttoCenter);
        }
    }

    return edgeList;
}

CleanEdgeResult DBHMeasurer::CleanEdgeListEdgeLengthCenter(const std::vector<EdgeRow>& edgeList, float minStdRowLen_px, float sigmaKeepK)
{
    CleanEdgeResult out;
    out.ok = false;
    out.edgeListCleaned.clear();
    out.centerListCleaned.clear();

    if (edgeList.empty()) {
        return out;
    }

    // Step A: filter by row length (|r-l|)
    std::vector<float> rowLengthList;
    rowLengthList.reserve(edgeList.size());
    for (const auto& e : edgeList) {
        rowLengthList.push_back(std::abs(e.r - e.l));
    }

    const float meanLen = Mean(rowLengthList);
    const float stdLen = Std(rowLengthList, meanLen);

    std::vector<EdgeRow> edgeListCleanedByRowLength;
    edgeListCleanedByRowLength.reserve(edgeList.size());

    if (stdLen > minStdRowLen_px) {
        const float lower = meanLen - 2.0f * stdLen;
        const float upper = meanLen + 2.0f * stdLen;
        for (size_t i = 0; i < edgeList.size(); ++i) {
            const float L = rowLengthList[i];
            if (L > lower && L < upper)
                edgeListCleanedByRowLength.push_back(edgeList[i]);
        }
    }
    else {
        edgeListCleanedByRowLength = edgeList;
    }

    // Step 1: compute centerList from cleaned-by-length
    std::vector<cv::Point2f> centerList;
    centerList.reserve(edgeListCleanedByRowLength.size());

    for (const auto& e : edgeListCleanedByRowLength) {
        const float u_center = 0.5f * (e.l + e.r);
        const float v_row = (float)e.row;
        centerList.emplace_back(u_center, v_row);
    }

    // if centerList is empty, fallback to original edgeList and return False
    if (centerList.empty()) {
        out.edgeListCleaned = edgeList;
        out.centerListCleaned.reserve(edgeList.size());
        for (const auto& e : edgeList) {
            out.centerListCleaned.emplace_back(0.5f * (e.l + e.r), (float)e.row);
        }
        out.ok = false;
        return out;
    }

    // Step 2: fit line x_center = a*y + b
    std::vector<float> centerRows; // y
    std::vector<float> centerCols; // x
    centerRows.reserve(centerList.size());
    centerCols.reserve(centerList.size());

    for (const auto& c : centerList) {
        centerCols.push_back(c.x);
        centerRows.push_back(c.y);
    }

    float a = 0.f, b = 0.f;
    if (!FitLine2D_X_as_aY_plus_b(centerRows, centerCols, a, b)) {
        // If we can't fit, keep what we have (closest to "keep all")
        out.edgeListCleaned = edgeListCleanedByRowLength;
        out.centerListCleaned = centerList;
        out.ok = true;
        return out;
    }

    // Step 3: signed perpendicular residuals
    // signed_residuals = (x - (a*y+b)) / sqrt(1+a^2)
    const float denom = std::sqrt(1.0f + a * a);
    std::vector<float> signed_residuals;
    signed_residuals.reserve(centerList.size());

    for (size_t i = 0; i < centerList.size(); ++i) {
        const float y = centerRows[i];
        const float x = centerCols[i];
        const float r = (x - (a * y + b)) / denom;
        signed_residuals.push_back(r);
    }

    const float meanRes = Mean(signed_residuals);
    const float stdRes = Std(signed_residuals, meanRes);

    // if std_res < minStdRowLen_px: keep all rows
    if (stdRes < minStdRowLen_px) {
        out.edgeListCleaned = edgeListCleanedByRowLength;
        out.centerListCleaned = centerList;
        out.ok = true;
        return out;
    }

    const float thr = sigmaKeepK * stdRes;

    out.edgeListCleaned.reserve(edgeListCleanedByRowLength.size());
    out.centerListCleaned.reserve(edgeListCleanedByRowLength.size());

    for (size_t i = 0; i < edgeListCleanedByRowLength.size(); ++i) {
        if (std::abs(signed_residuals[i]) <= thr) {
            const auto& e = edgeListCleanedByRowLength[i];
            out.edgeListCleaned.push_back(e);
            out.centerListCleaned.emplace_back(0.5f * (e.l + e.r), (float)e.row);
        }
    }

    out.ok = !out.edgeListCleaned.empty();
    return out;
}

cv::Mat DBHMeasurer::EdgeListToMask(const std::vector<EdgeRow>& edgeList, int imgHeight, int imgWidth)
{
    if (imgHeight <= 0 || imgWidth <= 0)
        throw std::invalid_argument("EdgeListToMask: invalid image size.");

    cv::Mat mask(imgHeight, imgWidth, CV_8UC1, cv::Scalar(0));

    for (const auto& e : edgeList)
    {
        const int row = e.row;
        if (row < 0 || row >= imgHeight) continue;

        const int l_clip = std::max(0, std::min(imgWidth - 1, (int)std::lround(e.l)));
        const int r_clip = std::max(0, std::min(imgWidth - 1, (int)std::lround(e.r)));

        if (l_clip <= r_clip) {
            uchar* mrow = mask.ptr<uchar>(row);
            for (int c = l_clip; c <= r_clip; ++c) mrow[c] = 255;
        }
    }

    return mask;
}

bool DBHMeasurer::FitLine3D_SVD(const std::vector<cv::Point3f>& points, cv::Point3f& centroid, cv::Vec3f& direction_unit)
{
    if (points.size() < 2) return false;

    // centroid = mean
    cv::Vec3f c(0, 0, 0);
    for (const auto& p : points) {
        c[0] += p.x; c[1] += p.y; c[2] += p.z;
    }
    c *= (1.0f / (float)points.size());
    centroid = cv::Point3f(c[0], c[1], c[2]);

    // X = points - centroid  => N x 3 matrix
    cv::Mat X((int)points.size(), 3, CV_32F);
    for (int i = 0; i < (int)points.size(); ++i) {
        X.at<float>(i, 0) = points[i].x - c[0];
        X.at<float>(i, 1) = points[i].y - c[1];
        X.at<float>(i, 2) = points[i].z - c[2];
    }

    // SVD: X = U * W * Vt
    cv::SVD svd(X, cv::SVD::FULL_UV);

    // direction = Vt[0] (first right-singular vector)
    cv::Mat Vt = svd.vt;            // 3x3
    cv::Vec3f dir(Vt.at<float>(0, 0),
        Vt.at<float>(0, 1),
        Vt.at<float>(0, 2));

    const float n = std::sqrt(dir.dot(dir));
    if (n < 1e-12f) return false;

    direction_unit = dir * (1.0f / n);
    return true;
}

bool DBHMeasurer::ComputeGrowDirection(const std::vector<cv::Point2f>& centerList, const cv::Mat& depthMap, const cv::Mat& K,
    cv::Vec3f& growthDirXYZ_unit, cv::Vec2f& growthDirUV_unit, cv::Vec3f& centroidXYZ)
{
    if (depthMap.empty() || depthMap.type() != CV_32FC1)
        throw std::invalid_argument("ComputeGrowDirection: depthMap must be CV_32FC1.");

    const int H = depthMap.rows;
    const int W = depthMap.cols;

    // Step 1. Back-project all centers to 3D
    std::vector<cv::Point3f> centersXYZ;
    centersXYZ.reserve(centerList.size());

    /*std::cout << "The size of CentersXYZ: " << centerList.size() << std::endl;*/

    for (const auto& uv : centerList)
    {
        const int u_int = (int)std::lround(uv.x);
        const int v_int = (int)std::lround(uv.y);

        if (u_int < 0 || u_int >= W || v_int < 0 || v_int >= H)
            continue;

        const float depth = depthMap.at<float>(v_int, u_int);
        if (depth < 0.0f || depth >= 5.0f)
            continue;

        const cv::Point3f ptXYZ = LidarPoint2XYZ(cv::Point2i(u_int, v_int), depth, K);
        centersXYZ.push_back(ptXYZ);
    }

    if (centersXYZ.size() < 2)
    {
        std::cout << "[ComputeGrowDirection] Not enough centers to fit 3D direction.\n";
        return false;
    }

    // Step 2. Fit 3D line using SVD
    cv::Vec3f dir_unit;
    cv::Point3f centroidPt;

    if (!FitLine3D_SVD(centersXYZ, centroidPt, dir_unit))
    {
        std::cout << "[ComputeGrowDirection] FitLine3D_SVD failed.\n";
        return false;
    }

    centroidXYZ = cv::Vec3f(centroidPt.x, centroidPt.y, centroidPt.z);

    // Ensure consistent direction orientation (upwards in 3D)
    if (dir_unit[1] > 0.0f)
        dir_unit *= -1.0f;

    growthDirXYZ_unit = dir_unit;

    // Step 3. Project the 3D direction back to 2D
    const cv::Point3f p0XYZ(centroidXYZ[0], centroidXYZ[1], centroidXYZ[2]);

    const cv::Point3f p1XYZ(centroidXYZ[0] + growthDirXYZ_unit[0], centroidXYZ[1] + growthDirXYZ_unit[1], centroidXYZ[2] + growthDirXYZ_unit[2]);

    const cv::Point2i p0uv_i = XYZ2LidarPoint(p0XYZ, K);
    const cv::Point2i p1uv_i = XYZ2LidarPoint(p1XYZ, K);

    cv::Vec2f growthDirUV(
        static_cast<float>(p1uv_i.x - p0uv_i.x),
        static_cast<float>(p1uv_i.y - p0uv_i.y));

    const float norm2d = std::sqrt(growthDirUV[0] * growthDirUV[0] +
        growthDirUV[1] * growthDirUV[1]);

    if (norm2d > 1e-12f)
        growthDirUV *= (1.0f / norm2d);

    // Ensure direction points upward in the image (v decreases upward)
    if (growthDirUV[1] > 0.0f)
        growthDirUV *= -1.0f;

    growthDirUV_unit = growthDirUV;

    if (DBH_DEBUG) {
        std::cout << "[ComputeGrowDirection] Growth direction 3D: ["
            << growthDirXYZ_unit[0] << ", "
            << growthDirXYZ_unit[1] << ", "
            << growthDirXYZ_unit[2] << "]\n";

        std::cout << "[ComputeGrowDirection] Growth direction 2D: ["
            << growthDirUV_unit[0] << ", "
            << growthDirUV_unit[1] << "]\n";
    }


    return true;
}

bool DBHMeasurer::FitUVPlane(const std::vector<cv::Point>& pointsUV, const cv::Mat& depthMap, const cv::Mat& cameraMatrix, cv::Matx<float, 9, 1>& planeFunction_out)
{
    if (pointsUV.empty())
        throw std::invalid_argument("FitUVPlane: Empty pointsUV.");

    if (depthMap.empty() || depthMap.type() != CV_32FC1)
        throw std::invalid_argument("FitUVPlane: depthMap must be CV_32FC1.");

    const int H = depthMap.rows;
    const int W = depthMap.cols;

    const cv::Mat K = cameraMatrix;

    const float fx = K.at<float>(0, 0);
    const float fy = K.at<float>(1, 1);
    const float cx = K.at<float>(0, 2);
    const float cy = K.at<float>(1, 2);

    std::vector<cv::Vec3f> A_rows;
    std::vector<cv::Vec3f> XYZ_rows;
    A_rows.reserve(pointsUV.size());
    XYZ_rows.reserve(pointsUV.size());

    for (const auto& uv : pointsUV)
    {
        const int u_int = (int)std::lround(uv.x); // np.rint
        const int v_int = (int)std::lround(uv.y);

        // bounds
        if (u_int < 0 || u_int >= W || v_int < 0 || v_int >= H)
            continue;

        const float z = depthMap.at<float>(v_int, u_int);
        if (!(z > 0.0f))
            continue;

        // vectorized unprojection (same math as python):
        // x = (u_int - cx) * z / fx
        // y = (v_int - cy) * z / fy
        const float u = (float)u_int;
        const float v = (float)v_int;
        const float x = (u - cx) * z / fx;
        const float y = (v - cy) * z / fy;

        XYZ_rows.emplace_back(x, y, z);

        // IMPORTANT: A uses uv_valid (possibly non-integer) in python
        A_rows.emplace_back(uv.x, uv.y, 1.0f);
    }

    const int M = (int)A_rows.size();
    if (M < 3)
        throw std::runtime_error("FitUVPlane: Not enough valid points to fit UV planes (need >= 3).");

    cv::Mat A(M, 3, CV_32F);
    cv::Mat XYZ(M, 3, CV_32F);

    for (int i = 0; i < M; ++i)
    {
        A.at<float>(i, 0) = A_rows[i][0];
        A.at<float>(i, 1) = A_rows[i][1];
        A.at<float>(i, 2) = A_rows[i][2];

        XYZ.at<float>(i, 0) = XYZ_rows[i][0];
        XYZ.at<float>(i, 1) = XYZ_rows[i][1];
        XYZ.at<float>(i, 2) = XYZ_rows[i][2];
    }

    // Solve A @ C = XYZ in least squares sense, C is 3x3
    // OpenCV: solve(A, XYZ, C, DECOMP_SVD) gives least-squares when A is not square.
    cv::Mat C; // 3x3
    const bool ok = cv::solve(A, XYZ, C, cv::DECOMP_SVD);
    if (!ok || C.empty() || C.rows != 3 || C.cols != 3)
        throw std::runtime_error("FitUVPlane: cv::solve failed.");

    // Flatten to python format: [C[:,0], C[:,1], C[:,2]]
    // i.e. columns: x,y,z; rows: u,v,1
    planeFunction_out(0) = C.at<float>(0, 0); // cx0
    planeFunction_out(1) = C.at<float>(1, 0); // cx1
    planeFunction_out(2) = C.at<float>(2, 0); // cx2

    planeFunction_out(3) = C.at<float>(0, 1); // cy0
    planeFunction_out(4) = C.at<float>(1, 1); // cy1
    planeFunction_out(5) = C.at<float>(2, 1); // cy2

    planeFunction_out(6) = C.at<float>(0, 2); // cz0
    planeFunction_out(7) = C.at<float>(1, 2); // cz1
    planeFunction_out(8) = C.at<float>(2, 2); // cz2

    return true;
}

cv::Point3f DBHMeasurer::FindXYZFromFittedUVPlanes(const cv::Point2f& pointuv, const cv::Matx<float, 9, 1>& coeffs)
{
    const float u = pointuv.x;
    const float v = pointuv.y;

    // x = cx0*u + cx1*v + cx2
    const float x = coeffs(0) * u + coeffs(1) * v + coeffs(2);
    const float y = coeffs(3) * u + coeffs(4) * v + coeffs(5);
    const float z = coeffs(6) * u + coeffs(7) * v + coeffs(8);

    return cv::Point3f(x, y, z);
}

bool DBHMeasurer::GetPointsForPlaneFitting(const cv::Mat& depthMap, const cv::Mat& mask, const cv::Mat& cameraMatrix, const std::vector<cv::Point2f>& centerPtList,
    std::vector<cv::Point>& uvPointsList_out, std::vector<cv::Point3f>& xyzPointsList_out, float depthThreshold_m, float estimatedDBH_m)
{
    uvPointsList_out.clear();
    xyzPointsList_out.clear();

    if (depthMap.empty() || depthMap.type() != CV_32FC1)
        throw std::invalid_argument("GetPointsForPlaneFitting: depthMap must be CV_32FC1.");
    if (mask.empty() || mask.size() != depthMap.size() || mask.type() != CV_8UC1)
        throw std::invalid_argument("GetPointsForPlaneFitting: mask must be CV_8UC1 and same size as depthMap.");

    const int H = depthMap.rows;
    const int W = depthMap.cols;

    const cv::Mat K = cameraMatrix;
    const float fx = K.at<float>(0, 0);
    const float fy = K.at<float>(1, 1);
    const float cx = K.at<float>(0, 2);
    const float cy = K.at<float>(1, 2);

    // --- 1) per-row center depth lookup ---
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> row_center_depth(H, NaN);

    for (const auto& pt : centerPtList)
    {
        const int u_c = (int)std::lround(pt.x);
        const int v_c = (int)std::lround(pt.y);
        if (0 <= v_c && v_c < H && 0 <= u_c && u_c < W)
        {
            const float d_c = depthMap.at<float>(v_c, u_c);
            if (d_c > 0.0f && d_c < depthThreshold_m)
                row_center_depth[v_c] = d_c;
        }
    }

    // Reserve a rough upper bound to reduce realloc
    uvPointsList_out.reserve(10000);
    xyzPointsList_out.reserve(10000);

    // --- 2) iterate all mask pixels (equiv to np.nonzero(mask_bool)) ---
    for (int v = 0; v < H; ++v)
    {
        const uchar* mrow = mask.ptr<uchar>(v);
        for (int u = 0; u < W; ++u)
        {
            if (mrow[u] == 0) continue;

            const float d = depthMap.at<float>(v, u);

            // depth validity: (d > 0) & (d < depthThreshold)
            if (!(d > 0.0f && d < depthThreshold_m))
                continue;

            // optional DBH-based filtering
            if (estimatedDBH_m > 0.0f)
            {
                const float cd = row_center_depth[v];
                const bool has_cd = !std::isnan(cd);
                if (has_cd)
                {
                    if (!(d <= cd + estimatedDBH_m * 0.4f))
                        continue;
                }
            }

            // --- 3) pinhole unprojection to XYZ (vectorized in python) ---
            const float z = d;
            const float x = ((float)u - cx) * z / fx;
            const float y = ((float)v - cy) * z / fy;

            uvPointsList_out.emplace_back(u, v);        // int32
            xyzPointsList_out.emplace_back(x, y, z);    // float
        }
    }

    return (!uvPointsList_out.empty());
}

float DBHMeasurer::ComputeChordLength(const std::vector<EdgeRow>& edgeList)
{
    if (edgeList.empty())
        return 0.0f;

    double sum = 0.0;
    for (const auto& e : edgeList)
    {
        sum += std::abs(e.r - e.l);
    }

    return static_cast<float>(sum / edgeList.size());
}

bool DBHMeasurer::Compute3DChordLength(const cv::Point2f& centroiduv, const cv::Vec2f& growthDirUV, float chordLength_px, const cv::Mat& depthMap, const cv::Mat& cameraMatrix,
    const cv::Matx<float, 9, 1>& planeFuncUV, cv::Point& leftEdgePointUV_out, cv::Point& rightEdgePointUV_out, float& chordLengthInMeterUV_out)
{
    chordLengthInMeterUV_out = 0.0f;
    leftEdgePointUV_out = cv::Point(-1, -1);
    rightEdgePointUV_out = cv::Point(-1, -1);

    if (depthMap.empty() || depthMap.type() != CV_32FC1)
        throw std::invalid_argument("Compute3DChordLength: depthMap must be CV_32FC1.");

    const int H = depthMap.rows;
    const int W = depthMap.cols;

    // crossSectionDir = [growthDirUV[1], -growthDirUV[0]] normalized
    cv::Vec2f crossSectionDir(growthDirUV[1], -growthDirUV[0]);
    const float csn = std::sqrt(crossSectionDir[0] * crossSectionDir[0] + crossSectionDir[1] * crossSectionDir[1]);
    if (csn < 1e-12f) return false;
    crossSectionDir *= (1.0f / csn);

    // cosTheta = abs(dot(growthDirUV, [0, -1]))
    const float cosTheta = std::abs(growthDirUV.dot(cv::Vec2f(0.0f, -1.0f)));

    // offset = 0.5 * chordLength * crossSectionDir * cosTheta
    const float scale = 0.5f * chordLength_px * cosTheta;
    const cv::Point2f offset(scale * crossSectionDir[0], scale * crossSectionDir[1]);

    // left/right endpoints (rounded to int)
    const cv::Point2f leftF = centroiduv - offset;
    const cv::Point2f rightF = centroiduv + offset;

    const int uL = (int)std::lround(leftF.x);
    const int vL = (int)std::lround(leftF.y);
    const int uR = (int)std::lround(rightF.x);
    const int vR = (int)std::lround(rightF.y);

    // bounds check
    if (uL < 0 || uL >= W || vL < 0 || vL >= H) return false;
    if (uR < 0 || uR >= W || vR < 0 || vR >= H) return false;

    leftEdgePointUV_out = cv::Point(uL, vL);
    rightEdgePointUV_out = cv::Point(uR, vR);

    const float dL = depthMap.at<float>(vL, uL);
    const float dR = depthMap.at<float>(vR, uR);
    if (!(dL > 0.0f) || !(dR > 0.0f)) return false;

    // left/right 3D using fitted UV plane (u,v)->(x,y,z)
    const cv::Point3f leftEdgePointXYZUV = FindXYZFromFittedUVPlanes(cv::Point2f((float)uL, (float)vL), planeFuncUV);
    const cv::Point3f rightEdgePointXYZUV = FindXYZFromFittedUVPlanes(cv::Point2f((float)uR, (float)vR), planeFuncUV);

    const cv::Point3f dUV = rightEdgePointXYZUV - leftEdgePointXYZUV;
    chordLengthInMeterUV_out = std::sqrt(dUV.x * dUV.x + dUV.y * dUV.y + dUV.z * dUV.z);

    if (DBH_DEBUG) {
        std::cout << "[Input]\n";
        std::cout << "  centroiduv        = (" << centroiduv.x << ", " << centroiduv.y << ")\n";
        std::cout << "  growthDirUV       = (" << growthDirUV[0] << ", " << growthDirUV[1] << ")\n";
        std::cout << "  chordLength_px    = " << chordLength_px << "\n";

        std::cout << "[DepthMap]\n";
        std::cout << "  size              = (" << W << ", " << H << ")\n";

        std::cout << "[Cross-section]\n";
        std::cout << "  raw crossSectionDir = ("
            << crossSectionDir[0] << ", " << crossSectionDir[1] << ")\n";
        std::cout << "  norm                = " << csn << "\n";

        std::cout << "[Angle correction]\n";
        std::cout << "  cosTheta            = " << cosTheta << "\n";

        std::cout << "[Offset]\n";
        std::cout << "  scale               = " << scale << "\n";
        std::cout << "  offset              = (" << offset.x << ", " << offset.y << ")\n";

        std::cout << "[Endpoints (float)]\n";
        std::cout << "  leftF               = (" << leftF.x << ", " << leftF.y << ")\n";
        std::cout << "  rightF              = (" << rightF.x << ", " << rightF.y << ")\n";

        std::cout << "[Endpoints (int)]\n";
        std::cout << "  left  (uL, vL)      = (" << uL << ", " << vL << ")\n";
        std::cout << "  right (uR, vR)      = (" << uR << ", " << vR << ")\n";
    }

    return (chordLengthInMeterUV_out > 0.0f);
}

float DBHMeasurer::ComputeCentroidDistance(const cv::Point3f& centroidxyz)
{
    return std::abs(centroidxyz.z);
}

float DBHMeasurer::ComputeDBHInMeter(float distanceInMeter, float chordLengthInMeter)
{
    if (distanceInMeter <= 0.0f || chordLengthInMeter <= 0.0f)
        return 0.0f;

    const float alpha = std::atan(chordLengthInMeter / (2.0f * distanceInMeter)); // radians

    const float sinAlpha = std::sin(alpha);
    const float cosAlpha = std::cos(alpha);

    const float denominator =
        (1.0f / sinAlpha)
        - (std::asin(cosAlpha) / (2.0f * cosAlpha))
        - (sinAlpha / 2.0f);

    if (std::abs(denominator) < 1e-6f)
        return 0.0f;

    const float dbhInMeter = (2.0f * distanceInMeter) / denominator;
    return dbhInMeter;
}

// Functions Implementation for depth-guided SAM method

RotationMatrices DBHMeasurer::computeRotationMatrixFromGrowthDirection(
    const cv::Vec2d& growthDirUV,
    const cv::Point2d& rotationCenter,
    int imageHeight,
    int imageWidth)
{
    // ---- Parse direction vector ----
    const double gx = growthDirUV[0];
    const double gy = growthDirUV[1];

    // ---- Angle from vector ----
    const double theta = std::atan2(gy, gx);              // radians
    const double angleDeg = theta * 180.0 / CV_PI + 90.0; // degrees

    // ---- Raw rotation matrix ----
    const cv::Point2d imgCenter(imageWidth * 0.5, imageHeight * 0.5);
    cv::Mat M_raw = cv::getRotationMatrix2D(imgCenter, angleDeg, 1.0); // 2x3, usually CV_64F

    // Ensure CV_64F (Python uses float64)
    if (M_raw.type() != CV_64F)
        M_raw.convertTo(M_raw, CV_64F);

    const int h = imageHeight;
    const int w = imageWidth;
    const double cx = rotationCenter.x;
    const double cy = rotationCenter.y;

    // ---- Compute bounding-box size after rotation ----
    const double cos_a = std::abs(M_raw.at<double>(0, 0));
    const double sin_a = std::abs(M_raw.at<double>(0, 1));

    const int new_w = static_cast<int>(h * sin_a + w * cos_a); // floor like Python int()
    const int new_h = static_cast<int>(h * cos_a + w * sin_a);

    // ---- Adjust translation so original center maps to new center ----
    cv::Mat M_forward = M_raw.clone();
    M_forward.at<double>(0, 2) += (new_w * 0.5) - cx;
    M_forward.at<double>(1, 2) += (new_h * 0.5) - cy;

    // ---- Inverse matrix (2x3 form) ----
    cv::Mat M_full = cv::Mat::eye(3, 3, CV_64F);
    M_forward.copyTo(M_full(cv::Rect(0, 0, 3, 2)));

    cv::Mat M_inv_full = M_full.inv(); // same as numpy.linalg.inv
    cv::Mat M_backward = M_inv_full(cv::Rect(0, 0, 3, 2)).clone();

    RotationMatrices out;
    out.M_forward = M_forward;
    out.M_backward = M_backward;
    out.outSize = cv::Size(new_w, new_h);
    return out;
}


cv::Mat DBHMeasurer::rotateImage(const cv::Mat& image, const cv::Mat& M, const cv::Size& outSize)
{
    if (M.rows != 2 || M.cols != 3)
        throw std::invalid_argument("rotateImage: M must be 2x3.");

    cv::Mat rotated;
    cv::warpAffine(
        image,
        rotated,
        M,
        outSize,
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0) 
    );
    return rotated;
}


cv::Point2f DBHMeasurer::rotatePointAroundCenter(const cv::Point2f& ptUV, const cv::Mat& M)
{
    if (M.rows != 2 || M.cols != 3)
        throw std::invalid_argument("rotatePointAroundCenter: M must be 2x3.");

    // ptHomo = [x, y, 1]^T
    const double x = ptUV.x;
    const double y = ptUV.y;

    const double rx = M.at<double>(0, 0) * x + M.at<double>(0, 1) * y + M.at<double>(0, 2);
    const double ry = M.at<double>(1, 0) * x + M.at<double>(1, 1) * y + M.at<double>(1, 2);

    return cv::Point2f(static_cast<float>(rx), static_cast<float>(ry));
}

cv::Rect DBHMeasurer::createSAMBoxRect(const cv::Point2f& breastHeightPt, const cv::Point2f& basePt, float chordLengthInPixel, int imageHeight, int imageWidth,
    float extension_rate, float top_lift, float base_lift)
{
    const int H = imageHeight;
    const int W = imageWidth;
    if (H <= 0 || W <= 0) throw std::invalid_argument("createSAMBox: image size must be positive.");
    if (!(std::isfinite(chordLengthInPixel)) || chordLengthInPixel <= 0.0f)
        throw std::invalid_argument("createSAMBox: chordLengthInPixel must be finite and > 0.");

    // ---- Horizontal extent: center at BH, wider than chord length ----
    const float half_total = 0.5f * chordLengthInPixel * (1.0f + extension_rate);

    const float x_center = breastHeightPt.x;
    float x_min_f = x_center - half_total;
    float x_max_f = x_center + half_total;

    // ---- Vertical extent: move the entire box upwards ----
    float y_top_f = std::min(breastHeightPt.y, basePt.y) - top_lift;
    float y_bottom_f = std::max(breastHeightPt.y, basePt.y) - base_lift;

    auto clampf = [](float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    };

    // ---- Clamp to image boundary ----
    x_min_f = clampf(x_min_f, 0.0f, float(W - 1));
    x_max_f = clampf(x_max_f, 0.0f, float(W - 1));
    y_top_f = clampf(y_top_f, 0.0f, float(H - 1));
    y_bottom_f = clampf(y_bottom_f, 0.0f, float(H - 1));

    // ---- Convert to int for SAM box ----
    int x_min = (int)std::floor(x_min_f);
    int x_max = (int)std::ceil(x_max_f);
    int y_min = (int)std::floor(y_top_f);
    int y_max = (int)std::ceil(y_bottom_f);

    // Ensure valid ordering after clamping/rounding
    x_min = std::max(0, std::min(x_min, W - 1));
    x_max = std::max(0, std::min(x_max, W - 1));
    y_min = std::max(0, std::min(y_min, H - 1));
    y_max = std::max(0, std::min(y_max, H - 1));

    if (x_max < x_min) std::swap(x_max, x_min);
    if (y_max < y_min) std::swap(y_max, y_min);

    // cv::Rect uses width/height, while Python returns x_min,y_min,x_max,y_max
    const int rect_w = std::max(1, x_max - x_min + 1);
    const int rect_h = std::max(1, y_max - y_min + 1);
    return cv::Rect(x_min, y_min, rect_w, rect_h);
}

cv::Rect DBHMeasurer::scaleBoxToSamInput(const cv::Rect& box_in, const cv::Size& srcSize, const cv::Size& dstSize)
{
    const float sx = static_cast<float>(dstSize.width) / static_cast<float>(srcSize.width);
    const float sy = static_cast<float>(dstSize.height) / static_cast<float>(srcSize.height);

    int x1 = static_cast<int>(std::lround(box_in.x * sx));
    int y1 = static_cast<int>(std::lround(box_in.y * sy));
    int x2 = static_cast<int>(std::lround((box_in.x + box_in.width) * sx));
    int y2 = static_cast<int>(std::lround((box_in.y + box_in.height) * sy));

    int left = std::min(x1, x2);
    int right = std::max(x1, x2);
    int top = std::min(y1, y2);
    int bottom = std::max(y1, y2);

    left = std::max(0, std::min(left, dstSize.width - 1));
    right = std::max(0, std::min(right, dstSize.width - 1));
    top = std::max(0, std::min(top, dstSize.height - 1));
    bottom = std::max(0, std::min(bottom, dstSize.height - 1));

    const int w = std::max(1, right - left);
    const int h = std::max(1, bottom - top);
    return cv::Rect(left, top, w, h);
}

// Sam::preprocessImage accepts BGR and packs RGB once, inside the wrapper.
// Reuse embeddings only when the resized pixels are identical (rotated images
// and new captures must always be encoded, even when needPreprocess is false).
static cv::Mat prepareSAMImage(Sam& sam, const cv::Mat& image_bgr,
                               bool forcePreprocess, bool* terminated) {
    if (terminated) *terminated = false;
    if (image_bgr.empty()) throw std::invalid_argument("SAM: image is empty");
    cv::Mat image_u8, bgr, resized;
    if (image_bgr.depth() != CV_8U) image_bgr.convertTo(image_u8, CV_8U);
    else image_u8 = image_bgr;
    if (image_u8.channels() == 4) cv::cvtColor(image_u8, bgr, cv::COLOR_BGRA2BGR);
    else if (image_u8.channels() == 1) cv::cvtColor(image_u8, bgr, cv::COLOR_GRAY2BGR);
    else if (image_u8.channels() == 3) bgr = image_u8;
    else throw std::invalid_argument("SAM: unsupported channel count");
    const cv::Size inputSize = sam.getInputSize();
    if (inputSize.empty()) throw std::runtime_error("SAM: model pair is not loaded");
    // The supplied export is fixed at 1x3x720x1024 and includes its own padding
    // and normalization. Match its direct resize contract; do not letterbox here.
    cv::resize(bgr, resized, inputSize, 0, 0, cv::INTER_LINEAR);
    if (forcePreprocess || !sam.hasPreprocessedImage(resized)) {
        if (!sam.preprocessImage(resized, terminated))
            throw std::runtime_error("SAM: image encoding failed");
    }
    return resized;
}

cv::Mat DBHMeasurer::SAMBoxSegmentation(Sam& sam, const cv::Mat& image_bgr, const cv::Rect& box_xywh, bool* terminated, bool needPreprocess) {
    prepareSAMImage(sam, image_bgr, needPreprocess, terminated);
    const cv::Rect roi = scaleBoxToSamInput(box_xywh, image_bgr.size(), sam.getInputSize());
    const std::list<cv::Point> points, negativePoints;
    cv::Mat mask = sam.getMask(points, negativePoints, roi, -1, false);
    if (mask.empty()) throw std::runtime_error("SAM: box decoder returned no mask");
    cv::Mat mask_full;
    cv::resize(mask, mask_full, image_bgr.size(), 0, 0, cv::INTER_NEAREST);
    return mask_full;
}

// Helper: scale a point from original image coords -> SAM input coords
cv::Point DBHMeasurer::scalePointToSamInput(const cv::Point& p_full, const cv::Size& fullSize, const cv::Size& samSize)
{
    const float sx = static_cast<float>(samSize.width) / std::max(1, fullSize.width);
    const float sy = static_cast<float>(samSize.height) / std::max(1, fullSize.height);

    const int x = static_cast<int>(std::lround(p_full.x * sx));
    const int y = static_cast<int>(std::lround(p_full.y * sy));

    return cv::Point(
        std::clamp(x, 0, samSize.width - 1),
        std::clamp(y, 0, samSize.height - 1)
    );
}

cv::Rect makeBoxAroundPoint(const cv::Point& p,
                            const cv::Size& imgSize,
                            int halfSize = 100)
{
    int x1 = std::max(0, p.x - halfSize);
    int y1 = std::max(0, p.y - halfSize);
    int x2 = std::min(imgSize.width  - 1, p.x + halfSize);
    int y2 = std::min(imgSize.height - 1, p.y + halfSize);

    return cv::Rect(cv::Point(x1, y1),
                    cv::Point(x2, y2));
}

cv::Mat DBHMeasurer::SAMPointSegmentation(Sam& sam, const cv::Mat& image_bgr, const cv::Point& point_full, bool* terminated, bool needPreprocess) {
    prepareSAMImage(sam, image_bgr, needPreprocess, terminated);
    const cv::Point point = scalePointToSamInput(point_full, image_bgr.size(), sam.getInputSize());
    // Diameter mode supplies one foreground tap and an empty ROI.
    // The ONNX adapter appends only its required padding token, never a box.
    const std::list<cv::Point> points{point}, negativePoints;
    cv::Mat mask = sam.getMask(points, negativePoints, cv::Rect(), -1, false);
    if (mask.empty()) throw std::runtime_error("SAM: point decoder returned no mask");
    cv::Mat mask_full;
    // getMask validates H,W against the encoder contract; no transpose heuristic.
    cv::resize(mask, mask_full, image_bgr.size(), 0, 0, cv::INTER_NEAREST);
    return mask_full;
}

int DBHMeasurer::clampInt(int v, int lo, int hi)
{
    return std::max(lo, std::min(v, hi));
}

cv::Point2f DBHMeasurer::applyAffine2x3(const cv::Point2f& p, const cv::Mat& M2x3)
{
    CV_Assert(M2x3.rows == 2 && M2x3.cols == 3);
    const float x = p.x, y = p.y;
    const float nx = static_cast<float>(M2x3.at<double>(0, 0) * x + M2x3.at<double>(0, 1) * y + M2x3.at<double>(0, 2));
    const float ny = static_cast<float>(M2x3.at<double>(1, 0) * x + M2x3.at<double>(1, 1) * y + M2x3.at<double>(1, 2));
    return cv::Point2f(nx, ny);
}

cv::Point2f DBHMeasurer::applyAffine2x3_f(const cv::Point2f& p, const cv::Mat& M2x3)
{
    CV_Assert(M2x3.rows == 2 && M2x3.cols == 3);
    const float x = p.x, y = p.y;
    const float nx = M2x3.at<float>(0, 0) * x + M2x3.at<float>(0, 1) * y + M2x3.at<float>(0, 2);
    const float ny = M2x3.at<float>(1, 0) * x + M2x3.at<float>(1, 1) * y + M2x3.at<float>(1, 2);
    return cv::Point2f(nx, ny);
}


//cv::Point2f DBHMeasurer::rotatePointAroundCenter(const cv::Point2f& p, const cv::Mat& M_backward)
//{
//    if (M_backward.type() == CV_64F) return applyAffine2x3(p, M_backward);
//    if (M_backward.type() == CV_32F) return applyAffine2x3_f(p, M_backward);
//    throw std::invalid_argument("M_backward must be CV_64F or CV_32F (2x3 affine).");
//}

std::vector<EdgeRow> DBHMeasurer::findRGBEdgePointListFromSAMMask(const cv::Mat& rotatedImage, const cv::Mat& maskBySAMOriginal, const cv::Point2f& breastHeightPtOriginal, int verticalSearchHalfRange /*=100*/)
{
    const int H_orig = maskBySAMOriginal.rows;
    const int W_orig = maskBySAMOriginal.cols;

    int bh_y = static_cast<int>(std::lround(breastHeightPtOriginal.y));
    bh_y = clampInt(bh_y, 0, H_orig - 1);

    // Match your "latest logic": anchor vertical band around BH (ignore mapped upper bound)
    int upper_bound_orig = bh_y - verticalSearchHalfRange;
    upper_bound_orig = clampInt(upper_bound_orig, 0, H_orig - 1);

    int y_start = upper_bound_orig;
    int y_end = bh_y + verticalSearchHalfRange;
    y_end = clampInt(y_end, 0, H_orig - 1);

    // 3) Edge search in original mask (per-row leftmost/rightmost non-zero)
    std::vector<EdgeRow> edgeRows;
    edgeRows.reserve(static_cast<size_t>(std::max(0, y_end - y_start + 1)));

    for (int y = y_start; y <= y_end; ++y)
    {
        const uchar* rowPtr = maskBySAMOriginal.ptr<uchar>(y);

        int xL = -1;
        int xR = -1;

        // left -> right
        for (int x = 0; x < W_orig; ++x)
        {
            if (rowPtr[x] != 0) { xL = x; break; }
        }

        if (xL < 0) continue; // no mask pixels in this row

        // right -> left
        for (int x = W_orig - 1; x >= 0; --x)
        {
            if (rowPtr[x] != 0) { xR = x; break; }
        }

        if (xR >= xL)
        {
            EdgeRow er;
            er.l = static_cast<float>(xL);
            er.row = y;
            er.r = static_cast<float>(xR);
            edgeRows.push_back(er);
        }
    }

    return edgeRows;
}

cv::Mat DBHMeasurer::findTreeTrunkFromSAMMask(const cv::Mat& maskBySAMOriginal, float widthRatio /*=0.5f*/)
{
    if (maskBySAMOriginal.empty())
        throw std::invalid_argument("maskBySAMOriginal is empty.");

    CV_Assert(maskBySAMOriginal.type() == CV_8U);

    const int H = maskBySAMOriginal.rows;
    const int W = maskBySAMOriginal.cols;

    // --- Convert to boolean mask (0 or 1) ---
    cv::Mat mask_bool;
    cv::threshold(maskBySAMOriginal, mask_bool, 0, 1, cv::THRESH_BINARY);

    // --- 1) Apply the caller's row-width policy ---
    filterTrunkMaskRows(mask_bool.ptr<uchar>(), H, W, mask_bool.step[0], widthRatio);

    // If everything is gone, return empty boolean mask
    if (cv::countNonZero(mask_bool) == 0)
        return cv::Mat::zeros(H, W, CV_8U);

    // --- 2) Keep only largest connected component ---
    cv::Mat labels;
    int num_labels = cv::connectedComponents(mask_bool, labels, 8, CV_32S);

    if (num_labels <= 1)
    {
        // Only background or single small component
        return mask_bool;
    }

    // Count pixel area for each label
    std::vector<int> counts(num_labels, 0);

    for (int y = 0; y < H; ++y)
    {
        const int* labelPtr = labels.ptr<int>(y);
        for (int x = 0; x < W; ++x)
        {
            int lbl = labelPtr[x];
            if (lbl > 0)
                counts[lbl]++;
        }
    }

    // Find largest component (exclude background label 0)
    int max_label = 1;
    int max_count = counts[1];

    for (int lbl = 2; lbl < num_labels; ++lbl)
    {
        if (counts[lbl] > max_count)
        {
            max_count = counts[lbl];
            max_label = lbl;
        }
    }

    // Create trunk mask (boolean output: 0 or 1)
    cv::Mat trunkMask = cv::Mat::zeros(H, W, CV_8U);

    for (int y = 0; y < H; ++y)
    {
        const int* labelPtr = labels.ptr<int>(y);
        uchar* outPtr = trunkMask.ptr<uchar>(y);

        for (int x = 0; x < W; ++x)
        {
            if (labelPtr[x] == max_label)
                outPtr[x] = 1;
        }
    }

    return trunkMask;
}


std::vector<EdgeRow> DBHMeasurer::findEdgeListWithTrunkMask(const std::vector<EdgeRow>& edgeList, const cv::Mat& trunkMask)
{
    if (trunkMask.empty())
        throw std::invalid_argument("trunkMask is empty.");

    CV_Assert(trunkMask.type() == CV_8U);

    const int H = trunkMask.rows;
    const int W = trunkMask.cols;

    std::vector<EdgeRow> edgeListWithinTrunk;
    edgeListWithinTrunk.reserve(edgeList.size());

    for (const auto& e : edgeList)
    {
        const int y = e.row;

        if (y < 0 || y >= H)
            continue;

        const uchar* rowPtr = trunkMask.ptr<uchar>(y);

        bool hasMask = false;
        for (int x = 0; x < W; ++x)
        {
            if (rowPtr[x] != 0)
            {
                hasMask = true;
                break;
            }
        }

        if (hasMask)
            edgeListWithinTrunk.push_back(e);
    }

    return edgeListWithinTrunk;
}

BHCheckResult DBHMeasurer::CheckBHWithinFittedEdges(const std::vector<EdgeRow>& rawRGBEdgeListWithinTrunk, const cv::Point2f& breastHeightPtRGB)
{
    BHCheckResult out;
    out.updatedBreastHeightPtRGB = breastHeightPtRGB;

    if (rawRGBEdgeListWithinTrunk.size() < 2)
    {
        out.ok = false;
        return out;
    }

    // Collect samples for left/right edge fits: x = a*y + b
    std::vector<float> yL, xL, yR, xR;
    yL.reserve(rawRGBEdgeListWithinTrunk.size());
    xL.reserve(rawRGBEdgeListWithinTrunk.size());
    yR.reserve(rawRGBEdgeListWithinTrunk.size());
    xR.reserve(rawRGBEdgeListWithinTrunk.size());

    for (const auto& e : rawRGBEdgeListWithinTrunk)
    {
        // Skip invalid rows (optional but recommended)
        if (e.r <= e.l) continue;

        const float yy = (float)e.row;

        yL.push_back(yy);
        xL.push_back((float)e.l);

        yR.push_back(yy);
        xR.push_back((float)e.r);
    }

    if (yL.size() < 2 || yR.size() < 2)
    {
        out.ok = false;
        return out;
    }

    float aL = 0.f, bL = 0.f;
    float aR = 0.f, bR = 0.f;

    if (!FitLine2D_X_as_aY_plus_b(yL, xL, aL, bL) || !FitLine2D_X_as_aY_plus_b(yR, xR, aR, bR))
    {
        out.ok = false;
        return out;
    }

    const float yBH = breastHeightPtRGB.y;
    float xLeft = aL * yBH + bL;
    float xRight = aR * yBH + bR;

    if (xLeft > xRight) std::swap(xLeft, xRight);

    const float xBH = breastHeightPtRGB.x;
    const bool inside = (xBH >= xLeft && xBH <= xRight);

    if (inside)
    {
        out.ok = true;
        out.updatedBreastHeightPtRGB = breastHeightPtRGB;
        return out;
    }

    // Not inside -> snap to fitted center at the same row
    const float xCenter = 0.5f * (xLeft + xRight);
    out.ok = false;
    out.updatedBreastHeightPtRGB = cv::Point2f(xCenter, yBH);
    return out;
}

