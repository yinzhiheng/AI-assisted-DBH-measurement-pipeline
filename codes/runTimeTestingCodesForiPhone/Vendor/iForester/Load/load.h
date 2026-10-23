#pragma once
#include <opencv2/opencv.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cctype>

// ------------------------- globals for mouse -------------------------

extern cv::Point2i g_tap_full;
extern bool g_hasTap;
extern float g_displayScale;

void OnMouseRGB(int event, int x, int y, int, void*);

cv::Mat readDepthMap(const std::string& combinePngPath);

std::string BytesToPrintablePreserveDelims(const std::vector<unsigned char>& bytes);

std::vector<std::string> SplitByPipe(const std::string& s);

bool ExtractQuotedValueAfterKey(const std::string& text, const std::string& key, std::string& valueOut);

std::string RemoveAllSpaces(const std::string& s);

std::string readIntrinsicStringFromCombinePNG(const std::string& combinePngPath);

cv::Mat readIntrinsicMatrixFromString(const std::string& intrinsicString, float scale);

std::string JoinPath(const std::string& dir, const std::string& file);

std::string Trim(const std::string& s);

void DrawPointLabel(cv::Mat& img, const cv::Point2i& p, const std::string& label, const cv::Scalar& color);

cv::Point2i FullToDepth(const cv::Point2i& p_full, float s);

cv::Mat VisualizeDepth(const cv::Mat& depth_m, float clipMin = 0.2f, float clipMax = 6.0f);

void DrawDirectionArrow(
    cv::Mat& img,
    const cv::Point2f& origin,
    const cv::Vec2f& dir,          // should be unit or near-unit
    float length_px,
    const cv::Scalar& color,
    int thickness = 2
);