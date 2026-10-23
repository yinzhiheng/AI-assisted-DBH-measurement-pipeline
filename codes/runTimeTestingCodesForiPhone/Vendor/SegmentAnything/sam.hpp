#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_
#include "sam_inference_timing.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <list>
#include <fstream>
#include <iostream>

#include <codecvt>
#include <locale>
#include <opencv2/opencv.hpp>
#include <vector>

class Sam {
  SamInferenceTiming inferenceTiming;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};
  Ort::SessionOptions sessionOptions[2];
  std::unique_ptr<Ort::Session> sessionPre, sessionSam;
  bool shouldPreprocessTerminate = false;
  Ort::RunOptions runOptionsPre;
  std::vector<int64_t> inputShapePre, outputShapePre;
  Ort::MemoryInfo memoryInfo{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::vector<float> outputTensorValuesPre;
  cv::Mat preprocessedImageBGR;
  std::vector<std::vector<float>> previousMasks;
  const char *inputNamesSam[6]{"image_embeddings", "point_coords", "point_labels",
                               "mask_input", "has_mask_input", "orig_im_size"},
  *outputNamesSam[3]{"masks", "iou_predictions", "low_res_masks"};

 public:
    void resetInferenceTiming() { inferenceTiming = {}; }
    const SamInferenceTiming& getInferenceTiming() const { return inferenceTiming; }
    Sam();
    ~Sam();
    void clear();
    void clearLoadModel();
    void clearPreviousMasks();
    void resizePreviousMasks(int previousMaskIdx);
    void terminatePreprocessing();
    bool loadModel(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber, bool *terminated);
    cv::Size getInputSize() const;
    // Input must be CV_8UC3 BGR, resized to getInputSize(). The ONNX encoder
    // includes normalization and padding; this class packs it as planar RGB.
    bool hasPreprocessedImage(const cv::Mat& image) const;
    bool preprocessImage(const cv::Mat& image, bool *terminated);
    cv::Mat getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask);
    cv::Mat getMask_confidenceLevel(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask, int confidenceIDX);
    float get_confidenceScore(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask, int confidenceIDX);
    

    
    using cbProgress = void (*)(double);
    cv::Mat autoSegment(const cv::Size& numPoints, cbProgress cb = {},
                        const double iouThreshold = 0.86, const double minArea = 100,
                        int* numObjects = nullptr);
};

#endif
