#include "sam.hpp"
#include <numeric>
#include <filesystem>

#ifdef _WIN32
#  include <windows.h>
#endif

// ONNX Runtime uses ORTCHAR_T for model paths. On Windows this is wchar_t.
// Convert UTF-8 std::string paths to the correct type.
static std::basic_string<ORTCHAR_T> ToOrtPath(const std::string& utf8Path)
{
#ifdef _WIN32
    if (utf8Path.empty()) return std::basic_string<ORTCHAR_T>();

    // UTF-8 -> UTF-16 (wide) for Windows APIs / ORTCHAR_T.
    const int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8Path.data(), static_cast<int>(utf8Path.size()), nullptr, 0);
    if (wlen <= 0) {
        // Fallback: best-effort ASCII widening
        return std::basic_string<ORTCHAR_T>(utf8Path.begin(), utf8Path.end());
    }

    std::wstring wpath;
    wpath.resize(static_cast<size_t>(wlen));
    MultiByteToWideChar(CP_UTF8, 0, utf8Path.data(), static_cast<int>(utf8Path.size()), wpath.data(), wlen);
    return wpath;
#else
    // Non-Windows builds typically use char.
    return utf8Path;
#endif
}


Sam::Sam(){}
Sam::~Sam(){
    clear();
}

void Sam::clear(){
    clearLoadModel();
    clearPreviousMasks();
}

void Sam::clearLoadModel(){
    Ort::Session* pre = sessionPre.release();
    Ort::Session* sam = sessionSam.release();
    delete pre;
    delete sam;
    shouldPreprocessTerminate = false;
    inputShapePre.resize(0);
    outputShapePre.resize(0);
    outputTensorValuesPre.resize(0);
}

void Sam::clearPreviousMasks(){
    previousMasks.resize(0);
}

void Sam::resizePreviousMasks(int previousMaskIdx){
    if(previousMasks.size() > previousMaskIdx + 1){
        previousMasks.resize(previousMaskIdx + 1);
    }
}

void Sam::terminatePreprocessing(){
    shouldPreprocessTerminate = true;
    runOptionsPre.SetTerminate();
}

bool modelExists(const std::string& modelPath){
    std::ifstream f(std::filesystem::u8path(modelPath), std::ios::binary);
    if (!f.good()) {
        return false;
    }
    return true;
}

bool Sam::loadModel(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber, bool *terminated){
    
    if (sessionPre != NULL) {
        return true;
    }
    
    if(!modelExists(preModelPath) || !modelExists(samModelPath)){
        return false;
    }

    try{
        clearLoadModel();
        for(int i = 0; i < 2; i++){
            auto& option = sessionOptions[i];
            option.SetIntraOpNumThreads(threadsNumber);
            option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }
        // Fix:02/11 for Windows
        const auto preOrtPath = ToOrtPath(preModelPath);
        const auto samOrtPath = ToOrtPath(samModelPath);

        sessionPre = std::make_unique<Ort::Session>(env, preOrtPath.c_str(), sessionOptions[0]);
        sessionSam = std::make_unique<Ort::Session>(env, samOrtPath.c_str(), sessionOptions[1]);


        //sessionPre = std::make_unique<Ort::Session>(env, preModelPath.c_str(), sessionOptions[0]);
        if(sessionPre->GetInputCount() != 1 || sessionPre->GetOutputCount() != 1){
            return false;
        }
        //sessionSam = std::make_unique<Ort::Session>(env, samModelPath.c_str(), sessionOptions[1]);
        if(sessionSam->GetInputCount() != 6 || sessionSam->GetOutputCount() != 3){
            return false;
        }
        inputShapePre = sessionPre->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        outputShapePre = sessionPre->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if(inputShapePre.size() != 4 || outputShapePre.size() != 4){
            return false;
        }
        if(shouldPreprocessTerminate){
            *terminated = true;
            return false;
        }
    }catch(Ort::Exception& e){
        return false;
    }
    return true;
}

cv::Size Sam::getInputSize() const{
    return cv::Size((int)inputShapePre[3], (int)inputShapePre[2]);
}

bool Sam::preprocessImage(const cv::Mat& image, bool *terminated){
    if(image.size() != cv::Size((int)inputShapePre[3], (int)inputShapePre[2])){
        return false;
    }
    if(image.channels() != 3){
        return false;
    }
    try{
        std::vector<uint8_t> inputTensorValues(inputShapePre[0] * inputShapePre[1] * inputShapePre[2] * inputShapePre[3]);
        for(int i = 0; i < inputShapePre[2]; i++){
            for(int j = 0; j < inputShapePre[3]; j++){
                inputTensorValues[i * inputShapePre[3] + j] = image.at<cv::Vec3b>(i, j)[2];
                inputTensorValues[inputShapePre[2] * inputShapePre[3] + i * inputShapePre[3] + j] =
                image.at<cv::Vec3b>(i, j)[1];
                inputTensorValues[2 * inputShapePre[2] * inputShapePre[3] + i * inputShapePre[3] + j] =
                image.at<cv::Vec3b>(i, j)[0];
            }
        }
        auto inputTensor = Ort::Value::CreateTensor<uint8_t>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShapePre.data(), inputShapePre.size());
        outputTensorValuesPre = std::vector<float>(outputShapePre[0] * outputShapePre[1] * outputShapePre[2] * outputShapePre[3]);
        auto outputTensorPre = Ort::Value::CreateTensor<float>(memoryInfo, outputTensorValuesPre.data(), outputTensorValuesPre.size(), outputShapePre.data(), outputShapePre.size());
        const char *inputNamesPre[] = {"input"}, *outputNamesPre[] = {"output"};
        runOptionsPre.UnsetTerminate();
        
        if(shouldPreprocessTerminate){
            *terminated = true;
            return false;
        }
        sessionPre->Run(runOptionsPre, inputNamesPre, &inputTensor, 1, outputNamesPre, &outputTensorPre, 1);
    }catch(Ort::Exception& e){
        *terminated = true;
        return false;
    }
    return true;
}

//cv::Mat Sam::getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask){
//    const size_t maskInputSize = 256 * 256;
//    float maskInputValues[maskInputSize];
//    memset(maskInputValues, 0, sizeof(maskInputValues));
//    float hasMaskValues[] = {0};
//    std::vector<float> previousMaskInputValues;
//    resizePreviousMasks(previousMaskIdx);
//    if(isNextGetMask){
//    }else if(previousMaskIdx >= 0){
//        hasMaskValues[0] = 1;
//        previousMaskInputValues = previousMasks[previousMaskIdx];
//    }
//    std::vector<float> inputPointValues, inputLabelValues;
//    for(auto& point : points){
//        inputPointValues.push_back((float)point.x);
//        inputPointValues.push_back((float)point.y);
//        inputLabelValues.push_back(1);
//    }
//    for(auto& point : negativePoints){
//        inputPointValues.push_back((float)point.x);
//        inputPointValues.push_back((float)point.y);
//        inputLabelValues.push_back(0);
//    }
//    if(!roi.empty()){
//        inputPointValues.push_back((float)roi.x);
//        inputPointValues.push_back((float)roi.y);
//        inputLabelValues.push_back(2);
//        inputPointValues.push_back((float)roi.br().x);
//        inputPointValues.push_back((float)roi.br().y);
//        inputLabelValues.push_back(3);
//    }
//    const int numPoints = (int)inputLabelValues.size();
//    std::vector<int64_t> inputPointShape = {1, numPoints, 2},
//    pointLabelsShape = {1, numPoints},
//    maskInputShape = {1, 1, 256, 256},
//    hasMaskInputShape = {1},
//    origImSizeShape = {2};
//    float orig_im_size_values[] = {(float)inputShapePre[2], (float)inputShapePre[3]};
//    std::vector<Ort::Value> inputTensorsSam;
//    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)outputTensorValuesPre.data(), outputTensorValuesPre.size(), outputShapePre.data(), outputShapePre.size()));
//    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputPointValues.data(), 2 * numPoints, inputPointShape.data(), inputPointShape.size()));
//    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputLabelValues.data(), numPoints, pointLabelsShape.data(), pointLabelsShape.size()));
//    if(hasMaskValues[0] == 1){
//        inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, previousMaskInputValues.data(), maskInputSize, maskInputShape.data(), maskInputShape.size()));
//    }else{
//        inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
//    }
//    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
//    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, orig_im_size_values, 2, origImSizeShape.data(), origImSizeShape.size()));
//    Ort::RunOptions runOptionsSam;
//    auto outputTensorsSam = sessionSam->Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(), inputTensorsSam.size(), outputNamesSam, 3);
//    auto outputMasksValues = outputTensorsSam[0].GetTensorMutableData<float>();
//    cv::Mat outputMaskSam = cv::Mat((int)inputShapePre[2], (int)inputShapePre[3], CV_8UC1);
//    for (int i = 0; i < outputMaskSam.rows; i++) {
//        for (int j = 0; j < outputMaskSam.cols; j++) {
//            outputMaskSam.at<uchar>(i, j) = outputMasksValues[i * outputMaskSam.cols + j] > 0 ? 255 : 0;
//        }
//    }
//    auto low_res_logits = outputTensorsSam[2].GetTensorMutableData<float>();
//    previousMaskInputValues = std::vector<float>(maskInputSize);
//    for (int i = 0; i < maskInputSize; i++) {
//        previousMaskInputValues[i] = low_res_logits[i];
//    }
//    previousMasks.push_back(previousMaskInputValues);
//    return outputMaskSam;
//}

cv::Mat Sam::getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask) {
    const size_t maskInputSize = 256 * 256;
    float maskInputValues[maskInputSize];
    memset(maskInputValues, 0, sizeof(maskInputValues));
    float hasMaskValues[] = {0};
    std::vector<float> previousMaskInputValues;

    resizePreviousMasks(previousMaskIdx);
    if (isNextGetMask) {
        // Handle next mask logic if needed
    } else if (previousMaskIdx >= 0) {
        hasMaskValues[0] = 1;
        previousMaskInputValues = previousMasks[previousMaskIdx];
    }

    std::vector<float> inputPointValues, inputLabelValues;
    for (auto& point : points) {
        inputPointValues.push_back((float)point.x);
        inputPointValues.push_back((float)point.y);
        inputLabelValues.push_back(1);
    }
    for (auto& point : negativePoints) {
        inputPointValues.push_back((float)point.x);
        inputPointValues.push_back((float)point.y);
        inputLabelValues.push_back(0);
    }
    if (!roi.empty()) {
        inputPointValues.push_back((float)roi.x);
        inputPointValues.push_back((float)roi.y);
        inputLabelValues.push_back(2);
        inputPointValues.push_back((float)roi.br().x);
        inputPointValues.push_back((float)roi.br().y);
        inputLabelValues.push_back(3);
    }

    const int numPoints = (int)inputLabelValues.size();
    std::vector<int64_t> inputPointShape = {1, numPoints, 2},
                         pointLabelsShape = {1, numPoints},
                         maskInputShape = {1, 1, 256, 256},
                         hasMaskInputShape = {1},
                         origImSizeShape = {2};
    float orig_im_size_values[] = {(float)inputShapePre[2], (float)inputShapePre[3]};
    std::vector<Ort::Value> inputTensorsSam;
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)outputTensorValuesPre.data(), outputTensorValuesPre.size(), outputShapePre.data(), outputShapePre.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputPointValues.data(), 2 * numPoints, inputPointShape.data(), inputPointShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputLabelValues.data(), numPoints, pointLabelsShape.data(), pointLabelsShape.size()));
    if (hasMaskValues[0] == 1) {
        inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, previousMaskInputValues.data(), maskInputSize, maskInputShape.data(), maskInputShape.size()));
    } else {
        inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
    }
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, orig_im_size_values, 2, origImSizeShape.data(), origImSizeShape.size()));

    Ort::RunOptions runOptionsSam;
    auto outputTensorsSam = sessionSam->Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(), inputTensorsSam.size(), outputNamesSam, 3);

    // Extract confidence scores
    auto confidenceScoresRaw = outputTensorsSam[1].GetTensorMutableData<float>();
    auto outputMasksValues = outputTensorsSam[0].GetTensorMutableData<float>();

//    // Determine number of masks
//    size_t numMasks = outputTensorsSam[1].GetTensorTypeAndShapeInfo().GetElementCount();
//    std::cout << "Number of masks: " << numMasks << std::endl;
    
    // Get the shape of the outputMasksValues tensor
    auto maskShape = outputTensorsSam[0].GetTensorTypeAndShapeInfo().GetShape();
    //std::cout << "Shape of outputMasksValues: ";
    for (auto dim : maskShape) std::cout << dim << " ";
    std::cout << std::endl;

    // Parse the dimensions
    size_t batchSize = maskShape[0];  // Batch size (should be 1)
    size_t numMasks = maskShape[1];  // Number of masks
    size_t height = maskShape[2];    // Height of each mask
    size_t width = maskShape[3];     // Width of each mask

    // Store confidence scores with mask indices
    std::vector<std::pair<int, float>> maskConfidencePairs;
    for (size_t i = 0; i < numMasks; i++) {
        maskConfidencePairs.emplace_back(i, confidenceScoresRaw[i]);
    }

    // Sort masks by confidence scores in descending order
    std::sort(maskConfidencePairs.begin(), maskConfidencePairs.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Get the best mask index
    int bestMaskIdx = maskConfidencePairs[0].first;
    float bestConfidence = maskConfidencePairs[0].second;
    std::cout << "Selected Mask Index: " << bestMaskIdx << ", Confidence Score: " << bestConfidence << std::endl;

    // Generate the binary mask for the selected mask
    cv::Mat outputMaskSam = cv::Mat((int)inputShapePre[2], (int)inputShapePre[3], CV_8UC1);
    for (int i = 0; i < outputMaskSam.rows; i++) {
        for (int j = 0; j < outputMaskSam.cols; j++) {
            outputMaskSam.at<uchar>(i, j) = outputMasksValues[bestMaskIdx * height * width + i * outputMaskSam.cols + j] > 0 ? 255 : 0;
        }
    }

    // Save logits for the best mask
    auto low_res_logits = outputTensorsSam[2].GetTensorMutableData<float>();
    previousMaskInputValues = std::vector<float>(maskInputSize);
    for (int i = 0; i < maskInputSize; i++) {
        previousMaskInputValues[i] = low_res_logits[i];
    }
    previousMasks.push_back(previousMaskInputValues);

    return outputMaskSam;
}


///////////////////

cv::Mat Sam::getMask_confidenceLevel(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask, int confidenceIDX) {
    const size_t maskInputSize = 256 * 256;
    float maskInputValues[maskInputSize];
    memset(maskInputValues, 0, sizeof(maskInputValues));
    float hasMaskValues[] = {0};
    std::vector<float> previousMaskInputValues;

    resizePreviousMasks(previousMaskIdx);
    if (isNextGetMask) {
        // Handle next mask logic if needed
    } else if (previousMaskIdx >= 0) {
        hasMaskValues[0] = 1;
        previousMaskInputValues = previousMasks[previousMaskIdx];
    }

    std::vector<float> inputPointValues, inputLabelValues;
    for (auto& point : points) {
        inputPointValues.push_back((float)point.x);
        inputPointValues.push_back((float)point.y);
        inputLabelValues.push_back(1);
    }
    for (auto& point : negativePoints) {
        inputPointValues.push_back((float)point.x);
        inputPointValues.push_back((float)point.y);
        inputLabelValues.push_back(0);
    }
    if (!roi.empty()) {
        inputPointValues.push_back((float)roi.x);
        inputPointValues.push_back((float)roi.y);
        inputLabelValues.push_back(2);
        inputPointValues.push_back((float)roi.br().x);
        inputPointValues.push_back((float)roi.br().y);
        inputLabelValues.push_back(3);
    }

    const int numPoints = (int)inputLabelValues.size();
    std::vector<int64_t> inputPointShape = {1, numPoints, 2},
                         pointLabelsShape = {1, numPoints},
                         maskInputShape = {1, 1, 256, 256},
                         hasMaskInputShape = {1},
                         origImSizeShape = {2};
    float orig_im_size_values[] = {(float)inputShapePre[2], (float)inputShapePre[3]};
    std::vector<Ort::Value> inputTensorsSam;
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)outputTensorValuesPre.data(), outputTensorValuesPre.size(), outputShapePre.data(), outputShapePre.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputPointValues.data(), 2 * numPoints, inputPointShape.data(), inputPointShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputLabelValues.data(), numPoints, pointLabelsShape.data(), pointLabelsShape.size()));
    if (hasMaskValues[0] == 1) {
        inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, previousMaskInputValues.data(), maskInputSize, maskInputShape.data(), maskInputShape.size()));
    } else {
        inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
    }
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, orig_im_size_values, 2, origImSizeShape.data(), origImSizeShape.size()));

    Ort::RunOptions runOptionsSam;
    auto outputTensorsSam = sessionSam->Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(), inputTensorsSam.size(), outputNamesSam, 3);

    // Extract confidence scores
    auto confidenceScoresRaw = outputTensorsSam[1].GetTensorMutableData<float>();
    auto outputMasksValues = outputTensorsSam[0].GetTensorMutableData<float>();

//    // Determine number of masks
//    size_t numMasks = outputTensorsSam[1].GetTensorTypeAndShapeInfo().GetElementCount();
//    std::cout << "Number of masks: " << numMasks << std::endl;
    
    // Get the shape of the outputMasksValues tensor
    auto maskShape = outputTensorsSam[0].GetTensorTypeAndShapeInfo().GetShape();
    //std::cout << "Shape of outputMasksValues: ";
    for (auto dim : maskShape) std::cout << dim << " ";
    std::cout << std::endl;

    // Parse the dimensions
    size_t batchSize = maskShape[0];  // Batch size (should be 1)
    size_t numMasks = maskShape[1];  // Number of masks
    size_t height = maskShape[2];    // Height of each mask
    size_t width = maskShape[3];     // Width of each mask

    // Store confidence scores with mask indices
    std::vector<std::pair<int, float>> maskConfidencePairs;
    for (size_t i = 0; i < numMasks; i++) {
        maskConfidencePairs.emplace_back(i, confidenceScoresRaw[i]);
    }

    // Sort masks by confidence scores in descending order
    std::sort(maskConfidencePairs.begin(), maskConfidencePairs.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Get the best mask index
    int bestMaskIdx = maskConfidencePairs[confidenceIDX].first;
    float bestConfidence = maskConfidencePairs[confidenceIDX].second;
    std::cout << "Selected Mask Index: " << bestMaskIdx << ", Confidence Score: " << bestConfidence << std::endl;

    // Generate the binary mask for the selected mask
    cv::Mat outputMaskSam = cv::Mat((int)inputShapePre[2], (int)inputShapePre[3], CV_8UC1);
    for (int i = 0; i < outputMaskSam.rows; i++) {
        for (int j = 0; j < outputMaskSam.cols; j++) {
            outputMaskSam.at<uchar>(i, j) = outputMasksValues[bestMaskIdx * height * width + i * outputMaskSam.cols + j] > 0 ? 255 : 0;
        }
    }

    // Save logits for the best mask
    auto low_res_logits = outputTensorsSam[2].GetTensorMutableData<float>();
    previousMaskInputValues = std::vector<float>(maskInputSize);
    for (int i = 0; i < maskInputSize; i++) {
        previousMaskInputValues[i] = low_res_logits[i];
    }
    previousMasks.push_back(previousMaskInputValues);

    return outputMaskSam;
}

float Sam::get_confidenceScore(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask, int confidenceIDX) {
    const size_t maskInputSize = 256 * 256;
    float maskInputValues[maskInputSize];
    memset(maskInputValues, 0, sizeof(maskInputValues));
    float hasMaskValues[] = {0};
    std::vector<float> previousMaskInputValues;

    resizePreviousMasks(previousMaskIdx);
    if (isNextGetMask) {
        // Handle next mask logic if needed
    } else if (previousMaskIdx >= 0) {
        hasMaskValues[0] = 1;
        previousMaskInputValues = previousMasks[previousMaskIdx];
    }

    std::vector<float> inputPointValues, inputLabelValues;
    for (auto& point : points) {
        inputPointValues.push_back((float)point.x);
        inputPointValues.push_back((float)point.y);
        inputLabelValues.push_back(1);
    }
    for (auto& point : negativePoints) {
        inputPointValues.push_back((float)point.x);
        inputPointValues.push_back((float)point.y);
        inputLabelValues.push_back(0);
    }
    if (!roi.empty()) {
        inputPointValues.push_back((float)roi.x);
        inputPointValues.push_back((float)roi.y);
        inputLabelValues.push_back(2);
        inputPointValues.push_back((float)roi.br().x);
        inputPointValues.push_back((float)roi.br().y);
        inputLabelValues.push_back(3);
    }

    const int numPoints = (int)inputLabelValues.size();
    std::vector<int64_t> inputPointShape = {1, numPoints, 2},
                         pointLabelsShape = {1, numPoints},
                         maskInputShape = {1, 1, 256, 256},
                         hasMaskInputShape = {1},
                         origImSizeShape = {2};
    float orig_im_size_values[] = {(float)inputShapePre[2], (float)inputShapePre[3]};
    std::vector<Ort::Value> inputTensorsSam;
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)outputTensorValuesPre.data(), outputTensorValuesPre.size(), outputShapePre.data(), outputShapePre.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputPointValues.data(), 2 * numPoints, inputPointShape.data(), inputPointShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputLabelValues.data(), numPoints, pointLabelsShape.data(), pointLabelsShape.size()));
    if (hasMaskValues[0] == 1) {
        inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, previousMaskInputValues.data(), maskInputSize, maskInputShape.data(), maskInputShape.size()));
    } else {
        inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, maskInputValues, maskInputSize, maskInputShape.data(), maskInputShape.size()));
    }
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, hasMaskValues, 1, hasMaskInputShape.data(), hasMaskInputShape.size()));
    inputTensorsSam.push_back(Ort::Value::CreateTensor<float>(memoryInfo, orig_im_size_values, 2, origImSizeShape.data(), origImSizeShape.size()));

    Ort::RunOptions runOptionsSam;
    auto outputTensorsSam = sessionSam->Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(), inputTensorsSam.size(), outputNamesSam, 3);

    // Extract confidence scores
    auto confidenceScoresRaw = outputTensorsSam[1].GetTensorMutableData<float>();
    auto outputMasksValues = outputTensorsSam[0].GetTensorMutableData<float>();

//    // Determine number of masks
//    size_t numMasks = outputTensorsSam[1].GetTensorTypeAndShapeInfo().GetElementCount();
//    std::cout << "Number of masks: " << numMasks << std::endl;
    
    // Get the shape of the outputMasksValues tensor
    auto maskShape = outputTensorsSam[0].GetTensorTypeAndShapeInfo().GetShape();
    //std::cout << "Shape of outputMasksValues: ";
    for (auto dim : maskShape) std::cout << dim << " ";
    std::cout << std::endl;

    // Parse the dimensions
    size_t batchSize = maskShape[0];  // Batch size (should be 1)
    size_t numMasks = maskShape[1];  // Number of masks
    size_t height = maskShape[2];    // Height of each mask
    size_t width = maskShape[3];     // Width of each mask

    // Store confidence scores with mask indices
    std::vector<std::pair<int, float>> maskConfidencePairs;
    for (size_t i = 0; i < numMasks; i++) {
        maskConfidencePairs.emplace_back(i, confidenceScoresRaw[i]);
    }

    // Sort masks by confidence scores in descending order
    std::sort(maskConfidencePairs.begin(), maskConfidencePairs.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    // Get the best mask index
    int bestMaskIdx = maskConfidencePairs[confidenceIDX].first;
    float bestConfidence = maskConfidencePairs[confidenceIDX].second;
    std::cout << "Selected Mask Index: " << bestMaskIdx << ", Confidence Score: " << bestConfidence << std::endl;


    return bestConfidence;
}

