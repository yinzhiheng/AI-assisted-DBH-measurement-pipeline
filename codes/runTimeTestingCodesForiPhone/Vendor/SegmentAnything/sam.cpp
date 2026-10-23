#include "sam.hpp"
#include "sam_prompts.hpp"
#include <numeric>
#include <algorithm>
#include <stdexcept>

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
    preprocessedImageBGR.release();
    clearPreviousMasks();
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
    std::ifstream f(modelPath);
    if (!f.good()) {
        return false;
    }
    return true;
}

bool Sam::loadModel(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber, bool *terminated){
    if (terminated) *terminated = false;
    if (sessionPre && sessionSam) return true;
    clearLoadModel();
    try {
        if (!modelExists(preModelPath) || !modelExists(samModelPath))
            throw std::runtime_error("SAM model file is missing");
        for (auto& option : sessionOptions) {
            option.SetIntraOpNumThreads(threadsNumber);
            option.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }
        // Commit only a fully validated pair; a failed decoder must be retryable.
        auto encoder = std::make_unique<Ort::Session>(env, preModelPath.c_str(), sessionOptions[0]);
        auto decoder = std::make_unique<Ort::Session>(env, samModelPath.c_str(), sessionOptions[1]);
        if (encoder->GetInputCount() != 1 || encoder->GetOutputCount() != 1 ||
            decoder->GetInputCount() != 6 || decoder->GetOutputCount() != 3)
            throw std::runtime_error("Unsupported SAM ONNX input/output count");

        Ort::AllocatorWithDefaultOptions allocator;
        auto checkTensor = [&](Ort::Session& session, bool input, size_t index,
                               const char* name, ONNXTensorElementDataType type) {
            auto actual = input ? session.GetInputNameAllocated(index, allocator)
                                : session.GetOutputNameAllocated(index, allocator);
            auto info = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
            if (std::string(actual.get()) != name || info.GetTensorTypeAndShapeInfo().GetElementType() != type)
                throw std::runtime_error(std::string("Unsupported SAM tensor: ") + name);
        };
        checkTensor(*encoder, true, 0, "input", ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
        checkTensor(*encoder, false, 0, "output", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        for (size_t i = 0; i < 6; ++i)
            checkTensor(*decoder, true, i, inputNamesSam[i], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        for (size_t i = 0; i < 3; ++i)
            checkTensor(*decoder, false, i, outputNamesSam[i], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        auto inShape = encoder->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        auto outShape = encoder->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (inShape.size() != 4 || inShape[0] != 1 || inShape[1] != 3 ||
            inShape[2] <= 0 || inShape[3] <= 0 ||
            outShape != std::vector<int64_t>({1, 256, 64, 64}) ||
            decoder->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != outShape)
            throw std::runtime_error("Unsupported SAM encoder/embedding dimensions");
        if (shouldPreprocessTerminate) {
            if (terminated) *terminated = true;
            return false;
        }
        inputShapePre = std::move(inShape);
        outputShapePre = std::move(outShape);
        sessionPre = std::move(encoder);
        sessionSam = std::move(decoder);
    } catch (const std::exception& e) {
        std::cerr << "SAM loadModel: " << e.what() << std::endl;
        clearLoadModel();
        return false;
    }
    return true;
}

cv::Size Sam::getInputSize() const {
    if (!sessionPre || !sessionSam || inputShapePre.size() != 4) return {};
    return cv::Size((int)inputShapePre[3], (int)inputShapePre[2]);
}

bool Sam::hasPreprocessedImage(const cv::Mat& image) const {
    return !preprocessedImageBGR.empty() && !outputTensorValuesPre.empty() &&
           image.size() == preprocessedImageBGR.size() &&
           image.type() == preprocessedImageBGR.type() &&
           cv::norm(image, preprocessedImageBGR, cv::NORM_INF) == 0;
}

bool Sam::preprocessImage(const cv::Mat& image, bool *terminated) {
    if (terminated) *terminated = false;
    preprocessedImageBGR.release();
    outputTensorValuesPre.clear();
    clearPreviousMasks();
    if (!sessionPre || !sessionSam || image.empty() ||
        image.size() != getInputSize() || image.type() != CV_8UC3) return false;
    try {
        const size_t planeSize = (size_t)image.rows * image.cols;
        std::vector<uint8_t> inputTensorValues(3 * planeSize);
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                const auto pixel = image.at<cv::Vec3b>(i, j);
                const size_t offset = (size_t)i * image.cols + j;
                inputTensorValues[offset] = pixel[2];
                inputTensorValues[planeSize + offset] = pixel[1];
                inputTensorValues[2 * planeSize + offset] = pixel[0];
            }
        }
        auto inputTensor = Ort::Value::CreateTensor<uint8_t>(memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputShapePre.data(), inputShapePre.size());
        std::vector<float> embeddings(1 * 256 * 64 * 64);
        auto outputTensor = Ort::Value::CreateTensor<float>(memoryInfo, embeddings.data(), embeddings.size(), outputShapePre.data(), outputShapePre.size());
        const char *inputNames[] = {"input"}, *outputNames[] = {"output"};
        runOptionsPre.UnsetTerminate();
        if (shouldPreprocessTerminate) {
            if (terminated) *terminated = true;
            return false;
        }
        timeSamInference(inferenceTiming.encoderMs, [&] {
            sessionPre->Run(runOptionsPre, inputNames, &inputTensor, 1, outputNames, &outputTensor, 1);
        });
        preprocessedImageBGR = image.clone();
        outputTensorValuesPre = std::move(embeddings);
    } catch (const std::exception& e) {
        std::cerr << "SAM preprocessImage: " << e.what() << std::endl;
        if (terminated) *terminated = shouldPreprocessTerminate;
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
//        previousMaskInputValues[i] = low_res_logits[bestMaskIdx * maskInputSize + i];
//    }
//    previousMasks.push_back(previousMaskInputValues);
//    return outputMaskSam;
//}

cv::Mat Sam::getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask) {
    if (!sessionSam || preprocessedImageBGR.empty() || outputTensorValuesPre.empty())
        throw std::runtime_error("SAM requires successful image preprocessing before decoding");
    if (!isNextGetMask && previousMaskIdx >= 0 &&
        static_cast<size_t>(previousMaskIdx) >= previousMasks.size())
        throw std::out_of_range("SAM previous mask index is invalid");

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

    auto prompts = makeSamPromptInputs(points, negativePoints, roi);
    auto& inputPointValues = prompts.coordinates;
    auto& inputLabelValues = prompts.labels;

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
    auto outputTensorsSam = timeSamInference(inferenceTiming.decoderMs, [&] {
        return sessionSam->Run(runOptionsSam, inputNamesSam, inputTensorsSam.data(), inputTensorsSam.size(), outputNamesSam, 3);
    });

    // Extract confidence scores
    auto confidenceScoresRaw = outputTensorsSam[1].GetTensorMutableData<float>();
    auto outputMasksValues = outputTensorsSam[0].GetTensorMutableData<float>();

//    // Determine number of masks
//    size_t numMasks = outputTensorsSam[1].GetTensorTypeAndShapeInfo().GetElementCount();
//    std::cout << "Number of masks: " << numMasks << std::endl;
    
    // Get the shape of the outputMasksValues tensor
    auto maskShape = outputTensorsSam[0].GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "Shape of outputMasksValues: ";
    for (auto dim : maskShape) std::cout << dim << " ";
    std::cout << std::endl;

    if (maskShape.size() != 4 || maskShape[0] != 1 || maskShape[1] <= 0 ||
        maskShape[2] != inputShapePre[2] || maskShape[3] != inputShapePre[3] ||
        outputTensorsSam[1].GetTensorTypeAndShapeInfo().GetElementCount() != (size_t)maskShape[1] ||
        outputTensorsSam[2].GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>({1, maskShape[1], 256, 256}))
        throw std::runtime_error("Unexpected SAM decoder output dimensions");

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
        previousMaskInputValues[i] = low_res_logits[bestMaskIdx * maskInputSize + i];
    }
    previousMasks.push_back(previousMaskInputValues);

    return outputMaskSam;
}


///////////////////

cv::Mat Sam::getMask_confidenceLevel(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask, int confidenceIDX) {
    if (!sessionSam || preprocessedImageBGR.empty() || outputTensorValuesPre.empty())
        throw std::runtime_error("SAM requires successful image preprocessing before decoding");
    if (!isNextGetMask && previousMaskIdx >= 0 &&
        static_cast<size_t>(previousMaskIdx) >= previousMasks.size())
        throw std::out_of_range("SAM previous mask index is invalid");

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

    auto prompts = makeSamPromptInputs(points, negativePoints, roi);
    auto& inputPointValues = prompts.coordinates;
    auto& inputLabelValues = prompts.labels;

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
    std::cout << "Shape of outputMasksValues: ";
    for (auto dim : maskShape) std::cout << dim << " ";
    std::cout << std::endl;

    if (maskShape.size() != 4 || maskShape[0] != 1 || maskShape[1] <= 0 ||
        maskShape[2] != inputShapePre[2] || maskShape[3] != inputShapePre[3] ||
        outputTensorsSam[1].GetTensorTypeAndShapeInfo().GetElementCount() != (size_t)maskShape[1] ||
        outputTensorsSam[2].GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>({1, maskShape[1], 256, 256}))
        throw std::runtime_error("Unexpected SAM decoder output dimensions");

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
    if (confidenceIDX < 0 || static_cast<size_t>(confidenceIDX) >= maskConfidencePairs.size())
        throw std::out_of_range("SAM confidence rank is invalid");
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
        previousMaskInputValues[i] = low_res_logits[bestMaskIdx * maskInputSize + i];
    }
    previousMasks.push_back(previousMaskInputValues);

    return outputMaskSam;
}

float Sam::get_confidenceScore(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints, const cv::Rect& roi, int previousMaskIdx, bool isNextGetMask, int confidenceIDX) {
    if (!sessionSam || preprocessedImageBGR.empty() || outputTensorValuesPre.empty())
        throw std::runtime_error("SAM requires successful image preprocessing before decoding");
    if (!isNextGetMask && previousMaskIdx >= 0 &&
        static_cast<size_t>(previousMaskIdx) >= previousMasks.size())
        throw std::out_of_range("SAM previous mask index is invalid");

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

    auto prompts = makeSamPromptInputs(points, negativePoints, roi);
    auto& inputPointValues = prompts.coordinates;
    auto& inputLabelValues = prompts.labels;

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
    std::cout << "Shape of outputMasksValues: ";
    for (auto dim : maskShape) std::cout << dim << " ";
    std::cout << std::endl;

    if (maskShape.size() != 4 || maskShape[0] != 1 || maskShape[1] <= 0 ||
        maskShape[2] != inputShapePre[2] || maskShape[3] != inputShapePre[3] ||
        outputTensorsSam[1].GetTensorTypeAndShapeInfo().GetElementCount() != (size_t)maskShape[1] ||
        outputTensorsSam[2].GetTensorTypeAndShapeInfo().GetShape() !=
            std::vector<int64_t>({1, maskShape[1], 256, 256}))
        throw std::runtime_error("Unexpected SAM decoder output dimensions");

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
    if (confidenceIDX < 0 || static_cast<size_t>(confidenceIDX) >= maskConfidencePairs.size())
        throw std::out_of_range("SAM confidence rank is invalid");
    int bestMaskIdx = maskConfidencePairs[confidenceIDX].first;
    float bestConfidence = maskConfidencePairs[confidenceIDX].second;
    std::cout << "Selected Mask Index: " << bestMaskIdx << ", Confidence Score: " << bestConfidence << std::endl;


    return bestConfidence;
}

