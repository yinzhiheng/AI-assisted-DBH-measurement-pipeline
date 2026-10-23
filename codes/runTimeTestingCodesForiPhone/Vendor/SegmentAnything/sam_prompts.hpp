#ifndef IFORESTER_SAM_PROMPTS_HPP
#define IFORESTER_SAM_PROMPTS_HPP

#include <opencv2/core/types.hpp>
#include <list>
#include <vector>

struct SamPromptInputs {
    std::vector<float> coordinates;
    std::vector<float> labels;
};

// Match MobileSAMv2 PromptEncoder.forward: point-only prompts use
// _embed_points(..., pad=(boxes is None)). The ONNX helper embeds the supplied
// tokens directly, so the caller must supply the label -1 padding token.
// This token is not a second tap or a box corner.
inline SamPromptInputs makeSamPromptInputs(const std::list<cv::Point>& points,
                                          const std::list<cv::Point>& negativePoints,
                                          const cv::Rect& roi) {
    SamPromptInputs inputs;
    auto append = [&](float x, float y, float label) {
        inputs.coordinates.push_back(x);
        inputs.coordinates.push_back(y);
        inputs.labels.push_back(label);
    };
    for (const auto& p : points) append(p.x, p.y, 1.f);
    for (const auto& p : negativePoints) append(p.x, p.y, 0.f);
    if (!roi.empty()) {
        append(roi.x, roi.y, 2.f);
        append(roi.br().x, roi.br().y, 3.f);
    } else if (!inputs.labels.empty()) {
        append(0.f, 0.f, -1.f);
    }
    return inputs;
}

#endif
