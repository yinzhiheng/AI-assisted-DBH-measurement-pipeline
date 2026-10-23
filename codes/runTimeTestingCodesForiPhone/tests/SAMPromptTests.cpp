#include "../Vendor/SegmentAnything/sam_prompts.hpp"
#include <cassert>
#include <iostream>

int main() {
    // Diameter mode: one tap, no fabricated box or negative point.
    const auto tap = makeSamPromptInputs({{321, 456}}, {}, {});
    assert((tap.coordinates == std::vector<float>{321, 456, 0, 0}));
    assert((tap.labels == std::vector<float>{1, -1}));

    // Point-only foreground/background and multi-click prompts pad once.
    const auto multiple = makeSamPromptInputs({{20, 30}, {40, 50}}, {{60, 70}}, {});
    assert((multiple.coordinates == std::vector<float>{20, 30, 40, 50, 60, 70, 0, 0}));
    assert((multiple.labels == std::vector<float>{1, 1, 0, -1}));
    const auto negative = makeSamPromptInputs({}, {{8, 9}}, {});
    assert((negative.labels == std::vector<float>{0, -1}));

    // DBH mode: preserve both box corners, with no point-only padding.
    const cv::Rect roi(10, 20, 100, 200);
    const auto box = makeSamPromptInputs({}, {}, roi);
    assert((box.coordinates == std::vector<float>{10, 20, 110, 220}));
    assert((box.labels == std::vector<float>{2, 3}));
    const auto mixed = makeSamPromptInputs({{50, 60}}, {{5, 6}}, roi);
    assert((mixed.labels == std::vector<float>{1, 0, 2, 3}));
    assert((mixed.coordinates == std::vector<float>{50, 60, 5, 6, 10, 20, 110, 220}));

    // No caller-provided prompts means no invented user prompt.
    const auto none = makeSamPromptInputs({}, {}, {});
    assert(none.coordinates.empty() && none.labels.empty());
    std::cout << "PASS: Diameter tap, multi-point, negative, box, mixed, empty prompts\n";
}
