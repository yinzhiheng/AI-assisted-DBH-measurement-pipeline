#include "../Vendor/SegmentAnything/sam_inference_timing.hpp"
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>

// A deterministic clock verifies the production timer's boundaries without
// sleeps or machine-dependent latency expectations.
struct TestClock {
    using duration = std::chrono::milliseconds;
    using time_point = std::chrono::time_point<TestClock, duration>;
    static inline time_point current{};
    static time_point now() { return current; }
    static void advance(int milliseconds) { current += duration(milliseconds); }
};

int main() {
    SamInferenceTiming timing;
    assert(timing.totalMs() == -1); // No inference means a blank CSV time.
    TestClock::advance(100); // Input resize and RGB/tensor preparation.
    timeSamInference<TestClock>(timing.encoderMs, [] { TestClock::advance(7); });
    TestClock::advance(200); // Embedding copy and prompt preparation.
    auto output = timeSamInference<TestClock>(timing.decoderMs, [] {
        TestClock::advance(11);
        return std::make_unique<int>(42); // Preserve move-only decoder results.
    });
    TestClock::advance(300); // Mask selection, threshold, resize and rendering.
    assert(*output == 42);
    assert(timing.encoderMs == 7 && timing.decoderMs == 11);
    assert(timing.totalMs() == 18);
    const double firstPass = timing.totalMs();

    timing = {}; // A retry/new image must not reuse a previous measurement.
    assert(timing.totalMs() == -1);
    timeSamInference<TestClock>(timing.encoderMs, [] { TestClock::advance(5); });
    timeSamInference<TestClock>(timing.decoderMs, [] { TestClock::advance(9); });
    assert(firstPass + timing.totalMs() == 32);

    timing = {};
    bool propagated = false;
    try {
        timeSamInference<TestClock>(timing.encoderMs, [] {
            TestClock::advance(3);
            throw std::runtime_error("Inference failed");
        });
    } catch (const std::runtime_error&) { propagated = true; }
    assert(propagated && timing.encoderMs == 3 && timing.decoderMs == -1);
    assert(timing.totalMs() == 3); // Retain attempted model time, not other work.
    std::cout << "PASS: encoder/decoder-only timing, excluded preparation/postprocessing, retries, reset and exceptions\n";
}
