#pragma once
#include <algorithm>
#include <chrono>
#include <utility>

// Benchmark instrumentation only. Negative means this model was not invoked.
struct SamInferenceTiming {
    double encoderMs = -1;
    double decoderMs = -1;

    double totalMs() const {
        if (encoderMs < 0 && decoderMs < 0) return -1;
        return std::max(0.0, encoderMs) + std::max(0.0, decoderMs);
    }
};

// Measure only the synchronous ONNX Run call supplied by the caller. The scope
// also records elapsed inference if Run throws; the exception still propagates.
template <typename Clock = std::chrono::steady_clock, typename Operation>
decltype(auto) timeSamInference(double& milliseconds, Operation&& operation) {
    struct Scope {
        double& elapsed;
        typename Clock::time_point start = Clock::now();
        ~Scope() {
            elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
    } timer{milliseconds};
    return std::forward<Operation>(operation)();
}
