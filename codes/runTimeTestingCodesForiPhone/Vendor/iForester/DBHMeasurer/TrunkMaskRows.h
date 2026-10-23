#pragma once
#include <algorithm>
#include <cstddef>

// Filter the binary mask in place. Diameter mode passes 1.0 to retain wide
// close-up trunks; the DBH path retains its existing 0.5 row-width policy.
inline void filterTrunkMaskRows(unsigned char* data, int height, int width,
                               std::size_t stride, float widthRatio) {
    const int maxWidth = static_cast<int>(widthRatio * width);
    for (int y = 0; y < height; ++y) {
        auto* row = data + static_cast<std::size_t>(y) * stride;
        int left = -1, right = -1;
        for (int x = 0; x < width; ++x) {
            if (row[x]) { left = x; break; }
        }
        if (left < 0) continue;
        for (int x = width - 1; x >= 0; --x) {
            if (row[x]) { right = x; break; }
        }
        if (right - left > maxWidth) std::fill(row, row + width, 0);
    }
}

// Image borders are not observable trunk boundaries. Keep such pixels for the
// segmentation overlay, but do not use the clipped span to estimate diameter.
inline bool hasVisibleTrunkEdges(float left, float right, int imageWidth) {
    return left > 0 && right < imageWidth - 1 && right > left;
}
