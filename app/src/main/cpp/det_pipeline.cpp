#include "det_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace pb {

namespace {
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

struct Uf {
    std::vector<int> parent;
    explicit Uf(size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent[b] = a;
    }
};
}  // namespace

void preprocessDet(const uint8_t* rgb, int srcW, int srcH, int side, float* outNchw) {
    const float scale = static_cast<float>(side) / static_cast<float>(std::max(srcW, srcH));
    const int dstW = std::max(1, static_cast<int>(srcW * scale + 0.5f));
    const int dstH = std::max(1, static_cast<int>(srcH * scale + 0.5f));

    // Bilinear resize into the top-left of the canvas, then pad right/bottom.
    std::vector<uint8_t> resized(static_cast<size_t>(dstH) * dstW * 3, 0);
    const float invScale = 1.0f / scale;
    for (int y = 0; y < dstH; ++y) {
        const float sy = std::min(static_cast<float>(srcH - 1),
                                  std::max(0.0f, y * invScale));
        const int y0 = static_cast<int>(sy);
        const int y1 = std::min(srcH - 1, y0 + 1);
        const float wy = sy - y0;
        for (int x = 0; x < dstW; ++x) {
            const float sx = std::min(static_cast<float>(srcW - 1),
                                      std::max(0.0f, x * invScale));
            const int x0 = static_cast<int>(sx);
            const int x1 = std::min(srcW - 1, x0 + 1);
            const float wx = sx - x0;
            for (int c = 0; c < 3; ++c) {
                // Index every corner through x1/y1 (both clamped to the last
                // valid row/col). Using `x0 + 1` directly reads up to 3 bytes
                // past the buffer on the final column -- harmless on the P20
                // (glibc malloc slack) but a hard SEGV_ACCERR under Scudo on
                // the S21U. The stale weight is 0 there, so no numeric change.
                const uint8_t* p0 = rgb + (static_cast<size_t>(y0) * srcW + x0) * 3 + c;
                const uint8_t* p1 = rgb + (static_cast<size_t>(y0) * srcW + x1) * 3 + c;
                const uint8_t* p2 = rgb + (static_cast<size_t>(y1) * srcW + x0) * 3 + c;
                const uint8_t* p3 = rgb + (static_cast<size_t>(y1) * srcW + x1) * 3 + c;
                const float p00 = p0[0];
                const float p01 = p1[0];
                const float p10 = p2[0];
                const float p11 = p3[0];
                const float v = (1 - wy) * ((1 - wx) * p00 + wx * p01) +
                                wy * ((1 - wx) * p10 + wx * p11);
                resized[(static_cast<size_t>(y) * dstW + x) * 3 + c] =
                        static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
            }
        }
    }

    const size_t plane = static_cast<size_t>(side) * side;
    for (int c = 0; c < 3; ++c) {
        float* dst = outNchw + static_cast<size_t>(c) * plane;
        for (int y = 0; y < side; ++y) {
            for (int x = 0; x < side; ++x) {
                float v;
                if (y < dstH && x < dstW) {
                    v = resized[(static_cast<size_t>(y) * dstW + x) * 3 + c];
                } else {
                    const int py = std::min(y, dstH - 1);
                    const int px = std::min(x, dstW - 1);
                    v = resized[(static_cast<size_t>(py) * dstW + px) * 3 + c];
                }
                dst[static_cast<size_t>(y) * side + x] = (v / 255.0f - kMean[c]) / kStd[c];
            }
        }
    }
}

int postprocessDet(const float* prob, int h, int w, float threshold, float unclip,
                   std::vector<Box>& boxes) {
    const int n = h * w;
    std::vector<char> mask(static_cast<size_t>(n), 0);
    for (int i = 0; i < n; ++i) mask[i] = prob[i] >= threshold ? 1 : 0;

    Uf uf(static_cast<size_t>(n));
    for (int y = 0; y < h; ++y) {
        const int base = y * w;
        for (int x = 0; x < w; ++x) {
            const int idx = base + x;
            if (!mask[idx]) continue;
            // left neighbour and upper row neighbours (incl. diagonals)
            if (x > 0 && mask[idx - 1]) uf.unite(idx, idx - 1);
            if (y > 0) {
                if (mask[idx - w]) uf.unite(idx, idx - w);
                if (x > 0 && mask[idx - w - 1]) uf.unite(idx, idx - w - 1);
                if (x < w - 1 && mask[idx - w + 1]) uf.unite(idx, idx - w + 1);
            }
        }
    }

    struct Acc {
        int minx = 1 << 30, miny = 1 << 30, maxx = -1, maxy = -1, count = 0;
    };
    std::vector<int> rootIndex(static_cast<size_t>(n), -1);
    std::vector<Acc> acc;
    for (int i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        const int r = uf.find(i);
        if (rootIndex[r] < 0) {
            rootIndex[r] = static_cast<int>(acc.size());
            acc.emplace_back();
        }
        Acc& a = acc[rootIndex[r]];
        const int x = i % w;
        const int y = i / w;
        a.minx = std::min(a.minx, x);
        a.maxx = std::max(a.maxx, x);
        a.miny = std::min(a.miny, y);
        a.maxy = std::max(a.maxy, y);
        ++a.count;
    }

    boxes.clear();
    for (const Acc& a : acc) {
        if (a.count < 16) continue;  // drop specks, same spirit as min_size in Paddle
        float cx = (a.minx + a.maxx) * 0.5f;
        float cy = (a.miny + a.maxy) * 0.5f;
        float bw = (a.maxx - a.minx + 1) * unclip;
        float bh = (a.maxy - a.miny + 1) * unclip;
        Box b;
        b.x0 = std::max(0.0f, cx - bw * 0.5f);
        b.y0 = std::max(0.0f, cy - bh * 0.5f);
        b.x1 = std::min(static_cast<float>(w - 1), cx + bw * 0.5f);
        b.y1 = std::min(static_cast<float>(h - 1), cy + bh * 0.5f);
        boxes.push_back(b);
    }
    return static_cast<int>(boxes.size());
}

}  // namespace pb
