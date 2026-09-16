#pragma once

#include <cstdint>
#include <vector>

namespace pb {

struct Box {
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

// Single shared implementation of DBNet-style DET preprocessing and decoding.
// Every engine runs this identical code, so differences in the pipeline numbers
// come from the runtime, never from a divergent pre/post choice.
//
// preprocess: keep aspect ratio, longest side -> side, then pad bottom/right
// into a square canvas (the frozen graph demands exactly side x side), and
// apply the standard PP-OCR det mean/std normalisation.
void preprocessDet(const uint8_t* rgb, int srcW, int srcH, int side, float* outNchw);

// postprocess: threshold the probability map, label connected components with
// union-find, then take each component's bounds expanded by unclip.
// NOTE: this is a cost-faithful approximation of Paddle's pipeline (which uses
// contour extraction plus Vatti polygon unclipping). It is *not* an accuracy
// reference -- it exists to measure how much DET time is not engine time.
int postprocessDet(const float* prob, int h, int w, float threshold, float unclip,
                   std::vector<Box>& boxes);

}  // namespace pb
