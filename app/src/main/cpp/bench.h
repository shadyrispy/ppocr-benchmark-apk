#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pb {

struct Stats {
    double mean_ms = 0;
    double median_ms = 0;
    double p95_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
    int samples = 0;
};

Stats computeStats(const std::vector<double>& ms);

int64_t nowNs();

long readRssKb();
long readPeakRssKb();

std::vector<float> readF32(const std::string& path);
bool readU8(const std::string& path, std::vector<uint8_t>& out);

// One engine, one graph. Every backend takes planar NCHW fp32 input and writes
// the raw graph output -- no preprocessing, no postprocessing. That is what
// makes the four implementations comparable.
class Backend {
public:
    virtual ~Backend() = default;
    virtual const char* name() const = 0;
    // inputShape is NCHW; it is authoritative where the runtime needs it at
    // session-creation time (lw.PPOCR.C) and used only for validation elsewhere.
    virtual bool load(const std::string& path, const std::vector<int>& inputShape,
                      std::string& err) = 0;
    virtual void unload() = 0;
    virtual void setThreads(int n) = 0;
    virtual bool forward(const float* input, int64_t inElems, float* output,
                         int64_t outElems, std::string& err) = 0;
    virtual int64_t outputElems() const = 0;
    // True when the backend can be pinned to n threads; false means it ignores
    // setThreads() and the caller must not claim a parallel result for it.
    virtual bool supportsThreadPinning() const { return true; }

    // Vendor-specific "strongest configuration" knob. Recognised values:
    //   ""            fp32 CPU (the baseline every engine was measured at)
    //   "nnapi"       ORT  -> NNAPI EP, fp32
    //   "nnapi-fp16"  ORT  -> NNAPI EP with NNAPI_FLAG_USE_FP16
    //   "opencl"      MNN  -> OpenCL GPU, fp32
    //   "opencl-fp16" MNN  -> OpenCL GPU, Precision_Low (fp16)
    //   "vulkan"      MNN/ncnn -> Vulkan GPU, fp32
    //   "vulkan-fp16" MNN/ncnn -> Vulkan GPU with fp16 storage + arithmetic
    //   "int8"        ORT/MNN/ncnn -> quantised weights (dir <engine>-int8)
    // Unknown values fall back to the fp32 CPU path instead of failing, so a
    // typo degrades to the baseline rather than killing the whole sweep.
    virtual void setConfig(const std::string& cfg) { (void)cfg; }
};

std::unique_ptr<Backend> createLwBackend();
std::unique_ptr<Backend> createOrtBackend();
std::unique_ptr<Backend> createMnnBackend();
std::unique_ptr<Backend> createNcnnBackend();

inline std::unique_ptr<Backend> createBackend(const std::string& engine,
                                              const std::string& config = "") {
    std::unique_ptr<Backend> b;
    if (engine == "lw") b = createLwBackend();
    else if (engine == "ort") b = createOrtBackend();
    else if (engine == "mnn") b = createMnnBackend();
    else if (engine == "ncnn") b = createNcnnBackend();
    if (b) b->setConfig(config);
    return b;
}

}  // namespace pb
