#include "bench.h"

#include <ncnn/net.h>

#include <cstring>
#include <vector>

namespace pb {
namespace {

// pnnx normalises graph blobs to in0/out0.

class NcnnBackend : public Backend {
public:
    const char* name() const override { return "ncnn"; }

    bool load(const std::string& path, const std::vector<int>& inputShape,
              std::string& err) override {
        std::string param = path.substr(0, path.find(".param")) + ".param";
        std::string bin = path.substr(0, path.find(".param")) + ".bin";
        net_.opt.num_threads = threads_;
        net_.opt.use_packing_layout = true;
        net_.opt.use_fp16_storage = false;
        net_.opt.use_fp16_arithmetic = false;
        net_.opt.use_int8_storage = false;
        // The prebuilt ncnn shipped with this repo was built NCNN_VULKAN=OFF,
        // so the GPU path needs the rebuilt library under third_party/stage/
        // ncnn-vulkan (see scripts/build_ncnn_vulkan_android.sh).
        if (cfg_ == "vulkan" || cfg_ == "vulkan-fp16") {
            net_.opt.use_vulkan_compute = true;
        }
        if (cfg_ == "vulkan-fp16") {
            net_.opt.use_fp16_packed = true;
            net_.opt.use_fp16_storage = true;
            net_.opt.use_fp16_arithmetic = true;
        }
        if (cfg_ == "int8") {
            net_.opt.use_int8_inference = true;
            net_.opt.use_int8_storage = true;
        }
        if (net_.load_param(param.c_str()) != 0) {
            err = "ncnn: load_param failed " + param;
            return false;
        }
        if (net_.load_model(bin.c_str()) != 0) {
            err = "ncnn: load_model failed " + bin;
            return false;
        }
        if (net_.input_names().empty() || net_.output_names().empty()) {
            err = "ncnn: graph has no named io";
            return false;
        }
        inputName_ = net_.input_names()[0];
        outputName_ = net_.output_names()[0];
        inputShape_ = inputShape;
        return true;
    }

    void unload() override { net_.clear(); }

    void setThreads(int n) override {
        threads_ = n;
        net_.opt.num_threads = n;
    }

    void setConfig(const std::string& cfg) override { cfg_ = cfg; }

    bool forward(const float* input, int64_t, float* output, int64_t outElems,
                 std::string& err) override {
        ncnn::Extractor ex = net_.create_extractor();
        const int c = inputShape_[1];
        const int h = inputShape_[2];
        const int w = inputShape_[3];
        // Canonical ncnn image Mat: (w, h, c) with each channel plane w*h long,
        // which is exactly the planar CHW layout every other backend receives.
        // A dims=4 Mat also compiles but routes some layers down a different
        // path and produced a wrong blob for DET, so dims=3 is required here.
        ncnn::Mat in(w, h, c, const_cast<float*>(input));
        if (ex.input(inputName_.c_str(), in) != 0) {
            err = "ncnn: input failed";
            return false;
        }
        ncnn::Mat out;
        if (ex.extract(outputName_.c_str(), out) != 0) {
            err = "ncnn: extract failed";
            return false;
        }
        const int64_t got = static_cast<int64_t>(out.w) * out.h * out.d * out.c;
        if (got != outElems) {
            char buf[192];
            snprintf(buf, sizeof(buf),
                     "ncnn: output shape mismatch got w=%d h=%d d=%d c=%d elems=%lld expected=%lld",
                     out.w, out.h, out.d, out.c, (long long)got, (long long)outElems);
            err = buf;
            return false;
        }
        if (out.elemsize == sizeof(float)) {
            std::memcpy(output, out.data, static_cast<size_t>(outElems) * sizeof(float));
        } else if (out.elemsize == 2) {
            // fp16 storage/arithmetic keeps the last blob in half precision;
            // widen it so the comparison against the fp32 golden is meaningful.
            const unsigned short* src = reinterpret_cast<const unsigned short*>(out.data);
            for (int64_t i = 0; i < outElems; ++i) {
                output[i] = ncnn::float16_to_float32(src[i]);
            }
        } else {
            err = "ncnn: unexpected output elemsize";
            return false;
        }
        return true;
    }

    int64_t outputElems() const override { return 0; }

private:
    int threads_ = 1;
    std::string cfg_;
    std::string inputName_;
    std::string outputName_;
    std::vector<int> inputShape_;
    ncnn::Net net_;
};

}  // namespace

std::unique_ptr<Backend> createNcnnBackend() {
    return std::unique_ptr<Backend>(new NcnnBackend());
}

}  // namespace pb
