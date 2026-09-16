#include "bench.h"

#include <cstring>

#include <lw_infer.h>

namespace pb {

// lw.PPOCR.C v0.2.0-preview.1 has NO public forward-only entry point: the
// exported handles (lw_detector_*_bgr_u8, lw_ocr_run_bgr_u8, ...) all bake in
// PPOCR preprocessing and postprocessing. Comparing those against a bare
// ORT/MNN/ncnn forward would measure the pipeline, not the runtime.
//
// The runtime does expose the primitive internally. Both symbols below are
// present and exported in liblw_ppocr_c_static.a (verified with llvm-nm):
//   lw_execute_session_f32            src/runtime/executor.c:1506
//   lw_session_set_intra_op_thread_count  src/runtime/session_internal.h:125
//
// We declare them locally rather than including private headers, so the
// benchmark depends only on the public contract plus two named internals that
// we have verified exist in this exact build. Recorded in the report.
extern "C" {
lw_status lw_execute_session_f32(lw_session* session, const float* input,
                                 uint64_t input_element_count, float* output,
                                 uint64_t output_element_count, lw_error* error);
void lw_session_set_intra_op_thread_count(lw_session* session, uint32_t thread_count);
}

namespace {

class LwBackend : public Backend {
public:
    const char* name() const override { return "lw.PPOCR.C"; }

    bool load(const std::string& path, const std::vector<int>& inputShape,
              std::string& err) override {
        if (inputShape.size() != 4) {
            err = "lw: need NCHW input shape";
            return false;
        }
        for (int i = 0; i < 4; ++i) shape_[i] = inputShape[i];
        inElems_ = 1;
        for (int v : inputShape) inElems_ *= static_cast<int64_t>(v);

        lw_error e;
        lw_error_init(&e);
        lw_model_options mo;
        lw_model_options_init(&mo);
        if (lw_model_load(path.c_str(), &mo, &model_, &e) != LW_STATUS_OK) {
            err = std::string("lw_model_load: ") + e.message;
            return false;
        }
        lw_session_options so;
        lw_session_options_init(&so);
        lw_tensor_desc input;
        lw_tensor_desc_init(&input);
        input.dtype = LW_DTYPE_F32;
        input.rank = 4;
        input.dimensions[0] = shape_[0];
        input.dimensions[1] = shape_[1];
        input.dimensions[2] = shape_[2];
        input.dimensions[3] = shape_[3];
        if (lw_session_create(model_, &input, 1, &so, &session_, &e) != LW_STATUS_OK) {
            err = std::string("lw_session_create: ") + e.message;
            return false;
        }
        lw_tensor_desc out;
        lw_tensor_desc_init(&out);
        if (lw_session_get_output_desc(session_, 0, &out) != LW_STATUS_OK) {
            err = "lw_session_get_output_desc failed";
            return false;
        }
        outElems_ = 1;
        for (uint32_t i = 0; i < out.rank; ++i) outElems_ *= out.dimensions[i];
        return true;
    }

    void unload() override {
        if (session_) { lw_session_free(session_); session_ = nullptr; }
        if (model_) { lw_model_free(model_); model_ = nullptr; }
    }

    void setThreads(int n) override {
        if (session_ && n > 0) {
            lw_session_set_intra_op_thread_count(session_, static_cast<uint32_t>(n));
        }
    }

    bool forward(const float* input, int64_t inElems, float* output, int64_t outElems,
                 std::string& err) override {
        if (outElems != outElems_) {
            err = "lw: unexpected output element count";
            return false;
        }
        lw_error e;
        lw_error_init(&e);
        lw_status st = lw_execute_session_f32(session_, input, inElems, output, outElems, &e);
        if (st != LW_STATUS_OK) {
            err = std::string("lw_execute_session_f32: ") + e.message;
            return false;
        }
        return true;
    }

    int64_t outputElems() const override { return outElems_; }
    int64_t inputElems() const { return inElems_; }

    // lw.PPOCR.C exposes no backend or precision knob at all: there is no
    // quantisation, no fp16 and no GPU/NNAPI delegate in the public API, and
    // its SIMD table (src/simd/) has NEON entries for only three operators.
    // So "strongest configuration" for lw IS the fp32 CPU baseline -- the
    // config is accepted and ignored rather than treated as an error.
    void setConfig(const std::string& cfg) override { (void)cfg; }

private:
    lw_model* model_ = nullptr;
    lw_session* session_ = nullptr;
    int32_t shape_[4] = {1, 3, 0, 0};
    int64_t outElems_ = 0;
    int64_t inElems_ = 0;
};

}  // namespace

std::unique_ptr<Backend> createLwBackend() {
    return std::unique_ptr<Backend>(new LwBackend());
}

}  // namespace pb
