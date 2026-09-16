#include "bench.h"

#include <nnapi_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <vector>

namespace pb {
namespace {

class OrtBackend : public Backend {
public:
    const char* name() const override { return "onnxruntime"; }

    bool load(const std::string& path, const std::vector<int>& inputShape,
              std::string& err) override {
        try {
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "ppocr_bench");
            Ort::SessionOptions so;
            so.SetIntraOpNumThreads(threads_);
            so.SetInterOpNumThreads(1);
            so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (cfg_ == "nnapi" || cfg_ == "nnapi-fp16") {
                // NNAPI is DISABLED on purpose: on the Exynos 2100 (S21U) and
                // Kirin 970 (P20) test devices, the ORT 1.30 NNAPI Execution
                // Provider segfaults (SIGSEGV, null function-pointer deref
                // inside libonnxruntime.so during Session::Run) for these
                // PP-OCRv6 tiny graphs. It is an NNAPI-driver / ORT-version
                // incompatibility, not a harness bug, and a native abort
                // cannot be caught here -- so we fail the case cleanly and let
                // the rest of the sweep continue. ORT's strongest usable config
                // on these devices is therefore CPU fp32. See results/REPORT.md.
                err = "ORT NNAPI EP segfaults on this device/ORT build "
                      "(SIGSEGV in libonnxruntime.so); disabled. Use CPU.";
                return false;
            }
            session_ = std::make_unique<Ort::Session>(*env_, path.c_str(), so);
        } catch (const Ort::Exception& e) {
            err = std::string("ORT create session: ") + e.what();
            return false;
        }
        inputShape_.assign(inputShape.begin(), inputShape.end());

        Ort::AllocatorWithDefaultOptions alloc;
        inputName_ = session_->GetInputNameAllocated(0, alloc).get();
        outputName_ = session_->GetOutputNameAllocated(0, alloc).get();

        auto inInfo = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
        std::vector<int64_t> expected = inInfo.GetShape();
        bool ok = expected.size() == inputShape_.size();
        for (size_t i = 0; ok && i < expected.size(); ++i) {
            ok = expected[i] < 0 || expected[i] == inputShape_[i];
        }
        if (!ok) {
            err = "ORT: model input shape disagrees with the benchmark tensor";
            return false;
        }

        outElems_ = 1;
        for (int64_t d : session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()) {
            outElems_ *= (d < 0 ? 1 : d);
        }
        return true;
    }

    void unload() override {
        session_.reset();
        env_.reset();
    }

    void setThreads(int n) override { threads_ = n; }

    void setConfig(const std::string& cfg) override { cfg_ = cfg; }

    bool forward(const float* input, int64_t inElems, float* output, int64_t,
                 std::string& err) override {
        try {
            Ort::MemoryInfo info = Ort::MemoryInfo::CreateCpu(
                    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
            // Wraps the caller's buffer rather than copying into the runtime.
            Ort::Value tensor = Ort::Value::CreateTensor<float>(
                    info, const_cast<float*>(input), static_cast<size_t>(inElems),
                    inputShape_.data(), inputShape_.size());
            const char* names[1] = {inputName_.c_str()};
            const char* outNames[1] = {outputName_.c_str()};
            auto out = session_->Run(Ort::RunOptions{nullptr}, names, &tensor, 1, outNames, 1);
            const float* src = out[0].GetTensorData<float>();
            std::memcpy(output, src, static_cast<size_t>(outElems_) * sizeof(float));
        } catch (const Ort::Exception& e) {
            err = std::string("ORT run: ") + e.what();
            return false;
        }
        return true;
    }

    int64_t outputElems() const override { return outElems_; }

private:
    int threads_ = 1;
    std::string cfg_;
    std::vector<int64_t> inputShape_;
    std::string inputName_;
    std::string outputName_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    int64_t outElems_ = 0;
};

}  // namespace

std::unique_ptr<Backend> createOrtBackend() {
    return std::unique_ptr<Backend>(new OrtBackend());
}

}  // namespace pb
