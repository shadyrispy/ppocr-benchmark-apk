#include "bench.h"

#include <Interpreter.hpp>
#include <MNNDefine.h>
#include <Tensor.hpp>

#include <cstring>
#include <vector>

namespace pb {
namespace {

// MNN requires an explicit host->device upload, so this backend pays exactly
// one memcpy that ORT/ncnn/lw avoid by wrapping the caller's pointer. Kept to
// a single copy per call; noted in the report.
class MnnBackend : public Backend {
public:
    const char* name() const override { return "MNN"; }

    bool load(const std::string& path, const std::vector<int>& inputShape,
              std::string& err) override {
        net_.reset(MNN::Interpreter::createFromFile(path.c_str()));
        if (!net_) {
            err = "MNN: createFromFile failed";
            return false;
        }
        inputShape_.assign(inputShape.begin(), inputShape.end());

        MNN::ScheduleConfig cfg;
        if (cfg_ == "opencl" || cfg_ == "opencl-fp16") {
            cfg.type = MNN_FORWARD_OPENCL;
        } else if (cfg_ == "vulkan" || cfg_ == "vulkan-fp16") {
            cfg.type = MNN_FORWARD_VULKAN;
        } else if (cfg_ == "nn") {
            cfg.type = MNN_FORWARD_NN;  // NNAPI
        } else {
            cfg.type = MNN_FORWARD_CPU;
        }
        cfg.numThread = threads_;
        MNN::BackendConfig backend;
        // Precision_Low is what actually turns on fp16 math for the GPU
        // backends; on CPU it only relaxes accumulation, so keep High there.
        const bool fp16 = cfg_ == "opencl-fp16" || cfg_ == "vulkan-fp16";
        backend.precision = fp16 ? MNN::BackendConfig::Precision_Low
                                 : (cfg.type == MNN_FORWARD_CPU
                                            ? MNN::BackendConfig::Precision_High
                                            : MNN::BackendConfig::Precision_Normal);
        backend.power = MNN::BackendConfig::Power_High;
        cfg.backendConfig = &backend;
        session_ = net_->createSession(cfg);
        if (!session_) {
            err = "MNN: createSession failed";
            return false;
        }
        input_ = net_->getSessionInput(session_, nullptr);
        output_ = net_->getSessionOutput(session_, nullptr);
        if (!input_ || !output_) {
            err = "MNN: missing graph io";
            return false;
        }
        net_->resizeTensor(input_, inputShape_);
        net_->resizeSession(session_);

        outElems_ = 1;
        for (int d : output_->shape()) outElems_ *= static_cast<int64_t>(d);
        return true;
    }

    void unload() override {
        if (net_ && session_) net_->releaseSession(session_);
        net_.reset();
        session_ = nullptr;
    }

    void setThreads(int n) override { threads_ = n; }

    void setConfig(const std::string& cfg) override { cfg_ = cfg; }

    bool forward(const float* input, int64_t, float* output, int64_t,
                 std::string& err) override {
        MNN::Tensor host(input_, MNN::Tensor::CAFFE);
        std::memcpy(host.host<float>(), input, host.size());
        input_->copyFromHostTensor(&host);
        if (net_->runSession(session_) != MNN::NO_ERROR) {
            err = "MNN: runSession failed";
            return false;
        }
        MNN::Tensor outHost(output_, MNN::Tensor::CAFFE);
        output_->copyToHostTensor(&outHost);
        std::memcpy(output, outHost.host<float>(), static_cast<size_t>(outElems_) * sizeof(float));
        return true;
    }

    int64_t outputElems() const override { return outElems_; }

private:
    int threads_ = 1;
    std::string cfg_;
    std::vector<int> inputShape_;
    std::shared_ptr<MNN::Interpreter> net_;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_ = nullptr;
    MNN::Tensor* output_ = nullptr;
    int64_t outElems_ = 0;
};

}  // namespace

std::unique_ptr<Backend> createMnnBackend() {
    return std::unique_ptr<Backend>(new MnnBackend());
}

}  // namespace pb
