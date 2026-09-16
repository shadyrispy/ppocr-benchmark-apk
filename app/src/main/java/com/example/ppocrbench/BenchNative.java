package com.example.ppocrbench;

final class BenchNative {
    static {
        System.loadLibrary("ppocr_bench");
    }

    static native void setDataDir(String dir);

    // config: "" = fp32 CPU baseline. See Backend::setConfig in bench.h for the
    // full list (nnapi / opencl / vulkan / *-fp16 / int8).
    static native String runForwardCase(String engine, String model, String config,
                                        int threads, int warmup, int iters);

    static native String runDetPipeline(String engine, String config, int threads,
                                        int warmup, int iters);
}
