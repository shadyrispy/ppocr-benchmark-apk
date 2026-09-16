// Offline ncnn check on macOS, linked against the release framework.
// Feeds the exact same tensors the Android harness uses, so a divergence here
// is either the pnnx conversion or my ncnn usage -- not Android.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <ncnn/net.h>

static std::vector<float> loadRaw(const std::string& p) {
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) { printf("cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> v(n / 4);
    fread(v.data(), 1, n, f);
    fclose(f);
    return v;
}

static void run(const char* model, const std::string& base, int c, int h, int w,
                int64_t expectElems) {
    printf("\n===== %s =====\n", model);
    auto input = loadRaw(base + "/" + model + ".input.f32");
    auto expect = loadRaw(base + "/" + model + ".expected.f32");
    printf("input elems=%zu expect elems=%zu (c=%d h=%d w=%d)\n", input.size(), expect.size(),
           c, h, w);

    ncnn::Net net;
    net.opt.num_threads = 1;
    net.opt.use_packing_layout = true;
    net.opt.use_fp16_storage = false;
    if (net.load_param((base + "/models/ncnn/" + model + ".ncnn.param").c_str())) {
        printf("load_param FAILED\n");
        return;
    }
    if (net.load_model((base + "/models/ncnn/" + model + ".ncnn.bin").c_str())) {
        printf("load_model FAILED\n");
        return;
    }
    printf("inputs=%zu outputs=%zu\n", net.input_names().size(), net.output_names().size());
    for (size_t i = 0; i < net.input_names().size(); i++)
        printf("  in  '%s'\n", net.input_names()[i]);
    for (size_t i = 0; i < net.output_names().size(); i++)
        printf("  out '%s'\n", net.output_names()[i]);

    ncnn::Extractor ex = net.create_extractor();
    ncnn::Mat in(w, h, c, input.data());
    int rc = ex.input(net.input_names()[0], in);
    printf("ex.input rc=%d\n", rc);
    ncnn::Mat out;
    rc = ex.extract(net.output_names()[0], out);
    printf("ex.extract rc=%d  -> w=%d h=%d d=%d c=%d elemsize=%zu total=%lld\n", rc, out.w,
           out.h, out.d, out.c, out.elemsize, (long long)out.w * out.h * out.d * out.c);
    if (rc != 0) return;

    int64_t total = (int64_t)out.w * out.h * out.d * out.c;
    {
        const float* q = (const float*)out.data;
        for (int i = 0; i < 8 && i < total; i++) printf("  raw[%d]=%.5f\n", i, q[i]);
    }
    if (total != expectElems) {
        printf("!! shape mismatch, skipping numerics\n");
        return;
    }
    const float* p = (const float*)out.data;
    double maxd = 0;
    for (int64_t i = 0; i < total; i++) {
        double d = fabs((double)p[i] - expect[i]);
        if (d > maxd) maxd = d;
    }
    printf("max_abs_diff=%.6f\n", maxd);
    if (strcmp(model, "rec") == 0 && out.d == 1 && out.c == 1) {
        int T = out.h, C = out.w, agree = 0;
        for (int t = 0; t < T; t++) {
            int a = 0, b = 0;
            float va = p[t * C], vb = expect[(size_t)t * C];
            for (int c = 1; c < C; c++) {
                if (p[(size_t)t * C + c] > va) { va = p[(size_t)t * C + c]; a = c; }
                if (expect[(size_t)t * C + c] > vb) { vb = expect[(size_t)t * C + c]; b = c; }
            }
            if (a == b) agree++;
        }
        printf("argmax agreement: %d/%d (%.1f%%)\n", agree, T, 100.0 * agree / T);
    }
    for (int i = 0; i < 4 && i < total; i++)
        printf("  got=%.5f expect=%.5f\n", p[i], expect[i]);
}

int main(int argc, char** argv) {
    std::string base = argc > 1 ? argv[1] : ".";
    run("cls", base, 3, 80, 160, 2);
    run("rec", base, 3, 48, 320, 40 * 6906);
    run("det", base, 3, 960, 960, 960 * 960);
    return 0;
}
