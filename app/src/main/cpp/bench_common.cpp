#include "bench.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>

namespace pb {

int64_t nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

Stats computeStats(const std::vector<double>& ms) {
    Stats s;
    if (ms.empty()) return s;
    std::vector<double> v = ms;
    std::sort(v.begin(), v.end());
    s.samples = static_cast<int>(v.size());
    s.min_ms = v.front();
    s.max_ms = v.back();
    s.median_ms = v[v.size() / 2];
    s.p95_ms = v[std::min(v.size() - 1, static_cast<size_t>(std::ceil(0.95 * v.size())) - 1)];
    double sum = 0;
    for (double x : v) sum += x;
    s.mean_ms = sum / static_cast<double>(v.size());
    return s;
}

long readRssKb() {
    FILE* f = fopen("/proc/self/status", "re");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &kb);
            break;
        }
    }
    fclose(f);
    return kb;
}

long readPeakRssKb() {
    FILE* f = fopen("/proc/self/status", "re");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%ld", &kb);
            break;
        }
    }
    fclose(f);
    return kb;
}

std::vector<float> readF32(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> out(static_cast<size_t>(n) / sizeof(float));
    size_t got = fread(out.data(), 1, static_cast<size_t>(n), f);
    fclose(f);
    if (got != static_cast<size_t>(n)) return {};
    return out;
}

bool readU8(const std::string& path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.resize(static_cast<size_t>(n));
    size_t got = fread(out.data(), 1, static_cast<size_t>(n), f);
    fclose(f);
    return got == static_cast<size_t>(n);
}

}  // namespace pb
