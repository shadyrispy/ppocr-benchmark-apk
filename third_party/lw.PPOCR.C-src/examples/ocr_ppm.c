#include "lw_infer.h"

/* Full DET/CLS/REC example using only the public C ABI and a tiny PPM loader. */
#include "ppm_image.h"

#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#  include <shellapi.h>
#endif

/* Capacities cross the public ABI as fixed-width integers. Check them through
 * uint64_t so the same source remains warning-clean on 32-bit and 64-bit. */
#define LW_OCR_DEMO_DEFAULT_REC_MAX_WIDTH 960u

static int valid_rec_max_width(uint32_t value) {
    return value == 192u || value == 320u || value == 480u || value == 640u || value == 960u;
}

static int parse_rec_max_width(const char* text, uint32_t* value) {
    char* end = NULL;
    unsigned long parsed;
    if (text == NULL || value == NULL || *text == '\0')
        return 0;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed > UINT32_MAX ||
        !valid_rec_max_width((uint32_t)parsed))
        return 0;
    *value = (uint32_t)parsed;
    return 1;
}

static int allocation_fits(uint64_t count, size_t element_size) {
    return element_size != 0u && count <= (uint64_t)(SIZE_MAX / element_size);
}

static int demo_main(int argc, char** argv) {
    lw_ocr* ocr = NULL;
    lw_ocr_info info;
    lw_ocr_result result;
    lw_ocr_line* lines = NULL;
    char* text = NULL;
    lw_error error;
    lw_example_ppm_image image;
    lw_status status;
    lw_ocr_options options;
    uint32_t rec_max_width = LW_OCR_DEMO_DEFAULT_REC_MAX_WIDTH;
    uint32_t index;
    int return_code = 1;
    memset(&image, 0, sizeof(image));
    if (argc != 6 && argc != 7) {
        fprintf(stderr, "usage: lw-ocr-ppm <det.lwm> <cls.lwm> <rec.lwm> "
                        "<dictionary.txt> <image.ppm> [rec-max-width]\n"
                        "rec-max-width: 192, 320, 480, 640, or 960 (default: 960)\n");
        return 2;
    }
    if (argc == 7 && !parse_rec_max_width(argv[6], &rec_max_width)) {
        fprintf(stderr, "invalid rec-max-width: %s\n", argv[6]);
        return 2;
    }
    if (!lw_example_ppm_image_load_bgr(argv[5], &image) || image.width > UINT32_MAX / 3u) {
        fprintf(stderr, "invalid P6 PPM image: %s\n", argv[5]);
        goto cleanup;
    }
    lw_error_init(&error);
    lw_ocr_options_init(&options);
    options.recognizer.target_width = rec_max_width;
    status = lw_ocr_create(argv[1], argv[2], argv[3], argv[4], &options, &ocr, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "create failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    lw_ocr_info_init(&info);
    if (lw_ocr_get_info(ocr, &info) != LW_STATUS_OK || info.max_line_capacity == 0u ||
        info.max_text_capacity == 0u ||
        !allocation_fits(info.max_line_capacity, sizeof(*lines)) ||
        !allocation_fits(info.max_text_capacity, sizeof(*text))) {
        fprintf(stderr, "unable to query OCR capacities\n");
        goto cleanup;
    }
    lines = (lw_ocr_line*)calloc(info.max_line_capacity, sizeof(*lines));
    text = (char*)malloc((size_t)info.max_text_capacity);
    if (lines == NULL || text == NULL) {
        fprintf(stderr, "unable to allocate OCR output buffers\n");
        goto cleanup;
    }
    lw_ocr_result_init(&result);
    lw_error_init(&error);
    status = lw_ocr_run_bgr_u8(ocr, image.pixels, image.byte_count, image.width, image.height,
                               image.width * 3u, lines, info.max_line_capacity, text,
                               info.max_text_capacity, &result, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "OCR failed: %s: %s\n", lw_status_string(status), error.message);
        goto cleanup;
    }
    printf("config rec_max_width=%u adaptive=%s\n", rec_max_width,
           rec_max_width > 320u ? "yes" : "no");
    printf("lines=%u image=%ux%u detector_input=%ux%u\n", result.line_count, image.width,
           image.height, result.detector_resized_width, result.detector_resized_height);
    for (index = 0u; index < result.line_count; ++index) {
        const lw_ocr_line* line = &lines[index];
        printf("%u text=%s rec=%.6f det=%.6f cls=%u/%.6f rotate=%u "
               "[(%.1f,%.1f),(%.1f,%.1f),(%.1f,%.1f),(%.1f,%.1f)]\n",
               index, text + line->text_offset, line->recognition_score, line->box.score,
               line->classification_label, line->classification_score,
               line->applied_rotation_degrees, line->box.x1, line->box.y1, line->box.x2,
               line->box.y2, line->box.x3, line->box.y3, line->box.x4, line->box.y4);
    }
    return_code = 0;
cleanup:
    free(text);
    free(lines);
    lw_ocr_free(ocr);
    lw_example_ppm_image_free(&image);
    return return_code;
}

#if defined(_WIN32)
int main(void) {
    wchar_t** wide_argv;
    char** utf8_argv;
    int argc;
    int index;
    int result = 2;
    SetConsoleOutputCP(CP_UTF8);
    wide_argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (wide_argv == NULL || argc <= 0)
        return 2;
    utf8_argv = (char**)calloc((size_t)argc, sizeof(*utf8_argv));
    if (utf8_argv == NULL) {
        LocalFree(wide_argv);
        return 2;
    }
    for (index = 0; index < argc; ++index) {
        int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1, NULL,
                                        0, NULL, NULL);
        if (bytes <= 0)
            goto cleanup;
        utf8_argv[index] = (char*)malloc((size_t)bytes);
        if (utf8_argv[index] == NULL ||
            WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, wide_argv[index], -1,
                                utf8_argv[index], bytes, NULL, NULL) <= 0)
            goto cleanup;
    }
    result = demo_main(argc, utf8_argv);
cleanup:
    for (index = 0; index < argc; ++index)
        free(utf8_argv[index]);
    free(utf8_argv);
    LocalFree(wide_argv);
    return result;
}
#else
int main(int argc, char** argv) {
    return demo_main(argc, argv);
}
#endif
