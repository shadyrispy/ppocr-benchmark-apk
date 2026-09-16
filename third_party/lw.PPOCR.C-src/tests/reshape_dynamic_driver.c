#include "lw_infer.h"

#include <inttypes.h>
#include <stdio.h>

static void make_input(lw_tensor_desc* input) {
    lw_tensor_desc_init(input);
    input->dtype = LW_DTYPE_F32;
    input->rank = 3u;
    input->dimensions[0] = 2;
    input->dimensions[1] = 3;
    input->dimensions[2] = 4;
}

static int expect_valid(const char* path) {
    lw_model* model = NULL;
    lw_session* session = NULL;
    lw_tensor_desc input;
    lw_tensor_desc output;
    lw_error error;
    lw_status status;
    make_input(&input);
    lw_error_init(&error);
    status = lw_model_load(path, NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "valid fixture load failed: %s\n", error.message);
        return 0;
    }
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_OK) {
        fprintf(stderr, "valid fixture session failed: %s\n", error.message);
        lw_model_free(model);
        return 0;
    }
    lw_tensor_desc_init(&output);
    status = lw_session_get_output_desc(session, 0u, &output);
    if (status != LW_STATUS_OK || output.rank != 3u || output.dimensions[0] != 2 ||
        output.dimensions[1] != 6 || output.dimensions[2] != 2) {
        fprintf(stderr, "unexpected inferred output shape\n");
        lw_session_free(session);
        lw_model_free(model);
        return 0;
    }
    lw_session_free(session);
    lw_model_free(model);
    return 1;
}

static int expect_invalid(const char* path) {
    lw_model* model = NULL;
    lw_session* session = NULL;
    lw_tensor_desc input;
    lw_error error;
    lw_status status;
    make_input(&input);
    lw_error_init(&error);
    status = lw_model_load(path, NULL, &model, &error);
    if (status != LW_STATUS_OK) {
        if (status == LW_STATUS_OUT_OF_BOUNDS) {
            return 1;
        }
        fprintf(stderr, "invalid fixture load failed: %s\n", error.message);
        return 0;
    }
    lw_error_init(&error);
    status = lw_session_create(model, &input, 1u, NULL, &session, &error);
    if (status != LW_STATUS_INVALID_SHAPE || session != NULL) {
        fprintf(stderr, "invalid fixture was accepted: %s (%s)\n",
                lw_status_string(status), error.message);
        if (session != NULL) lw_session_free(session);
        lw_model_free(model);
        return 0;
    }
    lw_model_free(model);
    return 1;
}

int main(int argc, char** argv) {
    int index;
    if (argc != 6) {
        fprintf(stderr, "usage: reshape-dynamic-driver valid two-unknown non-divisible zero overflow\n");
        return 2;
    }
    if (!expect_valid(argv[1])) return 1;
    for (index = 2; index < argc; ++index) {
        if (!expect_invalid(argv[index])) return 1;
    }
    puts("dynamic reshape contract: ok");
    return 0;
}
