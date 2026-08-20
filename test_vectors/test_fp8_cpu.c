#include "../iris_kernels.h"
#include "../iris_safetensors.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Provide the loader utility dependency without linking the full inference engine */
char *iris_strdup(const char *s) {
    /* Duplicate the test string with ordinary process memory */
    size_t bytes = strlen(s) + 1;
    char *copy = (char *)malloc(bytes);
    if (copy) memcpy(copy, s, bytes);
    return copy;
}

int main(void) {
    /* Compare the vectorizable converter against the format reference for every byte */
    uint8_t encoded[256];
    float converted[256];
    for (int i = 0; i < 256; i++) encoded[i] = (uint8_t)i;
    iris_f8_e4m3_to_f32(converted, encoded, 256, 0.375f);
    for (int i = 0; i < 256; i++) {
        float expected = safetensor_f8_e4m3_to_f32(encoded[i]) * 0.375f;
        if ((isnan(expected) && !isnan(converted[i])) ||
            (!isnan(expected) && memcmp(&expected, &converted[i], sizeof(float)) != 0)) {
            fprintf(stderr, "FP8 conversion mismatch at 0x%02x\n", i);
            return 1;
        }
    }

    /* Compare a tiled FP8 projection with a direct decoded reference multiplication */
    enum { M = 3, K = 19, N = 5 };
    float input[M * K];
    uint8_t weights[N * K];
    float decoded[N * K];
    float actual[M * N];
    float expected[M * N];
    float workspace[2 * K];
    for (int i = 0; i < M * K; i++) input[i] = (float)((i % 13) - 6) * 0.125f;
    for (int i = 0; i < N * K; i++) weights[i] = (uint8_t)((i * 37 + 11) & 0x7e);
    iris_f8_e4m3_to_f32(decoded, weights, N * K, 0.25f);
    if (!iris_matmul_t_f8_e4m3(actual, input, weights, 0.25f,
                               M, K, N, workspace, 2 * K)) {
        fprintf(stderr, "FP8 matrix multiplication rejected valid inputs\n");
        return 1;
    }
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += input[m * K + k] * decoded[n * K + k];
            expected[m * N + n] = sum;
        }
    }
    for (int i = 0; i < M * N; i++) {
        float tolerance = 1e-5f * (1.0f + fabsf(expected[i]));
        if (fabsf(actual[i] - expected[i]) > tolerance) {
            fprintf(stderr, "FP8 matrix mismatch at %d: %.9g != %.9g\n",
                    i, actual[i], expected[i]);
            return 1;
        }
    }

    /* Report success for command-line test runners */
    puts("FP8 CPU tests passed");
    return 0;
}
