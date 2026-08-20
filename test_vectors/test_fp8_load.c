#include <stdio.h>

typedef struct zi_transformer zi_transformer_t;

zi_transformer_t *zi_transformer_load_safetensors(
    const char *model_dir, const char *transformer_path,
    int dim, int n_heads, int n_layers, int n_refiner,
    int cap_feat_dim, int in_channels, int patch_size,
    float rope_theta, const int *axes_dims);
void iris_transformer_free_zimage(zi_transformer_t *tf);

int main(int argc, char **argv) {
    /* Require the standalone scaled FP8 transformer selected by the caller */
    if (argc != 2) {
        fprintf(stderr, "usage: %s transformer.safetensors\n", argv[0]);
        return 2;
    }

    /* Load the production Z-Image configuration without text encoding or inference */
    const int axes_dims[3] = {32, 48, 48};
    zi_transformer_t *tf = zi_transformer_load_safetensors(
        ".", argv[1], 3840, 30, 30, 2, 2560, 16, 2, 256.0f, axes_dims);
    if (!tf) {
        fprintf(stderr, "scaled FP8 transformer load failed\n");
        return 1;
    }

    /* Verify that mapped files and temporary CPU storage can be released cleanly */
    iris_transformer_free_zimage(tf);
    puts("FP8 transformer load passed");
    return 0;
}
