/*
 * Z-Image S3-DiT Transformer Implementation
 *
 * Implements the Z-Image-Turbo (6B) Scalable Single-Stream DiT architecture.
 *
 * Architecture:
 * - 2 noise_refiner blocks (modulated, image-only self-attention)
 * - 2 context_refiner blocks (unmodulated, text-only self-attention)
 * - 30 main transformer blocks (modulated, full self-attention)
 * - 30 heads, 128 dim per head (3840 hidden)
 * - 3-axis RoPE (32+48+48 = 128 dims, theta=256)
 * - SwiGLU activation (8/3 expansion)
 * - AdaLN modulation: scale + tanh(gate) only (no shift)
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include "iris_platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#if defined(USE_METAL) || defined(USE_VULKAN)
#include "iris_metal.h"
#define IRIS_ZIMAGE_GPU 1
#endif

/* ========================================================================
 * Constants
 * ======================================================================== */

#define ZI_SEQ_MULTI_OF     32      /* Pad sequences to multiples of 32 */
#define ZI_NORM_EPS         1e-5f   /* RMSNorm epsilon */
#define ZI_BF16_SDPA_SEQ    1024    /* Prefer bf16 SDPA at large sequence lengths */
#define ZI_MAX_SHARDS       32
#define ZI_FP8_PANEL_ROWS   256     /* Output rows decoded per CPU BLAS call */

/* Cumulative zImage timing counters (defined in iris_sample.c). */
extern double iris_timing_zi_total;
extern double iris_timing_zi_embeddings;
extern double iris_timing_zi_noise_refiner;
extern double iris_timing_zi_context_refiner;
extern double iris_timing_zi_main_blocks;
extern double iris_timing_zi_final;

static inline double zi_time_ms(void) {
    /* Use the platform timer implementation for this measurement. */
    return iris_time_ms();
}

/* ========================================================================
 * Data Structures
 * ======================================================================== */

/* Single transformer block weights */
typedef struct {
    const uint8_t *data;
    size_t elements;
    size_t offset;
    float scale;
} zi_fp8_weight_t;

typedef struct {
    /* Attention */
    float *attn_q_weight;       /* [dim, dim] */
    float *attn_k_weight;       /* [dim, dim] */
    float *attn_v_weight;       /* [dim, dim] */
    float *attn_out_weight;     /* [dim, dim] */
    float *attn_norm_q;         /* [n_heads, head_dim] for QK norm */
    float *attn_norm_k;         /* [n_heads, head_dim] */
    float *attn_norm1;          /* [dim] RMSNorm before attention */
    float *attn_norm2;          /* [dim] RMSNorm after attention */

    /* FFN (SwiGLU) */
    float *ffn_w1;              /* [ffn_dim, dim] gate projection */
    float *ffn_w2;              /* [dim, ffn_dim] down projection */
    float *ffn_w3;              /* [ffn_dim, dim] up projection */
    float *ffn_norm1;           /* [dim] RMSNorm before FFN */
    float *ffn_norm2;           /* [dim] RMSNorm after FFN */

    /* AdaLN modulation (NULL for context_refiner blocks) */
    float *adaln_weight;        /* [4*dim, adaln_dim] */
    float *adaln_bias;          /* [4*dim] */

    /* Mapped scaled E4M3 matrices shared by CPU and Vulkan paths */
    zi_fp8_weight_t attn_q_fp8;
    zi_fp8_weight_t attn_k_fp8;
    zi_fp8_weight_t attn_v_fp8;
    zi_fp8_weight_t attn_out_fp8;
    zi_fp8_weight_t ffn_w1_fp8;
    zi_fp8_weight_t ffn_w2_fp8;
    zi_fp8_weight_t ffn_w3_fp8;

#ifdef IRIS_ZIMAGE_GPU
    /* BF16 weight pointers for the GPU path */
    uint16_t *attn_q_weight_bf16;   /* [dim, dim] */
    uint16_t *attn_k_weight_bf16;   /* [dim, dim] */
    uint16_t *attn_v_weight_bf16;   /* [dim, dim] */
    uint16_t *attn_out_weight_bf16; /* [dim, dim] */
    uint16_t *ffn_w1_bf16;          /* [ffn_dim, dim] */
    uint16_t *ffn_w2_bf16;          /* [dim, ffn_dim] */
    uint16_t *ffn_w3_bf16;          /* [ffn_dim, dim] */
    unsigned int bf16_mapped_mask;  /* Native BF16 pointers owned by safetensors mappings */
    unsigned int bf16_cached_mask;  /* Converted BF16 matrices already uploaded to Vulkan */
    unsigned int f32_mapped_mask;   /* F32 fallback pointers owned by safetensors mappings */
    int bf16_host_released;         /* Temporary BF16 host buffers released after cache warmup */
#endif
} zi_block_t;

#ifdef IRIS_ZIMAGE_GPU
#define ZI_BF16_ATTN_Q   (1u << 0)
#define ZI_BF16_ATTN_K   (1u << 1)
#define ZI_BF16_ATTN_V   (1u << 2)
#define ZI_BF16_ATTN_OUT (1u << 3)
#define ZI_BF16_FFN_W1   (1u << 4)
#define ZI_BF16_FFN_W2   (1u << 5)
#define ZI_BF16_FFN_W3   (1u << 6)
#endif

/* Final layer weights */
typedef struct {
    float *adaln_weight;        /* [dim, adaln_dim] */
    float *adaln_bias;          /* [dim] */
    float *norm_weight;         /* NULL (no affine) or [dim] */
    float *linear_weight;       /* [out_ch, dim] */
    float *linear_bias;         /* [out_ch] */
} zi_final_t;

/* Z-Image transformer context */
typedef struct zi_transformer {
    /* Architecture config */
    int dim;                    /* 3840 */
    int n_heads;                /* 30 */
    int head_dim;               /* 128 */
    int n_layers;               /* 30 */
    int n_refiner;              /* 2 */
    int ffn_dim;                /* 8*dim/3 = 10240 */
    int in_channels;            /* 16 */
    int patch_size;             /* 2 */
    int adaln_dim;              /* min(dim, 256) = 256 */
    float rope_theta;           /* 256.0 */
    int axes_dims[3];           /* [32, 48, 48] */
    int axes_lens[3];           /* [1024, 512, 512] */

    /* Embedders */
    float *t_emb_mlp0_weight;   /* [mid_size, 256] */
    float *t_emb_mlp0_bias;     /* [mid_size] */
    float *t_emb_mlp2_weight;   /* [adaln_dim, mid_size] */
    float *t_emb_mlp2_bias;     /* [adaln_dim] */
    int t_emb_mid_size;         /* intermediate timestep MLP size */

    float *cap_emb_norm;        /* [cap_feat_dim] RMSNorm weight */
    float *cap_emb_linear_w;    /* [dim, cap_feat_dim] */
    float *cap_emb_linear_b;    /* [dim] */
    int cap_feat_dim;           /* 2560 */

    float *x_emb_weight;        /* [dim, patch_feat] where patch_feat = ps*ps*in_ch */
    float *x_emb_bias;          /* [dim] */

    float *x_pad_token;         /* [dim] */
    float *cap_pad_token;       /* [dim] */

    /* Transformer blocks */
    zi_block_t *noise_refiner;  /* [n_refiner] */
    zi_block_t *context_refiner;/* [n_refiner] */
    zi_block_t *layers;         /* [n_layers] */

    /* Final layer */
    zi_final_t final_layer;

    /* CPU mmap mode: keep shard files open and use direct f32 pointers. */
    int mmap_f32_weights;
    int mmap_bf16_weights;
    int fp8_weights;
    float *fp8_panel;
    size_t fp8_panel_elements;
    float *load_f32_workspace;
    uint16_t *load_bf16_workspace;
    safetensors_file_t *sf_files[ZI_MAX_SHARDS];
    int num_sf_files;

    /* Precomputed RoPE frequencies (complex pairs) */
    float *rope_cos[3];         /* [axes_lens[i], axes_dims[i]/2] */
    float *rope_sin[3];         /* [axes_lens[i], axes_dims[i]/2] */

    /* Working memory */
    float *work_x;              /* Main token buffer */
    float *work_tmp;            /* Temporary buffer */
    float *work_qkv;            /* Q, K, V buffers */
    float *work_attn;           /* Attention scores */
    float *work_ffn;            /* FFN intermediate */
    size_t work_alloc;          /* Total allocated */
    int max_seq;                /* Max sequence length allocated for */

#ifdef IRIS_ZIMAGE_GPU
    int use_gpu;                /* 1 if GPU path available */
    /* Cached preassembled RoPE tables for GPU path (reused across steps) */
    int gpu_rope_img_seq;
    int gpu_rope_cap_seq;
    int gpu_rope_uni_seq;
    int gpu_rope_h_tokens;
    int gpu_rope_w_tokens;
    float *gpu_img_rope_cos;
    float *gpu_img_rope_sin;
    float *gpu_cap_rope_cos;
    float *gpu_cap_rope_sin;
    float *gpu_uni_rope_cos;
    float *gpu_uni_rope_sin;
#endif
} zi_transformer_t;

void iris_transformer_free_zimage(zi_transformer_t *tf);

#ifdef IRIS_ZIMAGE_GPU
static int zi_prepare_cpu_fallback(zi_transformer_t *tf);
#endif

static void zi_final_forward(float *out, const float *x, const zi_final_t *fl,
                             const float *t_emb, int seq, zi_transformer_t *tf);

#ifdef IRIS_ZIMAGE_GPU
/* GPU scratch buffers for block forward pass.
 * Pre-allocated once for max sequence length, reused across all blocks. */
typedef struct {
    int seq, dim, ffn_dim;
    iris_gpu_tensor_t norm;     /* [seq, dim] */
    iris_gpu_tensor_t q;        /* [seq, dim] */
    iris_gpu_tensor_t k;        /* [seq, dim] */
    iris_gpu_tensor_t v;        /* [seq, dim] */
    iris_gpu_tensor_t attn_out; /* [seq, dim] */
    iris_gpu_tensor_t proj;     /* [seq, dim] */
    iris_gpu_tensor_t norm2;    /* [seq, dim] */
    iris_gpu_tensor_t gate_up;  /* [seq, ffn_dim] */
    iris_gpu_tensor_t up;       /* [seq, ffn_dim] */
    iris_gpu_tensor_t down;     /* [seq, dim] */
    /* BF16 attention scratch (for SDPA path via iris_gpu_attention_fused_bf16) */
    iris_gpu_tensor_t q_bf16;       /* [seq, dim] bf16 */
    iris_gpu_tensor_t k_bf16;       /* [seq, dim] bf16 */
    iris_gpu_tensor_t v_bf16;       /* [seq, dim] bf16 */
    iris_gpu_tensor_t attn_out_bf16;/* [seq, dim] bf16 */
    float *mod;                     /* [4*dim] CPU modulation scratch */
    float *fused_attn_norm;         /* [dim] CPU fused RMS weight scratch */
    float *fused_ffn_norm;          /* [dim] CPU fused RMS weight scratch */
} zi_gpu_scratch_t;

static void zi_gpu_scratch_free(zi_gpu_scratch_t *s) {
    if (!s) return;
    if (s->norm) iris_gpu_tensor_free(s->norm);
    if (s->q) iris_gpu_tensor_free(s->q);
    if (s->k) iris_gpu_tensor_free(s->k);
    if (s->v) iris_gpu_tensor_free(s->v);
    if (s->attn_out) iris_gpu_tensor_free(s->attn_out);
    if (s->proj) iris_gpu_tensor_free(s->proj);
    if (s->norm2) iris_gpu_tensor_free(s->norm2);
    if (s->gate_up) iris_gpu_tensor_free(s->gate_up);
    if (s->up) iris_gpu_tensor_free(s->up);
    if (s->down) iris_gpu_tensor_free(s->down);
    if (s->q_bf16) iris_gpu_tensor_free(s->q_bf16);
    if (s->k_bf16) iris_gpu_tensor_free(s->k_bf16);
    if (s->v_bf16) iris_gpu_tensor_free(s->v_bf16);
    if (s->attn_out_bf16) iris_gpu_tensor_free(s->attn_out_bf16);
    if (s->mod) free(s->mod);
    if (s->fused_attn_norm) free(s->fused_attn_norm);
    if (s->fused_ffn_norm) free(s->fused_ffn_norm);
    memset(s, 0, sizeof(*s));
}

static int zi_gpu_scratch_init(zi_gpu_scratch_t *s, int seq, int dim, int ffn_dim) {
    memset(s, 0, sizeof(*s));
    s->seq = seq;
    s->dim = dim;
    s->ffn_dim = ffn_dim;

    s->norm = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->q = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->k = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->v = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->attn_out = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->proj = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->norm2 = iris_gpu_tensor_alloc((size_t)seq * dim);
    s->gate_up = iris_gpu_tensor_alloc((size_t)seq * ffn_dim);
    s->up = iris_gpu_tensor_alloc((size_t)seq * ffn_dim);
    s->down = iris_gpu_tensor_alloc((size_t)seq * dim);

    if (!s->norm || !s->q || !s->k || !s->v || !s->attn_out ||
        !s->proj || !s->norm2 || !s->gate_up || !s->up || !s->down) {
        zi_gpu_scratch_free(s);
        return 0;
    }

    {
        size_t qkv_elems = (size_t)seq * dim;
        s->q_bf16 = iris_gpu_tensor_alloc_f16(qkv_elems);
        s->k_bf16 = iris_gpu_tensor_alloc_f16(qkv_elems);
        s->v_bf16 = iris_gpu_tensor_alloc_f16(qkv_elems);
        s->attn_out_bf16 = iris_gpu_tensor_alloc_f16(qkv_elems);
        if (!s->q_bf16 || !s->k_bf16 || !s->v_bf16 || !s->attn_out_bf16) {
            zi_gpu_scratch_free(s);
            return 0;
        }
    }

    s->mod = (float *)malloc(4 * (size_t)dim * sizeof(float));
    s->fused_attn_norm = (float *)malloc((size_t)dim * sizeof(float));
    s->fused_ffn_norm = (float *)malloc((size_t)dim * sizeof(float));
    if (!s->mod || !s->fused_attn_norm || !s->fused_ffn_norm) {
        zi_gpu_scratch_free(s);
        return 0;
    }

    return 1;
}

#ifdef USE_VULKAN
typedef struct {
    int seq, dim, ffn_dim;
    iris_gpu_tensor_t norm;
    iris_gpu_tensor_t q;
    iris_gpu_tensor_t k;
    iris_gpu_tensor_t v;
    iris_gpu_tensor_t attn_out;
    iris_gpu_tensor_t proj;
    iris_gpu_tensor_t norm2;
    iris_gpu_tensor_t gate_up;
    iris_gpu_tensor_t up;
    iris_gpu_tensor_t down;
    float *mod;
    float *fused_attn_norm;
    float *fused_ffn_norm;
} zi_gpu_bf16_scratch_t;

static void zi_gpu_bf16_scratch_free(zi_gpu_bf16_scratch_t *s) {
    if (!s) return;
    /* Release the compact BF16 activation workspace */
    if (s->norm) iris_gpu_tensor_free(s->norm);
    if (s->q) iris_gpu_tensor_free(s->q);
    if (s->k) iris_gpu_tensor_free(s->k);
    if (s->v) iris_gpu_tensor_free(s->v);
    if (s->attn_out) iris_gpu_tensor_free(s->attn_out);
    if (s->proj) iris_gpu_tensor_free(s->proj);
    if (s->norm2) iris_gpu_tensor_free(s->norm2);
    if (s->gate_up) iris_gpu_tensor_free(s->gate_up);
    if (s->up) iris_gpu_tensor_free(s->up);
    if (s->down) iris_gpu_tensor_free(s->down);
    free(s->mod);
    free(s->fused_attn_norm);
    free(s->fused_ffn_norm);
    memset(s, 0, sizeof(*s));
}

static int zi_gpu_bf16_scratch_init(zi_gpu_bf16_scratch_t *s,
                                    int seq, int dim, int ffn_dim) {
    memset(s, 0, sizeof(*s));
    s->seq = seq;
    s->dim = dim;
    s->ffn_dim = ffn_dim;

    /* Allocate every transformer activation in native BF16 storage */
    s->norm = iris_gpu_tensor_alloc_f16((size_t)seq * dim);
    s->q = iris_gpu_tensor_alloc_f16((size_t)seq * dim);
    s->k = iris_gpu_tensor_alloc_f16((size_t)seq * dim);
    s->v = iris_gpu_tensor_alloc_f16((size_t)seq * dim);
    s->attn_out = iris_gpu_tensor_alloc_f16((size_t)seq * dim);
    s->proj = iris_gpu_tensor_alloc_f16((size_t)seq * dim);
    s->norm2 = iris_gpu_tensor_alloc_f16((size_t)seq * dim);
    s->gate_up = iris_gpu_tensor_alloc_f16((size_t)seq * ffn_dim);
    s->up = iris_gpu_tensor_alloc_f16((size_t)seq * ffn_dim);
    s->down = iris_gpu_tensor_alloc_f16((size_t)seq * dim);
    if (!s->norm || !s->q || !s->k || !s->v || !s->attn_out ||
        !s->proj || !s->norm2 || !s->gate_up || !s->up || !s->down) {
        zi_gpu_bf16_scratch_free(s);
        return 0;
    }

    /* Keep modulation and fused affine vectors in FP32 for numerical stability */
    s->mod = (float *)malloc(4u * (size_t)dim * sizeof(float));
    s->fused_attn_norm = (float *)malloc((size_t)dim * sizeof(float));
    s->fused_ffn_norm = (float *)malloc((size_t)dim * sizeof(float));
    if (!s->mod || !s->fused_attn_norm || !s->fused_ffn_norm) {
        zi_gpu_bf16_scratch_free(s);
        return 0;
    }
    return 1;
}
#endif

typedef struct {
    zi_gpu_scratch_t f32;
#ifdef USE_VULKAN
    zi_gpu_bf16_scratch_t bf16;
    int use_bf16;
#endif
} zi_gpu_graph_scratch_t;

static int zi_gpu_graph_scratch_init(zi_gpu_graph_scratch_t *s,
                                     int use_bf16, int seq, int dim,
                                     int ffn_dim) {
    memset(s, 0, sizeof(*s));
#ifdef USE_VULKAN
    /* Select the compact graph only for the fully supported FP8 model path */
    s->use_bf16 = use_bf16;
    if (s->use_bf16)
        return zi_gpu_bf16_scratch_init(&s->bf16, seq, dim, ffn_dim);
#else
    (void)use_bf16;
#endif
    return zi_gpu_scratch_init(&s->f32, seq, dim, ffn_dim);
}

static void zi_gpu_graph_scratch_free(zi_gpu_graph_scratch_t *s) {
    if (!s) return;
#ifdef USE_VULKAN
    /* Release the workspace matching the active graph precision */
    if (s->use_bf16) {
        zi_gpu_bf16_scratch_free(&s->bf16);
        return;
    }
#endif
    zi_gpu_scratch_free(&s->f32);
}

static void zi_build_rope_table(float *cos_out, float *sin_out,
                                 const int *pos_ids, int seq,
                                 zi_transformer_t *tf);

/* GPU linear projection writing into a preallocated f32 output tensor.
 * Tries bf16 weight path first (fast), falls back to f32 weights. The "into"
 * variant avoids allocating a new tensor each call, which matters when
 * running 30+ blocks per step. */
static int zi_gpu_linear_into_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x,
                                   const uint16_t *W_bf16, const void *bf16_cache_key,
                                   const float *W_f32,
                                   const zi_fp8_weight_t *W_fp8,
                                   int seq_len, int in_dim, int out_dim) {
    size_t n = (size_t)seq_len * (size_t)out_dim;

#ifdef USE_VULKAN
    /* Decode scaled E4M3FN weights directly in the Vulkan projection */
    if (W_fp8 && W_fp8->data && iris_gpu_linear_fp8_stream_into(
            out, x, W_fp8->data, W_fp8->elements, W_fp8->offset,
            W_fp8->scale, seq_len, in_dim, out_dim)) {
        return 1;
    }

    /* Stream mapped F32 weights through reusable BF16 Vulkan buffers */
    if (W_f32 && iris_gpu_linear_f32_stream_into(
            out, x, W_f32, seq_len, in_dim, out_dim)) {
        return 1;
    }

    /* Try the stable-key BF16 cache before the F32 path */
    if (bf16_cache_key && iris_gpu_linear_bf16_into_key(
            out, x, bf16_cache_key, W_bf16, seq_len, in_dim, out_dim)) {
        return 1;
    }
#else
    /* Try the pointer-key BF16 cache before the F32 path */
    if (W_bf16) {
        if (iris_gpu_linear_bf16_into(out, x, W_bf16, seq_len, in_dim, out_dim)) {
            return 1;
        }
#endif
    if (W_bf16) {
        iris_gpu_tensor_t tmp_bf16 = iris_gpu_linear_bf16(x, W_bf16, seq_len, in_dim, out_dim);
        if (tmp_bf16) {
            iris_gpu_copy_f32(out, tmp_bf16, n);
            iris_gpu_tensor_free(tmp_bf16);
            return 1;
        }
    }

    if (W_f32) {
        iris_gpu_tensor_t tmp_f32 = iris_gpu_linear(x, W_f32, NULL, seq_len, in_dim, out_dim);
        if (tmp_f32) {
            iris_gpu_copy_f32(out, tmp_f32, n);
            iris_gpu_tensor_free(tmp_f32);
            return 1;
        }
    }

    return 0;
}

/* Self-attention dispatcher for GPU. Tries bf16 SDPA first for large
 * sequences (>= 1024 tokens) since bf16 attention fits in memory better,
 * then falls back to f32 fused attention, then tries the other precision,
 * and finally falls back to the legacy f32->f16->f32 path. This cascading
 * fallback ensures attention works at any sequence length. */
static int zi_gpu_attention(iris_gpu_tensor_t out_f32,
                             iris_gpu_tensor_t q_f32, iris_gpu_tensor_t k_f32, iris_gpu_tensor_t v_f32,
                             int seq, int n_heads, int head_dim, float attn_scale,
                             zi_gpu_scratch_t *scratch) {
    int prefer_bf16 = (seq >= ZI_BF16_SDPA_SEQ);

    if (prefer_bf16) {
        if (iris_gpu_convert_f32_to_bf16_into(scratch->q_bf16, q_f32) &&
            iris_gpu_convert_f32_to_bf16_into(scratch->k_bf16, k_f32) &&
            iris_gpu_convert_f32_to_bf16_into(scratch->v_bf16, v_f32) &&
            iris_gpu_attention_fused_bf16(scratch->attn_out_bf16,
                                          scratch->q_bf16, scratch->k_bf16, scratch->v_bf16,
                                          seq, seq, n_heads, head_dim, attn_scale) &&
            iris_gpu_convert_bf16_to_f32_into(out_f32, scratch->attn_out_bf16)) {
            return 1;
        }
    }

    if (iris_gpu_attention_fused(out_f32, q_f32, k_f32, v_f32,
                                 seq, seq, n_heads, head_dim, attn_scale)) {
        return 1;
    }

    if (!prefer_bf16) {
        if (iris_gpu_convert_f32_to_bf16_into(scratch->q_bf16, q_f32) &&
            iris_gpu_convert_f32_to_bf16_into(scratch->k_bf16, k_f32) &&
            iris_gpu_convert_f32_to_bf16_into(scratch->v_bf16, v_f32) &&
            iris_gpu_attention_fused_bf16(scratch->attn_out_bf16,
                                          scratch->q_bf16, scratch->k_bf16, scratch->v_bf16,
                                          seq, seq, n_heads, head_dim, attn_scale) &&
            iris_gpu_convert_bf16_to_f32_into(out_f32, scratch->attn_out_bf16)) {
            return 1;
        }
    }

    return iris_gpu_attention_bf16(out_f32, q_f32, k_f32, v_f32,
                                    seq, seq, n_heads, head_dim, attn_scale);
}

static void zi_gpu_rope_cache_clear(zi_transformer_t *tf) {
    free(tf->gpu_img_rope_cos); tf->gpu_img_rope_cos = NULL;
    free(tf->gpu_img_rope_sin); tf->gpu_img_rope_sin = NULL;
    free(tf->gpu_cap_rope_cos); tf->gpu_cap_rope_cos = NULL;
    free(tf->gpu_cap_rope_sin); tf->gpu_cap_rope_sin = NULL;
    free(tf->gpu_uni_rope_cos); tf->gpu_uni_rope_cos = NULL;
    free(tf->gpu_uni_rope_sin); tf->gpu_uni_rope_sin = NULL;
    tf->gpu_rope_img_seq = 0;
    tf->gpu_rope_cap_seq = 0;
    tf->gpu_rope_uni_seq = 0;
    tf->gpu_rope_h_tokens = 0;
    tf->gpu_rope_w_tokens = 0;
}

/* Preassembles and caches RoPE cos/sin tables for the current image geometry
 * (H_tokens, W_tokens, cap_seq_len). The geometry is stable across denoising
 * steps, so this avoids rebuilding tables every transformer call. Invalidated
 * when dimensions change (e.g., different image size). Builds separate tables
 * for noise refiner (image-only), context refiner (caption-only), and main
 * blocks (unified [img, cap] sequence). */
static int zi_gpu_rope_cache_prepare(zi_transformer_t *tf,
                                      int cap_seq_len, int H_tokens, int W_tokens) {
    int img_valid_seq = H_tokens * W_tokens;
    int img_seq = ((img_valid_seq + ZI_SEQ_MULTI_OF - 1) / ZI_SEQ_MULTI_OF) * ZI_SEQ_MULTI_OF;
    int cap_seq = ((cap_seq_len + ZI_SEQ_MULTI_OF - 1) / ZI_SEQ_MULTI_OF) * ZI_SEQ_MULTI_OF;
    int uni_seq = img_seq + cap_seq;

    if (tf->gpu_img_rope_cos &&
        tf->gpu_rope_img_seq == img_seq &&
        tf->gpu_rope_cap_seq == cap_seq &&
        tf->gpu_rope_uni_seq == uni_seq &&
        tf->gpu_rope_h_tokens == H_tokens &&
        tf->gpu_rope_w_tokens == W_tokens) {
        return 1;
    }

    zi_gpu_rope_cache_clear(tf);

    int head_dim = tf->head_dim;
    tf->gpu_img_rope_cos = (float *)malloc((size_t)img_seq * head_dim * sizeof(float));
    tf->gpu_img_rope_sin = (float *)malloc((size_t)img_seq * head_dim * sizeof(float));
    tf->gpu_cap_rope_cos = (float *)malloc((size_t)cap_seq * head_dim * sizeof(float));
    tf->gpu_cap_rope_sin = (float *)malloc((size_t)cap_seq * head_dim * sizeof(float));
    tf->gpu_uni_rope_cos = (float *)malloc((size_t)uni_seq * head_dim * sizeof(float));
    tf->gpu_uni_rope_sin = (float *)malloc((size_t)uni_seq * head_dim * sizeof(float));

    if (!tf->gpu_img_rope_cos || !tf->gpu_img_rope_sin ||
        !tf->gpu_cap_rope_cos || !tf->gpu_cap_rope_sin ||
        !tf->gpu_uni_rope_cos || !tf->gpu_uni_rope_sin) {
        zi_gpu_rope_cache_clear(tf);
        return 0;
    }

    int cap_padded_for_pos = ((cap_seq_len + ZI_SEQ_MULTI_OF - 1) / ZI_SEQ_MULTI_OF)
                              * ZI_SEQ_MULTI_OF;
    int *img_pos = (int *)calloc((size_t)img_seq * 3, sizeof(int));
    int *cap_pos = (int *)calloc((size_t)cap_seq * 3, sizeof(int));
    int *uni_pos = (int *)malloc((size_t)uni_seq * 3 * sizeof(int));
    if (!img_pos || !cap_pos || !uni_pos) {
        free(img_pos);
        free(cap_pos);
        free(uni_pos);
        zi_gpu_rope_cache_clear(tf);
        return 0;
    }

    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int idx = h * W_tokens + w;
            img_pos[idx * 3 + 0] = cap_padded_for_pos + 1;
            img_pos[idx * 3 + 1] = h;
            img_pos[idx * 3 + 2] = w;
        }
    }

    for (int s = 0; s < cap_seq; s++) {
        cap_pos[s * 3 + 0] = 1 + s;
        cap_pos[s * 3 + 1] = 0;
        cap_pos[s * 3 + 2] = 0;
    }

    memcpy(uni_pos, img_pos, (size_t)img_seq * 3 * sizeof(int));
    memcpy(uni_pos + (size_t)img_seq * 3, cap_pos, (size_t)cap_seq * 3 * sizeof(int));

    zi_build_rope_table(tf->gpu_img_rope_cos, tf->gpu_img_rope_sin, img_pos, img_seq, tf);
    zi_build_rope_table(tf->gpu_cap_rope_cos, tf->gpu_cap_rope_sin, cap_pos, cap_seq, tf);
    zi_build_rope_table(tf->gpu_uni_rope_cos, tf->gpu_uni_rope_sin, uni_pos, uni_seq, tf);

    free(img_pos);
    free(cap_pos);
    free(uni_pos);

    tf->gpu_rope_img_seq = img_seq;
    tf->gpu_rope_cap_seq = cap_seq;
    tf->gpu_rope_uni_seq = uni_seq;
    tf->gpu_rope_h_tokens = H_tokens;
    tf->gpu_rope_w_tokens = W_tokens;
    return 1;
}

static void iris_warmup_bf16_zimage(zi_transformer_t *tf) {
    if (!tf || !tf->use_gpu) return;
    if (!iris_metal_available()) return;

#ifndef USE_VULKAN
    /* Keep the existing Metal-wide warmup behavior */
    size_t attn_elems = (size_t)tf->dim * tf->dim;
    size_t ffn_up_elems = (size_t)tf->ffn_dim * tf->dim;
    size_t ffn_down_elems = (size_t)tf->dim * tf->ffn_dim;

    zi_block_t *groups[3] = { tf->noise_refiner, tf->context_refiner, tf->layers };
    int counts[3] = { tf->n_refiner, tf->n_refiner, tf->n_layers };
    int all_cached = 1;

    for (int g = 0; g < 3; g++) {
        zi_block_t *blocks = groups[g];
        int n = counts[g];
        if (!blocks) continue;

        for (int i = 0; i < n; i++) {
            zi_block_t *b = &blocks[i];

            /* Upload each immutable matrix and retain its cache key */
            if (b->attn_q_weight_bf16) all_cached &= iris_metal_warmup_bf16(b->attn_q_weight_bf16, attn_elems);
            if (b->attn_k_weight_bf16) all_cached &= iris_metal_warmup_bf16(b->attn_k_weight_bf16, attn_elems);
            if (b->attn_v_weight_bf16) all_cached &= iris_metal_warmup_bf16(b->attn_v_weight_bf16, attn_elems);
            if (b->attn_out_weight_bf16) all_cached &= iris_metal_warmup_bf16(b->attn_out_weight_bf16, attn_elems);
            if (b->ffn_w1_bf16) all_cached &= iris_metal_warmup_bf16(b->ffn_w1_bf16, ffn_up_elems);
            if (b->ffn_w2_bf16) all_cached &= iris_metal_warmup_bf16(b->ffn_w2_bf16, ffn_down_elems);
            if (b->ffn_w3_bf16) all_cached &= iris_metal_warmup_bf16(b->ffn_w3_bf16, ffn_up_elems);
        }
    }
#else
    /* Retain each unique FP8 payload once to remove denoising-time Copy traffic */
    if (tf->fp8_weights) {
        zi_block_t *groups[3] = {tf->noise_refiner, tf->context_refiner, tf->layers};
        int counts[3] = {tf->n_refiner, tf->n_refiner, tf->n_layers};
        int all_cached = 1;
        for (int g = 0; g < 3; g++) {
            for (int i = 0; i < counts[g]; i++) {
                zi_block_t *block = &groups[g][i];
                zi_fp8_weight_t *weights[7] = {
                    &block->attn_q_fp8, &block->attn_k_fp8,
                    &block->attn_v_fp8, &block->attn_out_fp8,
                    &block->ffn_w1_fp8, &block->ffn_w2_fp8,
                    &block->ffn_w3_fp8
                };
                for (int j = 0; j < 7; j++) {
                    if (!weights[j]->data) continue;
                    int duplicate = 0;
                    for (int k = 0; k < j; k++) {
                        if (weights[k]->data == weights[j]->data &&
                            weights[k]->elements == weights[j]->elements) {
                            duplicate = 1;
                            break;
                        }
                    }
                    if (!duplicate && !iris_metal_warmup_fp8(
                            weights[j]->data, weights[j]->elements))
                        all_cached = 0;
                }
            }
        }

        /* Summarize whether denoising can proceed without streamed weight copies */
        size_t cached_bytes = iris_metal_fp8_cache_used();
        if (cached_bytes) {
            fprintf(stderr, "  Z-Image: cached %.2f GiB FP8 weights in VRAM%s\n",
                    (double)cached_bytes / (1024.0 * 1024.0 * 1024.0),
                    all_cached ? "" : " (remaining weights will stream)");
        }
    }
#endif
}
#endif /* IRIS_ZIMAGE_GPU */

/* ========================================================================
 * Forward declarations
 * ======================================================================== */

void iris_transformer_free_zimage(zi_transformer_t *tf);

/* Forward declarations for functions used by GPU path */
static void zi_patchify(float *out, const float *latent,
                         int in_ch, int H, int W, int ps);
static void zi_unpatchify(float *latent, const float *patches,
                            int in_ch, int H, int W, int ps);
static int zi_final_compute_scale(float *scale, const zi_final_t *fl,
                                   const float *t_emb, zi_transformer_t *tf);
static void zi_final_forward(float *out, const float *x, const zi_final_t *fl,
                               const float *t_emb, int seq, zi_transformer_t *tf);
static void zi_rms_norm(float *out, const float *x, const float *weight,
                         int rows, int dim, float eps);

/* ========================================================================
 * Timestep Embedding
 * ======================================================================== */

/* Converts scalar timestep to a 256-dim vector using log-spaced frequencies,
 * the same idea as the original Transformer positional encoding but here it
 * encodes the denoising step. The caller scales the input by 1000 before
 * calling (t * 1000.0f), mapping the [0,1] sigma range to [0,1000]. */
static void zi_sinusoidal_embedding(float *out, float t, int dim) {
    int half = dim / 2;
    float log_max_period = logf(10000.0f);
    for (int i = 0; i < half; i++) {
        float freq = expf(-log_max_period * (float)i / (float)half);
        float angle = t * freq;
        out[i] = cosf(angle);
        out[i + half] = sinf(angle);
    }
}

/* Projects the sinusoidal timestep embedding through an MLP
 * (Linear -> SiLU -> Linear) to produce the adaln_dim-sized conditioning
 * vector. This drives all AdaLN modulation in the transformer -- it is how
 * every block knows which denoising step it is operating on. */
static void zi_timestep_embed(zi_transformer_t *tf, float *out, float t) {
    float sin_emb[256];
    zi_sinusoidal_embedding(sin_emb, t * 1000.0f, 256);

    /* MLP: Linear(256 -> mid) + SiLU + Linear(mid -> adaln_dim) */
    int mid = tf->t_emb_mid_size;
    float *hidden = (float *)malloc(mid * sizeof(float));

    /* Linear 0 */
    iris_matmul_t(hidden, sin_emb, tf->t_emb_mlp0_weight, 1, 256, mid);
    for (int i = 0; i < mid; i++) hidden[i] += tf->t_emb_mlp0_bias[i];

    /* SiLU */
    iris_silu(hidden, mid);

    /* Linear 2 */
    iris_matmul_t(out, hidden, tf->t_emb_mlp2_weight, 1, mid, tf->adaln_dim);
    for (int i = 0; i < tf->adaln_dim; i++) out[i] += tf->t_emb_mlp2_bias[i];

    free(hidden);
}

/* ========================================================================
 * RoPE
 * ======================================================================== */

/* Precomputes cos/sin frequency tables for all 3 RoPE axes
 * (T=32 dims, H=48 dims, W=48 dims) up to max_pos=1024 per axis.
 * Uses theta=256.0, much smaller than the usual 10000, giving shorter-range
 * position sensitivity suited to Z-Image's spatial layout. Tables are
 * allocated once at load time and reused across all denoising steps. */
static void zi_precompute_rope(zi_transformer_t *tf) {
    for (int ax = 0; ax < 3; ax++) {
        int d = tf->axes_dims[ax];
        int half_d = d / 2;
        int max_pos = tf->axes_lens[ax];

        tf->rope_cos[ax] = (float *)malloc(max_pos * half_d * sizeof(float));
        tf->rope_sin[ax] = (float *)malloc(max_pos * half_d * sizeof(float));

        for (int pos = 0; pos < max_pos; pos++) {
            for (int i = 0; i < half_d; i++) {
                float freq = 1.0f / powf(tf->rope_theta, (float)(2 * i) / (float)d);
                float angle = (float)pos * freq;
                tf->rope_cos[ax][pos * half_d + i] = cosf(angle);
                tf->rope_sin[ax][pos * half_d + i] = sinf(angle);
            }
        }
    }
}

/* Applies 3-axis RoPE to Q or K in-place using consecutive-pair rotation:
 * (x0*cos - x1*sin, x1*cos + x0*sin) on elements (d, d+1).
 * Each axis section of head_dim (T=32, H=48, W=48) gets its own position
 * from pos_ids[s,3]. This differs from Flux's split-half convention --
 * Z-Image pairs adjacent elements (d, d+1) rather than (d, d+half). */
static void zi_apply_rope(float *x, const int *pos_ids, int seq, int n_heads,
                           zi_transformer_t *tf) {
    int head_dim = tf->head_dim;
    int offset = 0;

    for (int ax = 0; ax < 3; ax++) {
        int d = tf->axes_dims[ax];
        int half_d = d / 2;

        for (int s = 0; s < seq; s++) {
            int pos = pos_ids[s * 3 + ax];
            if (pos < 0 || pos >= tf->axes_lens[ax]) continue;

            const float *cos_tab = tf->rope_cos[ax] + pos * half_d;
            const float *sin_tab = tf->rope_sin[ax] + pos * half_d;

            for (int h = 0; h < n_heads; h++) {
                float *head = x + (s * n_heads + h) * head_dim + offset;
                for (int i = 0; i < half_d; i++) {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    float c = cos_tab[i];
                    float sn = sin_tab[i];
                    head[2 * i]     = x0 * c - x1 * sn;
                    head[2 * i + 1] = x1 * c + x0 * sn;
                }
            }
        }
        offset += d;
    }
}

/* ========================================================================
 * Block Forward Pass (BLAS)
 * ======================================================================== */

/* Run a CPU projection from either ordinary f32 or mapped scaled FP8 weights */
static int zi_cpu_linear(float *out, const float *x, const float *weight,
                          const zi_fp8_weight_t *fp8, int rows,
                          int in_dim, int out_dim, zi_transformer_t *tf) {
    /* Prefer compact mapped weights when the complete selected matrix is present */
    if (fp8 && fp8->data) {
        size_t matrix_elements = (size_t)in_dim * out_dim;
        if (fp8->offset > fp8->elements ||
            matrix_elements > fp8->elements - fp8->offset)
            return 0;
        return iris_matmul_t_f8_e4m3(
            out, x, fp8->data + fp8->offset, fp8->scale,
            rows, in_dim, out_dim, tf->fp8_panel, tf->fp8_panel_elements);
    }

    /* Preserve the existing f32 projection for unquantized models */
    if (!weight) return 0;
    iris_matmul_t(out, x, weight, rows, in_dim, out_dim);
    return 1;
}

/* RMSNorm: out = x * weight / sqrt(mean(x^2) + eps) */
static void zi_rms_norm(float *out, const float *x, const float *weight,
                         int rows, int dim, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * dim;
        float *or_ = out + r * dim;
        float sum_sq = 0;
        for (int i = 0; i < dim; i++) sum_sq += xr[i] * xr[i];
        float rms = 1.0f / sqrtf(sum_sq / dim + eps);
        for (int i = 0; i < dim; i++) or_[i] = xr[i] * rms * weight[i];
    }
}

/* Per-head RMSNorm for QK normalization.
 * x: [seq, n_heads * head_dim], norm_weight: [head_dim] (shared across heads) */
static void zi_qk_norm(float *x, const float *norm_weight, int seq,
                         int n_heads, int head_dim, float eps) {
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < n_heads; h++) {
            float *ptr = x + s * n_heads * head_dim + h * head_dim;
            float sum_sq = 0;
            for (int i = 0; i < head_dim; i++) sum_sq += ptr[i] * ptr[i];
            float rms = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int i = 0; i < head_dim; i++) ptr[i] = ptr[i] * rms * norm_weight[i];
        }
    }
}

/* Scaled dot-product self-attention on the CPU path.
 * Computes Q@K^T per head, applies padding mask (sets masked positions to
 * -1e9 so softmax zeros them out), then scores@V. The mask distinguishes
 * real tokens from padding in the sequence. This is the slow reference path;
 * the GPU path uses fused SDPA kernels instead. */
static void zi_attention(float *out, const float *x,
                          const zi_block_t *block, const int *pos_ids,
                          const int *mask, int seq,
                          zi_transformer_t *tf) {
    int dim = tf->dim;
    int n_heads = tf->n_heads;
    int head_dim = tf->head_dim;

    float *q = tf->work_qkv;
    float *k = q + seq * dim;
    float *v = k + seq * dim;

    /* Q, K, V projections */
    if (!zi_cpu_linear(q, x, block->attn_q_weight, &block->attn_q_fp8,
                       seq, dim, dim, tf) ||
        !zi_cpu_linear(k, x, block->attn_k_weight, &block->attn_k_fp8,
                       seq, dim, dim, tf) ||
        !zi_cpu_linear(v, x, block->attn_v_weight, &block->attn_v_fp8,
                       seq, dim, dim, tf)) {
        memset(out, 0, (size_t)seq * dim * sizeof(float));
        return;
    }

    /* QK normalization */
    zi_qk_norm(q, block->attn_norm_q, seq, n_heads, head_dim, ZI_NORM_EPS);
    zi_qk_norm(k, block->attn_norm_k, seq, n_heads, head_dim, ZI_NORM_EPS);

    /* Apply RoPE */
    zi_apply_rope(q, pos_ids, seq, n_heads, tf);
    zi_apply_rope(k, pos_ids, seq, n_heads, tf);

    /* Scaled dot-product attention per head */
    float scale = 1.0f / sqrtf((float)head_dim);
    /* Keep the attention scratch separate from the projected output buffer */
    float *attn_out = tf->work_ffn;

    for (int h = 0; h < n_heads; h++) {
        float *scores = tf->work_attn;

        /* Compute Q @ K^T for this head */
        for (int i = 0; i < seq; i++) {
            const float *qi = q + i * dim + h * head_dim;
            for (int j = 0; j < seq; j++) {
                const float *kj = k + j * dim + h * head_dim;
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += qi[d] * kj[d];
                scores[i * seq + j] = dot * scale;
            }
        }

        /* Apply mask: set padding positions to -inf */
        if (mask) {
            for (int i = 0; i < seq; i++) {
                for (int j = 0; j < seq; j++) {
                    if (!mask[j])
                        scores[i * seq + j] = -1e9f;
                }
            }
        }

        /* Softmax */
        iris_softmax(scores, seq, seq);

        /* Scores @ V */
        for (int i = 0; i < seq; i++) {
            float *oi = attn_out + i * dim + h * head_dim;
            memset(oi, 0, head_dim * sizeof(float));
            for (int j = 0; j < seq; j++) {
                float s = scores[i * seq + j];
                const float *vj = v + j * dim + h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    oi[d] += s * vj[d];
            }
        }
    }

    /* Output projection */
    if (!zi_cpu_linear(out, attn_out, block->attn_out_weight,
                       &block->attn_out_fp8, seq, dim, dim, tf))
        memset(out, 0, (size_t)seq * dim * sizeof(float));
}

/* SwiGLU FFN: silu(W1 @ x) * (W3 @ x) then W2 */
static void zi_ffn(float *out, const float *x, const zi_block_t *block,
                    int seq, zi_transformer_t *tf) {
    int dim = tf->dim;
    int ffn_dim = tf->ffn_dim;
    float *gate = tf->work_ffn;
    float *up = gate + seq * ffn_dim;

    /* W1 (gate) and W3 (up) projections */
    if (!zi_cpu_linear(gate, x, block->ffn_w1, &block->ffn_w1_fp8,
                       seq, dim, ffn_dim, tf) ||
        !zi_cpu_linear(up, x, block->ffn_w3, &block->ffn_w3_fp8,
                       seq, dim, ffn_dim, tf)) {
        memset(out, 0, (size_t)seq * dim * sizeof(float));
        return;
    }

    /* SiLU(gate) * up */
    int n = seq * ffn_dim;
    iris_silu(gate, n);
    for (int i = 0; i < n; i++) gate[i] *= up[i];

    /* W2 (down) projection */
    if (!zi_cpu_linear(out, gate, block->ffn_w2, &block->ffn_w2_fp8,
                       seq, ffn_dim, dim, tf))
        memset(out, 0, (size_t)seq * dim * sizeof(float));
}

/* One S3-DiT block on CPU. Two modes: modulated (noise_refiner + main layers)
 * applies AdaLN with scale and tanh-gated residuals; unmodulated
 * (context_refiner) is a plain pre-norm attention + FFN block. The modulation
 * uses 4 parameters per block: scale_msa, gate_msa, scale_mlp, gate_mlp.
 * Note: no additive shift in Z-Image's block modulation (unlike Flux's
 * AdaLN which has shift). */
static void zi_block_forward(float *x, const zi_block_t *block,
                              const int *pos_ids, const int *mask,
                              const float *t_emb, int seq,
                              zi_transformer_t *tf) {
    int dim = tf->dim;
    int n = seq * dim;
    float *attn_out = tf->work_tmp;
    float *norm_out = tf->work_tmp + n;
    float *scaled = tf->work_tmp + 2 * n;
    float *ffn_out = tf->work_tmp + 3 * n;

    if (!tf->work_tmp || tf->max_seq < seq) return;

    if (block->adaln_weight) {
        /* Modulated block: extract scale_msa, gate_msa, scale_mlp, gate_mlp */
        float mod[4 * dim];
        iris_matmul_t(mod, t_emb, block->adaln_weight, 1, tf->adaln_dim, 4 * dim);
        for (int i = 0; i < 4 * dim; i++) mod[i] += block->adaln_bias[i];

        float *scale_msa = mod;
        float *gate_msa  = mod + dim;
        float *scale_mlp = mod + 2 * dim;
        float *gate_mlp  = mod + 3 * dim;

        /* Apply tanh to gates, 1+scale */
        for (int i = 0; i < dim; i++) {
            scale_msa[i] = 1.0f + scale_msa[i];
            gate_msa[i] = tanhf(gate_msa[i]);
            scale_mlp[i] = 1.0f + scale_mlp[i];
            gate_mlp[i] = tanhf(gate_mlp[i]);
        }

        /* Attention: h = attention(norm1(x) * scale_msa) */
        zi_rms_norm(norm_out, x, block->attn_norm1, seq, dim, ZI_NORM_EPS);
        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim; i++)
                scaled[s * dim + i] = norm_out[s * dim + i] * scale_msa[i];

        zi_attention(attn_out, scaled, block, pos_ids, mask, seq, tf);

        /* x = x + gate_msa * norm2(attn_out) */
        zi_rms_norm(norm_out, attn_out, block->attn_norm2, seq, dim, ZI_NORM_EPS);

        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim; i++)
                x[s * dim + i] += gate_msa[i] * norm_out[s * dim + i];

        /* FFN: h = ffn(norm1(x) * scale_mlp) */
        zi_rms_norm(norm_out, x, block->ffn_norm1, seq, dim, ZI_NORM_EPS);
        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim; i++)
                scaled[s * dim + i] = norm_out[s * dim + i] * scale_mlp[i];

        zi_ffn(ffn_out, scaled, block, seq, tf);

        /* x = x + gate_mlp * norm2(ffn_out) */
        zi_rms_norm(norm_out, ffn_out, block->ffn_norm2, seq, dim, ZI_NORM_EPS);
        for (int s = 0; s < seq; s++)
            for (int i = 0; i < dim; i++)
                x[s * dim + i] += gate_mlp[i] * norm_out[s * dim + i];

    } else {
        /* Unmodulated block (context_refiner): no scale/gate */

        /* Attention: h = attention(norm1(x)) */
        zi_rms_norm(norm_out, x, block->attn_norm1, seq, dim, ZI_NORM_EPS);

        zi_attention(attn_out, norm_out, block, pos_ids, mask, seq, tf);

        /* x = x + norm2(attn_out) */
        zi_rms_norm(norm_out, attn_out, block->attn_norm2, seq, dim, ZI_NORM_EPS);
        for (int i = 0; i < n; i++) x[i] += norm_out[i];
        /* FFN */
        zi_rms_norm(norm_out, x, block->ffn_norm1, seq, dim, ZI_NORM_EPS);

        zi_ffn(ffn_out, norm_out, block, seq, tf);

        zi_rms_norm(norm_out, ffn_out, block->ffn_norm2, seq, dim, ZI_NORM_EPS);
        for (int i = 0; i < n; i++) x[i] += norm_out[i];
    }
}

/* ========================================================================
 * GPU Forward Pass (Metal)
 * ======================================================================== */

#ifdef IRIS_ZIMAGE_GPU

/* Convert f32 array to bf16 (CPU-side, for weight conversion at load time).
 * Uses round-to-nearest-even for best accuracy.
 * Caller owns the returned buffer. */
static int zi_f32_to_bf16_into(uint16_t *dst, const float *src, size_t n) {
    if (!dst || !src) return 0;

    /* Convert the source array into caller-owned BF16 storage */
    for (size_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &src[i], 4);
        /* Round to nearest even: add rounding bias, handle tie-breaking */
        uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        bits += rounding_bias;
        dst[i] = (uint16_t)(bits >> 16);
    }
    return 1;
}

static uint16_t *zi_f32_to_bf16(const float *src, size_t n) {
    uint16_t *dst = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!dst) return NULL;

    /* Convert into the newly allocated BF16 array */
    if (!zi_f32_to_bf16_into(dst, src, n)) {
        free(dst);
        return NULL;
    }
    return dst;
}

/* Build a pre-assembled [seq, head_dim] RoPE cos/sin table by merging 3 axes.
 * pos_ids: [seq, 3] with (T, H, W) per position.
 * The table has cos/sin values laid out as consecutive pairs so that
 * iris_gpu_rope_2d (axis_dim=head_dim) applies rotation across the full head. */
static void zi_build_rope_table(float *cos_out, float *sin_out,
                                 const int *pos_ids, int seq,
                                 zi_transformer_t *tf) {
    int head_dim = tf->head_dim;

    for (int s = 0; s < seq; s++) {
        int offset = 0;
        for (int ax = 0; ax < 3; ax++) {
            int d = tf->axes_dims[ax];
            int half_d = d / 2;
            int pos = pos_ids[s * 3 + ax];

            /* Clamp pos to valid range */
            if (pos < 0) pos = 0;
            if (pos >= tf->axes_lens[ax]) pos = tf->axes_lens[ax] - 1;

            const float *ax_cos = tf->rope_cos[ax] + pos * half_d;
            const float *ax_sin = tf->rope_sin[ax] + pos * half_d;

            /* Write as consecutive pairs [cos_0, cos_0, cos_1, cos_1, ...] */
            for (int i = 0; i < half_d; i++) {
                cos_out[s * head_dim + offset + 2 * i]     = ax_cos[i];
                cos_out[s * head_dim + offset + 2 * i + 1] = ax_cos[i];
                sin_out[s * head_dim + offset + 2 * i]     = ax_sin[i];
                sin_out[s * head_dim + offset + 2 * i + 1] = ax_sin[i];
            }
            offset += d;
        }
    }
}

/* GPU-accelerated block forward. Fuses norm weight with modulation scale on
 * CPU (one multiply per dim), then passes the fused weight to GPU RMSNorm to
 * avoid an extra GPU kernel. Uses fused QKV and W1/W3 matmuls when available.
 * Returns 0 on any failure so the caller can fall back to CPU. */
static int zi_block_forward_gpu(iris_gpu_tensor_t hidden_gpu,
                                 const zi_block_t *block,
                                 const float *rope_cos, const float *rope_sin,
                                 const float *t_emb, const float *precomputed_mod, int seq,
                                 zi_transformer_t *tf,
                                 zi_gpu_scratch_t *scratch) {
    if (!iris_metal_available() || !iris_metal_shaders_available()) return 0;
    if (!hidden_gpu || !scratch) return 0;

    int dim = tf->dim;
    int n_heads = tf->n_heads;
    int head_dim = tf->head_dim;
    int ffn_dim = tf->ffn_dim;

    if (block->adaln_weight) {
        /* ---- Modulated block ---- */

        const float *mod = precomputed_mod;
        if (!mod) {
            /* Fallback path: compute modulation on the fly. */
            float *scratch_mod = scratch->mod;
            iris_matmul_t(scratch_mod, t_emb, block->adaln_weight, 1, tf->adaln_dim, 4 * dim);
            for (int i = 0; i < 4 * dim; i++) scratch_mod[i] += block->adaln_bias[i];

            /* Apply 1+scale to scales, tanh to gates */
            for (int i = 0; i < dim; i++) {
                scratch_mod[i] = 1.0f + scratch_mod[i];
                scratch_mod[dim + i] = tanhf(scratch_mod[dim + i]);
                scratch_mod[2 * dim + i] = 1.0f + scratch_mod[2 * dim + i];
                scratch_mod[3 * dim + i] = tanhf(scratch_mod[3 * dim + i]);
            }
            mod = scratch_mod;
        }

        const float *scale_msa = mod;
        const float *gate_msa  = mod + dim;
        const float *scale_mlp = mod + 2 * dim;
        const float *gate_mlp  = mod + 3 * dim;

        /* CPU: fuse norm_weight * scale into a single weight for RMSNorm */
        float *fused_attn_norm = scratch->fused_attn_norm;
        for (int i = 0; i < dim; i++)
            fused_attn_norm[i] = block->attn_norm1[i] * scale_msa[i];

        /* GPU: RMSNorm with fused weight (= rms_norm(x) * attn_norm1 * scale_msa) */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, fused_attn_norm,
                               seq, dim, ZI_NORM_EPS);

        /* GPU: Project Q, K, and V separately because fused output is row-interleaved */
        if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                    block->attn_q_weight_bf16,
                                    &block->attn_q_weight_bf16, block->attn_q_weight,
                                    &block->attn_q_fp8,
                                    seq, dim, dim)) return 0;
        if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                    block->attn_k_weight_bf16,
                                    &block->attn_k_weight_bf16, block->attn_k_weight,
                                    &block->attn_k_fp8,
                                    seq, dim, dim)) return 0;
        if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                    block->attn_v_weight_bf16,
                                    &block->attn_v_weight_bf16, block->attn_v_weight,
                                    &block->attn_v_fp8,
                                    seq, dim, dim)) return 0;

        /* GPU: QK normalization */
        iris_gpu_qk_rms_norm(scratch->q, scratch->k,
                              block->attn_norm_q, block->attn_norm_k,
                              seq, n_heads, head_dim, ZI_NORM_EPS);

        /* GPU: RoPE (full head_dim, pre-assembled 3-axis table) */
        iris_gpu_rope_single_pair_f32(scratch->q, scratch->k,
                                      rope_cos, rope_sin,
                                      seq, n_heads, head_dim);

        /* GPU: Self-attention */
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        if (!zi_gpu_attention(scratch->attn_out, scratch->q, scratch->k, scratch->v,
                               seq, n_heads, head_dim, attn_scale, scratch)) {
            return 0;
        }

        /* GPU: Output projection */
        if (!zi_gpu_linear_into_f32(scratch->proj, scratch->attn_out,
                                    block->attn_out_weight_bf16,
                                    &block->attn_out_weight_bf16, block->attn_out_weight,
                                    &block->attn_out_fp8,
                                    seq, dim, dim)) return 0;

        /* GPU: attn_norm2 + gated residual: x += gate_msa * norm2(proj) */
        iris_gpu_rms_norm_f32(scratch->norm2, scratch->proj, block->attn_norm2,
                               seq, dim, ZI_NORM_EPS);
        iris_gpu_gated_add(hidden_gpu, gate_msa, scratch->norm2, seq, dim);

        /* CPU: fuse FFN norm weight * scale_mlp */
        float *fused_ffn_norm = scratch->fused_ffn_norm;
        for (int i = 0; i < dim; i++)
            fused_ffn_norm[i] = block->ffn_norm1[i] * scale_mlp[i];

        /* GPU: FFN input norm with fused weight */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, fused_ffn_norm,
                               seq, dim, ZI_NORM_EPS);

        /* GPU: Project the two SwiGLU inputs separately because fused output is row-interleaved */
        if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                    block->ffn_w1_bf16,
                                    &block->ffn_w1_bf16, block->ffn_w1,
                                    &block->ffn_w1_fp8,
                                    seq, dim, ffn_dim)) return 0;
        if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                    block->ffn_w3_bf16,
                                    &block->ffn_w3_bf16, block->ffn_w3,
                                    &block->ffn_w3_fp8,
                                    seq, dim, ffn_dim)) return 0;
        iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);

        /* GPU: FFN down projection */
        if (!zi_gpu_linear_into_f32(scratch->down, scratch->gate_up,
                                    block->ffn_w2_bf16,
                                    &block->ffn_w2_bf16, block->ffn_w2,
                                    &block->ffn_w2_fp8,
                                    seq, ffn_dim, dim)) return 0;

        /* GPU: ffn_norm2 + gated residual: x += gate_mlp * norm2(ffn_out) */
        iris_gpu_rms_norm_f32(scratch->norm2, scratch->down, block->ffn_norm2,
                               seq, dim, ZI_NORM_EPS);
        iris_gpu_gated_add(hidden_gpu, gate_mlp, scratch->norm2, seq, dim);

    } else {
        /* ---- Unmodulated block (context_refiner) ---- */

        /* GPU: RMSNorm (plain weight, no scale) */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, block->attn_norm1,
                               seq, dim, ZI_NORM_EPS);

        /* GPU: Project Q, K, and V separately because fused output is row-interleaved */
        if (!zi_gpu_linear_into_f32(scratch->q, scratch->norm,
                                    block->attn_q_weight_bf16,
                                    &block->attn_q_weight_bf16, block->attn_q_weight,
                                    &block->attn_q_fp8,
                                    seq, dim, dim)) return 0;
        if (!zi_gpu_linear_into_f32(scratch->k, scratch->norm,
                                    block->attn_k_weight_bf16,
                                    &block->attn_k_weight_bf16, block->attn_k_weight,
                                    &block->attn_k_fp8,
                                    seq, dim, dim)) return 0;
        if (!zi_gpu_linear_into_f32(scratch->v, scratch->norm,
                                    block->attn_v_weight_bf16,
                                    &block->attn_v_weight_bf16, block->attn_v_weight,
                                    &block->attn_v_fp8,
                                    seq, dim, dim)) return 0;

        /* GPU: QK normalization */
        iris_gpu_qk_rms_norm(scratch->q, scratch->k,
                              block->attn_norm_q, block->attn_norm_k,
                              seq, n_heads, head_dim, ZI_NORM_EPS);

        /* GPU: RoPE */
        iris_gpu_rope_single_pair_f32(scratch->q, scratch->k,
                                      rope_cos, rope_sin,
                                      seq, n_heads, head_dim);

        /* GPU: Self-attention */
        float attn_scale = 1.0f / sqrtf((float)head_dim);
        if (!zi_gpu_attention(scratch->attn_out, scratch->q, scratch->k, scratch->v,
                               seq, n_heads, head_dim, attn_scale, scratch)) {
            return 0;
        }

        /* GPU: Output projection */
        if (!zi_gpu_linear_into_f32(scratch->proj, scratch->attn_out,
                                    block->attn_out_weight_bf16,
                                    &block->attn_out_weight_bf16, block->attn_out_weight,
                                    &block->attn_out_fp8,
                                    seq, dim, dim)) return 0;

        /* GPU: norm2(proj) + residual: x += norm2(attn_out) */
        iris_gpu_rms_norm_f32(scratch->norm2, scratch->proj, block->attn_norm2,
                               seq, dim, ZI_NORM_EPS);
        iris_gpu_add_f32(hidden_gpu, hidden_gpu, scratch->norm2, seq * dim);

        /* GPU: FFN */
        iris_gpu_rms_norm_f32(scratch->norm, hidden_gpu, block->ffn_norm1,
                               seq, dim, ZI_NORM_EPS);
        /* GPU: Project the two SwiGLU inputs separately because fused output is row-interleaved */
        if (!zi_gpu_linear_into_f32(scratch->gate_up, scratch->norm,
                                    block->ffn_w1_bf16,
                                    &block->ffn_w1_bf16, block->ffn_w1,
                                    &block->ffn_w1_fp8,
                                    seq, dim, ffn_dim)) return 0;
        if (!zi_gpu_linear_into_f32(scratch->up, scratch->norm,
                                    block->ffn_w3_bf16,
                                    &block->ffn_w3_bf16, block->ffn_w3,
                                    &block->ffn_w3_fp8,
                                    seq, dim, ffn_dim)) return 0;
        iris_gpu_silu_mul(scratch->gate_up, scratch->up, seq * ffn_dim);
        if (!zi_gpu_linear_into_f32(scratch->down, scratch->gate_up,
                                    block->ffn_w2_bf16,
                                    &block->ffn_w2_bf16, block->ffn_w2,
                                    &block->ffn_w2_fp8,
                                    seq, ffn_dim, dim)) return 0;

        /* GPU: ffn_norm2 + residual: x += norm2(ffn_out) */
        iris_gpu_rms_norm_f32(scratch->norm2, scratch->down, block->ffn_norm2,
                               seq, dim, ZI_NORM_EPS);
        iris_gpu_add_f32(hidden_gpu, hidden_gpu, scratch->norm2, seq * dim);
    }

    return 1;
}

#ifdef USE_VULKAN
static int zi_block_forward_gpu_bf16(iris_gpu_tensor_t hidden_gpu,
                                     const zi_block_t *block,
                                     const float *rope_cos,
                                     const float *rope_sin,
                                     const float *t_emb,
                                     const float *precomputed_mod,
                                     int seq, zi_transformer_t *tf,
                                     zi_gpu_bf16_scratch_t *scratch) {
    if (!hidden_gpu || !block || !scratch || !tf->fp8_weights) return 0;
    int dim = tf->dim;
    int heads = tf->n_heads;
    int head_dim = tf->head_dim;
    int ffn_dim = tf->ffn_dim;
    size_t qkv_matrix = (size_t)dim * (size_t)dim;

    /* Compute or reuse the four modulation vectors for this block */
    const float *mod = precomputed_mod;
    if (block->adaln_weight && !mod) {
        iris_matmul_t(scratch->mod, t_emb, block->adaln_weight,
                      1, tf->adaln_dim, 4 * dim);
        for (int i = 0; i < 4 * dim; i++)
            scratch->mod[i] += block->adaln_bias[i];
        for (int i = 0; i < dim; i++) {
            scratch->mod[i] = 1.0f + scratch->mod[i];
            scratch->mod[dim + i] = tanhf(scratch->mod[dim + i]);
            scratch->mod[2 * dim + i] = 1.0f + scratch->mod[2 * dim + i];
            scratch->mod[3 * dim + i] = tanhf(scratch->mod[3 * dim + i]);
        }
        mod = scratch->mod;
    }
    const float *gate_msa = mod ? mod + dim : NULL;
    const float *gate_mlp = mod ? mod + 3 * dim : NULL;

    /* Fold attention modulation into the FP32 RMSNorm affine vector */
    const float *attn_norm = block->attn_norm1;
    if (mod) {
        for (int i = 0; i < dim; i++)
            scratch->fused_attn_norm[i] = block->attn_norm1[i] * mod[i];
        attn_norm = scratch->fused_attn_norm;
    }
    iris_gpu_rms_norm_bf16_f32_weight(scratch->norm, hidden_gpu,
                                       attn_norm, seq, dim, ZI_NORM_EPS);

    /* Fuse contiguous QKV projections and share each activation tile */
    int qkv_contiguous = block->attn_q_fp8.data &&
        block->attn_q_fp8.data == block->attn_k_fp8.data &&
        block->attn_q_fp8.data == block->attn_v_fp8.data &&
        block->attn_q_fp8.elements == block->attn_k_fp8.elements &&
        block->attn_q_fp8.elements == block->attn_v_fp8.elements &&
        block->attn_k_fp8.offset == block->attn_q_fp8.offset + qkv_matrix &&
        block->attn_v_fp8.offset == block->attn_q_fp8.offset + 2u * qkv_matrix &&
        block->attn_q_fp8.scale == block->attn_k_fp8.scale &&
        block->attn_q_fp8.scale == block->attn_v_fp8.scale;
    if (qkv_contiguous) {
        if (!iris_gpu_linear_fp8_qkv_bf16_into(
                scratch->q, scratch->k, scratch->v, scratch->norm,
                block->attn_q_fp8.data, block->attn_q_fp8.elements,
                block->attn_q_fp8.offset, block->attn_q_fp8.scale,
                seq, dim, dim)) return 0;
    } else {
        if (!iris_gpu_linear_fp8_bf16_into(
                scratch->q, scratch->norm, block->attn_q_fp8.data,
                block->attn_q_fp8.elements, block->attn_q_fp8.offset,
                block->attn_q_fp8.scale, seq, dim, dim) ||
            !iris_gpu_linear_fp8_bf16_into(
                scratch->k, scratch->norm, block->attn_k_fp8.data,
                block->attn_k_fp8.elements, block->attn_k_fp8.offset,
                block->attn_k_fp8.scale, seq, dim, dim) ||
            !iris_gpu_linear_fp8_bf16_into(
                scratch->v, scratch->norm, block->attn_v_fp8.data,
                block->attn_v_fp8.elements, block->attn_v_fp8.offset,
                block->attn_v_fp8.scale, seq, dim, dim)) return 0;
    }

    /* Keep normalization, RoPE, and attention output entirely in BF16 */
    iris_gpu_qk_rms_norm_bf16_f32_weight(
        scratch->q, scratch->k, block->attn_norm_q, block->attn_norm_k,
        seq, heads, head_dim, ZI_NORM_EPS);
    iris_gpu_rope_pair_bf16(scratch->q, scratch->k, rope_cos, rope_sin,
                            seq, heads, head_dim);
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    if (!iris_gpu_attention_fused_bf16(
            scratch->attn_out, scratch->q, scratch->k, scratch->v,
            seq, seq, heads, head_dim, attn_scale)) return 0;
    if (!iris_gpu_linear_fp8_bf16_into(
            scratch->proj, scratch->attn_out, block->attn_out_fp8.data,
            block->attn_out_fp8.elements, block->attn_out_fp8.offset,
            block->attn_out_fp8.scale, seq, dim, dim)) return 0;

    /* Normalize the attention result before its gated residual update */
    iris_gpu_rms_norm_bf16_f32_weight(
        scratch->norm2, scratch->proj, block->attn_norm2,
        seq, dim, ZI_NORM_EPS);
    if (gate_msa)
        iris_gpu_gated_add_bf16_f32_gate(
            hidden_gpu, gate_msa, scratch->norm2, seq, dim);
    else
        iris_gpu_add_bf16(hidden_gpu, hidden_gpu, scratch->norm2, seq * dim);

    /* Fold FFN modulation into the FP32 RMSNorm affine vector */
    const float *ffn_norm = block->ffn_norm1;
    if (mod) {
        for (int i = 0; i < dim; i++)
            scratch->fused_ffn_norm[i] = block->ffn_norm1[i] * mod[2 * dim + i];
        ffn_norm = scratch->fused_ffn_norm;
    }
    iris_gpu_rms_norm_bf16_f32_weight(
        scratch->norm, hidden_gpu, ffn_norm, seq, dim, ZI_NORM_EPS);

    /* Fuse gate and up projections before the in-place SwiGLU activation */
    if (!iris_gpu_linear_fp8_gate_up_bf16_into(
            scratch->gate_up, scratch->up, scratch->norm,
            block->ffn_w1_fp8.data, block->ffn_w1_fp8.elements,
            block->ffn_w1_fp8.offset, block->ffn_w1_fp8.scale,
            block->ffn_w3_fp8.data, block->ffn_w3_fp8.elements,
            block->ffn_w3_fp8.offset, block->ffn_w3_fp8.scale,
            seq, dim, ffn_dim)) return 0;
    iris_gpu_silu_mul_bf16(scratch->gate_up, scratch->up, seq * ffn_dim);
    if (!iris_gpu_linear_fp8_bf16_into(
            scratch->down, scratch->gate_up, block->ffn_w2_fp8.data,
            block->ffn_w2_fp8.elements, block->ffn_w2_fp8.offset,
            block->ffn_w2_fp8.scale, seq, ffn_dim, dim)) return 0;

    /* Normalize the FFN result before its final residual update */
    iris_gpu_rms_norm_bf16_f32_weight(
        scratch->norm2, scratch->down, block->ffn_norm2,
        seq, dim, ZI_NORM_EPS);
    if (gate_mlp)
        iris_gpu_gated_add_bf16_f32_gate(
            hidden_gpu, gate_mlp, scratch->norm2, seq, dim);
    else
        iris_gpu_add_bf16(hidden_gpu, hidden_gpu, scratch->norm2, seq * dim);
    return 1;
}
#endif

static int zi_block_forward_gpu_graph(iris_gpu_tensor_t hidden_gpu,
                                      const zi_block_t *block,
                                      const float *rope_cos,
                                      const float *rope_sin,
                                      const float *t_emb,
                                      const float *precomputed_mod,
                                      int seq, zi_transformer_t *tf,
                                      zi_gpu_graph_scratch_t *scratch) {
#ifdef USE_VULKAN
    /* Route FP8 models through the end-to-end BF16 transformer graph */
    if (scratch->use_bf16)
        return zi_block_forward_gpu_bf16(
            hidden_gpu, block, rope_cos, rope_sin, t_emb,
            precomputed_mod, seq, tf, &scratch->bf16);
#endif
    return zi_block_forward_gpu(
        hidden_gpu, block, rope_cos, rope_sin, t_emb,
        precomputed_mod, seq, tf, &scratch->f32);
}

/* Precompute modulation for one block:
 * mod_out layout = [scale_msa, gate_msa, scale_mlp, gate_mlp], each dim.
 * Scales are stored as (1 + scale), gates are tanh(gate). */
static int zi_precompute_block_modulation(float *mod_out, const zi_block_t *block,
                                          const float *t_emb, int adaln_dim, int dim) {
    if (!mod_out || !block || !block->adaln_weight || !block->adaln_bias || !t_emb) return 0;

    iris_matmul_t(mod_out, t_emb, block->adaln_weight, 1, adaln_dim, 4 * dim);
    for (int i = 0; i < 4 * dim; i++) mod_out[i] += block->adaln_bias[i];

    for (int i = 0; i < dim; i++) {
        mod_out[i] = 1.0f + mod_out[i];
        mod_out[dim + i] = tanhf(mod_out[dim + i]);
        mod_out[2 * dim + i] = 1.0f + mod_out[2 * dim + i];
        mod_out[3 * dim + i] = tanhf(mod_out[3 * dim + i]);
    }

    return 1;
}

/* Full GPU-accelerated Z-Image transformer forward pass. Pipeline:
 * CPU timestep embed + patchify + caption norm -> GPU embedding projections ->
 * GPU noise refiner (2 blocks, image only) -> GPU context refiner (2 blocks,
 * caption only) -> GPU concat [img, cap] -> GPU main blocks (30, unified) ->
 * GPU final layer -> CPU unpatchify. Pre-computes all block modulations once
 * per step. Uses batch mode to submit all GPU work in one command buffer.
 * Returns NULL on failure (caller falls back to CPU). */
static float *zi_transformer_forward_gpu(zi_transformer_t *tf,
                                          const float *latent,
                                          int latent_h, int latent_w,
                                          float timestep,
                                          const float *cap_feats,
                                          int cap_seq_len) {
#ifdef USE_VULKAN
    /* Restore FP8 weights after a preview or VAE decode reclaimed VRAM */
    if (tf->fp8_weights && iris_metal_fp8_cache_used() == 0) {
        iris_metal_clear_weight_cache_only();
        iris_warmup_bf16_zimage(tf);
    }
#endif
    int dim = tf->dim;
    int ps = tf->patch_size;
    int in_ch = tf->in_channels;
    int patch_feat = ps * ps * in_ch;

    int H_tokens = latent_h / ps;
    int W_tokens = latent_w / ps;
    int img_seq = H_tokens * W_tokens;
    int refiner_total = tf->n_refiner * 2;

    /* Match the CPU sequence padding so both paths execute the same graph */
    int img_padded = ((img_seq + ZI_SEQ_MULTI_OF - 1) / ZI_SEQ_MULTI_OF) * ZI_SEQ_MULTI_OF;
    int cap_padded = ((cap_seq_len + ZI_SEQ_MULTI_OF - 1) / ZI_SEQ_MULTI_OF) * ZI_SEQ_MULTI_OF;
    int unified_seq = img_padded + cap_padded;
    double t_embed_ms = 0.0, t_noise_ms = 0.0, t_context_ms = 0.0;
    double t_main_ms = 0.0, t_final_ms = 0.0;
    double stage_start = zi_time_ms();

    /* === CPU: Timestep embedding === */
    float t_emb[256];
    zi_timestep_embed(tf, t_emb, timestep);

    /* === CPU: Patchify image === */
    float *img_patches = (float *)malloc(img_seq * patch_feat * sizeof(float));
    if (!img_patches) return NULL;
    zi_patchify(img_patches, latent, in_ch, latent_h, latent_w, ps);

    /* === CPU: Caption RMSNorm === */
    float *cap_normed = (float *)malloc(cap_seq_len * tf->cap_feat_dim * sizeof(float));
    if (!cap_normed) {
        free(img_patches);
        return NULL;
    }
    zi_rms_norm(cap_normed, cap_feats, tf->cap_emb_norm,
                cap_seq_len, tf->cap_feat_dim, ZI_NORM_EPS);

    /* === Embed image/caption (prefer GPU linear, fall back to CPU) === */
    iris_gpu_tensor_t img_gpu = NULL;
    iris_gpu_tensor_t cap_gpu = NULL;

    /* Image projection on GPU */
    iris_gpu_tensor_t img_patch_gpu = iris_gpu_tensor_create(img_patches, (size_t)img_seq * patch_feat);
    if (img_patch_gpu) {
        img_gpu = iris_gpu_linear(img_patch_gpu, tf->x_emb_weight, tf->x_emb_bias,
                                  img_seq, patch_feat, dim);
        iris_gpu_tensor_free(img_patch_gpu);
    }

    /* Caption projection on GPU */
    iris_gpu_tensor_t cap_norm_gpu = iris_gpu_tensor_create(cap_normed, (size_t)cap_seq_len * tf->cap_feat_dim);
    if (cap_norm_gpu) {
        cap_gpu = iris_gpu_linear(cap_norm_gpu, tf->cap_emb_linear_w, tf->cap_emb_linear_b,
                                  cap_seq_len, tf->cap_feat_dim, dim);
        iris_gpu_tensor_free(cap_norm_gpu);
    }

    /* CPU fallback if either embedding projection failed */
    if (!img_gpu || !cap_gpu) {
        if (img_gpu) {
            iris_gpu_tensor_free(img_gpu);
            img_gpu = NULL;
        }
        if (cap_gpu) {
            iris_gpu_tensor_free(cap_gpu);
            cap_gpu = NULL;
        }

        float *img_emb = (float *)malloc((size_t)img_seq * dim * sizeof(float));
        float *cap_emb = (float *)malloc((size_t)cap_seq_len * dim * sizeof(float));
        if (!img_emb || !cap_emb) {
            free(img_emb);
            free(cap_emb);
            free(img_patches);
            free(cap_normed);
            return NULL;
        }

        iris_matmul_t(img_emb, img_patches, tf->x_emb_weight, img_seq, patch_feat, dim);
        for (int s = 0; s < img_seq; s++) {
            for (int i = 0; i < dim; i++) {
                img_emb[s * dim + i] += tf->x_emb_bias[i];
            }
        }

        iris_matmul_t(cap_emb, cap_normed, tf->cap_emb_linear_w,
                      cap_seq_len, tf->cap_feat_dim, dim);
        for (int s = 0; s < cap_seq_len; s++) {
            for (int i = 0; i < dim; i++) {
                cap_emb[s * dim + i] += tf->cap_emb_linear_b[i];
            }
        }

        img_gpu = iris_gpu_tensor_create(img_emb, (size_t)img_seq * dim);
        cap_gpu = iris_gpu_tensor_create(cap_emb, (size_t)cap_seq_len * dim);
        free(img_emb);
        free(cap_emb);
    }

    free(img_patches);
    free(cap_normed);

    /* Expand valid embeddings with the same pad tokens as the CPU reference */
    if (img_padded != img_seq || cap_padded != cap_seq_len) {
        float *img_valid = (float *)malloc((size_t)img_seq * dim * sizeof(float));
        float *cap_valid = (float *)malloc((size_t)cap_seq_len * dim * sizeof(float));
        float *img_full = (float *)calloc((size_t)img_padded * dim, sizeof(float));
        float *cap_full = (float *)calloc((size_t)cap_padded * dim, sizeof(float));
        if (!img_valid || !cap_valid || !img_full || !cap_full) {
            free(img_valid);
            free(cap_valid);
            free(img_full);
            free(cap_full);
            iris_gpu_tensor_free(img_gpu);
            iris_gpu_tensor_free(cap_gpu);
            return NULL;
        }
        iris_gpu_tensor_read(img_gpu, img_valid);
        iris_gpu_tensor_read(cap_gpu, cap_valid);
        memcpy(img_full, img_valid, (size_t)img_seq * dim * sizeof(float));
        memcpy(cap_full, cap_valid, (size_t)cap_seq_len * dim * sizeof(float));
        for (int s = img_seq; s < img_padded; s++)
            memcpy(img_full + (size_t)s * dim, tf->x_pad_token, dim * sizeof(float));
        for (int s = cap_seq_len; s < cap_padded; s++)
            memcpy(cap_full + (size_t)s * dim, tf->cap_pad_token, dim * sizeof(float));
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        img_gpu = iris_gpu_tensor_create(img_full, (size_t)img_padded * dim);
        cap_gpu = iris_gpu_tensor_create(cap_full, (size_t)cap_padded * dim);
        free(img_valid);
        free(cap_valid);
        free(img_full);
        free(cap_full);
    }

    /* === CPU: Pre-assemble RoPE tables (cached across steps) === */
    if (!zi_gpu_rope_cache_prepare(tf, cap_seq_len, H_tokens, W_tokens)) {
        if (img_gpu) iris_gpu_tensor_free(img_gpu);
        if (cap_gpu) iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }
    const float *img_rope_cos = tf->gpu_img_rope_cos;
    const float *img_rope_sin = tf->gpu_img_rope_sin;
    const float *cap_rope_cos = tf->gpu_cap_rope_cos;
    const float *cap_rope_sin = tf->gpu_cap_rope_sin;
    const float *uni_rope_cos = tf->gpu_uni_rope_cos;
    const float *uni_rope_sin = tf->gpu_uni_rope_sin;
    t_embed_ms = zi_time_ms() - stage_start;

    /* === GPU: Process embedded tokens === */
    if (!img_gpu || !cap_gpu) {
        if (img_gpu) iris_gpu_tensor_free(img_gpu);
        if (cap_gpu) iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }
#ifdef USE_VULKAN
    int use_bf16_graph = tf->fp8_weights;
    if (use_bf16_graph) {
        /* Convert embeddings once at the transformer graph boundary */
        iris_gpu_tensor_t img_bf16 = iris_gpu_tensor_f32_to_bf16(img_gpu);
        iris_gpu_tensor_t cap_bf16 = iris_gpu_tensor_f32_to_bf16(cap_gpu);
        if (!img_bf16 || !cap_bf16) {
            if (img_bf16) iris_gpu_tensor_free(img_bf16);
            if (cap_bf16) iris_gpu_tensor_free(cap_bf16);
            iris_gpu_tensor_free(img_gpu);
            iris_gpu_tensor_free(cap_gpu);
            return NULL;
        }
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        img_gpu = img_bf16;
        cap_gpu = cap_bf16;
        /* Report the selected graph once rather than once per denoising step */
        static int bf16_graph_reported;
        if (!bf16_graph_reported) {
            fprintf(stderr, "Vulkan Z-Image activations: end-to-end BF16, fused QKV/SwiGLU\n");
            bf16_graph_reported = 1;
        }
    }
#else
    int use_bf16_graph = 0;
#endif
    iris_gpu_tensor_set_persistent(img_gpu, 1);
    iris_gpu_tensor_set_persistent(cap_gpu, 1);

    /* Allocate scratch for max sequence length (unified_seq) */
    zi_gpu_graph_scratch_t scratch;
    if (!zi_gpu_graph_scratch_init(&scratch, use_bf16_graph,
                                   unified_seq, dim, tf->ffn_dim)) {
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }

    /* Precompute modulation once per step for all modulated blocks. */
    int n_mod_blocks = tf->n_refiner + tf->n_layers;
    float *step_mod = NULL;
    if (n_mod_blocks > 0) {
        step_mod = (float *)malloc((size_t)n_mod_blocks * 4 * dim * sizeof(float));
        if (step_mod) {
            int mod_idx = 0;
            int mod_ok = 1;
            for (int i = 0; i < tf->n_refiner && mod_ok; i++) {
                mod_ok = zi_precompute_block_modulation(
                    step_mod + (size_t)mod_idx * 4 * dim,
                    &tf->noise_refiner[i], t_emb, tf->adaln_dim, dim);
                mod_idx++;
            }
            for (int i = 0; i < tf->n_layers && mod_ok; i++) {
                mod_ok = zi_precompute_block_modulation(
                    step_mod + (size_t)mod_idx * 4 * dim,
                    &tf->layers[i], t_emb, tf->adaln_dim, dim);
                mod_idx++;
            }
            if (!mod_ok) {
                free(step_mod);
                step_mod = NULL;
            }
        }
    }

    iris_gpu_batch_begin();

    /* === Noise refiner: 2 modulated blocks on image tokens === */
    int gpu_ok = 1;
    int mod_idx = 0;
    stage_start = zi_time_ms();
    for (int i = 0; i < tf->n_refiner && gpu_ok; i++) {
        const float *block_mod = step_mod ? (step_mod + (size_t)mod_idx * 4 * dim) : NULL;
        mod_idx++;
        gpu_ok = zi_block_forward_gpu_graph(
            img_gpu, &tf->noise_refiner[i], img_rope_cos, img_rope_sin,
            t_emb, block_mod, img_padded, tf, &scratch);
#ifdef USE_VULKAN
        /* Submit each block separately to stay below the Windows GPU watchdog */
        if (gpu_ok) {
            iris_gpu_batch_end();
            iris_gpu_batch_begin();
        }
#endif
        if (gpu_ok && iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, refiner_total);
    }
    t_noise_ms = zi_time_ms() - stage_start;

    /* === Context refiner: 2 unmodulated blocks on caption tokens === */
    stage_start = zi_time_ms();
    for (int i = 0; i < tf->n_refiner && gpu_ok; i++) {
        gpu_ok = zi_block_forward_gpu_graph(
            cap_gpu, &tf->context_refiner[i], cap_rope_cos, cap_rope_sin,
            NULL, NULL, cap_padded, tf, &scratch);
#ifdef USE_VULKAN
        /* Submit each block separately to stay below the Windows GPU watchdog */
        if (gpu_ok) {
            iris_gpu_batch_end();
            iris_gpu_batch_begin();
        }
#endif
        if (gpu_ok && iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, tf->n_refiner + i, refiner_total);
    }
    t_context_ms = zi_time_ms() - stage_start;

    if (!gpu_ok) {
        iris_gpu_batch_end();
        zi_gpu_graph_scratch_free(&scratch);
        free(step_mod);
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }

    /* === Concatenate: unified = [img, cap] === */
    iris_gpu_tensor_t unified_gpu = use_bf16_graph
        ? iris_gpu_tensor_alloc_f16((size_t)unified_seq * dim)
        : iris_gpu_tensor_alloc((size_t)unified_seq * dim);
    if (!unified_gpu) {
        iris_gpu_batch_end();
        zi_gpu_graph_scratch_free(&scratch);
        free(step_mod);
        iris_gpu_tensor_free(img_gpu);
        iris_gpu_tensor_free(cap_gpu);
        return NULL;
    }
    iris_gpu_tensor_set_persistent(unified_gpu, 1);

    /* Copy img then cap into unified entirely on GPU (no CPU sync). */
    size_t img_elems = (size_t)img_padded * dim;
    size_t cap_elems = (size_t)cap_padded * dim;
    if (use_bf16_graph) {
        /* Concatenate image and caption tokens without widening BF16 storage */
        iris_gpu_copy_region_bf16(unified_gpu, 0, img_gpu, 0, img_elems);
        iris_gpu_copy_region_bf16(unified_gpu, img_elems, cap_gpu, 0, cap_elems);
    } else {
        iris_gpu_copy_region_f32(unified_gpu, 0, img_gpu, 0, img_elems);
        iris_gpu_copy_region_f32(unified_gpu, img_elems, cap_gpu, 0, cap_elems);
    }

    iris_gpu_tensor_free(img_gpu);
    iris_gpu_tensor_free(cap_gpu);

    /* === Main transformer: 30 modulated blocks on unified sequence === */
    stage_start = zi_time_ms();
    for (int i = 0; i < tf->n_layers && gpu_ok; i++) {
        const float *block_mod = step_mod ? (step_mod + (size_t)mod_idx * 4 * dim) : NULL;
        mod_idx++;
        gpu_ok = zi_block_forward_gpu_graph(
            unified_gpu, &tf->layers[i], uni_rope_cos, uni_rope_sin,
            t_emb, block_mod, unified_seq, tf, &scratch);
#ifdef USE_VULKAN
        /* Submit each block separately to stay below the Windows GPU watchdog */
        if (gpu_ok) {
            iris_gpu_batch_end();
            iris_gpu_batch_begin();
        }
#endif
        if (gpu_ok && iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->n_layers);
    }
    t_main_ms = zi_time_ms() - stage_start;

    if (!gpu_ok) {
        iris_gpu_batch_end();
        zi_gpu_graph_scratch_free(&scratch);
        free(step_mod);
        iris_gpu_tensor_free(unified_gpu);
        return NULL;
    }

    /* === Final layer: synchronize image tokens and use the exact LayerNorm path === */
    stage_start = zi_time_ms();
    int out_ch = ps * ps * in_ch;
    /* Read back the image tokens after all main-block work has completed */
    iris_gpu_batch_end();
    size_t unified_hidden_count = (size_t)unified_seq * dim;
    float *img_hidden = (float *)malloc(unified_hidden_count * sizeof(float));
    if (!img_hidden) {
        free(img_hidden);
        zi_gpu_graph_scratch_free(&scratch);
        free(step_mod);
        iris_gpu_tensor_free(unified_gpu);
        return NULL;
    }
    /* Widen once at the final graph boundary for the exact CPU final layer */
    iris_gpu_tensor_t readback_gpu = unified_gpu;
    if (use_bf16_graph) {
        readback_gpu = iris_gpu_tensor_bf16_to_f32(unified_gpu);
        if (!readback_gpu) {
            free(img_hidden);
            zi_gpu_graph_scratch_free(&scratch);
            free(step_mod);
            iris_gpu_tensor_free(unified_gpu);
            return NULL;
        }
    }
    iris_gpu_tensor_read(readback_gpu, img_hidden);
    if (readback_gpu != unified_gpu) iris_gpu_tensor_free(readback_gpu);
    zi_gpu_graph_scratch_free(&scratch);
    free(step_mod);
    iris_gpu_tensor_free(unified_gpu);

    /* Apply the reference LayerNorm, modulation, projection, and bias on CPU */
    float *final_out = (float *)malloc((size_t)img_seq * out_ch * sizeof(float));
    if (!final_out) {
        free(img_hidden);
        return NULL;
    }
    zi_final_forward(final_out, img_hidden, &tf->final_layer, t_emb, img_seq, tf);
    free(img_hidden);
    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);

    /* === CPU: Unpatchify === */
    float *output = (float *)calloc(in_ch * latent_h * latent_w, sizeof(float));
    zi_unpatchify(output, final_out, in_ch, latent_h, latent_w, ps);
    free(final_out);
    t_final_ms = zi_time_ms() - stage_start;

    /* Accumulate per-step zImage GPU timing. */
    iris_timing_zi_embeddings += t_embed_ms;
    iris_timing_zi_noise_refiner += t_noise_ms;
    iris_timing_zi_context_refiner += t_context_ms;
    iris_timing_zi_main_blocks += t_main_ms;
    iris_timing_zi_final += t_final_ms;
    iris_timing_zi_total += t_embed_ms + t_noise_ms + t_context_ms + t_main_ms + t_final_ms;

    return output;
}

#endif /* IRIS_ZIMAGE_GPU */

/* ========================================================================
 * Final Layer
 * ======================================================================== */

/* Final layer AdaLN modulation: scale = 1 + Linear(SiLU(t_emb)) */
static int zi_final_compute_scale(float *scale, const zi_final_t *fl,
                                   const float *t_emb, zi_transformer_t *tf) {
    if (!scale || !fl || !t_emb || !tf) return 0;

    float silu_emb[256];
    memcpy(silu_emb, t_emb, tf->adaln_dim * sizeof(float));
    iris_silu(silu_emb, tf->adaln_dim);

    iris_matmul_t(scale, silu_emb, fl->adaln_weight, 1, tf->adaln_dim, tf->dim);
    for (int i = 0; i < tf->dim; i++) scale[i] = 1.0f + scale[i] + fl->adaln_bias[i];
    return 1;
}

/* Z-Image final layer: LayerNorm (no affine) -> scale by
 * (1 + SiLU(Linear(t_emb))) -> Linear projection to patch channels.
 * Note the SiLU activation in the final layer's modulation -- this differs
 * from the block modulation which has no activation. Output shape is
 * [img_seq, patch_size^2 * in_channels]. */
static void zi_final_forward(float *out, const float *x, const zi_final_t *fl,
                               const float *t_emb, int seq, zi_transformer_t *tf) {
    int dim = tf->dim;
    int out_dim = tf->patch_size * tf->patch_size * tf->in_channels;

    float *scale = (float *)malloc(dim * sizeof(float));
    if (!scale || !zi_final_compute_scale(scale, fl, t_emb, tf)) {
        free(scale);
        return;
    }

    /* LayerNorm (no affine) -> scale */
    float *normed = (float *)malloc(seq * dim * sizeof(float));
    for (int s = 0; s < seq; s++) {
        const float *xr = x + s * dim;
        float *nr = normed + s * dim;

        /* Compute mean and variance */
        float mean = 0;
        for (int i = 0; i < dim; i++) mean += xr[i];
        mean /= dim;

        float var = 0;
        for (int i = 0; i < dim; i++) {
            float d = xr[i] - mean;
            var += d * d;
        }
        var /= dim;
        float inv_std = 1.0f / sqrtf(var + 1e-6f); /* Final LayerNorm uses 1e-6 */

        for (int i = 0; i < dim; i++)
            nr[i] = (xr[i] - mean) * inv_std * scale[i];
    }

    /* Linear projection: dim -> out_dim */
    iris_matmul_t(out, normed, fl->linear_weight, seq, dim, out_dim);
    for (int s = 0; s < seq; s++)
        for (int i = 0; i < out_dim; i++)
            out[s * out_dim + i] += fl->linear_bias[i];

    free(scale);
    free(normed);
}

/* ========================================================================
 * Patchify / Unpatchify
 * ======================================================================== */

/* Converts latent [in_ch, H, W] to patch sequence [n_patches, ps*ps*in_ch].
 * Gathers each ps x ps spatial block into a flat vector, ordering as
 * (ph, pw, channel). This is the inverse of unpatchify and creates the
 * token sequence the transformer operates on. */
static void zi_patchify(float *out, const float *latent,
                         int in_ch, int H, int W, int ps) {
    int H_tokens = H / ps;
    int W_tokens = W / ps;
    int patch_feat = ps * ps * in_ch;

    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int patch_idx = h * W_tokens + w;
            float *dst = out + patch_idx * patch_feat;
            int di = 0;

            /* Gather patch: iterate (ph, pw, c) */
            for (int ph = 0; ph < ps; ph++) {
                for (int pw = 0; pw < ps; pw++) {
                    for (int c = 0; c < in_ch; c++) {
                        int sy = h * ps + ph;
                        int sx = w * ps + pw;
                        dst[di++] = latent[c * H * W + sy * W + sx];
                    }
                }
            }
        }
    }
}

/* Unpatchify: [n_patches, patch_feat_dim] -> [in_ch, H, W] */
static void zi_unpatchify(float *latent, const float *patches,
                            int in_ch, int H, int W, int ps) {
    int H_tokens = H / ps;
    int W_tokens = W / ps;
    int patch_feat = ps * ps * in_ch;

    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int patch_idx = h * W_tokens + w;
            const float *src = patches + patch_idx * patch_feat;
            int si = 0;

            for (int ph = 0; ph < ps; ph++) {
                for (int pw = 0; pw < ps; pw++) {
                    for (int c = 0; c < in_ch; c++) {
                        int sy = h * ps + ph;
                        int sx = w * ps + pw;
                        latent[c * H * W + sy * W + sx] = src[si++];
                    }
                }
            }
        }
    }
}

/* ========================================================================
 * Main Forward Pass
 * ======================================================================== */

/* Top-level Z-Image transformer entry point. Tries GPU path first, falls
 * back to CPU on failure. CPU path pads sequences to multiples of 32 and
 * uses padding masks. Pipeline: patchify -> embed image/caption ->
 * noise refiner (image self-attention) -> context refiner (caption
 * self-attention) -> concatenate [image, caption] -> main blocks (full
 * self-attention) -> final layer -> unpatchify. */
float *iris_transformer_forward_zimage(zi_transformer_t *tf,
                                const float *latent,
                                int latent_h, int latent_w,
                                float timestep,
                                const float *cap_feats,
                                int cap_seq_len) {
#ifdef IRIS_ZIMAGE_GPU
    /* Try GPU-accelerated path first */
    if (tf->use_gpu) {
        float *result = zi_transformer_forward_gpu(tf, latent, latent_h, latent_w,
                                                    timestep, cap_feats, cap_seq_len);
        if (result) return result;
        /* Fall back to CPU on GPU failure */
        fprintf(stderr, "Z-Image GPU path failed, falling back to CPU\n");
        /* Reuse mapped FP8 directly or reload ordinary F32 fallback weights */
        if (!tf->fp8_weights && !tf->mmap_f32_weights &&
            !zi_prepare_cpu_fallback(tf)) {
            fprintf(stderr, "Z-Image CPU fallback unavailable after GPU failure\n");
            return NULL;
        }
        tf->use_gpu = 0;
    }
#endif

    int dim = tf->dim;
    int ps = tf->patch_size;
    int in_ch = tf->in_channels;
    int patch_feat = ps * ps * in_ch;  /* 64 */

    int H_tokens = latent_h / ps;
    int W_tokens = latent_w / ps;
    int img_seq = H_tokens * W_tokens;
    int refiner_total = tf->n_refiner * 2;

    /* Pad sequences to multiples of ZI_SEQ_MULTI_OF */
    int img_pad = (ZI_SEQ_MULTI_OF - (img_seq % ZI_SEQ_MULTI_OF)) % ZI_SEQ_MULTI_OF;
    int cap_pad = (ZI_SEQ_MULTI_OF - (cap_seq_len % ZI_SEQ_MULTI_OF)) % ZI_SEQ_MULTI_OF;
    int img_padded = img_seq + img_pad;
    int cap_padded = cap_seq_len + cap_pad;
    int unified_seq = img_padded + cap_padded;

    /* Ensure working memory is sufficient */
    size_t needed = (size_t)unified_seq * dim * 4 +
                    (size_t)unified_seq * dim * 3 +  /* QKV */
                    (size_t)unified_seq * unified_seq + /* attention scores */
                    (size_t)unified_seq * tf->ffn_dim * 2;
    if (needed > tf->work_alloc) {
        free(tf->work_x);
        free(tf->work_tmp);
        free(tf->work_qkv);
        free(tf->work_attn);
        free(tf->work_ffn);
        tf->work_x = (float *)malloc(unified_seq * dim * sizeof(float));
        tf->work_tmp = (float *)malloc(unified_seq * dim * 4 * sizeof(float));
        tf->work_qkv = (float *)malloc(unified_seq * dim * 3 * sizeof(float));
        tf->work_attn = (float *)malloc((size_t)unified_seq * unified_seq * sizeof(float));
        tf->work_ffn = (float *)malloc((size_t)unified_seq * tf->ffn_dim * 2 * sizeof(float));
        if (!tf->work_x || !tf->work_tmp || !tf->work_qkv || !tf->work_attn || !tf->work_ffn) {
            free(tf->work_x); tf->work_x = NULL;
            free(tf->work_tmp); tf->work_tmp = NULL;
            free(tf->work_qkv); tf->work_qkv = NULL;
            free(tf->work_attn); tf->work_attn = NULL;
            free(tf->work_ffn); tf->work_ffn = NULL;
            tf->work_alloc = 0;
            tf->max_seq = 0;
            return NULL;
        }
        tf->work_alloc = needed;
        tf->max_seq = unified_seq;
    }

    /* 1. Timestep embedding */
    float t_emb[256];
    zi_timestep_embed(tf, t_emb, timestep);

    /* 2. Patchify image -> [img_seq, patch_feat] */
    float *img_patches = (float *)malloc(img_padded * patch_feat * sizeof(float));
    if (!img_patches) return NULL;
    zi_patchify(img_patches, latent, in_ch, latent_h, latent_w, ps);

    /* Pad image patches (repeat last token) */
    for (int i = img_seq; i < img_padded; i++)
        memcpy(img_patches + i * patch_feat,
               img_patches + (img_seq - 1) * patch_feat,
               patch_feat * sizeof(float));

    /* Embed image: [img_padded, patch_feat] -> [img_padded, dim] */
    float *img_emb = (float *)malloc(img_padded * dim * sizeof(float));
    if (!img_emb) {
        free(img_patches);
        return NULL;
    }
    iris_matmul_t(img_emb, img_patches, tf->x_emb_weight, img_padded, patch_feat, dim);
    for (int s = 0; s < img_padded; s++)
        for (int i = 0; i < dim; i++)
            img_emb[s * dim + i] += tf->x_emb_bias[i];
    free(img_patches);

    /* Apply pad token to image padding positions */
    for (int s = img_seq; s < img_padded; s++)
        memcpy(img_emb + s * dim, tf->x_pad_token, dim * sizeof(float));

    /* 3. Caption embedding: RMSNorm -> Linear */
    float *cap_emb = (float *)malloc(cap_padded * dim * sizeof(float));
    float *cap_normed = (float *)malloc(cap_padded * tf->cap_feat_dim * sizeof(float));
    if (!cap_emb || !cap_normed) {
        free(img_emb);
        free(cap_emb);
        free(cap_normed);
        return NULL;
    }

    /* Pad caption features (repeat last token) */
    float *cap_padded_feats = (float *)malloc(cap_padded * tf->cap_feat_dim * sizeof(float));
    if (!cap_padded_feats) {
        free(img_emb);
        free(cap_emb);
        free(cap_normed);
        return NULL;
    }
    memcpy(cap_padded_feats, cap_feats, cap_seq_len * tf->cap_feat_dim * sizeof(float));
    for (int s = cap_seq_len; s < cap_padded; s++)
        memcpy(cap_padded_feats + s * tf->cap_feat_dim,
               cap_feats + (cap_seq_len - 1) * tf->cap_feat_dim,
               tf->cap_feat_dim * sizeof(float));

    zi_rms_norm(cap_normed, cap_padded_feats, tf->cap_emb_norm,
                cap_padded, tf->cap_feat_dim, ZI_NORM_EPS);
    free(cap_padded_feats);

    iris_matmul_t(cap_emb, cap_normed, tf->cap_emb_linear_w,
                  cap_padded, tf->cap_feat_dim, dim);
    for (int s = 0; s < cap_padded; s++)
        for (int i = 0; i < dim; i++)
            cap_emb[s * dim + i] += tf->cap_emb_linear_b[i];
    free(cap_normed);

    /* Apply pad token to caption padding positions */
    for (int s = cap_seq_len; s < cap_padded; s++)
        memcpy(cap_emb + s * dim, tf->cap_pad_token, dim * sizeof(float));

    /* 4. Build position IDs */

    /* Image position IDs: (T=cap_padded+1, H=h_idx, W=w_idx)
     * All image tokens share the same T position (one frame). */
    int *img_pos = (int *)calloc(img_padded * 3, sizeof(int));
    if (!img_pos) {
        free(img_emb);
        free(cap_emb);
        return NULL;
    }
    for (int h = 0; h < H_tokens; h++) {
        for (int w = 0; w < W_tokens; w++) {
            int idx = h * W_tokens + w;
            img_pos[idx * 3 + 0] = cap_padded + 1;  /* T (same for all) */
            img_pos[idx * 3 + 1] = h;                /* H */
            img_pos[idx * 3 + 2] = w;                /* W */
        }
    }
    /* Padding tokens get (0, 0, 0) */

    /* Caption position IDs: (T=1+seq_idx, H=0, W=0) */
    int *cap_pos = (int *)calloc(cap_padded * 3, sizeof(int));
    if (!cap_pos) {
        free(img_emb);
        free(cap_emb);
        free(img_pos);
        return NULL;
    }
    for (int s = 0; s < cap_padded; s++) {
        cap_pos[s * 3 + 0] = 1 + s;  /* T */
        cap_pos[s * 3 + 1] = 0;       /* H */
        cap_pos[s * 3 + 2] = 0;       /* W */
    }

    /* Image and caption masks */
    int *img_mask = (int *)malloc(img_padded * sizeof(int));
    if (!img_mask) {
        free(img_emb);
        free(cap_emb);
        free(img_pos);
        free(cap_pos);
        return NULL;
    }
    for (int i = 0; i < img_seq; i++) img_mask[i] = 1;
    for (int i = img_seq; i < img_padded; i++) img_mask[i] = 0;

    int *cap_mask = (int *)malloc(cap_padded * sizeof(int));
    if (!cap_mask) {
        free(img_emb);
        free(cap_emb);
        free(img_pos);
        free(cap_pos);
        free(img_mask);
        return NULL;
    }
    for (int i = 0; i < cap_seq_len; i++) cap_mask[i] = 1;
    for (int i = cap_seq_len; i < cap_padded; i++) cap_mask[i] = 0;

    /* Let padded tokens participate in attention like the reference model */
    /* 5. Noise refiner: image-only self-attention with modulation */
    for (int i = 0; i < tf->n_refiner; i++) {
        zi_block_forward(img_emb, &tf->noise_refiner[i], img_pos, NULL,
                          t_emb, img_padded, tf);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, refiner_total);
    }

    /* 6. Context refiner: caption-only self-attention without modulation */
    for (int i = 0; i < tf->n_refiner; i++) {
        zi_block_forward(cap_emb, &tf->context_refiner[i], cap_pos, NULL,
                          NULL, cap_padded, tf);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, tf->n_refiner + i, refiner_total);
    }

    /* 7. Build unified sequence: [image_tokens, caption_tokens] */
    float *unified = tf->work_x;
    memcpy(unified, img_emb, img_padded * dim * sizeof(float));
    memcpy(unified + img_padded * dim, cap_emb, cap_padded * dim * sizeof(float));
    free(img_emb);
    free(cap_emb);

    /* Unified position IDs */
    int *unified_pos = (int *)malloc(unified_seq * 3 * sizeof(int));
    if (!unified_pos) {
        free(img_pos);
        free(cap_pos);
        free(img_mask);
        free(cap_mask);
        return NULL;
    }
    memcpy(unified_pos, img_pos, img_padded * 3 * sizeof(int));
    memcpy(unified_pos + img_padded * 3, cap_pos, cap_padded * 3 * sizeof(int));
    free(img_pos);
    free(cap_pos);

    /* Unified mask */
    int *unified_mask = (int *)malloc(unified_seq * sizeof(int));
    if (!unified_mask) {
        free(unified_pos);
        free(img_mask);
        free(cap_mask);
        return NULL;
    }
    memcpy(unified_mask, img_mask, img_padded * sizeof(int));
    memcpy(unified_mask + img_padded, cap_mask, cap_padded * sizeof(int));
    free(img_mask);
    free(cap_mask);

    /* 8. Main transformer layers */
    for (int i = 0; i < tf->n_layers; i++) {
        zi_block_forward(unified, &tf->layers[i], unified_pos, NULL,
                          t_emb, unified_seq, tf);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->n_layers);
    }

    free(unified_pos);
    free(unified_mask);

    /* 9. Final layer: extract image tokens only, then project */
    float *img_out = (float *)malloc(img_seq * dim * sizeof(float));
    if (!img_out) return NULL;
    memcpy(img_out, unified, img_seq * dim * sizeof(float));

    int out_ch = ps * ps * in_ch;  /* 64 */
    float *final_out = (float *)malloc(img_seq * out_ch * sizeof(float));
    if (!final_out) {
        free(img_out);
        return NULL;
    }
    zi_final_forward(final_out, img_out, &tf->final_layer, t_emb, img_seq, tf);
    free(img_out);
    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);

    /* 10. Unpatchify: [n_patches, 64] -> [16, latent_h, latent_w] */
    float *output = (float *)calloc(in_ch * latent_h * latent_w, sizeof(float));
    if (!output) {
        free(final_out);
        return NULL;
    }
    zi_unpatchify(output, final_out, in_ch, latent_h, latent_w, ps);
    free(final_out);

    return output;
}

/* ========================================================================
 * Weight Loading (Safetensors)
 * ======================================================================== */

static float *zi_get_tensor(safetensors_file_t **files, int n_files,
                              const char *name, int mmap_f32_weights) {
    /* Keep space for an alternate name used by native FP8 exports */
    char alias[256];
    const char *lookup = name;

    /* Translate diffusers tensor names to the native names used by FP8 exports */
    if (strncmp(name, "all_x_embedder.", 15) == 0) {
        const char *suffix = strrchr(name, '.');
        if (suffix) {
            snprintf(alias, sizeof(alias), "x_embedder%s", suffix);
            lookup = alias;
        }
    } else if (strncmp(name, "all_final_layer.", 16) == 0) {
        const char *component = strstr(name, ".adaLN_modulation.");
        if (!component) component = strstr(name, ".linear.");
        if (!component) component = strstr(name, ".norm_final.");
        if (component) {
            snprintf(alias, sizeof(alias), "final_layer%s", component);
            lookup = alias;
        }
    } else if (strstr(name, ".attention.norm_q.weight")) {
        snprintf(alias, sizeof(alias), "%s", name);
        char *part = strstr(alias, ".attention.norm_q.weight");
        snprintf(part, sizeof(alias) - (size_t)(part - alias), ".attention.q_norm.weight");
        lookup = alias;
    } else if (strstr(name, ".attention.norm_k.weight")) {
        snprintf(alias, sizeof(alias), "%s", name);
        char *part = strstr(alias, ".attention.norm_k.weight");
        snprintf(part, sizeof(alias) - (size_t)(part - alias), ".attention.k_norm.weight");
        lookup = alias;
    } else if (strstr(name, ".attention.to_out.0.weight")) {
        snprintf(alias, sizeof(alias), "%s", name);
        char *part = strstr(alias, ".attention.to_out.0.weight");
        snprintf(part, sizeof(alias) - (size_t)(part - alias), ".attention.out.weight");
        lookup = alias;
    }

    for (int f = 0; f < n_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (!t && lookup != name) t = safetensors_find(files[f], lookup);
        if (!t) continue;
        if (mmap_f32_weights) {
            if (t->dtype != DTYPE_F32) {
                fprintf(stderr, "Error: Z-Image tensor '%s' is not F32 in mmap mode\n", name);
                return NULL;
            }
            return (float *)safetensors_data(files[f], t);
        }
        /* Materialize CPU-used tensors and apply legacy FP8 scaling once */
        float *result = safetensors_get_f32(files[f], t);
        if (result && t->dtype == DTYPE_F8_E4M3) {
            char scale_name[256];
            snprintf(scale_name, sizeof(scale_name), "%s", lookup);
            char *weight = strstr(scale_name, ".weight");
            if (!weight) {
                free(result);
                return NULL;
            }
            snprintf(weight, sizeof(scale_name) - (size_t)(weight - scale_name), ".scale_weight");
            const safetensor_t *scale_tensor = safetensors_find(files[f], scale_name);
            float *scale = scale_tensor ? safetensors_get_f32(files[f], scale_tensor) : NULL;
            if (!scale) {
                free(result);
                return NULL;
            }
            int64_t elements = safetensor_numel(t);
            for (int64_t i = 0; i < elements; i++) result[i] *= scale[0];
            free(scale);
        }
        return result;
    }
    fprintf(stderr, "Warning: Z-Image tensor '%s' not found\n", name);
    return NULL;
}

static int zi_get_fp8_tensor(safetensors_file_t **files, int n_files,
                             const char *name, size_t matrix_offset,
                             zi_fp8_weight_t *result) {
    /* Track the mapped payload and its matching scalar scale tensor */
    const safetensor_t *tensor = NULL;
    safetensors_file_t *file = NULL;
    char scale_name[256];

    /* Locate the raw matrix payload without allocating a dequantized copy */
    for (int i = 0; i < n_files; i++) {
        tensor = safetensors_find(files[i], name);
        if (tensor) {
            file = files[i];
            break;
        }
    }
    if (!tensor || !file || tensor->dtype != DTYPE_F8_E4M3) return 0;

    /* Read the legacy Kijai per-tensor multiplier */
    snprintf(scale_name, sizeof(scale_name), "%s", name);
    char *weight = strstr(scale_name, ".weight");
    if (!weight) return 0;
    snprintf(weight, sizeof(scale_name) - (size_t)(weight - scale_name), ".scale_weight");
    const safetensor_t *scale_tensor = safetensors_find(file, scale_name);
    if (!scale_tensor || scale_tensor->dtype != DTYPE_F32 ||
        safetensor_numel(scale_tensor) != 1)
        return 0;
    float scale;
    memcpy(&scale, safetensors_data(file, scale_tensor), sizeof(scale));

    /* Retain a view of the mapped FP8 payload and its selected matrix offset */
    result->data = (const uint8_t *)safetensors_data(file, tensor);
    result->elements = (size_t)safetensor_numel(tensor);
    result->offset = matrix_offset;
    result->scale = scale;
    return result->offset <= result->elements;
}

static float *zi_get_tensor_optional(safetensors_file_t **files, int n_files,
                                       const char *name, int mmap_f32_weights) {
    for (int f = 0; f < n_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (!t) continue;
        if (mmap_f32_weights) {
            if (t->dtype != DTYPE_F32) return NULL;
            return (float *)safetensors_data(files[f], t);
        }
        return safetensors_get_f32(files[f], t);
    }
    return NULL;
}

static int zi_all_tensors_f32(safetensors_file_t **files, int n_files) {
    for (int f = 0; f < n_files; f++) {
        safetensors_file_t *sf = files[f];
        if (!sf) return 0;
        for (int i = 0; i < sf->num_tensors; i++) {
            if (sf->tensors[i].dtype != DTYPE_F32) return 0;
        }
    }
    return 1;
}

#ifdef IRIS_ZIMAGE_GPU
/* Reload GPU-only large matrices as F32 so the CPU fallback has valid weights */
static int zi_prepare_cpu_fallback(zi_transformer_t *tf) {
    zi_block_t *groups[3] = {tf->noise_refiner, tf->context_refiner, tf->layers};
    int counts[3] = {tf->n_refiner, tf->n_refiner, tf->n_layers};
    const char *prefixes[3] = {"noise_refiner", "context_refiner", "layers"};
    char name[256];

    /* Require retained safetensors mappings before converting fallback weights */
    if (!tf->sf_files[0] || tf->num_sf_files <= 0) return 0;

    /* Require F32 shards because the CPU fallback maps the original matrices */
    if (!zi_all_tensors_f32(tf->sf_files, tf->num_sf_files)) return 0;

    /* Reuse mapped F32 matrix data without allocating a second transformer copy */
    for (int g = 0; g < 3; g++) {
        for (int i = 0; i < counts[g]; i++) {
            zi_block_t *block = &groups[g][i];
            snprintf(name, sizeof(name), "%s.%d.attention.to_q.weight", prefixes[g], i);
            block->attn_q_weight = zi_get_tensor(tf->sf_files, tf->num_sf_files, name, 1);
            if (!block->attn_q_weight) goto fallback_error;
            block->f32_mapped_mask |= ZI_BF16_ATTN_Q;
            snprintf(name, sizeof(name), "%s.%d.attention.to_k.weight", prefixes[g], i);
            block->attn_k_weight = zi_get_tensor(tf->sf_files, tf->num_sf_files, name, 1);
            if (!block->attn_k_weight) goto fallback_error;
            block->f32_mapped_mask |= ZI_BF16_ATTN_K;
            snprintf(name, sizeof(name), "%s.%d.attention.to_v.weight", prefixes[g], i);
            block->attn_v_weight = zi_get_tensor(tf->sf_files, tf->num_sf_files, name, 1);
            if (!block->attn_v_weight) goto fallback_error;
            block->f32_mapped_mask |= ZI_BF16_ATTN_V;
            snprintf(name, sizeof(name), "%s.%d.attention.to_out.0.weight", prefixes[g], i);
            block->attn_out_weight = zi_get_tensor(tf->sf_files, tf->num_sf_files, name, 1);
            if (!block->attn_out_weight) goto fallback_error;
            block->f32_mapped_mask |= ZI_BF16_ATTN_OUT;

            /* Load the three mapped FFN matrices needed by the CPU block path */
            snprintf(name, sizeof(name), "%s.%d.feed_forward.w1.weight", prefixes[g], i);
            block->ffn_w1 = zi_get_tensor(tf->sf_files, tf->num_sf_files, name, 1);
            if (!block->ffn_w1) goto fallback_error;
            block->f32_mapped_mask |= ZI_BF16_FFN_W1;
            snprintf(name, sizeof(name), "%s.%d.feed_forward.w2.weight", prefixes[g], i);
            block->ffn_w2 = zi_get_tensor(tf->sf_files, tf->num_sf_files, name, 1);
            if (!block->ffn_w2) goto fallback_error;
            block->f32_mapped_mask |= ZI_BF16_FFN_W2;
            snprintf(name, sizeof(name), "%s.%d.feed_forward.w3.weight", prefixes[g], i);
            block->ffn_w3 = zi_get_tensor(tf->sf_files, tf->num_sf_files, name, 1);
            if (!block->ffn_w3) goto fallback_error;
            block->f32_mapped_mask |= ZI_BF16_FFN_W3;
        }
    }

    return 1;

fallback_error:
    /* Release partial F32 fallback weights before reporting failure */
    for (int g = 0; g < 3; g++) {
        for (int i = 0; i < counts[g]; i++) {
            zi_block_t *block = &groups[g][i];
            /* Release only heap-backed fields from a partial fallback */
            if (!(block->f32_mapped_mask & ZI_BF16_ATTN_Q)) free(block->attn_q_weight);
            if (!(block->f32_mapped_mask & ZI_BF16_ATTN_K)) free(block->attn_k_weight);
            if (!(block->f32_mapped_mask & ZI_BF16_ATTN_V)) free(block->attn_v_weight);
            if (!(block->f32_mapped_mask & ZI_BF16_ATTN_OUT)) free(block->attn_out_weight);
            if (!(block->f32_mapped_mask & ZI_BF16_FFN_W1)) free(block->ffn_w1);
            if (!(block->f32_mapped_mask & ZI_BF16_FFN_W2)) free(block->ffn_w2);
            if (!(block->f32_mapped_mask & ZI_BF16_FFN_W3)) free(block->ffn_w3);
            block->attn_q_weight = NULL;
            block->attn_k_weight = NULL;
            block->attn_v_weight = NULL;
            block->attn_out_weight = NULL;
            block->ffn_w1 = NULL;
            block->ffn_w2 = NULL;
            block->ffn_w3 = NULL;
            block->f32_mapped_mask = 0;
        }
    }
    return 0;
}
#endif

#ifdef IRIS_ZIMAGE_GPU
#ifdef USE_VULKAN
static int zi_warmup_bf16_key(const void *cache_key, const uint16_t *weights,
                              size_t num_elements);
#endif

/* Load a transformer matrix directly as BF16 without retaining a full F32 copy */
static uint16_t *zi_get_bf16_tensor(safetensors_file_t **files, int n_files,
                                    const char *name, unsigned int *mapped_mask,
                                    unsigned int mapped_bit,
                                    float *f32_workspace,
                                    size_t f32_workspace_elements,
                                    uint16_t *bf16_workspace,
                                    const void *cache_key,
                                    unsigned int *cached_mask) {
    const safetensor_t *tensor = NULL;
    safetensors_file_t *file = NULL;
    int64_t elements;
    float *f32;

    /* Locate the matrix and reuse an existing BF16 payload when available */
    for (int i = 0; i < n_files; i++) {
        tensor = safetensors_find(files[i], name);
        if (tensor) {
            file = files[i];
            break;
        }
    }
    if (!tensor || !file) return NULL;

    /* Reuse native BF16 data directly from the memory-mapped shard */
    if (tensor->dtype == DTYPE_BF16) {
        if (mapped_mask) *mapped_mask |= mapped_bit;
        return safetensors_get_bf16_direct(file, tensor);
    }

    /* Determine the conversion size before selecting reusable storage */
    elements = safetensor_numel(tensor);
    if (elements <= 0) return NULL;

    /* Fill the early-allocated F32 workspace when this matrix fits */
    if (f32_workspace && (size_t)elements <= f32_workspace_elements) {
        if (!safetensors_get_f32_into(file, tensor, f32_workspace,
                                      f32_workspace_elements))
            return NULL;
        f32 = f32_workspace;
    } else {
        /* Fall back to a private F32 allocation for an unexpectedly large matrix */
        f32 = safetensors_get_f32(file, tensor);
    }
    if (!f32) return NULL;

    /* Upload through reusable BF16 workspace when Vulkan owns the converted matrix */
#ifdef USE_VULKAN
    if (bf16_workspace && (size_t)elements <= f32_workspace_elements &&
        cache_key && cached_mask &&
        zi_f32_to_bf16_into(bf16_workspace, f32, (size_t)elements) &&
        zi_warmup_bf16_key(cache_key, bf16_workspace, (size_t)elements)) {
        *cached_mask |= mapped_bit;
        if (f32 != f32_workspace) free(f32);
        return NULL;
    }
#endif

    /* Convert the F32 values into a private BF16 array when reuse is unavailable */
    uint16_t *bf16 = zi_f32_to_bf16(f32, (size_t)elements);
    if (f32 != f32_workspace) free(f32);
    return bf16;
}
#endif /* IRIS_ZIMAGE_GPU */

#ifdef USE_VULKAN
static int zi_warmup_bf16_key(const void *cache_key, const uint16_t *weights,
                              size_t num_elements) {
    /* Use the stable block field as the Vulkan cache key */
    return iris_metal_warmup_bf16_key(cache_key, weights, num_elements);
}
#endif

static int zi_load_block(zi_block_t *block, safetensors_file_t **files,
                          int n_files, const char *prefix, int has_modulation,
                          int dim, int ffn_dim, int use_gpu,
                          int mmap_f32_weights, float *f32_workspace,
                          size_t f32_workspace_elements,
                          uint16_t *bf16_workspace) {
    char name[256];

    /* Bind fused and individual FP8 matrices as zero-copy mapped views */
    size_t matrix_elements = (size_t)dim * dim;
    snprintf(name, sizeof(name), "%s.attention.qkv.weight", prefix);
    if (zi_get_fp8_tensor(files, n_files, name, 0, &block->attn_q_fp8)) {
        block->attn_k_fp8 = block->attn_q_fp8;
        block->attn_v_fp8 = block->attn_q_fp8;
        block->attn_k_fp8.offset = matrix_elements;
        block->attn_v_fp8.offset = matrix_elements * 2;
    }
    snprintf(name, sizeof(name), "%s.attention.out.weight", prefix);
    zi_get_fp8_tensor(files, n_files, name, 0, &block->attn_out_fp8);
    snprintf(name, sizeof(name), "%s.feed_forward.w1.weight", prefix);
    zi_get_fp8_tensor(files, n_files, name, 0, &block->ffn_w1_fp8);
    snprintf(name, sizeof(name), "%s.feed_forward.w2.weight", prefix);
    zi_get_fp8_tensor(files, n_files, name, 0, &block->ffn_w2_fp8);
    snprintf(name, sizeof(name), "%s.feed_forward.w3.weight", prefix);
    zi_get_fp8_tensor(files, n_files, name, 0, &block->ffn_w3_fp8);

#ifndef IRIS_ZIMAGE_GPU
    /* Mark GPU-only conversion storage unused in the generic build */
    (void)f32_workspace;
    (void)f32_workspace_elements;
    (void)bf16_workspace;
#endif

    /* Load large GPU matrices directly in their execution precision */
#ifdef IRIS_ZIMAGE_GPU
    if (use_gpu && !mmap_f32_weights && !block->attn_q_fp8.data) {
        snprintf(name, sizeof(name), "%s.attention.to_q.weight", prefix);
         block->attn_q_weight_bf16 = zi_get_bf16_tensor(files, n_files, name,
                                                         &block->bf16_mapped_mask,
                                                         ZI_BF16_ATTN_Q,
                                                         f32_workspace,
                                                         f32_workspace_elements,
                                                         bf16_workspace,
                                                         &block->attn_q_weight_bf16,
                                                         &block->bf16_cached_mask);
        snprintf(name, sizeof(name), "%s.attention.to_k.weight", prefix);
         block->attn_k_weight_bf16 = zi_get_bf16_tensor(files, n_files, name,
                                                         &block->bf16_mapped_mask,
                                                         ZI_BF16_ATTN_K,
                                                         f32_workspace,
                                                         f32_workspace_elements,
                                                         bf16_workspace,
                                                         &block->attn_k_weight_bf16,
                                                         &block->bf16_cached_mask);
        snprintf(name, sizeof(name), "%s.attention.to_v.weight", prefix);
         block->attn_v_weight_bf16 = zi_get_bf16_tensor(files, n_files, name,
                                                         &block->bf16_mapped_mask,
                                                         ZI_BF16_ATTN_V,
                                                         f32_workspace,
                                                         f32_workspace_elements,
                                                         bf16_workspace,
                                                         &block->attn_v_weight_bf16,
                                                         &block->bf16_cached_mask);
        snprintf(name, sizeof(name), "%s.attention.to_out.0.weight", prefix);
         block->attn_out_weight_bf16 = zi_get_bf16_tensor(files, n_files, name,
                                                         &block->bf16_mapped_mask,
                                                         ZI_BF16_ATTN_OUT,
                                                         f32_workspace,
                                                         f32_workspace_elements,
                                                         bf16_workspace,
                                                         &block->attn_out_weight_bf16,
                                                         &block->bf16_cached_mask);
    } else if (!block->attn_q_fp8.data)
#else
    if (!block->attn_q_fp8.data)
#endif
    {
        snprintf(name, sizeof(name), "%s.attention.to_q.weight", prefix);
        block->attn_q_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
        snprintf(name, sizeof(name), "%s.attention.to_k.weight", prefix);
        block->attn_k_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
        snprintf(name, sizeof(name), "%s.attention.to_v.weight", prefix);
        block->attn_v_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
        snprintf(name, sizeof(name), "%s.attention.to_out.0.weight", prefix);
        block->attn_out_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    }

    /* QK norm */
    snprintf(name, sizeof(name), "%s.attention.norm_q.weight", prefix);
    block->attn_norm_q = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.attention.norm_k.weight", prefix);
    block->attn_norm_k = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* Pre/post attention norms */
    snprintf(name, sizeof(name), "%s.attention_norm1.weight", prefix);
    block->attn_norm1 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.attention_norm2.weight", prefix);
    block->attn_norm2 = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* Load the three SwiGLU matrices directly as BF16 on the GPU path */
#ifdef IRIS_ZIMAGE_GPU
    if (use_gpu && !mmap_f32_weights && !block->ffn_w1_fp8.data) {
        snprintf(name, sizeof(name), "%s.feed_forward.w1.weight", prefix);
         block->ffn_w1_bf16 = zi_get_bf16_tensor(files, n_files, name,
                                                 &block->bf16_mapped_mask,
                                                 ZI_BF16_FFN_W1,
                                                 f32_workspace,
                                                 f32_workspace_elements,
                                                 bf16_workspace,
                                                 &block->ffn_w1_bf16,
                                                 &block->bf16_cached_mask);
        snprintf(name, sizeof(name), "%s.feed_forward.w2.weight", prefix);
         block->ffn_w2_bf16 = zi_get_bf16_tensor(files, n_files, name,
                                                 &block->bf16_mapped_mask,
                                                 ZI_BF16_FFN_W2,
                                                 f32_workspace,
                                                 f32_workspace_elements,
                                                 bf16_workspace,
                                                 &block->ffn_w2_bf16,
                                                 &block->bf16_cached_mask);
        snprintf(name, sizeof(name), "%s.feed_forward.w3.weight", prefix);
         block->ffn_w3_bf16 = zi_get_bf16_tensor(files, n_files, name,
                                                 &block->bf16_mapped_mask,
                                                 ZI_BF16_FFN_W3,
                                                 f32_workspace,
                                                 f32_workspace_elements,
                                                 bf16_workspace,
                                                 &block->ffn_w3_bf16,
                                                 &block->bf16_cached_mask);

    } else if (!block->ffn_w1_fp8.data)
#else
    if (!block->ffn_w1_fp8.data)
#endif
    {
        snprintf(name, sizeof(name), "%s.feed_forward.w1.weight", prefix);
        block->ffn_w1 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
        snprintf(name, sizeof(name), "%s.feed_forward.w2.weight", prefix);
        block->ffn_w2 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
        snprintf(name, sizeof(name), "%s.feed_forward.w3.weight", prefix);
        block->ffn_w3 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    }

    /* FFN norms */
    snprintf(name, sizeof(name), "%s.ffn_norm1.weight", prefix);
    block->ffn_norm1 = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "%s.ffn_norm2.weight", prefix);
    block->ffn_norm2 = zi_get_tensor(files, n_files, name, mmap_f32_weights);

    /* AdaLN modulation (only for modulated blocks) */
    if (has_modulation) {
        snprintf(name, sizeof(name), "%s.adaLN_modulation.0.weight", prefix);
        block->adaln_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
        snprintf(name, sizeof(name), "%s.adaLN_modulation.0.bias", prefix);
        block->adaln_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    } else {
        block->adaln_weight = NULL;
        block->adaln_bias = NULL;
    }

    /* Validate the precision-specific large matrices and shared small weights */
#ifdef IRIS_ZIMAGE_GPU
    if (use_gpu) {
        if ((!block->attn_q_weight && !block->attn_q_weight_bf16 && !block->attn_q_fp8.data && !(block->bf16_cached_mask & ZI_BF16_ATTN_Q)) ||
            (!block->attn_k_weight && !block->attn_k_weight_bf16 && !block->attn_k_fp8.data && !(block->bf16_cached_mask & ZI_BF16_ATTN_K)) ||
            (!block->attn_v_weight && !block->attn_v_weight_bf16 && !block->attn_v_fp8.data && !(block->bf16_cached_mask & ZI_BF16_ATTN_V)) ||
            (!block->attn_out_weight && !block->attn_out_weight_bf16 && !block->attn_out_fp8.data && !(block->bf16_cached_mask & ZI_BF16_ATTN_OUT)) ||
            (!block->ffn_w1 && !block->ffn_w1_bf16 && !block->ffn_w1_fp8.data && !(block->bf16_cached_mask & ZI_BF16_FFN_W1)) ||
            (!block->ffn_w2 && !block->ffn_w2_bf16 && !block->ffn_w2_fp8.data && !(block->bf16_cached_mask & ZI_BF16_FFN_W2)) ||
            (!block->ffn_w3 && !block->ffn_w3_bf16 && !block->ffn_w3_fp8.data && !(block->bf16_cached_mask & ZI_BF16_FFN_W3)))
            return 0;
    } else
#endif
    if ((!block->attn_q_weight && !block->attn_q_fp8.data) ||
        (!block->attn_k_weight && !block->attn_k_fp8.data) ||
        (!block->attn_v_weight && !block->attn_v_fp8.data) ||
        (!block->attn_out_weight && !block->attn_out_fp8.data) ||
        !block->attn_norm_q || !block->attn_norm_k ||
        !block->attn_norm1 || !block->attn_norm2 ||
        (!block->ffn_w1 && !block->ffn_w1_fp8.data) ||
        (!block->ffn_w2 && !block->ffn_w2_fp8.data) ||
        (!block->ffn_w3 && !block->ffn_w3_fp8.data) || block->ffn_norm1 == NULL ||
        !block->ffn_norm2) {
        return 0;
    }
    if (has_modulation && (!block->adaln_weight || !block->adaln_bias)) {
        return 0;
    }

#ifdef USE_VULKAN
    /* Upload this block before loading the next block to bound CPU peak memory */
    if (use_gpu) {
        size_t attn_elems = (size_t)dim * dim;
        size_t ffn_up_elems = (size_t)ffn_dim * dim;
        size_t ffn_down_elems = (size_t)dim * ffn_dim;
        int cached = 1;
        if (block->attn_q_weight_bf16)
            cached &= zi_warmup_bf16_key(&block->attn_q_weight_bf16,
                                         block->attn_q_weight_bf16, attn_elems);
        if (block->attn_k_weight_bf16)
            cached &= zi_warmup_bf16_key(&block->attn_k_weight_bf16,
                                         block->attn_k_weight_bf16, attn_elems);
        if (block->attn_v_weight_bf16)
            cached &= zi_warmup_bf16_key(&block->attn_v_weight_bf16,
                                         block->attn_v_weight_bf16, attn_elems);
        if (block->attn_out_weight_bf16)
            cached &= zi_warmup_bf16_key(&block->attn_out_weight_bf16,
                                         block->attn_out_weight_bf16, attn_elems);
        if (block->ffn_w1_bf16)
            cached &= zi_warmup_bf16_key(&block->ffn_w1_bf16,
                                         block->ffn_w1_bf16, ffn_up_elems);
        if (block->ffn_w2_bf16)
            cached &= zi_warmup_bf16_key(&block->ffn_w2_bf16,
                                         block->ffn_w2_bf16, ffn_down_elems);
        if (block->ffn_w3_bf16)
            cached &= zi_warmup_bf16_key(&block->ffn_w3_bf16,
                                         block->ffn_w3_bf16, ffn_up_elems);
        if (cached) {
            /* Release only converted BF16 buffers; mapped data stays owned by the shard */
            if (!(block->bf16_mapped_mask & ZI_BF16_ATTN_Q)) {
                free(block->attn_q_weight_bf16);
                block->attn_q_weight_bf16 = NULL;
            }
            if (!(block->bf16_mapped_mask & ZI_BF16_ATTN_K)) {
                free(block->attn_k_weight_bf16);
                block->attn_k_weight_bf16 = NULL;
            }
            if (!(block->bf16_mapped_mask & ZI_BF16_ATTN_V)) {
                free(block->attn_v_weight_bf16);
                block->attn_v_weight_bf16 = NULL;
            }
            if (!(block->bf16_mapped_mask & ZI_BF16_ATTN_OUT)) {
                free(block->attn_out_weight_bf16);
                block->attn_out_weight_bf16 = NULL;
            }
            if (!(block->bf16_mapped_mask & ZI_BF16_FFN_W1)) {
                free(block->ffn_w1_bf16);
                block->ffn_w1_bf16 = NULL;
            }
            if (!(block->bf16_mapped_mask & ZI_BF16_FFN_W2)) {
                free(block->ffn_w2_bf16);
                block->ffn_w2_bf16 = NULL;
            }
            if (!(block->bf16_mapped_mask & ZI_BF16_FFN_W3)) {
                free(block->ffn_w3_bf16);
                block->ffn_w3_bf16 = NULL;
            }
            block->bf16_host_released = 1;
        }
    }
#endif

#ifdef IRIS_ZIMAGE_GPU
    /* Keep GPU weights in their direct BF16 representation */
    if (use_gpu) {
        (void)dim;
        (void)ffn_dim;
    }
#else
    (void)use_gpu; (void)dim; (void)ffn_dim; (void)mmap_f32_weights;
#endif
    return 1;
}

static void zi_free_block(zi_block_t *block, int free_f32_weights) {
    if (free_f32_weights) {
#ifdef IRIS_ZIMAGE_GPU
        /* Release ordinary F32 weights while preserving mapped fallback views */
        if (!(block->f32_mapped_mask & ZI_BF16_ATTN_Q)) free(block->attn_q_weight);
        if (!(block->f32_mapped_mask & ZI_BF16_ATTN_K)) free(block->attn_k_weight);
        if (!(block->f32_mapped_mask & ZI_BF16_ATTN_V)) free(block->attn_v_weight);
        if (!(block->f32_mapped_mask & ZI_BF16_ATTN_OUT)) free(block->attn_out_weight);
        free(block->attn_norm_q);
        free(block->attn_norm_k);
        free(block->attn_norm1);
        free(block->attn_norm2);
        if (!(block->f32_mapped_mask & ZI_BF16_FFN_W1)) free(block->ffn_w1);
        if (!(block->f32_mapped_mask & ZI_BF16_FFN_W2)) free(block->ffn_w2);
        if (!(block->f32_mapped_mask & ZI_BF16_FFN_W3)) free(block->ffn_w3);
#else
        /* Release ordinary F32 weights in the generic build */
        free(block->attn_q_weight);
        free(block->attn_k_weight);
        free(block->attn_v_weight);
        free(block->attn_out_weight);
        free(block->ffn_w1);
        free(block->ffn_w2);
        free(block->ffn_w3);
#endif
        free(block->ffn_norm1);
        free(block->ffn_norm2);
        free(block->adaln_weight);
        free(block->adaln_bias);
    }
#ifdef IRIS_ZIMAGE_GPU
    /* Free CPU BF16 matrices only when the GPU cache does not own them */
    if (!block->bf16_host_released) {
        /* Release only converted BF16 buffers; mapped native BF16 data belongs to the shard mapping */
        if (!(block->bf16_mapped_mask & ZI_BF16_ATTN_Q)) free(block->attn_q_weight_bf16);
        if (!(block->bf16_mapped_mask & ZI_BF16_ATTN_K)) free(block->attn_k_weight_bf16);
        if (!(block->bf16_mapped_mask & ZI_BF16_ATTN_V)) free(block->attn_v_weight_bf16);
        if (!(block->bf16_mapped_mask & ZI_BF16_ATTN_OUT)) free(block->attn_out_weight_bf16);
        if (!(block->bf16_mapped_mask & ZI_BF16_FFN_W1)) free(block->ffn_w1_bf16);
        if (!(block->bf16_mapped_mask & ZI_BF16_FFN_W2)) free(block->ffn_w2_bf16);
        if (!(block->bf16_mapped_mask & ZI_BF16_FFN_W3)) free(block->ffn_w3_bf16);
    }
#endif
}

/* Loads Z-Image transformer weights from sharded safetensors files.
 * Auto-discovers shards from index JSON, probes weights to determine FFN dim
 * and timestep MLP size. In CPU mode: uses mmap zero-copy pointers for f32
 * weights. In GPU mode: loads large matrices directly as bf16 to avoid a
 * second full-precision copy. Pre-warms the GPU buffer cache after loading. */
zi_transformer_t *zi_transformer_load_safetensors(const char *model_dir,
                                                     const char *transformer_path,
                                                     int dim, int n_heads,
                                                     int n_layers, int n_refiner,
                                                     int cap_feat_dim, int in_channels,
                                                     int patch_size, float rope_theta,
                                                     const int *axes_dims) {
    zi_transformer_t *tf = calloc(1, sizeof(zi_transformer_t));
    if (!tf) return NULL;

    char name[256];

    /* Set config */
    tf->dim = dim;
    tf->n_heads = n_heads;
    tf->head_dim = dim / n_heads;
    tf->n_layers = n_layers;
    tf->n_refiner = n_refiner;
    tf->ffn_dim = (8 * dim / 3 + 255) / 256 * 256;  /* Round up to 256 */
    tf->in_channels = in_channels;
    tf->patch_size = patch_size;
    tf->adaln_dim = dim < 256 ? dim : 256;
    tf->rope_theta = rope_theta;
    tf->cap_feat_dim = cap_feat_dim;

    for (int i = 0; i < 3; i++) {
        tf->axes_dims[i] = axes_dims[i];
        tf->axes_lens[i] = 1024;  /* Default max positions */
    }

    /* Open safetensors files */
    char path[1024];

    /* Try index file first for sharded models */
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors.index.json", model_dir);
    FILE *idx_f = transformer_path && transformer_path[0] ? NULL : fopen(path, "r");

    safetensors_file_t *files[ZI_MAX_SHARDS] = {0};
    int n_files = 0;

    if (idx_f) {
        /* Sharded: parse index to find shard files */
        fseek(idx_f, 0, SEEK_END);
        long fsize = ftell(idx_f);
        fseek(idx_f, 0, SEEK_SET);
        char *json = (char *)malloc(fsize + 1);
        if (!json) {
            fclose(idx_f);
            goto error;
        }
        fread(json, 1, fsize, idx_f);
        json[fsize] = 0;
        fclose(idx_f);

        /* Find unique shard filenames */
        char seen[32][128];
        int n_seen = 0;
        char *p = json;
        while ((p = strstr(p, ".safetensors")) != NULL) {
            /* Find start of filename */
            char *end = p + strlen(".safetensors");
            char *start = p;
            while (start > json && *(start - 1) != '"') start--;

            int len = (int)(end - start);
            if (len < 128) {
                char fname[128];
                memcpy(fname, start, len);
                fname[len] = 0;

                /* Check if already seen */
                int found = 0;
                for (int i = 0; i < n_seen; i++) {
                    if (strcmp(seen[i], fname) == 0) { found = 1; break; }
                }
                if (!found && n_seen < ZI_MAX_SHARDS) {
                    strcpy(seen[n_seen], fname);
                    n_seen++;
                }
            }
            p = end;
        }
        free(json);

        /* Open each shard */
        for (int i = 0; i < n_seen && n_files < ZI_MAX_SHARDS; i++) {
            snprintf(path, sizeof(path), "%s/transformer/%s", model_dir, seen[i]);
            files[n_files] = safetensors_open(path);
            if (files[n_files]) n_files++;
        }
    } else {
        /* Single file */
        if (transformer_path && transformer_path[0])
            snprintf(path, sizeof(path), "%s", transformer_path);
        else
            snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
        files[0] = safetensors_open(path);
        if (files[0]) n_files = 1;
    }

    if (n_files == 0) {
        fprintf(stderr, "Z-Image: failed to open transformer safetensors\n");
        goto error;
    }

    tf->num_sf_files = n_files;
    for (int i = 0; i < n_files; i++) tf->sf_files[i] = files[i];

    if (iris_verbose)
        fprintf(stderr, "  Loading Z-Image transformer (%d shards)...\n", n_files);

    /* Determine FFN dimension from weights */
    const safetensor_t *w1_probe = NULL;
    for (int f = 0; f < n_files && !w1_probe; f++)
        w1_probe = safetensors_find(files[f], "layers.0.feed_forward.w1.weight");
    if (w1_probe) {
        tf->ffn_dim = (int)w1_probe->shape[0];
        tf->fp8_weights = w1_probe->dtype == DTYPE_F8_E4M3;
    }

    /* Determine t_embedder mid_size from weights */
    tf->t_emb_mid_size = 1024;  /* Default */
    const safetensor_t *t_probe = NULL;
    for (int f = 0; f < n_files && !t_probe; f++)
        t_probe = safetensors_find(files[f], "t_embedder.mlp.0.weight");
    if (t_probe) {
        tf->t_emb_mid_size = (int)t_probe->shape[0];
    }

    /* Check if GPU acceleration is available */
    int use_gpu = 0;
#ifdef IRIS_ZIMAGE_GPU
    if (iris_metal_available() && iris_metal_shaders_available()) {
        use_gpu = 1;
        tf->use_gpu = 1;
        if (iris_verbose)
            fprintf(stderr, "  Z-Image: GPU acceleration enabled (%s weights)\n",
                    tf->fp8_weights ? "scaled fp8" : "bf16");
    }
#endif
    /* BLAS/CPU fast-load mode: keep mmap files open and use direct f32 pointers. */
#ifdef USE_VULKAN
    /* Keep F32 shards mapped so Vulkan can stream one BF16 matrix at a time */
    int mmap_f32_weights = zi_all_tensors_f32(files, n_files);
#else
    int mmap_f32_weights = (!use_gpu && zi_all_tensors_f32(files, n_files));
#endif
    tf->mmap_f32_weights = mmap_f32_weights;
    if (mmap_f32_weights) {
        if (iris_verbose)
            fprintf(stderr, "  Z-Image: CPU mmap mode enabled (zero-copy f32 weights)\n");
    }

    /* Reserve one reusable panel for compact CPU FP8 projections and GPU fallback */
    if (tf->fp8_weights) {
        size_t largest_inner = tf->ffn_dim > dim ? (size_t)tf->ffn_dim : (size_t)dim;
        tf->fp8_panel_elements = ZI_FP8_PANEL_ROWS * largest_inner;
        tf->fp8_panel = (float *)malloc(tf->fp8_panel_elements * sizeof(float));
        if (!tf->fp8_panel) goto error;
        if (iris_verbose)
            fprintf(stderr, "  Z-Image: CPU scaled FP8 panel %.1f MiB (%d rows)\n",
                    (double)(tf->fp8_panel_elements * sizeof(float)) /
                        (1024.0 * 1024.0), ZI_FP8_PANEL_ROWS);
    }

#ifdef IRIS_ZIMAGE_GPU
    /* Reserve reusable conversion buffers before loading transformer blocks */
    size_t load_workspace_elements = use_gpu
        && !mmap_f32_weights && !tf->fp8_weights
        ? (size_t)tf->ffn_dim * dim
        : 0;
    if (load_workspace_elements) {
        tf->load_f32_workspace = (float *)malloc(
            load_workspace_elements * sizeof(float));
        tf->load_bf16_workspace = (uint16_t *)malloc(
            load_workspace_elements * sizeof(uint16_t));
        if (!tf->load_f32_workspace || !tf->load_bf16_workspace) {
            goto error;
        }
    }
#else
    size_t load_workspace_elements = 0;
#endif

    /* Load timestep embedder */
    tf->t_emb_mlp0_weight = zi_get_tensor(files, n_files, "t_embedder.mlp.0.weight", mmap_f32_weights);
    tf->t_emb_mlp0_bias = zi_get_tensor(files, n_files, "t_embedder.mlp.0.bias", mmap_f32_weights);
    tf->t_emb_mlp2_weight = zi_get_tensor(files, n_files, "t_embedder.mlp.2.weight", mmap_f32_weights);
    tf->t_emb_mlp2_bias = zi_get_tensor(files, n_files, "t_embedder.mlp.2.bias", mmap_f32_weights);
    if (!tf->t_emb_mlp0_weight || !tf->t_emb_mlp0_bias ||
        !tf->t_emb_mlp2_weight || !tf->t_emb_mlp2_bias) {
        goto error;
    }

    /* Load caption embedder: RMSNorm + Linear */
    tf->cap_emb_norm = zi_get_tensor(files, n_files, "cap_embedder.0.weight", mmap_f32_weights);
    tf->cap_emb_linear_w = zi_get_tensor(files, n_files, "cap_embedder.1.weight", mmap_f32_weights);
    tf->cap_emb_linear_b = zi_get_tensor(files, n_files, "cap_embedder.1.bias", mmap_f32_weights);
    if (!tf->cap_emb_norm || !tf->cap_emb_linear_w || !tf->cap_emb_linear_b) {
        goto error;
    }

    /* Load image embedder */
    snprintf(name, sizeof(name), "all_x_embedder.%d-1.weight", patch_size);
    tf->x_emb_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_x_embedder.%d-1.bias", patch_size);
    tf->x_emb_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    if (!tf->x_emb_weight || !tf->x_emb_bias) {
        goto error;
    }

    /* Pad tokens */
    tf->x_pad_token = zi_get_tensor(files, n_files, "x_pad_token", mmap_f32_weights);
    tf->cap_pad_token = zi_get_tensor(files, n_files, "cap_pad_token", mmap_f32_weights);
    if (!tf->x_pad_token || !tf->cap_pad_token) {
        goto error;
    }

    /* Load noise refiner blocks */
    tf->noise_refiner = calloc(n_refiner, sizeof(zi_block_t));
    if (!tf->noise_refiner) goto error;
    for (int i = 0; i < n_refiner; i++) {
        snprintf(name, sizeof(name), "noise_refiner.%d", i);
        if (!zi_load_block(&tf->noise_refiner[i], files, n_files, name, 1,
                           dim, tf->ffn_dim, use_gpu, mmap_f32_weights,
                           tf->load_f32_workspace, load_workspace_elements,
                           tf->load_bf16_workspace)) {
            goto error;
        }
    }

    /* Load context refiner blocks (no modulation) */
    tf->context_refiner = calloc(n_refiner, sizeof(zi_block_t));
    if (!tf->context_refiner) goto error;
    for (int i = 0; i < n_refiner; i++) {
        snprintf(name, sizeof(name), "context_refiner.%d", i);
        if (!zi_load_block(&tf->context_refiner[i], files, n_files, name, 0,
                           dim, tf->ffn_dim, use_gpu, mmap_f32_weights,
                           tf->load_f32_workspace, load_workspace_elements,
                           tf->load_bf16_workspace)) {
            goto error;
        }
    }

    /* Load main transformer blocks */
    tf->layers = calloc(n_layers, sizeof(zi_block_t));
    if (!tf->layers) goto error;
    for (int i = 0; i < n_layers; i++) {
        snprintf(name, sizeof(name), "layers.%d", i);
        if (!zi_load_block(&tf->layers[i], files, n_files, name, 1,
                           dim, tf->ffn_dim, use_gpu, mmap_f32_weights,
                           tf->load_f32_workspace, load_workspace_elements,
                           tf->load_bf16_workspace)) {
            goto error;
        }
    }

#ifdef IRIS_ZIMAGE_GPU
    /* Keep shard mappings alive while native BF16 weights are referenced by the GPU path */
    if (use_gpu) {
        zi_block_t *groups[3] = {tf->noise_refiner, tf->context_refiner, tf->layers};
        int counts[3] = {tf->n_refiner, tf->n_refiner, tf->n_layers};
        for (int g = 0; g < 3 && !tf->mmap_bf16_weights; g++) {
            for (int i = 0; i < counts[g]; i++) {
                if (groups[g][i].bf16_mapped_mask) {
                    tf->mmap_bf16_weights = 1;
                    break;
                }
            }
        }
    }
#endif

    /* Load final layer */
    snprintf(name, sizeof(name), "all_final_layer.%d-1.adaLN_modulation.1.weight", patch_size);
    tf->final_layer.adaln_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.adaLN_modulation.1.bias", patch_size);
    tf->final_layer.adaln_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.norm_final.weight", patch_size);
    tf->final_layer.norm_weight = zi_get_tensor_optional(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.linear.weight", patch_size);
    tf->final_layer.linear_weight = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    snprintf(name, sizeof(name), "all_final_layer.%d-1.linear.bias", patch_size);
    tf->final_layer.linear_bias = zi_get_tensor(files, n_files, name, mmap_f32_weights);
    if (!tf->final_layer.adaln_weight || !tf->final_layer.adaln_bias ||
        !tf->final_layer.linear_weight || !tf->final_layer.linear_bias) {
        goto error;
    }

    /* Precompute RoPE tables */
    zi_precompute_rope(tf);

    /* Allocate initial working memory (will be resized as needed) */
    tf->work_alloc = 0;
    tf->work_x = NULL;
    tf->work_tmp = NULL;
    tf->work_qkv = NULL;
    tf->work_attn = NULL;
    tf->work_ffn = NULL;
    tf->max_seq = 0;

    /* Close safetensors files unless CPU mmap mode is active. */
    if (!mmap_f32_weights && !tf->fp8_weights && !use_gpu && !tf->mmap_bf16_weights) {
        for (int f = 0; f < n_files; f++) {
            if (tf->sf_files[f]) {
                safetensors_close(tf->sf_files[f]);
                tf->sf_files[f] = NULL;
            }
        }
        tf->num_sf_files = 0;
    }

#ifdef IRIS_ZIMAGE_GPU
    /* Release conversion buffers after all block weights have been uploaded */
    free(tf->load_f32_workspace);
    free(tf->load_bf16_workspace);
    tf->load_f32_workspace = NULL;
    tf->load_bf16_workspace = NULL;
#endif

    if (iris_verbose) {
        fprintf(stderr, "  Z-Image transformer loaded: dim=%d, heads=%d, layers=%d+%d+%d, ffn=%d\n",
                dim, n_heads, n_refiner, n_refiner, n_layers, tf->ffn_dim);
    }

#ifdef IRIS_ZIMAGE_GPU
    /* Pre-warm bf16->Metal buffer cache so first denoising step avoids misses. */
    iris_warmup_bf16_zimage(tf);
#endif

    return tf;

error:
    iris_transformer_free_zimage(tf);
    return NULL;
}

void iris_transformer_free_zimage(zi_transformer_t *tf) {
    if (!tf) return;

#ifdef IRIS_ZIMAGE_GPU
    /* Release conversion storage if loading stopped before normal cleanup */
    free(tf->load_f32_workspace);
    free(tf->load_bf16_workspace);
    tf->load_f32_workspace = NULL;
    tf->load_bf16_workspace = NULL;
#endif

    int free_f32_weights = !tf->mmap_f32_weights;

    if (free_f32_weights) {
        free(tf->t_emb_mlp0_weight);
        free(tf->t_emb_mlp0_bias);
        free(tf->t_emb_mlp2_weight);
        free(tf->t_emb_mlp2_bias);
        free(tf->cap_emb_norm);
        free(tf->cap_emb_linear_w);
        free(tf->cap_emb_linear_b);
        free(tf->x_emb_weight);
        free(tf->x_emb_bias);
        free(tf->x_pad_token);
        free(tf->cap_pad_token);
    }

    if (tf->noise_refiner) {
        for (int i = 0; i < tf->n_refiner; i++)
            zi_free_block(&tf->noise_refiner[i], free_f32_weights);
        free(tf->noise_refiner);
    }
    if (tf->context_refiner) {
        for (int i = 0; i < tf->n_refiner; i++)
            zi_free_block(&tf->context_refiner[i], free_f32_weights);
        free(tf->context_refiner);
    }
    if (tf->layers) {
        for (int i = 0; i < tf->n_layers; i++)
            zi_free_block(&tf->layers[i], free_f32_weights);
        free(tf->layers);
    }

    if (free_f32_weights) {
        free(tf->final_layer.adaln_weight);
        free(tf->final_layer.adaln_bias);
        free(tf->final_layer.norm_weight);
        free(tf->final_layer.linear_weight);
        free(tf->final_layer.linear_bias);
    }

    for (int i = 0; i < tf->num_sf_files; i++) {
        if (tf->sf_files[i]) {
            safetensors_close(tf->sf_files[i]);
            tf->sf_files[i] = NULL;
        }
    }
    tf->num_sf_files = 0;

    for (int i = 0; i < 3; i++) {
        free(tf->rope_cos[i]);
        free(tf->rope_sin[i]);
    }

    free(tf->work_x);
    free(tf->work_tmp);
    free(tf->work_qkv);
    free(tf->work_attn);
    free(tf->work_ffn);
    free(tf->fp8_panel);

#ifdef IRIS_ZIMAGE_GPU
    zi_gpu_rope_cache_clear(tf);
#endif

    free(tf);
}
