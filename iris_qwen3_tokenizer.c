/*
 * Qwen3 Tokenizer Implementation
 *
 * BPE (Byte Pair Encoding) tokenizer for Qwen3 text encoder.
 * Loads directly from HuggingFace tokenizer.json format.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include "iris_kernels.h"
#include "iris_platform.h"

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define QWEN3_VOCAB_SIZE 151936
#define QWEN3_MAX_TOKEN_LEN 256
#define QWEN3_MAX_SEQ_LEN 512
#define QWEN3_HASH_SIZE 300007  /* Prime > 2 * vocab_size */

/* Special token IDs */
#define QWEN3_PAD_ID 151643      /* <|endoftext|> */
#define QWEN3_IM_START_ID 151644 /* <|im_start|> */
#define QWEN3_IM_END_ID 151645   /* <|im_end|> */
#define QWEN3_THINK_START_ID 151667 /* <think> */
#define QWEN3_THINK_END_ID 151668   /* </think> */

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    const char *token;
    int id;
} vocab_entry_t;

typedef struct {
    const char *left;
    const char *right;
    int rank;  /* Lower rank = higher priority (merge first) */
} bpe_merge_t;

typedef struct qwen3_tokenizer {
    /* Vocabulary: id -> token string */
    const char **vocab;
    int vocab_size;

    /* Hash table: token string -> id */
    vocab_entry_t *vocab_hash;
    int hash_size;

    /* BPE merges */
    bpe_merge_t *merges;
    int num_merges;

    /* Merge rank lookup: "left right" -> rank */
    int *merge_ranks;  /* Hash table: hash("left right") -> rank, or -1 */

    /* Persistent storage for decoded JSON strings */
    char *string_pool;
    size_t string_pool_size;
    size_t string_pool_used;
} qwen3_tokenizer_t;

/* ========================================================================
 * Byte-Level BPE Encoding Table
 * ======================================================================== */

/*
 * GPT-2/Qwen style byte-to-unicode mapping.
 * Bytes 33-126 and 161-172 and 174-255 map to themselves.
 * Other bytes (0-32, 127-160, 173) map to 256+i for uniqueness.
 */
static int byte_to_unicode[256];
static int unicode_to_byte[512];
static int byte_encoder_initialized = 0;

static void init_byte_encoder(void) {
    if (byte_encoder_initialized) return;

    /* Printable ASCII and extended Latin */
    for (int i = 33; i <= 126; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }
    for (int i = 161; i <= 172; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }
    for (int i = 174; i <= 255; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }

    /* Map remaining bytes to 256+ range */
    int offset = 256;
    for (int i = 0; i < 256; i++) {
        if (byte_to_unicode[i] == 0 && i != 33) {  /* 33 maps to itself */
            byte_to_unicode[i] = offset;
            unicode_to_byte[offset] = i;
            offset++;
        }
    }
    /* Fix: byte 0 should also be mapped */
    byte_to_unicode[0] = 256;
    unicode_to_byte[256] = 0;

    byte_encoder_initialized = 1;
}

/* Encode a byte to its unicode character (UTF-8) */
static int encode_byte_to_utf8(unsigned char b, char *out) {
    init_byte_encoder();
    int cp = byte_to_unicode[b];
    if (cp < 128) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 2048) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    /* Shouldn't reach here for byte-level BPE */
    out[0] = '?';
    return 1;
}


/* ========================================================================
 * Hash Functions
 * ======================================================================== */

static unsigned int hash_string(const char *str) {
    unsigned int hash = 2166136261u;
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 16777619u;
    }
    return hash;
}

/* Hash a merge pair without constructing a temporary joined string */
static unsigned int hash_string_pair(const char *left, const char *right) {
    unsigned int hash = 2166136261u;

    /* Hash the left side of the pair */
    while (*left) {
        hash ^= (unsigned char)*left++;
        hash *= 16777619u;
    }

    /* Hash the separator used by the serialized merge key */
    hash ^= (unsigned char)' ';
    hash *= 16777619u;

    /* Hash the right side of the pair */
    while (*right) {
        hash ^= (unsigned char)*right++;
        hash *= 16777619u;
    }

    return hash;
}

static void vocab_hash_insert(vocab_entry_t *table, int hash_size,
                              const char *token, int id) {
    unsigned int h = hash_string(token) % hash_size;
    int probes = 0;
    while (table[h].token != NULL && probes < hash_size) {
        if (strcmp(table[h].token, token) == 0) {
            return;  /* Already exists */
        }
        h = (h + 1) % hash_size;
        probes++;
    }
    if (probes < hash_size) {
        /* Keep hash entries as non-owning references to the string pool */
        table[h].token = token;
        table[h].id = id;
    }
}

static int vocab_hash_lookup(const vocab_entry_t *table, int hash_size,
                             const char *token) {
    unsigned int h = hash_string(token) % hash_size;
    int probes = 0;
    while (table[h].token != NULL && probes < hash_size) {
        if (strcmp(table[h].token, token) == 0) {
            return table[h].id;
        }
        h = (h + 1) % hash_size;
        probes++;
    }
    return -1;
}

/* ========================================================================
 * JSON Parsing Helpers (minimal, just for tokenizer.json)
 * ======================================================================== */

/* Skip whitespace */
static const char *skip_ws(const char *p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Convert one hexadecimal digit to its numeric value */
static int json_hex_value(char c) {
    /* Decode decimal digits */
    if (c >= '0' && c <= '9') return c - '0';

    /* Decode lowercase hexadecimal digits */
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;

    /* Decode uppercase hexadecimal digits */
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;

    return -1;
}

/* Append a Unicode code point as UTF-8 */
static int append_utf8(char *out, size_t capacity, size_t *length, uint32_t cp) {
    size_t count;

    /* Replace invalid Unicode scalar values */
    if (cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
        cp = 0xfffd;

    /* Determine the encoded byte count */
    if (cp < 0x80) count = 1;
    else if (cp < 0x800) count = 2;
    else if (cp < 0x10000) count = 3;
    else count = 4;

    /* Keep one byte available for the terminating NUL */
    if (*length >= capacity || count > capacity - *length - 1)
        return -1;

    /* Encode the code point */
    if (count == 1) {
        out[(*length)++] = (char)cp;
    } else if (count == 2) {
        out[(*length)++] = (char)(0xc0 | (cp >> 6));
        out[(*length)++] = (char)(0x80 | (cp & 0x3f));
    } else if (count == 3) {
        out[(*length)++] = (char)(0xe0 | (cp >> 12));
        out[(*length)++] = (char)(0x80 | ((cp >> 6) & 0x3f));
        out[(*length)++] = (char)(0x80 | (cp & 0x3f));
    } else {
        out[(*length)++] = (char)(0xf0 | (cp >> 18));
        out[(*length)++] = (char)(0x80 | ((cp >> 12) & 0x3f));
        out[(*length)++] = (char)(0x80 | ((cp >> 6) & 0x3f));
        out[(*length)++] = (char)(0x80 | (cp & 0x3f));
    }

    return 0;
}

/* Parse a JSON string into the tokenizer string pool */
static char *parse_json_string(qwen3_tokenizer_t *tok, const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;

    /* Find the end of the encoded string before reserving pool space */
    const char *start = p;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            p += 2;
        } else {
            p++;
        }
    }
    if (*p != '"') return NULL;

    /* Reserve the remaining pool range and let decoding enforce its bounds */
    if (!tok || !tok->string_pool || tok->string_pool_used >= tok->string_pool_size)
        return NULL;
    char *result = tok->string_pool + tok->string_pool_used;
    size_t capacity = tok->string_pool_size - tok->string_pool_used;
    size_t length = 0;

    /* Decode the string into the reserved pool range */
    p = start;
    while (*p && *p != '"') {
        if (*p != '\\') {
            /* Copy an unescaped UTF-8 byte */
            if (length + 1 >= capacity) return NULL;
            result[length++] = *p++;
            continue;
        }

        /* Reject a trailing escape instead of reading past the string */
        if (!p[1]) return NULL;

        /* Decode the JSON escape sequence */
        switch (p[1]) {
            case 'b':
                if (length + 1 >= capacity) return NULL;
                result[length++] = '\b';
                p += 2;
                break;
            case 'f':
                if (length + 1 >= capacity) return NULL;
                result[length++] = '\f';
                p += 2;
                break;
            case 'n':
                if (length + 1 >= capacity) return NULL;
                result[length++] = '\n';
                p += 2;
                break;
            case 'r':
                if (length + 1 >= capacity) return NULL;
                result[length++] = '\r';
                p += 2;
                break;
            case 't':
                if (length + 1 >= capacity) return NULL;
                result[length++] = '\t';
                p += 2;
                break;
            case '\\':
            case '/':
            case '"':
                if (length + 1 >= capacity) return NULL;
                result[length++] = p[1];
                p += 2;
                break;
            case 'u': {
                /* Parse a four-digit Unicode escape */
                /* Validate the complete escape before reading its digits */
                if (!p[2] || !p[3] || !p[4] || !p[5]) return NULL;
                int cp = 0;
                for (int digit = 0; digit < 4; digit++) {
                    int value = json_hex_value(p[2 + digit]);
                    if (value < 0) return NULL;
                    cp = (cp << 4) | value;
                }
                p += 6;

                /* Combine a valid UTF-16 surrogate pair */
                /* Validate a following low surrogate before reading it */
                if (cp >= 0xd800 && cp <= 0xdbff && p[0] == '\\' && p[1] == 'u' &&
                    p[2] && p[3] && p[4] && p[5]) {
                    int low = 0;
                    for (int digit = 0; digit < 4; digit++) {
                        int value = json_hex_value(p[2 + digit]);
                        if (value < 0) return NULL;
                        low = (low << 4) | value;
                    }
                    if (low >= 0xdc00 && low <= 0xdfff) {
                        cp = 0x10000 + ((cp - 0xd800) << 10) + (low - 0xdc00);
                        p += 6;
                    }
                }

                /* Append the decoded code point safely */
                if (append_utf8(result, capacity, &length, (uint32_t)cp) != 0)
                    return NULL;
                break;
            }
            default:
                /* Preserve unknown escapes as their escaped character */
                if (length + 1 >= capacity) return NULL;
                result[length++] = p[1];
                p += 2;
                break;
        }
    }
    result[length] = '\0';

    if (*p == '"') p++;
    *pp = p;
    tok->string_pool_used += length + 1;
    return result;
}

/* Parse a JSON integer */
static int parse_json_int(const char **pp) {
    const char *p = *pp;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *pp = p;
    return neg ? -val : val;
}

/* Skip a JSON value (string, number, object, array, bool, null) */
static const char *skip_json_value(const char *p) {
    p = skip_ws(p);
    if (*p == '"') {
        p++;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) p += 2;
            else p++;
        }
        if (*p == '"') p++;
    } else if (*p == '{') {
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && p[1]) p += 2;
                    else p++;
                }
            }
            p++;
        }
    } else if (*p == '[') {
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && p[1]) p += 2;
                    else p++;
                }
            }
            p++;
        }
    } else {
        /* number, bool, null */
        while (*p && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') p++;
    }
    return p;
}

/* ========================================================================
 * Tokenizer Loading
 * ======================================================================== */

qwen3_tokenizer_t *qwen3_tokenizer_load(const char *tokenizer_json_path) {
    /* Read file */
    FILE *f = fopen(tokenizer_json_path, "rb");
    if (!f) {
        fprintf(stderr, "qwen3_tokenizer_load: cannot open %s\n", tokenizer_json_path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = malloc(size + 1);
    if (!json) {
        fclose(f);
        return NULL;
    }
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    qwen3_tokenizer_t *tok = calloc(1, sizeof(qwen3_tokenizer_t));
    if (!tok) {
        free(json);
        return NULL;
    }

    /* Allocate one persistent pool for all decoded tokenizer strings */
    tok->string_pool_size = (size_t)size + 1;
    tok->string_pool = malloc(tok->string_pool_size);
    if (!tok->string_pool) {
        free(tok);
        free(json);
        return NULL;
    }

    tok->hash_size = QWEN3_HASH_SIZE;
    tok->vocab_hash = calloc(tok->hash_size, sizeof(vocab_entry_t));
    if (!tok->vocab_hash) {
        free(tok->string_pool);
        free(tok);
        free(json);
        return NULL;
    }

    /* Parse vocabulary from "model": { "vocab": { ... } } */
    const char *p = strstr(json, "\"model\"");
    if (!p) {
        fprintf(stderr, "qwen3_tokenizer_load: no model section\n");
        goto error;
    }

    p = strstr(p, "\"vocab\"");
    if (!p) {
        fprintf(stderr, "qwen3_tokenizer_load: no vocab section\n");
        goto error;
    }

    /* Skip to opening brace */
    p = strchr(p, '{');
    if (!p) goto error;
    p++;

    /* Count vocab entries first */
    int vocab_count = 0;
    const char *count_p = p;
    int depth = 1;
    while (*count_p && depth > 0) {
        if (*count_p == '{') depth++;
        else if (*count_p == '}') depth--;
        else if (*count_p == '"' && depth == 1) {
            vocab_count++;
            /* Skip the string */
            count_p++;
            while (*count_p && *count_p != '"') {
                if (*count_p == '\\' && count_p[1]) count_p += 2;
                else count_p++;
            }
        }
        count_p++;
    }
    /* vocab_count is the number of string keys (values are integers, not strings) */

    tok->vocab_size = vocab_count;
    tok->vocab = calloc(vocab_count + 1000, sizeof(*tok->vocab));  /* Extra for added_tokens */
    if (!tok->vocab) goto error;

    /* Parse vocab entries */
    p = skip_ws(p);
    int max_id = 0;
    while (*p && *p != '}') {
        if (*p == '"') {
            char *token = parse_json_string(tok, &p);
            /* Reject malformed vocabulary strings before advancing the parser */
            if (!token) goto error;
            p = skip_ws(p);
            /* Require the vocabulary key-value separator */
            if (*p != ':') goto error;
            p++;
            p = skip_ws(p);
            int id = parse_json_int(&p);

            if (token && id >= 0 && id < vocab_count + 1000) {
                tok->vocab[id] = token;
                vocab_hash_insert(tok->vocab_hash, tok->hash_size, token, id);
                if (id > max_id) max_id = id;
            }

            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
        } else {
            p++;
        }
    }

    /* Parse merges from "model": { "merges": [ ... ] }
     * Merges are arrays like: [["Ġ", "Ġ"], ["ĠĠ", "ĠĠ"], ...]
     */
    p = strstr(json, "\"merges\"");
    if (!p) {
        fprintf(stderr, "qwen3_tokenizer_load: no merges section\n");
        goto error;
    }

    p = strchr(p, '[');
    if (!p) goto error;
    p++;

    /* Count merges by counting '[' characters at depth 1 */
    int merge_count = 0;
    count_p = p;
    depth = 1;
    while (*count_p && depth > 0) {
        if (*count_p == '[') {
            if (depth == 1) merge_count++;
            depth++;
        } else if (*count_p == ']') {
            depth--;
        } else if (*count_p == '"') {
            count_p++;
            while (*count_p && *count_p != '"') {
                if (*count_p == '\\' && count_p[1]) count_p += 2;
                else count_p++;
            }
        }
        if (*count_p) count_p++;
    }

    tok->num_merges = merge_count;
    tok->merges = calloc(merge_count, sizeof(bpe_merge_t));
    tok->merge_ranks = calloc(tok->hash_size, sizeof(int));
    if (!tok->merges || !tok->merge_ranks) goto error;

    /* Initialize merge_ranks to -1 */
    for (int i = 0; i < tok->hash_size; i++) {
        tok->merge_ranks[i] = -1;
    }

    /* Parse merges - format is [["left", "right"], ...] */
    p = skip_ws(p);
    int merge_idx = 0;
    while (*p && *p != ']' && merge_idx < merge_count) {
        if (*p == '[') {
            p++;
            p = skip_ws(p);

            /* Parse left string */
            char *left = NULL;
            char *right = NULL;
            /* Require and decode the complete left merge token */
            if (*p != '"') goto error;
            left = parse_json_string(tok, &p);
            if (!left) goto error;
            p = skip_ws(p);
            /* Require the separator between merge tokens */
            if (*p != ',') goto error;
            p++;
            p = skip_ws(p);

            /* Parse right string */
            /* Require and decode the complete right merge token */
            if (*p != '"') goto error;
            right = parse_json_string(tok, &p);
            if (!right) goto error;

            /* Skip to closing ] */
            while (*p && *p != ']') p++;
            if (*p == ']') p++;

            if (left && right) {
                tok->merges[merge_idx].left = left;
                tok->merges[merge_idx].right = right;
                tok->merges[merge_idx].rank = merge_idx;

                /* Add to merge rank lookup: hash "left right" */
                /* Hash the pair directly without allocating a joined key */
                unsigned int h = hash_string_pair(left, right) % tok->hash_size;
                int probes = 0;
                while (tok->merge_ranks[h] != -1 && probes < tok->hash_size) {
                    h = (h + 1) % tok->hash_size;
                    probes++;
                }
                if (probes < tok->hash_size) {
                    tok->merge_ranks[h] = merge_idx;
                }
            }

            merge_idx++;

            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
        } else {
            p++;
        }
    }

    /* Parse added_tokens for special tokens */
    p = strstr(json, "\"added_tokens\"");
    if (p) {
        p = strchr(p, '[');
        if (p) {
            p++;
            while (*p && *p != ']') {
                if (*p == '{') {
                    /* Parse added token object */
                    p++;
                    char *content = NULL;
                    int id = -1;

                    while (*p && *p != '}') {
                        p = skip_ws(p);
                        if (*p == '"') {
                            char *key = parse_json_string(tok, &p);
                            /* Reject malformed added-token object keys */
                            if (!key) goto error;
                            p = skip_ws(p);
                            /* Require the added-token key-value separator */
                            if (*p != ':') goto error;
                            p++;
                            p = skip_ws(p);

                            if (key && strcmp(key, "content") == 0 && *p == '"') {
                                content = parse_json_string(tok, &p);
                            } else if (key && strcmp(key, "id") == 0) {
                                id = parse_json_int(&p);
                            } else {
                                p = skip_json_value(p);
                            }
                        }
                        p = skip_ws(p);
                        if (*p == ',') p++;
                    }

                    if (content && id >= 0) {
                        if (id < vocab_count + 1000) {
                            if (!tok->vocab[id]) {
                                tok->vocab[id] = content;
                                vocab_hash_insert(tok->vocab_hash, tok->hash_size, content, id);
                                if (id > max_id) max_id = id;
                                content = NULL;  /* Don't free */
                            }
                        }
                    }
                    if (*p == '}') p++;
                }
                p = skip_ws(p);
                if (*p == ',') p++;
            }
        }
    }

    tok->vocab_size = max_id + 1;

    free(json);

    if (iris_verbose)
        fprintf(stderr, " Qwen3 tokenizer loaded (%d vocab)\n",
                tok->vocab_size);

    return tok;

error:
    free(json);
    if (tok) {
        free(tok->vocab_hash);
        free(tok->vocab);
        free(tok->merges);
        free(tok->merge_ranks);
        free(tok->string_pool);
        free(tok);
    }
    return NULL;
}

void qwen3_tokenizer_free(qwen3_tokenizer_t *tok) {
    if (!tok) return;

    if (tok->vocab) {
        free(tok->vocab);
    }

    if (tok->vocab_hash) {
        free(tok->vocab_hash);
    }

    if (tok->merges) {
        free(tok->merges);
    }

    free(tok->merge_ranks);
    free(tok->string_pool);
    free(tok);
}

/* ========================================================================
 * Merge Rank Lookup
 * ======================================================================== */

static int get_merge_rank(qwen3_tokenizer_t *tok, const char *left, const char *right) {
    /* Lookup the pair without constructing a temporary key */
    unsigned int h = hash_string_pair(left, right) % tok->hash_size;
    int probes = 0;
    while (tok->merge_ranks[h] != -1 && probes < tok->hash_size) {
        int rank = tok->merge_ranks[h];
        if (rank >= 0 && rank < tok->num_merges) {
            /* Check if this is the right merge */
            if (strcmp(tok->merges[rank].left, left) == 0 &&
                strcmp(tok->merges[rank].right, right) == 0) {
                return rank;
            }
        }
        h = (h + 1) % tok->hash_size;
        probes++;
    }

    return -1;
}

/* ========================================================================
 * BPE Tokenization
 * ======================================================================== */

/* Token list node for BPE */
typedef struct token_node {
    struct token_node *next;
    size_t length;
    char text[];
} token_node_t;

static token_node_t *create_node(const char *text) {
    size_t length = strlen(text);
    /* Reject a size calculation that would wrap */
    if (length > SIZE_MAX - sizeof(token_node_t) - 1)
        return NULL;
    token_node_t *node = malloc(sizeof(token_node_t) + length + 1);
    if (!node) return NULL;

    /* Initialize the combined node and text allocation */
    node->next = NULL;
    node->length = length;
    memcpy(node->text, text, length + 1);
    return node;
}

static void free_token_list(token_node_t *head) {
    while (head) {
        token_node_t *next = head->next;
        free(head);
        head = next;
    }
}

/* Core BPE (Byte-Pair Encoding) algorithm. Starts with character-level
 * tokens, then iteratively merges the adjacent pair with the lowest rank
 * (highest priority) until no more merges apply. The merge ranks come from
 * the tokenizer's trained vocabulary. This produces the subword tokenization
 * that Qwen3 expects as input. */
static token_node_t *bpe_encode_word(qwen3_tokenizer_t *tok, const char *word) {
    int len = strlen(word);
    if (len == 0) return NULL;

    /* Start with character-level tokens */
    token_node_t *head = NULL;
    token_node_t *tail = NULL;

    const char *p = word;
    while (*p) {
        /* Get one UTF-8 character */
        int char_len = 1;
        unsigned char c = (unsigned char)*p;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        char buf[8];
        memcpy(buf, p, char_len);
        buf[char_len] = '\0';

        token_node_t *node = create_node(buf);
        if (!node) {
            /* Release the partial token list when a node allocation fails */
            free_token_list(head);
            return NULL;
        }
        if (!head) head = node;
        else tail->next = node;
        tail = node;

        p += char_len;
    }

    /* Apply BPE merges */
    int changed = 1;
    while (changed) {
        changed = 0;

        /* Find best merge (lowest rank) */
        int best_rank = tok->num_merges + 1;
        token_node_t *best_node = NULL;

        token_node_t **best_link = &head;
        for (token_node_t **link = &head; *link && (*link)->next; link = &(*link)->next) {
            token_node_t *node = *link;
            int rank = get_merge_rank(tok, node->text, node->next->text);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_node = node;
                best_link = link;
            }
        }

        /* Apply best merge */
        if (best_node) {
            /* Merge best_node and best_node->next */
            size_t len1 = best_node->length;
            size_t len2 = best_node->next->length;
            token_node_t *right = best_node->next;
            /* Reject a merged node size that would wrap */
            if (len1 > SIZE_MAX - sizeof(token_node_t) - 1 ||
                len2 > SIZE_MAX - sizeof(token_node_t) - len1 - 1) {
                free_token_list(head);
                return NULL;
            }
            token_node_t *merged = realloc(best_node,
                                           sizeof(token_node_t) + len1 + len2 + 1);
            if (!merged) {
                free_token_list(head);
                return NULL;
            }

            /* Extend the left node with the right node text */
            memcpy(merged->text + len1, right->text, len2 + 1);
            merged->length = len1 + len2;
            merged->next = right->next;
            *best_link = merged;

            /* Release the consumed right node */
            free(right);

            changed = 1;
        }
    }

    return head;
}

/* Convert text to byte-level encoding */
static int text_to_bytes(const char *text, char *result, size_t capacity) {
    init_byte_encoder();
    size_t len = strlen(text);

    /* Verify that the shared output buffer can hold the encoded chunk */
    if (len > (SIZE_MAX - 1) / 2 || len * 2 + 1 > capacity)
        return -1;

    /* Encode each source byte into the shared output buffer */
    size_t j = 0;
    for (size_t i = 0; i < len; i++) {
        j += encode_byte_to_utf8((unsigned char)text[i], result + j);
    }
    result[j] = '\0';

    return 0;
}

/* ========================================================================
 * Pre-tokenization (GPT-style regex split)
 * ======================================================================== */

/*
 * Simplified pre-tokenizer that handles the main cases:
 * - Contractions: 's, 't, 're, 've, 'm, 'll, 'd
 * - Words with optional leading space
 * - Numbers
 * - Punctuation/symbols
 * - Whitespace
 */
static char **pretokenize(const char *text, int *num_chunks,
                          char **chunk_storage_out) {
    int capacity = 64;
    char **chunks = malloc(capacity * sizeof(char *));
    /* Reserve one temporary buffer for all copied chunks */
    size_t text_length = strlen(text);
    size_t storage_capacity = text_length <= (SIZE_MAX - 1) / 2
        ? text_length * 2 + 1 : 0;
    char *storage = storage_capacity ? malloc(storage_capacity) : NULL;
    size_t storage_used = 0;
    int count = 0;

    if (!chunks || !storage) {
        /* Release the pre-tokenization buffers when setup fails */
        free(chunks);
        free(storage);
        return NULL;
    }

    const char *p = text;
    while (*p) {
        const char *start = p;

        /* Check for contractions */
        if (*p == '\'' && p[1]) {
            char lower = tolower(p[1]);
            if (lower == 's' || lower == 't' || lower == 'm' || lower == 'd') {
                p += 2;
            } else if ((lower == 'r' || lower == 'v' || lower == 'l') && p[2] &&
                       (tolower(p[2]) == 'e' || tolower(p[2]) == 'l')) {
                p += 3;
            } else {
                p++;
            }
        }
        /* Letters (possibly with leading space) */
        else if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                 (unsigned char)*p >= 128) {
            while (*p && ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                          (unsigned char)*p >= 128)) {
                if ((unsigned char)*p >= 128) {
                    /* Skip UTF-8 continuation bytes */
                    if (((unsigned char)*p & 0xE0) == 0xC0) p += 2;
                    else if (((unsigned char)*p & 0xF0) == 0xE0) p += 3;
                    else if (((unsigned char)*p & 0xF8) == 0xF0) p += 4;
                    else p++;
                } else {
                    p++;
                }
            }
        }
        /* Numbers */
        else if (*p >= '0' && *p <= '9') {
            while (*p >= '0' && *p <= '9') p++;
        }
        /* Space followed by word - keep space with word */
        else if (*p == ' ' && p[1] && (isalpha(p[1]) || (unsigned char)p[1] >= 128)) {
            p++;  /* Include the space */
            while (*p && ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                          (unsigned char)*p >= 128)) {
                if ((unsigned char)*p >= 128) {
                    if (((unsigned char)*p & 0xE0) == 0xC0) p += 2;
                    else if (((unsigned char)*p & 0xF0) == 0xE0) p += 3;
                    else if (((unsigned char)*p & 0xF8) == 0xF0) p += 4;
                    else p++;
                } else {
                    p++;
                }
            }
        }
        /* Space followed by number */
        else if (*p == ' ' && p[1] >= '0' && p[1] <= '9') {
            p++;
            while (*p >= '0' && *p <= '9') p++;
        }
        /* Whitespace */
        else if (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
            while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++;
        }
        /* Single character/punctuation */
        else {
            p++;
        }

        /* Add chunk */
        if (p > start) {
            size_t len = (size_t)(p - start);

            if (count >= capacity) {
                capacity *= 2;
                char **expanded = realloc(chunks, capacity * sizeof(char *));
                if (!expanded) {
                    /* Release all pre-tokenization buffers on growth failure */
                    free(chunks);
                    free(storage);
                    return NULL;
                }
                chunks = expanded;
            }

            /* Copy chunks into one contiguous temporary string allocation */
            memcpy(storage + storage_used, start, len);
            storage[storage_used + len] = '\0';
            chunks[count++] = storage + storage_used;
            storage_used += len + 1;
        }
    }

    *num_chunks = count;
    *chunk_storage_out = storage;
    return chunks;
}

/* ========================================================================
 * Main Tokenization API
 * ======================================================================== */

/*
 * Tokenize text to token IDs.
 * Returns array of token IDs, caller must free.
 */
int *qwen3_tokenize(qwen3_tokenizer_t *tok, const char *text,
                    int *num_tokens, int max_len) {
    if (max_len <= 0) max_len = QWEN3_MAX_SEQ_LEN;

    /* Pre-tokenize */
    int num_chunks;
    char *chunk_storage = NULL;
    char **chunks = pretokenize(text, &num_chunks, &chunk_storage);
    if (!chunks) {
        /* Return an empty result when pre-tokenization cannot allocate */
        *num_tokens = 0;
        return NULL;
    }

    /* Allocate the bounded token result once */
    int *tokens = malloc((size_t)max_len * sizeof(*tokens));
    int total = 0;
    size_t byte_capacity = strlen(text) <= (SIZE_MAX - 1) / 2
        ? strlen(text) * 2 + 1 : 0;
    char *byte_storage = byte_capacity ? malloc(byte_capacity) : NULL;
    if (!tokens || !byte_storage) {
        /* Release temporary storage when an output buffer fails */
        free(chunks);
        free(chunk_storage);
        free(tokens);
        free(byte_storage);
        *num_tokens = 0;
        return NULL;
    }

    for (int c = 0; c < num_chunks && total < max_len; c++) {
        /* Convert to byte-level encoding */
        if (text_to_bytes(chunks[c], byte_storage, byte_capacity) != 0)
            continue;

        /* Apply BPE */
        token_node_t *bpe_tokens = bpe_encode_word(tok, byte_storage);

        /* Convert to token IDs */
        for (token_node_t *node = bpe_tokens; node && total < max_len; node = node->next) {
            int id = vocab_hash_lookup(tok->vocab_hash, tok->hash_size, node->text);
            if (id >= 0) {
                tokens[total++] = id;
            }
        }

        free_token_list(bpe_tokens);
    }
    free(chunks);
    free(chunk_storage);
    free(byte_storage);

    *num_tokens = total;
    return tokens;
}

/* Apply the Qwen3 chat template and tokenize the result. Template:
 * <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 * For Flux, also appends <think>\n\n</think>\n\n to match the training
 * template that triggers direct generation. For Z-Image the think tags
 * are skipped (controlled by skip_think_tags). */
int *qwen3_tokenize_chat(qwen3_tokenizer_t *tok, const char *prompt,
                         int *num_tokens, int max_len, int skip_think_tags) {
    if (max_len <= 0) max_len = QWEN3_MAX_SEQ_LEN;

    /* Allocate the bounded chat-template result once */
    int *tokens = malloc((size_t)max_len * sizeof(*tokens));
    int total = 0;
    if (!tokens) {
        /* Report chat-template allocation failure to the caller */
        *num_tokens = 0;
        return NULL;
    }

    /* Add special tokens and text */
    /* <|im_start|> */
    tokens[total++] = QWEN3_IM_START_ID;

    /* "user\n" */
    int n;
    int *user_tokens = qwen3_tokenize(tok, "user\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        tokens[total++] = user_tokens[i];
    }
    free(user_tokens);

    /* prompt */
    int *prompt_tokens = qwen3_tokenize(tok, prompt, &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        tokens[total++] = prompt_tokens[i];
    }
    free(prompt_tokens);

    /* <|im_end|>\n */
    if (total < max_len) {
        tokens[total++] = QWEN3_IM_END_ID;
    }
    int *newline_tokens = qwen3_tokenize(tok, "\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        tokens[total++] = newline_tokens[i];
    }
    free(newline_tokens);

    /* <|im_start|> */
    if (total < max_len) {
        tokens[total++] = QWEN3_IM_START_ID;
    }

    /* "assistant\n" */
    int *asst_tokens = qwen3_tokenize(tok, "assistant\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        tokens[total++] = asst_tokens[i];
    }
    free(asst_tokens);

    /* <think>\n\n</think>\n\n — only for Flux, not Z-Image. */
    if (!skip_think_tags) {
        if (total < max_len)
            tokens[total++] = QWEN3_THINK_START_ID;
        int *think_newlines = qwen3_tokenize(tok, "\n\n", &n, max_len - total);
        for (int i = 0; i < n && total < max_len; i++)
            tokens[total++] = think_newlines[i];
        free(think_newlines);

        if (total < max_len)
            tokens[total++] = QWEN3_THINK_END_ID;

        think_newlines = qwen3_tokenize(tok, "\n\n", &n, max_len - total);
        for (int i = 0; i < n && total < max_len; i++)
            tokens[total++] = think_newlines[i];
        free(think_newlines);
    }

    *num_tokens = total;
    return tokens;
}

/*
 * Pad token sequence to max_len with PAD tokens.
 * Returns new array, caller must free original.
 */
int *qwen3_pad_tokens(int *tokens, int num_tokens, int max_len, int *attention_mask) {
    int *padded = malloc(max_len * sizeof(int));
    if (!padded) return NULL;

    for (int i = 0; i < max_len; i++) {
        if (i < num_tokens) {
            padded[i] = tokens[i];
            if (attention_mask) attention_mask[i] = 1;
        } else {
            padded[i] = QWEN3_PAD_ID;
            if (attention_mask) attention_mask[i] = 0;
        }
    }

    return padded;
}

/* ========================================================================
 * Debug / Utility Functions
 * ======================================================================== */

/* Get token string by ID */
const char *qwen3_get_token(qwen3_tokenizer_t *tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size) return NULL;
    return tok->vocab[id];
}

/* Get token ID by string */
int qwen3_get_id(qwen3_tokenizer_t *tok, const char *token) {
    if (!tok || !token) return -1;
    return vocab_hash_lookup(tok->vocab_hash, tok->hash_size, token);
}
