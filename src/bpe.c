// jt_bpe: llm-jp-tokenizer v2.2 最小トークナイザ (Unigram Viterbi)。
// C11, restrict, errno + goto cleanup。依存はlibcのみ。
//
// 方式: 語彙ピースのバイトトライ木 + 加算スコアViterbi + byteフォールバック。
// v2.2語彙に境界横断ピースが0件のためハード分割なし全域Viterbiで等価
// (詳細は include/jimotono/bpe.h の D1〜D3)。

#include "jimotono/bpe.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <errno.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ---- トライ ----
// nodes[0] が root。child/sibling の 0 は「なし」(実ノードは1から)。
typedef struct jt_bpe_node {
    uint32_t child;
    uint32_t sibling;
    uint8_t byte;
    uint8_t pad[3];
    int32_t id;  // 終端でなければ -1
    float score;
} jt_bpe_node_t;

// ---- 復号表エントリ ----
typedef struct jt_bpe_dec {
    uint32_t off;  // blob先頭からのオフセット
    uint32_t len;
    uint8_t is_byte;
    uint8_t val;  // is_byte時の実バイト値
    uint8_t pad[2];
} jt_bpe_dec_t;

typedef struct jt_bpe_inner {
    unsigned char *base;  // mmapまたはmalloc領域 (語彙ファイル像)
    size_t len;
    int is_mmap;
    unsigned char *blob;  // ピース連結 (base内への参照。所有権はbase)
    jt_bpe_dec_t *dec;    // (max_id+1) 要素。範囲外IDは len=UINT32_MAX 扱いで識別
    uint32_t max_id;
    uint32_t n_pieces;
    int32_t byte_id[256];  // byte値 → ピースID。欠番は -1
    float byte_score[256];
    jt_bpe_node_t *nodes;
    uint32_t n_nodes;
    uint32_t cap_nodes;
    uint32_t max_plen;  // 最長ピース長 (Viterbi歩行上限)
} jt_bpe_inner_t;

static uint32_t jt_bpe_rd32(const unsigned char *p) {
    uint32_t v = 0;
    memcpy(&v, p, 4);
    return v;
}

static float jt_bpe_rdfloat(const unsigned char *p) {
    float v = 0.0f;
    memcpy(&v, p, 4);
    return v;
}

// ---- ファイル読み (mmap優先、失敗時malloc+read。Windowsはfopen) ----
static int jt_bpe_read_file(const char *restrict path,
                            unsigned char **restrict out_base,
                            size_t *restrict out_len, int *restrict out_mmap) {
    if (path == NULL || out_base == NULL || out_len == NULL ||
        out_mmap == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_base = NULL;
    *out_len = 0;
    *out_mmap = 0;
#if defined(_WIN32)
    {
        FILE *f = fopen(path, "rb");
        long n = 0;
        unsigned char *buf = NULL;
        size_t done = 0;
        int rc = JT_OK;
        if (f == NULL) {
            return JT_ERR_IO;
        }
        if (fseek(f, 0, SEEK_END) != 0) {
            rc = JT_ERR_IO;
            goto wcleanup;
        }
        n = ftell(f);
        if (n < 0) {
            rc = JT_ERR_IO;
            goto wcleanup;
        }
        if (fseek(f, 0, SEEK_SET) != 0) {
            rc = JT_ERR_IO;
            goto wcleanup;
        }
        buf = (unsigned char *)malloc((size_t)n == 0 ? 1 : (size_t)n);
        if (buf == NULL) {
            errno = ENOMEM;
            rc = JT_ERR_NOMEM;
            goto wcleanup;
        }
        while (done < (size_t)n) {
            size_t r = fread(buf + done, 1, (size_t)n - done, f);
            if (r == 0) {
                if (ferror(f)) {
                    free(buf);
                    rc = JT_ERR_IO;
                    goto wcleanup;
                }
                break;
            }
            done += r;
        }
        if (done != (size_t)n) {
            free(buf);
            errno = EIO;
            rc = JT_ERR_IO;
            goto wcleanup;
        }
        *out_base = buf;
        *out_len = (size_t)n;
        *out_mmap = 0;
        rc = JT_OK;
    wcleanup:
        fclose(f);
        return rc;
    }
#else
    {
        int fd = -1;
        struct stat st;
        size_t len = 0;
        unsigned char *base = NULL;
        int is_mmap = 0;
        int rc = JT_OK;
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            return JT_ERR_IO;
        }
        if (fstat(fd, &st) != 0) {
            rc = JT_ERR_IO;
            goto cleanup;
        }
        if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        len = (size_t)st.st_size;
        if (len == 0) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        {
            void *m = mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
            if (m != MAP_FAILED) {
                base = (unsigned char *)m;
                is_mmap = 1;
            } else {
                unsigned char *buf =
                    (unsigned char *)malloc(len == 0 ? 1 : len);
                size_t done = 0;
                if (buf == NULL) {
                    errno = ENOMEM;
                    rc = JT_ERR_NOMEM;
                    goto cleanup;
                }
                while (done < len) {
                    ssize_t r = read(fd, buf + done, len - done);
                    if (r < 0) {
                        if (errno == EINTR) {
                            continue;
                        }
                        free(buf);
                        rc = JT_ERR_IO;
                        goto cleanup;
                    }
                    if (r == 0) {
                        free(buf);
                        errno = EIO;
                        rc = JT_ERR_IO;
                        goto cleanup;
                    }
                    done += (size_t)r;
                }
                base = buf;
                is_mmap = 0;
            }
        }
        *out_base = base;
        *out_len = len;
        *out_mmap = is_mmap;
        rc = JT_OK;
    cleanup:
        if (fd >= 0) {
            close(fd);
        }
        return rc;
    }
#endif
}

static void jt_bpe_release_base(jt_bpe_inner_t *restrict in) {
    if (in->base == NULL) {
        return;
    }
#if defined(_WIN32)
    free(in->base);
#else
    if (in->is_mmap) {
        munmap(in->base, in->len);
    } else {
        free(in->base);
    }
#endif
    in->base = NULL;
}

// ---- トライ操作 ----
static int jt_bpe_node_new(jt_bpe_inner_t *restrict in, uint32_t *restrict out) {
    if (in->n_nodes == in->cap_nodes) {
        size_t ncap = in->cap_nodes == 0 ? 1024 : (size_t)in->cap_nodes * 2;
        jt_bpe_node_t *np = NULL;
        if (ncap > SIZE_MAX / sizeof(jt_bpe_node_t)) {
            errno = ENOMEM;
            return JT_ERR_NOMEM;
        }
        np = (jt_bpe_node_t *)realloc(in->nodes, ncap * sizeof(*np));
        if (np == NULL) {
            errno = ENOMEM;
            return JT_ERR_NOMEM;
        }
        in->nodes = np;
        in->cap_nodes = (uint32_t)ncap;
    }
    {
        jt_bpe_node_t *nd = &in->nodes[in->n_nodes];
        nd->child = 0;
        nd->sibling = 0;
        nd->byte = 0;
        nd->id = -1;
        nd->score = 0.0f;
        memset(nd->pad, 0, sizeof(nd->pad));
    }
    *out = in->n_nodes;
    in->n_nodes++;
    return JT_OK;
}

// ピース挿入。同一バイト列の重複は INVAL (fail-closed)。
static int jt_bpe_trie_insert(jt_bpe_inner_t *restrict in,
                              const unsigned char *restrict bytes, uint32_t len,
                              int32_t id, float score) {
    uint32_t cur = 0;  // root
    for (uint32_t k = 0; k < len; k++) {
        unsigned char want = bytes[k];
        uint32_t prev = 0;
        uint32_t c = in->nodes[cur].child;
        while (c != 0 && in->nodes[c].byte != want) {
            prev = c;
            c = in->nodes[c].sibling;
        }
        if (c == 0) {
            uint32_t nn = 0;
            int rc = jt_bpe_node_new(in, &nn);
            if (rc != JT_OK) {
                return rc;
            }
            in->nodes[nn].byte = want;
            if (prev == 0) {
                in->nodes[cur].child = nn;
            } else {
                in->nodes[prev].sibling = nn;
            }
            c = nn;
        }
        cur = c;
    }
    if (in->nodes[cur].id >= 0) {
        errno = EINVAL;  // 重複ピース
        return JT_ERR_INVAL;
    }
    in->nodes[cur].id = id;
    in->nodes[cur].score = score;
    if (len > in->max_plen) {
        in->max_plen = len;
    }
    return JT_OK;
}

// ---- 厳格UTF-8検証 (overlong/surrogate/範囲外/切詰めを拒否) ----
static int jt_bpe_valid_utf8(const unsigned char *restrict s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = s[i];
        if (c < 0x80u) {
            i += 1;
        } else if (c >= 0xC2u && c <= 0xDFu) {
            if (i + 1 >= n || (s[i + 1] & 0xC0u) != 0x80u) {
                return 0;
            }
            i += 2;
        } else if (c == 0xE0u) {
            if (i + 2 >= n || s[i + 1] < 0xA0u || s[i + 1] > 0xBFu ||
                (s[i + 2] & 0xC0u) != 0x80u) {
                return 0;
            }
            i += 3;
        } else if (c >= 0xE1u && c <= 0xECu) {
            if (i + 2 >= n || (s[i + 1] & 0xC0u) != 0x80u ||
                (s[i + 2] & 0xC0u) != 0x80u) {
                return 0;
            }
            i += 3;
        } else if (c == 0xEDu) {
            if (i + 2 >= n || s[i + 1] < 0x80u || s[i + 1] > 0x9Fu ||
                (s[i + 2] & 0xC0u) != 0x80u) {
                return 0;
            }
            i += 3;
        } else if (c == 0xEEu || c == 0xEFu) {
            if (i + 2 >= n || (s[i + 1] & 0xC0u) != 0x80u ||
                (s[i + 2] & 0xC0u) != 0x80u) {
                return 0;
            }
            i += 3;
        } else if (c == 0xF0u) {
            if (i + 3 >= n || s[i + 1] < 0x90u || s[i + 1] > 0xBFu ||
                (s[i + 2] & 0xC0u) != 0x80u ||
                (s[i + 3] & 0xC0u) != 0x80u) {
                return 0;
            }
            i += 4;
        } else if (c >= 0xF1u && c <= 0xF3u) {
            if (i + 3 >= n || (s[i + 1] & 0xC0u) != 0x80u ||
                (s[i + 2] & 0xC0u) != 0x80u ||
                (s[i + 3] & 0xC0u) != 0x80u) {
                return 0;
            }
            i += 4;
        } else if (c == 0xF4u) {
            if (i + 3 >= n || s[i + 1] < 0x80u || s[i + 1] > 0x8Fu ||
                (s[i + 2] & 0xC0u) != 0x80u ||
                (s[i + 3] & 0xC0u) != 0x80u) {
                return 0;
            }
            i += 4;
        } else {
            return 0;
        }
    }
    return 1;
}

// U+2581 (▁) のUTF-8。
static const unsigned char JT_BPE_SP[3] = {0xE2u, 0x96u, 0x81u};

int jt_bpe_open(jt_bpe_t *restrict b, const char *restrict vocab_path) {
    jt_bpe_inner_t *in = NULL;
    unsigned char *base = NULL;
    size_t flen = 0;
    int is_mmap = 0;
    int rc = JT_OK;
    uint32_t n = 0, max_id = 0;
    size_t off = 0;
    uint32_t i = 0;

    if (b == NULL || vocab_path == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    b->opaque = NULL;

    rc = jt_bpe_read_file(vocab_path, &base, &flen, &is_mmap);
    if (rc != JT_OK) {
        return rc;
    }
    if (flen < JT_BPE_HEADER_LEN) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (base[0] != JT_BPE_MAGIC0 || base[1] != JT_BPE_MAGIC1 ||
        base[2] != JT_BPE_MAGIC2 || base[3] != JT_BPE_MAGIC3) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (jt_bpe_rd32(base + 4) != JT_BPE_VERSION) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    n = jt_bpe_rd32(base + 8);
    max_id = jt_bpe_rd32(base + 12);
    for (i = 16; i < 32; i++) {
        if (base[i] != 0) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
    }
    if (n == 0 || n > 1000000u || max_id >= JT_BPE_MAX_ID) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }

    in = (jt_bpe_inner_t *)calloc(1, sizeof(*in));
    if (in == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    in->base = base;
    in->len = flen;
    in->is_mmap = is_mmap;
    base = NULL;  // 所有権を in へ
    in->max_id = max_id;
    in->n_pieces = n;
    for (i = 0; i < 256; i++) {
        in->byte_id[i] = -1;
        in->byte_score[i] = 0.0f;
    }
    {
        uint32_t nn = 0;
        // root ノード確保。
        in->nodes = NULL;
        in->n_nodes = 0;
        in->cap_nodes = 0;
        rc = jt_bpe_node_new(in, &nn);
        if (rc != JT_OK || nn != 0) {
            if (rc == JT_OK) {
                errno = EINVAL;
                rc = JT_ERR_INVAL;
            }
            goto cleanup;
        }
    }
    {
        size_t tab = 0;
        if ((size_t)max_id + 1u > SIZE_MAX / sizeof(jt_bpe_dec_t)) {
            errno = ENOMEM;
            rc = JT_ERR_NOMEM;
            goto cleanup;
        }
        tab = ((size_t)max_id + 1u) * sizeof(jt_bpe_dec_t);
        in->dec = (jt_bpe_dec_t *)malloc(tab == 0 ? 1 : tab);
        if (in->dec == NULL) {
            errno = ENOMEM;
            rc = JT_ERR_NOMEM;
            goto cleanup;
        }
        for (i = 0; i <= max_id; i++) {
            in->dec[i].off = 0;
            in->dec[i].len = UINT32_MAX;  // 未登録印
            in->dec[i].is_byte = 0;
            in->dec[i].val = 0;
            memset(in->dec[i].pad, 0, sizeof(in->dec[i].pad));
        }
    }

    off = JT_BPE_HEADER_LEN;
    for (i = 0; i < n; i++) {
        uint32_t id = 0, flags = 0, plen = 0;
        float score = 0.0f;
        const unsigned char *bytes = NULL;
        if (off + 12 > flen) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        id = jt_bpe_rd32(in->base + off);
        score = jt_bpe_rdfloat(in->base + off + 4);
        flags = jt_bpe_rd32(in->base + off + 8);
        off += 12;
        if (off + 4 > flen) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        plen = jt_bpe_rd32(in->base + off);
        off += 4;
        if (plen == 0 || plen > 256u || off + plen > flen) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        bytes = in->base + off;
        if (id > max_id || in->dec[id].len != UINT32_MAX) {
            errno = EINVAL;  // 範囲外または重複ID
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        if ((flags & (uint32_t)(~JT_BPE_FLAG_BYTE)) != 0u) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        in->dec[id].off = (uint32_t)(bytes - in->base);
        in->dec[id].len = plen;
        if ((flags & JT_BPE_FLAG_BYTE) != 0u) {
            unsigned v = 0;
            if (plen != 1) {
                errno = EINVAL;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
            v = bytes[0];
            in->dec[id].is_byte = 1;
            in->dec[id].val = (uint8_t)v;
            if (in->byte_id[v] >= 0) {
                errno = EINVAL;  // byte重複
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
            in->byte_id[v] = (int32_t)id;
            in->byte_score[v] = score;
        } else {
            rc = jt_bpe_trie_insert(in, bytes, plen, (int32_t)id, score);
            if (rc != JT_OK) {
                goto cleanup;
            }
        }
        off += plen;
    }
    if (off != flen) {
        errno = EINVAL;  // 末尾ゴミ
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    in->blob = in->base;  // 復号は base 内参照 (offはbase相対)

    b->opaque = in;
    in = NULL;
    rc = JT_OK;

cleanup:
    if (base != NULL) {
#if defined(_WIN32)
        free(base);
#else
        if (is_mmap) {
            munmap(base, flen);
        } else {
            free(base);
        }
#endif
    }
    if (in != NULL) {
        free(in->dec);
        free(in->nodes);
        if (in->base != NULL) {
            jt_bpe_release_base(in);
        }
        free(in);
    }
    return rc;
}

void jt_bpe_close(jt_bpe_t *restrict b) {
    jt_bpe_inner_t *in = NULL;
    if (b == NULL || b->opaque == NULL) {
        return;
    }
    in = (jt_bpe_inner_t *)b->opaque;
    free(in->dec);
    free(in->nodes);
    jt_bpe_release_base(in);
    free(in);
    b->opaque = NULL;
}

static const jt_bpe_inner_t *jt_bpe_get(const jt_bpe_t *restrict b) {
    if (b == NULL || b->opaque == NULL) {
        errno = EINVAL;
        return NULL;
    }
    return (const jt_bpe_inner_t *)b->opaque;
}

uint32_t jt_bpe_vocab_size(const jt_bpe_t *restrict b) {
    const jt_bpe_inner_t *in = jt_bpe_get(b);
    if (in == NULL) {
        return 0;
    }
    return in->n_pieces;
}

uint32_t jt_bpe_vocab_max(const jt_bpe_t *restrict b) {
    const jt_bpe_inner_t *in = jt_bpe_get(b);
    if (in == NULL) {
        return 0;
    }
    return in->max_id + 1u;
}

// 正規化: ダミー▁ + U+0020→▁。戻り後は *nlen。失敗時NULL。
static unsigned char *jt_bpe_normalize(const unsigned char *restrict s,
                                       size_t n, size_t *restrict nlen) {
    unsigned char *buf = NULL;
    size_t w = 0;
    size_t k = 0;
    if (n > (SIZE_MAX - 3u) / 3u) {
        errno = ENOMEM;
        return NULL;
    }
    buf = (unsigned char *)malloc(n * 3u + 3u == 0 ? 1 : n * 3u + 3u);
    if (buf == NULL) {
        errno = ENOMEM;
        return NULL;
    }
    memcpy(buf, JT_BPE_SP, 3);
    w = 3;
    for (k = 0; k < n; k++) {
        if (s[k] == 0x20u) {
            memcpy(buf + w, JT_BPE_SP, 3);
            w += 3;
        } else {
            buf[w] = s[k];
            w += 1;
        }
    }
    *nlen = w;
    return buf;
}

// -inf 判定用 (MSVCは定数0除算を拒否するため -INFINITY を使用)。
// Unigram Viterbiは対数確率の最大和 (0に近いほど良い。byte片-80が最悪)。
static float jt_bpe_neginf(void) {
    return -INFINITY;
}

int jt_bpe_encode(const jt_bpe_t *restrict b, const char *restrict text,
                  size_t len, uint32_t *restrict out, size_t cap,
                  size_t *restrict out_n) {
    const jt_bpe_inner_t *in = NULL;
    const jt_bpe_node_t *nodes = NULL;
    unsigned char *norm = NULL;
    size_t nlen = 0;
    // dpはfloat32で蓄積する (参照実装のLatticeと同精度にし、丸めによる
    // 最良経路の不一致を避ける。doubleだとタイ付近の和が反転しうる)。
    float *dp = NULL;
    uint32_t *prev = NULL;
    uint32_t *pid = NULL;
    size_t i = 0;
    size_t cnt = 0;
    size_t p = 0;
    int rc = JT_OK;

    in = jt_bpe_get(b);
    if (in == NULL || out_n == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_n = 0;
    if (len > 0 && text == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (len == 0) {
        return JT_OK;  // 空入力は0トークン
    }
    if (out == NULL && cap > 0) {
        // out==NULL かつ cap==0 は「必要数照会」として許容しない
        // (fail-closed: 出力先不明の成功を返さない)。cap==0でもout必須。
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (!jt_bpe_valid_utf8((const unsigned char *)text, len)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    nodes = in->nodes;

    norm = jt_bpe_normalize((const unsigned char *)text, len, &nlen);
    if (norm == NULL) {
        return JT_ERR_NOMEM;  // errno設定済み
    }
    if (nlen + 1 < nlen || (nlen + 1) > SIZE_MAX / sizeof(float)) {
        free(norm);
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    dp = (float *)malloc((nlen + 1) * sizeof(*dp));
    prev = (uint32_t *)malloc((nlen + 1) * sizeof(*prev));
    pid = (uint32_t *)malloc((nlen + 1) * sizeof(*pid));
    if (dp == NULL || prev == NULL || pid == NULL) {
        free(dp);
        free(prev);
        free(pid);
        free(norm);
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    {
        float ninf = jt_bpe_neginf();
        for (i = 0; i <= nlen; i++) {
            dp[i] = ninf;
            prev[i] = UINT32_MAX;
            pid[i] = UINT32_MAX;
        }
    }
    dp[0] = 0.0f;

    for (i = 0; i < nlen; i++) {
        float base = 0.0f;
        uint32_t cur = 0;
        size_t j = 0;
        size_t jmax = 0;
        int32_t bid = 0;
        if (dp[i] <= jt_bpe_neginf()) {
            continue;  // 到達不能 (通常ありえない)
        }
        base = dp[i];
        // トライ一致遷移を先に評価する。byteフォールバックは後に回し、
        // 完全タイ時は通常ピースを優先する (参照実装との一致条件)。
        cur = 0;
        jmax = i + (size_t)in->max_plen;
        if (jmax > nlen) {
            jmax = nlen;
        }
        for (j = i; j < jmax; j++) {
            unsigned char want = norm[j];
            uint32_t c = nodes[cur].child;
            while (c != 0 && nodes[c].byte != want) {
                c = nodes[c].sibling;
            }
            if (c == 0) {
                break;
            }
            cur = c;
            if (nodes[cur].id >= 0) {
                float cand = base + nodes[cur].score;
                if (cand > dp[j + 1]) {
                    dp[j + 1] = cand;
                    prev[j + 1] = (uint32_t)i;
                    pid[j + 1] = (uint32_t)nodes[cur].id;
                }
            }
        }
        // byteフォールバック遷移 (常に候補。スコア最低のため最終手段)。
        bid = in->byte_id[norm[i]];
        if (bid < 0) {
            // 当該バイトの代替ピースなし → 符号化不能 (fail-closed)。
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto ecleanup;
        }
        {
            float cand = base + in->byte_score[norm[i]];
            // タイは先勝ち (D3: strict > のみ更新。トライ優先の順序と合わせ
            // 通常ピースが残る)。
            if (cand > dp[i + 1]) {
                dp[i + 1] = cand;
                prev[i + 1] = (uint32_t)i;
                pid[i + 1] = (uint32_t)bid;
            }
        }
    }

    if (dp[nlen] <= jt_bpe_neginf() || prev[nlen] == UINT32_MAX) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto ecleanup;
    }
    // トークン数勘定。
    p = nlen;
    cnt = 0;
    while (p > 0) {
        uint32_t q = prev[p];
        if (q == UINT32_MAX || q >= p) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto ecleanup;
        }
        cnt++;
        p = q;
    }
    if (cnt > 0 && out == NULL) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto ecleanup;
    }
    *out_n = cnt;
    if (cnt > cap) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto ecleanup;
    }
    // 逆向きに充填。
    p = nlen;
    while (p > 0) {
        cnt--;
        out[cnt] = pid[p];
        p = prev[p];
    }
    rc = JT_OK;

ecleanup:
    free(dp);
    free(prev);
    free(pid);
    free(norm);
    return rc;
}

int jt_bpe_decode(const jt_bpe_t *restrict b, const uint32_t *restrict ids,
                  size_t n, char *restrict out, size_t cap,
                  size_t *restrict out_n) {
    const jt_bpe_inner_t *in = NULL;
    unsigned char *tmp = NULL;
    size_t total = 0;
    size_t w = 0;
    size_t k = 0;
    int rc = JT_OK;

    in = jt_bpe_get(b);
    if (in == NULL || out_n == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_n = 0;
    if (n > 0 && ids == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n == 0) {
        if (out == NULL) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        return JT_OK;
    }
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // 1パス目: 総バイト数。
    for (k = 0; k < n; k++) {
        uint32_t id = ids[k];
        if (id > in->max_id || in->dec[id].len == UINT32_MAX) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        if (in->dec[id].is_byte) {
            if (total + 1 < total) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
            total += 1;
        } else {
            uint32_t L = in->dec[id].len;
            if (total + L < total) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
            total += L;
        }
    }
    tmp = (unsigned char *)malloc(total == 0 ? 1 : total);
    if (tmp == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    // 2パス目: 連結。
    for (k = 0; k < n; k++) {
        uint32_t id = ids[k];
        if (in->dec[id].is_byte) {
            tmp[w] = in->dec[id].val;
            w += 1;
        } else {
            uint32_t L = in->dec[id].len;
            memcpy(tmp + w, in->base + in->dec[id].off, L);
            w += L;
        }
    }
    // 3パス目: ▁→U+0020 へその場変換 (縮小のみ)。
    {
        size_t r = 0;
        size_t w2 = 0;
        while (r < total) {
            if (r + 2 < total && tmp[r] == JT_BPE_SP[0] &&
                tmp[r + 1] == JT_BPE_SP[1] && tmp[r + 2] == JT_BPE_SP[2]) {
                tmp[w2] = 0x20u;
                w2 += 1;
                r += 3;
            } else {
                tmp[w2] = tmp[r];
                w2 += 1;
                r += 1;
            }
        }
        total = w2;
    }
    // ダミー接頭辞の逆操作: 先頭1空白を除去。
    if (total > 0 && tmp[0] == 0x20u) {
        memmove(tmp, tmp + 1, total - 1);
        total -= 1;
    }
    *out_n = total;
    if (total > cap) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto dcleanup;
    }
    if (total > 0) {
        memcpy(out, tmp, total);
    }
    rc = JT_OK;

dcleanup:
    free(tmp);
    return rc;
}
