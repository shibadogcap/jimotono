// bench: bpe unit test (hermetic: 合成小語彙のみ。外部語彙不要)。
// 成功時 exit 0、失敗時 exit 1 + stderr。対象: src/bpe.c。
// NOTE: 命名ファイル (build dir直下) を使う。ctestはbuild-tok内で走る想定。
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/bpe.h"

#define TB_VOCAB "test_bpe_tmp.jtvocab"
#define TB_BAD "test_bpe_bad.jtvocab"

static int fails = 0;

#define CHECK(cond, ...)                                   \
    do {                                                   \
        if (!(cond)) {                                     \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                  \
            fprintf(stderr, "\n");                         \
            fails++;                                       \
        }                                                  \
    } while (0)

// 合成語彙エントリ: (id, score, flags, bytes)
typedef struct {
    uint32_t id;
    float score;
    uint32_t flags;
    const char *bytes;
} tb_entry_t;

// 最小語彙: ▁/a/b/ab + byte片 (byte遷移は全位置で候補のため、使用バイト
// 全ての代替が必要。'q'=0x71 のみ欠番とし、fail-closed経路の検査に使う)。
// スコアは実語彙の magnitude に合わせる (通常片: 数単位、byte片: -80)。
// Viterbiは最小和のため、通常経路がbyte蓄積 (-80/B) に勝つ必要がある。
static const tb_entry_t TB_ENTRIES[] = {
    {0, -4.0f, 0, "ab"},
    {1, -3.0f, 0, "a"},
    {2, -3.0f, 0, "b"},
    {3, -3.0f, 0, "\xE2\x96\x81"},  // ▁
    {4, -80.0f, 1, "x"},
    {5, -80.0f, 1, "\x09"},
    {6, -80.0f, 1, "\xE2"},
    {7, -80.0f, 1, "\x96"},
    {8, -80.0f, 1, "\x81"},
    {9, -80.0f, 1, "a"},
    {10, -80.0f, 1, "b"},
};
#define TB_N (sizeof(TB_ENTRIES) / sizeof(TB_ENTRIES[0]))

static int tb_write_vocab(const char *path, const tb_entry_t *ents,
                          size_t n, uint32_t max_id) {
    FILE *f = fopen(path, "wb");
    size_t k = 0;
    if (f == NULL) {
        return -1;
    }
    {
        unsigned char hdr[32];
        memset(hdr, 0, sizeof(hdr));
        hdr[0] = 'J';
        hdr[1] = 'T';
        hdr[2] = 'V';
        hdr[3] = 'B';
        {
            uint32_t v = 1;
            memcpy(hdr + 4, &v, 4);
        }
        {
            uint32_t vv = (uint32_t)n;
            memcpy(hdr + 8, &vv, 4);
        }
        memcpy(hdr + 12, &max_id, 4);
        if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
            fclose(f);
            return -1;
        }
    }
    for (k = 0; k < n; k++) {
        uint32_t len = (uint32_t)strlen(ents[k].bytes);
        if (fwrite(&ents[k].id, 4, 1, f) != 1 ||
            fwrite(&ents[k].score, 4, 1, f) != 1 ||
            fwrite(&ents[k].flags, 4, 1, f) != 1 ||
            fwrite(&len, 4, 1, f) != 1 ||
            (len > 0 && fwrite(ents[k].bytes, 1, len, f) != len)) {
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    return 0;
}

static void test_open_errors(void) {
    jt_bpe_t b = {0};
    CHECK(jt_bpe_open(NULL, TB_VOCAB) == JT_ERR_INVAL, "open NULL handle");
    CHECK(jt_bpe_open(&b, NULL) == JT_ERR_INVAL, "open NULL path");
    CHECK(jt_bpe_open(&b, "test_bpe_no_such_file.jtvocab") == JT_ERR_IO,
          "open missing should be IO");
    CHECK(jt_bpe_vocab_size(&b) == 0, "size of unopened should be 0");
    CHECK(jt_bpe_vocab_size(NULL) == 0, "size NULL should be 0");
    jt_bpe_close(&b);
    jt_bpe_close(NULL);
    // 破損: マジック改竄
    CHECK(tb_write_vocab(TB_VOCAB, TB_ENTRIES, TB_N, 10) == 0, "write vocab");
    {
        FILE *f = fopen(TB_VOCAB, "r+b");
        CHECK(f != NULL, "reopen for corrupt");
        if (f != NULL) {
            fputc('X', f);
            fclose(f);
        }
    }
    CHECK(jt_bpe_open(&b, TB_VOCAB) == JT_ERR_INVAL, "bad magic should fail");
    jt_bpe_close(&b);
    remove(TB_VOCAB);
}

static void test_viterbi_and_fallback(void) {
    jt_bpe_t b = {0};
    CHECK(tb_write_vocab(TB_VOCAB, TB_ENTRIES, TB_N, 10) == 0, "write vocab");
    CHECK(jt_bpe_open(&b, TB_VOCAB) == JT_OK, "open failed: %s",
          strerror(errno));
    if (b.opaque == NULL) {
        return;
    }
    CHECK(jt_bpe_vocab_size(&b) == TB_N, "size=%u want %u",
          jt_bpe_vocab_size(&b), (unsigned)TB_N);
    CHECK(jt_bpe_vocab_max(&b) == 11, "max=%u want 11", jt_bpe_vocab_max(&b));

    // "ab" → 正規化 "▁ab" → [▁(3), ab(0)]。Viterbiが a+b より ab を選ぶ。
    {
        uint32_t out[8];
        size_t n = 99;
        CHECK(jt_bpe_encode(&b, "ab", 2, out, 8, &n) == JT_OK, "encode ab");
        CHECK(n == 2 && out[0] == 3 && out[1] == 0, "ab -> [3,0], got n=%zu",
              n);
    }
    // "axb" → [3,1,4(x:byte),2]
    {
        uint32_t out[8];
        size_t n = 99;
        CHECK(jt_bpe_encode(&b, "axb", 3, out, 8, &n) == JT_OK, "encode axb");
        CHECK(n == 4 && out[0] == 3 && out[1] == 1 && out[2] == 4 &&
                  out[3] == 2,
              "axb mismatch n=%zu", n);
    }
    // "a\tb": 正規化で\tは維持 → [3,1,5,2]
    {
        uint32_t out[8];
        size_t n = 99;
        CHECK(jt_bpe_encode(&b, "a\tb", 3, out, 8, &n) == JT_OK,
              "encode tab");
        CHECK(n == 4 && out[2] == 5, "tab should use byte piece n=%zu", n);
    }
    // 空入力 → 0トークン。
    {
        uint32_t out[8];
        size_t n = 99;
        CHECK(jt_bpe_encode(&b, "", 0, out, 8, &n) == JT_OK && n == 0,
              "empty should be 0 tokens");
    }
    // 不正UTF-8 (fail-closed)。
    {
        uint32_t out[8];
        size_t n = 99;
        CHECK(jt_bpe_encode(&b, "\xff", 1, out, 8, &n) == JT_ERR_INVAL,
              "0xFF should fail");
        CHECK(jt_bpe_encode(&b, "\xE3\x81", 2, out, 8, &n) == JT_ERR_INVAL,
              "truncated should fail");
        CHECK(jt_bpe_encode(&b, "\xC0\xAF", 2, out, 8, &n) == JT_ERR_INVAL,
              "overlong should fail");
        CHECK(jt_bpe_encode(&b, "\xED\xA0\x80", 3, out, 8, &n) ==
                  JT_ERR_INVAL,
              "surrogate should fail");
    }
    // 代替なしバイト: 'q' (0x71) のbyteピース不在 → INVAL。
    {
        uint32_t out[8];
        size_t n = 99;
        CHECK(jt_bpe_encode(&b, "q", 1, out, 8, &n) == JT_ERR_INVAL,
              "missing byte piece should fail");
    }
    // cap不足 → NOMEM + 必要数。
    {
        uint32_t out[1];
        size_t n = 99;
        CHECK(jt_bpe_encode(&b, "ab", 2, out, 1, &n) == JT_ERR_NOMEM,
              "short cap should be NOMEM");
        CHECK(n == 2, "need=%zu want 2", n);
    }
    // NULL系。
    {
        uint32_t out[8];
        size_t n = 0;
        CHECK(jt_bpe_encode(&b, NULL, 2, out, 8, &n) == JT_ERR_INVAL,
              "NULL text should fail");
        CHECK(jt_bpe_encode(&b, "ab", 2, NULL, 8, &n) == JT_ERR_INVAL,
              "NULL out should fail");
        CHECK(jt_bpe_encode(&b, "ab", 2, out, 8, NULL) == JT_ERR_INVAL,
              "NULL out_n should fail");
        CHECK(jt_bpe_encode(NULL, "ab", 2, out, 8, &n) == JT_ERR_INVAL,
              "NULL handle should fail");
    }
    jt_bpe_close(&b);
    remove(TB_VOCAB);
}

static void test_decode(void) {
    jt_bpe_t b = {0};
    CHECK(tb_write_vocab(TB_VOCAB, TB_ENTRIES, TB_N, 10) == 0, "write vocab");
    CHECK(jt_bpe_open(&b, TB_VOCAB) == JT_OK, "open failed");
    if (b.opaque == NULL) {
        return;
    }
    // [3,0] = "▁ab" → "ab" (ダミー接頭辞除去)。
    {
        const uint32_t ids[2] = {3, 0};
        char out[16];
        size_t n = 99;
        CHECK(jt_bpe_decode(&b, ids, 2, out, sizeof(out), &n) == JT_OK,
              "decode [3,0]");
        CHECK(n == 2 && memcmp(out, "ab", 2) == 0, "decode [3,0]->ab n=%zu",
              n);
    }
    // 往復: encode→decode。
    {
        const char *src = "a b";
        uint32_t ids[8];
        size_t ni = 0;
        char back[16];
        size_t nb = 0;
        CHECK(jt_bpe_encode(&b, src, strlen(src), ids, 8, &ni) == JT_OK,
              "rt encode");
        CHECK(jt_bpe_decode(&b, ids, ni, back, sizeof(back), &nb) == JT_OK,
              "rt decode");
        CHECK(nb == strlen(src) && memcmp(back, src, nb) == 0,
              "roundtrip mismatch nb=%zu", nb);
    }
    // byte復号: [5] → "\x09"。
    {
        const uint32_t ids[1] = {5};
        char out[8];
        size_t n = 99;
        CHECK(jt_bpe_decode(&b, ids, 1, out, sizeof(out), &n) == JT_OK,
              "decode byte");
        CHECK(n == 1 && out[0] == '\x09', "byte decode n=%zu", n);
    }
    // 語彙外ID → INVAL。
    {
        const uint32_t ids[1] = {11};
        char out[8];
        size_t n = 0;
        CHECK(jt_bpe_decode(&b, ids, 1, out, sizeof(out), &n) == JT_ERR_INVAL,
              "oob id should fail");
    }
    // cap不足 → NOMEM + 必要数。
    {
        const uint32_t ids[2] = {3, 0};
        char out[1];
        size_t n = 99;
        CHECK(jt_bpe_decode(&b, ids, 2, out, 1, &n) == JT_ERR_NOMEM,
              "short cap should be NOMEM");
        CHECK(n == 2, "need=%zu want 2", n);
    }
    // NULL系。
    {
        const uint32_t ids[1] = {0};
        char out[8];
        size_t n = 0;
        CHECK(jt_bpe_decode(&b, NULL, 1, out, sizeof(out), &n) ==
                  JT_ERR_INVAL,
              "NULL ids should fail");
        CHECK(jt_bpe_decode(&b, ids, 1, NULL, sizeof(out), &n) ==
                  JT_ERR_INVAL,
              "NULL out should fail");
        CHECK(jt_bpe_decode(&b, ids, 1, out, sizeof(out), NULL) ==
                  JT_ERR_INVAL,
              "NULL out_n should fail");
        CHECK(jt_bpe_decode(NULL, ids, 1, out, sizeof(out), &n) ==
                  JT_ERR_INVAL,
              "NULL handle should fail");
    }
    // 空復号 → 成功0バイト。
    {
        char out[8];
        size_t n = 99;
        uint32_t dummy = 0;
        (void)dummy;
        CHECK(jt_bpe_decode(&b, NULL, 0, out, sizeof(out), &n) == JT_OK &&
                  n == 0,
              "empty decode");
    }
    jt_bpe_close(&b);
    remove(TB_VOCAB);
    remove(TB_BAD);
}

int main(void) {
    test_open_errors();
    test_viterbi_and_fallback();
    test_decode();
    if (fails == 0) {
        printf("bpe: OK\n");
        return 0;
    }
    fprintf(stderr, "bpe: %d FAIL(s)\n", fails);
    return 1;
}
