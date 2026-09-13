// bench: data_pack unit test (往復一致 + 破損検出)。
// 成功時 exit 0、失敗時 exit 1 + stderr。
// 対象: src/data_pack.c (HDF5不在のため独自バイナリ+mmap)。
// NOTE: 命名ファイル (build dir直下) を使う。ctestはbuild-data内で走る想定。
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/data_pack.h"

#define TD_OUT "test_data_pack_tmp.jtdp"
#define TD_BAD "test_data_pack_bad.jtdp"

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

static void test_roundtrip(void) {
    // 空・1要素・通常・長文の4seq (語彙上限48587を含む)。
    static const uint32_t s1[] = {0};
    static const uint32_t s2[] = {1, 2, 3, 100, 48587, 0, 42};
    static uint32_t s3[3000];
    for (int i = 0; i < 3000; i++) {
        s3[i] = (uint32_t)(i % 48588);
    }
    const uint32_t *seqs[4] = {NULL, s1, s2, s3};
    const size_t lens[4] = {0, 1, 7, 3000};

    remove(TD_OUT);
    CHECK(jt_dp_write_file(TD_OUT, seqs, lens, 4) == JT_OK, "write rc=%d",
          0);

    jt_dp_reader_t r = {0};
    CHECK(jt_dp_open(TD_OUT, &r) == JT_OK, "open failed");
    if (r.opaque == NULL) {
        return;
    }
    uint64_t n = 99;
    CHECK(jt_dp_nseq(&r, &n) == JT_OK && n == 4, "nseq=%llu want 4",
          (unsigned long long)n);
    for (uint64_t i = 0; i < 4; i++) {
        uint64_t sl = 99;
        CHECK(jt_dp_seq_len(&r, i, &sl) == JT_OK && sl == lens[i],
              "seq%llu len=%llu want %zu", (unsigned long long)i,
              (unsigned long long)sl, lens[i]);
        const uint32_t *ptr = (const uint32_t *)0x1;
        uint64_t plen = 99;
        CHECK(jt_dp_seq_ptr(&r, i, &ptr, &plen) == JT_OK && plen == lens[i],
              "seq%llu ptr len mismatch", (unsigned long long)i);
        if (lens[i] == 0) {
            CHECK(ptr == NULL, "empty seq ptr should be NULL");
        } else {
            CHECK(ptr != NULL && memcmp(ptr, seqs[i], lens[i] * 4) == 0,
                  "seq%llu payload mismatch", (unsigned long long)i);
        }
        // copy経路も検証。
        uint32_t *buf = (uint32_t *)calloc(lens[i] + 1, 4);
        CHECK(buf != NULL, "oom");
        if (buf != NULL) {
            uint64_t got = 99;
            CHECK(jt_dp_seq_copy(&r, i, buf, lens[i], &got) == JT_OK &&
                      got == lens[i],
                  "seq%llu copy failed", (unsigned long long)i);
            if (lens[i] > 0) {
                CHECK(memcmp(buf, seqs[i], lens[i] * 4) == 0,
                      "seq%llu copy mismatch", (unsigned long long)i);
            }
            // cap不足はNOMEM + 必要数。
            if (lens[i] > 0) {
                got = 99;
                CHECK(jt_dp_seq_copy(&r, i, buf, lens[i] - 1, &got) ==
                          JT_ERR_NOMEM,
                      "seq%llu short cap should be NOMEM",
                      (unsigned long long)i);
                CHECK(got == lens[i], "short cap need=%llu",
                      (unsigned long long)got);
            }
            free(buf);
        }
    }
    // 範囲外idxはINVAL。
    {
        uint64_t sl = 0;
        const uint32_t *ptr = NULL;
        CHECK(jt_dp_seq_len(&r, 4, &sl) == JT_ERR_INVAL, "oob len should fail");
        CHECK(jt_dp_seq_ptr(&r, 4, &ptr, &sl) == JT_ERR_INVAL,
              "oob ptr should fail");
    }
    jt_dp_close(&r);
    jt_dp_close(NULL);  // 無害
    remove(TD_OUT);
}

static void test_streaming_writer(void) {
    // writer_open/add/close経路 (pack_jsonlと同一API) の往復。
    jt_dp_writer_t w = {0};
    remove(TD_OUT);
    CHECK(jt_dp_writer_open(&w, TD_OUT) == JT_OK, "writer_open failed");
    {
        uint32_t a[] = {7, 8};
        CHECK(jt_dp_writer_add(&w, a, 2) == JT_OK, "add a failed");
        CHECK(jt_dp_writer_add(&w, NULL, 0) == JT_OK, "add empty failed");
        CHECK(jt_dp_writer_add(&w, NULL, 1) == JT_ERR_INVAL,
              "add NULL+len should fail");
        uint32_t bad[] = {48588};
        CHECK(jt_dp_writer_add(&w, bad, 1) == JT_ERR_INVAL,
              "add vocab-overflow should fail");
    }
    CHECK(jt_dp_writer_close(&w) == JT_OK, "writer_close failed");

    jt_dp_reader_t r = {0};
    CHECK(jt_dp_open(TD_OUT, &r) == JT_OK, "reopen failed");
    if (r.opaque != NULL) {
        uint64_t n = 0;
        CHECK(jt_dp_nseq(&r, &n) == JT_OK && n == 2, "nseq=%llu want 2",
              (unsigned long long)n);
        jt_dp_close(&r);
    }
    remove(TD_OUT);
}

static void test_empty_pack(void) {
    remove(TD_OUT);
    CHECK(jt_dp_write_file(TD_OUT, NULL, NULL, 0) == JT_OK, "empty write");
    jt_dp_reader_t r = {0};
    CHECK(jt_dp_open(TD_OUT, &r) == JT_OK, "empty open");
    if (r.opaque != NULL) {
        uint64_t n = 99;
        CHECK(jt_dp_nseq(&r, &n) == JT_OK && n == 0, "empty nseq=%llu",
              (unsigned long long)n);
        jt_dp_close(&r);
    }
    remove(TD_OUT);
}

// ファイル全体を読んで1バイト反転コピーを作る (破損テスト用)。
static int make_corrupt(const char *src, const char *dst, long off) {
    FILE *f = fopen(src, "rb");
    if (f == NULL) {
        return -1;
    }
    int rc = -1;
    long len = 0;
    unsigned char *buf = NULL;
    FILE *g = NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        goto cleanup;
    }
    len = ftell(f);
    if (len < 0) {
        goto cleanup;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        goto cleanup;
    }
    buf = (unsigned char *)malloc((size_t)len == 0 ? 1 : (size_t)len);
    if (buf == NULL) {
        goto cleanup;
    }
    if (len > 0 && fread(buf, 1, (size_t)len, f) != (size_t)len) {
        goto cleanup;
    }
    if (off < 0 || off >= len) {
        goto cleanup;
    }
    buf[off] ^= 0xFFu;
    g = fopen(dst, "wb");
    if (g == NULL) {
        goto cleanup;
    }
    if (len > 0 && fwrite(buf, 1, (size_t)len, g) != (size_t)len) {
        goto cleanup;
    }
    rc = 0;

cleanup:
    if (f != NULL) {
        fclose(f);
    }
    if (g != NULL) {
        fclose(g);
    }
    free(buf);
    return rc;
}

static void test_corrupt(void) {
    static const uint32_t s0[] = {5, 6, 7, 8, 1000};
    static const uint32_t s1[] = {9};
    const uint32_t *seqs[2] = {s0, s1};
    const size_t lens[2] = {5, 1};
    remove(TD_OUT);
    remove(TD_BAD);
    CHECK(jt_dp_write_file(TD_OUT, seqs, lens, 2) == JT_OK, "write for corrupt");

    // payload先頭 (64+0)・payload末尾・表領域・マジックを各1バイト破損。
    // いずれもopen失敗 (fail-closed) であること。
    const long spots[4] = {0, 64, 64 + 5 * 4 - 1, 64 + 5 * 4 + 8};
    for (int k = 0; k < 4; k++) {
        CHECK(make_corrupt(TD_OUT, TD_BAD, spots[k]) == 0,
              "make_corrupt spot %ld failed", spots[k]);
        jt_dp_reader_t r = {0};
        CHECK(jt_dp_open(TD_BAD, &r) != JT_OK, "corrupt spot %ld opened!",
              spots[k]);
        jt_dp_close(&r);
    }
    // 存在しないファイルはIO。
    {
        jt_dp_reader_t r = {0};
        CHECK(jt_dp_open("test_data_pack_no_such_file.jtdp", &r) == JT_ERR_IO,
              "missing file should be IO");
        jt_dp_close(&r);
    }
    // NULL引数はINVAL。
    {
        jt_dp_reader_t r = {0};
        CHECK(jt_dp_open(NULL, &r) == JT_ERR_INVAL, "NULL path should fail");
        CHECK(jt_dp_open(TD_OUT, NULL) == JT_ERR_INVAL, "NULL out should fail");
        CHECK(jt_dp_write_file(NULL, seqs, lens, 2) == JT_ERR_INVAL,
              "NULL write path should fail");
        jt_dp_close(&r);
    }
    remove(TD_OUT);
    remove(TD_BAD);
}

int main(void) {
    test_roundtrip();
    test_streaming_writer();
    test_empty_pack();
    test_corrupt();
    if (fails == 0) {
        printf("data_pack: OK\n");
        return 0;
    }
    fprintf(stderr, "data_pack: %d FAIL(s)\n", fails);
    return 1;
}
