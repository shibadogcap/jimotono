// bench: io_batch smoke test (Phase 1 SSD streaming足場)。
// runs圧縮正しさ + pread_batch正しさ (一時ファイルに書いて読み戻し一致)。
// 成功時 exit 0、失敗時 exit 1 + stderr。
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "jimotono/io_batch.h"

#if defined(_WIN32)
#include <io.h>
#define JT_FD_OF(f) _fileno(f)
#else
#include <unistd.h>
#define JT_FD_OF(f) fileno(f)
#endif

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

static void test_runs_basic(void) {
    // 仕様例: {0,1,2,5,6} → {(0,3),(5,2)} (record_bytes=1, base=0)
    uint32_t ids[] = {0, 1, 2, 5, 6};
    jt_io_run_t out[4] = {{0, 0}};
    size_t out_n = 0;
    int rc = jt_io_runs_compress(ids, 5, 1, 0, out, 4, &out_n);
    CHECK(rc == JT_OK, "basic rc=%d", rc);
    CHECK(out_n == 2, "basic out_n=%zu want 2", out_n);
    if (out_n == 2) {
        CHECK(out[0].offset == 0 && out[0].length == 3,
              "run0=(%llu,%zu) want (0,3)",
              (unsigned long long)out[0].offset, out[0].length);
        CHECK(out[1].offset == 5 && out[1].length == 2,
              "run1=(%llu,%zu) want (5,2)",
              (unsigned long long)out[1].offset, out[1].length);
    }
}

static void test_runs_scaled(void) {
    // record_bytes/base_offset付き: ids={3,4,7}, rec=16, base=64
    // → {(64+48,32),(64+112,16)} = {(112,32),(176,16)}
    uint32_t ids[] = {3, 4, 7};
    jt_io_run_t out[4] = {{0, 0}};
    size_t out_n = 0;
    int rc = jt_io_runs_compress(ids, 3, 16, 64, out, 4, &out_n);
    CHECK(rc == JT_OK, "scaled rc=%d", rc);
    CHECK(out_n == 2, "scaled out_n=%zu want 2", out_n);
    if (out_n == 2) {
        CHECK(out[0].offset == 112 && out[0].length == 32,
              "srun0=(%llu,%zu) want (112,32)",
              (unsigned long long)out[0].offset, out[0].length);
        CHECK(out[1].offset == 176 && out[1].length == 16,
              "srun1=(%llu,%zu) want (176,16)",
              (unsigned long long)out[1].offset, out[1].length);
    }
}

static void test_runs_edge(void) {
    jt_io_run_t out[4] = {{0, 0}};
    size_t out_n = 99;

    // 空入力: *out_n=0, JT_OK (ids/outはNULL可)
    CHECK(jt_io_runs_compress(NULL, 0, 1, 0, NULL, 0, &out_n) == JT_OK,
          "empty should be OK");
    CHECK(out_n == 0, "empty out_n=%zu want 0", out_n);

    // 重複畳み込み: {2,2,3} → {(2,2)}
    {
        uint32_t ids[] = {2, 2, 3};
        out_n = 0;
        int rc = jt_io_runs_compress(ids, 3, 1, 0, out, 4, &out_n);
        CHECK(rc == JT_OK && out_n == 1, "dup rc=%d n=%zu", rc, out_n);
        if (out_n == 1) {
            CHECK(out[0].offset == 2 && out[0].length == 2,
                  "dup run=(%llu,%zu) want (2,2)",
                  (unsigned long long)out[0].offset, out[0].length);
        }
    }

    // 単一ID: {9} → {(9,1)}
    {
        uint32_t ids[] = {9};
        out_n = 0;
        int rc = jt_io_runs_compress(ids, 1, 1, 0, out, 4, &out_n);
        CHECK(rc == JT_OK && out_n == 1, "single rc=%d n=%zu", rc, out_n);
        if (out_n == 1) {
            CHECK(out[0].offset == 9 && out[0].length == 1,
                  "single run=(%llu,%zu)", (unsigned long long)out[0].offset,
                  out[0].length);
        }
    }

    // 未ソートは INVAL
    {
        uint32_t ids[] = {1, 3, 2};
        out_n = 0;
        int rc = jt_io_runs_compress(ids, 3, 1, 0, out, 4, &out_n);
        CHECK(rc == JT_ERR_INVAL, "unsorted rc=%d want INVAL", rc);
    }

    // record_bytes==0 は INVAL
    {
        uint32_t ids[] = {0, 1};
        out_n = 0;
        int rc = jt_io_runs_compress(ids, 2, 0, 0, out, 4, &out_n);
        CHECK(rc == JT_ERR_INVAL, "rec0 rc=%d want INVAL", rc);
    }

    // out_cap不足は NOMEM + 必要数を返す
    {
        uint32_t ids[] = {0, 1, 2, 5, 6};
        out_n = 0;
        int rc = jt_io_runs_compress(ids, 5, 1, 0, out, 1, &out_n);
        CHECK(rc == JT_ERR_NOMEM, "smallcap rc=%d want NOMEM", rc);
        CHECK(out_n == 2, "smallcap need=%zu want 2", out_n);
    }

    // NULL引数
    {
        uint32_t ids[] = {0};
        CHECK(jt_io_runs_compress(NULL, 1, 1, 0, out, 4, &out_n) == JT_ERR_INVAL,
              "NULL ids should fail");
        CHECK(jt_io_runs_compress(ids, 1, 1, 0, NULL, 4, &out_n) == JT_ERR_INVAL,
              "NULL out should fail");
        CHECK(jt_io_runs_compress(ids, 1, 1, 0, out, 4, NULL) == JT_ERR_INVAL,
              "NULL out_n should fail");
    }
}

static void test_pread_batch(void) {
    // 256Bパターン (0..255) を一時ファイルに書き、spec配列で読み戻し一致を確認。
    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed: %s", strerror(errno));
    if (f == NULL) {
        return;
    }
    unsigned char pattern[256];
    for (int i = 0; i < 256; i++) {
        pattern[i] = (unsigned char)i;
    }
    CHECK(fwrite(pattern, 1, sizeof(pattern), f) == sizeof(pattern),
          "fwrite failed");
    CHECK(fflush(f) == 0, "fflush failed");
    int fd = JT_FD_OF(f);
    CHECK(fd >= 0, "fileno failed");

    unsigned char b0[16] = {0};
    unsigned char b1[10] = {0};
    unsigned char b2[32] = {0};
    jt_io_spec_t specs[4];
    specs[0].offset = 0;
    specs[0].length = sizeof(b0);
    specs[0].dst = b0;
    specs[1].offset = 100;
    specs[1].length = sizeof(b1);
    specs[1].dst = b1;
    specs[2].offset = 200;
    specs[2].length = sizeof(b2);
    specs[2].dst = b2;
    specs[3].offset = 0;
    specs[3].length = 0;  // no-op (dst NULL可)
    specs[3].dst = NULL;

    int rc = JT_OK;
    int prc = 0;
    // goto cleanup パターン (AGENTS.MD 7.1)。
    if (fd < 0) {
        prc = 1;
        goto cleanup;
    }
    rc = jt_io_pread_batch(fd, specs, 4);
    CHECK(rc == JT_OK, "pread_batch rc=%d", rc);
    if (rc == JT_OK) {
        CHECK(memcmp(b0, pattern + 0, sizeof(b0)) == 0, "b0 mismatch");
        CHECK(memcmp(b1, pattern + 100, sizeof(b1)) == 0, "b1 mismatch");
        CHECK(memcmp(b2, pattern + 200, sizeof(b2)) == 0, "b2 mismatch");
    }

    // n==0 は何もせず OK (specs NULL可)
    CHECK(jt_io_pread_batch(fd, NULL, 0) == JT_OK, "n==0 should be OK");

    // 引数不正
    CHECK(jt_io_pread_batch(-1, specs, 4) == JT_ERR_INVAL, "bad fd should fail");
    CHECK(jt_io_pread_batch(fd, NULL, 4) == JT_ERR_INVAL, "NULL specs should fail");
    {
        jt_io_spec_t bad = {0, 8, NULL};
        CHECK(jt_io_pread_batch(fd, &bad, 1) == JT_ERR_INVAL,
              "NULL dst should fail");
    }
    // EOF前打ち切りは IO エラー
    {
        unsigned char tmp[8] = {0};
        jt_io_spec_t over = {1000000, sizeof(tmp), tmp};
        CHECK(jt_io_pread_batch(fd, &over, 1) == JT_ERR_IO,
              "over-EOF should be IO error");
    }

cleanup:
    fclose(f);
    if (prc != 0) {
        fails++;
    }
}

int main(void) {
    test_runs_basic();
    test_runs_scaled();
    test_runs_edge();
    test_pread_batch();
    if (fails == 0) {
        printf("io_batch: OK\n");
        return 0;
    }
    fprintf(stderr, "io_batch: %d FAIL(s)\n", fails);
    return 1;
}
