// bench: io_uring batch-read skeleton test (Phase 2足場)。
// bench/kernel/test_io.c には手を出さず、本ファイルを新設。
//   - fallback (Linux && !HAVE_LIBURING): n==0はJT_OK、n>0はJT_ERR_NOSUP+ENOSYS。
//   - 有効時 (Linux && HAVE_LIBURING): 往復一致 + 引数不正 + EOF + 窓進行(>256)。
//   - 非Linux: Linux-only APIのためSKIPしてexit 0 (ヘッダ宣言なしに対応)。
// 成功時 exit 0、失敗時 exit 1 + stderr。
// C11、errnoベース (AGENTS.MD 7.1)。
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "jimotono/io_batch.h"

#ifdef __linux__
#include <unistd.h>
#define JT_FD_OF(f) fileno(f)

static int fails = 0;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        if (!(cond)) {                                         \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                      \
            fprintf(stderr, "\n");                             \
            fails++;                                           \
        }                                                      \
    } while (0)

static void test_n0_ok(void) {
    // n==0はfd/specs検証なしでJT_OK (fallback/有効時共通)。
    errno = 0;
    CHECK(jt_io_pread_batch_uring(-1, NULL, 0) == JT_OK, "n==0 should be OK");
}

#ifdef HAVE_LIBURING
static void test_roundtrip(void) {
    // 256Bパターン (0..255) を一時ファイルに書き、uring経路で読み戻し一致。
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
    if (fd < 0) {
        prc = 1;
        goto cleanup;
    }
    rc = jt_io_pread_batch_uring(fd, specs, 4);
    CHECK(rc == JT_OK, "uring roundtrip rc=%d errno=%d", rc, errno);
    if (rc == JT_OK) {
        CHECK(memcmp(b0, pattern + 0, sizeof(b0)) == 0, "b0 mismatch");
        CHECK(memcmp(b1, pattern + 100, sizeof(b1)) == 0, "b1 mismatch");
        CHECK(memcmp(b2, pattern + 200, sizeof(b2)) == 0, "b2 mismatch");
    }

    // 引数不正は INVAL。
    CHECK(jt_io_pread_batch_uring(-1, specs, 4) == JT_ERR_INVAL,
          "bad fd should fail");
    CHECK(jt_io_pread_batch_uring(fd, NULL, 4) == JT_ERR_INVAL,
          "NULL specs should fail");
    {
        jt_io_spec_t bad = {0, 8, NULL};
        CHECK(jt_io_pread_batch_uring(fd, &bad, 1) == JT_ERR_INVAL,
              "NULL dst should fail");
    }
    // EOF前打ち切りは IO エラー (同期版と同一)。
    {
        unsigned char tmp[8] = {0};
        jt_io_spec_t over = {1000000, sizeof(tmp), tmp};
        CHECK(jt_io_pread_batch_uring(fd, &over, 1) == JT_ERR_IO,
              "over-EOF should be IO error");
    }

cleanup:
    fclose(f);
    if (prc != 0) {
        fails++;
    }
}

static void test_window(void) {
    // sliding-window進行の確認: batch=256を超える300件の1B読み。
    // buffered fd前提・O_DIRECTなし (4096整列要求なし)。
    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed: %s", strerror(errno));
    if (f == NULL) {
        return;
    }
    unsigned char pattern[256];
    for (int i = 0; i < 256; i++) {
        pattern[i] = (unsigned char)(i ^ 0x5A);
    }
    CHECK(fwrite(pattern, 1, sizeof(pattern), f) == sizeof(pattern),
          "fwrite failed");
    CHECK(fflush(f) == 0, "fflush failed");
    int fd = JT_FD_OF(f);
    CHECK(fd >= 0, "fileno failed");
    if (fd < 0) {
        fclose(f);
        fails++;
        return;
    }
#define JT_NWIN 300
    static unsigned char cells[JT_NWIN];
    jt_io_spec_t specs[JT_NWIN];
    for (int i = 0; i < JT_NWIN; i++) {
        specs[i].offset = (uint64_t)(i % 256);
        specs[i].length = 1;
        specs[i].dst = &cells[i];
        cells[i] = 0;
    }
    // 途中にno-opを混ぜてskip経路も踏む。
    specs[10].length = 0;
    specs[10].dst = NULL;
    cells[10] = 0xAA;  // no-opなので触られない想定 (検証は他セル中心)
    int rc = jt_io_pread_batch_uring(fd, specs, JT_NWIN);
    CHECK(rc == JT_OK, "window rc=%d errno=%d", rc, errno);
    if (rc == JT_OK) {
        for (int i = 0; i < JT_NWIN; i++) {
            if (i == 10) {
                continue;
            }
            unsigned char want = pattern[i % 256];
            if (cells[i] != want) {
                CHECK(cells[i] == want, "cell[%d]=%u want %u", i, cells[i],
                      want);
                break;
            }
        }
    }
    fclose(f);
#undef JT_NWIN
}
#else
static void test_fallback_enosys(void) {
    // liburing無効ビルドのfallback: n>0はJT_ERR_NOSUP + errno=ENOSYS。
    // NOSUP追従済み (MINOR-1)。
    FILE *f = tmpfile();
    CHECK(f != NULL, "tmpfile failed: %s", strerror(errno));
    if (f == NULL) {
        return;
    }
    unsigned char cell[8] = {0};
    jt_io_spec_t spec;
    spec.offset = 0;
    spec.length = sizeof(cell);
    spec.dst = cell;
    int fd = JT_FD_OF(f);
    CHECK(fd >= 0, "fileno failed");
    if (fd >= 0) {
        errno = 0;
        int rc = jt_io_pread_batch_uring(fd, &spec, 1);
        CHECK(rc == JT_ERR_NOSUP, "fallback rc=%d want NOSUP", rc);
        CHECK(errno == ENOSYS, "fallback errno=%d want ENOSYS(%d)", errno,
              ENOSYS);
    }
    fclose(f);
}
#endif

int main(void) {
    test_n0_ok();
#ifdef HAVE_LIBURING
    test_roundtrip();
    test_window();
#else
    test_fallback_enosys();
#endif
    if (fails == 0) {
#ifdef HAVE_LIBURING
        printf("uring: OK (backend enabled)\n");
#else
        printf("uring: OK (fallback NOSUP+ENOSYS)\n");
#endif
        return 0;
    }
    fprintf(stderr, "uring: %d FAIL(s)\n", fails);
    return 1;
}

#else
// 非Linux: jt_io_pread_batch_uringはヘッダ宣言なし (Linux-only)。
// コンパイル時NOSUP相当としてSKIP扱いでgreenとする。
int main(void) {
    printf("uring: SKIP (non-Linux, Linux-only API)\n");
    return 0;
}
#endif
