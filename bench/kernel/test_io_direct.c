// bench: O_DIRECT抽象化足場テスト (P2 GAP-3, DESIGN.MD §3)。
//   - 非Linux: Linux-only APIのためSKIPしてexit 0 (test_uring.cの流儀)。
//   - Linux: tmpdir往復一致 + アライン違反拒否 (fail-closed)。
// 成功時 exit 0、失敗時 exit 1 + stderr。
// C11、errnoベース (AGENTS.MD 7.1)。
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "jimotono/io_direct.h"

#ifdef __linux__
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

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

// テスト用定数: セクタ(512)/ページ(4096)両方の倍数でO_DIRECT可。
#define JT_OD_SIZE 8192

static void test_alloc_free(void) {
    void *p = NULL;
    // 正常: 4096整列で確保でき、実際に4096整列している。
    CHECK(jt_io_direct_alloc(&p, 4096, 4096) == JT_OK, "alloc 4K should be OK");
    if (p != NULL) {
        CHECK(((uintptr_t)p & (uintptr_t)4095) == 0, "alloc 4K misaligned");
        CHECK(jt_io_direct_is_aligned(p, 0, 4096, 4096) == 1,
              "is_aligned should accept 4K");
        jt_io_direct_free(p);
        p = NULL;
    }
    // 正常: 512整列も受付ける (512/4Kの両対応)。
    CHECK(jt_io_direct_alloc(&p, 512, 512) == JT_OK, "alloc 512 should be OK");
    if (p != NULL) {
        CHECK(((uintptr_t)p & (uintptr_t)511) == 0, "alloc 512 misaligned");
        jt_io_direct_free(p);
        p = NULL;
    }
    // 異常: 引数不正はINVAL (fail-closed)。
    CHECK(jt_io_direct_alloc(NULL, 4096, 4096) == JT_ERR_INVAL,
          "NULL out should fail");
    CHECK(jt_io_direct_alloc(&p, 0, 4096) == JT_ERR_INVAL,
          "size 0 should fail");
    CHECK(p == NULL, "failed alloc must leave *out NULL");
    CHECK(jt_io_direct_alloc(&p, 4096, 100) == JT_ERR_INVAL,
          "align 100 should fail");
    CHECK(jt_io_direct_alloc(&p, 4096, 0) == JT_ERR_INVAL,
          "align 0 should fail");
    CHECK(jt_io_direct_alloc(&p, 4096, 511) == JT_ERR_INVAL,
          "align 511 should fail");
    jt_io_direct_free(NULL);  // NULL安全 (何もしない)
}

static void test_open_invalid(void) {
    int fd = -99;
    CHECK(jt_io_direct_open(NULL, &fd) == JT_ERR_INVAL, "NULL path INVAL");
    CHECK(jt_io_direct_open("", &fd) == JT_ERR_INVAL, "empty path INVAL");
    CHECK(jt_io_direct_open("/no/such/file", NULL) == JT_ERR_INVAL,
          "NULL out_fd INVAL");
    CHECK(jt_io_direct_open("/no/such/file_xyz_jt", &fd) == JT_ERR_IO,
          "missing file should be IO");
    CHECK(jt_io_direct_close(-1) == JT_ERR_INVAL, "close(-1) INVAL");
}

static void test_roundtrip_and_align_reject(void) {
    // tmpdirに8192Bパターンをbuffered書きし、O_DIRECT openで読み戻す。
    // /tmpがtmpfs等でO_DIRECT非対応でもopen側fallbackで読めるはず。
    char dir[] = "/tmp/jt_odirect_XXXXXX";
    CHECK(mkdtemp(dir) != NULL, "mkdtemp failed: %s", strerror(errno));
    if (dir[0] == '\0') {
        return;
    }
    char path[4096];
    snprintf(path, sizeof(path), "%s/cold.bin", dir);

    static unsigned char pattern[JT_OD_SIZE];
    for (size_t i = 0; i < (size_t)JT_OD_SIZE; i++) {
        pattern[i] = (unsigned char)(i ^ 0xA5);
    }
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL, "fopen write failed: %s", strerror(errno));
    if (f == NULL) {
        rmdir(dir);
        return;
    }
    CHECK(fwrite(pattern, 1, sizeof(pattern), f) == sizeof(pattern),
          "fwrite failed");
    CHECK(fclose(f) == 0, "fclose failed");

    int fd = -1;
    int rc = jt_io_direct_open(path, &fd);
    CHECK(rc == JT_OK, "direct open rc=%d errno=%d", rc, errno);
    if (rc != JT_OK || fd < 0) {
        unlink(path);
        rmdir(dir);
        fails++;
        return;
    }

    // 全文一致読み (offset 0, len 8192, 4096整列バッファ)。
    void *buf = NULL;
    CHECK(jt_io_direct_alloc(&buf, JT_OD_SIZE, 4096) == JT_OK, "alloc failed");
    if (buf == NULL) {
        jt_io_direct_close(fd);
        unlink(path);
        rmdir(dir);
        fails++;
        return;
    }
    rc = jt_io_direct_pread(fd, buf, JT_OD_SIZE, 0);
    CHECK(rc == JT_OK, "full pread rc=%d errno=%d", rc, errno);
    if (rc == JT_OK) {
        CHECK(memcmp(buf, pattern, JT_OD_SIZE) == 0, "full roundtrip mismatch");
    }
    // 後半ページ読み (offset 4096, len 4096)。
    memset(buf, 0, JT_OD_SIZE);
    rc = jt_io_direct_pread(fd, buf, 4096, 4096);
    CHECK(rc == JT_OK, "half pread rc=%d errno=%d", rc, errno);
    if (rc == JT_OK) {
        CHECK(memcmp(buf, pattern + 4096, 4096) == 0, "half roundtrip mismatch");
    }

    // アライン違反はfail-closed (I/O発行せずINVAL+errno=EINVAL)。
    {
        errno = 0;
        unsigned char *mis = (unsigned char *)buf + 1;  // 不整列dst
        rc = jt_io_direct_pread(fd, mis, 4096, 0);
        CHECK(rc == JT_ERR_INVAL, "misaligned dst rc=%d want INVAL", rc);
        CHECK(errno == EINVAL, "misaligned dst errno=%d want EINVAL", errno);
    }
    {
        errno = 0;
        rc = jt_io_direct_pread(fd, buf, 4096, 1);  // 不整列offset
        CHECK(rc == JT_ERR_INVAL, "misaligned off rc=%d want INVAL", rc);
        CHECK(errno == EINVAL, "misaligned off errno=%d want EINVAL", errno);
    }
    {
        errno = 0;
        rc = jt_io_direct_pread(fd, buf, 100, 0);  // 不整列length
        CHECK(rc == JT_ERR_INVAL, "misaligned len rc=%d want INVAL", rc);
        CHECK(errno == EINVAL, "misaligned len errno=%d want EINVAL", errno);
    }
    // 引数不正。
    CHECK(jt_io_direct_pread(-1, buf, 4096, 0) == JT_ERR_INVAL,
          "bad fd should fail");
    // len==0はno-opでOK (fd/dst検証なし)。
    CHECK(jt_io_direct_pread(-1, NULL, 0, 1) == JT_OK,
          "len==0 should be OK");
    CHECK(jt_io_direct_pread(fd, NULL, 0, 0) == JT_OK,
          "len==0 should be OK");
    {
        // len>0でdst NULLはINVAL。
        errno = 0;
        CHECK(jt_io_direct_pread(fd, NULL, 4096, 0) == JT_ERR_INVAL,
              "NULL dst should fail");
    }
    // EOF前打ち切りはIO (整列済みだが範囲外)。
    {
        errno = 0;
        // ファイルは8192B。offset 8192から512B読むとEOF。
        rc = jt_io_direct_pread(fd, buf, 512, (uint64_t)JT_OD_SIZE);
        CHECK(rc == JT_ERR_IO, "over-EOF rc=%d want IO", rc);
    }

    jt_io_direct_free(buf);
    CHECK(jt_io_direct_close(fd) == JT_OK, "close failed errno=%d", errno);
    unlink(path);
    rmdir(dir);
}

int main(void) {
    test_alloc_free();
    test_open_invalid();
    test_roundtrip_and_align_reject();
    if (fails == 0) {
        printf("io_direct: OK (O_DIRECT scaffold)\n");
        return 0;
    }
    fprintf(stderr, "io_direct: %d FAIL(s)\n", fails);
    return 1;
}

#else
// 非Linux: Linux-only APIのためSKIP扱いでgreenとする (test_uring.c流儀)。
#include <stdio.h>
int main(void) {
    printf("io_direct: SKIP (non-Linux, Linux-only API)\n");
    return 0;
}
#endif
