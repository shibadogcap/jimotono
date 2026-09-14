// bench: 3OS同一pack読み込みテスト (analysis/p3-io-abstraction.md §4.2)。
// 一致性テスト。性能合否なし。短時間correctnessのみ (RULE.md §6相互排除)。
//
// 対象:
//   .jtdp小fixture (nseq=8・total<=1K・全ID<48588・checksum正) +
//   neuron-major合成coldファイル (64行x256B、各バイト (row^0xA5)^col)。
// 配布: 生成器は固定パターン (乱数なし)。出力バイト列のSHA-256は実行後に
//   shell (shasum/sha256sum) で記録し、3OSで同一ハッシュを用いる。
//   本テスト自体はFNV-1aを表示する (SHA実装を持ち込まないため)。
//
// 手順 (3OS共通):
//   1. jt_dp_open→全検証→jt_dp_seq_copy全seq一致 (壊しfixtureはopen失敗)。
//   2. coldファイルをbuffered open→jt_io_pread_batchでspec配列
//      (runs_compress出力＋no-op混在＋EOF超過1件) 読み→先頭2件一致・
//      EOF超過はJT_ERR_IO。
//   3. 薄層 jt_io_submit/poll/waitのHOT/COLD往復一致＋fail-closed検査。
//   Linux追加: direct open→alloc(4096)→pread往復＋整列違反3件INVAL+EINVAL。
//     tmpfs fallbackは既存test_io_directが担保 (ここでは成功のみ期待)。
//     HAVE_LIBURING時のみuring一致、無効時はNOSUP+ENOSYS＋同期fallback一致。
//   macOS追加: F_NOCACHE適用fdと非適用fdで(2)の一致結果が同一
//     (性能差は記録のみ・合否に使わない) ＋kqueue readiness==1。
//     F_NOCACHEは汚染低減に留まる (O_DIRECT相当の保証なし)。
//   Windows追加: _lseeki64+_read経路で(2)が一致 (単一スレッド)。
//     NO_BUFFERING/IOCPスタブは常時NOSUP (将来用)。
//
// C11、errnoベース。同一ソースが3OSでビルド可能であること。
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/common.h"
#include "jimotono/data_pack.h"
#include "jimotono/io.h"
#include "jimotono/io_batch.h"
#include "jimotono/io_direct.h"

#if defined(_WIN32)
#include <io.h>
#define XP_FD_OF(f) _fileno(f)
#include <malloc.h>  // _aligned_malloc/_aligned_free (Windows実機用)
#else
#include <fcntl.h>
#include <unistd.h>
#define XP_FD_OF(f) fileno(f)
#endif

#define XP_JTDP "test_io_xpack_tmp.jtdp"
#define XP_BAD "test_io_xpack_bad.jtdp"
#define XP_COLD "test_io_xpack_cold.bin"

// cold: 64行 x 256B = 16384B。byte(off) = (row ^ 0xA5) ^ col。
#define XP_ROWS 64
#define XP_ROWBYTES 256
#define XP_COLDSZ ((size_t)XP_ROWS * (size_t)XP_ROWBYTES)

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

static unsigned char xp_cold_expect(uint64_t off) {
    uint64_t row = off / (uint64_t)XP_ROWBYTES;
    uint64_t col = off % (uint64_t)XP_ROWBYTES;
    return (unsigned char)(((row & 0xFFu) ^ 0xA5u) ^ (col & 0xFFu));
}

static uint32_t xp_fnv1a(const unsigned char *p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

// ポータブルなアライン付き確保 (COLD用512/4096)。
// POSIXはposix_memalign、Windowsは_aligned_malloc。
static int xp_aligned_alloc(void **out, size_t size, size_t align) {
    if (out == NULL || size == 0 || align < 512 || (align & (align - 1)) != 0) {
        return -1;
    }
    *out = NULL;
#if defined(_WIN32)
    void *p = _aligned_malloc(size, align);
    if (p == NULL) {
        return -1;
    }
    *out = p;
    return 0;
#else
    // C11 aligned_alloc (sizeはalign倍数であること。呼び出し側は512/512等)。
    if (size % align != 0) {
        return -1;
    }
    void *p = aligned_alloc(align, size);
    if (p == NULL) {
        return -1;
    }
    *out = p;
    return 0;
#endif
}

static void xp_aligned_free(void *p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

// ---- 1. .jtdp小fixture (nseq=8, total<=1K) ----
static const size_t XP_LENS[8] = {0, 1, 7, 16, 64, 128, 200, 100};

static void xp_fill_seq(uint64_t seq, uint32_t *dst, size_t len) {
    for (size_t j = 0; j < len; j++) {
        uint32_t v =
            (uint32_t)((seq * 131u + j * 17u + j * j) % JT_DP_VOCAB_SIZE);
        dst[j] = v;
    }
    // 語彙上限48587をどこかに含める (seq1の先頭)。
    if (seq == 1 && len > 0) {
        dst[0] = JT_DP_VOCAB_SIZE - 1u;
    }
}

static void test_pack_roundtrip(void) {
    uint32_t *seqs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t *cseqs[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 8; i++) {
        if (XP_LENS[i] > 0) {
            seqs[i] = (uint32_t *)calloc(XP_LENS[i], 4);
            if (seqs[i] == NULL) {
                fprintf(stderr, "FAIL oom seq %d\n", i);
                fails++;
                goto done;
            }
            xp_fill_seq((uint64_t)i, seqs[i], XP_LENS[i]);
        }
        cseqs[i] = seqs[i];
    }
    {
        size_t total = 0;
        for (int i = 0; i < 8; i++) {
            total += XP_LENS[i];
        }
        CHECK(total <= 1000, "total=%zu want <=1000", total);
    }

    remove(XP_JTDP);
    CHECK(jt_dp_write_file(XP_JTDP, cseqs, XP_LENS, 8) == JT_OK,
          "pack write failed errno=%d", errno);

    jt_dp_reader_t r = {0};
    CHECK(jt_dp_open(XP_JTDP, &r) == JT_OK, "pack open failed errno=%d",
          errno);
    if (r.opaque == NULL) {
        goto done;
    }
    {
        uint64_t n = 99;
        CHECK(jt_dp_nseq(&r, &n) == JT_OK && n == 8, "nseq=%llu want 8",
              (unsigned long long)n);
    }
    for (uint64_t i = 0; i < 8; i++) {
        uint64_t sl = 99;
        CHECK(jt_dp_seq_len(&r, i, &sl) == JT_OK && sl == XP_LENS[i],
              "seq%llu len=%llu want %zu", (unsigned long long)i,
              (unsigned long long)sl, XP_LENS[i]);
        if (XP_LENS[i] > 0) {
            uint32_t *buf =
                (uint32_t *)calloc(XP_LENS[i] + 1, sizeof(uint32_t));
            CHECK(buf != NULL, "oom copy buf");
            if (buf != NULL) {
                uint64_t got = 99;
                int rc =
                    jt_dp_seq_copy(&r, i, buf, XP_LENS[i], &got);
                CHECK(rc == JT_OK && got == XP_LENS[i],
                      "seq%llu copy rc=%d got=%llu", (unsigned long long)i,
                      rc, (unsigned long long)got);
                if (rc == JT_OK && got == XP_LENS[i]) {
                    CHECK(memcmp(buf, seqs[i], XP_LENS[i] * 4) == 0,
                          "seq%llu payload mismatch", (unsigned long long)i);
                    for (size_t j = 0; j < XP_LENS[i]; j++) {
                        if (buf[j] >= JT_DP_VOCAB_SIZE) {
                            CHECK(0, "seq%llu vocab overflow %u",
                                  (unsigned long long)i, buf[j]);
                            break;
                        }
                    }
                }
                free(buf);
            }
        }
    }
    // FNV表示 (SHA-256はshellで別途記録)。
    {
        uint32_t h = 2166136261u;
        for (int i = 0; i < 8; i++) {
            if (XP_LENS[i] > 0) {
                h = xp_fnv1a((const unsigned char *)seqs[i],
                             XP_LENS[i] * 4) ^
                    (h * 16777619u);
            }
        }
        printf("io_xpack: pack payload fnv1a=%08x total<=1K nseq=8\n", h);
    }
    jt_dp_close(&r);

    // 語彙値域外のwriteはINVAL (fail-closed)。
    {
        uint32_t bad[] = {JT_DP_VOCAB_SIZE};
        const uint32_t *bs[1] = {bad};
        const size_t bl[1] = {1};
        CHECK(jt_dp_write_file(XP_BAD, bs, bl, 1) == JT_ERR_INVAL,
              "vocab-overflow write should be INVAL");
        remove(XP_BAD);
    }

    // 壊しfixture: payload先頭1B反転はopen失敗。
    {
        FILE *f = fopen(XP_JTDP, "rb");
        CHECK(f != NULL, "reopen pack failed");
        if (f != NULL) {
            CHECK(fseek(f, 0, SEEK_END) == 0, "fseek end failed");
            long len = ftell(f);
            CHECK(len > 64, "pack len=%ld too small", len);
            rewind(f);
            unsigned char *buf = (unsigned char *)malloc(
                (size_t)(len <= 0 ? 1 : len));
            CHECK(buf != NULL, "oom corrupt buf");
            if (buf != NULL && len > 0) {
                CHECK(fread(buf, 1, (size_t)len, f) == (size_t)len,
                      "fread pack failed");
                fclose(f);
                f = NULL;
                buf[64] ^= 0xFFu;  // payload先頭
                FILE *g = fopen(XP_BAD, "wb");
                CHECK(g != NULL, "open bad failed");
                if (g != NULL) {
                    CHECK(fwrite(buf, 1, (size_t)len, g) == (size_t)len,
                          "fwrite bad failed");
                    fclose(g);
                    jt_dp_reader_t rb = {0};
                    CHECK(jt_dp_open(XP_BAD, &rb) != JT_OK,
                          "corrupt payload opened!");
                    jt_dp_close(&rb);
                    remove(XP_BAD);
                }
                free(buf);
                buf = NULL;
            }
            if (f != NULL) {
                fclose(f);
            }
        }
    }
    // 末尾ゴミ1B追加はopen失敗。
    {
        FILE *f = fopen(XP_JTDP, "rb");
        CHECK(f != NULL, "reopen pack2 failed");
        if (f != NULL) {
            CHECK(fseek(f, 0, SEEK_END) == 0, "fseek end2 failed");
            long len = ftell(f);
            rewind(f);
            unsigned char *buf =
                (unsigned char *)malloc((size_t)(len <= 0 ? 1 : len) + 1);
            CHECK(buf != NULL, "oom tail buf");
            if (buf != NULL && len > 0) {
                CHECK(fread(buf, 1, (size_t)len, f) == (size_t)len,
                      "fread pack2 failed");
                buf[len] = 0xAB;
                fclose(f);
                f = NULL;
                FILE *g = fopen(XP_BAD, "wb");
                CHECK(g != NULL, "open bad2 failed");
                if (g != NULL) {
                    CHECK(fwrite(buf, 1, (size_t)len + 1, g) ==
                              (size_t)len + 1,
                          "fwrite bad2 failed");
                    fclose(g);
                    jt_dp_reader_t rb = {0};
                    CHECK(jt_dp_open(XP_BAD, &rb) != JT_OK,
                          "trailing-garbage opened!");
                    jt_dp_close(&rb);
                    remove(XP_BAD);
                }
                free(buf);
                buf = NULL;
            }
            if (f != NULL) {
                fclose(f);
            }
        }
    }

done:
    for (int i = 0; i < 8; i++) {
        free(seqs[i]);
    }
}

// ---- 2. coldファイル + batch読み ----
static void test_cold_batch(void) {
    static unsigned char pattern[XP_COLDSZ];
    for (uint64_t o = 0; o < (uint64_t)XP_COLDSZ; o++) {
        pattern[(size_t)o] = xp_cold_expect(o);
    }
    printf("io_xpack: cold fnv1a=%08x size=%zu\n",
           xp_fnv1a(pattern, sizeof(pattern)), sizeof(pattern));

    FILE *f = fopen(XP_COLD, "wb");
    CHECK(f != NULL, "fopen cold write failed: %s", strerror(errno));
    if (f == NULL) {
        return;
    }
    CHECK(fwrite(pattern, 1, sizeof(pattern), f) == sizeof(pattern),
          "fwrite cold failed");
    CHECK(fclose(f) == 0, "fclose cold failed");
    f = NULL;

    f = fopen(XP_COLD, "rb");
    CHECK(f != NULL, "fopen cold read failed: %s", strerror(errno));
    if (f == NULL) {
        return;
    }
    int fd = XP_FD_OF(f);
    CHECK(fd >= 0, "fileno cold failed");

    // runs_compress出力相当: ids={0,1,2,5,6}, rec=256, base=0
    // → {(0,768),(1280,512)}。
    uint32_t ids[] = {0, 1, 2, 5, 6};
    jt_io_run_t runs[4] = {{0, 0}};
    size_t nruns = 0;
    CHECK(jt_io_runs_compress(ids, 5, XP_ROWBYTES, 0, runs, 4, &nruns) ==
              JT_OK,
          "runs_compress failed");
    CHECK(nruns == 2, "nruns=%zu want 2", nruns);
    if (nruns == 2) {
        CHECK(runs[0].offset == 0 && runs[0].length == 768,
              "run0=(%llu,%zu) want (0,768)",
              (unsigned long long)runs[0].offset, runs[0].length);
        CHECK(runs[1].offset == 1280 && runs[1].length == 512,
              "run1=(%llu,%zu) want (1280,512)",
              (unsigned long long)runs[1].offset, runs[1].length);
    }

    unsigned char *b0 = (unsigned char *)calloc(768, 1);
    unsigned char *b1 = (unsigned char *)calloc(512, 1);
    CHECK(b0 != NULL && b1 != NULL, "oom cold bufs");
    if (b0 != NULL && b1 != NULL && nruns == 2 && fd >= 0) {
        jt_io_spec_t specs[4];
        specs[0].offset = runs[0].offset;
        specs[0].length = 768;
        specs[0].dst = b0;
        specs[1].offset = runs[1].offset;
        specs[1].length = 512;
        specs[1].dst = b1;
        specs[2].offset = 0;
        specs[2].length = 0;  // no-op混在
        specs[2].dst = NULL;
        specs[3].offset = 0;
        specs[3].length = 0;
        specs[3].dst = NULL;
        CHECK(jt_io_pread_batch(fd, specs, 3) == JT_OK,
              "cold batch first2 rc=%d errno=%d", JT_OK, errno);
        if (memcmp(b0, pattern + 0, 768) != 0) {
            CHECK(0, "cold b0 mismatch");
        }
        if (memcmp(b1, pattern + 1280, 512) != 0) {
            CHECK(0, "cold b1 mismatch");
        }
        // EOF超過1件はJT_ERR_IO。
        {
            unsigned char tmp[8] = {0};
            jt_io_spec_t over = {(uint64_t)XP_COLDSZ + 1000000u, sizeof(tmp),
                                 tmp};
            CHECK(jt_io_pread_batch(fd, &over, 1) == JT_ERR_IO,
                  "over-EOF should be IO");
        }
    }
    free(b0);
    free(b1);
    if (f != NULL) {
        fclose(f);
    }
}

// ---- 3. 薄層 submit/poll/wait ----
static void test_thin_layer(void) {
    FILE *f = fopen(XP_COLD, "rb");
    CHECK(f != NULL, "thin: reopen cold failed");
    if (f == NULL) {
        return;
    }
    int fd = XP_FD_OF(f);
    CHECK(fd >= 0, "thin: fileno failed");
    if (fd < 0) {
        fclose(f);
        return;
    }

    // HOT往復 (整列要求なし)。
    {
        unsigned char h0[16] = {0};
        unsigned char h1[10] = {0};
        jt_io_req_t reqs[3];
        reqs[0].offset = 0;
        reqs[0].length = sizeof(h0);
        reqs[0].dst = h0;
        reqs[1].offset = 100;
        reqs[1].length = sizeof(h1);
        reqs[1].dst = h1;
        reqs[2].offset = 0;
        reqs[2].length = 0;
        reqs[2].dst = NULL;
        jt_io_token_t tok = {0};
        errno = 0;
        int src = jt_io_submit(fd, JT_IO_HOT, reqs, 3, &tok);
        CHECK(src == JT_OK && tok.id != 0, "thin HOT submit rc=%d", src);
        if (src == JT_OK && tok.id != 0) {
            size_t done = 99;
            CHECK(jt_io_poll(tok, &done) == JT_OK && done == 2,
                  "thin HOT poll done=%zu", done);
            CHECK(jt_io_wait(tok) == JT_OK, "thin HOT wait failed");
            for (size_t i = 0; i < sizeof(h0); i++) {
                if (h0[i] != xp_cold_expect(i)) {
                    CHECK(0, "thin HOT h0[%zu] mismatch", i);
                    break;
                }
            }
            for (size_t i = 0; i < sizeof(h1); i++) {
                if (h1[i] != xp_cold_expect(100 + i)) {
                    CHECK(0, "thin HOT h1[%zu] mismatch", i);
                    break;
                }
            }
        }
    }

    // COLD往復 (512整列必須)。
    {
        void *c0 = NULL;
        void *c1 = NULL;
        CHECK(xp_aligned_alloc(&c0, 512, 512) == 0, "thin COLD alloc0");
        CHECK(xp_aligned_alloc(&c1, 512, 512) == 0, "thin COLD alloc1");
        if (c0 != NULL && c1 != NULL) {
            memset(c0, 0, 512);
            memset(c1, 0, 512);
            jt_io_req_t reqs[2];
            reqs[0].offset = 0;
            reqs[0].length = 512;
            reqs[0].dst = c0;
            reqs[1].offset = 1024;
            reqs[1].length = 512;
            reqs[1].dst = c1;
            jt_io_token_t tok = {0};
            errno = 0;
            int src = jt_io_submit(fd, JT_IO_COLD, reqs, 2, &tok);
            CHECK(src == JT_OK && tok.id != 0,
                  "thin COLD submit rc=%d errno=%d", src, errno);
            if (src == JT_OK && tok.id != 0) {
                size_t done = 99;
                CHECK(jt_io_poll(tok, &done) == JT_OK && done == 2,
                      "thin COLD poll done=%zu", done);
                CHECK(jt_io_wait(tok) == JT_OK, "thin COLD wait failed");
                for (size_t i = 0; i < 512; i++) {
                    if (((unsigned char *)c0)[i] != xp_cold_expect(i)) {
                        CHECK(0, "thin COLD c0[%zu] mismatch", i);
                        break;
                    }
                    if (((unsigned char *)c1)[i] != xp_cold_expect(1024 + i)) {
                        CHECK(0, "thin COLD c1[%zu] mismatch", i);
                        break;
                    }
                }
            }
        }
        // COLD整列違反3件はfail-closed (I/O発行なし)。
        if (c0 != NULL) {
            jt_io_token_t tok = {0};
            jt_io_req_t r0 = {0, 512, (unsigned char *)c0 + 1};  // dst不整列
            errno = 0;
            CHECK(jt_io_submit(fd, JT_IO_COLD, &r0, 1, &tok) ==
                      JT_ERR_INVAL,
                  "COLD mis-dst should be INVAL");
            CHECK(errno == EINVAL, "COLD mis-dst errno=%d want EINVAL",
                  errno);
            jt_io_req_t r1 = {1, 512, c0};  // offset不整列
            errno = 0;
            CHECK(jt_io_submit(fd, JT_IO_COLD, &r1, 1, &tok) ==
                      JT_ERR_INVAL,
                  "COLD mis-off should be INVAL");
            CHECK(errno == EINVAL, "COLD mis-off errno=%d want EINVAL",
                  errno);
            jt_io_req_t r2 = {0, 100, c0};  // length不整列
            errno = 0;
            CHECK(jt_io_submit(fd, JT_IO_COLD, &r2, 1, &tok) ==
                      JT_ERR_INVAL,
                  "COLD mis-len should be INVAL");
            CHECK(errno == EINVAL, "COLD mis-len errno=%d want EINVAL",
                  errno);
        }
        xp_aligned_free(c0);
        xp_aligned_free(c1);
    }

    // n==0 / len==0 no-op、unknown token、引数不正。
    {
        jt_io_token_t tok = {0};
        CHECK(jt_io_submit(fd, JT_IO_HOT, NULL, 0, &tok) == JT_OK &&
                  tok.id != 0,
              "thin n==0 should be OK");
        if (tok.id != 0) {
            size_t done = 99;
            CHECK(jt_io_poll(tok, &done) == JT_OK && done == 0,
                  "thin n==0 poll done=%zu", done);
            CHECK(jt_io_wait(tok) == JT_OK, "thin n==0 wait failed");
        }
        jt_io_token_t bad = {0xFFFFFFFFFFFFFFFFull};
        CHECK(jt_io_poll(bad, &(size_t){0}) == JT_ERR_INVAL,
              "unknown poll should be INVAL");
        CHECK(jt_io_wait(bad) == JT_ERR_INVAL,
              "unknown wait should be INVAL");
        CHECK(jt_io_submit(fd, (jt_io_kind_t)99, NULL, 0, &tok) ==
                  JT_ERR_INVAL,
              "bad kind should be INVAL");
        CHECK(jt_io_submit(-1, JT_IO_HOT, NULL, 0, &tok) == JT_OK,
              "n==0 bad fd should still be OK");
    }
    fclose(f);
}

#ifdef __linux__
static void test_linux_extra(void) {
    // direct open→alloc(4096)→pread往復＋整列違反3件。
    // /tmpが非対応FSでもopen側fallbackで読めるはず。
    int fd = -1;
    CHECK(jt_io_direct_open(XP_COLD, &fd) == JT_OK, "linux direct open");
    if (fd < 0) {
        return;
    }
    void *buf = NULL;
    CHECK(jt_io_direct_alloc(&buf, 4096, 4096) == JT_OK, "linux alloc");
    if (buf == NULL) {
        jt_io_direct_close(fd);
        return;
    }
    CHECK(jt_io_direct_pread(fd, buf, 4096, 0) == JT_OK,
          "linux pread0 rc errno=%d", errno);
    if (1) {
        for (size_t i = 0; i < 4096; i++) {
            if (((unsigned char *)buf)[i] != xp_cold_expect(i)) {
                CHECK(0, "linux pread0 mismatch at %zu", i);
                break;
            }
        }
    }
    {
        errno = 0;
        CHECK(jt_io_direct_pread(fd, (unsigned char *)buf + 1, 4096, 0) ==
                  JT_ERR_INVAL,
              "linux mis-dst");
        CHECK(errno == EINVAL, "linux mis-dst errno=%d", errno);
        errno = 0;
        CHECK(jt_io_direct_pread(fd, buf, 4096, 1) == JT_ERR_INVAL,
              "linux mis-off");
        CHECK(errno == EINVAL, "linux mis-off errno=%d", errno);
        errno = 0;
        CHECK(jt_io_direct_pread(fd, buf, 100, 0) == JT_ERR_INVAL,
              "linux mis-len");
        CHECK(errno == EINVAL, "linux mis-len errno=%d", errno);
    }
#ifdef HAVE_LIBURING
    {
        // O_DIRECT fdのため整列済みbuf(4096B)を再利用する。
        // 非整列16B読みはカーネルがEINVALにする（direct契約）。
        jt_io_spec_t s = {0, 4096, buf};
        CHECK(jt_io_pread_batch_uring(fd, &s, 1) == JT_OK,
              "linux uring roundtrip");
        for (size_t i = 0; i < 16; i++) {
            if (((unsigned char *)buf)[i] != xp_cold_expect(i)) {
                CHECK(0, "linux uring mismatch at %zu", i);
                break;
            }
        }
    }
#else
    {
        // O_DIRECT fdのため整列済みbuf(4096B)でsync fallbackを確認する。
        jt_io_spec_t s = {0, 4096, buf};
        errno = 0;
        CHECK(jt_io_pread_batch_uring(fd, &s, 1) == JT_ERR_NOSUP,
              "linux uring fallback should be NOSUP");
        CHECK(errno == ENOSYS, "linux uring errno=%d want ENOSYS", errno);
        CHECK(jt_io_pread_batch(fd, &s, 1) == JT_OK,
              "linux sync fallback failed");
    }
#endif
    jt_io_direct_free(buf);
    CHECK(jt_io_direct_close(fd) == JT_OK, "linux close failed");
}
#endif

#ifdef __APPLE__
static void test_macos_extra(void) {
    // F_NOCACHE適用fdと非適用fdで同一一致 (性能差は記録のみ)。
    FILE *f0 = fopen(XP_COLD, "rb");
    FILE *f1 = fopen(XP_COLD, "rb");
    CHECK(f0 != NULL && f1 != NULL, "macos reopen failed");
    if (f0 == NULL || f1 == NULL) {
        if (f0 != NULL) {
            fclose(f0);
        }
        if (f1 != NULL) {
            fclose(f1);
        }
        return;
    }
    int fd0 = XP_FD_OF(f0);
    int fd1 = XP_FD_OF(f1);
    // COLD fdにのみ適用 (HOTには適用しない)。失敗は無視扱いでOK。
    CHECK(jt_io_macos_set_nocache(fd1) == JT_OK, "macos nocache failed");
    CHECK(jt_io_macos_set_nocache(-1) == JT_ERR_INVAL,
          "macos nocache bad fd should be INVAL");
    unsigned char a[32] = {0};
    unsigned char b[32] = {0};
    jt_io_spec_t s0 = {64, sizeof(a), a};
    jt_io_spec_t s1 = {64, sizeof(b), b};
    CHECK(jt_io_pread_batch(fd0, &s0, 1) == JT_OK, "macos plain read failed");
    CHECK(jt_io_pread_batch(fd1, &s1, 1) == JT_OK,
          "macos nocache read failed");
    CHECK(memcmp(a, b, sizeof(a)) == 0, "macos nocache mismatch");
    for (size_t i = 0; i < sizeof(a); i++) {
        if (a[i] != xp_cold_expect(64 + i)) {
            CHECK(0, "macos content mismatch at %zu", i);
            break;
        }
    }
    // kqueue readinessは1を期待 (regular fileはreadable)。
    {
        int ready = 0;
        CHECK(jt_io_macos_kqueue_ready(fd0, 1000, &ready) == JT_OK,
              "macos kqueue failed errno=%d", errno);
        CHECK(ready == 1, "macos kqueue ready=%d want 1", ready);
        CHECK(jt_io_macos_kqueue_ready(-1, 100, &ready) == JT_ERR_INVAL,
              "macos kqueue bad fd should be INVAL");
    }
    printf("io_xpack: macos F_NOCACHE consistency OK (pollution-reduction "
           "only, no O_DIRECT guarantee)\n");
    fclose(f0);
    fclose(f1);
}
#endif

#ifdef _WIN32
static void test_windows_extra(void) {
    // _lseeki64+_read経路 (単一スレッド1C1T前提) で一致。
    FILE *f = fopen(XP_COLD, "rb");
    CHECK(f != NULL, "win reopen failed");
    if (f == NULL) {
        return;
    }
    int fd = XP_FD_OF(f);
    unsigned char a[32] = {0};
    jt_io_spec_t s = {64, sizeof(a), a};
    CHECK(jt_io_pread_batch(fd, &s, 1) == JT_OK, "win batch read failed");
    for (size_t i = 0; i < sizeof(a); i++) {
        if (a[i] != xp_cold_expect(64 + i)) {
            CHECK(0, "win content mismatch at %zu", i);
            break;
        }
    }
    // NO_BUFFERING/IOCPスタブは常時NOSUP (将来用)。
    CHECK(jt_io_win_direct_open(XP_COLD, &(int){0}) == JT_ERR_NOSUP,
          "win direct stub should be NOSUP");
    CHECK(jt_io_win_iocp_submit(fd, NULL, 0, NULL) == JT_ERR_NOSUP,
          "win iocp stub should be NOSUP");
    fclose(f);
}
#endif

static void test_cross_stubs(void) {
// 各OSで他OSヘルパーがNOSUPであること (ビルド阻害しないことの確認)。
#ifndef __APPLE__
    CHECK(jt_io_macos_set_nocache(0) == JT_ERR_NOSUP,
          "non-mac nocache should be NOSUP");
    CHECK(jt_io_macos_kqueue_ready(0, 0, &(int){0}) == JT_ERR_NOSUP,
          "non-mac kqueue should be NOSUP");
#endif
    // winスタブは全OSでNOSUP (Windows上でも将来までNOSUP)。
    CHECK(jt_io_win_direct_open("x", &(int){0}) == JT_ERR_NOSUP,
          "win direct stub should be NOSUP");
    CHECK(jt_io_win_iocp_submit(0, NULL, 0, NULL) == JT_ERR_NOSUP,
          "win iocp stub should be NOSUP");
}

int main(void) {
    test_pack_roundtrip();
    test_cold_batch();
    test_thin_layer();
#ifdef __linux__
    test_linux_extra();
#endif
#ifdef __APPLE__
    test_macos_extra();
#endif
#ifdef _WIN32
    test_windows_extra();
#endif
    test_cross_stubs();
    remove(XP_JTDP);
    remove(XP_BAD);
    remove(XP_COLD);
    if (fails == 0) {
#if defined(__linux__)
        printf("io_xpack: OK (Linux real-run, consistency only)\n");
#elif defined(__APPLE__)
        printf("io_xpack: OK (macOS real-run, consistency only)\n");
#elif defined(_WIN32)
        printf("io_xpack: OK (Windows real-run, consistency only)\n");
#else
        printf("io_xpack: OK\n");
#endif
        return 0;
    }
    fprintf(stderr, "io_xpack: %d FAIL(s)\n", fails);
    return 1;
}
