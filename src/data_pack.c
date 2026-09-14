// jt_data_pack: トークンID列の素朴バイナリ形式 writer/reader。
// C11, restrict, errno + goto cleanup。依存はlibcのみ (HDF5不要)。
//
// NOTE: Windowsのmmap readerはTODOスタブ (JT_ERR_NOSUP)。fopen/fwrite系の
// writerはWindowsでも動作する (バイナリモード"wb")。

#include "jimotono/data_pack.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
// TODO(Windows-mmap): CreateFileMapping + MapViewOfFile によるreader対応。
// 現状は jt_dp_open が JT_ERR_NOSUP (+ENOSYS) を返すスタブ。
#include <errno.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// ---- FNV-1a 32bit (checksum簡易。衝突耐性より検出用) ----
#define JT_DP_FNV_BASIS 2166136261u
#define JT_DP_FNV_PRIME 16777619u

static uint32_t jt_dp_fnv_update(uint32_t h, const void *restrict data,
                                 size_t len) {
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= JT_DP_FNV_PRIME;
    }
    return h;
}

// ---- 内部状態 ----
typedef struct jt_dp_wstate {
    FILE *f;
    uint64_t *prefix;  // 累積トークン数。prefix[0]=0、要素数nseq+1、capは+1込み
    size_t nseq;
    size_t cap;
    uint64_t total;
    uint32_t hash;  // payload部分まで更新済みのFNV状態
} jt_dp_wstate_t;

typedef struct jt_dp_rstate {
    unsigned char *base;  // mmapまたはmalloc領域
    size_t len;           // ファイル長
    int is_mmap;          // 1=mmap (munmapで解放)、0=malloc (freeで解放)
    uint64_t nseq;
    uint64_t total;
    const uint32_t *tokens;  // base + 64
    const uint64_t *prefix;  // base + off_table_off
} jt_dp_rstate_t;

// ヘッダ64Bを素朴に組立/解釈する (LE前提のためmemcpyでよい)。
static void jt_dp_pack_u32(unsigned char *dst, uint32_t v) {
    memcpy(dst, &v, 4);
}

static void jt_dp_pack_u64(unsigned char *dst, uint64_t v) {
    memcpy(dst, &v, 8);
}

static uint32_t jt_dp_unpack_u32(const unsigned char *src) {
    uint32_t v = 0;
    memcpy(&v, src, 4);
    return v;
}

static uint64_t jt_dp_unpack_u64(const unsigned char *src) {
    uint64_t v = 0;
    memcpy(&v, src, 8);
    return v;
}

// writer内部: prefix領域の拡張 (必要要素数need=(nseq+1)+1を確保)。
static int jt_dp_wgrow(jt_dp_wstate_t *restrict ws) {
    size_t need = ws->nseq + 2;  // 新規1件追加後の要素数
    if (need <= ws->cap) {
        return JT_OK;
    }
    size_t ncap = ws->cap == 0 ? 16 : ws->cap * 2;
    while (ncap < need) {
        if (ncap > (SIZE_MAX / 2)) {
            errno = ENOMEM;
            return JT_ERR_NOMEM;
        }
        ncap *= 2;
    }
    if (ncap > SIZE_MAX / sizeof(uint64_t)) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    uint64_t *np = (uint64_t *)realloc(ws->prefix, ncap * sizeof(uint64_t));
    if (np == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    ws->prefix = np;
    ws->cap = ncap;
    return JT_OK;
}

int jt_dp_writer_open(jt_dp_writer_t *restrict w, const char *restrict path) {
    if (w == NULL || path == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    w->opaque = NULL;
    jt_dp_wstate_t *ws = (jt_dp_wstate_t *)calloc(1, sizeof(*ws));
    if (ws == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    ws->hash = JT_DP_FNV_BASIS;
    FILE *f = fopen(path, "wb");  // バイナリモード (Windows対応)
    if (f == NULL) {
        free(ws);  // errnoはfopen由来
        return JT_ERR_IO;
    }
    ws->f = f;
    // prefix表はopen時に16要素calloc確保しprefix[0]=0に固定する。
    // 空pack (addゼロ) のcloseでもprefix[0]が読めるようにする (SEGFAULT対策)。
    // 以降の拡張はrealloc (wgrow) で、インデックス1以降はadd時に代入される。
    ws->prefix = (uint64_t *)calloc(16, sizeof(uint64_t));
    if (ws->prefix == NULL) {
        fclose(f);
        free(ws);
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    ws->cap = 16;
    // プレースホルダ64B (close時にbackpatch)。書式不正の部分読取りを避ける
    // ため、マジック・版数・header_lenだけ先に埋める。
    unsigned char hdr[JT_DP_HEADER_LEN];
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = JT_DP_MAGIC0;
    hdr[1] = JT_DP_MAGIC1;
    hdr[2] = JT_DP_MAGIC2;
    hdr[3] = JT_DP_MAGIC3;
    jt_dp_pack_u32(hdr + 4, JT_DP_VERSION);
    jt_dp_pack_u32(hdr + 8, JT_DP_HEADER_LEN);
    if (fwrite(hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        int rc = JT_ERR_IO;  // errnoはfwrite由来
        fclose(f);
        free(ws);
        return rc;
    }
    w->opaque = ws;
    return JT_OK;
}

int jt_dp_writer_add(jt_dp_writer_t *restrict w,
                     const uint32_t *restrict ids, size_t len) {
    if (w == NULL || w->opaque == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (len > 0 && ids == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    jt_dp_wstate_t *ws = (jt_dp_wstate_t *)w->opaque;
    // 値域検証 (fail-closed: 語彙外IDは書込まない)。
    for (size_t i = 0; i < len; i++) {
        if (ids[i] >= JT_DP_VOCAB_SIZE) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    if (len > UINT64_MAX - ws->total) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (ws->total + len > UINT64_MAX / 4u) {
        errno = EINVAL;  // payloadバイト数オーバーフロー
        return JT_ERR_INVAL;
    }
    int rc = jt_dp_wgrow(ws);
    if (rc != JT_OK) {
        return rc;
    }
    if (len > 0) {
        if (fwrite(ids, sizeof(uint32_t), len, ws->f) != len) {
            return JT_ERR_IO;  // errnoはfwrite由来
        }
        // 書込んだLEバイト列と同一のメモリ像をハッシュ (LE前提)。
        ws->hash = jt_dp_fnv_update(ws->hash, ids, len * sizeof(uint32_t));
    }
    ws->total += (uint64_t)len;
    ws->nseq++;
    ws->prefix[ws->nseq] = ws->total;  // prefix[0]はcallocで0
    return JT_OK;
}

int jt_dp_writer_close(jt_dp_writer_t *restrict w) {
    if (w == NULL || w->opaque == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    jt_dp_wstate_t *ws = (jt_dp_wstate_t *)w->opaque;
    int rc = JT_OK;
    uint64_t off = 0;
    // payloadバイト数と表位置の決定 (オーバーフロー検査)。
    if (ws->total > UINT64_MAX / 4u ||
        (uint64_t)JT_DP_HEADER_LEN > UINT64_MAX - ws->total * 4u) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    off = (uint64_t)JT_DP_HEADER_LEN + ws->total * 4u;
    if (ws->nseq > (UINT64_MAX - 8u) / 8u) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    // オフセット表を追記し、表バイトもハッシュに含める (検証順と同一)。
    for (size_t i = 0; i <= ws->nseq; i++) {
        unsigned char buf[8];
        jt_dp_pack_u64(buf, ws->prefix[i]);
        if (fwrite(buf, 1, sizeof(buf), ws->f) != sizeof(buf)) {
            rc = JT_ERR_IO;
            goto cleanup;
        }
        ws->hash = jt_dp_fnv_update(ws->hash, buf, sizeof(buf));
    }
    // ヘッダbackpatch。
    {
        unsigned char hdr[JT_DP_HEADER_LEN];
        memset(hdr, 0, sizeof(hdr));
        hdr[0] = JT_DP_MAGIC0;
        hdr[1] = JT_DP_MAGIC1;
        hdr[2] = JT_DP_MAGIC2;
        hdr[3] = JT_DP_MAGIC3;
        jt_dp_pack_u32(hdr + 4, JT_DP_VERSION);
        jt_dp_pack_u32(hdr + 8, JT_DP_HEADER_LEN);
        jt_dp_pack_u32(hdr + 12, 0u);  // flags (uint16-pack予約、現状0)
        jt_dp_pack_u64(hdr + 16, (uint64_t)ws->nseq);
        jt_dp_pack_u64(hdr + 24, ws->total);
        jt_dp_pack_u64(hdr + 32, off);
        jt_dp_pack_u32(hdr + 40, ws->hash);
        // hdr[44..64)はreserved=0
        if (fseek(ws->f, 0, SEEK_SET) != 0) {
            rc = JT_ERR_IO;
            goto cleanup;
        }
        if (fwrite(hdr, 1, sizeof(hdr), ws->f) != sizeof(hdr)) {
            rc = JT_ERR_IO;
            goto cleanup;
        }
    }

cleanup:
    if (ws->f != NULL) {
        // fflush失敗もIO扱い (ENOSPC等の取こぼし防止)。
        if (fflush(ws->f) != 0) {
            rc = JT_ERR_IO;
        }
        fclose(ws->f);
        ws->f = NULL;
    }
    free(ws->prefix);
    free(ws);
    w->opaque = NULL;
    return rc;
}

int jt_dp_write_file(const char *restrict path,
                     const uint32_t *const *restrict seqs,
                     const size_t *restrict lens, size_t nseq) {
    if (path == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (nseq > 0 && (seqs == NULL || lens == NULL)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    jt_dp_writer_t w = {0};
    int rc = jt_dp_writer_open(&w, path);
    if (rc != JT_OK) {
        return rc;
    }
    for (size_t i = 0; i < nseq; i++) {
        // lens[i]==0 (空シーケンス) のときseqs[i]はNULL可。
        rc = jt_dp_writer_add(&w, (lens[i] == 0) ? NULL : seqs[i], lens[i]);
        if (rc != JT_OK) {
            // 部分ファイルはchecksum不一致でreaderに拒否される。
            // 書込み途中のFILEはcloseして資源だけ解放する。
            jt_dp_wstate_t *ws = (jt_dp_wstate_t *)w.opaque;
            if (ws != NULL) {
                if (ws->f != NULL) {
                    fclose(ws->f);
                }
                free(ws->prefix);
                free(ws);
                w.opaque = NULL;
            }
            return rc;
        }
    }
    return jt_dp_writer_close(&w);
}

// ---- reader ----

// 後方定義（Windows/POSIX共用）の前方宣言。Windows版jt_dp_openから使う。
static int jt_dp_validate(const unsigned char *restrict base, size_t len,
                          jt_dp_rstate_t *restrict rs);

#if defined(_WIN32)

// Windows: CreateFileMapping対応までの暫定としてfopen/fread＋mallocで読む
// （mmap相当の常駐。is_mmap=0でjt_dp_closeがfreeする）。
int jt_dp_open(const char *restrict path, jt_dp_reader_t *restrict out) {
    if (path == NULL || out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    out->opaque = NULL;
    int rc = JT_OK;
    FILE *f = NULL;
    unsigned char *base = NULL;
    size_t len = 0;
    jt_dp_rstate_t *rs = NULL;
    long flen = 0;

    f = fopen(path, "rb");
    if (f == NULL) {
        return JT_ERR_IO;  // errnoはfopen由来
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        rc = JT_ERR_IO;
        goto cleanup;
    }
    flen = ftell(f);
    if (flen <= 0) {
        errno = (flen == 0) ? EINVAL : EIO;  // 空ファイルはfail-closed
        rc = (flen == 0) ? JT_ERR_INVAL : JT_ERR_IO;
        goto cleanup;
    }
    if ((uint64_t)flen > SIZE_MAX) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        rc = JT_ERR_IO;
        goto cleanup;
    }
    len = (size_t)flen;
    base = (unsigned char *)malloc(len);
    if (base == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    if (fread(base, 1, len, f) != len) {
        errno = EIO;
        rc = JT_ERR_IO;
        goto cleanup;
    }
    rs = (jt_dp_rstate_t *)calloc(1, sizeof(*rs));
    if (rs == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    rs->base = base;
    rs->len = len;
    rs->is_mmap = 0;
    base = NULL;  // 所有権をrsへ
    rc = jt_dp_validate(rs->base, rs->len, rs);
    if (rc != JT_OK) {
        goto cleanup;
    }
    out->opaque = rs;
    rs = NULL;
    rc = JT_OK;

cleanup:
    if (f != NULL) {
        fclose(f);
    }
    free(base);
    if (rs != NULL) {
        free(rs->base);
        free(rs);
    }
    return rc;
}

#endif  // _WIN32 (Windows fread backendここまで)

// 全検証の本体（Windows/POSIX共用。MSVC C2129回避のためガード外）。
// base[0..len)を検査し、rstateを埋める。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM。
static int jt_dp_validate(const unsigned char *restrict base, size_t len,
                          jt_dp_rstate_t *restrict rs) {
    if (len < JT_DP_HEADER_LEN) {
        errno = EINVAL;  // 空・切詰めファイル
        return JT_ERR_INVAL;
    }
    if (base[0] != JT_DP_MAGIC0 || base[1] != JT_DP_MAGIC1 ||
        base[2] != JT_DP_MAGIC2 || base[3] != JT_DP_MAGIC3) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (jt_dp_unpack_u32(base + 4) != JT_DP_VERSION) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (jt_dp_unpack_u32(base + 8) != JT_DP_HEADER_LEN) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (jt_dp_unpack_u32(base + 12) != 0u) {
        // flags!=0: 将来形式 (uint16-pack等)。v1は拒否する。
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t nseq = jt_dp_unpack_u64(base + 16);
    uint64_t total = jt_dp_unpack_u64(base + 24);
    uint64_t taboff = jt_dp_unpack_u64(base + 32);
    uint32_t expect_ck = jt_dp_unpack_u32(base + 40);
    for (int i = 44; i < 64; i++) {
        if (base[i] != 0) {
            errno = EINVAL;  // reserved非ゼロ
            return JT_ERR_INVAL;
        }
    }
    if (total > UINT64_MAX / 4u) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t paybytes = total * 4u;
    if ((uint64_t)JT_DP_HEADER_LEN > UINT64_MAX - paybytes) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t want_taboff = (uint64_t)JT_DP_HEADER_LEN + paybytes;
    if (taboff != want_taboff) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (nseq > (UINT64_MAX - 8u) / 8u) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t tabbytes = (nseq + 1u) * 8u;
    if (taboff > UINT64_MAX - tabbytes) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t want_len = taboff + tabbytes;
    if (want_len != (uint64_t)len) {
        errno = EINVAL;  // 切詰め・末尾ゴミ
        return JT_ERR_INVAL;
    }
    // オフセット表の単調性と端点。
    const unsigned char *tab = base + taboff;
    uint64_t prev = jt_dp_unpack_u64(tab);
    if (prev != 0u) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (uint64_t i = 1; i <= nseq; i++) {
        uint64_t cur = jt_dp_unpack_u64(tab + i * 8u);
        if (cur < prev || cur > total) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        prev = cur;
    }
    if (prev != total) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // checksum (payload + 表の順。writerと同一順序)。
    uint32_t h = JT_DP_FNV_BASIS;
    h = jt_dp_fnv_update(h, base + JT_DP_HEADER_LEN, (size_t)paybytes);
    h = jt_dp_fnv_update(h, tab, (size_t)tabbytes);
    if (h != expect_ck) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // 語彙値域 (fail-closed: 範囲外IDを含むファイルは開かない)。
    const uint32_t *toks =
        (const uint32_t *)(const void *)(base + JT_DP_HEADER_LEN);
    for (uint64_t i = 0; i < total; i++) {
        uint32_t v = 0;
        memcpy(&v, &toks[i], 4);  // 非整列安全 (mmap先頭+64は整列するが念のため)
        if (v >= JT_DP_VOCAB_SIZE) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    rs->nseq = nseq;
    rs->total = total;
    rs->tokens = toks;
    rs->prefix = (const uint64_t *)(const void *)tab;
    return JT_OK;
}

#if !defined(_WIN32)  // POSIX (Linux/macOS): mmap + malloc fallback

int jt_dp_open(const char *restrict path, jt_dp_reader_t *restrict out) {
    if (path == NULL || out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    out->opaque = NULL;
    int rc = JT_OK;
    int fd = -1;
    struct stat st;
    unsigned char *base = NULL;
    size_t len = 0;
    int is_mmap = 0;
    jt_dp_rstate_t *rs = NULL;

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        return JT_ERR_IO;  // errnoはopen由来
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
        errno = EINVAL;  // 空ファイルはfail-closed
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    // mmap試行 (macOS/Linux共通)。失敗時はmalloc+readにfallback。
    {
        void *m =
            mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m != MAP_FAILED) {
            base = (unsigned char *)m;
            is_mmap = 1;
        } else {
            unsigned char *buf = (unsigned char *)malloc(len == 0 ? 1 : len);
            if (buf == NULL) {
                errno = ENOMEM;
                rc = JT_ERR_NOMEM;
                goto cleanup;
            }
            size_t done = 0;
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
    rs = (jt_dp_rstate_t *)calloc(1, sizeof(*rs));
    if (rs == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    rs->base = base;
    rs->len = len;
    rs->is_mmap = is_mmap;
    base = NULL;  // 所有権をrsへ
    rc = jt_dp_validate(rs->base, rs->len, rs);
    if (rc != JT_OK) {
        goto cleanup;
    }
    out->opaque = rs;
    rs = NULL;
    rc = JT_OK;

cleanup:
    if (fd >= 0) {
        close(fd);  // mmap後はfdを閉じてよい
    }
    if (base != NULL) {
        if (is_mmap) {
            munmap(base, len);
        } else {
            free(base);
        }
    }
    if (rs != NULL) {
        if (rs->base != NULL) {
            if (rs->is_mmap) {
                munmap(rs->base, rs->len);
            } else {
                free(rs->base);
            }
        }
        free(rs);
    }
    return rc;
}

#endif  // !defined(_WIN32) POSIX jt_dp_open

void jt_dp_close(jt_dp_reader_t *restrict r) {
    if (r == NULL || r->opaque == NULL) {
        return;
    }
#if defined(_WIN32)
    // Windows暫定実装はmalloc常駐（is_mmap=0）のためfreeで解放。
    // CreateFileMapping対応時はここを分岐させること。
    {
        jt_dp_rstate_t *rs = (jt_dp_rstate_t *)r->opaque;
        free(rs->base);
        free(rs);
    }
    r->opaque = NULL;
    return;
#else
    jt_dp_rstate_t *rs = (jt_dp_rstate_t *)r->opaque;
    if (rs->base != NULL) {
        if (rs->is_mmap) {
            munmap(rs->base, rs->len);
        } else {
            free(rs->base);
        }
    }
    free(rs);
    r->opaque = NULL;
#endif
}

static const jt_dp_rstate_t *jt_dp_get(const jt_dp_reader_t *restrict r) {
    if (r == NULL || r->opaque == NULL) {
        errno = EINVAL;
        return NULL;
    }
    return (const jt_dp_rstate_t *)r->opaque;
}

int jt_dp_nseq(const jt_dp_reader_t *restrict r, uint64_t *restrict out_n) {
    const jt_dp_rstate_t *rs = jt_dp_get(r);
    if (rs == NULL || out_n == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_n = rs->nseq;
    return JT_OK;
}

int jt_dp_seq_len(const jt_dp_reader_t *restrict r, uint64_t idx,
                  uint64_t *restrict out_len) {
    const jt_dp_rstate_t *rs = jt_dp_get(r);
    if (rs == NULL || out_len == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (idx >= rs->nseq) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t a = 0, b = 0;
    memcpy(&a, &rs->prefix[idx], 8);
    memcpy(&b, &rs->prefix[idx + 1], 8);
    *out_len = b - a;  // validate済み単調性によりb>=a
    return JT_OK;
}

int jt_dp_seq_ptr(const jt_dp_reader_t *restrict r, uint64_t idx,
                  const uint32_t **restrict out_ptr,
                  uint64_t *restrict out_len) {
    const jt_dp_rstate_t *rs = jt_dp_get(r);
    if (rs == NULL || out_ptr == NULL || out_len == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (idx >= rs->nseq) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t a = 0, b = 0;
    memcpy(&a, &rs->prefix[idx], 8);
    memcpy(&b, &rs->prefix[idx + 1], 8);
    *out_len = b - a;
    if (*out_len == 0) {
        *out_ptr = NULL;  // 空シーケンス (NULL + len 0)
        return JT_OK;
    }
    // tokensはu32整列 (base+64)。memcpy経由でなく直接参照でよい。
    *out_ptr = rs->tokens + a;
    return JT_OK;
}

int jt_dp_seq_copy(const jt_dp_reader_t *restrict r, uint64_t idx,
                   uint32_t *restrict dst, uint64_t cap,
                   uint64_t *restrict out_len) {
    const uint32_t *ptr = NULL;
    uint64_t slen = 0;
    if (dst == NULL || out_len == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    int rc = jt_dp_seq_ptr(r, idx, &ptr, &slen);
    if (rc != JT_OK) {
        return rc;
    }
    *out_len = slen;
    if (slen > cap) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    if (slen > 0) {
        memcpy(dst, ptr, (size_t)(slen * 4u));
    }
    return JT_OK;
}
