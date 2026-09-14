// jt_io: 共通I/O薄層 (submit/poll/wait) + OS別backend最小実装。
// C11、restrict積極使用、errnoベース + goto cleanup (AGENTS.MD 7.1)。
// 既存3経路 (io_batch.c / io_direct.c / io_batch_uring.c) は変更せずラップする。
//
// 方針 (analysis/p3-io-abstraction.md §1.4):
//   HOT : jt_io_pread_batchへ直結 (同期実行＋即時完了token)。
//   COLD: 整列事前検査 (fail-closed) 後、OS能力で選択。
//     Linux+HAVE_LIBURING: jt_io_pread_batch_uring (buffered fd前提)。
//       direct fd + uringの結合は意図的に行わない (分離維持)。
//     Linux (liburing無効): jt_io_direct_pread per-reqループ
//       (O_DIRECT足場への接続。整列済みのためdirect/buffered両fdで動作)。
//       JIMOTONO_USE_URING既定OFFはCMake側で維持 (本ファイルはHAVE_LIBURING
//       ガードのみ)。
//     macOS/Windows: jt_io_pread_batch同期fallback (NOSUPを返さず完了扱い)。
//
// macOS F_NOCACHEは汚染低減に留まる (ヘッダのRULE転記参照)。O_DIRECT相当の
// 保証はない旨を明記し、fcntl失敗は無視する (best-effort)。
// macOS kqueueはreadiness通知のみ (完了通知ではない)。I/O発行はしない。
// Windows NO_BUFFERING/IOCPは将来用スタブ (常時NOSUP。ビルド阻害しない)。

#include "jimotono/io.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/io_batch.h"
#include "jimotono/io_direct.h"

#if defined(__APPLE__)
#include <fcntl.h>
#include <sys/event.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif

// token表: 固定64枠round-robin。新規スレッドなし (1C1T前提)。
// 同一fdへの並行submitは呼び出し側で直列化すること。
#define JT_IO_SLOTS 64

static struct jt_io_slot {
    int used;
    uint64_t id;
    int rc;
    size_t total;  // 有効req数 (len>0のみ)
    size_t done;   // 完了数 (同期実装のためrc==OKならtotal、失敗時は0)
} g_slots[JT_IO_SLOTS];
static uint64_t g_next_id = 1;

static struct jt_io_slot *jt_io_find(uint64_t id) {
    if (id == 0) {
        return NULL;
    }
    for (size_t i = 0; i < (size_t)JT_IO_SLOTS; i++) {
        if (g_slots[i].used && g_slots[i].id == id) {
            return &g_slots[i];
        }
    }
    return NULL;
}

static struct jt_io_slot *jt_io_alloc_slot(uint64_t *out_id) {
    // round-robinで上書き (inflightは小窓想定)。
    uint64_t id = g_next_id++;
    if (g_next_id == 0) {
        g_next_id = 1;  // 0は無効値のためskip
    }
    size_t idx = (size_t)((id - 1u) % (uint64_t)JT_IO_SLOTS);
    g_slots[idx].used = 1;
    g_slots[idx].id = id;
    g_slots[idx].rc = JT_OK;
    g_slots[idx].total = 0;
    g_slots[idx].done = 0;
    *out_id = id;
    return &g_slots[idx];
}

// reqs→specs複写 (HOT/COLD共通)。len==0はskip対象として複写する
// (既存batch側がno-op扱いするため意味論一致)。戻り値は有効件数。
static size_t jt_io_copy_specs(const jt_io_req_t *restrict reqs, size_t n,
                               jt_io_spec_t *restrict specs) {
    size_t eff = 0;
    for (size_t i = 0; i < n; i++) {
        specs[i].offset = reqs[i].offset;
        specs[i].length = reqs[i].length;
        specs[i].dst = reqs[i].dst;
        if (reqs[i].length > 0) {
            eff++;
        }
    }
    return eff;
}

int jt_io_submit(int fd, jt_io_kind_t kind,
                 const jt_io_req_t *restrict reqs, size_t n,
                 jt_io_token_t *restrict out_tok) {
    if (out_tok == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    out_tok->id = 0;
    if (kind != JT_IO_HOT && kind != JT_IO_COLD) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n == 0) {
        // 何もせずJT_OK (fd/reqs検証なし、reqsはNULL可。既存流儀)。
        uint64_t id = 0;
        struct jt_io_slot *sl = jt_io_alloc_slot(&id);
        sl->rc = JT_OK;
        sl->total = 0;
        sl->done = 0;
        out_tok->id = id;
        return JT_OK;
    }
    if (reqs == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (fd < 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }

    int rc = JT_OK;
    jt_io_spec_t *specs = NULL;
    struct jt_io_slot *sl = NULL;
    uint64_t id = 0;

    specs = (jt_io_spec_t *)malloc(n * sizeof(jt_io_spec_t));
    if (specs == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    size_t total = jt_io_copy_specs(reqs, n, specs);
    if (total == 0) {
        // 全てno-op。
        sl = jt_io_alloc_slot(&id);
        sl->rc = JT_OK;
        sl->total = 0;
        sl->done = 0;
        out_tok->id = id;
        rc = JT_OK;
        goto cleanup;
    }

    if (kind == JT_IO_HOT) {
        // HOT: buffered同期。O_DIRECT禁止 (batch経路のみ)。
        int brc = jt_io_pread_batch(fd, specs, n);
        sl = jt_io_alloc_slot(&id);
        sl->rc = brc;
        sl->total = total;
        sl->done = (brc == JT_OK) ? total : 0;
        out_tok->id = id;
        rc = brc;
        goto cleanup;
    }

    // ---- COLD: 投入前整列検査 (fail-closed、I/O発行なし) ----
    for (size_t i = 0; i < n; i++) {
        if (reqs[i].length == 0) {
            continue;
        }
        if (reqs[i].dst == NULL) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        if (reqs[i].offset > (uint64_t)0x7FFFFFFFFFFFFFFFULL) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        if (!jt_io_direct_is_aligned(reqs[i].dst, reqs[i].offset,
                                     reqs[i].length,
                                     (size_t)JT_IO_DIRECT_ALIGN_SECTOR)) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
    }

#if defined(__linux__) && defined(HAVE_LIBURING)
    // Linux+liburing実経路 (buffered fd前提。direct+uring結合は行わない)。
    {
        int urc = jt_io_pread_batch_uring(fd, specs, n);
        sl = jt_io_alloc_slot(&id);
        sl->rc = urc;
        sl->total = total;
        sl->done = (urc == JT_OK) ? total : 0;
        out_tok->id = id;
        rc = urc;
        goto cleanup;
    }
#elif defined(__linux__)
    // Linux liburing無効時: O_DIRECT足場 (jt_io_direct_pread) のper-reqループ。
    // 整列済みのためdirect fd・buffered fallback fdのいずれでも動作する。
    {
        int lrc = JT_OK;
        for (size_t i = 0; i < n; i++) {
            if (specs[i].length == 0) {
                continue;
            }
            int prc = jt_io_direct_pread(fd, specs[i].dst, specs[i].length,
                                         specs[i].offset);
            if (prc != JT_OK) {
                lrc = prc;  // errnoはdirect_pread由来
                break;
            }
        }
        sl = jt_io_alloc_slot(&id);
        sl->rc = lrc;
        sl->total = total;
        sl->done = (lrc == JT_OK) ? total : 0;
        out_tok->id = id;
        rc = lrc;
        goto cleanup;
    }
#else
    // macOS/Windows: 同期fallback。新層はNOSUPを返さず完了扱いにする。
    // macOS NOTE: F_NOCACHE適用は呼び出し側判断 (本層では触らない)。
    //   F_NOCACHEは汚染低減に留まり、COLD＝汚染ゼロを保証しない。
    // Windows NOTE: FILE_FLAG_NO_BUFFERING/IOCPは将来用スタブ
    //   (jt_io_win_*は常時NOSUP)。本経路は_lseeki64+_read buffered fallback。
    {
        int brc = jt_io_pread_batch(fd, specs, n);
        sl = jt_io_alloc_slot(&id);
        sl->rc = brc;
        sl->total = total;
        sl->done = (brc == JT_OK) ? total : 0;
        out_tok->id = id;
        rc = brc;
        goto cleanup;
    }
#endif

cleanup:
    free(specs);
    return rc;
}

int jt_io_poll(jt_io_token_t tok, size_t *restrict done) {
    if (done == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *done = 0;
    struct jt_io_slot *sl = jt_io_find(tok.id);
    if (sl == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *done = sl->done;
    return sl->rc;
}

int jt_io_wait(jt_io_token_t tok) {
    struct jt_io_slot *sl = jt_io_find(tok.id);
    if (sl == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // 同期実装のため即時確定。EINTRリトライは既存層で済み。
    return sl->rc;
}

// ---- macOS: F_NOCACHE助言 (best-effort) ----
int jt_io_macos_set_nocache(int fd) {
    if (fd < 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
#ifdef __APPLE__
    // F_NOCACHEはfd単位の助言に留まる (汚染低減。O_DIRECT相当の保証なし)。
    // 失敗してもJT_ERR_IOにせず無視し、bufferedとして動作継続する。
    (void)fcntl(fd, F_NOCACHE, 1);
    return JT_OK;
#else
    errno = ENOSYS;
    return JT_ERR_NOSUP;
#endif
}

// ---- macOS: kqueue readiness通知 (最小実装) ----
int jt_io_macos_kqueue_ready(int fd, int timeout_ms, int *restrict out_ready) {
    if (out_ready == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_ready = 0;
    if (fd < 0 || timeout_ms < 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
#ifdef __APPLE__
    int rc = JT_OK;
    int kq = -1;
    kq = kqueue();
    if (kq < 0) {
        rc = JT_ERR_IO;  // errnoはkqueue由来
        goto cleanup;
    }
    {
        struct kevent ch;
        EV_SET(&ch, (uintptr_t)fd, EVFILT_READ, EV_ADD | EV_ONESHOT, 0, 0,
               NULL);
        struct timespec ts;
        ts.tv_sec = timeout_ms / 1000;
        ts.tv_nsec = (long)(timeout_ms % 1000) * 1000000L;
        struct kevent ev;
        // readiness通知のみ (完了通知ではない)。I/O自体は発行しない。
        int nev = kevent(kq, &ch, 1, &ev, 1, &ts);
        if (nev < 0) {
            if (errno == EINTR) {
                // 単発probeのためEINTRはtimeout扱い (not-ready) とする。
                *out_ready = 0;
                rc = JT_OK;
                goto cleanup;
            }
            rc = JT_ERR_IO;  // errnoはkevent由来
            goto cleanup;
        }
        *out_ready = (nev > 0) ? 1 : 0;
        rc = JT_OK;
    }
cleanup:
    if (kq >= 0) {
        close(kq);
    }
    return rc;
#else
    (void)fd;
    (void)timeout_ms;
    errno = ENOSYS;
    return JT_ERR_NOSUP;
#endif
}

// ---- Windows: FILE_FLAG_NO_BUFFERING + IOCP (将来用スタブ) ----
int jt_io_win_direct_open(const char *restrict path, int *restrict out_fd) {
    (void)path;
    (void)out_fd;
    // 将来の実経路予約。セクタ整列・倍数長・tailパディングはP3本体判断。
    // 現段階では buffered fallback (jt_io_pread_batch) が正経路。
    errno = ENOSYS;
    return JT_ERR_NOSUP;
}

int jt_io_win_iocp_submit(int fd, const jt_io_req_t *restrict reqs, size_t n,
                          jt_io_token_t *restrict out_tok) {
    (void)fd;
    (void)reqs;
    (void)n;
    (void)out_tok;
    // 将来: OVERLAPPED+GetQueuedCompletionStatus。別関数に分離すること。
    errno = ENOSYS;
    return JT_ERR_NOSUP;
}
