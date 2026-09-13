// jt_io_pread_batch_uring: Linux io_uring batch-read (Phase 2足場, skeleton).
// C11, restrict積極使用, errnoベース + goto cleanup (AGENTS.MD 7.1).
//
// buffered fd前提。O_DIRECTは使わない (io_batch.hのHOT/COLD方針とおり、
// 本スケルトンでO_DIRECT有効化禁止。4096整列要求なし。Phase 1と同一条件)。
// dense prefix計算とのオーバーラップは呼び出し側 (将来のNeuroPrefetcher経路)。
//
// sliding-window: batch=256、ring depthは2の冪(256)、
// prep_read → submit → wait_cqe + peekでdrain → resubmit の窓進行。
// GIL相当の排他なし (C層、1C1T前提。複数スレッドから同一fdへの並行呼び出しは
// 呼び出し側で直列化すること。pread同様オフセット指定読みのため安全だが、
// エラー時の取り消しは行わない)。
//
// liburing無効ビルド (JIMOTONO_USE_URING=OFF既定) および非Linuxでは
// fallback stub (JT_ERR_INVAL + errno=ENOSYS) を返す。
// TODO(NOSUP追従): MINOR-1で JT_ERR_NOSUP が新設され次第、本fallbackの
// JT_ERR_INVAL+ENOSYS を JT_ERR_NOSUP に置き換える。common.hには手を出さない。

#include "jimotono/io_batch.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__linux__) && defined(HAVE_LIBURING)
// ---- liburing有効経路 (Linuxのみ) ----
#include <limits.h>
#include <sys/types.h>
#include <unistd.h>

#include <liburing.h>

// sliding-window幅。papers.md §4のbatch=256に準拠。
#define JT_URING_BATCH 256
// ring depthは2の冪であること (io_uring要件)。BATCHと同値で窓を1:1対応。
#define JT_URING_DEPTH 256

int jt_io_pread_batch_uring(int fd, const jt_io_spec_t *restrict specs, size_t n) {
    if (n == 0) {
        return JT_OK;
    }
    if (fd < 0 || specs == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }

    // 事前検証: ring初期化前に引数不正を弾く (fallbackと同一条件)。
    size_t total = 0;
    for (size_t i = 0; i < n; i++) {
        size_t len = specs[i].length;
        if (len == 0) {
            continue;  // no-op (dstはNULL可)
        }
        if (specs[i].dst == NULL) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        if (specs[i].offset > (uint64_t)0x7FFFFFFFFFFFFFFFULL) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        total++;
    }
    if (total == 0) {
        return JT_OK;  // 全てno-op
    }

    int rc = JT_OK;
    size_t *done = NULL;
    size_t *retry = NULL;
    size_t nretry = 0;
    struct io_uring ring;
    int ring_inited = 0;

    done = (size_t *)calloc(n, sizeof(size_t));
    if (done == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    // retry stack: 1 drainで生じる再投入は高々inflight(<=BATCH)件のため、
    // 容量BATCHで十分 (fill前にdrain完結させる設計)。
    retry = (size_t *)malloc((size_t)JT_URING_BATCH * sizeof(size_t));
    if (retry == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }

    {
        int qret = io_uring_queue_init(JT_URING_DEPTH, &ring, 0);
        if (qret < 0) {
            errno = -qret;
            rc = JT_ERR_IO;
            goto cleanup;
        }
        ring_inited = 1;
    }

    {
        size_t next = 0;
        while (next < n && specs[next].length == 0) {
            next++;
        }
        size_t inflight = 0;
        size_t completed = 0;

        while (completed < total) {
            // ---- fill: retry優先で窓を埋める ----
            while (inflight < (size_t)JT_URING_BATCH && (nretry > 0 || next < n)) {
                size_t idx = 0;
                int from_retry = 0;
                if (nretry > 0) {
                    idx = retry[--nretry];
                    from_retry = 1;
                } else {
                    while (next < n && specs[next].length == 0) {
                        next++;
                    }
                    if (next >= n) {
                        break;
                    }
                    idx = next;
                    next++;
                }
                // 進捗不変条件: done[idx] < length (完了済みは再投入しない)。
                if (done[idx] >= specs[idx].length) {
                    // 論理矛盾 (二重完了)。次へ進める。
                    continue;
                }
                struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                if (sqe == NULL) {
                    // ring full: 消費を取り消してsubmit/drainへ。
                    if (from_retry) {
                        retry[nretry++] = idx;
                    } else {
                        next = idx;
                    }
                    break;
                }
                size_t d = done[idx];
                // offset加算のwraparound検出。
                if (specs[idx].offset > UINT64_MAX - (uint64_t)d) {
                    errno = EINVAL;
                    rc = JT_ERR_INVAL;
                    goto cleanup;
                }
                uint64_t cur_off = specs[idx].offset + (uint64_t)d;
                size_t rem = specs[idx].length - d;
                unsigned int chunk =
                    (rem > (size_t)UINT_MAX) ? UINT_MAX : (unsigned int)rem;
                if (chunk == 0) {
                    continue;
                }
                io_uring_prep_read(sqe, fd,
                                   (char *)specs[idx].dst + d, chunk, cur_off);
                io_uring_sqe_set_data(sqe, (void *)(uintptr_t)idx);
                inflight++;
            }

            if (inflight == 0) {
                // 残りがあるのにflight無しは論理矛盾。安全側でIO扱い。
                errno = EIO;
                rc = JT_ERR_IO;
                goto cleanup;
            }

            {
                int sub = io_uring_submit(&ring);
                if (sub < 0) {
                    errno = -sub;
                    rc = JT_ERR_IO;
                    goto cleanup;
                }
            }

            // ---- drain: 最低1件wait + peekで全件回収 ----
            {
                struct io_uring_cqe *cqe = NULL;
                int w = io_uring_wait_cqe(&ring, &cqe);
                if (w < 0) {
                    if (-w == EINTR) {
                        continue;
                    }
                    errno = -w;
                    rc = JT_ERR_IO;
                    goto cleanup;
                }
                for (;;) {
                    size_t idx = (size_t)(uintptr_t)io_uring_cqe_get_data(cqe);
                    int res = cqe->res;
                    io_uring_cqe_seen(&ring, cqe);
                    cqe = NULL;
                    inflight--;
                    if (idx >= n) {
                        errno = EIO;
                        rc = JT_ERR_IO;
                        goto cleanup;
                    }
                    if (res < 0) {
                        if (-res == EINTR) {
                            // 再投入 (retry容量はBATCHで足りる)。
                            if (nretry >= (size_t)JT_URING_BATCH) {
                                errno = EIO;
                                rc = JT_ERR_IO;
                                goto cleanup;
                            }
                            retry[nretry++] = idx;
                        } else {
                            errno = -res;
                            rc = JT_ERR_IO;
                            goto cleanup;
                        }
                    } else if (res == 0) {
                        errno = EIO;  // EOF前打ち切り (同期版と同一)
                        rc = JT_ERR_IO;
                        goto cleanup;
                    } else {
                        done[idx] += (size_t)res;
                        if (done[idx] < specs[idx].length) {
                            // short-read / 巨大specの分割chunk残: 再投入。
                            if (nretry >= (size_t)JT_URING_BATCH) {
                                errno = EIO;
                                rc = JT_ERR_IO;
                                goto cleanup;
                            }
                            retry[nretry++] = idx;
                        } else {
                            completed++;
                        }
                    }
                    if (io_uring_peek_cqe(&ring, &cqe) != 0) {
                        break;
                    }
                }
            }
        }
    }

cleanup:
    if (ring_inited) {
        io_uring_queue_exit(&ring);
    }
    free(retry);
    free(done);
    return rc;
}

#else
// ---- fallback stub (非Linux または HAVE_LIBURING未定義) ----
// buffered同期版 jt_io_pread_batch を呼び出し側で使うこと。
// TODO(NOSUP追従): MINOR-1で JT_ERR_NOSUP が新設され次第、本fallbackの
// JT_ERR_INVAL+ENOSYS を JT_ERR_NOSUP に置き換える。common.hには手を出さない。
int jt_io_pread_batch_uring(int fd, const jt_io_spec_t *restrict specs, size_t n) {
    if (n == 0) {
        return JT_OK;
    }
    (void)fd;
    (void)specs;
    errno = ENOSYS;
    return JT_ERR_INVAL;
}
#endif
