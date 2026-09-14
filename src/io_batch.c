// jt_io_batch: SSD streaming batch-read (Phase 1: buffered pread wrapper).
// C11, restrict, errno + goto cleanup. O_DIRECT未使用のためアライメント要求なし.
//
// TODO(Phase 2, O_DIRECT時): dstは4096整列・offset/lengthは4096倍数にすること。
// TODO(uring, Linuxのみ): jt_io_pread_batch_uring() を別関数として新設し、
//   liburing sliding-window (batch=256, ring sizeは2の冪,
//   prep_read → submit → wait_cqe + peekでdrain → resubmit) で非同期化する。
//   本関数は同期fallbackとして残す。

#include "jimotono/io_batch.h"

#include <errno.h>
#include <limits.h>
#include <stdint.h>

#if defined(_WIN32)
#include <io.h>
#include <stdio.h>
// _lseeki64 / _read 用。SEEK_SET は stdio.h 由来。
// _lseeki64 / _read 用。Phase 1はbuffered fallback (OVERLAPPED化はPhase 2 TODO)。
// NOTE: _readのcountはunsigned int幅のため、大きいlengthは分割して読む。
#else
#include <sys/types.h>
#include <unistd.h>
#endif

int jt_io_pread_batch(int fd, const jt_io_spec_t *restrict specs, size_t n) {
    if (n == 0) {
        return JT_OK;
    }
    if (fd < 0 || specs == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }

    int rc = JT_OK;
    for (size_t i = 0; i < n; i++) {
        uint64_t off = specs[i].offset;
        size_t len = specs[i].length;
        unsigned char *dst = (unsigned char *)specs[i].dst;
        if (len == 0) {
            continue;  // no-op (dstはNULL可)
        }
        if (dst == NULL) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
#if defined(_WIN32)
        // Windows buffered fallback: オフセット移動 + _read ループ。
        // NOTE: 1C1T前提。ファイルオフセットを動かすためスレッド共有FD不可。
        // FILE_FLAG_NO_BUFFERING / OVERLAPPED は Phase 2 TODO。
        if (off > (uint64_t)_I64_MAX) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        if (_lseeki64(fd, (__int64)off, SEEK_SET) < 0) {
            rc = JT_ERR_IO;  // errnoは_lseeki64由来
            goto cleanup;
        }
        size_t done = 0;
        while (done < len) {
            size_t rem = len - done;
            unsigned int chunk = (rem > (size_t)UINT_MAX) ? UINT_MAX : (unsigned int)rem;
            int r = _read(fd, dst + done, chunk);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                rc = JT_ERR_IO;
                goto cleanup;
            }
            if (r == 0) {
                errno = EIO;  // EOF前打ち切り
                rc = JT_ERR_IO;
                goto cleanup;
            }
            done += (size_t)r;
        }
#else
        // POSIX pread ループ (Linux/macOS共通のPhase 1正経路)。
        // macOS NOTE: O_DIRECTなし。コールド行のキャッシュ汚染が問題になれば
        //   fcntl(fd, F_NOCACHE, 1) を呼び出し側で検討 (本関数では触らない)。
        // Linux NOTE: io_uring+O_DIRECTは別関数に分離 (ヘッダのTODO参照)。
        if (off > (uint64_t)0x7FFFFFFFFFFFFFFFULL) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        off_t base = (off_t)off;
        size_t done = 0;
        while (done < len) {
            ssize_t r = pread(fd, dst + done, len - done, base + (off_t)done);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                rc = JT_ERR_IO;  // errnoはpread由来
                goto cleanup;
            }
            if (r == 0) {
                errno = EIO;  // EOF前打ち切り
                rc = JT_ERR_IO;
                goto cleanup;
            }
            done += (size_t)r;
        }
#endif
    }

cleanup:
    return rc;
}

int jt_io_runs_compress(const uint32_t *restrict ids, size_t n,
                        uint64_t record_bytes, uint64_t base_offset,
                        jt_io_run_t *restrict out, size_t out_cap,
                        size_t *restrict out_n) {
    int rc = JT_OK;
    if (out_n == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_n = 0;
    if (n == 0) {
        if (record_bytes == 0) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        return JT_OK;  // ids/outはNULL可
    }
    if (ids == NULL || out == NULL || record_bytes == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }

    // 単一パス: run確定ごとに nruns<out_cap なら書き込む。
    // out_cap不足時は *out_n=必要数 + JT_ERR_NOMEM (errno=ENOMEM)。
    size_t nruns = 0;
    uint64_t run_start = ids[0];
    uint64_t run_len = 1;
    uint32_t prev = ids[0];

    for (size_t i = 1; i < n; i++) {
        uint32_t cur = ids[i];
        if (cur < prev) {
            errno = EINVAL;  // 未ソート
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        if (cur == prev) {
            continue;  // 重複は畳む
        }
        // prev==UINT32_MAXの次は連続とみなさない (wraparound防止)。
        if ((uint64_t)cur == (uint64_t)prev + 1u) {
            run_len++;
        } else {
            // run確定: オーバーフローチェック後に書き込み。
            if (run_len > UINT64_MAX / record_bytes ||
                run_start > (UINT64_MAX - base_offset) / record_bytes) {
                errno = EINVAL;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
            uint64_t off = base_offset + run_start * record_bytes;
            size_t rlen = (size_t)(run_len * record_bytes);
            // size_tが64bit未満環境での切り詰め検出。
            if ((uint64_t)rlen != run_len * record_bytes) {
                errno = EINVAL;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
            if (nruns < out_cap) {
                out[nruns].offset = off;
                out[nruns].length = rlen;
            }
            nruns++;
            run_start = cur;
            run_len = 1;
        }
        prev = cur;
    }
    // 最終run確定。
    if (run_len > UINT64_MAX / record_bytes ||
        run_start > (UINT64_MAX - base_offset) / record_bytes) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    {
        uint64_t off = base_offset + run_start * record_bytes;
        size_t rlen = (size_t)(run_len * record_bytes);
        if ((uint64_t)rlen != run_len * record_bytes) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        if (nruns < out_cap) {
            out[nruns].offset = off;
            out[nruns].length = rlen;
        }
        nruns++;
    }

    *out_n = nruns;
    if (nruns > out_cap) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }

cleanup:
    return rc;
}
