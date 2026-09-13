// jt_io_direct: O_DIRECT抽象化足場 (P2 GAP-3対応, DESIGN.MD §3)。
// C11, restrict積極使用, errnoベース + goto cleanup (AGENTS.MD 7.1)。
//
// Colibri式LRU+ページキャッシュ階層のうち、COLD側 (使い捨てに近い
// 低頻度routed expert / delta行) のみO_DIRECT streamingする。HOT側は
// buffered I/O (jt_io_pread_batch) を使い、本APIを呼ばないこと。
//
// Linux実経路 (#ifdef __linux__):
//   open(path, O_RDONLY|O_DIRECT) し、非対応FS (tmpfs等) の
//   EINVAL/EOPNOTSUPP/ENOTSUP/ENOSYS時は buffered open (O_RDONLY) に
//   fallbackする。allocはposix_memalign(512/4K)。preadは512整列を
//   事前検査 (fail-closed) してからpreadループする。
// 非Linux (#else):
//   fallback stub (JT_ERR_NOSUP + errno=ENOSYS)。既存流儀
//   (io_batch_uring.c fallback) に準じ、len==0等のno-opはJT_OK。

#include "jimotono/io_direct.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__linux__)
// ---- Linux実経路 ----
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

int jt_io_direct_open(const char *restrict path, int *restrict out_fd) {
    if (path == NULL || path[0] == '\0' || out_fd == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        int e = errno;
        if (e == EINVAL || e == EOPNOTSUPP || e == ENOTSUP || e == ENOSYS) {
            // 非対応FS (tmpfs等) / カーネル無効時: buffered fallback。
            fd = open(path, O_RDONLY);
            if (fd < 0) {
                return JT_ERR_IO;  // errnoはfallback open由来
            }
            *out_fd = fd;
            return JT_OK;
        }
        return JT_ERR_IO;  // errnoはopen由来 (ENOENT等)
    }
    *out_fd = fd;
    return JT_OK;
}

int jt_io_direct_alloc(void **out, size_t size, size_t align) {
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out = NULL;
    if (size == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (align < (size_t)JT_IO_DIRECT_ALIGN_SECTOR ||
        (align & (align - 1)) != 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    void *p = NULL;
    int prc = posix_memalign(&p, align, size);
    if (prc != 0) {
        if (prc == EINVAL) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    *out = p;
    return JT_OK;
}

void jt_io_direct_free(void *p) { free(p); }

int jt_io_direct_pread(int fd, void *restrict dst, size_t len,
                       uint64_t offset) {
    if (len == 0) {
        return JT_OK;
    }
    if (fd < 0 || dst == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // fail-closed: セクタ整列を事前検査 (I/O発行なし)。
    if (!jt_io_direct_is_aligned(dst, offset, len,
                                 (size_t)JT_IO_DIRECT_ALIGN_SECTOR)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (offset > (uint64_t)0x7FFFFFFFFFFFFFFFULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    int rc = JT_OK;
    off_t base = (off_t)offset;
    unsigned char *d = (unsigned char *)dst;
    size_t done = 0;
    while (done < len) {
        ssize_t r = pread(fd, d + done, len - done, base + (off_t)done);
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

cleanup:
    return rc;
}

int jt_io_direct_close(int fd) {
    if (fd < 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    while (close(fd) != 0) {
        if (errno == EINTR) {
            continue;
        }
        return JT_ERR_IO;  // errnoはclose由来
    }
    return JT_OK;
}

#else
// ---- 非Linux fallback stub (macOS/Windows) ----
// O_DIRECTなし。本足場ではF_NOCACHE / FILE_FLAG_NO_BUFFERINGに手を出さず
// NOSUPを返す。既存流儀 (io_batch_uring.c fallback) に準拠。

int jt_io_direct_open(const char *restrict path, int *restrict out_fd) {
    (void)path;
    (void)out_fd;
    errno = ENOSYS;
    return JT_ERR_NOSUP;
}

int jt_io_direct_alloc(void **out, size_t size, size_t align) {
    (void)size;
    (void)align;
    if (out != NULL) {
        *out = NULL;
    }
    errno = ENOSYS;
    return JT_ERR_NOSUP;
}

void jt_io_direct_free(void *p) { free(p); }

int jt_io_direct_pread(int fd, void *restrict dst, size_t len,
                       uint64_t offset) {
    if (len == 0) {
        return JT_OK;
    }
    (void)fd;
    (void)dst;
    (void)offset;
    errno = ENOSYS;
    return JT_ERR_NOSUP;
}

int jt_io_direct_close(int fd) {
    (void)fd;
    errno = ENOSYS;
    return JT_ERR_NOSUP;
}

#endif  // defined(__linux__)
