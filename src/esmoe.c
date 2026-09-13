// jt_esmoe: ES-MoE SSDオフロード足場。
// C11, restrict, errno + goto cleanup (AGENTS.MD 7.1)。malloc/freeのみ。
// 現状は buffered pread で機能正しさ優先。O_DIRECT/io_uringはTODO。
// llama.cpp等リンクなし。

#include "jimotono/esmoe.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/io_batch.h"

#if defined(_WIN32)
#include <io.h>
#define JT_ESMOE_FD_OF(f) _fileno(f)
#else
#include <unistd.h>
#define JT_ESMOE_FD_OF(f) fileno(f)
#endif

// 内部pwrite: backingへのoffset書き込み (POSIXはpwrite、Windowsはseek+write)。
// 呼び出し側は単一スレッド (1C1T) 前提。
static int jt_esmoe_pwrite(int fd, FILE *f, uint64_t off, const void *buf,
                           size_t len) {
    if (len == 0) {
        return JT_OK;
    }
    if (buf == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
#if defined(_WIN32)
    (void)fd;
    // Windows: _fseeki64 + fwrite ループ (CRT fd共有のためFILE経由で統一)。
    if (_fseeki64(f, (__int64)off, SEEK_SET) != 0) {
        return JT_ERR_IO;  // errnoは_fseeki64由来
    }
    const unsigned char *p = (const unsigned char *)buf;
    size_t done = 0;
    while (done < len) {
        size_t r = fwrite(p + done, 1, len - done, f);
        if (r == 0) {
            if (ferror(f)) {
                return JT_ERR_IO;  // errnoはfwrite由来に準ずる
            }
            errno = EIO;
            return JT_ERR_IO;
        }
        done += r;
    }
    if (fflush(f) != 0) {
        return JT_ERR_IO;
    }
    return JT_OK;
#else
    (void)f;
    const unsigned char *p = (const unsigned char *)buf;
    if (off > (uint64_t)0x7FFFFFFFFFFFFFFFULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    off_t base = (off_t)off;
    size_t done = 0;
    while (done < len) {
        ssize_t r = pwrite(fd, p + done, len - done, base + (off_t)done);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return JT_ERR_IO;  // errnoはpwrite由来
        }
        if (r == 0) {
            errno = EIO;
            return JT_ERR_IO;
        }
        done += (size_t)r;
    }
    return JT_OK;
#endif
}

static int jt_esmoe_valid(const jt_esmoe_t *restrict es) {
    return es != NULL && es->file != NULL && es->fd >= 0 &&
           es->n_experts > 0 && es->expert_bytes > 0 && es->slots != NULL &&
           es->present != NULL && es->registered != NULL;
}

int jt_esmoe_init(jt_esmoe_t *restrict es, const jt_esmoe_cfg_t *restrict cfg) {
    if (es == NULL || cfg == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (cfg->n_experts == 0 || cfg->expert_bytes == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // expert_bytes * n_experts のオーバーフローチェック。
    if (cfg->expert_bytes > UINT64_MAX / (uint64_t)cfg->n_experts) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (cfg->row_bytes != 0) {
        if (cfg->row_bytes == 0 || cfg->row_bytes > cfg->expert_bytes ||
            cfg->expert_bytes % cfg->row_bytes != 0) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }

    int rc = JT_OK;
    FILE *f = NULL;
    unsigned char **slots = NULL;
    unsigned char *present = NULL;
    unsigned char *registered = NULL;

    f = tmpfile();
    if (f == NULL) {
        rc = JT_ERR_IO;  // errnoはtmpfile由来
        goto cleanup;
    }
    int fd = JT_ESMOE_FD_OF(f);
    if (fd < 0) {
        rc = JT_ERR_IO;
        goto cleanup;
    }
    slots = (unsigned char **)calloc(cfg->n_experts, sizeof(unsigned char *));
    if (slots == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    present = (unsigned char *)calloc(cfg->n_experts, sizeof(unsigned char));
    if (present == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    registered =
        (unsigned char *)calloc(cfg->n_experts, sizeof(unsigned char));
    if (registered == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }

    es->file = f;
    es->fd = fd;
    es->n_experts = cfg->n_experts;
    es->expert_bytes = cfg->expert_bytes;
    es->row_bytes =
        (cfg->row_bytes == 0) ? cfg->expert_bytes : cfg->row_bytes;
    es->slots = slots;
    es->present = present;
    es->registered = registered;
    memset(&es->stats, 0, sizeof(es->stats));
    f = NULL;
    slots = NULL;
    present = NULL;
    registered = NULL;

cleanup:
    free(registered);
    free(present);
    free(slots);
    if (f != NULL) {
        fclose(f);
    }
    return rc;
}

void jt_esmoe_fini(jt_esmoe_t *restrict es) {
    if (es == NULL) {
        return;
    }
    if (es->slots != NULL) {
        for (uint32_t i = 0; i < es->n_experts; i++) {
            free(es->slots[i]);
        }
        free(es->slots);
    }
    free(es->present);
    free(es->registered);
    if (es->file != NULL) {
        fclose((FILE *)es->file);
    }
    es->file = NULL;
    es->fd = -1;
    es->slots = NULL;
    es->present = NULL;
    es->registered = NULL;
    es->n_experts = 0;
    es->expert_bytes = 0;
    es->row_bytes = 0;
    memset(&es->stats, 0, sizeof(es->stats));
}

int jt_esmoe_register(jt_esmoe_t *restrict es, uint32_t expert_id,
                      const void *restrict data, size_t len) {
    if (!jt_esmoe_valid(es) || data == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (expert_id >= es->n_experts || len != es->expert_bytes) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t off = (uint64_t)expert_id * (uint64_t)es->expert_bytes;
    int rc = jt_esmoe_pwrite(es->fd, (FILE *)es->file, off, data, len);
    if (rc != JT_OK) {
        return rc;
    }
    es->registered[expert_id] = 1;
    // 登録済みキャッシュがあれば無効化 (古い重みの残留防止)。
    if (es->present[expert_id]) {
        free(es->slots[expert_id]);
        es->slots[expert_id] = NULL;
        es->present[expert_id] = 0;
    }
    return JT_OK;
}

int jt_esmoe_evict(jt_esmoe_t *restrict es, uint32_t expert_id) {
    if (!jt_esmoe_valid(es)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (expert_id >= es->n_experts) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (es->present[expert_id]) {
        free(es->slots[expert_id]);
        es->slots[expert_id] = NULL;
        es->present[expert_id] = 0;
        es->stats.evicts += 1;
    }
    return JT_OK;
}

int jt_esmoe_prefetch(jt_esmoe_t *restrict es, const uint32_t *restrict ids,
                      size_t n) {
    if (!jt_esmoe_valid(es)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n == 0) {
        return JT_OK;
    }
    if (ids == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (size_t i = 0; i < n; i++) {
        if (ids[i] >= es->n_experts || !es->registered[ids[i]]) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }

    int rc = JT_OK;
    // ミス分のみ jt_io_pread_batch で一括読み (spec配列構築)。
    jt_io_spec_t *specs = NULL;
    uint32_t *miss_ids = NULL;
    size_t nmiss = 0;
    for (size_t i = 0; i < n; i++) {
        if (!es->present[ids[i]]) {
            nmiss++;
        }
    }
    if (nmiss == 0) {
        es->stats.hits += (uint64_t)n;
        es->stats.prefetches += 1;
        return JT_OK;
    }
    specs = (jt_io_spec_t *)malloc(nmiss * sizeof(jt_io_spec_t));
    if (specs == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    miss_ids = (uint32_t *)malloc(nmiss * sizeof(uint32_t));
    if (miss_ids == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    {
        size_t k = 0;
        for (size_t i = 0; i < n; i++) {
            uint32_t id = ids[i];
            if (es->present[id]) {
                continue;
            }
            // 同一prefetch内の重複idは1回だけ読む。
            int dup = 0;
            for (size_t j = 0; j < k; j++) {
                if (miss_ids[j] == id) {
                    dup = 1;
                    break;
                }
            }
            if (dup) {
                continue;
            }
            unsigned char *buf =
                (unsigned char *)malloc(es->expert_bytes);
            if (buf == NULL) {
                errno = ENOMEM;
                rc = JT_ERR_NOMEM;
                // k件目までの確保分を巻き戻す (slots未登録のため直接free)。
                for (size_t j = 0; j < k; j++) {
                    free((void *)specs[j].dst);
                    specs[j].dst = NULL;
                }
                goto cleanup;
            }
            miss_ids[k] = id;
            specs[k].offset = (uint64_t)id * (uint64_t)es->expert_bytes;
            specs[k].length = es->expert_bytes;
            specs[k].dst = buf;
            k++;
        }
        nmiss = k;
    }

    rc = jt_io_pread_batch(es->fd, specs, nmiss);
    if (rc != JT_OK) {
        for (size_t j = 0; j < nmiss; j++) {
            free((void *)specs[j].dst);
        }
        goto cleanup;
    }
    for (size_t j = 0; j < nmiss; j++) {
        uint32_t id = miss_ids[j];
        // register/evict競合なし (1C1T) のため上書きは起きないはずだが、
        // 念のため既存slotがあれば解放して差し替え。
        free(es->slots[id]);
        es->slots[id] = (unsigned char *)specs[j].dst;
        specs[j].dst = NULL;
        es->present[id] = 1;
    }
    es->stats.misses += (uint64_t)nmiss;
    es->stats.hits += (uint64_t)n - (uint64_t)nmiss;
    es->stats.loads += (uint64_t)nmiss;
    es->stats.bytes_read += (uint64_t)nmiss * (uint64_t)es->expert_bytes;
    es->stats.prefetches += 1;
    rc = JT_OK;

cleanup:
    free(miss_ids);
    free(specs);
    return rc;
}

int jt_esmoe_load(jt_esmoe_t *restrict es, uint32_t expert_id,
                  void *restrict dst, size_t len) {
    if (!jt_esmoe_valid(es) || dst == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (expert_id >= es->n_experts || len != es->expert_bytes ||
        !es->registered[expert_id]) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (es->present[expert_id]) {
        memcpy(dst, es->slots[expert_id], es->expert_bytes);
        es->stats.hits += 1;
        return JT_OK;
    }
    unsigned char *buf = (unsigned char *)malloc(es->expert_bytes);
    if (buf == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    int rc = JT_OK;
    jt_io_spec_t spec;
    spec.offset = (uint64_t)expert_id * (uint64_t)es->expert_bytes;
    spec.length = es->expert_bytes;
    spec.dst = buf;
    rc = jt_io_pread_batch(es->fd, &spec, 1);
    if (rc != JT_OK) {
        free(buf);
        return rc;
    }
    memcpy(dst, buf, es->expert_bytes);
    free(es->slots[expert_id]);
    es->slots[expert_id] = buf;
    es->present[expert_id] = 1;
    es->stats.misses += 1;
    es->stats.loads += 1;
    es->stats.bytes_read += (uint64_t)es->expert_bytes;
    return JT_OK;
}

int jt_esmoe_read_rows(jt_esmoe_t *restrict es, uint32_t expert_id,
                       size_t row_start, size_t row_count,
                       void *restrict dst) {
    if (!jt_esmoe_valid(es) || dst == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (expert_id >= es->n_experts || !es->registered[expert_id]) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (row_count == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    size_t rb = es->row_bytes;
    if (rb == 0 || rb > es->expert_bytes) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    size_t rows_per_expert = es->expert_bytes / rb;
    if (row_start >= rows_per_expert ||
        row_count > rows_per_expert - row_start) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (row_count > UINT64_MAX / rb) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t len64 = (uint64_t)row_count * (uint64_t)rb;
    if (len64 > (uint64_t)SIZE_MAX) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    uint64_t base = (uint64_t)expert_id * (uint64_t)es->expert_bytes;
    uint64_t off = base + (uint64_t)row_start * (uint64_t)rb;
    jt_io_spec_t spec;
    spec.offset = off;
    spec.length = (size_t)len64;
    spec.dst = dst;
    int rc = jt_io_pread_batch(es->fd, &spec, 1);
    if (rc != JT_OK) {
        return rc;
    }
    es->stats.loads += 1;
    es->stats.bytes_read += len64;
    return JT_OK;
}

int jt_esmoe_stats(const jt_esmoe_t *restrict es,
                   jt_esmoe_stats_t *restrict out) {
    if (es == NULL || out == NULL || !jt_esmoe_valid(es)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out = es->stats;
    return JT_OK;
}
