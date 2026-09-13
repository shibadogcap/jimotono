// jt_arena: bump arena, C11, cross-platform (Linux/macOS/Windows).
// AGENTS.MD 7.1: errnoベース + goto cleanup パターンは呼び出し側で。

#include "jimotono/arena.h"

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

#define JT_ARENA_ALIGN JT_CACHELINE

// 2の冪アライメント切り上げ。オーバーフロー時は0を返す番兵。
// P0レビュー指摘: SIZE_MAX付近のwraparoundで検査迂回・ヒープ破壊を防ぐ。
static size_t jt_align_up(size_t n, size_t a) {
    size_t mask = a - (size_t)1;
    if ((a & (a - (size_t)1)) != 0) {
        return 0;  // 2の冪以外は受け付けない
    }
    if (n > SIZE_MAX - mask) {
        return 0;  // 加算オーバーフロー番兵
    }
    return (n + mask) & ~mask;
}

int jt_arena_init(jt_arena_t *restrict a, size_t cap) {
    if (a == NULL || cap == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    size_t aligned = jt_align_up(cap, JT_ARENA_ALIGN);
    if (aligned == 0) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    void *p = NULL;
#if defined(_WIN32)
    p = _aligned_malloc(aligned, JT_ARENA_ALIGN);
    if (p == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
#else
    // aligned_alloc requires size % alignment == 0 (C11).
    p = aligned_alloc(JT_ARENA_ALIGN, aligned);
    if (p == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
#endif
    a->base = (unsigned char *)p;
    a->cap = aligned;
    a->off = 0;
    return JT_OK;
}

void jt_arena_fini(jt_arena_t *restrict a) {
    if (a == NULL || a->base == NULL) {
        return;
    }
#if defined(_WIN32)
    _aligned_free(a->base);
#else
    free(a->base);
#endif
    a->base = NULL;
    a->cap = 0;
    a->off = 0;
}

void *jt_arena_alloc(jt_arena_t *restrict a, size_t size) {
    if (a == NULL || a->base == NULL || size == 0) {
        errno = EINVAL;
        return NULL;
    }
    if (a->off > a->cap) {
        errno = EINVAL;  // 不変条件崩壊(未初期化直渡し等)の防御
        return NULL;
    }
    size_t n = jt_align_up(size, JT_ARENA_ALIGN);
    if (n == 0) {
        errno = ENOMEM;  // 切り上げオーバーフロー
        return NULL;
    }
    if (n > a->cap - a->off) {
        errno = ENOMEM;
        return NULL;
    }
    void *p = a->base + a->off;
    a->off += n;
    return p;
}

void jt_arena_reset(jt_arena_t *restrict a) {
    if (a == NULL) {
        return;
    }
    a->off = 0;
}

size_t jt_arena_used(const jt_arena_t *restrict a) {
    return (a != NULL) ? a->off : 0;
}

size_t jt_arena_free(const jt_arena_t *restrict a) {
    if (a == NULL || a->base == NULL) {
        return 0;
    }
    return a->cap - a->off;
}
