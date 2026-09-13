// jt_arena: bump arena, C11, cross-platform (Linux/macOS/Windows).
// AGENTS.MD 7.1: errnoベース + goto cleanup パターンは呼び出し側で。

#include "jimotono/arena.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

#define JT_ARENA_ALIGN JT_CACHELINE

static size_t jt_align_up(size_t n, size_t a) {
    return (n + (a - 1u)) & ~(a - 1u);
}

int jt_arena_init(jt_arena_t *restrict a, size_t cap) {
    if (a == NULL || cap == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    size_t aligned = jt_align_up(cap, JT_ARENA_ALIGN);
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
    size_t n = jt_align_up(size, JT_ARENA_ALIGN);
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
