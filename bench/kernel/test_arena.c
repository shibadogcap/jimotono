// bench: arena smoke test (Phase 0足場の動作確認用)。
// 成功時 exit 0、失敗時 exit 1 + stderr。
#include <stdio.h>
#include "jimotono/arena.h"

int main(void) {
    jt_arena_t a = {0};
    if (jt_arena_init(&a, 4096) != JT_OK) {
        fprintf(stderr, "arena_init failed\n");
        return 1;
    }
    void *p1 = jt_arena_alloc(&a, 64);
    void *p2 = jt_arena_alloc(&a, 128);
    if (!p1 || !p2) {
        fprintf(stderr, "arena_alloc failed\n");
        jt_arena_fini(&a);
        return 1;
    }
    // 64B整列チェック
    if (((uintptr_t)p1 % 64u) != 0 || ((uintptr_t)p2 % 64u) != 0) {
        fprintf(stderr, "arena alignment broken\n");
        jt_arena_fini(&a);
        return 1;
    }
    // 枯渇チェック
    void *p3 = jt_arena_alloc(&a, 1u << 30);
    if (p3 != NULL) {
        fprintf(stderr, "arena over-alloc should fail\n");
        jt_arena_fini(&a);
        return 1;
    }
    jt_arena_reset(&a);
    if (jt_arena_used(&a) != 0) {
        fprintf(stderr, "arena reset broken\n");
        jt_arena_fini(&a);
        return 1;
    }
    jt_arena_fini(&a);
    printf("arena: OK\n");
    return 0;
}
