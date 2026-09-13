// bench: arena smoke test (Phase 0足場の動作確認用)。
// 成功時 exit 0、失敗時 exit 1 + stderr。
#include <stdint.h>
#include <stdio.h>
#include "jimotono/arena.h"

static int check(int cond, const char *msg, jt_arena_t *a) {
    if (!cond) {
        fprintf(stderr, "%s\n", msg);
        jt_arena_fini(a);
        return 0;
    }
    return 1;
}

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
    void *p3 = jt_arena_alloc(&a, ((size_t)1 << 30));
    if (!check(p3 == NULL, "arena over-alloc should fail", &a)) {
        return 1;
    }
    // 不正入力チェック (P0レビュー指摘)
    jt_arena_t bad = {0};
    if (!check(jt_arena_init(NULL, 4096) == JT_ERR_INVAL, "init NULL should fail", &a)) {
        return 1;
    }
    if (!check(jt_arena_init(&bad, 0) == JT_ERR_INVAL, "init cap==0 should fail", &a)) {
        return 1;
    }
    if (!check(jt_arena_alloc(NULL, 64) == NULL, "alloc NULL should fail", &a)) {
        return 1;
    }
    if (!check(jt_arena_alloc(&a, 0) == NULL, "alloc size==0 should fail", &a)) {
        return 1;
    }
    if (!check(jt_arena_alloc(&a, SIZE_MAX) == NULL, "alloc SIZE_MAX should fail", &a)) {
        return 1;
    }
    jt_arena_fini(NULL);  // クラッシュしなければOK
    jt_arena_fini(&bad);  // 未初期化finiは無害
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
