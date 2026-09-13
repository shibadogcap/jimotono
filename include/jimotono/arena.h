#pragma once
#ifndef JIMOTONO_ARENA_H
#define JIMOTONO_ARENA_H
// Arena / pool allocator (AGENTS.MD 7.1).
// 単調bump arena: 推論の一時バッファ用。freeはリセットのみ。
// 設計: 64B整列保証、O(1)確保、スレッド非安全(1C1T前提, DESIGN 2.2)。

#include <stddef.h>
#include "jimotono/common.h"

typedef struct jt_arena {
    unsigned char *base;  // 64B aligned (restrictは引数側で付与)
    size_t cap;           // total bytes
    size_t off;           // bump offset
} jt_arena_t;

// capバイトで初期化。baseは内部でaligned_alloc相当により確保。
// 戻り値: JT_OK / JT_ERR_NOMEM / JT_ERR_INVAL
int jt_arena_init(jt_arena_t *restrict a, size_t cap);

// 解放 (二重free安全: base==NULLなら何もしない)
void jt_arena_fini(jt_arena_t *restrict a);

// sizeバイト確保 (64B整列切り上げ)。失敗時NULL + errno=ENOMEM。
void *jt_arena_alloc(jt_arena_t *restrict a, size_t size);

// 使用量リセット (メモリは保持)
void jt_arena_reset(jt_arena_t *restrict a);

// 使用量 / 残量
size_t jt_arena_used(const jt_arena_t *restrict a);
size_t jt_arena_free(const jt_arena_t *restrict a);

#endif  // JIMOTONO_ARENA_H
