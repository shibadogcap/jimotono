#pragma once
#ifndef JIMOTONO_COMMON_H
#define JIMOTONO_COMMON_H
// JIMOTONO common definitions: C11, little-endian only, cross-platform.
// AGENTS.MD 7.1: restrict積極使用, errnoベース, goto cleanup.

#include <stddef.h>
#include <stdint.h>

#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#error "JIMOTONO supports little-endian only"
#endif

// Cache line alignment for weight/LUT pages (DESIGN 5.1, T-MAC kAllocAlignment=64).
#define JT_CACHELINE 64
#define JT_ALIGN64 _Alignas(JT_CACHELINE)

// Error codes (errnoベースの前段として自前コードも返す)
typedef enum jt_err {
    JT_OK = 0,
    JT_ERR_NOMEM = 1,
    JT_ERR_INVAL = 2,
    JT_ERR_IO = 3,
    JT_ERR_ALIGN = 4,
} jt_err_t;

// Clamp helper (分岐除去: CMOV化を期待、DESIGN 4.2)
static inline int jt_imin(int a, int b) { return a < b ? a : b; }
static inline int jt_imax(int a, int b) { return a > b ? a : b; }

// Version
#define JT_VERSION_MAJOR 0
#define JT_VERSION_MINOR 0
#define JT_VERSION_PATCH 1

#endif  // JIMOTONO_COMMON_H
