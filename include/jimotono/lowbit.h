#pragma once
#ifndef JIMOTONO_LOWBIT_H
#define JIMOTONO_LOWBIT_H
// Routed expert 低ビット副線 (Stage 3b-2, analysis/p3b-tmac-design.md 準拠)。
//
// 使い分け: gate/up INT2・down INT4。共有expert・注意・emb/headは対象外
// (W8A8本線またはfp32のまま)。
// モード (既定0=fp32。明示フラグ切替のみ。自動切替なし):
//   0 = OFF (fp32従来核のみ。速度・数値不変)。
//   1 = fake-quant (学習用): 量子化→逆量子化後にfp32 GEMM。bwdはSTE
//       (既存fp32 bwdをそのまま使う。top-k/drop/renormalize 2次項は素通し)。
//   2 = 真LUT (推論用): オンラインQLUT＋bit-plane参照。decode GEMV禁止
//       (GEMM体制のみ。M==1ではfp32にfallback)。
// 量子化レベル (T-MAC bit-serial準拠): bits-bit値は奇対称
//   value(li) = (2*li-(2^bits-1))/(2^bits-1) * max_abs, li∈[0,2^bits).
//   INT2: {-1,-1/3,+1/3,+1}×max。INT4: {-1,..,-1/15,..,+1}×max。
// スケール: per-column max_abs (単一w_scale縮退より精度優先。選択を記録)。
// 規約: C11, restrict, errno+goto cleanup。fail-closed。
// リトルエンディアン前提。SIMDなしスカラー核 (集計はtmac核に委譲)。

#include <stddef.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// モード設定/取得 (0/1/2以外はINVAL)。
int jt_lowbit_set_mode(int mode);
int jt_lowbit_get_mode(void);

// 層種→ビット幅 (gate/up=2, down=4。それ以外はINVAL)。
int jt_lowbit_bits_for(int is_down, int *restrict out_bits);

// fake-quant: w[0..n)を奇対称レベルに量子化→逆量子化してfqへ。
// スケールは max_abs (per-array)。非有限はINVAL。
int jt_lowbit_fakequant(const float *restrict w, float *restrict fq,
                        size_t n, int bits);

// 重み列のbit-plane化: col[K] (1出力列) → idx[bits*ngroups] + w_scale。
// ngroups=K/4。K%4!=0はINVAL。非有限はINVAL。
int jt_lowbit_col_to_idx(const float *restrict col, size_t k, int bits,
                         uint8_t *restrict idx, float *restrict w_scale);

// 真LUT GEMM (1 expert分): Y[Me][H] = X[Me][K] · W[K][H] (Wはrow-major [K][H])。
// bitsは全列共通。K%32!=0のときはfp32 GEMMにfallbackする (QLUT ctor制約)。
// M==0は何もせずJT_OK。非有限入力はINVAL (Y不変)。
// 符号注記: LUT核は符号なしレベル和 Σa·li を返すため、gemm内で
//   真値 (2·o-(n-1)·ws·A)/(n-1) (Aは行和) に戻す。これによりfake-quantの
//   奇対称レベル (2li-(n-1))/(n-1) と一致する (構成上の等価性)。
int jt_lowbit_gemm_lut(const float *restrict X, const float *restrict W,
                       float *restrict Y, int me, int h, int k, int bits);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_LOWBIT_H
