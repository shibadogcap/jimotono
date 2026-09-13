#pragma once
#ifndef JIMOTONO_MIXED_PREC_H
#define JIMOTONO_MIXED_PREC_H
// JIMOTONO 混合精度ポリシー + per-block量子化/逆量子化 (Phase 2足場)。
// 準拠: AGENTS.MD §2.2 (shared=INT8 / routing down=INT4 / gate,up=INT2 /
//   attn=INT8 / emb,head=INT8 tied)、§3.3 HGQ-LUT (学習時は通常テンソル演算)。
//
// FRI-MxMoE方針 (https://aclanthology.org/2026.acl-long.982/):
//   エキスパート内サブレイヤーでロバスト性が異なるため、層種別にビット幅を
//   割り当てる。FRIはプロファイリング不要で感度を推定し最大15.7倍高速化する
//   ため、本モジュールも「安価なダミー統計→ビット割当提案」の純粋関数として
//   感度APIを持つ (jt_mp_sensitivity_score / jt_mp_propose_bits)。
//   実プロファイリング値は将来差し替え、現状は分布仮定のスタブ。
//
// 量子化方式 (素朴・対称per-block):
//   block毎に max_abs = max|x|、scale = max_abs/qmax、
//   q = clamp(round(x/scale), qmin, qmax)。
//   qmax = 2^(bits-1)-1 (8→127, 4→7, 2→1)、qmin = -(qmax+1)。
//   ゼロブロックは scale=1.0, q=0。除算回避。
//   対応bitsは {2,4,8} のみ (policy表の値域)。他はINVAL。
//
// 規約: C11, restrict, errnoベース + goto cleanup (実装側)。
// fail-closed: 非有限入力・不正引数は JT_ERR_INVAL (errno=EINVAL) を返し、
//   正常系の出力値は信用しないこと (呼び出し側は破棄する)。
// リトルエンディアン前提 (common.h)。SIMDなしスカラー核。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// 層種別 (AGENTS.MD §2.2の行に対応)。
typedef enum jt_mp_layer {
    JT_MP_SHARED = 0,       // 共有エキスパート (常時発火) → INT8
    JT_MP_ROUTING_DOWN = 1, // routing down_proj → INT4
    JT_MP_ROUTING_GATE = 2, // routing gate_proj → INT2
    JT_MP_ROUTING_UP = 3,   // routing up_proj → INT2
    JT_MP_ATTN = 4,         // attention射影 (毎トークン発火) → INT8
    JT_MP_EMB = 5,          // embedding / 出力head (tied) → INT8
    JT_MP_LAYER_COUNT = 6
} jt_mp_layer_t;

// 既定ブロック長 (T-MAC act_group=32と整合。任意blockも可)。
#define JT_MP_DEFAULT_BLOCK 32

// 層→ビット幅ポリシー表 (純粋関数)。
// 戻り値: JT_OK / JT_ERR_INVAL (不正layer, NULL)。
int jt_mp_bits_for(jt_mp_layer_t layer, int *restrict out_bits);

// ビット幅→対称量子化の正側最大値 (2→1, 4→7, 8→127)。
// 戻り値: JT_OK / JT_ERR_INVAL (bitsが{2,4,8}外, NULL)。
int jt_mp_qmax_for(int bits, int *restrict out_qmax);

// 必要ブロック数 ((n+block-1)/block)。block==0/n==0はINVAL。
int jt_mp_nblocks(size_t n, size_t block, size_t *restrict out);

// per-block量子化: src[0..n) → dst[0..n) + scales[0..nblocks)。
// nblocksは jt_mp_nblocks(n, block) と一致すること。
// 丸めはround (half-away)、範囲外はクリップ、非有限srcはINVAL。
// ゼロブロックは scale=1.0, q=0。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=EINVAL) / JT_ERR_NOMEM (積オーバーフロー)。
int jt_mp_quantize(const float *restrict src, size_t n, int bits,
                   size_t block, int8_t *restrict dst,
                   float *restrict scales, size_t nblocks);

// 逆量子化: dst[i] = (float)src[i] * scales[i/block]。
// 非有限scaleはINVAL。q値域外 (qmin..qmax外) もINVAL (fail-closed)。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM。
int jt_mp_dequantize(const int8_t *restrict src, const float *restrict scales,
                     size_t n, int bits, size_t block, size_t nblocks,
                     float *restrict dst);

// ---- FRI-MxMoE風 感度スコアAPI (純粋関数・プロファイリング不要) ----

// 安価なダミー統計 (将来のFRI推定量の差し替え位置)。
// 全て有限・非負を要求 (負値はINVAL)。
typedef struct jt_mp_sensitivity {
    float grad_var;    // 勾配分散の代理 (>=0)
    float act_range;   // 活性レンジの代理 (>=0)
    float hess_trace;  // Hessian trace代理 (>=0)
} jt_mp_sensitivity_t;

// 感度スコア化: score = log1p(gv)+log1p(ar)+log1p(ht) (double内部計算)。
// スコアが高いほど量子化に敏感 (=高ビット推奨)。
// 戻り値: JT_OK / JT_ERR_INVAL (NULL, 非有限/負値)。
int jt_mp_sensitivity_score(const jt_mp_sensitivity_t *restrict st,
                            float *restrict out_score);

// ビット割当提案: policy基準値からスコアで一段階だけ補正するスタブ。
//   score > 2.0 → 1段上げ (2→4→8)、score < 0.5 → 1段下げ (8→4→2)。
// 段数は {2,4,8} のみ。FRI本実装までの予約位置 (しきい値は暫定)。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_mp_propose_bits(jt_mp_layer_t base_layer,
                       const jt_mp_sensitivity_t *restrict st,
                       int *restrict out_bits);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_MIXED_PREC_H
