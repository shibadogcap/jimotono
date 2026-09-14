#pragma once
#ifndef JIMOTONO_W8A8_H
#define JIMOTONO_W8A8_H
// JIMOTONO Stage 3: W8A8 INT8 GEMM本線 (P3b §3準拠)。
// 位置づけ: decode GEMVを含む確実な帯域半減＋整数SIMD加速の本線。
// T-MAC/INT4/INT2は非スコープ (Stage 3b) であり本モジュールで扱わない。
//
// [スコープ]
//   - 対称ゼロ点0 INT8 (q ∈ [-127,127]。-128は非対称のため不使用)。
//   - per-tensorスケール (行列全体で1 scale) と per-channelスケール
//     (row-major行単位=出力チャネル単位で1 scale/行) のみ。
//   - 量子化・逆量子化カーネル (スカラー)。
//   - int8 GEMM: C[M][N] = sA * sB[n] * Σ_k Aq[M][K]·Bq[K][N] (int32蓄積)。
//     B側は per-tensor (sB単一) または per-channel (列単位=N単位) に対応。
//     A側は per-tensor単一scale、または行単位scale (per-token) に対応。
//   - GEMM整数蓄積部のSIMDバックエンド (Stage 3b-1。同一関数内dispatch。
//     公開API・検証・tol不変。整数exactのため出力はbit同一):
//     AVX2 emul (#ifdef __AVX2__) / AVX512-VNNI (Ryzen用。__AVX512F__＋
//     __AVX512VNNI__ガード。__AVX512__マクロは存在しない) /
//     AVX-VNNI (N100用。__AVXVNNI__＋-mavxvnni。u8*s8＋0x80補正でexact) /
//     AVX-VNNI-INT8 (将来CPU用。__AVXVNNIINT8__＋-mavxvnniint8。s8*s8直接。
//     N100では#UDするため使わない) /
//     非対応CPUは同一k順スカラー。FMA不使用。
//
// [fp32並存・フラグ]
//   - 既存 jt_gemm_mat_f32 (moe_gemm.c) には一切手を入れない (既定経路)。
//   - 本モジュールの static フラグ (既定OFF) で切替える。
//     OFF時は呼出し側が従来通り jt_gemm_mat_f32 を呼ぶ (速度不変)。
//     ON時のみ本モジュールの W8A8 経路を使う。既定はfp32のまま。
//   - 学習時の --w8a8 は fake-quant (quant→dequant後にfp32 GEMM) として
//     既存核を再利用する。推論時の真のint8 GEMMは jt_w8a8_gemm_* を使う。
//
// [fail-closed]
//   - 全公開APIは書込み前に検証を完了し、失敗時は出力を更新せず
//     JT_ERR_INVAL (errno=EINVAL) を返す。非有限入力は拒否する。
//   - int32蓄積のオーバーフロー防止: K上限を JT_BWD_MAX_WIDE (4096) とし、
//     |acc| ≤ 127*127*4096 ≈ 6.6e7 < 2^31 のため安全。
//
// [規約]
//   - C11・restrict・errnoベース＋goto cleanup (AGENTS.MD 7.1)。
//   - リトルエンディアン前提 (common.h)。クロスプラット (SIMDはifdef分岐)。
//   - AVX-512はVNNIドット経路 (Ryzen用) に限定し、一般GEMMには使わない。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// INT8対称量子化の正側最大値 (127)。負側最小は -127 (-128不使用)。
#define JT_W8A8_QMAX 127
#define JT_W8A8_QMIN (-127)

// ---- モードフラグ (既定OFF=fp32。プロセス内グローバル) ----

// 0=fp32既定経路 (OFF)。1=W8A8 INT8経路 (ON)。
// 戻り値: JT_OK / JT_ERR_INVAL (NULL)。
int jt_w8a8_set_enabled(int enabled);
int jt_w8a8_get_enabled(int *restrict out_enabled);
// Cでの分岐用インライン判定 (0/1)。NULL不可の軽量版ではないため
// ホットパスでは jt_w8a8_get_enabled ではなく戻り値キャッシュを使うこと。
int jt_w8a8_is_enabled(void);

// ---- per-tensor量子化 (行列全体で単一scale) ----
// scale = max|x|/127。ゼロ行列時は scale=1.0, q=0。
// 丸めはround (half-away)、範囲外はクリップ、非有限srcはINVAL。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=EINVAL)。
int jt_w8a8_quant_per_tensor(const float *restrict src, size_t n,
                             int8_t *restrict dst,
                             float *restrict out_scale);
// 逆量子化: dst[i] = (float)src[i] * scale。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_w8a8_dequant_per_tensor(const int8_t *restrict src, float scale,
                               size_t n, float *restrict dst);

// ---- per-channel量子化 (row-major行単位。1行=1チャネル=1 scale) ----
// src[rows][cols] → dst[rows][cols] + scales[rows]。
// 各行は per-tensor と同一式 (行内max/127)。ゼロ行は scale=1.0, q=0。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_w8a8_quant_per_channel(const float *restrict src, size_t rows,
                              size_t cols, int8_t *restrict dst,
                              float *restrict scales);
// 逆量子化: dst[r][c] = (float)src[r][c] * scales[r]。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_w8a8_dequant_per_channel(const int8_t *restrict src,
                                const float *restrict scales, size_t rows,
                                size_t cols, float *restrict dst);

// ---- int8 GEMM (真のW8A8。推論本線) ----
// C[M][N] = sA * sB[n] * Σ_k Aq[m][k]*Bq[k][n] (int32蓄積→float)。
//   Aq: [M][K] int8 row-major。Bq: [K][N] int8 row-major。C: [M][N] float。
//   sA: A側scale (per-tensor単一)。sB_col: B側scale (per-tensor時は全N同一値
//     の配列、またはNULLで単一sB使用。下記2関数の使い分け)。
// M==0は起動スキップ (C不変でJT_OK。jt_gemm_mat_f32と同一契約)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。
//
// per-tensor版: 単一sB (B全体で1 scale)。
int jt_w8a8_gemm_per_tensor(const int8_t *restrict Aq,
                            const int8_t *restrict Bq, float sA, float sB,
                            float *restrict C, int M, int N, int K);
// per-channel版: B側は列単位scale (sB_col[N]。N列=出力チャネル単位)。
//   (量子化時にBを転置 [N][K] row-majorで行量子化した想定。GEMM時は
//   Bq[K][N]配置で読むため、列nのscaleは sB_col[n])。
// sB_col==NULL時はINVAL (per-tensor版を使うこと)。
int jt_w8a8_gemm_per_channel(const int8_t *restrict Aq,
                             const int8_t *restrict Bq, float sA,
                             const float *restrict sB_col,
                             float *restrict C, int M, int N, int K);

// ---- fake-quantヘルパー (学習時の精度評価用。推論本線ではない) ----
// out[i] = dequant(quant(in[i]))。per-tensor版と行単位版。
// 既存fp32 GEMM核の前段に挟み、量子化ノイズのみを付加する。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_w8a8_fakequant_per_tensor(const float *restrict src, size_t n,
                                 float *restrict dst);
int jt_w8a8_fakequant_per_channel(const float *restrict src, size_t rows,
                                  size_t cols, float *restrict dst);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_W8A8_H
