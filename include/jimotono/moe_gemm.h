#pragma once
#ifndef JIMOTONO_MOE_GEMM_H
#define JIMOTONO_MOE_GEMM_H
// Phase G Step 4a: AVX2マイクロカーネル（Mr12×Nr4・acc12・f32蓄積＋f64加算・テール）。
// C11・restrict・errnoベース＋goto cleanup（AGENTS.MD 7.1）。AVX-512不使用。
// スコープ：MoEバッチGEMMのみ。KV・decode・量子化・G1融合に触れない。
//
// [タイル定義]
//   JT_GEMM_MR=12（M方向ブロック。M_e mod 12のテール単位）。
//   JT_GEMM_NR=4（N方向論理ブロック。スカラー換算4。Nr mod 4のテール単位）。
//   AVX2 8-wideではN方向に2ブロック融合（8 f32）で処理し、剰余（N%8・N%4）は
//   スカラーで吸収する。§4.1の「Mr=12行×Nr=4トークン」と本ヘッダのMR/M方向は
//   命名が転置しているが、48出力（12×4）のマイクロタイル寸法は同一である。
//   f32側48/8=6 regs、f64側48/4=12 regs（acc12本はf64側を指す）。
//   内側マイクロ（M≤12 × N=8）ではf32 acc 12本（ymm0–ymm11）＋B用1本＋A broadcast用
//   2本 = 15本 ≤ 16でスピルなし（roofline D6準拠）。
//
// [精度]
//   内側K縮約はf32蓄積（mul/add分離・FMA不使用でbit同一）。
//   最終のY/dX合算は呼出し側の既存f64加算（jt_moe_avx2_f64_add相当の要素wise、
//   bit同一）を用いる。すなわち「f32蓄積＋f64加算」はカーネル＋呼出し側の系として
//   成立する。FMAは使用しない（bit同一要請のため）。
//
// [順序・bit同一]
//   K（縮約）方向はk=0..K-1逐次で固定。N/M方向のベクトル化は出力独立のため
//   K順序を変えず、スカラーf32三重ループ（同一k順）とbit一致する。
//   テール（M%12・N%8・N%4）も同一k順のスカラーフォールバックでbit一致する。
//   M_e=0は起動スキップ（C不変）。M_e<12はテール経路のみで処理する。
//
// [fail-closed]
//   公開MoE API（Y/dX/dW不変）の内側スクラッチ核である。dims/NULL/overflow検査を
//   書込み前に完了し、検査失敗時はCに触れずJT_ERR_INVALを返す。
//   計算途中の非有限はJT_ERR_INVALで返す（この場合Cスクラッチは部分更新され得るが、
//   呼出し側が破棄し公開出力を更新しない。公開fail-closedは呼出し側で担保）。
//   有限プリスキャン（O(MK+KN)）は呼出し側のvalidateに委ね、核では行わない
//   （unchecked使用条件と同一）。
//
// [共有expert・attention]
//   共有expert（M=T=512）・attention/head相当の密GEMM（M=T=512）も同一の
//   jt_gemm_mat_f32に乗せる。呼出し側（moe_layer.c）が同一APIを使う。
//   本ヘッダはレイアウトを問わず C[M][N]=A[M][K]·B[K][N]（row-major）のみ扱う。
//   gate/upのB[N][K] row-major（ドット形式）は呼出し側で転置（jt_transpose_f32、
//   exact）してから本核に渡す。転置はexactのためbit同一に影響しない。

#include <stddef.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define JT_GEMM_MR 12
#define JT_GEMM_NR 4

// C[M][N] = A[M][K] · B[K][N]（row-major、lda=K・ldb=N・ldc=N）。
// A[M*K]・B[K*N]・C[M*N]は重なり禁止。M==0はC不変でJT_OK（起動スキップ）。
// 戻り値：JT_OK / JT_ERR_INVAL（errno併用）。
int jt_gemm_mat_f32(const float *restrict A, const float *restrict B,
                    float *restrict C, int M, int N, int K);

// dst[cols][rows] = src[rows][cols]（row-major、exact）。
// rows==0||cols==0はJT_ERR_INVAL（M_e=0の転置は呼出し側でスキップすること）。
// 戻り値：JT_OK / JT_ERR_INVAL（errno併用）。
int jt_transpose_f32(const float *restrict src, float *restrict dst, int rows,
                     int cols);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_MOE_GEMM_H
