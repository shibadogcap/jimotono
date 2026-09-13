#pragma once
#ifndef JIMOTONO_GDN2_H
#define JIMOTONO_GDN2_H
// Gated DeltaNet-2 デコード逐次核 + プリフィル chunk 核 (WY 型、C=16/32)。
// knowledge/papers.md §2:
//
//   S_t = (I - k_t (b_t⊙k_t)^T) D_t S_{t-1} + k_t (w_t⊙v_t)^T
//   o_t = S_t^T q_t
//
// 設計 (AGENTS.MD 7.1, papers.md §2 申し送り):
// - state S (dk×dv, row-major, 先頭次元=dk) は fp32 維持。量子化しない。
// - q/k は核内で L2 正規化 (eps ガード)。呼び出し側の事前正規化は不要。
// - D_t=Diag(alpha): log-decay の累積・exp は fp32 で核外計算し、
//   本核には alpha ([0,1]。範囲外・非有限は核内で拒否) を渡すこと。
// - b_t (erase, key側) は精度優先で常に fp32 ベクトル維持。
//   削減は w_t (write, value側) から行う (スカラー化・量子化の候補)。
// - S は 64B 整列必須。不正時は JT_ERR_ALIGN。
// - レイアウトは密・連続 (ld==dv 固定)。可変長は cuSeqlens 境界で
//   jt_gdn2_state_reset() し、チャンクを跨がないこと。
// - リトルエンディアン前提 (common.h が BE で #error)。
// - SIMD: 下部の JT_GDN2_SIMD_WIDTH ifdef ガード参照。
//   現状スカラー核。AVX-512/AVX2/NEON パスは別タスクで追加予定。

#include <stddef.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// 次元上限 (論文既定 dk=dv=128。検査・scratch 見積り用)。
#define JT_GDN2_MAX_D 1024
// L2 正規化のゼロ割ガード。
#define JT_GDN2_EPS_DEFAULT 1e-6f

// SIMD 幅ヒント (P2 ベクトル化用。P1 スカラー核は値を使わないが、
// dv 方向の分割設計はこの幅に合わせる)。
#if defined(__AVX512F__)
#define JT_GDN2_SIMD_WIDTH 16
#elif defined(__AVX2__)
#define JT_GDN2_SIMD_WIDTH 8
#elif defined(__ARM_NEON)
#define JT_GDN2_SIMD_WIDTH 4
#else
#define JT_GDN2_SIMD_WIDTH 1
#endif

// S (dk×dv float) のバイト数。戻り値: JT_OK / JT_ERR_INVAL。
int jt_gdn2_state_bytes(int dk, int dv, size_t *restrict out_bytes);

// 64B 整列・ゼロ初期化で S を確保。解放は jt_gdn2_state_free()。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM (errno 併用)。
int jt_gdn2_state_alloc(float **restrict out_S, int dk, int dv);

// 解放 (NULL 安全)。
void jt_gdn2_state_free(float *S);

// S をゼロ埋め (cuSeqlens 境界での state リセット用)。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_gdn2_state_reset(float *restrict S, int dk, int dv);

// jt_gdn2_decode_step() に渡す scratch の float 要素数 (2*dk + 2*dv)。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_gdn2_scratch_floats(int dk, int dv, size_t *restrict out_n);

// L2 正規化 y = x / ||x||。||x|| < eps なら y=0 (NaN を出さない)。
// eps <= 0 / NaN は JT_ERR_INVAL。戻り値: JT_OK / JT_ERR_INVAL。
int jt_gdn2_l2norm(const float *restrict x, float *restrict y, int n, float eps);

// デコード逐次核 (1トークン更新)。完全実装。
//   S: [dk][dv] 入出力 (64B 整列必須)。out_o: [dv] 出力。
//   q/k: [dk] (核内で正規化)。v/w: [dv]。b: [dk] (erase, fp32維持)。
//   alpha: [dk] (核外で fp32 累積済みの decay。範囲 [0,1])。
//   scratch: jt_gdn2_scratch_floats() 個以上の作業域 (S/out_o と非重複)。
// 別名禁止: S・out_o・scratch・各入力は互いに重ねないこと (restrict)。
// fail-closed (tmac/routing と同一方針): q/k/v/b/w/alpha に非有限
// (NaN/Inf) がある場合、および alpha が [0,1] 外の場合は
// JT_ERR_INVAL (errno=EINVAL) を返し、S・out_o を更新しない (state不変)。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_ALIGN (errno 併用)。
int jt_gdn2_decode_step(float *restrict S, float *restrict out_o,
                        const float *restrict q, const float *restrict k,
                        const float *restrict v, const float *restrict b,
                        const float *restrict w, const float *restrict alpha,
                        int dk, int dv, float *restrict scratch,
                        size_t scratch_n);

// プリフィル用チャンク核 (P2 実装済み、WY 型、C 固定)。
// intra-chunk: C×C 下三角 solve ((I+L)^{-1}) + 密行列積、
// inter-chunk: チャンク間のみ漸化式 (papers.md §2、decay 吸収で純粋非対称 delta 化)。
// Q/K/B/Alpha: [C][dk]、V/W: [C][dv]、Out: [C][dv]、S: [dk][dv] 入出力。
// C は関数名に固定 (16 / 32)。C 回の jt_gdn2_decode_step() 逐次実行と等価
// (同一初期 S・同一順序入力に対し fp32 丸めを除き一致。bench は倍精度参照と tol=2e-5)。
// q/k は核内で L2 正規化 (eps ガード)。alpha は核外累積済み [0,1]。
// b (erase) は fp32 維持。S は fp32 維持・64B 整列必須。
// scratch: 4*C*dk + 2*C*dv + C*C + dk 個以上の作業域
// (内訳 Qn/Kn/E/P + U/G + L + tmp)。不足時は JT_ERR_INVAL。
// S・Out・各入力・scratch は互いに重ねないこと (restrict)。
// fail-closed (デコード核と同一方針): Q/K/V/B/W/Alpha に非有限がある場合、
// alpha が [0,1] 外の場合は JT_ERR_INVAL (errno=EINVAL) を返し S・Out 不変。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_ALIGN (errno 併用)。NOSUP は返さない。
int jt_gdn2_prefill_chunk16(float *restrict S, float *restrict Out,
                            const float *restrict Q, const float *restrict K,
                            const float *restrict V, const float *restrict B,
                            const float *restrict W,
                            const float *restrict Alpha, int dk, int dv,
                            float *restrict scratch, size_t scratch_n);
int jt_gdn2_prefill_chunk32(float *restrict S, float *restrict Out,
                            const float *restrict Q, const float *restrict K,
                            const float *restrict V, const float *restrict B,
                            const float *restrict W,
                            const float *restrict Alpha, int dk, int dv,
                            float *restrict scratch, size_t scratch_n);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_GDN2_H
