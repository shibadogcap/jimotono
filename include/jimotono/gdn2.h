#pragma once
#ifndef JIMOTONO_GDN2_H
#define JIMOTONO_GDN2_H
// Gated DeltaNet-2 デコード逐次核 + プリフィル chunk スタブ (Phase 1)。
// knowledge/papers.md §2:
//
//   S_t = (I - k_t (b_t⊙k_t)^T) D_t S_{t-1} + k_t (w_t⊙v_t)^T
//   o_t = S_t^T q_t
//
// 設計 (AGENTS.MD 7.1, papers.md §2 申し送り):
// - state S (dk×dv, row-major, 先頭次元=dk) は fp32 維持。量子化しない。
// - q/k は核内で L2 正規化 (eps ガード)。呼び出し側の事前正規化は不要。
// - D_t=Diag(alpha): log-decay の累積・exp は fp32 で核外計算し、
//   本核には alpha (呼び出し側で [0,1] 保証) を渡すこと。
// - b_t (erase, key側) は精度優先で常に fp32 ベクトル維持。
//   削減は w_t (write, value側) から行う (スカラー化・量子化の候補)。
// - S は 64B 整列必須。不正時は JT_ERR_ALIGN。
// - レイアウトは密・連続 (ld==dv 固定)。可変長は cuSeqlens 境界で
//   jt_gdn2_state_reset() し、P1 ではチャンク核を使わない。
// - リトルエンディアン前提 (common.h が BE で #error)。
// - SIMD: 下部の JT_GDN2_SIMD_WIDTH ifdef ガード参照。
//   P1 はスカラー核。P2 で AVX-512/AVX2/NEON パスを追加する。

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
//   alpha: [dk] (核外で fp32 累積済みの decay)。
//   scratch: jt_gdn2_scratch_floats() 個以上の作業域 (S/out_o と非重複)。
// 別名禁止: S・out_o・scratch・各入力は互いに重ねないこと (restrict)。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_ALIGN (errno 併用)。
int jt_gdn2_decode_step(float *restrict S, float *restrict out_o,
                        const float *restrict q, const float *restrict k,
                        const float *restrict v, const float *restrict b,
                        const float *restrict w, const float *restrict alpha,
                        int dk, int dv, float *restrict scratch,
                        size_t scratch_n);

// プリフィル用チャンク核 (P1 はスタブ: 宣言 + ENOSYS 返却のみ)。
// P2 で WY 型 (intra-chunk 64×64→C=16/32 縮小 solve + inter-chunk 漸化式、
// L1/L2 常駐優先) を実装予定。Q/K/B/Alpha: [C][dk]、V/W: [C][dv]、
// Out: [C][dv]、S: [dk][dv] 入出力。C は関数名に固定 (16 / 32)。
// 現状の戻り値: 正常入力には errno=ENOSYS で JT_ERR_INVAL、不正入力は
// errno=EINVAL で JT_ERR_INVAL。
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
