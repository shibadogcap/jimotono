#pragma once
#ifndef JIMOTONO_TRAIN_BWD_H
#define JIMOTONO_TRAIN_BWD_H
// JIMOTONO 学習エンジン: backward + checkpoint の逆伝播核 (Phase 2 足場)。
// 準拠: ARCHITECTURE.MD §1 (GDN-2), DESIGN.MD §4.1 (dX先行・dW後回し)。
// 規約: C11, restrict積極使用, errnoベース + goto cleanup (AGENTS.MD 7.1)。
// オプティマイザ本体は別wt担当のため本モジュールに含めない。
//
// [GDN-2 forward再掲 (gdn2.h一致)]
//   qn = q/||q||, kn = k/||k|| (epsガード、ゼロ時は0ベクトル)
//   S1 = D S_prev, D = Diag(alpha)
//   ke = b ⊙ kn, c^T = ke^T S1
//   S2 = S1 - kn c^T
//   vw = w ⊙ v
//   Sn = S2 + kn vw^T = (I - kn ke^T) D S_prev + kn (w⊙v)^T
//   o = Sn^T qn
// backwardは上式に整合し、dO (=dL/do, [dv]) と dS_next (=dL/dSn, [dk][dv])
// から各入力勾配を返す。損失スカラーはテスト側で
//   L = dot(o,dO) + sum(Sn*dS_next) (dO/dS_nextは定数扱い)
// と定義し、finite-diffと対照する。

#include <stddef.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// backward用次元上限 (gdn2.hのJT_GDN2_MAX_Dと整合)。
#define JT_BWD_MAX_D 1024
// L2ノルム逆伝播のゼロ割ガード (forwardのJT_GDN2_EPS_DEFAULTと同一値)。
#define JT_BWD_EPS_DEFAULT 1e-6f
// SwiGLU/RMSNorm用の次元上限 (隠れ層拡張を見込み広め)。
#define JT_BWD_MAX_WIDE 4096

// jt_gdn2_decode_bwd() 用scratchのfloat要素数 (2*dk + 2*dv + 2*dk*dv)。
// 内訳: qn[dk] + kn[dk] + c[dv] + vw[dv] + S1[dk*dv] + Sn[dk*dv]。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。
int jt_gdn2_decode_bwd_scratch_floats(int dk, int dv,
                                      size_t *restrict out_n);

// GDN-2デコード1ステップの逆伝播 (完全実装・スカラー核)。
//   S_prev: [dk][dv] forward入力 (64B整列必須、gdn2.hと同一)。
//   q/k/b/alpha: [dk], v/w: [dv] (forward入力と同一値)。
//   dO: [dv], dS_next: [dk][dv] (上流勾配、定数扱い)。
//   dQ/dK/dB/dAlpha: [dk], dV/dW: [dv], dS_prev: [dk][dv] (出力)。
//   scratch: 上記要素数以上の作業域 (入出力・入力のいずれとも非重複)。
// 別名禁止: 全ポインタは互いに重ねないこと (restrict)。
// fail-closed: forwardと同一方針で、q/k/v/b/w/alpha/dO/dS_nextに非有限が
// ある場合・alphaが[0,1]外の場合はJT_ERR_INVAL (errno=EINVAL) を返し、
// 出力7点+ dS_prevを更新しない。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_ALIGN (errno併用)。
int jt_gdn2_decode_bwd(const float *restrict S_prev,
                       const float *restrict q, const float *restrict k,
                       const float *restrict v, const float *restrict b,
                       const float *restrict w, const float *restrict alpha,
                       const float *restrict dO, const float *restrict dS_next,
                       float *restrict dQ, float *restrict dK,
                       float *restrict dV, float *restrict dB,
                       float *restrict dW, float *restrict dAlpha,
                       float *restrict dS_prev, int dk, int dv,
                       float *restrict scratch, size_t scratch_n);

// SwiGLU融合 (gate/up/downを1カーネル相当で処理、DESIGN.MD §4.1) の逆伝播。
// forward定義 (本関数の逆伝播が整合する式):
//   G[i] = sum_j X[j] Wg[i][j], U[i] = sum_j X[j] Wu[i][j]  (i<h, j<n)
//   s[i] = silu(G[i]) * U[i], silu(z) = z * sigmoid(z)
//   Y[j] = sum_i s[i] Wd[i][j]
//   Wg/Wu/Wdはrow-major ([h][n])、G/Uはforward時に保存された中間値。
// 計算順序 (DESIGN.MD §4.1): dXを先に計算して伝播させ、dWは後回し。
// dW計算→更新→書き戻しのオーバーラップを想定し、本関数内でもdX→dWの順に
// 記述する (呼び出し側がdX転送とdW計算をパイプライン化できるよう配慮)。
// dWg/dWu/dWdは上書き (加算ではない)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=EINVAL: NULL, n/h<=0・上限超過,
//   非有限入力)。
int jt_swiglu_bwd(const float *restrict dY, const float *restrict X,
                  const float *restrict G, const float *restrict U,
                  const float *restrict Wd, const float *restrict Wg,
                  const float *restrict Wu, float *restrict dX,
                  float *restrict dWg, float *restrict dWu,
                  float *restrict dWd, int n, int h);

// jt_swiglu_bwd の入力有限スキャンを省略する内部高速経路。
// 通常APIと同一の計算核 (bit一致)。省略するのは入力バッファの
// O(n)/O(h*n) 有限プリスキャンのみで、NULL/次元検査と計算途中の
// isfiniteガード (acc等、O(出力)で安価) は残る。
// [unchecked使用条件] 呼び出し側が当該区間で以下を保証する場合のみ:
//   (1) 重み・入力バッファを事前に有限検証済みであること、
//   (2) 当該区間で重みバッファが不変であること (更新は区間外)。
// 活性化由来の非有限は計算途中ガード・loss合算点・更新前ガードで検出する。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用: NULL・次元不正・途中非有限)。
int jt_swiglu_bwd_unchecked(const float *restrict dY,
                            const float *restrict X,
                            const float *restrict G, const float *restrict U,
                            const float *restrict Wd, const float *restrict Wg,
                            const float *restrict Wu, float *restrict dX,
                            float *restrict dWg, float *restrict dWu,
                            float *restrict dWd, int n, int h);

// RMSNormの逆伝播 (最小実装)。
// forward定義: r = 1/sqrt(mean(X^2)+eps), Y[i] = W[i]*X[i]*r。
//   dW[i] = dY[i]*X[i]*r
//   dX[j] = r*W[j]*dY[j] - r^3*X[j]/n * sum_i(dY[i]*W[i]*X[i])
// 内部はdouble累積。eps<=0・非有限はJT_ERR_INVAL。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。
int jt_rmsnorm_bwd(const float *restrict dY, const float *restrict X,
                   const float *restrict W, float *restrict dX,
                   float *restrict dW, int n, float eps);

// SwiGLU融合の順伝播 (jt_swiglu_bwdと整合するforward核)。
// 定義 (bwdコメントと同一):
//   G[i] = sum_j X[j] Wg[i][j], U[i] = sum_j X[j] Wu[i][j]  (i<h, j<n)
//   s[i] = silu(G[i]) * U[i], silu(z) = z * sigmoid(z)
//   Y[j] = sum_i s[i] Wd[i][j]
//   Wg/Wu/Wdはrow-major ([h][n])。
// G/Uはcheckpoint/逆伝播用の保存中間値 (bwdのG/U入力にそのまま渡せる)。
// 内部はdouble累積、sigmoidはdouble評価。fail-closed: 非有限入力・
// NULL・次元不正時はJT_ERR_INVAL (errno=EINVAL) を返し、G/U/Yを更新しない。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。
int jt_swiglu_fwd(const float *restrict X, const float *restrict Wg,
                  const float *restrict Wu, const float *restrict Wd,
                  float *restrict G, float *restrict U,
                  float *restrict Y, int n, int h);

// jt_swiglu_fwd の入力有限スキャンを省略する内部高速経路。
// 通常APIと同一の計算核 (bit一致)。省略するのは入力バッファの
// O(n)/O(h*n) 有限プリスキャンのみで、NULL/次元検査と計算途中の
// isfiniteガード (g/u/acc等、O(出力)で安価) は残る。
// [unchecked使用条件] jt_swiglu_bwd_uncheckedに同じ (事前検証済み＋区間内不変)。
// 活性化由来の非有限は計算途中ガード・loss合算点・更新前ガードで検出する。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用: NULL・次元不正・途中非有限)。
int jt_swiglu_fwd_unchecked(const float *restrict X,
                            const float *restrict Wg,
                            const float *restrict Wu, const float *restrict Wd,
                            float *restrict G, float *restrict U,
                            float *restrict Y, int n, int h);

// RMSNormの順伝播 (jt_rmsnorm_bwdと整合するforward核)。
// 定義: r = 1/sqrt(mean(X^2)+eps), Y[i] = W[i]*X[i]*r (内部double累積)。
// fail-closed: 非有限入力・NULL・次元不正・eps<=0/非有限時は
// JT_ERR_INVAL (errno=EINVAL) を返し、Yを更新しない。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。
int jt_rmsnorm_fwd(const float *restrict X, const float *restrict W,
                   float *restrict Y, int n, float eps);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_TRAIN_BWD_H
