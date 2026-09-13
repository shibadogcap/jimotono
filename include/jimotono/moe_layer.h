#pragma once
#ifndef JIMOTONO_MOE_LAYER_H
#define JIMOTONO_MOE_LAYER_H
// MoE層: gate線形→softmax→実top-k→選択expert SwiGLU→重み付き和＋共有expert。
// P2最終レビューMAJOR-3対応 (e2eの「gate勾配なし・実top-k不使用」を解消する
// 足場)。jt_routing_topk と jt_swiglu_fwd/bwd を再利用する薄い結合層。
// 規約: C11, restrict積極使用, errnoベース + goto cleanup (AGENTS.MD 7.1)。
// クロスプラットフォーム (SIMD分岐なしのスカラー参照核)。
//
// [共有expert]
// 共有expertは常駐のため損失対象外 (routing.hと同一方針)。本層では順伝播で
// 重み1の加算 (Y += sum_s Y_s) のみ行う。n_shared=0で無効化可。
// 共有expert数は0〜JT_MOE_MAX_SHARED (既定2を想定)。
//
// [gate勾配 (STE的直通)]
// top-k選択自体は非微分 (離散選択) のため、 backwardでは選択を固定した
// Straight-Throughとして扱う:
//   選択外expert: dLogit=0, dWgate行=0, dW行=0
//   選択内: softmaxヤコビアン J=diag(w)-w w^T 経由
//     dL/dlogit_e = w_e * (dL/dw_e - sum_q w_q*dL/dw_q)
//     dL/dw_p = dot(Y_{id_p}, dY)
// L_CE・μL_balとの合算は呼び出し側で行うこと (routing.h準拠、TODO)。
//
// [系列集約]
// jt_moe_sticky_seq_loss() は jt_routing_sticky_loss() の系列平均ヘルパー。
// 層平均・L_CE合算は呼び出し側TODO (routing.hコメント準拠)。
//
// [restrict / 別名禁止]
// 全ポインタは互いに重ならないこと。gate 3点の制約と同様、XとdY等も
// 同一バッファを使い回さないこと。同一値の連続でも別バッファを渡すこと。
//
// [fail-closed]
// 検証失敗時は主要出力 (fwdのY / bwdのdX・dW群・dLogits / seqのout_loss) を
// 更新せず JT_ERR_INVAL (errno=EINVAL) を返す。fwdのcache
// (out_ids/out_weights/out_logits/Gsel/Usel/Ysel/Gs/Us) はスクラッチ扱いで
// エラー時に部分更新される場合がある (Yの不変のみ保証)。

#include <stddef.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// 次元上限 (train_bwd.hのJT_BWD_MAX_WIDEと整合)。
#define JT_MOE_MAX_EXPERTS 4096
#define JT_MOE_MAX_TOPK 64
#define JT_MOE_MAX_SHARED 8

// MoE順伝播 (1トークン分)。
//   X: [n] 入力。
//   Wgate: [E][n] row-major。logit[e]=sum_j X[j]*Wgate[e][j] (double累積)。
//   Wg/Wu/Wd: routed expert重み。各[E][h][n] row-major ([h][n]が行単位)。
//   Wg_s/Wu_s/Wd_s: 共有expert重み。各[S][h][n]。S==0時はNULL可 (無視)。
//   Y: [n] 出力。Y = sum_{p<k} w_p*Y_{id_p} + sum_{s<S} Y_s。
//   n: model dim (1..JT_BWD_MAX_WIDE)。h: expert hidden (同)。
//   n_experts (E): routed数 (1..MAX)。topk (k): 1..E かつ <=MAX_TOPK。
//   n_shared (S): 0..MAX_SHARED。
//   out_ids [k], out_weights [k]: jt_routing_topkの結果 (NULL不可)。
//   out_logits [E] or NULL: gate生logit (デバッグ用、NULLで省略)。
//   cache_Gsel/Usel [k*h] or NULL, cache_Ysel [k*n] or NULL:
//     bwd用中間値。bwdを使う場合は3点とも非NULLで渡すこと。
//     NULL時は順伝播のみ (内部tempで代替)。
//   cache_Gs/Us [S*h] or NULL: 共有expert用中間値 (S>0でbwdを使う場合は必須)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=EINVAL: NULL, 次元不正, 非有限入力,
//   routing失敗 (+INF等), expert forward失敗)。
int jt_moe_fwd(const float *restrict X,
               const float *restrict Wgate,
               const float *restrict Wg, const float *restrict Wu,
               const float *restrict Wd,
               const float *restrict Wg_s, const float *restrict Wu_s,
               const float *restrict Wd_s,
               float *restrict Y,
               int n, int h, int n_experts, int topk, int n_shared,
               size_t *restrict out_ids, float *restrict out_weights,
               float *restrict out_logits,
               float *restrict cache_Gsel, float *restrict cache_Usel,
               float *restrict cache_Ysel,
               float *restrict cache_Gs, float *restrict cache_Us);

// jt_moe_fwd の重み有限スキャンを省略する内部高速経路。
// 通常APIと同一の計算核 (bit一致)。省略するのは重みバッファの
// O(E*H*N) 有限プリスキャンと内側 jt_swiglu_fwd の重みスキャンのみで、
// NULL/次元検査・logits/重み範囲検査 (O(k))・計算途中のisfiniteガード
// (gate acc・yacc等、O(出力)で安価)・内側 routing_topk のNaN検査は残る。
// [unchecked使用条件] 呼び出し側が当該区間で以下を保証する場合のみ:
//   (1) 全重みバッファを区間冒頭で有限検証済みであること、
//   (2) 当該区間で重みバッファが不変であること (更新は区間外)。
// 活性化由来の非有限は計算途中ガード・loss合算点・更新前ガードで検出する
// (train_longrunではステップ冒頭のP全走査＋loss/gradガードが該当)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用: NULL・次元不正・途中非有限)。
int jt_moe_fwd_unchecked(const float *restrict X,
                         const float *restrict Wgate,
                         const float *restrict Wg, const float *restrict Wu,
                         const float *restrict Wd,
                         const float *restrict Wg_s, const float *restrict Wu_s,
                         const float *restrict Wd_s,
                         float *restrict Y,
                         int n, int h, int n_experts, int topk, int n_shared,
                         size_t *restrict out_ids, float *restrict out_weights,
                         float *restrict out_logits,
                         float *restrict cache_Gsel, float *restrict cache_Usel,
                         float *restrict cache_Ysel,
                         float *restrict cache_Gs, float *restrict cache_Us);

// MoE逆伝播 (1トークン分)。
//   dY: [n] 上流勾配 (定数扱い)。X/W群はfwdと同一値。
//   ids [k], weights [k]: fwdのout_ids/out_weightsと同一値。
//   Gsel/Usel [k*h], Ysel [k*n]: fwdのcacheと同一値 (NULL不可)。
//   Gs/Us [S*h]: fwdの共有cache (S==0時はNULL可、S>0時はNULL不可)。
//   dX [n]: 入力勾配 = gate経路 + 選択expert経路 + 共有経路の和。
//   dWgate [E][n]: gate線形重み勾配。非選択行は0。
//   dWg/dWu/dWd [E][h][n]: routed重み勾配 (上書き、加算ではない)。
//     非選択expertの全行は0。
//   dWg_s/dWu_s/dWd_s [S][h][n]: 共有重み勾配。S==0時はNULL可 (無視)。
//   dLogits [E] or NULL: gate logit勾配 (選択外0、選択内softmaxヤコビアン)。
//     NULLで省略可 (dWgate/dXの計算は継続)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用: NULL, 次元不正, 非有限入力,
//   ids範囲外・重複, weights不正)。
int jt_moe_bwd(const float *restrict dY, const float *restrict X,
               const float *restrict Wgate,
               const float *restrict Wg, const float *restrict Wu,
               const float *restrict Wd,
               const float *restrict Wg_s, const float *restrict Wu_s,
               const float *restrict Wd_s,
               const size_t *restrict ids, const float *restrict weights,
               const float *restrict Gsel, const float *restrict Usel,
               const float *restrict Ysel,
               const float *restrict Gs, const float *restrict Us,
               float *restrict dX,
               float *restrict dWgate,
               float *restrict dWg, float *restrict dWu,
               float *restrict dWd,
               float *restrict dWg_s, float *restrict dWu_s,
               float *restrict dWd_s,
               float *restrict dLogits,
               int n, int h, int n_experts, int topk, int n_shared);

// jt_moe_bwd の重み・キャッシュ有限スキャンを省略する内部高速経路。
// 通常APIと同一の計算核 (bit一致)。省略するのは重み・キャッシュ・活性化
// バッファの O(E*H*N) 有限プリスキャンと内側 jt_swiglu_bwd の重みスキャン
// のみで、NULL/次元検査・計算途中のisfiniteガード (dw_dp・dlog・dx_acc等、
// O(出力)で安価) は残る。ids/weightsの範囲・重複・総和検査 (O(k^2)) も
// 省略するため、ids/weightsは同一ステップ内fwd産の値をそのまま渡すこと。
// [unchecked使用条件] jt_moe_fwd_uncheckedに同じ (事前検証済み＋区間内不変)。
// 活性化由来の非有限は計算途中ガード・loss合算点・更新前ガードで検出する。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用: NULL・次元不正・途中非有限)。
int jt_moe_bwd_unchecked(const float *restrict dY, const float *restrict X,
                         const float *restrict Wgate,
                         const float *restrict Wg, const float *restrict Wu,
                         const float *restrict Wd,
                         const float *restrict Wg_s, const float *restrict Wu_s,
                         const float *restrict Wd_s,
                         const size_t *restrict ids,
                         const float *restrict weights,
                         const float *restrict Gsel, const float *restrict Usel,
                         const float *restrict Ysel,
                         const float *restrict Gs, const float *restrict Us,
                         float *restrict dX,
                         float *restrict dWgate,
                         float *restrict dWg, float *restrict dWu,
                         float *restrict dWd,
                         float *restrict dWg_s, float *restrict dWu_s,
                         float *restrict dWd_s,
                         float *restrict dLogits,
                         int n, int h, int n_experts, int topk, int n_shared);

// StickyMoE損失の系列合算ヘルパー。
//   gates: [T][n] row-majorのgate分布 (routedのみ)。
//   T: 系列長 (>0)。n: expert数 (>0)。w (W): 窓幅 (>0)。
//   lambda/alpha: soft/hard重み (>=0・有限)。
//   窓アンカー s(t)=(t/w)*w, t_rel=t-s(t) とし、各tで
//     jt_routing_sticky_loss(g_t, t>0?g_{t-1}:NULL, g_{s(t)}, ...)
//   を呼び double累積し T で平均化して返す (T==1時は先頭のみで0)。
//   L_CE・μL_bal・層平均との合算は呼び出し側TODO (routing.h準拠)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。
int jt_moe_sticky_seq_loss(const float *restrict gates, size_t T, size_t n,
                           float lambda, float alpha, size_t w,
                           float *restrict out_loss);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_MOE_LAYER_H
