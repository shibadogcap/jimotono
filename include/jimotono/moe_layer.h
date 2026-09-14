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

// [Phase G Step 1: ソート＋dispatch (計算は既存GEMVのまま。GEMM化なし)]
// jt_moe_batch_sort() はルーティング確定後の (token, slot) ペア列を expert 順へ
// 並べ替える counting sort (E が小さい前提。O(E + T*k))。安定ソートであり、
// 同一入力→同一 perm を保証する (単一スレッドで構築。タイブレークは
// (expert_id, token_pos, slot) の辞書式順序。重みの大小比較で順序を決めない)。
// capacity 上限 cap = ceil(cap_factor * T * k / E) を超えた分は drop
// (当該ペアの寄与を 0 とし、残り重みは renormalize する (G2改訂。旧「しない」を撤回)。
// トークン毎に kept 和Sで w'=w/S。共有 expert は
// 対象外で呼び出し側が別経路とする)。cap_factor は 1.0–1.5 の範囲でのみ受付。
//   ids [T*k]: token-major の expert 割当て ([t][k] row-major)。全要素 < E であること。
//   T: トークン数 (>0)。k: top-k (1..E)。E: routed 数 (1..MAX)。
//   cap_factor: 容量係数 (1.0–1.5 の有限値。既定 1.5=G2改訂)。
//   perm [T*k]: kept のみを expert 順に並べた flattened index (長さ = *out_kept)。
//     flattened index q = t*k+p (token-major)。NULL 不可。
//   off [E+1]: expert 境界 (off[E] == *out_kept)。NULL 不可。
//   drop [T*k]: flattened 順の drop マスク (0=kept, 1=dropped)。NULL 不可。
//   out_kept/out_dropped: kept/drop 総数 (NULL で省略可)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=EINVAL: NULL, 次元不正, ids範囲外,
//   cap_factor範囲外)。検証失敗時は perm/off/drop/out_* を更新しない。
// 備考: AVX-512 不使用。スカラーのみ (ソートはメモリ律速のため)。
int jt_moe_batch_sort(const size_t *restrict ids, int T, int k, int E,
                      float cap_factor,
                      size_t *restrict perm, size_t *restrict off,
                      unsigned char *restrict drop,
                      size_t *restrict out_kept, size_t *restrict out_dropped);

// MoE順伝播のバッチ版 (Tトークン分。Step 2: expert単位バッチfwd・素朴GEMM)。
// gate logits→実top-k→softmax まではトークン毎に単体版と同一核で計算し、
// その後 jt_moe_batch_sort() で perm/off/drop を作る。
// Step 2 では expert 連続バッファ Xe [M_e][n] に gather し、M 方向に既存
// jt_swiglu_fwd 核を拡張して expert-outer/M-inner 順 (§2.1) で計算する
// (素朴 GEMM 参照実装。ブロッキング・SIMD 新規最適化なし・AVX-512 不使用。
// 本格マイクロカーネルは Step 4)。各行の計算核と結合の token 順は Step 1 と
// 同一のため、drop なし時は単体版 jt_moe_fwd のトークンループと bit 一致する
// (同一順序の token 順 combine のため。AVX2 有効時も同一ヘルパー使用)。
// drop あり時は dropped ペアの寄与を 0 とし kept を renormalize する (G2改訂)。
// 共有 expert は常時オン・容量制限対象外でトークン毎に単体版と同一に加算する。
// bwd は単体版 jt_moe_bwd をそのまま使う (同一 ids/weights/cache 形式)。
// 注意 (Step 3 への申送り): drop あり時の bwd 側 drop マスク適用
// (dropped ペアの gate 勾配 0 化) は Step 3 の範囲。本関数は dropped 対応
// cache スロットを 0 埋めするに留める (不定値の混入防止)。
//   X [T][n], Y [T][n]: 入出力 (重なり禁止)。
//   Wgate/Wg/Wu/Wd/Wg_s/Wu_s/Wd_s: 単体版と同一形式・同一条件。
//   T: トークン数 (>0)。n/h/E/k/S: 単体版と同一範囲。
//   out_ids [T][k], out_weights [T][k]: トークン毎の top-k 結果 (NULL不可。
//     bwd へそのまま渡せる形式。cache 同様スクラッチ扱いで、エラー時は
//     部分更新される場合がある。Y の不変のみ保証)。
//   cache_Gsel/Usel [T][k*h] or NULL, cache_Ysel [T][k*n] or NULL,
//   cache_Gs/Us [T][S*h] or NULL: 単体版の cache のバッチ敷き詰め版。
//     bwd を使う場合は単体版同様に非NULLで渡すこと。NULL 時は順伝播のみ。
//   cap_factor: 1.0–1.5 (既定 1.5=G2改訂)。
//   out_perm [T*k] or NULL, out_off [E+1] or NULL, out_drop [T*k] or NULL:
//     ソート結果の写し (決定論性・drop 率の観測用。NULL で省略)。
//   out_kept/out_dropped or NULL: kept/drop 総数。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。検証失敗時は Y を更新しない。
// 備考: 内部作業域 (yacc 等) は malloc 確保。AVX-512 不使用。
int jt_moe_fwd_batch(const float *restrict X,
                     const float *restrict Wgate,
                     const float *restrict Wg, const float *restrict Wu,
                     const float *restrict Wd,
                     const float *restrict Wg_s, const float *restrict Wu_s,
                     const float *restrict Wd_s,
                     float *restrict Y,
                     int T, int n, int h, int n_experts, int topk,
                     int n_shared,
                     size_t *restrict out_ids, float *restrict out_weights,
                     float *restrict cache_Gsel, float *restrict cache_Usel,
                     float *restrict cache_Ysel,
                     float *restrict cache_Gs, float *restrict cache_Us,
                     float cap_factor,
                     size_t *restrict out_perm, size_t *restrict out_off,
                     unsigned char *restrict out_drop,
                     size_t *restrict out_kept, size_t *restrict out_dropped);

// jt_moe_fwd_batch の重み有限スキャンを省略する内部高速経路。
// 通常バッチ版と同一の計算核 (bit一致)。省略範囲・使用条件は
// jt_moe_fwd_unchecked に同じ (区間冒頭で重み検証済み＋区間内不変の場合のみ)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。
int jt_moe_fwd_batch_unchecked(const float *restrict X,
                               const float *restrict Wgate,
                               const float *restrict Wg,
                               const float *restrict Wu,
                               const float *restrict Wd,
                               const float *restrict Wg_s,
                               const float *restrict Wu_s,
                               const float *restrict Wd_s,
                               float *restrict Y,
                               int T, int n, int h, int n_experts, int topk,
                               int n_shared,
                               size_t *restrict out_ids,
                               float *restrict out_weights,
                               float *restrict cache_Gsel,
                               float *restrict cache_Usel,
                               float *restrict cache_Ysel,
                               float *restrict cache_Gs,
                               float *restrict cache_Us,
                               float cap_factor,
                               size_t *restrict out_perm,
                               size_t *restrict out_off,
                               unsigned char *restrict out_drop,
                               size_t *restrict out_kept,
                               size_t *restrict out_dropped);

// MoE逆伝播のバッチ版 (Tトークン分。Step 3: 同一perm再利用のdW/dX GEMM)。
// fwd (jt_moe_fwd_batch) と同一 perm・同一 expert 境界で処理する
// (bwdでの再ルーティング・再ソート禁止。再ソートしない)。
// 素朴GEMM参照実装 (triple-loop相当。ブロッキング・SIMD新規最適化なし・
// AVX-512不使用。本格マイクロカーネルはStep 4)。
// 順序固定 (決定論性。f32非結合則のため):
//   - dX scatter-addは expert_id 昇順に固定 (同一tokenのk=2寄与の加算順)。
//   - gate勾配 (dWgate/dLogits) のtoken方向加算は token_pos 昇順に固定。
//     expert内はperm順 (安定ソートのためtoken_pos昇順と一致) に加算する。
//   - dWのM_e縮約は m昇順 (expert内token_pos昇順)。可変M_eによるbit変動は
//     許容範囲 (§5.2) で評価する (1e-8〜1e-7程度は正常、1e-6超は原因特定)。
// dropされたペアの寄与は0 (gate勾配も0。keptはrenormalizeする。§1.2/§3.3=G2改訂)。
// SwiGLU非線形のbwdはperm順のまま要素wiseに処理し、同一数式
// (jt_swiglu_bwdと同一のsigmoid/silu/ヤコビアン) を使う。
//   dY [T][n]: 上流勾配。X [T][n]: fwd入力と同一値。
//   Wgate/Wg/Wu/Wd/Wg_s/Wu_s/Wd_s: fwdと同一値・同一形式。
//   ids [T][k], weights [T][k]: fwdのout_ids/out_weightsと同一値。
//   Gsel/Usel [T][k*h], Ysel [T][k*n]: fwdのcacheと同一値。
//   Gs/Us [T][S*h]: fwdの共有cache (S==0時はNULL可)。
//   perm [kept], off [E+1], drop [T*k]: fwdのout_perm/out_off/out_dropと
//     同一値 (再ソートせず読み出すのみ)。NULL不可。
//   dX [T][n]: 入力勾配 (上書き)。
//   dWgate [E][n]: gate勾配 (上書き。非選択行は0)。
//   dWg/dWu/dWd [E][h][n]: routed勾配 (上書き。非選択expertは0)。
//   dWg_s/dWu_s/dWd_s [S][h][n]: 共有勾配 (上書き。S==0時はNULL可)。
//   dLogits [T][E] or NULL: token毎のgate logit勾配 (選択外・dropは0)。
//   T/n/h/E/k/S: fwd_batchと同一範囲。cap_factorは本関数では使わない
//     (perm/off/dropが既に確定済みのため。引数なし)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。検証失敗時は
//   dX・dW群・dLogitsを更新しない (fail-closed。単体版と同一条件)。
// 備考: 内部作業域はmalloc確保。AVX-512不使用。
int jt_moe_bwd_batch(const float *restrict dY, const float *restrict X,
                     const float *restrict Wgate,
                     const float *restrict Wg, const float *restrict Wu,
                     const float *restrict Wd,
                     const float *restrict Wg_s, const float *restrict Wu_s,
                     const float *restrict Wd_s,
                     const size_t *restrict ids, const float *restrict weights,
                     const float *restrict Gsel, const float *restrict Usel,
                     const float *restrict Ysel,
                     const float *restrict Gs, const float *restrict Us,
                     const size_t *restrict perm, const size_t *restrict off,
                     const unsigned char *restrict drop,
                     float *restrict dX,
                     float *restrict dWgate,
                     float *restrict dWg, float *restrict dWu,
                     float *restrict dWd,
                     float *restrict dWg_s, float *restrict dWu_s,
                     float *restrict dWd_s,
                     float *restrict dLogits,
                     int T, int n, int h, int n_experts, int topk,
                     int n_shared);

// jt_moe_bwd_batchの重み・キャッシュ有限スキャンを省略する内部高速経路。
// 通常バッチbwdと同一の計算核 (bit一致)。省略範囲・使用条件は
// jt_moe_bwd_uncheckedに同じ (区間冒頭で重み検証済み＋区間内不変、
// ids/weightsは同一ステップ内fwd産の場合のみ)。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。
int jt_moe_bwd_batch_unchecked(const float *restrict dY,
                               const float *restrict X,
                               const float *restrict Wgate,
                               const float *restrict Wg,
                               const float *restrict Wu,
                               const float *restrict Wd,
                               const float *restrict Wg_s,
                               const float *restrict Wu_s,
                               const float *restrict Wd_s,
                               const size_t *restrict ids,
                               const float *restrict weights,
                               const float *restrict Gsel,
                               const float *restrict Usel,
                               const float *restrict Ysel,
                               const float *restrict Gs,
                               const float *restrict Us,
                               const size_t *restrict perm,
                               const size_t *restrict off,
                               const unsigned char *restrict drop,
                               float *restrict dX,
                               float *restrict dWgate,
                               float *restrict dWg, float *restrict dWu,
                               float *restrict dWd,
                               float *restrict dWg_s, float *restrict dWu_s,
                               float *restrict dWd_s,
                               float *restrict dLogits,
                               int T, int n, int h, int n_experts, int topk,
                               int n_shared);

// StickyMoE損失の系列合算ヘルパー.
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
