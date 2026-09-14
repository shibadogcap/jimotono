#pragma once
#ifndef JIMOTONO_ROUTING_H
#define JIMOTONO_ROUTING_H
// MoE routing: top-k選択 + StickyMoE soft-hard損失 (Phase 1)。
// 準拠: knowledge/papers.md §3 (StickyMoE soft-hard variant, 2607.08780v1)。
// 規約: C11, restrict, errnoベース (AGENTS.MD 7.1)。ホットループは分岐除去意識
// (CMOV化を期待し三項演算子を使用。引数検証の分岐はコールド側に隔離)。
//
// [共有expertの扱い]
// 共有expert 2つは常駐のため本モジュールの対象外。呼び出し側でlogitsから
// 除外し、routed expertのみを渡すこと。損失計算もrouted expert分布のみで行う
// (papers.md §3: 「共有expert 2つは損失対象外(常駐のため)」)。
//
// [集合一致の再定義注意]
// 原典のSR/CHR (SR 59%減・miss 3.92倍) は top-1・C=2・4 experts・WikiText-2・
// 8.8M/22M小規模条件の定義である。JIMOTONO (数千expert×top-8〜16) では
// top-1一致ではなく「top-k集合の一致」で再定義し、W・Cもtop-k倍率で拡大
// (W=8〜16) しないと過大評価になる (papers.md §3適用注意1, §5申し送り)。
// 本ヘッダの関数は集合定義に依存しない (選択と損失のみ提供し、SR/CHR計測は
// 呼び出し側・bench側で集合一致として定義すること)。
//
// [Wの選び方]
// 初期値は W=2〜4 から開始 (W=8はPareto確認用に留める)。Hard単体はW=2が最良、
// W=8は逆に悪化の報告あり。併用最良は λ=0.1, α=0.05〜1.0, W=4。
// 学習開始は λ=0.05〜0.1 推奨 (λ>=0.2でPPL悪化のトレードオフ顕在)。
//
// [層別λ・境界マスク・集約は呼び出し側]
// L0は改善鈍化 (embedding未混合) のため層別λ (L0小・L1/L2大) を推奨し、
// 文・関数境界ではペナルティをマスクすること (papers.md §3適用注意2,3)。
// 本関数は1トークン・1層分の寄与のみ返し、系列平均 1/(T-1)・全層平均・
// L_CE・μL_bal (μ=0.01) との合算は呼び出し側で行うこと。
//
// [restrict / NaN方針]
// gate 3点 (g_t/g_prev/g_anchor) は互いにエイリアス禁止。同一値の連続でも
// 別バッファを渡すこと (SIMD化のためrestrict維持)。logits中のNaNは
// JT_ERR_INVAL (-INFマスクは許容し選択対象外として扱う)。gate中の非有限値・
// 非有限lambda/alpha・非有限lossはJT_ERR_INVAL。

#include <stddef.h>
#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// top-k選択 + softmax正規化。
// logits: routed expertの生logit (共有expert除外済み), 長さn。
// k: 可変 (実行時動的, RULE.MD 1)。1 <= k <= n。
// out_ids: 長さk。logit降順 (同値は小さいindex優先の決定的順序)。
// out_weights: 長さk。選択k個に対するsoftmax (合計1)。数値安定のため
//   選択内max引きで正規化。
// 計算量 O(n*k^2)。Phase 1は正しさ優先の線形選択。高速化 (heap-based
// O(n log k) 等) は後回し。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=EINVAL: NULL引数, n==0, k==0, k>n,
//   非有限softmax)。
int jt_routing_topk(const float *restrict logits, size_t n, size_t k,
                    size_t *restrict out_ids, float *restrict out_weights);

// load-balancing補助損失 + ルーティングエントロピー (Switch Transformer流)。
//   f_e = cnt_e / (T*k)（割当て率）, P_e = sum_w_e / (T*k)（平均確率）,
//   L_aux = E * Σ_e f_e * P_e（均等時は k^2/E→約1、崩壊時は k に近づく）。
//   f_e は hard 選択のため定数扱い（straight-through。微分は P_e 経由のみ）。
//   entropy は top-k 重みのトークン平均エントロピー H = -1/T Σ_{t,p} w*ln w
//  （密分布ではなく疎top-kの代理指標。崩壊検出用ログ列）。
//   ids [T*k]（token-major、全要素<E）、weights [T*k]（0..1・有限）。
//   T>0、k>=1、E>=1（E<=4096、k<=E）。out_aux/out_entropyはNULL可（片方のみ取得可）。
//   cap検証は含まない（capは jt_moe_batch_sort 側で1.0–1.5に固定。2.0での隠蔽禁止）。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=EINVAL)。
int jt_routing_balance_loss(const size_t *restrict ids,
                            const float *restrict weights,
                            size_t T, size_t k, size_t E,
                            float *restrict out_aux,
                            float *restrict out_entropy);

// StickyMoE soft-hard損失 (1トークン分の寄与)。
//   L_cons = ||g_t - g_{t-1}||_2^2 (全層平均の前の1トークン分)
//   L_hard = (t-s(t))/W * ||g_t - g_{s(t)}||_2^2 (線形ランプ付き窓アンカー拘束)
//   L = λ*L_cons + α*L_hard
// g_t: 現トークンのgate分布 (routed expertのみ, 長さn)。NULL不可。
// g_prev: 前トークンのgate分布。NULL可 (=系列先頭, L_cons=0として扱う)。
// g_anchor: 窓先頭アンカー g_{s(t)}。NULL可 (=L_hard=0として扱う)。
// lambda (λ): soft重み。0.05〜0.1開始 (負値不可)。
// alpha (α): hard重み。0.05〜1.0 (負値不可)。
// t_rel: t-s(t)。窓先頭で0。0 <= t_rel < w。
// w (W): 窓幅。2〜4開始 (0不可)。
// out_loss: 結果格納。内部はdouble累積しfloatで返す。
// 仕様(MINOR-3): λ=α=0でも g_prev/g_anchor の非有限値は INVAL (0*INF=NaN防止のfail-closed)。重み0でも検査する。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=EINVAL)。
int jt_routing_sticky_loss(const float *restrict g_t,
                           const float *restrict g_prev,
                           const float *restrict g_anchor,
                           size_t n, float lambda, float alpha,
                           size_t t_rel, size_t w,
                           float *restrict out_loss);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_ROUTING_H
