#pragma once
#ifndef JIMOTONO_CHECKPOINT_H
#define JIMOTONO_CHECKPOINT_H
// 勾配チェックポインティング足場 (O(√n)) (Phase 2)。
// 方針: 層区間のforward再計算スケジューラ。区間境界のみ保存し、中間活性は
// backward時に再計算する。AGENTS.MD §4.1 / DESIGN.MD §4.2準拠。
// 規約: C11, restrict, errnoベース (AGENTS.MD 7.1)。
//
// 本ヘッダはモデル非依存の純粋関数 (区間分割数・境界・メモリ見積り) から提供
// し、単体テスト可能にする。実再計算コールバック (層forward本体) はモデル側
// 実装のため、スケジューラ本体はAPIスケルトン (JT_ERR_NOSUP) とする。
// オプティマイザ・ES-MoE本体は別wt担当のため含めない。

#include <stddef.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// 上限 (1B MoE検証: 数十層〜数百層を想定)。
#define JT_CKPT_MAX_LAYERS 4096

// O(√n)既定の区間数: ceil(sqrt(n_layers))。
// 戻り値: JT_OK / JT_ERR_INVAL (errno併用)。
int jt_ckpt_num_segments(int n_layers, int *restrict out_seg);

// 区間境界の均等分割 (純粋関数・テスト容易)。
// bounds[0]=0, bounds[n_seg]=n_layers、残りはできるだけ均等
// (先頭区間から1つずつ余りを配分)。bounds長はn_seg+1以上。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_ckpt_boundaries(int n_layers, int n_seg, int *restrict bounds,
                       size_t bounds_n);

// メモリ見積り (純粋関数、単位byte)。
//   full: 全活性保存 = n_layers * per_layer。
//   stored: 境界のみ保存 = (n_seg+1) * per_layer。
//   peak: 再計算ピーク = stored + seg_max * per_layer
//         (seg_maxは最長区間長=ceil(n_layers/n_seg))。
// オーバーフロー時はJT_ERR_NOMEM。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM (errno併用)。
int jt_ckpt_mem_estimate(int n_layers, int n_seg, size_t per_layer,
                         size_t *restrict out_full,
                         size_t *restrict out_stored,
                         size_t *restrict out_peak);

// 削減率のbench足場 (純粋関数、double)。
//   ratio = 1 - stored/full (0..1)。full==0はINVAL。
// メモリ削減効果の測定は本関数+上記estimateで行い、実時間計測は
// bench harness (bench/common) 側に委ねる。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_ckpt_saving_ratio(size_t full, size_t stored,
                         double *restrict out_ratio);

// 再計算スケジューラのプラン (境界配列は呼び出し側所有)。
typedef struct jt_ckpt_plan {
    int n_layers;
    int n_seg;
    const int *bounds;  // 長さn_seg+1、所有権は呼び出し側
} jt_ckpt_plan_t;

// プラン検査 (境界の単調性・端点0/n_layersを検証)。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_ckpt_plan_init(const jt_ckpt_plan_t *restrict plan);

// 層idxが属する区間番号を返す (純粋関数、二分探索不要の線形走査)。
// 戻り値: JT_OK / JT_ERR_INVAL (範囲外はINVAL)。
int jt_ckpt_find_segment(const jt_ckpt_plan_t *restrict plan, int layer,
                         int *restrict out_seg_idx);

// 区間再計算スタブ: forward再計算コールバックはモデル側のためP2では未実装。
// 正常入力はJT_ERR_NOSUP+errno=ENOSYS、不正入力はJT_ERR_INVAL+errno=EINVAL
// (gdn2 chunkスタブと同一契約、戻り値のみで区別可能)。
typedef int (*jt_ckpt_fwd_fn)(int layer, void *ctx);
int jt_ckpt_recompute_range(const jt_ckpt_plan_t *restrict plan, int seg_idx,
                            jt_ckpt_fwd_fn fwd, void *ctx);

// bench足場: 見積り一括取得 (full/stored/peak/ratio)。
typedef struct jt_ckpt_bench {
    size_t full;
    size_t stored;
    size_t peak;
    double ratio;
} jt_ckpt_bench_t;
int jt_ckpt_bench_estimate(int n_layers, int n_seg, size_t per_layer,
                           jt_ckpt_bench_t *restrict out);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_CHECKPOINT_H
