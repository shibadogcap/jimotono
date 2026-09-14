// jt_routing: MoE top-k選択 + StickyMoE soft-hard損失 (Phase 1)。
// papers.md §3準拠。C11, errnoベース, クロスプラットフォーム (分岐なし)。
// 共有expert 2つは呼び出し側で除外済みであることが前提 (ヘッダ参照)。

#include "jimotono/routing.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>

// MSVC旧版などINFINITY未定義環境へのフォールバック (C11 portable)。
#ifndef INFINITY
#define INFINITY HUGE_VALF
#endif

int jt_routing_topk(const float *restrict logits, size_t n, size_t k,
                    size_t *restrict out_ids, float *restrict out_weights) {
    if (logits == NULL || out_ids == NULL || out_weights == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n == 0 || k == 0 || k > n) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }

    // NaN混入は未定義のため拒否 (-INFマスクは許容し選択対象外として扱う)。
    // +INFは後段softmaxの有限検査で拒否する。
    for (size_t i = 0; i < n; i++) {
        if (isnan(logits[i])) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }

    // Phase 1は正しさ優先の線形選択 O(n*k^2)。高速化は後回し。
    // 同値tie-break: 小さいindex優先 (v > best_v のstrict比較)。
    // 三項演算子で記述しCMOV化を期待 (分岐除去意識)。
    for (size_t p = 0; p < k; p++) {
        size_t best = n;  // 番兵 (未確定)
        float best_v = 0.0f;
        for (size_t i = 0; i < n; i++) {
            int taken = 0;
            for (size_t q = 0; q < p; q++) {
                taken |= (out_ids[q] == i);
            }
            float v = taken ? -INFINITY : logits[i];
            int better = (best == n) ? 1 : (v > best_v);
            best = better ? i : best;
            best_v = better ? v : best_v;
        }
        out_ids[p] = best;
    }

    // 選択k個に対するsoftmax (選択内max引きで数値安定化)。
    float mx = logits[out_ids[0]];
    for (size_t p = 1; p < k; p++) {
        float v = logits[out_ids[p]];
        mx = (v > mx) ? v : mx;
    }
    double sum = 0.0;
    for (size_t p = 0; p < k; p++) {
        double e = exp((double)logits[out_ids[p]] - (double)mx);
        sum += e;
        out_weights[p] = (float)e;
    }
    if (!(sum > 0.0) || !isfinite(sum)) {
        errno = EINVAL;  // 全-inf/NaN入力等の非有限softmax
        return JT_ERR_INVAL;
    }
    float inv = (float)(1.0 / sum);
    for (size_t p = 0; p < k; p++) {
        out_weights[p] *= inv;
    }
    return JT_OK;
}

int jt_routing_balance_loss(const size_t *restrict ids,
                            const float *restrict weights,
                            size_t T, size_t k, size_t E,
                            float *restrict out_aux,
                            float *restrict out_entropy) {
    double *sumw = NULL;
    size_t *cnt = NULL;
    double aux = 0.0;
    double ent = 0.0;
    size_t Tk = 0;
    if (ids == NULL || weights == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (T == 0 || k == 0 || E == 0 || E > 4096 || k > E) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (out_aux == NULL && out_entropy == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (k > SIZE_MAX / (T > 0 ? T : 1)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    Tk = T * k;
    // 範囲・有限検査（出力更新前に完了。fail-closed）。
    for (size_t q = 0; q < Tk; q++) {
        float w = weights[q];
        if (ids[q] >= E) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        if (!(w >= 0.0f) || !(w <= 1.0f) || !isfinite((double)w)) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    // E<=4096のため固定上限の自動配列（VLA回避）。
    {
        static double s_sumw[4096];
        static size_t s_cnt[4096];
        double inv = 0.0;
        if (E > 4096) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        sumw = s_sumw;
        cnt = s_cnt;
        for (size_t e = 0; e < E; e++) {
            sumw[e] = 0.0;
            cnt[e] = 0;
        }
        for (size_t q = 0; q < Tk; q++) {
            size_t e = ids[q];
            sumw[e] += (double)weights[q];
            cnt[e]++;
        }
        inv = 1.0 / (double)Tk;
        for (size_t e = 0; e < E; e++) {
            double f = (double)cnt[e] * inv;
            double p = sumw[e] * inv;
            aux += f * p;
        }
        aux *= (double)E;
        // top-k疎エントロピーのトークン平均（w=0項は0とみなす）。
        for (size_t t = 0; t < T; t++) {
            for (size_t p = 0; p < k; p++) {
                double w = (double)weights[t * k + p];
                if (w > 0.0) {
                    ent += -(w * log(w));
                }
            }
        }
        ent /= (double)T;
    }
    if (!isfinite(aux) || !isfinite(ent)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (out_aux != NULL) {
        float o = (float)aux;
        if (!isfinite((double)o)) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        *out_aux = o;
    }
    if (out_entropy != NULL) {
        float o = (float)ent;
        if (!isfinite((double)o)) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        *out_entropy = o;
    }
    return JT_OK;
}

int jt_routing_sticky_loss(const float *restrict g_t,
                           const float *restrict g_prev,
                           const float *restrict g_anchor,
                           size_t n, float lambda, float alpha,
                           size_t t_rel, size_t w,
                           float *restrict out_loss) {
    if (g_t == NULL || out_loss == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // 負値・NaN・+INFを拒否 (INF重みはloss発散のため不可)。
    if (!(lambda >= 0.0f) || !(alpha >= 0.0f) ||
        !isfinite((double)lambda) || !isfinite((double)alpha)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (w == 0 || t_rel >= w) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }

    // g_t自体の非有限値は (両ポインタNULLでも) 拒否。g_prev/g_anchor側の
    // 非有限値は最終lossの有限検査で捕捉する。
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)g_t[i])) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }

    // 内部はdouble累積 (gate差分二乗の桁落ち防止)。
    double cons = 0.0;
    if (g_prev != NULL) {
        for (size_t i = 0; i < n; i++) {
            double d = (double)g_t[i] - (double)g_prev[i];
            cons += d * d;
        }
    }
    double hard = 0.0;
    // 窓先頭 (t_rel==0) は拘束ゼロのためanchor走査を省略。
    if (g_anchor != NULL && t_rel != 0) {
        for (size_t i = 0; i < n; i++) {
            double d = (double)g_t[i] - (double)g_anchor[i];
            hard += d * d;
        }
        // 線形ランプ (t-s(t))/W。三項演算子でCMOV化を期待。
        double ramp = (double)t_rel / (double)w;
        hard *= ramp;
    }

    double loss = (double)lambda * cons + (double)alpha * hard;
    // gate非有限・オーバーフロー由来の非有限lossは拒否。
    if (!isfinite(loss)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_loss = (float)loss;
    // double→float変換でのINF化 (値域超過) も拒否。
    if (!isfinite((double)*out_loss)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    return JT_OK;
}
