// jt_moe_layer: MoE層 forward/backward + sticky系列合算 (Phase 2足場)。
// AGENTS.MD 7.1: C11, restrict, errnoベース + goto cleanup, クロスプラット。
// jt_routing_topk (実top-k) と jt_swiglu_fwd/bwd を再利用する薄い結合層。
// 内部はdouble累積。ホットループの分岐は三項演算子でCMOV化を期待し、
// 引数検証の分岐はコールド側に隔離 (routing.hと同一方針)。

#include "jimotono/moe_layer.h"

#include <errno.h>
#include <math.h>
#include <string.h>

#include "jimotono/routing.h"
#include "jimotono/train_bwd.h"

static int jt_moe_valid_dims(int n, int h, int e, int k, int s) {
    if (n <= 0 || h <= 0) {
        return 0;
    }
    if (n > JT_BWD_MAX_WIDE || h > JT_BWD_MAX_WIDE) {
        return 0;
    }
    if (e <= 0 || e > JT_MOE_MAX_EXPERTS) {
        return 0;
    }
    if (k <= 0 || k > e || k > JT_MOE_MAX_TOPK) {
        return 0;
    }
    if (s < 0 || s > JT_MOE_MAX_SHARED) {
        return 0;
    }
    return 1;
}

static int jt_moe_all_finite(const float *restrict x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)x[i])) {
            return 0;
        }
    }
    return 1;
}

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
               float *restrict cache_Gs, float *restrict cache_Us) {
    int rc = JT_ERR_INVAL;
    if (X == NULL || Wgate == NULL || Wg == NULL || Wu == NULL ||
        Wd == NULL || Y == NULL || out_ids == NULL ||
        out_weights == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_moe_valid_dims(n, h, n_experts, topk, n_shared)) {
        errno = EINVAL;
        goto cleanup;
    }
    if (n_shared > 0) {
        if (Wg_s == NULL || Wu_s == NULL || Wd_s == NULL) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    // 入力の有限検査 (fail-closed: Y更新前に拒否)。
    {
        size_t e = (size_t)n_experts;
        size_t nn = (size_t)n;
        size_t hh = (size_t)h;
        if (!jt_moe_all_finite(X, nn) ||
            !jt_moe_all_finite(Wgate, e * nn) ||
            !jt_moe_all_finite(Wg, e * hh * nn) ||
            !jt_moe_all_finite(Wu, e * hh * nn) ||
            !jt_moe_all_finite(Wd, e * hh * nn)) {
            errno = EINVAL;
            goto cleanup;
        }
        if (n_shared > 0) {
            size_t ss = (size_t)n_shared;
            if (!jt_moe_all_finite(Wg_s, ss * hh * nn) ||
                !jt_moe_all_finite(Wu_s, ss * hh * nn) ||
                !jt_moe_all_finite(Wd_s, ss * hh * nn)) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }

    {
        // gate logits (double累積→float)。E<=4096のため固定上限の自動配列。
        // C11 VLA回避 (MSVC非対応のため固定上限+検査)。
        static const int kMaxE = JT_MOE_MAX_EXPERTS;
        static const int kMaxW = JT_BWD_MAX_WIDE;
        float logits[JT_MOE_MAX_EXPERTS];
        double yacc[JT_BWD_MAX_WIDE];
        float tmpG[JT_BWD_MAX_WIDE];
        float tmpU[JT_BWD_MAX_WIDE];
        float tmpY[JT_BWD_MAX_WIDE];
        if (n_experts > kMaxE || n > kMaxW || h > kMaxW) {
            errno = EINVAL;
            goto cleanup;
        }
        for (int e = 0; e < n_experts; e++) {
            const float *wr = Wgate + (size_t)e * (size_t)n;
            double acc = 0.0;
            for (int j = 0; j < n; j++) {
                acc += (double)X[j] * (double)wr[j];
            }
            if (!isfinite(acc)) {
                errno = EINVAL;
                goto cleanup;
            }
            logits[e] = (float)acc;
            if (!isfinite((double)logits[e])) {
                errno = EINVAL;
                goto cleanup;
            }
        }
        // 実top-k (選択内softmax)。ids/weightsはここで確定する。
        {
            int rrc = jt_routing_topk(logits, (size_t)n_experts,
                                      (size_t)topk, out_ids, out_weights);
            if (rrc != JT_OK) {
                goto cleanup;  // errnoは下位で設定済み
            }
        }
        for (int j = 0; j < n; j++) {
            yacc[j] = 0.0;
        }
        // 選択expertのSwiGLU forwardを再利用し重み付き和。
        for (int p = 0; p < topk; p++) {
            size_t e = out_ids[p];
            if (e >= (size_t)n_experts) {
                errno = EINVAL;
                goto cleanup;
            }
            float w = out_weights[p];
            if (!(w >= 0.0f) || !isfinite((double)w)) {
                errno = EINVAL;
                goto cleanup;
            }
            const float *wgr = Wg + e * (size_t)h * (size_t)n;
            const float *wur = Wu + e * (size_t)h * (size_t)n;
            const float *wdr = Wd + e * (size_t)h * (size_t)n;
            float *Gp = (cache_Gsel != NULL)
                            ? (cache_Gsel + (size_t)p * (size_t)h)
                            : tmpG;
            float *Up = (cache_Usel != NULL)
                            ? (cache_Usel + (size_t)p * (size_t)h)
                            : tmpU;
            float *Yp = (cache_Ysel != NULL)
                            ? (cache_Ysel + (size_t)p * (size_t)n)
                            : tmpY;
            int frc = jt_swiglu_fwd(X, wgr, wur, wdr, Gp, Up, Yp, n, h);
            if (frc != JT_OK) {
                goto cleanup;  // errnoは下位で設定済み。Yは未更新。
            }
            for (int j = 0; j < n; j++) {
                yacc[j] += (double)w * (double)Yp[j];
            }
        }
        // 共有expert加算 (重み1)。
        for (int s = 0; s < n_shared; s++) {
            const float *wgr =
                Wg_s + (size_t)s * (size_t)h * (size_t)n;
            const float *wur =
                Wu_s + (size_t)s * (size_t)h * (size_t)n;
            const float *wdr =
                Wd_s + (size_t)s * (size_t)h * (size_t)n;
            float *Gp = (cache_Gs != NULL)
                            ? (cache_Gs + (size_t)s * (size_t)h)
                            : tmpG;
            float *Up = (cache_Us != NULL)
                            ? (cache_Us + (size_t)s * (size_t)h)
                            : tmpU;
            // Ysはyaccへ直接加算するためtmpYへ出力する (cache_Ysは持たない)。
            int frc = jt_swiglu_fwd(X, wgr, wur, wdr, Gp, Up, tmpY, n, h);
            if (frc != JT_OK) {
                goto cleanup;
            }
            for (int j = 0; j < n; j++) {
                yacc[j] += (double)tmpY[j];
            }
        }
        for (int j = 0; j < n; j++) {
            if (!isfinite(yacc[j])) {
                errno = EINVAL;
                goto cleanup;
            }
        }
        // 全成功後に一括commit (fail-closed: 拒否時はY不変)。
        for (int j = 0; j < n; j++) {
            Y[j] = (float)yacc[j];
        }
        if (out_logits != NULL) {
            for (int e = 0; e < n_experts; e++) {
                out_logits[e] = logits[e];
            }
        }
    }

    rc = JT_OK;
cleanup:
    return rc;
}

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
               int n, int h, int n_experts, int topk, int n_shared) {
    int rc = JT_ERR_INVAL;
    if (dY == NULL || X == NULL || Wgate == NULL || Wg == NULL ||
        Wu == NULL || Wd == NULL || ids == NULL || weights == NULL ||
        Gsel == NULL || Usel == NULL || Ysel == NULL || dX == NULL ||
        dWgate == NULL || dWg == NULL || dWu == NULL || dWd == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_moe_valid_dims(n, h, n_experts, topk, n_shared)) {
        errno = EINVAL;
        goto cleanup;
    }
    if (n_shared > 0) {
        if (Wg_s == NULL || Wu_s == NULL || Wd_s == NULL || Gs == NULL ||
            Us == NULL || dWg_s == NULL || dWu_s == NULL ||
            dWd_s == NULL) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    // 有限検査 (出力更新前に全入力を検査)。
    {
        size_t e = (size_t)n_experts;
        size_t nn = (size_t)n;
        size_t hh = (size_t)h;
        size_t kk = (size_t)topk;
        if (!jt_moe_all_finite(dY, nn) || !jt_moe_all_finite(X, nn) ||
            !jt_moe_all_finite(Wgate, e * nn) ||
            !jt_moe_all_finite(Wg, e * hh * nn) ||
            !jt_moe_all_finite(Wu, e * hh * nn) ||
            !jt_moe_all_finite(Wd, e * hh * nn) ||
            !jt_moe_all_finite(weights, kk) ||
            !jt_moe_all_finite(Gsel, kk * hh) ||
            !jt_moe_all_finite(Usel, kk * hh) ||
            !jt_moe_all_finite(Ysel, kk * nn)) {
            errno = EINVAL;
            goto cleanup;
        }
        if (n_shared > 0) {
            size_t ss = (size_t)n_shared;
            if (!jt_moe_all_finite(Wg_s, ss * hh * nn) ||
                !jt_moe_all_finite(Wu_s, ss * hh * nn) ||
                !jt_moe_all_finite(Wd_s, ss * hh * nn) ||
                !jt_moe_all_finite(Gs, ss * hh) ||
                !jt_moe_all_finite(Us, ss * hh)) {
                errno = EINVAL;
                goto cleanup;
            }
        }
        // ids範囲・重複・weights範囲の検査。
        {
            double wsum = 0.0;
            for (int p = 0; p < topk; p++) {
                if (ids[p] >= (size_t)n_experts) {
                    errno = EINVAL;
                    goto cleanup;
                }
                float w = weights[p];
                if (!(w >= 0.0f) || !(w <= 1.0f) ||
                    !isfinite((double)w)) {
                    errno = EINVAL;
                    goto cleanup;
                }
                wsum += (double)w;
            }
            for (int p = 0; p < topk; p++) {
                for (int q = p + 1; q < topk; q++) {
                    if (ids[p] == ids[q]) {
                        errno = EINVAL;
                        goto cleanup;
                    }
                }
            }
            if (!(wsum > 0.999 && wsum < 1.001) || !isfinite(wsum)) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }

    {
        static const int kMaxW = JT_BWD_MAX_WIDE;
        double dw_dp[JT_MOE_MAX_TOPK];
        double dlog[JT_MOE_MAX_EXPERTS];
        double dx_acc[JT_BWD_MAX_WIDE];
        float dYe[JT_BWD_MAX_WIDE];
        float dxe[JT_BWD_MAX_WIDE];
        if (n > kMaxW || h > kMaxW || topk > JT_MOE_MAX_TOPK ||
            n_experts > JT_MOE_MAX_EXPERTS) {
            errno = EINVAL;
            goto cleanup;
        }
        // dL/dw_p = dot(Ysel_p, dY)。
        for (int p = 0; p < topk; p++) {
            const float *Yp = Ysel + (size_t)p * (size_t)n;
            double acc = 0.0;
            for (int j = 0; j < n; j++) {
                acc += (double)Yp[j] * (double)dY[j];
            }
            if (!isfinite(acc)) {
                errno = EINVAL;
                goto cleanup;
            }
            dw_dp[p] = acc;
        }
        // softmaxヤコビアン (選択内のみ。選択外は0のSTE直通)。
        {
            double s = 0.0;
            for (int p = 0; p < topk; p++) {
                s += (double)weights[p] * dw_dp[p];
            }
            for (int e = 0; e < n_experts; e++) {
                dlog[e] = 0.0;
            }
            for (int p = 0; p < topk; p++) {
                double v =
                    (double)weights[p] * (dw_dp[p] - s);
                if (!isfinite(v)) {
                    errno = EINVAL;
                    goto cleanup;
                }
                dlog[ids[p]] = v;
            }
        }
        for (int j = 0; j < n; j++) {
            dx_acc[j] = 0.0;
        }
        // gate経路のdX寄与。
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int e = 0; e < n_experts; e++) {
                double dl = dlog[e];
                if (dl != 0.0) {
                    acc += dl *
                           (double)Wgate[(size_t)e * (size_t)n +
                                         (size_t)j];
                }
            }
            dx_acc[j] += acc;
        }
        // 非選択expertのdWは0で埋める (上書き・加算ではない)。
        {
            size_t e = (size_t)n_experts;
            size_t hhn = (size_t)h * (size_t)n;
            // 選択集合のルックアップ (k<=64のため線形走査)。
            for (size_t ee = 0; ee < e; ee++) {
                int sel = 0;
                for (int p = 0; p < topk; p++) {
                    sel |= (ids[p] == ee);
                }
                if (!sel) {
                    float *rg = dWg + ee * hhn;
                    float *ru = dWu + ee * hhn;
                    float *rd = dWd + ee * hhn;
                    for (size_t i = 0; i < hhn; i++) {
                        rg[i] = 0.0f;
                        ru[i] = 0.0f;
                        rd[i] = 0.0f;
                    }
                }
            }
            // dWgate/dLogitsの非選択行も0 (選択行は後段で上書き)。
            for (size_t ee = 0; ee < e; ee++) {
                int sel = 0;
                for (int p = 0; p < topk; p++) {
                    sel |= (ids[p] == ee);
                }
                if (!sel) {
                    float *rg = dWgate + ee * (size_t)n;
                    for (int j = 0; j < n; j++) {
                        rg[j] = 0.0f;
                    }
                    if (dLogits != NULL) {
                        dLogits[ee] = 0.0f;
                    }
                }
            }
        }
        // 選択expertのSwiGLU bwdを再利用 (dY_e = w_p * dY)。
        for (int p = 0; p < topk; p++) {
            size_t e = ids[p];
            float w = weights[p];
            for (int j = 0; j < n; j++) {
                dYe[j] = (float)((double)w * (double)dY[j]);
            }
            const float *Gp = Gsel + (size_t)p * (size_t)h;
            const float *Up = Usel + (size_t)p * (size_t)h;
            const float *wdr = Wd + e * (size_t)h * (size_t)n;
            const float *wgr = Wg + e * (size_t)h * (size_t)n;
            const float *wur = Wu + e * (size_t)h * (size_t)n;
            float *oWg = dWg + e * (size_t)h * (size_t)n;
            float *oWu = dWu + e * (size_t)h * (size_t)n;
            float *oWd = dWd + e * (size_t)h * (size_t)n;
            int brc = jt_swiglu_bwd(dYe, X, Gp, Up, wdr, wgr, wur, dxe,
                                    oWg, oWu, oWd, n, h);
            if (brc != JT_OK) {
                goto cleanup;  // errnoは下位で設定済み
            }
            for (int j = 0; j < n; j++) {
                dx_acc[j] += (double)dxe[j];
            }
            // gate線形の選択行。
            {
                double dl = dlog[e];
                float *rg = dWgate + e * (size_t)n;
                for (int j = 0; j < n; j++) {
                    rg[j] = (float)(dl * (double)X[j]);
                }
                if (dLogits != NULL) {
                    dLogits[e] = (float)dl;
                }
            }
        }
        // 共有expert (重み1で常時発火)。
        for (int s = 0; s < n_shared; s++) {
            const float *Gp = Gs + (size_t)s * (size_t)h;
            const float *Up = Us + (size_t)s * (size_t)h;
            const float *wdr =
                Wd_s + (size_t)s * (size_t)h * (size_t)n;
            const float *wgr =
                Wg_s + (size_t)s * (size_t)h * (size_t)n;
            const float *wur =
                Wu_s + (size_t)s * (size_t)h * (size_t)n;
            float *oWg = dWg_s + (size_t)s * (size_t)h * (size_t)n;
            float *oWu = dWu_s + (size_t)s * (size_t)h * (size_t)n;
            float *oWd = dWd_s + (size_t)s * (size_t)h * (size_t)n;
            int brc = jt_swiglu_bwd(dY, X, Gp, Up, wdr, wgr, wur, dxe,
                                    oWg, oWu, oWd, n, h);
            if (brc != JT_OK) {
                goto cleanup;
            }
            for (int j = 0; j < n; j++) {
                dx_acc[j] += (double)dxe[j];
            }
        }
        for (int j = 0; j < n; j++) {
            if (!isfinite(dx_acc[j])) {
                errno = EINVAL;
                goto cleanup;
            }
        }
        // 一括commit (dW群は上記で直接書き込み済みだが、検証通過後のみ到達)。
        for (int j = 0; j < n; j++) {
            dX[j] = (float)dx_acc[j];
        }
    }

    rc = JT_OK;
cleanup:
    return rc;
}

int jt_moe_sticky_seq_loss(const float *restrict gates, size_t T, size_t n,
                           float lambda, float alpha, size_t w,
                           float *restrict out_loss) {
    int rc = JT_ERR_INVAL;
    if (gates == NULL || out_loss == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (T == 0 || n == 0 || w == 0) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!(lambda >= 0.0f) || !(alpha >= 0.0f) ||
        !isfinite((double)lambda) || !isfinite((double)alpha)) {
        errno = EINVAL;
        goto cleanup;
    }
    {
        double sum = 0.0;
        for (size_t t = 0; t < T; t++) {
            const float *gt = gates + t * n;
            const float *gp = (t > 0) ? (gates + (t - 1) * n) : NULL;
            size_t s_idx = (t / w) * w;
            size_t t_rel = t - s_idx;
            const float *ga = gates + s_idx * n;
            float lt = 0.0f;
            int lrc = jt_routing_sticky_loss(gt, gp, ga, n, lambda, alpha,
                                             t_rel, w, &lt);
            if (lrc != JT_OK) {
                goto cleanup;  // errnoは下位で設定済み。out_loss不変。
            }
            sum += (double)lt;
        }
        if (!isfinite(sum)) {
            errno = EINVAL;
            goto cleanup;
        }
        {
            double avg = sum / (double)T;
            float out = (float)avg;
            if (!isfinite((double)out)) {
                errno = EINVAL;
                goto cleanup;
            }
            *out_loss = out;
        }
    }
    // NOTE: L_CE・μL_bal・層平均との合算は呼び出し側TODO (routing.h準拠)。
    rc = JT_OK;
cleanup:
    return rc;
}
