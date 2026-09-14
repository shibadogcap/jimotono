// jt_moe_layer: MoE層 forward/backward + sticky系列合算 (Phase 2足場)。
// AGENTS.MD 7.1: C11, restrict, errnoベース + goto cleanup, クロスプラット。
// jt_routing_topk (実top-k) と jt_swiglu_fwd/bwd を再利用する薄い結合層。
// 内部はdouble累積。ホットループの分岐は三項演算子でCMOV化を期待し、
// 引数検証の分岐はコールド側に隔離 (routing.hと同一方針)。

#include "jimotono/moe_layer.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/routing.h"
#include "jimotono/train_bwd.h"

// P2 SIMD (Phase D) AVX2 パス (train_bwd.c と同一方針)。
// - 出力 dim 方向の elementwise/f64累積は演算順序同一・FMA 不使用で bit同一。
// - reduction dim 方向のドット (jt_moe_avx2_dot: gate logits・dw_dp) のみ
//   合算順序が変わるため tol内一致 (相対 ~1e-16。テスト tol=1e-5/1e-3)。
// - gate 経路の dX 寄与 (dl!=0 の分岐付き疎加算) はスカラー維持
//   (分岐混じりでベクトル化の利益なし)。
// - AVX-512 には手を出さない (tmac.c と同一理由)。
// - アライメント非依存 (loadu/storeu のみ)。フォールバックは既存スカラー。
#ifdef __AVX2__
#include <immintrin.h>

// ドット (f64累積)。reduction 順序が変わるため tol内一致 (bit一致ではない)。
static double jt_moe_avx2_dot(const float *restrict x,
                              const float *restrict y, int n) {
    __m256d acc = _mm256_setzero_pd();
    int j = 0;
    int n4 = n & ~3;
    for (; j < n4; j += 4) {
        __m256d vx =
            _mm256_cvtps_pd(_mm_loadu_ps((const float *)(x + j)));
        __m256d vy =
            _mm256_cvtps_pd(_mm_loadu_ps((const float *)(y + j)));
        acc = _mm256_add_pd(acc, _mm256_mul_pd(vx, vy));
    }
    double lane[4];
    _mm256_storeu_pd(lane, acc);
    double s = (lane[0] + lane[1]) + (lane[2] + lane[3]);
    for (; j < n; j++) {
        s += (double)x[j] * (double)y[j];
    }
    return s;
}

// dst[j] += k*src[j] (f64。yacc/dx_acc 累積と同一形。レーン毎順序同一で bit同一)。
static void jt_moe_avx2_f64_add(double *restrict dst, double k,
                                const float *restrict src, int n) {
    __m256d vk = _mm256_set1_pd(k);
    int j = 0;
    int n4 = n & ~3;
    for (; j < n4; j += 4) {
        __m256d vd = _mm256_loadu_pd(dst + j);
        __m256d vs =
            _mm256_cvtps_pd(_mm_loadu_ps((const float *)(src + j)));
        vd = _mm256_add_pd(vd, _mm256_mul_pd(vk, vs));
        _mm256_storeu_pd(dst + j, vd);
    }
    for (; j < n; j++) {
        dst[j] += k * (double)src[j];
    }
}

// dst[j] = (float)(k*src[j]) (f64→f32 commit。dWgate 行と同一形。bit同一)。
static void jt_moe_avx2_f64_mulk_commit(float *restrict dst, double k,
                                        const float *restrict src, int n) {
    __m256d vk = _mm256_set1_pd(k);
    int j = 0;
    int n4 = n & ~3;
    for (; j < n4; j += 4) {
        __m256d vs =
            _mm256_cvtps_pd(_mm_loadu_ps((const float *)(src + j)));
        __m128 v = _mm256_cvtpd_ps(_mm256_mul_pd(vk, vs));
        _mm_storeu_ps(dst + j, v);
    }
    for (; j < n; j++) {
        dst[j] = (float)(k * (double)src[j]);
    }
}

#endif

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

static int jt_moe_fwd_impl(const float *restrict X,
                             const float *restrict Wgate,
                             const float *restrict Wg,
                             const float *restrict Wu,
                             const float *restrict Wd,
                             const float *restrict Wg_s,
                             const float *restrict Wu_s,
                             const float *restrict Wd_s,
                             float *restrict Y,
                             int n, int h, int n_experts, int topk,
                             int n_shared,
                             size_t *restrict out_ids,
                             float *restrict out_weights,
                             float *restrict out_logits,
                             float *restrict cache_Gsel,
                             float *restrict cache_Usel,
                             float *restrict cache_Ysel,
                             float *restrict cache_Gs,
                             float *restrict cache_Us, int validate) {
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
    // O(E*H*N)の全重み走査はvalidate時のみ。uncheckedでは呼び出し側の
    // ステップ冒頭検証＋区間内不変に委ね、ここでは省略する。
    if (validate) {
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
#ifdef __AVX2__
            // reduction 方向のため tol内一致 (jt_moe_avx2_dot 参照)。
            double acc = jt_moe_avx2_dot(X, wr, n);
#else
            double acc = 0.0;
            for (int j = 0; j < n; j++) {
                acc += (double)X[j] * (double)wr[j];
            }
#endif
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
            // unchecked区間では内側の重みスキャンも省略 (外側の冒頭検証に一元化)。
            // 同一計算核のためbit一致。routing_topkはO(E)で安価なため検証版のまま。
            int frc = validate
                          ? jt_swiglu_fwd(X, wgr, wur, wdr, Gp, Up, Yp, n,
                                          h)
                          : jt_swiglu_fwd_unchecked(X, wgr, wur, wdr, Gp, Up,
                                                    Yp, n, h);
            if (frc != JT_OK) {
                goto cleanup;  // errnoは下位で設定済み。Yは未更新。
            }
#ifdef __AVX2__
            jt_moe_avx2_f64_add(yacc, (double)w, Yp, n);
#else
            for (int j = 0; j < n; j++) {
                yacc[j] += (double)w * (double)Yp[j];
            }
#endif
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
            int frc = validate
                          ? jt_swiglu_fwd(X, wgr, wur, wdr, Gp, Up, tmpY, n,
                                          h)
                          : jt_swiglu_fwd_unchecked(X, wgr, wur, wdr, Gp, Up,
                                                    tmpY, n, h);
            if (frc != JT_OK) {
                goto cleanup;
            }
#ifdef __AVX2__
            jt_moe_avx2_f64_add(yacc, 1.0, tmpY, n);
#else
            for (int j = 0; j < n; j++) {
                yacc[j] += (double)tmpY[j];
            }
#endif
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

// 公開API (検証あり、従来通りfail-closed)。
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
    return jt_moe_fwd_impl(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Y, n, h,
                           n_experts, topk, n_shared, out_ids, out_weights,
                           out_logits, cache_Gsel, cache_Usel, cache_Ysel,
                           cache_Gs, cache_Us, 1);
}

// 内部高速経路 (重み有限スキャンなし。ヘッダの使用条件コメント参照)。
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
                         float *restrict cache_Gs, float *restrict cache_Us) {
    return jt_moe_fwd_impl(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Y, n, h,
                           n_experts, topk, n_shared, out_ids, out_weights,
                           out_logits, cache_Gsel, cache_Usel, cache_Ysel,
                           cache_Gs, cache_Us, 0);
}

static int jt_moe_bwd_impl(const float *restrict dY, const float *restrict X,
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
               int n, int h, int n_experts, int topk, int n_shared,
               int validate) {
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
    // O(E*H*N)の全重み・キャッシュ走査はvalidate時のみ。uncheckedでは
    // 呼び出し側のステップ冒頭検証＋区間内不変に委ね、ここでは省略する。
    // ids/weightsの範囲検査 (O(k^2)) も省略し、同一ステップ内fwd産の値を
    // そのまま受け取る。計算途中のisfiniteガードは両経路で残る。
    if (validate) {
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
#ifdef __AVX2__
            // reduction 方向のため tol内一致 (jt_moe_avx2_dot 参照)。
            double acc = jt_moe_avx2_dot(Yp, dY, n);
#else
            double acc = 0.0;
            for (int j = 0; j < n; j++) {
                acc += (double)Yp[j] * (double)dY[j];
            }
#endif
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
#ifdef __AVX2__
            // f64評価で bit同一 (スカラー核の (double)w*(double)dY と同順序)。
            jt_moe_avx2_f64_mulk_commit(dYe, (double)w, dY, n);
#else
            for (int j = 0; j < n; j++) {
                dYe[j] = (float)((double)w * (double)dY[j]);
            }
#endif
            const float *Gp = Gsel + (size_t)p * (size_t)h;
            const float *Up = Usel + (size_t)p * (size_t)h;
            const float *wdr = Wd + e * (size_t)h * (size_t)n;
            const float *wgr = Wg + e * (size_t)h * (size_t)n;
            const float *wur = Wu + e * (size_t)h * (size_t)n;
            float *oWg = dWg + e * (size_t)h * (size_t)n;
            float *oWu = dWu + e * (size_t)h * (size_t)n;
            float *oWd = dWd + e * (size_t)h * (size_t)n;
            // unchecked区間では内側の重みスキャンも省略 (外側の冒頭検証に一元化)。
            int brc = validate ? jt_swiglu_bwd(dYe, X, Gp, Up, wdr, wgr,
                                               wur, dxe, oWg, oWu, oWd, n,
                                               h)
                               : jt_swiglu_bwd_unchecked(dYe, X, Gp, Up, wdr,
                                                         wgr, wur, dxe, oWg,
                                                         oWu, oWd, n, h);
            if (brc != JT_OK) {
                goto cleanup;  // errnoは下位で設定済み
            }
#ifdef __AVX2__
            jt_moe_avx2_f64_add(dx_acc, 1.0, dxe, n);
#else
            for (int j = 0; j < n; j++) {
                dx_acc[j] += (double)dxe[j];
            }
#endif
            // gate線形の選択行。
            {
                double dl = dlog[e];
                float *rg = dWgate + e * (size_t)n;
#ifdef __AVX2__
                jt_moe_avx2_f64_mulk_commit(rg, dl, X, n);
#else
                for (int j = 0; j < n; j++) {
                    rg[j] = (float)(dl * (double)X[j]);
                }
#endif
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
            int brc = validate ? jt_swiglu_bwd(dY, X, Gp, Up, wdr, wgr, wur,
                                               dxe, oWg, oWu, oWd, n, h)
                               : jt_swiglu_bwd_unchecked(dY, X, Gp, Up, wdr,
                                                         wgr, wur, dxe, oWg,
                                                         oWu, oWd, n, h);
            if (brc != JT_OK) {
                goto cleanup;
            }
#ifdef __AVX2__
            jt_moe_avx2_f64_add(dx_acc, 1.0, dxe, n);
#else
            for (int j = 0; j < n; j++) {
                dx_acc[j] += (double)dxe[j];
            }
#endif
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

// 公開API (検証あり、従来通りfail-closed)。
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
    return jt_moe_bwd_impl(dY, X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, ids,
                           weights, Gsel, Usel, Ysel, Gs, Us, dX, dWgate,
                           dWg, dWu, dWd, dWg_s, dWu_s, dWd_s, dLogits, n, h,
                           n_experts, topk, n_shared, 1);
}

// 内部高速経路 (重み・キャッシュ有限スキャンなし。ヘッダの使用条件コメント参照)。
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
                         int n, int h, int n_experts, int topk, int n_shared) {
    return jt_moe_bwd_impl(dY, X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, ids,
                           weights, Gsel, Usel, Ysel, Gs, Us, dX, dWgate,
                           dWg, dWu, dWd, dWg_s, dWu_s, dWd_s, dLogits, n, h,
                           n_experts, topk, n_shared, 0);
}

// ---- Phase G Step 1: ソート＋dispatch (計算は既存GEMVのまま) ----
// counting sort (E<=4096)。安定・決定論的・単一スレッド構築。
// AVX-512 不使用 (スカラーのみ)。
int jt_moe_batch_sort(const size_t *restrict ids, int T, int k, int E,
                      float cap_factor,
                      size_t *restrict perm, size_t *restrict off,
                      unsigned char *restrict drop,
                      size_t *restrict out_kept, size_t *restrict out_dropped) {
    size_t cnt[JT_MOE_MAX_EXPERTS];
    size_t Tk = 0;
    size_t cap = 0;
    size_t kept_total = 0;
    if (ids == NULL || perm == NULL || off == NULL || drop == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (T <= 0 || k <= 0 || E <= 0 || E > JT_MOE_MAX_EXPERTS || k > E ||
        k > JT_MOE_MAX_TOPK) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (!(cap_factor >= 1.0f) || !(cap_factor <= 1.5f) ||
        !isfinite((double)cap_factor)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if ((size_t)k > SIZE_MAX / (size_t)T) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    Tk = (size_t)T * (size_t)k;
    // ids 範囲検査 (出力更新前に完了。fail-closed)。
    for (size_t q = 0; q < Tk; q++) {
        if (ids[q] >= (size_t)E) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    {
        double capd =
            ceil((double)cap_factor * (double)Tk / (double)E);
        if (!isfinite(capd) || capd < 1.0) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        cap = (size_t)capd;
        if (cap == 0) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    for (int e = 0; e < E; e++) {
        cnt[e] = 0;
    }
    for (size_t q = 0; q < Tk; q++) {
        cnt[ids[q]]++;
    }
    // expert 境界 (kept のみ)。off[E] == kept 総数。
    off[0] = 0;
    for (int e = 0; e < E; e++) {
        size_t kc = (cnt[e] < cap) ? cnt[e] : cap;
        off[(size_t)e + 1] = off[(size_t)e] + kc;
    }
    kept_total = off[(size_t)E];
    // 安定配置: (token, slot) 昇順走査で kept 枠へ。超過分は drop。
    // cnt をカーソルに転用 (off は境界として保持)。
    for (int e = 0; e < E; e++) {
        cnt[e] = off[(size_t)e];
    }
    for (size_t q = 0; q < Tk; q++) {
        size_t e = ids[q];
        if (cnt[e] < off[e + 1]) {
            perm[cnt[e]] = q;
            cnt[e]++;
            drop[q] = 0;
        } else {
            drop[q] = 1;
        }
    }
    if (out_kept != NULL) {
        *out_kept = kept_total;
    }
    if (out_dropped != NULL) {
        *out_dropped = Tk - kept_total;
    }
    return JT_OK;
}

static int jt_moe_fwd_batch_impl(
    const float *restrict X,
    const float *restrict Wgate,
    const float *restrict Wg, const float *restrict Wu,
    const float *restrict Wd,
    const float *restrict Wg_s, const float *restrict Wu_s,
    const float *restrict Wd_s,
    float *restrict Y,
    int T, int n, int h, int n_experts, int topk, int n_shared,
    size_t *restrict out_ids, float *restrict out_weights,
    float *restrict cache_Gsel, float *restrict cache_Usel,
    float *restrict cache_Ysel,
    float *restrict cache_Gs, float *restrict cache_Us,
    float cap_factor,
    size_t *restrict out_perm, size_t *restrict out_off,
    unsigned char *restrict out_drop,
    size_t *restrict out_kept, size_t *restrict out_dropped,
    int validate) {
    int rc = JT_ERR_INVAL;
    static const int kMaxE = JT_MOE_MAX_EXPERTS;
    static const int kMaxW = JT_BWD_MAX_WIDE;
    float logits[JT_MOE_MAX_EXPERTS];
    float tmpG[JT_BWD_MAX_WIDE];
    float tmpU[JT_BWD_MAX_WIDE];
    float tmpY[JT_BWD_MAX_WIDE];
    size_t Tk = 0;
    size_t kept = 0;
    size_t dropped = 0;
    double *yacc = NULL;
    size_t *perm = NULL;
    size_t *off = NULL;
    unsigned char *drop = NULL;
    float *stage_Ysel = NULL;  // cache_Ysel==NULL 時のみ確保
    // Phase G Step 2: expert 連続ワークスペース ([maxMe][n/h])。
    float *Xe = NULL;
    float *Ye = NULL;
    float *Ge = NULL;
    float *Ue = NULL;
    if (X == NULL || Wgate == NULL || Wg == NULL || Wu == NULL ||
        Wd == NULL || Y == NULL || out_ids == NULL ||
        out_weights == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (T <= 0) {
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
    if (!(cap_factor >= 1.0f) || !(cap_factor <= 1.5f) ||
        !isfinite((double)cap_factor)) {
        errno = EINVAL;
        goto cleanup;
    }
    if (n_experts > kMaxE || n > kMaxW || h > kMaxW) {
        errno = EINVAL;
        goto cleanup;
    }
    if ((size_t)topk > SIZE_MAX / (size_t)T ||
        (size_t)T > SIZE_MAX / (size_t)n) {
        errno = EINVAL;
        goto cleanup;
    }
    Tk = (size_t)T * (size_t)topk;
    if (Tk > SIZE_MAX / (size_t)n) {
        errno = EINVAL;
        goto cleanup;
    }
    // 入力の有限検査 (fail-closed: Y更新前に拒否)。unchecked では省略し、
    // 呼び出し側のステップ冒頭検証＋区間内不変に委ねる (単体版と同一条件)。
    if (validate) {
        size_t e = (size_t)n_experts;
        size_t nn = (size_t)n;
        size_t hh = (size_t)h;
        size_t Tn = (size_t)T * nn;
        if (!jt_moe_all_finite(X, Tn) ||
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
    // 作業域確保 (失敗時は ENOMEM。Y は未更新)。
    yacc = (double *)calloc((size_t)T * (size_t)n, sizeof(double));
    perm = (size_t *)malloc(Tk * sizeof(size_t));
    off = (size_t *)malloc(((size_t)n_experts + 1) * sizeof(size_t));
    drop = (unsigned char *)malloc(Tk * sizeof(unsigned char));
    if (yacc == NULL || perm == NULL || off == NULL || drop == NULL) {
        errno = ENOMEM;
        goto cleanup;
    }
    if (cache_Ysel == NULL) {
        stage_Ysel = (float *)malloc(Tk * (size_t)n * sizeof(float));
        if (stage_Ysel == NULL) {
            errno = ENOMEM;
            goto cleanup;
        }
    }
    // gate logits → 実top-k (トークン毎に単体版と同一核)。
    for (int t = 0; t < T; t++) {
        const float *Xt = X + (size_t)t * (size_t)n;
        size_t *ids = out_ids + (size_t)t * (size_t)topk;
        float *weights = out_weights + (size_t)t * (size_t)topk;
        for (int e = 0; e < n_experts; e++) {
            const float *wr = Wgate + (size_t)e * (size_t)n;
#ifdef __AVX2__
            double acc = jt_moe_avx2_dot(Xt, wr, n);
#else
            double acc = 0.0;
            for (int j = 0; j < n; j++) {
                acc += (double)Xt[j] * (double)wr[j];
            }
#endif
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
        {
            int rrc = jt_routing_topk(logits, (size_t)n_experts,
                                      (size_t)topk, ids, weights);
            if (rrc != JT_OK) {
                goto cleanup;  // errnoは下位で設定済み
            }
        }
        for (int p = 0; p < topk; p++) {
            if (ids[(size_t)p] >= (size_t)n_experts) {
                errno = EINVAL;
                goto cleanup;
            }
            float w = weights[(size_t)p];
            if (!(w >= 0.0f) || !isfinite((double)w)) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    // ソート＋capacity 上限＋超過 drop (範囲検査は内部で再確認)。
    {
        int src = jt_moe_batch_sort(out_ids, T, topk, n_experts,
                                    cap_factor, perm, off, drop, &kept,
                                    &dropped);
        if (src != JT_OK) {
            goto cleanup;  // errnoは下位で設定済み
        }
    }
    // Phase G Step 2: expert 単位バッチ fwd (素朴 GEMM 参照実装)。
    // expert 連続バッファ Xe [M_e][n] に gather し、M 方向に既存
    // jt_swiglu_fwd 核を拡張して expert-outer/M-inner 順 (§2.1) で計算する。
    // ブロッキング・SIMD 新規最適化なし・AVX-512 不使用 (Step 4)。
    // bwd は単体版のまま。各行の計算核と結合の token 順は Step 1 と同一のため
    // drop なし時は単体ループと bit 一致 (AVX2 有無によらず同一核経由)。
    {
        size_t maxMe = 0;
        for (int e = 0; e < n_experts; e++) {
            size_t Me = off[(size_t)e + 1] - off[(size_t)e];
            if (Me > maxMe) {
                maxMe = Me;
            }
        }
        if (maxMe > 0) {
            if (maxMe > SIZE_MAX / (size_t)n ||
                maxMe > SIZE_MAX / (size_t)h) {
                errno = EINVAL;
                goto cleanup;
            }
            Xe = (float *)malloc(maxMe * (size_t)n * sizeof(float));
            Ye = (float *)malloc(maxMe * (size_t)n * sizeof(float));
            Ge = (float *)malloc(maxMe * (size_t)h * sizeof(float));
            Ue = (float *)malloc(maxMe * (size_t)h * sizeof(float));
            if (Xe == NULL || Ye == NULL || Ge == NULL || Ue == NULL) {
                errno = ENOMEM;
                goto cleanup;
            }
        }
        for (int e = 0; e < n_experts; e++) {
            size_t b0 = off[(size_t)e];
            size_t Me = off[(size_t)e + 1] - b0;
            const float *wgr;
            const float *wur;
            const float *wdr;
            if (Me == 0) {
                continue;  // M_e=0 の expert は起動スキップ (§4.2)
            }
            // gather: expert 連続配置 (perm 順。同一 expert 内は token 昇順で安定)。
            for (size_t m = 0; m < Me; m++) {
                size_t q = perm[b0 + m];
                size_t t = q / (size_t)topk;
                memcpy(Xe + m * (size_t)n, X + t * (size_t)n,
                       (size_t)n * sizeof(float));
            }
            wgr = Wg + (size_t)e * (size_t)h * (size_t)n;
            wur = Wu + (size_t)e * (size_t)h * (size_t)n;
            wdr = Wd + (size_t)e * (size_t)h * (size_t)n;
            // M 方向に既存核を拡張 (各行は Step 1 の per-pair 呼出しと同一)。
            for (size_t m = 0; m < Me; m++) {
                int frc =
                    validate
                        ? jt_swiglu_fwd(Xe + m * (size_t)n, wgr, wur, wdr,
                                        Ge + m * (size_t)h,
                                        Ue + m * (size_t)h,
                                        Ye + m * (size_t)n, n, h)
                        : jt_swiglu_fwd_unchecked(
                              Xe + m * (size_t)n, wgr, wur, wdr,
                              Ge + m * (size_t)h, Ue + m * (size_t)h,
                              Ye + m * (size_t)n, n, h);
                if (frc != JT_OK) {
                    goto cleanup;  // errnoは下位で設定済み。Yは未更新。
                }
            }
            // scatter: per-q cache 形式へ復元 (bwd は単体版のまま Step 3 申送り)。
            for (size_t m = 0; m < Me; m++) {
                size_t q = perm[b0 + m];
                if (cache_Gsel != NULL) {
                    memcpy(cache_Gsel + q * (size_t)h, Ge + m * (size_t)h,
                           (size_t)h * sizeof(float));
                }
                if (cache_Usel != NULL) {
                    memcpy(cache_Usel + q * (size_t)h, Ue + m * (size_t)h,
                           (size_t)h * sizeof(float));
                }
                if (cache_Ysel != NULL) {
                    memcpy(cache_Ysel + q * (size_t)n, Ye + m * (size_t)n,
                           (size_t)n * sizeof(float));
                } else {
                    memcpy(stage_Ysel + q * (size_t)n, Ye + m * (size_t)n,
                           (size_t)n * sizeof(float));
                }
            }
        }
    }
    // dropped 対応 cache スロットの 0 埋め (不定値混入防止。
    // bwd 側の drop マスク適用は Step 3 の範囲)。
    for (size_t q = 0; q < Tk; q++) {
        if (drop[q]) {
            if (cache_Gsel != NULL) {
                memset(cache_Gsel + q * (size_t)h, 0,
                       (size_t)h * sizeof(float));
            }
            if (cache_Usel != NULL) {
                memset(cache_Usel + q * (size_t)h, 0,
                       (size_t)h * sizeof(float));
            }
            if (cache_Ysel != NULL) {
                memset(cache_Ysel + q * (size_t)n, 0,
                       (size_t)n * sizeof(float));
            }
        }
    }
    // combine (scatter-add。token 順・slot 順で単体版と同一順序のため、
    // drop なし時は bit 一致)。共有 expert は容量制限対象外で同一に加算。
    for (int t = 0; t < T; t++) {
        double *ya = yacc + (size_t)t * (size_t)n;
        const float *Xt = X + (size_t)t * (size_t)n;
        for (int p = 0; p < topk; p++) {
            size_t q = (size_t)t * (size_t)topk + (size_t)p;
            float w;
            const float *Yp;
            if (drop[q]) {
                continue;  // renormalize しない (§1.2 仕様)
            }
            w = out_weights[q];
            Yp = (cache_Ysel != NULL) ? (cache_Ysel + q * (size_t)n)
                                      : (stage_Ysel + q * (size_t)n);
#ifdef __AVX2__
            jt_moe_avx2_f64_add(ya, (double)w, Yp, n);
#else
            for (int j = 0; j < n; j++) {
                ya[j] += (double)w * (double)Yp[j];
            }
#endif
        }
        for (int s = 0; s < n_shared; s++) {
            const float *wgr =
                Wg_s + (size_t)s * (size_t)h * (size_t)n;
            const float *wur =
                Wu_s + (size_t)s * (size_t)h * (size_t)n;
            const float *wdr =
                Wd_s + (size_t)s * (size_t)h * (size_t)n;
            float *Gp = (cache_Gs != NULL)
                            ? (cache_Gs +
                               ((size_t)t * (size_t)n_shared + (size_t)s) *
                                   (size_t)h)
                            : tmpG;
            float *Up = (cache_Us != NULL)
                            ? (cache_Us +
                               ((size_t)t * (size_t)n_shared + (size_t)s) *
                                   (size_t)h)
                            : tmpU;
            int frc = validate
                          ? jt_swiglu_fwd(Xt, wgr, wur, wdr, Gp, Up, tmpY,
                                          n, h)
                          : jt_swiglu_fwd_unchecked(Xt, wgr, wur, wdr, Gp,
                                                    Up, tmpY, n, h);
            if (frc != JT_OK) {
                goto cleanup;
            }
#ifdef __AVX2__
            jt_moe_avx2_f64_add(ya, 1.0, tmpY, n);
#else
            for (int j = 0; j < n; j++) {
                ya[j] += (double)tmpY[j];
            }
#endif
        }
    }
    for (size_t i = 0; i < (size_t)T * (size_t)n; i++) {
        if (!isfinite(yacc[i])) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    // 全成功後に一括 commit (fail-closed: 拒否時は Y 不変)。
    for (int t = 0; t < T; t++) {
        const double *ya = yacc + (size_t)t * (size_t)n;
        float *Yt = Y + (size_t)t * (size_t)n;
        for (int j = 0; j < n; j++) {
            Yt[j] = (float)ya[j];
        }
    }
    if (out_perm != NULL) {
        for (size_t i = 0; i < kept; i++) {
            out_perm[i] = perm[i];
        }
    }
    if (out_off != NULL) {
        for (int e = 0; e <= n_experts; e++) {
            out_off[(size_t)e] = off[(size_t)e];
        }
    }
    if (out_drop != NULL) {
        for (size_t q = 0; q < Tk; q++) {
            out_drop[q] = drop[q];
        }
    }
    if (out_kept != NULL) {
        *out_kept = kept;
    }
    if (out_dropped != NULL) {
        *out_dropped = dropped;
    }
    rc = JT_OK;
cleanup:
    free(yacc);
    free(perm);
    free(off);
    free(drop);
    free(stage_Ysel);
    free(Xe);
    free(Ye);
    free(Ge);
    free(Ue);
    return rc;
}

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
                     size_t *restrict out_kept, size_t *restrict out_dropped) {
    return jt_moe_fwd_batch_impl(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Y,
                                 T, n, h, n_experts, topk, n_shared,
                                 out_ids, out_weights, cache_Gsel,
                                 cache_Usel, cache_Ysel, cache_Gs, cache_Us,
                                 cap_factor, out_perm, out_off, out_drop,
                                 out_kept, out_dropped, 1);
}

int jt_moe_fwd_batch_unchecked(
    const float *restrict X,
    const float *restrict Wgate,
    const float *restrict Wg, const float *restrict Wu,
    const float *restrict Wd,
    const float *restrict Wg_s, const float *restrict Wu_s,
    const float *restrict Wd_s,
    float *restrict Y,
    int T, int n, int h, int n_experts, int topk, int n_shared,
    size_t *restrict out_ids, float *restrict out_weights,
    float *restrict cache_Gsel, float *restrict cache_Usel,
    float *restrict cache_Ysel,
    float *restrict cache_Gs, float *restrict cache_Us, float cap_factor,
    size_t *restrict out_perm, size_t *restrict out_off,
    unsigned char *restrict out_drop,
    size_t *restrict out_kept, size_t *restrict out_dropped) {
    return jt_moe_fwd_batch_impl(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Y,
                                 T, n, h, n_experts, topk, n_shared,
                                 out_ids, out_weights, cache_Gsel,
                                 cache_Usel, cache_Ysel, cache_Gs, cache_Us,
                                 cap_factor, out_perm, out_off, out_drop,
                                 out_kept, out_dropped, 0);
}

int jt_moe_sticky_seq_loss(const float *restrict gates, size_t T, size_t n,
                           float lambda, float alpha, size_t w,
                           float *restrict out_loss) {    int rc = JT_ERR_INVAL;
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
