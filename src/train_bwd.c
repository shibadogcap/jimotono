// jt_train_bwd: GDN-2デコード逆伝播 + SwiGLU融合逆伝播 + RMSNorm逆伝播。
// AGENTS.MD 7.1: C11, restrict, errnoベース + goto cleanup, クロスプラット。
// DESIGN.MD §6: 1誤差伝播20Mサイクル以下・逆伝播12M以下 (本P2足場は正しさ
// 優先のスカラー核。FLOP見積りはヘッダ/bench側の対応メモ参照)。
//
// GDN-2 backward導出メモ (forwardはtrain_bwd.h再掲):
//   G[i,j] = dS_next[i,j] + qn[i]*dO[j]   … Snへの全勾配 (o経路+未来経路)
//   dqn[i] = sum_j Sn[i,j]*dO[j]
//   dvw[j] = sum_i G[i,j]*kn[i] → dw=dvw*v, dv=dvw*w
//   dS2 = G
//   dc[j] = -sum_i G[i,j]*kn[i]
//   dke[p] = sum_j dc[j]*S1[p,j] → db=dke*kn, dkn_ke=dke*b
//   dkn_sn[i] = sum_j G[i,j]*vw[j]
//   dkn_s2[i] = -sum_j G[i,j]*c[j]
//   dkn[norm空間] = dkn_sn + dkn_s2 + dkn_ke
//   dS1[p,j] = G[p,j] + dc[j]*ke[p]
//   dSp = dS1*alpha, dAlpha[p] = sum_j dS1[p,j]*Sp[p,j]
//   q/kはL2正規化を経由: dx = (dy - xh*(xh·dy))/norm (norm<epsは0)。

#include "jimotono/train_bwd.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int jt_bwd_valid_dims(int dk, int dv) {
    return dk > 0 && dv > 0 && dk <= JT_BWD_MAX_D && dv <= JT_BWD_MAX_D;
}

static int jt_bwd_all_finite(const float *restrict x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)x[i])) {
            return 0;
        }
    }
    return 1;
}

int jt_gdn2_decode_bwd_scratch_floats(int dk, int dv,
                                      size_t *restrict out_n) {
    if (out_n == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_n = 0;
    if (!jt_bwd_valid_dims(dk, dv)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    {
        size_t dd = (size_t)(unsigned int)dk * (size_t)(unsigned int)dv;
        // 2*dk + 2*dv + 2*dk*dv。size_tオーバーフロー番兵。
        if (dd > (SIZE_MAX - (size_t)2 * (size_t)(unsigned int)dk -
                   (size_t)2 * (size_t)(unsigned int)dv) /
                      (size_t)2) {
            errno = ENOMEM;
            return JT_ERR_NOMEM;
        }
        *out_n = (size_t)2 * (size_t)(unsigned int)dk +
                 (size_t)2 * (size_t)(unsigned int)dv + (size_t)2 * dd;
    }
    return JT_OK;
}

int jt_gdn2_decode_bwd(const float *restrict S_prev,
                       const float *restrict q, const float *restrict k,
                       const float *restrict v, const float *restrict b,
                       const float *restrict w, const float *restrict alpha,
                       const float *restrict dO, const float *restrict dS_next,
                       float *restrict dQ, float *restrict dK,
                       float *restrict dV, float *restrict dB,
                       float *restrict dW, float *restrict dAlpha,
                       float *restrict dS_prev, int dk, int dv,
                       float *restrict scratch, size_t scratch_n) {
    int rc = JT_ERR_INVAL;
    size_t need = 0;

    if (S_prev == NULL || q == NULL || k == NULL || v == NULL || b == NULL ||
        w == NULL || alpha == NULL || dO == NULL || dS_next == NULL ||
        dQ == NULL || dK == NULL || dV == NULL || dB == NULL || dW == NULL ||
        dAlpha == NULL || dS_prev == NULL || scratch == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_bwd_valid_dims(dk, dv)) {
        errno = EINVAL;
        goto cleanup;
    }
    if (((uintptr_t)(const void *)S_prev % (uintptr_t)JT_CACHELINE) != 0) {
        errno = EINVAL;
        rc = JT_ERR_ALIGN;
        goto cleanup;
    }
    if (jt_gdn2_decode_bwd_scratch_floats(dk, dv, &need) != JT_OK) {
        goto cleanup;  // errnoは下位で設定済み
    }
    if (scratch_n < need) {
        errno = EINVAL;
        goto cleanup;
    }
    // fail-closed: 非有限・alpha範囲外は出力不変で拒否。
    if (!jt_bwd_all_finite(q, (size_t)dk) ||
        !jt_bwd_all_finite(k, (size_t)dk) ||
        !jt_bwd_all_finite(b, (size_t)dk) ||
        !jt_bwd_all_finite(alpha, (size_t)dk) ||
        !jt_bwd_all_finite(v, (size_t)dv) ||
        !jt_bwd_all_finite(w, (size_t)dv) ||
        !jt_bwd_all_finite(dO, (size_t)dv) ||
        !jt_bwd_all_finite(dS_next,
                           (size_t)(unsigned int)dk *
                               (size_t)(unsigned int)dv) ||
        !jt_bwd_all_finite(S_prev, (size_t)(unsigned int)dk *
                                       (size_t)(unsigned int)dv)) {
        errno = EINVAL;
        goto cleanup;
    }
    for (int i = 0; i < dk; i++) {
        float ai = alpha[i];
        if (!(ai >= 0.0f && ai <= 1.0f)) {
            errno = EINVAL;
            goto cleanup;
        }
    }

    {
        const float eps = JT_BWD_EPS_DEFAULT;
        float *qn = scratch;
        float *kn = scratch + (size_t)dk;
        float *c = scratch + (size_t)2 * (size_t)dk;
        float *vw = c + (size_t)dv;
        float *S1 = vw + (size_t)dv;
        float *Sn = S1 + (size_t)(unsigned int)dk * (size_t)(unsigned int)dv;
        size_t dd = (size_t)(unsigned int)dk * (size_t)(unsigned int)dv;
        double nq = 0.0;
        double nk = 0.0;
        (void)memset(c, 0, (size_t)dv * sizeof(float));

        // 0) 正規化係数 (forwardと同一式。ゼロ割ガード)。
        for (int i = 0; i < dk; i++) {
            nq += (double)q[i] * (double)q[i];
            nk += (double)k[i] * (double)k[i];
        }
        nq = sqrt(nq);
        nk = sqrt(nk);
        if (nq < (double)eps) {
            for (int i = 0; i < dk; i++) {
                qn[i] = 0.0f;
            }
        } else {
            float inv = (float)(1.0 / nq);
            for (int i = 0; i < dk; i++) {
                qn[i] = q[i] * inv;
            }
        }
        if (nk < (double)eps) {
            for (int i = 0; i < dk; i++) {
                kn[i] = 0.0f;
            }
        } else {
            float inv = (float)(1.0 / nk);
            for (int i = 0; i < dk; i++) {
                kn[i] = k[i] * inv;
            }
        }

        // 1) forward再計算: S1 = D S_prev。
        for (int i = 0; i < dk; i++) {
            float ai = alpha[i];
            const float *sp = S_prev + (size_t)i * (size_t)dv;
            float *s1 = S1 + (size_t)i * (size_t)dv;
            for (int j = 0; j < dv; j++) {
                s1[j] = ai * sp[j];
            }
        }
        // 2) c^T = ke^T S1, ke = b⊙kn。
        for (int j = 0; j < dv; j++) {
            c[j] = 0.0f;
        }
        for (int i = 0; i < dk; i++) {
            double ke = (double)b[i] * (double)kn[i];
            const float *s1 = S1 + (size_t)i * (size_t)dv;
            for (int j = 0; j < dv; j++) {
                c[j] = (float)((double)c[j] + ke * (double)s1[j]);
            }
        }
        // 3) vw = w⊙v。
        for (int j = 0; j < dv; j++) {
            vw[j] = w[j] * v[j];
        }
        // 4) Sn = S1 - kn c^T + kn vw^T。
        for (int i = 0; i < dk; i++) {
            float ki = kn[i];
            const float *s1 = S1 + (size_t)i * (size_t)dv;
            float *sn = Sn + (size_t)i * (size_t)dv;
            for (int j = 0; j < dv; j++) {
                sn[j] = s1[j] - ki * c[j] + ki * vw[j];
            }
        }

        // 5) G = dS_next + qn dO^T、dc、dqn/dkn_sn/dkn_s2/dvwをdouble累積。
        // 中間GはSnバッファ…ではなく別途必要だが、メモリ節約のため
        // GをS1バッファへ上書きせず、SnをGへ転用後に再利用する。
        // 単純化のためここではスタック固定ではなくscratch末尾…ではなく
        // SnをGに上書きする前にdqn用Snを先に消費する順序にする。
        {
            // dqn[i] = sum_j Sn[i,j]*dO[j] (Sn上書き前に計算)。
            double *dqn_acc = NULL;
            // 可変長配列を避け、scratchとは別に小さなスタック上限で処理。
            // dk<=1024のため固定1024の自動配列で十分。
            double dqn_tmp[JT_BWD_MAX_D];
            double dkn_sn[JT_BWD_MAX_D];
            double dkn_s2[JT_BWD_MAX_D];
            double dc_tmp[JT_BWD_MAX_D];
            double dvw_tmp[JT_BWD_MAX_D];
            (void)dqn_acc;
            for (int i = 0; i < dk; i++) {
                dqn_tmp[i] = 0.0;
                dkn_sn[i] = 0.0;
                dkn_s2[i] = 0.0;
            }
            for (int j = 0; j < dv; j++) {
                dc_tmp[j] = 0.0;
                dvw_tmp[j] = 0.0;
            }
            // dqn。
            for (int i = 0; i < dk; i++) {
                const float *sn = Sn + (size_t)i * (size_t)dv;
                double acc = 0.0;
                for (int j = 0; j < dv; j++) {
                    acc += (double)sn[j] * (double)dO[j];
                }
                dqn_tmp[i] = acc;
            }
            // G[i,j]をSn位置に上書き: G = dS_next + qn*dO^T。
            for (int i = 0; i < dk; i++) {
                float qi = qn[i];
                const float *dsn =
                    dS_next + (size_t)i * (size_t)dv;
                float *g = Sn + (size_t)i * (size_t)dv;  // Gで再利用
                for (int j = 0; j < dv; j++) {
                    g[j] = (float)((double)dsn[j] + (double)qi * (double)dO[j]);
                }
            }
            // dc[j] = -sum_i G[i,j]*kn[i]; dvw[j] = sum_i G[i,j]*kn[i]。
            for (int j = 0; j < dv; j++) {
                double s_dc = 0.0;
                double s_vw = 0.0;
                for (int i = 0; i < dk; i++) {
                    double g =
                        (double)(Sn[(size_t)i * (size_t)dv + (size_t)j]);
                    s_dc += g * (double)kn[i];
                    s_vw += g * (double)kn[i];
                }
                dc_tmp[j] = -s_dc;
                dvw_tmp[j] = s_vw;
            }
            // dkn_sn[i] = sum_j G[i,j]*vw[j]; dkn_s2[i] = -sum_j G[i,j]*c[j]。
            for (int i = 0; i < dk; i++) {
                const float *g = Sn + (size_t)i * (size_t)dv;
                double a_sn = 0.0;
                double a_s2 = 0.0;
                for (int j = 0; j < dv; j++) {
                    a_sn += (double)g[j] * (double)vw[j];
                    a_s2 += (double)g[j] * (double)c[j];
                }
                dkn_sn[i] = a_sn;
                dkn_s2[i] = -a_s2;
            }
            // dke[p] = sum_j dc[j]*S1[p,j] → dB, dkn_ke。
            // dS1 = G + dc*ke; dS_prev = dS1*alpha; dAlpha = sum dS1*Sp。
            for (int i = 0; i < dk; i++) {
                const float *s1 = S1 + (size_t)i * (size_t)dv;
                const float *sp = S_prev + (size_t)i * (size_t)dv;
                const float *g = Sn + (size_t)i * (size_t)dv;
                double ke = (double)b[i] * (double)kn[i];
                double dke = 0.0;
                for (int j = 0; j < dv; j++) {
                    dke += dc_tmp[j] * (double)s1[j];
                }
                dB[i] = (float)(dke * (double)kn[i]);
                {
                    double dkn_tot =
                        dkn_sn[i] + dkn_s2[i] + dke * (double)b[i];
                    // k正規化逆伝播: nk<epsは0 (forwardゼロガードに整合)。
                    if (nk < (double)eps) {
                        dK[i] = 0.0f;
                    } else {
                        // dot = kn·dkn。
                        // 2パス目でdotが必要なため先にdotを別ループで求める。
                        // ここではdkn_totを一時的にdKへ退避する。
                        dK[i] = (float)dkn_tot;
                    }
                }
                {
                    double da = 0.0;
                    float *dsp = dS_prev + (size_t)i * (size_t)dv;
                    for (int j = 0; j < dv; j++) {
                        double dS1 =
                            (double)g[j] + dc_tmp[j] * ke;
                        dsp[j] = (float)(dS1 * (double)alpha[i]);
                        da += dS1 * (double)sp[j];
                    }
                    dAlpha[i] = (float)da;
                }
                // dqn/dknの正規化逆伝播用にdke等は不要になったので、
                // dkn_s2をdkn_ke保持に転用しない (上式で合算済み)。
                (void)ke;
            }
            // k正規化逆伝播のdot補正: dot = sum kn*dkn_tot。
            if (nk >= (double)eps) {
                double dot = 0.0;
                for (int i = 0; i < dk; i++) {
                    dot += (double)kn[i] * (double)dK[i];
                }
                for (int i = 0; i < dk; i++) {
                    double dkn_tot = (double)dK[i];
                    double dq_raw =
                        (dkn_tot - (double)kn[i] * dot) / nk;
                    dK[i] = (float)dq_raw;
                }
            }
            // q正規化逆伝播。
            if (nq < (double)eps) {
                for (int i = 0; i < dk; i++) {
                    dQ[i] = 0.0f;
                }
            } else {
                double dot = 0.0;
                for (int i = 0; i < dk; i++) {
                    dot += (double)qn[i] * dqn_tmp[i];
                }
                for (int i = 0; i < dk; i++) {
                    double vq = (dqn_tmp[i] - (double)qn[i] * dot) / nq;
                    dQ[i] = (float)vq;
                }
            }
            // dv/dw。
            for (int j = 0; j < dv; j++) {
                dW[j] = (float)(dvw_tmp[j] * (double)v[j]);
                dV[j] = (float)(dvw_tmp[j] * (double)w[j]);
            }
            (void)dd;
        }
    }

    rc = JT_OK;
cleanup:
    return rc;
}

// sigmoid (double精度で評価しfloatで返す)。
static double jt_bwd_sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

static int jt_swiglu_bwd_impl(const float *restrict dY,
                                const float *restrict X,
                                const float *restrict G,
                                const float *restrict U,
                                const float *restrict Wd,
                                const float *restrict Wg,
                                const float *restrict Wu, float *restrict dX,
                                float *restrict dWg, float *restrict dWu,
                                float *restrict dWd, int n, int h,
                                int validate) {
    int rc = JT_ERR_INVAL;
    if (dY == NULL || X == NULL || G == NULL || U == NULL || Wd == NULL ||
        Wg == NULL || Wu == NULL || dX == NULL || dWg == NULL ||
        dWu == NULL || dWd == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (n <= 0 || h <= 0 || n > JT_BWD_MAX_WIDE || h > JT_BWD_MAX_WIDE) {
        errno = EINVAL;
        goto cleanup;
    }
    // 入力有限プリスキャン (O(n)+O(h*n)) はvalidate時のみ。uncheckedでは
    // 呼び出し側の事前検証＋区間内不変に委ね、ここでは省略する。
    if (validate) {
        if (!jt_bwd_all_finite(dY, (size_t)n) ||
            !jt_bwd_all_finite(X, (size_t)n) ||
            !jt_bwd_all_finite(G, (size_t)h) ||
            !jt_bwd_all_finite(U, (size_t)h) ||
            !jt_bwd_all_finite(Wd, (size_t)h * (size_t)n) ||
            !jt_bwd_all_finite(Wg, (size_t)h * (size_t)n) ||
            !jt_bwd_all_finite(Wu, (size_t)h * (size_t)n)) {
            errno = EINVAL;
            goto cleanup;
        }
    }

    {
        // 中間 dS/pre-activation勾配はh<=4096の自動配列で処理
        // (ヒープ確保を避け、1C1Tスタック利用を想定)。
        // C11 VLAは使わない (MSVC非対応のため固定上限+検査)。
        static const int kMax = JT_BWD_MAX_WIDE;
        double dg[JT_BWD_MAX_WIDE];
        double du[JT_BWD_MAX_WIDE];
        // s[i] = silu(G[i])*U[i] の保存域。下のdWループでsigmoid再計算を
        // 省くための使い回し (同一入力→同一ビットのため数値は不変)。
        double ss[JT_BWD_MAX_WIDE];
        if (h > kMax) {
            errno = EINVAL;
            goto cleanup;
        }
        for (int i = 0; i < h; i++) {
            double gi = (double)G[i];
            double ui = (double)U[i];
            double sig = jt_bwd_sigmoid(gi);
            double silu = gi * sig;
            // silu'(g) = sig*(1+g*(1-sig))。
            double dsilu = sig * (1.0 + gi * (1.0 - sig));
            const float *wdrow = Wd + (size_t)i * (size_t)n;
            double acc = 0.0;
            for (int j = 0; j < n; j++) {
                acc += (double)dY[j] * (double)wdrow[j];
            }
            dg[i] = acc * ui * dsilu;
            du[i] = acc * silu;
            ss[i] = silu * ui;
            (void)silu;
        }
        // DESIGN.MD §4.1: dXを先に計算して伝播させ、dWは後回し。
        // dW計算→更新→書き戻しのオーバーラップ前提の順序。
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int i = 0; i < h; i++) {
                acc += dg[i] * (double)Wg[(size_t)i * (size_t)n + (size_t)j];
                acc += du[i] * (double)Wu[(size_t)i * (size_t)n + (size_t)j];
            }
            dX[j] = (float)acc;
        }
        // dWは後回し (上記dX転送とパイプライン化可能)。
        // dWd[i,j] = s[i]*dY[j] (sは上記ループで保存した中間値の使い回し。
        // sigmoid再計算を省く。同一入力の再評価のためビット不変)。
        for (int i = 0; i < h; i++) {
            float *dwg = dWg + (size_t)i * (size_t)n;
            float *dwu = dWu + (size_t)i * (size_t)n;
            float *dwd = dWd + (size_t)i * (size_t)n;
            double gi_d = dg[i];
            double ui_d = du[i];
            double s = ss[i];
            for (int j = 0; j < n; j++) {
                double xj = (double)X[j];
                dwg[j] = (float)(gi_d * xj);
                dwu[j] = (float)(ui_d * xj);
                dwd[j] = (float)(s * (double)dY[j]);
            }
        }
    }

    rc = JT_OK;
cleanup:
    return rc;
}

// 公開API (検証あり、従来通りfail-closed)。
int jt_swiglu_bwd(const float *restrict dY, const float *restrict X,
                  const float *restrict G, const float *restrict U,
                  const float *restrict Wd, const float *restrict Wg,
                  const float *restrict Wu, float *restrict dX,
                  float *restrict dWg, float *restrict dWu,
                  float *restrict dWd, int n, int h) {
    return jt_swiglu_bwd_impl(dY, X, G, U, Wd, Wg, Wu, dX, dWg, dWu, dWd,
                              n, h, 1);
}

// 内部高速経路 (入力有限スキャンなし。ヘッダの使用条件コメント参照)。
int jt_swiglu_bwd_unchecked(const float *restrict dY,
                            const float *restrict X,
                            const float *restrict G, const float *restrict U,
                            const float *restrict Wd, const float *restrict Wg,
                            const float *restrict Wu, float *restrict dX,
                            float *restrict dWg, float *restrict dWu,
                            float *restrict dWd, int n, int h) {
    return jt_swiglu_bwd_impl(dY, X, G, U, Wd, Wg, Wu, dX, dWg, dWu, dWd,
                              n, h, 0);
}

int jt_rmsnorm_bwd(const float *restrict dY, const float *restrict X,
                   const float *restrict W, float *restrict dX,
                   float *restrict dW, int n, float eps) {
    int rc = JT_ERR_INVAL;
    if (dY == NULL || X == NULL || W == NULL || dX == NULL || dW == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (n <= 0 || n > JT_BWD_MAX_WIDE) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!(eps > 0.0f) || !isfinite((double)eps)) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_bwd_all_finite(dY, (size_t)n) ||
        !jt_bwd_all_finite(X, (size_t)n) ||
        !jt_bwd_all_finite(W, (size_t)n)) {
        errno = EINVAL;
        goto cleanup;
    }

    {
        double mean = 0.0;
        double r = 0.0;
        for (int i = 0; i < n; i++) {
            mean += (double)X[i] * (double)X[i];
        }
        mean /= (double)n;
        r = 1.0 / sqrt(mean + (double)eps);
        for (int i = 0; i < n; i++) {
            dW[i] = (float)((double)dY[i] * (double)X[i] * r);
        }
        {
            double r3n = r * r * r / (double)n;
            double s = 0.0;
            for (int i = 0; i < n; i++) {
                s += (double)dY[i] * (double)W[i] * (double)X[i];
            }
            for (int j = 0; j < n; j++) {
                double v =
                    r * (double)dY[j] * (double)W[j] - r3n * (double)X[j] * s;
                dX[j] = (float)v;
            }
        }
    }

    rc = JT_OK;
cleanup:
    return rc;
}

// SwiGLU順伝播 (bwd前提式と同一。double累積 + double sigmoid)。
static int jt_swiglu_fwd_impl(const float *restrict X,
                              const float *restrict Wg,
                              const float *restrict Wu, const float *restrict Wd,
                              float *restrict G, float *restrict U,
                              float *restrict Y, int n, int h, int validate) {
    int rc = JT_ERR_INVAL;
    if (X == NULL || Wg == NULL || Wu == NULL || Wd == NULL || G == NULL ||
        U == NULL || Y == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (n <= 0 || h <= 0 || n > JT_BWD_MAX_WIDE || h > JT_BWD_MAX_WIDE) {
        errno = EINVAL;
        goto cleanup;
    }
    // 入力有限プリスキャン (O(n)+O(h*n)) はvalidate時のみ。uncheckedでは
    // 呼び出し側の事前検証＋区間内不変に委ね、ここでは省略する。
    if (validate) {
        if (!jt_bwd_all_finite(X, (size_t)n) ||
            !jt_bwd_all_finite(Wg, (size_t)h * (size_t)n) ||
            !jt_bwd_all_finite(Wu, (size_t)h * (size_t)n) ||
            !jt_bwd_all_finite(Wd, (size_t)h * (size_t)n)) {
            errno = EINVAL;
            goto cleanup;
        }
    }

    {
        // G/U/Yは検証通過後のみ書き込む (fail-closed: 拒否時は出力不変)。
        // h<=4096のため固定上限の自動配列で中間sを保持する。
        // G/Uも途中書き込みせず一時配列に退避し、全要素の有限確認後に
        // 一括commitする。
        static const int kMax = JT_BWD_MAX_WIDE;
        double s[JT_BWD_MAX_WIDE];
        double gg[JT_BWD_MAX_WIDE];
        double uu[JT_BWD_MAX_WIDE];
        double yacc[JT_BWD_MAX_WIDE];
        if (h > kMax || n > kMax) {
            errno = EINVAL;
            goto cleanup;
        }
        for (int i = 0; i < h; i++) {
            double g = 0.0;
            double u = 0.0;
            const float *wgr = Wg + (size_t)i * (size_t)n;
            const float *wur = Wu + (size_t)i * (size_t)n;
            for (int j = 0; j < n; j++) {
                g += (double)X[j] * (double)wgr[j];
                u += (double)X[j] * (double)wur[j];
            }
            if (!isfinite(g) || !isfinite(u)) {
                errno = EINVAL;
                goto cleanup;
            }
            {
                double sig = jt_bwd_sigmoid(g);
                double silu = g * sig;
                s[i] = silu * u;
                gg[i] = g;
                uu[i] = u;
            }
        }
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int i = 0; i < h; i++) {
                acc += s[i] * (double)Wd[(size_t)i * (size_t)n + (size_t)j];
            }
            if (!isfinite(acc)) {
                errno = EINVAL;
                goto cleanup;
            }
            yacc[j] = acc;
        }
        // 全要素の有限確認後に一括commit (拒否時はG/U/Y不変)。
        for (int i = 0; i < h; i++) {
            if (!isfinite(s[i]) || !isfinite(gg[i]) || !isfinite(uu[i])) {
                errno = EINVAL;
                goto cleanup;
            }
        }
        for (int i = 0; i < h; i++) {
            G[i] = (float)gg[i];
            U[i] = (float)uu[i];
        }
        for (int j = 0; j < n; j++) {
            Y[j] = (float)yacc[j];
        }
    }

    rc = JT_OK;
cleanup:
    return rc;
}

// 公開API (検証あり、従来通りfail-closed)。
int jt_swiglu_fwd(const float *restrict X, const float *restrict Wg,
                  const float *restrict Wu, const float *restrict Wd,
                  float *restrict G, float *restrict U,
                  float *restrict Y, int n, int h) {
    return jt_swiglu_fwd_impl(X, Wg, Wu, Wd, G, U, Y, n, h, 1);
}

// 内部高速経路 (入力有限スキャンなし。ヘッダの使用条件コメント参照)。
int jt_swiglu_fwd_unchecked(const float *restrict X,
                            const float *restrict Wg,
                            const float *restrict Wu, const float *restrict Wd,
                            float *restrict G, float *restrict U,
                            float *restrict Y, int n, int h) {
    return jt_swiglu_fwd_impl(X, Wg, Wu, Wd, G, U, Y, n, h, 0);
}

// RMSNorm順伝播 (bwd前提式と同一。double累積)。
int jt_rmsnorm_fwd(const float *restrict X, const float *restrict W,
                   float *restrict Y, int n, float eps) {
    int rc = JT_ERR_INVAL;
    if (X == NULL || W == NULL || Y == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (n <= 0 || n > JT_BWD_MAX_WIDE) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!(eps > 0.0f) || !isfinite((double)eps)) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_bwd_all_finite(X, (size_t)n) ||
        !jt_bwd_all_finite(W, (size_t)n)) {
        errno = EINVAL;
        goto cleanup;
    }

    {
        double mean = 0.0;
        double r = 0.0;
        double yacc[JT_BWD_MAX_WIDE];
        for (int i = 0; i < n; i++) {
            mean += (double)X[i] * (double)X[i];
        }
        mean /= (double)n;
        r = 1.0 / sqrt(mean + (double)eps);
        if (!isfinite(r)) {
            errno = EINVAL;
            goto cleanup;
        }
        for (int i = 0; i < n; i++) {
            double v = (double)W[i] * (double)X[i] * r;
            if (!isfinite(v)) {
                errno = EINVAL;
                goto cleanup;
            }
            yacc[i] = v;
        }
        // 有限確認後に一括commit (拒否時はY不変)。
        for (int i = 0; i < n; i++) {
            Y[i] = (float)yacc[i];
        }
    }

    rc = JT_OK;
cleanup:
    return rc;
}
