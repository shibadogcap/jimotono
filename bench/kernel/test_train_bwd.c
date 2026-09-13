// test_train_bwd: backward核の数値勾配チェック + checkpoint足場検証。
// 方針: スカラー損失 L に対するcentral finite-diffと解析勾配をtol=1e-3で比較。
//   GDN-2: L = dot(o,dO) + sum(Sn*dS_next) (dO/dS_nextは定数)。
//   SwiGLU: L = dot(Y,dY)。G/UはX,Wから再計算するforwardと対照。
//   RMSNorm: L = dot(Y,dY)。
// 小規模 (dk/dv=4/8、n/h=4) で実施。成功時exit 0 + "train_bwd: OK"。
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/checkpoint.h"
#include "jimotono/common.h"
#include "jimotono/gdn2.h"
#include "jimotono/train_bwd.h"

static int g_fail = 0;

#define CHECK(cond, ...)                                                  \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);                \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
            g_fail = 1;                                                        \
        }                                                                      \
    } while (0)

static float fdval(int i, int j, int s) {
    double v = (double)(((i * 31 + j * 17 + s * 13) % 11) - 5) / 5.0;
    return (float)v;
}

// ---- double参照forward ----

static void ref_l2(const double *x, double *y, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        acc += x[i] * x[i];
    }
    double norm = sqrt(acc);
    if (norm < 1e-6) {
        for (int i = 0; i < n; i++) {
            y[i] = 0.0;
        }
        return;
    }
    for (int i = 0; i < n; i++) {
        y[i] = x[i] / norm;
    }
}

// GDN-2 forward (double)。Sは[dk][dv]入出力、oは[dv]出力。
static void ref_gdn2_fwd(double *S, double *o, const double *q,
                         const double *k, const double *v, const double *b,
                         const double *w, const double *alpha, int dk,
                         int dv) {
    double qn[1024];
    double kn[1024];
    double c[1024];
    double vw[1024];
    ref_l2(q, qn, dk);
    ref_l2(k, kn, dk);
    for (int i = 0; i < dk; i++) {
        for (int j = 0; j < dv; j++) {
            S[i * dv + j] *= alpha[i];
        }
    }
    for (int j = 0; j < dv; j++) {
        c[j] = 0.0;
        for (int i = 0; i < dk; i++) {
            c[j] += (b[i] * kn[i]) * S[i * dv + j];
        }
    }
    for (int i = 0; i < dk; i++) {
        for (int j = 0; j < dv; j++) {
            S[i * dv + j] -= kn[i] * c[j];
        }
    }
    for (int j = 0; j < dv; j++) {
        vw[j] = w[j] * v[j];
    }
    for (int i = 0; i < dk; i++) {
        for (int j = 0; j < dv; j++) {
            S[i * dv + j] += kn[i] * vw[j];
        }
    }
    for (int j = 0; j < dv; j++) {
        o[j] = 0.0;
        for (int i = 0; i < dk; i++) {
            o[j] += qn[i] * S[i * dv + j];
        }
    }
}

static double ref_sig(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// SwiGLU forward (double): G=XWg, U=XWu, s=silu(G)*U, Y=sWd。
static void ref_swiglu_fwd(const double *X, const double *Wg,
                           const double *Wu, const double *Wd, double *Y,
                           int n, int h) {
    double s[4096];
    for (int i = 0; i < h; i++) {
        double g = 0.0;
        double u = 0.0;
        for (int j = 0; j < n; j++) {
            g += X[j] * Wg[i * n + j];
            u += X[j] * Wu[i * n + j];
        }
        s[i] = g * ref_sig(g) * u;
    }
    for (int j = 0; j < n; j++) {
        double acc = 0.0;
        for (int i = 0; i < h; i++) {
            acc += s[i] * Wd[i * n + j];
        }
        Y[j] = acc;
    }
}

static void ref_rms_fwd(const double *X, const double *W, double *Y, int n,
                        double eps) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        mean += X[i] * X[i];
    }
    mean /= (double)n;
    {
        double r = 1.0 / sqrt(mean + eps);
        for (int i = 0; i < n; i++) {
            Y[i] = W[i] * X[i] * r;
        }
    }
}

static double gdn2_loss_fn(double *rS, double *ro, const double *rSp,
                          const double *rq, const double *rk,
                          const double *rv, const double *rb,
                          const double *rw, const double *ra,
                          const float *dO, const float *dSn, int dk,
                          int dv) {
    for (int i = 0; i < dk * dv; i++) {
        rS[i] = rSp[i];
    }
    ref_gdn2_fwd(rS, ro, rq, rk, rv, rb, rw, ra, dk, dv);
    double L = 0.0;
    for (int j = 0; j < dv; j++) {
        L += ro[j] * (double)dO[j];
    }
    for (int i = 0; i < dk * dv; i++) {
        L += rS[i] * (double)dSn[i];
    }
    return L;
}

static double sw_loss_fn(const double *rX, const double *rWg,
                         const double *rWu, const double *rWd,
                         const float *dY, int n, int h) {
    double rY[JT_BWD_MAX_WIDE];
    ref_swiglu_fwd(rX, rWg, rWu, rWd, rY, n, h);
    double L = 0.0;
    for (int j = 0; j < n; j++) {
        L += rY[j] * (double)dY[j];
    }
    return L;
}

static double rms_loss_fn(const double *rX, const double *rW,
                          const float *dY, int n, double eps) {
    double rY[JT_BWD_MAX_WIDE];
    ref_rms_fwd(rX, rW, rY, n, eps);
    double L = 0.0;
    for (int i = 0; i < n; i++) {
        L += rY[i] * (double)dY[i];
    }
    return L;
}

static int grad_close(float analytic, double numeric, double tol) {
    double d = fabs((double)analytic - numeric);
    double allowed = tol + tol * fabs(numeric);
    return d <= allowed;
}

// ---- GDN-2勾配チェック ----

static void test_gdn2_grad(int dk, int dv) {
    const double h = 1e-3;
    const double tol = 1e-3;
    float *Sp = NULL;
    CHECK(jt_gdn2_state_alloc(&Sp, dk, dv) == JT_OK && Sp != NULL,
          "gdn2 grad alloc dk=%d dv=%d", dk, dv);
    if (Sp == NULL) {
        return;
    }
    size_t need = 0;
    CHECK(jt_gdn2_decode_bwd_scratch_floats(dk, dv, &need) == JT_OK,
          "bwd scratch size");
    float *scratch = (float *)malloc(need * sizeof(float));
    CHECK(scratch != NULL, "bwd scratch malloc");
    if (scratch == NULL) {
        jt_gdn2_state_free(Sp);
        return;
    }
    // forward入力 (決定論的)。
    float *q = malloc((size_t)dk * sizeof(float));
    float *k = malloc((size_t)dk * sizeof(float));
    float *b = malloc((size_t)dk * sizeof(float));
    float *al = malloc((size_t)dk * sizeof(float));
    float *v = malloc((size_t)dv * sizeof(float));
    float *w = malloc((size_t)dv * sizeof(float));
    float *dO = malloc((size_t)dv * sizeof(float));
    float *dSn = malloc((size_t)dk * (size_t)dv * sizeof(float));
    float *dQ = malloc((size_t)dk * sizeof(float));
    float *dK = malloc((size_t)dk * sizeof(float));
    float *dB = malloc((size_t)dk * sizeof(float));
    float *dA = malloc((size_t)dk * sizeof(float));
    float *dV = malloc((size_t)dv * sizeof(float));
    float *dW = malloc((size_t)dv * sizeof(float));
    float *dSp = malloc((size_t)dk * (size_t)dv * sizeof(float));
    CHECK(q && k && b && al && v && w && dO && dSn && dQ && dK && dB && dA &&
              dV && dW && dSp,
          "gdn2 grad malloc");
    if (!(q && k && b && al && v && w && dO && dSn && dQ && dK && dB && dA &&
          dV && dW && dSp)) {
        goto done;
    }
    for (int i = 0; i < dk; i++) {
        q[i] = fdval(i, 0, 11) + 0.37f;
        k[i] = fdval(i, 1, 12) - 0.23f;
        b[i] = 0.5f + 0.1f * fdval(i, 2, 13);
        al[i] = 0.9f + 0.05f * fdval(i, 3, 14);
    }
    for (int j = 0; j < dv; j++) {
        v[j] = fdval(0, j, 15) + 0.11f;
        w[j] = 0.6f + 0.1f * fdval(1, j, 16);
        dO[j] = 0.3f * fdval(2, j, 17) + 0.1f;
    }
    for (int i = 0; i < dk; i++) {
        for (int j = 0; j < dv; j++) {
            Sp[i * dv + j] = 0.25f * fdval(i, j, 18);
            dSn[i * dv + j] = 0.2f * fdval(i, j, 19) + 0.05f;
        }
    }
    {
        int rc = jt_gdn2_decode_bwd(Sp, q, k, v, b, w, al, dO, dSn, dQ, dK,
                                    dV, dB, dW, dA, dSp, dk, dv, scratch,
                                    need);
        CHECK(rc == JT_OK, "gdn2_bwd rc=%d dk=%d dv=%d", rc, dk, dv);
        if (rc != JT_OK) {
            goto done;
        }
    }
    // double側バッファ。
    {
        double *rSp = malloc((size_t)dk * (size_t)dv * sizeof(double));
        double *rS = malloc((size_t)dk * (size_t)dv * sizeof(double));
        double *ro = malloc((size_t)dv * sizeof(double));
        double *rq = malloc((size_t)dk * sizeof(double));
        double *rk = malloc((size_t)dk * sizeof(double));
        double *rb = malloc((size_t)dk * sizeof(double));
        double *ra = malloc((size_t)dk * sizeof(double));
        double *rv = malloc((size_t)dv * sizeof(double));
        double *rw = malloc((size_t)dv * sizeof(double));
        CHECK(rSp && rS && ro && rq && rk && rb && ra && rv && rw,
              "gdn2 ref malloc");
        if (!(rSp && rS && ro && rq && rk && rb && ra && rv && rw)) {
            free(rSp);
            free(rS);
            free(ro);
            free(rq);
            free(rk);
            free(rb);
            free(ra);
            free(rv);
            free(rw);
            goto done;
        }
        for (int i = 0; i < dk; i++) {
            rq[i] = q[i];
            rk[i] = k[i];
            rb[i] = b[i];
            ra[i] = al[i];
        }
        for (int j = 0; j < dv; j++) {
            rv[j] = v[j];
            rw[j] = w[j];
        }
        for (int i = 0; i < dk * dv; i++) {
            rSp[i] = Sp[i];
        }
        // 損失関数: L(p) (pは摂動対象)。C11 portableな関数呼び出し。
        // 各入力のcentral差分。
        for (int i = 0; i < dk; i++) {
            double o = rq[i];
            rq[i] = o + h;
            double lp = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rq[i] = o - h;
            double lm = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rq[i] = o;
            CHECK(grad_close(dQ[i], (lp - lm) / (2 * h), tol),
                  "dQ[%d] a=%f n=%f dk=%d dv=%d", i, dQ[i],
                  (lp - lm) / (2 * h), dk, dv);
        }
        for (int i = 0; i < dk; i++) {
            double o = rk[i];
            rk[i] = o + h;
            double lp = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rk[i] = o - h;
            double lm = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rk[i] = o;
            CHECK(grad_close(dK[i], (lp - lm) / (2 * h), tol),
                  "dK[%d] a=%f n=%f", i, dK[i], (lp - lm) / (2 * h));
        }
        for (int j = 0; j < dv; j++) {
            double o = rv[j];
            rv[j] = o + h;
            double lp = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rv[j] = o - h;
            double lm = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rv[j] = o;
            CHECK(grad_close(dV[j], (lp - lm) / (2 * h), tol),
                  "dV[%d] a=%f n=%f", j, dV[j], (lp - lm) / (2 * h));
        }
        for (int i = 0; i < dk; i++) {
            double o = rb[i];
            rb[i] = o + h;
            double lp = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rb[i] = o - h;
            double lm = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rb[i] = o;
            CHECK(grad_close(dB[i], (lp - lm) / (2 * h), tol),
                  "dB[%d] a=%f n=%f", i, dB[i], (lp - lm) / (2 * h));
        }
        for (int j = 0; j < dv; j++) {
            double o = rw[j];
            rw[j] = o + h;
            double lp = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rw[j] = o - h;
            double lm = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rw[j] = o;
            CHECK(grad_close(dW[j], (lp - lm) / (2 * h), tol),
                  "dW[%d] a=%f n=%f", j, dW[j], (lp - lm) / (2 * h));
        }
        for (int i = 0; i < dk; i++) {
            double o = ra[i];
            ra[i] = o + h * 0.1;  // alphaは[0,1]内に留める
            double lp = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            ra[i] = o - h * 0.1;
            double lm = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            ra[i] = o;
            CHECK(grad_close(dA[i], (lp - lm) / (2 * h * 0.1), tol),
                  "dAlpha[%d] a=%f n=%f", i, dA[i],
                  (lp - lm) / (2 * h * 0.1));
        }
        for (int i = 0; i < dk * dv; i++) {
            double o = rSp[i];
            rSp[i] = o + h;
            double lp = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rSp[i] = o - h;
            double lm = gdn2_loss_fn(rS, ro, rSp, rq, rk, rv, rb, rw, ra, dO, dSn, dk, dv);
            rSp[i] = o;
            CHECK(grad_close(dSp[i], (lp - lm) / (2 * h), tol),
                  "dSp[%d] a=%f n=%f", i, dSp[i], (lp - lm) / (2 * h));
        }
        free(rSp);
        free(rS);
        free(ro);
        free(rq);
        free(rk);
        free(rb);
        free(ra);
        free(rv);
        free(rw);
    }
done:
    free(q);
    free(k);
    free(b);
    free(al);
    free(v);
    free(w);
    free(dO);
    free(dSn);
    free(dQ);
    free(dK);
    free(dB);
    free(dA);
    free(dV);
    free(dW);
    free(dSp);
    free(scratch);
    jt_gdn2_state_free(Sp);
}

static void test_gdn2_bwd_invalid(void) {
    float *Sp = NULL;
    CHECK(jt_gdn2_state_alloc(&Sp, 4, 4) == JT_OK, "bwd inv alloc");
    if (Sp == NULL) {
        return;
    }
    size_t need = 0;
    CHECK(jt_gdn2_decode_bwd_scratch_floats(4, 4, &need) == JT_OK,
          "bwd inv scratch");
    float *sc = (float *)malloc(need * sizeof(float));
    CHECK(sc != NULL, "bwd inv malloc");
    if (sc == NULL) {
        jt_gdn2_state_free(Sp);
        return;
    }
    float q[4] = {1, 0, 0, 0}, k[4] = {0, 1, 0, 0}, v[4] = {1, 1, 1, 1},
          b[4] = {1, 1, 1, 1}, w[4] = {1, 1, 1, 1}, a[4] = {1, 1, 1, 1},
          dO[4] = {1, 0, 0, 0}, oQ[4] = {0}, oK[4] = {0}, oV[4] = {0},
          oB[4] = {0}, oW[4] = {0}, oA[4] = {0};
    float dSn[16];
    float dSp[16];
    for (int i = 0; i < 16; i++) {
        Sp[i] = 0.1f * (float)i;
        dSn[i] = 0.05f;
        dSp[i] = 9.0f;
    }
    CHECK(jt_gdn2_decode_bwd(NULL, q, k, v, b, w, a, dO, dSn, oQ, oK, oV,
                             oB, oW, oA, dSp, 4, 4, sc, need) ==
              JT_ERR_INVAL,
          "bwd NULL Sp");
    CHECK(jt_gdn2_decode_bwd(Sp, q, k, v, b, w, a, dO, dSn, NULL, oK, oV,
                             oB, oW, oA, dSp, 4, 4, sc, need) ==
              JT_ERR_INVAL,
          "bwd NULL dQ");
    CHECK(jt_gdn2_decode_bwd(Sp, q, k, v, b, w, a, dO, dSn, oQ, oK, oV, oB,
                             oW, oA, dSp, 0, 4, sc, need) == JT_ERR_INVAL,
          "bwd dk=0");
    CHECK(jt_gdn2_decode_bwd_scratch_floats(0, 4, &need) == JT_ERR_INVAL,
          "bwd scratch bad dims");
    CHECK(jt_gdn2_decode_bwd_scratch_floats(4, 4, NULL) == JT_ERR_INVAL,
          "bwd scratch NULL");
    // 非整列S_prevはALIGN。
    {
        void *raw = malloc(16 * sizeof(float) + 64);
        CHECK(raw != NULL, "raw malloc");
        if (raw != NULL) {
            float *mis = (float *)((char *)raw + 1);
            int rc = jt_gdn2_decode_bwd(mis, q, k, v, b, w, a, dO, dSn, oQ,
                                        oK, oV, oB, oW, oA, dSp, 4, 4, sc,
                                        need);
            CHECK(rc == JT_ERR_ALIGN, "bwd misalign rc=%d", rc);
            free(raw);
        }
    }
    // 非有限拒否 (fail-closed: 出力不変)。
    {
        float tq[4] = {1, 0, 0, 0};
        float ref[16];
        memcpy(ref, dSp, sizeof(ref));
        tq[0] = (float)NAN;
        errno = 0;
        int rc = jt_gdn2_decode_bwd(Sp, tq, k, v, b, w, a, dO, dSn, oQ, oK,
                                    oV, oB, oW, oA, dSp, 4, 4, sc, need);
        CHECK(rc == JT_ERR_INVAL && errno == EINVAL, "bwd nan q");
        CHECK(memcmp(dSp, ref, sizeof(ref)) == 0, "bwd nan q mutated");
        tq[0] = 1.0f;
        float ta[4] = {1, 1, 1, 1};
        ta[1] = 2.0f;
        memcpy(ref, dSp, sizeof(ref));
        rc = jt_gdn2_decode_bwd(Sp, q, k, v, b, w, ta, dO, dSn, oQ, oK, oV,
                                oB, oW, oA, dSp, 4, 4, sc, need);
        CHECK(rc == JT_ERR_INVAL, "bwd alpha=2");
        CHECK(memcmp(dSp, ref, sizeof(ref)) == 0, "bwd alpha mutated");
    }
    free(sc);
    jt_gdn2_state_free(Sp);
}

// ---- SwiGLU勾配チェック ----

static void test_swiglu_grad(void) {
    const int n = 4;
    const int h = 4;
    const double hh = 1e-4;
    const double tol = 1e-3;
    float X[4], Wg[16], Wu[16], Wd[16], G[4], U[4], dY[4];
    float dX[4], dWg[16], dWu[16], dWd[16];
    for (int j = 0; j < n; j++) {
        X[j] = 0.4f * fdval(0, j, 21) + 0.2f;
        dY[j] = 0.5f * fdval(1, j, 22) + 0.1f;
    }
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < n; j++) {
            Wg[i * n + j] = 0.3f * fdval(i, j, 23);
            Wu[i * n + j] = 0.3f * fdval(i, j, 24);
            Wd[i * n + j] = 0.3f * fdval(i, j, 25);
        }
    }
    // cached G/U (forward再計算)。
    for (int i = 0; i < h; i++) {
        double g = 0.0;
        double u = 0.0;
        for (int j = 0; j < n; j++) {
            g += (double)X[j] * (double)Wg[i * n + j];
            u += (double)X[j] * (double)Wu[i * n + j];
        }
        G[i] = (float)g;
        U[i] = (float)u;
    }
    int rc = jt_swiglu_bwd(dY, X, G, U, Wd, Wg, Wu, dX, dWg, dWu, dWd, n,
                           h);
    CHECK(rc == JT_OK, "swiglu_bwd rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    {
        double rX[4], rWg[16], rWu[16], rWd[16];
        for (int j = 0; j < n; j++) {
            rX[j] = X[j];
        }
        for (int i = 0; i < 16; i++) {
            rWg[i] = Wg[i];
            rWu[i] = Wu[i];
            rWd[i] = Wd[i];
        }
        for (int j = 0; j < n; j++) {
            double o = rX[j];
            rX[j] = o + hh;
            double lp = sw_loss_fn(rX, rWg, rWu, rWd, dY, n, h);
            rX[j] = o - hh;
            double lm = sw_loss_fn(rX, rWg, rWu, rWd, dY, n, h);
            rX[j] = o;
            CHECK(grad_close(dX[j], (lp - lm) / (2 * hh), tol),
                  "swiglu dX[%d] a=%f n=%f", j, dX[j], (lp - lm) / (2 * hh));
        }
        for (int i = 0; i < 16; i++) {
            double o = rWg[i];
            rWg[i] = o + hh;
            double lp = sw_loss_fn(rX, rWg, rWu, rWd, dY, n, h);
            rWg[i] = o - hh;
            double lm = sw_loss_fn(rX, rWg, rWu, rWd, dY, n, h);
            rWg[i] = o;
            CHECK(grad_close(dWg[i], (lp - lm) / (2 * hh), tol),
                  "swiglu dWg[%d] a=%f n=%f", i, dWg[i],
                  (lp - lm) / (2 * hh));
        }
        for (int i = 0; i < 16; i++) {
            double o = rWu[i];
            rWu[i] = o + hh;
            double lp = sw_loss_fn(rX, rWg, rWu, rWd, dY, n, h);
            rWu[i] = o - hh;
            double lm = sw_loss_fn(rX, rWg, rWu, rWd, dY, n, h);
            rWu[i] = o;
            CHECK(grad_close(dWu[i], (lp - lm) / (2 * hh), tol),
                  "swiglu dWu[%d] a=%f n=%f", i, dWu[i],
                  (lp - lm) / (2 * hh));
        }
        for (int i = 0; i < 16; i++) {
            double o = rWd[i];
            rWd[i] = o + hh;
            double lp = sw_loss_fn(rX, rWg, rWu, rWd, dY, n, h);
            rWd[i] = o - hh;
            double lm = sw_loss_fn(rX, rWg, rWu, rWd, dY, n, h);
            rWd[i] = o;
            CHECK(grad_close(dWd[i], (lp - lm) / (2 * hh), tol),
                  "swiglu dWd[%d] a=%f n=%f", i, dWd[i],
                  (lp - lm) / (2 * hh));
        }
    }
    // 不正系。
    CHECK(jt_swiglu_bwd(NULL, X, G, U, Wd, Wg, Wu, dX, dWg, dWu, dWd, n,
                        h) == JT_ERR_INVAL,
          "swiglu NULL dY");
    CHECK(jt_swiglu_bwd(dY, X, G, U, Wd, Wg, Wu, dX, dWg, dWu, dWd, 0,
                        h) == JT_ERR_INVAL,
          "swiglu n=0");
}

// ---- RMSNorm勾配チェック ----

static void test_rmsnorm_grad(void) {
    const int n = 8;
    const double hh = 1e-4;
    const double tol = 1e-3;
    const float eps = 1e-6f;
    float X[8], W[8], dY[8], dX[8], dW[8];
    for (int i = 0; i < n; i++) {
        X[i] = 0.5f * fdval(i, 0, 31) + 0.3f;
        W[i] = 0.8f + 0.2f * fdval(i, 1, 32);
        dY[i] = 0.4f * fdval(i, 2, 33) - 0.1f;
    }
    int rc = jt_rmsnorm_bwd(dY, X, W, dX, dW, n, eps);
    CHECK(rc == JT_OK, "rmsnorm_bwd rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    {
        double rX[8], rW[8];
        for (int i = 0; i < n; i++) {
            rX[i] = X[i];
            rW[i] = W[i];
        }
        for (int i = 0; i < n; i++) {
            double o = rX[i];
            rX[i] = o + hh;
            double lp = rms_loss_fn(rX, rW, dY, n, (double)eps);
            rX[i] = o - hh;
            double lm = rms_loss_fn(rX, rW, dY, n, (double)eps);
            rX[i] = o;
            CHECK(grad_close(dX[i], (lp - lm) / (2 * hh), tol),
                  "rms dX[%d] a=%f n=%f", i, dX[i], (lp - lm) / (2 * hh));
        }
        for (int i = 0; i < n; i++) {
            double o = rW[i];
            rW[i] = o + hh;
            double lp = rms_loss_fn(rX, rW, dY, n, (double)eps);
            rW[i] = o - hh;
            double lm = rms_loss_fn(rX, rW, dY, n, (double)eps);
            rW[i] = o;
            CHECK(grad_close(dW[i], (lp - lm) / (2 * hh), tol),
                  "rms dW[%d] a=%f n=%f", i, dW[i], (lp - lm) / (2 * hh));
        }
    }
    CHECK(jt_rmsnorm_bwd(NULL, X, W, dX, dW, n, eps) == JT_ERR_INVAL,
          "rms NULL dY");
    CHECK(jt_rmsnorm_bwd(dY, X, W, dX, dW, 0, eps) == JT_ERR_INVAL,
          "rms n=0");
    CHECK(jt_rmsnorm_bwd(dY, X, W, dX, dW, n, 0.0f) == JT_ERR_INVAL,
          "rms eps=0");
}

// ---- SwiGLU/RMSNorm forward単体 (手計算値, tol=1e-5) ----

static int fwd_close(float got, double want, double tol) {
    double d = fabs((double)got - want);
    double allowed = tol + tol * fabs(want);
    return d <= allowed;
}

static void test_swiglu_fwd(void) {
    const int n = 2;
    const int h = 2;
    const double tol = 1e-5;
    // 手計算 (double):
    //   G0=1*1+(-0.5)*0.5=0.75, U0=1*0.5+(-0.5)*(-1)=1.0
    //   G1=1*(-1)+(-0.5)*2=-2.0, U1=1*2+(-0.5)*0=2.0
    //   s0=silu(0.75)*1.0=0.5093840244, s1=silu(-2)*2=-0.4768116881
    //   Y0=s0*1+s1*(-0.5)=0.7477898684, Y1=s0*2+s1*1.5=0.3035505166
    float X[2] = {1.0f, -0.5f};
    float Wg[4] = {1.0f, 0.5f, -1.0f, 2.0f};
    float Wu[4] = {0.5f, -1.0f, 2.0f, 0.0f};
    float Wd[4] = {1.0f, 2.0f, -0.5f, 1.5f};
    float G[2] = {0.0f, 0.0f};
    float U[2] = {0.0f, 0.0f};
    float Y[2] = {0.0f, 0.0f};
    int rc = jt_swiglu_fwd(X, Wg, Wu, Wd, G, U, Y, n, h);
    CHECK(rc == JT_OK, "swiglu_fwd rc=%d", rc);
    if (rc == JT_OK) {
        CHECK(fwd_close(G[0], 0.75, tol), "swiglu_fwd G0=%f", G[0]);
        CHECK(fwd_close(G[1], -2.0, tol), "swiglu_fwd G1=%f", G[1]);
        CHECK(fwd_close(U[0], 1.0, tol), "swiglu_fwd U0=%f", U[0]);
        CHECK(fwd_close(U[1], 2.0, tol), "swiglu_fwd U1=%f", U[1]);
        CHECK(fwd_close(Y[0], 0.7477898684257798, tol), "swiglu_fwd Y0=%f",
              Y[0]);
        CHECK(fwd_close(Y[1], 0.30355051663038424, tol), "swiglu_fwd Y1=%f",
              Y[1]);
    }
    // double参照との一致 (既存finite-diffと同一式)。
    {
        double rX[2], rWg[4], rWu[4], rWd[4], rY[2];
        for (int j = 0; j < n; j++) {
            rX[j] = (double)X[j];
        }
        for (int i = 0; i < 4; i++) {
            rWg[i] = (double)Wg[i];
            rWu[i] = (double)Wu[i];
            rWd[i] = (double)Wd[i];
        }
        ref_swiglu_fwd(rX, rWg, rWu, rWd, rY, n, h);
        CHECK(fwd_close(Y[0], rY[0], tol), "swiglu_fwd ref Y0=%f vs %f",
              Y[0], rY[0]);
        CHECK(fwd_close(Y[1], rY[1], tol), "swiglu_fwd ref Y1=%f vs %f",
              Y[1], rY[1]);
    }
    // bwdとの勾配整合: fwd核のG/Uをbwdへ渡して正常終了すること。
    {
        float dY[2] = {0.3f, -0.2f};
        float dX[2] = {0.0f, 0.0f};
        float dWg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float dWu[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float dWd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int brc = jt_swiglu_bwd(dY, X, G, U, Wd, Wg, Wu, dX, dWg, dWu,
                                dWd, n, h);
        CHECK(brc == JT_OK, "swiglu fwd->bwd rc=%d", brc);
        if (brc == JT_OK) {
            for (int i = 0; i < n; i++) {
                CHECK(isfinite(dX[i]), "swiglu fwd->bwd dX[%d]=%f", i,
                      dX[i]);
            }
        }
    }
    // 不正系 (fail-closed: 出力不変)。
    {
        float g0 = G[0], u0 = U[0], y0 = Y[0];
        CHECK(jt_swiglu_fwd(NULL, Wg, Wu, Wd, G, U, Y, n, h) ==
                  JT_ERR_INVAL,
              "swiglu_fwd NULL X");
        CHECK(jt_swiglu_fwd(X, Wg, Wu, Wd, G, U, Y, 0, h) ==
                  JT_ERR_INVAL,
              "swiglu_fwd n=0");
        CHECK(jt_swiglu_fwd(X, Wg, Wu, Wd, G, U, Y, n, 0) ==
                  JT_ERR_INVAL,
              "swiglu_fwd h=0");
        CHECK(jt_swiglu_fwd(X, Wg, Wu, Wd, G, U, Y, JT_BWD_MAX_WIDE + 1,
                            h) == JT_ERR_INVAL,
              "swiglu_fwd n over max");
        CHECK(G[0] == g0 && U[0] == u0 && Y[0] == y0,
              "swiglu_fwd mutated on inval");
        errno = 0;
        {
            float badX[2] = {1.0f, (float)NAN};
            float g1 = G[0];
            int nrc = jt_swiglu_fwd(badX, Wg, Wu, Wd, G, U, Y, n, h);
            CHECK(nrc == JT_ERR_INVAL && errno == EINVAL,
                  "swiglu_fwd nan");
            CHECK(G[0] == g1, "swiglu_fwd nan mutated");
        }
    }
}

static void test_rmsnorm_fwd(void) {
    const int n = 4;
    const double tol = 1e-5;
    const float eps = 1e-6f;
    // 手計算: mean=(1+4+1+0.25)/4=1.5625, r=1/sqrt(1.5625+1e-6)。
    float X[4] = {1.0f, 2.0f, -1.0f, 0.5f};
    float W[4] = {1.0f, 0.5f, 2.0f, 1.0f};
    float Y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int rc = jt_rmsnorm_fwd(X, W, Y, n, eps);
    CHECK(rc == JT_OK, "rmsnorm_fwd rc=%d", rc);
    if (rc == JT_OK) {
        CHECK(fwd_close(Y[0], 0.7999997440001229, tol),
              "rmsnorm_fwd Y0=%f", Y[0]);
        CHECK(fwd_close(Y[1], 0.7999997440001229, tol),
              "rmsnorm_fwd Y1=%f", Y[1]);
        CHECK(fwd_close(Y[2], -1.5999994880002457, tol),
              "rmsnorm_fwd Y2=%f", Y[2]);
        CHECK(fwd_close(Y[3], 0.39999987200006143, tol),
              "rmsnorm_fwd Y3=%f", Y[3]);
    }
    // double参照との一致。
    {
        double rX[4], rW[4], rY[4];
        for (int i = 0; i < n; i++) {
            rX[i] = (double)X[i];
            rW[i] = (double)W[i];
        }
        ref_rms_fwd(rX, rW, rY, n, (double)eps);
        for (int i = 0; i < n; i++) {
            CHECK(fwd_close(Y[i], rY[i], tol),
                  "rmsnorm_fwd ref Y[%d]=%f vs %f", i, Y[i], rY[i]);
        }
    }
    // bwdとの勾配整合: fwd核出力をbwd経路に渡して正常終了すること。
    {
        float dY[4] = {0.2f, -0.1f, 0.4f, 0.1f};
        float dX[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float dW[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        int brc = jt_rmsnorm_bwd(dY, X, W, dX, dW, n, eps);
        CHECK(brc == JT_OK, "rmsnorm fwd->bwd rc=%d", brc);
        if (brc == JT_OK) {
            for (int i = 0; i < n; i++) {
                CHECK(isfinite(dX[i]) && isfinite(dW[i]),
                      "rmsnorm fwd->bwd [%d]=%f/%f", i, dX[i], dW[i]);
            }
        }
    }
    // 不正系 (fail-closed: 出力不変)。
    {
        float y0 = Y[0];
        CHECK(jt_rmsnorm_fwd(NULL, W, Y, n, eps) == JT_ERR_INVAL,
              "rmsnorm_fwd NULL X");
        CHECK(jt_rmsnorm_fwd(X, W, Y, 0, eps) == JT_ERR_INVAL,
              "rmsnorm_fwd n=0");
        CHECK(jt_rmsnorm_fwd(X, W, Y, n, 0.0f) == JT_ERR_INVAL,
              "rmsnorm_fwd eps=0");
        CHECK(jt_rmsnorm_fwd(X, W, Y, n, -1.0f) == JT_ERR_INVAL,
              "rmsnorm_fwd eps<0");
        CHECK(Y[0] == y0, "rmsnorm_fwd mutated on inval");
        errno = 0;
        {
            float badX[4] = {1.0f, 2.0f, (float)INFINITY, 0.5f};
            int nrc = jt_rmsnorm_fwd(badX, W, Y, n, eps);
            CHECK(nrc == JT_ERR_INVAL && errno == EINVAL,
                  "rmsnorm_fwd inf");
            CHECK(Y[0] == y0, "rmsnorm_fwd inf mutated");
        }
    }
}

// ---- checkpoint再計算 (モデル非依存コールバック) ----

// 玩具層forward: x[layer+1] = 2*x[layer] + (layer+1)。境界x[0]保存→区間毎に
// 再実行し、参照forwardとの一致で検証する (2層2区間)。
typedef struct {
    float x[4];
    int calls;
    int segs[4];
    int layers[4];
    int fail_on;  // >=0の層でJT_ERR_INVALを返す (負値は失敗なし)
} ckpt_ctx_t;

static int ckpt_layer_fn(int seg_idx, int layer, void *ctx) {
    ckpt_ctx_t *c = (ckpt_ctx_t *)ctx;
    if (c == NULL || layer < 0 || layer >= 3) {
        return JT_ERR_INVAL;
    }
    if (c->fail_on == layer) {
        return JT_ERR_INVAL;
    }
    c->x[layer + 1] = 2.0f * c->x[layer] + (float)(layer + 1);
    if (c->calls >= 0 && c->calls < 4) {
        c->segs[c->calls] = seg_idx;
        c->layers[c->calls] = layer;
    }
    c->calls++;
    return JT_OK;
}

static void test_checkpoint(void) {
    int seg = 0;
    CHECK(jt_ckpt_num_segments(16, &seg) == JT_OK && seg == 4,
          "ckpt seg16=%d", seg);
    CHECK(jt_ckpt_num_segments(20, &seg) == JT_OK && seg == 5,
          "ckpt seg20=%d", seg);
    CHECK(jt_ckpt_num_segments(0, &seg) == JT_ERR_INVAL, "ckpt seg0");
    CHECK(jt_ckpt_num_segments(16, NULL) == JT_ERR_INVAL, "ckpt seg NULL");
    {
        int b[5] = {0};
        CHECK(jt_ckpt_boundaries(16, 4, b, 5) == JT_OK, "ckpt bounds rc");
        CHECK(b[0] == 0 && b[1] == 4 && b[2] == 8 && b[3] == 12 &&
                  b[4] == 16,
              "ckpt bounds16x4 %d %d %d %d %d", b[0], b[1], b[2], b[3],
              b[4]);
    }
    {
        int b[4] = {0};
        CHECK(jt_ckpt_boundaries(10, 3, b, 4) == JT_OK, "ckpt bounds10x3");
        CHECK(b[0] == 0 && b[1] == 4 && b[2] == 7 && b[3] == 10,
              "ckpt bounds10x3 val %d %d %d %d", b[0], b[1], b[2], b[3]);
    }
    CHECK(jt_ckpt_boundaries(10, 0, (int[1]){0}, 1) == JT_ERR_INVAL,
          "ckpt bounds seg0");
    {
        size_t full = 0;
        size_t stored = 0;
        size_t peak = 0;
        CHECK(jt_ckpt_mem_estimate(16, 4, 1000, &full, &stored, &peak) ==
                  JT_OK,
              "ckpt mem rc");
        CHECK(full == 16000 && stored == 5000 && peak == 9000,
              "ckpt mem %zu %zu %zu", full, stored, peak);
        double r = 0.0;
        CHECK(jt_ckpt_saving_ratio(full, stored, &r) == JT_OK,
              "ckpt ratio rc");
        CHECK(fabs(r - 0.6875) < 1e-12, "ckpt ratio=%f", r);
        CHECK(jt_ckpt_saving_ratio(0, 0, &r) == JT_ERR_INVAL,
              "ckpt ratio full=0");
    }
    {
        int b[5] = {0, 4, 8, 12, 16};
        jt_ckpt_plan_t p = {16, 4, b};
        CHECK(jt_ckpt_plan_init(&p) == JT_OK, "ckpt plan ok");
        int idx = -1;
        CHECK(jt_ckpt_find_segment(&p, 5, &idx) == JT_OK && idx == 1,
              "ckpt find5=%d", idx);
        CHECK(jt_ckpt_find_segment(&p, 16, &idx) == JT_ERR_INVAL,
              "ckpt find OOB");
        // 2層2区間で境界保存→再計算→一致 (MAJOR-2実実装テスト)。
        {
            int b2[3] = {0, 1, 2};
            jt_ckpt_plan_t p2 = {2, 2, b2};
            ckpt_ctx_t c;
            memset(&c, 0, sizeof(c));
            c.fail_on = -1;
            c.x[0] = 1.0f;  // 境界保存値
            // 参照forward: x1=2*1+1=3, x2=2*3+2=8。
            CHECK(jt_ckpt_recompute_range(&p2, 0, ckpt_layer_fn, &c) ==
                      JT_OK,
                  "ckpt recomp seg0");
            CHECK(c.calls == 1 && c.segs[0] == 0 && c.layers[0] == 0,
                  "ckpt seg0 trace calls=%d seg=%d layer=%d", c.calls,
                  c.segs[0], c.layers[0]);
            CHECK(fabsf(c.x[1] - 3.0f) < 1e-6f, "ckpt x1=%f", c.x[1]);
            CHECK(jt_ckpt_recompute_range(&p2, 1, ckpt_layer_fn, &c) ==
                      JT_OK,
                  "ckpt recomp seg1");
            CHECK(c.calls == 2 && c.segs[1] == 1 && c.layers[1] == 1,
                  "ckpt seg1 trace calls=%d seg=%d layer=%d", c.calls,
                  c.segs[1], c.layers[1]);
            CHECK(fabsf(c.x[2] - 8.0f) < 1e-6f, "ckpt x2=%f", c.x[2]);
        }
        // 不正入力 + コールバック失敗伝播。
        {
            int b2[3] = {0, 1, 2};
            jt_ckpt_plan_t p2 = {2, 2, b2};
            ckpt_ctx_t c;
            memset(&c, 0, sizeof(c));
            c.fail_on = -1;
            c.x[0] = 1.0f;
            CHECK(jt_ckpt_recompute_range(&p2, 9, ckpt_layer_fn, &c) ==
                      JT_ERR_INVAL,
                  "ckpt recompute bad seg");
            CHECK(jt_ckpt_recompute_range(&p2, 0, NULL, &c) ==
                      JT_ERR_INVAL,
                  "ckpt recompute NULL fn");
            CHECK(jt_ckpt_recompute_range(NULL, 0, ckpt_layer_fn, &c) ==
                      JT_ERR_INVAL,
                  "ckpt recompute NULL plan");
            c.fail_on = 1;
            CHECK(jt_ckpt_recompute_range(&p2, 1, ckpt_layer_fn, &c) ==
                      JT_ERR_INVAL,
                  "ckpt recompute cb fail");
        }
        jt_ckpt_plan_t bad = {16, 4, (int[5]){0, 4, 4, 12, 16}};
        CHECK(jt_ckpt_plan_init(&bad) == JT_ERR_INVAL, "ckpt plan nonmono");
    }
    {
        // bench足場: メモリ削減効果の見積り表示 (純粋関数、時間計測なし)。
        jt_ckpt_bench_t be = {0, 0, 0, 0.0};
        CHECK(jt_ckpt_bench_estimate(64, 8, 1 << 20, &be) == JT_OK,
              "ckpt bench rc");
        CHECK(be.full == (size_t)64 << 20 && be.ratio > 0.0,
              "ckpt bench val");
        printf("ckpt-bench: layers=64 seg=8 full=%zu stored=%zu peak=%zu "
               "ratio=%.3f\n",
               be.full, be.stored, be.peak, be.ratio);
    }
}

// ---- swiglu unchecked経路の数値一致 (検証ありAPIとbit一致) ----
// unchecked使用条件: 重み事前検証済み＋区間内不変の呼び出しでのみ使うこと。
// 有限の固定入力で通常経路とunchecked経路の全出力をmemcmpする。
// fail-closed不変 (INVAL系) は既存の不正系テストで担保する。
static void test_swiglu_unchecked_match(void) {
    float X[2] = {1.0f, -0.5f};
    float Wg[4] = {1.0f, 0.5f, -1.0f, 2.0f};
    float Wu[4] = {0.5f, -1.0f, 2.0f, 0.0f};
    float Wd[4] = {1.0f, 2.0f, -0.5f, 1.5f};
    float G1[2] = {0, 0}, G2[2] = {0, 0};
    float U1[2] = {0, 0}, U2[2] = {0, 0};
    float Y1[2] = {0, 0}, Y2[2] = {0, 0};
    int fr1 = jt_swiglu_fwd(X, Wg, Wu, Wd, G1, U1, Y1, 2, 2);
    int fr2 = jt_swiglu_fwd_unchecked(X, Wg, Wu, Wd, G2, U2, Y2, 2, 2);
    CHECK(fr1 == JT_OK && fr2 == JT_OK, "swiglu unchecked fwd rc %d/%d",
          fr1, fr2);
    if (fr1 == JT_OK && fr2 == JT_OK) {
        CHECK(memcmp(G1, G2, sizeof(G1)) == 0,
              "swiglu unchecked fwd G bits");
        CHECK(memcmp(U1, U2, sizeof(U1)) == 0,
              "swiglu unchecked fwd U bits");
        CHECK(memcmp(Y1, Y2, sizeof(Y1)) == 0,
              "swiglu unchecked fwd Y bits");
    }
    {
        float dY[2] = {0.3f, -0.2f};
        float dX1[2] = {0, 0}, dX2[2] = {0, 0};
        float dwg1[4] = {0, 0, 0, 0}, dwg2[4] = {0, 0, 0, 0};
        float dwu1[4] = {0, 0, 0, 0}, dwu2[4] = {0, 0, 0, 0};
        float dwd1[4] = {0, 0, 0, 0}, dwd2[4] = {0, 0, 0, 0};
        int br1 = jt_swiglu_bwd(dY, X, G1, U1, Wd, Wg, Wu, dX1, dwg1,
                                dwu1, dwd1, 2, 2);
        int br2 = jt_swiglu_bwd_unchecked(dY, X, G1, U1, Wd, Wg, Wu, dX2,
                                          dwg2, dwu2, dwd2, 2, 2);
        CHECK(br1 == JT_OK && br2 == JT_OK,
              "swiglu unchecked bwd rc %d/%d", br1, br2);
        if (br1 == JT_OK && br2 == JT_OK) {
            CHECK(memcmp(dX1, dX2, sizeof(dX1)) == 0,
                  "swiglu unchecked bwd dX bits");
            CHECK(memcmp(dwg1, dwg2, sizeof(dwg1)) == 0,
                  "swiglu unchecked bwd dWg bits");
            CHECK(memcmp(dwu1, dwu2, sizeof(dwu1)) == 0,
                  "swiglu unchecked bwd dWu bits");
            CHECK(memcmp(dwd1, dwd2, sizeof(dwd1)) == 0,
                  "swiglu unchecked bwd dWd bits");
        }
    }
}

int main(void) {
    test_gdn2_grad(4, 4);
    test_gdn2_grad(4, 8);
    test_gdn2_bwd_invalid();
    test_swiglu_grad();
    test_rmsnorm_grad();
    test_swiglu_fwd();
    test_rmsnorm_fwd();
    test_checkpoint();
    test_swiglu_unchecked_match();
    if (g_fail != 0) {
        fprintf(stderr, "train_bwd: FAIL\n");
        return 1;
    }
    printf("train_bwd: OK (gdn2/swiglu/rmsnorm grad + ckpt)\n");
    return 0;
}
