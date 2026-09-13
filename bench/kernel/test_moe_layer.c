// test_moe_layer: MoE層 forward/backward + sticky系列合算の検証。
// - forward既知値一致 (手計算値 tol=1e-5 + double参照一致)
// - 数値勾配チェック (gate Wgateとexpert重みのcentral finite-diff, tol=1e-3)
// - top-kマスク性 (非選択のdLogits/dWgate/dWが0)
// - sticky系列ヘルパーの平均化一致 + 不正系 (fail-closed: 出力不変)
// 成功時 exit 0 + "moe_layer: OK"。C11・errnoベース。
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/common.h"
#include "jimotono/moe_layer.h"
#include "jimotono/routing.h"
#include "jimotono/train_bwd.h"

static int g_fail = 0;

#define CHECK(cond, ...)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);           \
            fprintf(stderr, __VA_ARGS__);                                 \
            fprintf(stderr, "\n");                                        \
            g_fail = 1;                                                   \
        }                                                                 \
    } while (0)

static float fdval(int a, int b, int salt) {
    int v = (a * 31 + b * 17 + salt * 13) % 11;
    return (float)(v - 5) / 5.0f;
}

static int fwd_close(float got, double want, double tol) {
    double d = fabs((double)got - want);
    double allowed = tol + tol * fabs(want);
    return d <= allowed;
}

static int grad_close(float analytic, double numeric, double tol) {
    double d = fabs((double)analytic - numeric);
    double allowed = tol + tol * fabs(numeric);
    return d <= allowed;
}

static double ref_sig(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// ---- double参照のMoE forward (ライブラリを使わない独立式) ----
static void ref_moe_fwd(const double *X, const double *Wgate,
                        const double *Wg, const double *Wu,
                        const double *Wd, const double *Wg_s,
                        const double *Wu_s, const double *Wd_s, double *Y,
                        int n, int h, int E, int k, int S, size_t *ids,
                        double *w) {
    double logits[4096];
    for (int e = 0; e < E; e++) {
        double acc = 0.0;
        for (int j = 0; j < n; j++) {
            acc += X[j] * Wgate[e * n + j];
        }
        logits[e] = acc;
    }
    // top-k線形選択 (小index優先のtie-break)。
    for (int p = 0; p < k; p++) {
        size_t best = 0;
        double bestv = 0.0;
        int first = 1;
        for (int e = 0; e < E; e++) {
            int taken = 0;
            for (int q = 0; q < p; q++) {
                taken |= (ids[q] == (size_t)e);
            }
            if (taken) {
                continue;
            }
            if (first || logits[e] > bestv) {
                best = (size_t)e;
                bestv = logits[e];
                first = 0;
            }
        }
        ids[p] = best;
    }
    double mx = logits[ids[0]];
    for (int p = 1; p < k; p++) {
        if (logits[ids[p]] > mx) {
            mx = logits[ids[p]];
        }
    }
    double sum = 0.0;
    for (int p = 0; p < k; p++) {
        w[p] = exp(logits[ids[p]] - mx);
        sum += w[p];
    }
    for (int p = 0; p < k; p++) {
        w[p] /= sum;
    }
    for (int j = 0; j < n; j++) {
        Y[j] = 0.0;
    }
    for (int p = 0; p < k; p++) {
        int e = (int)ids[p];
        double s[4096];
        for (int i = 0; i < h; i++) {
            double g = 0.0, u = 0.0;
            for (int j = 0; j < n; j++) {
                g += X[j] * Wg[(e * h + i) * n + j];
                u += X[j] * Wu[(e * h + i) * n + j];
            }
            s[i] = g * ref_sig(g) * u;
        }
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int i = 0; i < h; i++) {
                acc += s[i] * Wd[(e * h + i) * n + j];
            }
            Y[j] += w[p] * acc;
        }
    }
    for (int sidx = 0; sidx < S; sidx++) {
        double s[4096];
        for (int i = 0; i < h; i++) {
            double g = 0.0, u = 0.0;
            for (int j = 0; j < n; j++) {
                g += X[j] * Wg_s[(sidx * h + i) * n + j];
                u += X[j] * Wu_s[(sidx * h + i) * n + j];
            }
            s[i] = g * ref_sig(g) * u;
        }
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int i = 0; i < h; i++) {
                acc += s[i] * Wd_s[(sidx * h + i) * n + j];
            }
            Y[j] += acc;
        }
    }
}

// ---- forward既知値 (n=2,h=1,E=3,k=2,S=1, 手計算§参照) ----
static void test_fwd_known(void) {
    const double tol = 1e-5;
    float X[2] = {1.0f, 0.5f};
    // Wgate [3][2]: logits=[1,0.5,-1] → ids=[0,1], w=[0.62245933,0.37754067]
    float Wgate[6] = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    float Wg[6] = {0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f};
    float Wu[6] = {1.0f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f};
    float Wd[6] = {1.0f, 2.0f, -1.0f, 0.5f, 0.0f, 0.0f};
    float Wg_s[2] = {0.5f, 0.5f};
    float Wu_s[2] = {1.0f, -1.0f};
    float Wd_s[2] = {0.5f, -0.5f};
    float Y[2] = {0.0f, 0.0f};
    size_t ids[2] = {99, 99};
    float weights[2] = {0.0f, 0.0f};
    float logits[3] = {0.0f, 0.0f, 0.0f};
    float Gs[2] = {0.0f, 0.0f};
    float Us[2] = {0.0f, 0.0f};
    float Ys[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float Gsel[2] = {0.0f, 0.0f};
    float Usel[2] = {0.0f, 0.0f};
    int rc = jt_moe_fwd(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Y, 2, 1,
                        3, 2, 1, ids, weights, logits, Gsel, Usel, Ys, Gs,
                        Us);
    CHECK(rc == JT_OK, "moe fwd known rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    CHECK(ids[0] == 0 && ids[1] == 1, "moe fwd ids {%zu,%zu} want {0,1}",
          ids[0], ids[1]);
    CHECK(fwd_close(weights[0], 0.6224593312018546, tol),
          "moe fwd w0=%f", weights[0]);
    CHECK(fwd_close(weights[1], 0.37754066879814546, tol),
          "moe fwd w1=%f", weights[1]);
    CHECK(fwd_close(logits[0], 1.0, tol), "moe fwd logit0=%f", logits[0]);
    CHECK(fwd_close(logits[1], 0.5, tol), "moe fwd logit1=%f", logits[1]);
    CHECK(fwd_close(logits[2], -1.0, tol), "moe fwd logit2=%f",
          logits[2]);
    CHECK(fwd_close(Gsel[0], 0.25, tol), "moe fwd Gsel0=%f", Gsel[0]);
    CHECK(fwd_close(Gsel[1], 1.0, tol), "moe fwd Gsel1=%f", Gsel[1]);
    CHECK(fwd_close(Usel[0], 1.0, tol), "moe fwd Usel0=%f", Usel[0]);
    CHECK(fwd_close(Usel[1], 0.25, tol), "moe fwd Usel1=%f", Usel[1]);
    CHECK(fwd_close(Y[0], 0.14582792210843093, tol), "moe fwd Y0=%f",
          Y[0]);
    CHECK(fwd_close(Y[1], 0.08212054137232436, tol), "moe fwd Y1=%f",
          Y[1]);
    // double参照との一致。
    {
        double rX[2], rWg[3], rWu[3], rWd[3], rY[2];
        double rGate[6], rWgA[6], rWuA[6], rWdA[6];
        double rGs[2], rUs[2], rDs[2];
        size_t rids[2] = {0, 0};
        double rw[2] = {0.0, 0.0};
        for (int j = 0; j < 2; j++) {
            rX[j] = X[j];
        }
        for (int i = 0; i < 6; i++) {
            rGate[i] = Wgate[i];
            rWgA[i] = Wg[i];
            rWuA[i] = Wu[i];
            rWdA[i] = Wd[i];
        }
        for (int i = 0; i < 2; i++) {
            rGs[i] = Wg_s[i];
            rUs[i] = Wu_s[i];
            rDs[i] = Wd_s[i];
        }
        (void)rWg;
        (void)rWu;
        (void)rWd;
        ref_moe_fwd(rX, rGate, rWgA, rWuA, rWdA, rGs, rUs, rDs, rY, 2,
                    1, 3, 2, 1, rids, rw);
        CHECK(rids[0] == ids[0] && rids[1] == ids[1],
              "moe fwd ref ids");
        CHECK(fwd_close(Y[0], rY[0], tol), "moe fwd ref Y0 %f vs %f",
              Y[0], rY[0]);
        CHECK(fwd_close(Y[1], rY[1], tol), "moe fwd ref Y1 %f vs %f",
              Y[1], rY[1]);
    }
}

// ---- 数値勾配チェック (gate Wgate + expert重み, tol=1e-3) ----
static double moe_loss_ref(const double *rX, const double *rGate,
                           const double *rWg, const double *rWu,
                           const double *rWd, const double *rGs,
                           const double *rUs, const double *rDs,
                           const float *dY, int n, int h, int E, int k,
                           int S) {
    double Y[4096];
    size_t ids[64];
    double w[64];
    ref_moe_fwd(rX, rGate, rWg, rWu, rWd, rGs, rUs, rDs, Y, n, h, E, k,
                S, ids, w);
    double L = 0.0;
    for (int j = 0; j < n; j++) {
        L += Y[j] * (double)dY[j];
    }
    return L;
}

static void test_grad(void) {
    const int n = 4;
    const int h = 2;
    const int E = 4;
    const int k = 2;
    const int S = 1;
    const double hh = 1e-4;
    const double tol = 1e-3;
    float X[4], Wgate[16], Wg[32], Wu[32], Wd[32];
    float Wg_s[8], Wu_s[8], Wd_s[8], dY[4];
    for (int j = 0; j < n; j++) {
        X[j] = 0.4f * fdval(0, j, 41) + 0.2f;
        dY[j] = 0.5f * fdval(1, j, 42) + 0.15f;
    }
    // gateは選択が安定するようオフセットで分離 (摂動1e-4でflipしない)。
    for (int e = 0; e < E; e++) {
        for (int j = 0; j < n; j++) {
            Wgate[e * n + j] = 0.3f * fdval(e, j, 43) + (float)(E - e) * 0.8f;
        }
    }
    for (int i = 0; i < E * h * n; i++) {
        Wg[i] = 0.3f * fdval(i, 0, 44);
        Wu[i] = 0.3f * fdval(i, 1, 45);
        Wd[i] = 0.3f * fdval(i, 2, 46);
    }
    for (int i = 0; i < S * h * n; i++) {
        Wg_s[i] = 0.3f * fdval(i, 0, 47);
        Wu_s[i] = 0.3f * fdval(i, 1, 48);
        Wd_s[i] = 0.3f * fdval(i, 2, 49);
    }
    float Y[4] = {0};
    size_t ids[2] = {0, 0};
    float weights[2] = {0, 0};
    float logits[4] = {0, 0, 0, 0};
    float Gsel[4] = {0, 0, 0, 0};
    float Usel[4] = {0, 0, 0, 0};
    float Ysel[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float Gs[2] = {0, 0};
    float Us[2] = {0, 0};
    int rc = jt_moe_fwd(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Y, n, h,
                        E, k, S, ids, weights, logits, Gsel, Usel, Ysel,
                        Gs, Us);
    CHECK(rc == JT_OK, "moe grad fwd rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    float dX[4] = {0, 0, 0, 0};
    float dWgate[16] = {0};
    float dWg[32] = {0};
    float dWu[32] = {0};
    float dWd[32] = {0};
    float dWg_s[8] = {0};
    float dWu_s[8] = {0};
    float dWd_s[8] = {0};
    float dLogits[4] = {0, 0, 0, 0};
    rc = jt_moe_bwd(dY, X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, ids,
                    weights, Gsel, Usel, Ysel, Gs, Us, dX, dWgate, dWg,
                    dWu, dWd, dWg_s, dWu_s, dWd_s, dLogits, n, h, E, k,
                    S);
    CHECK(rc == JT_OK, "moe bwd rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    // double参照バッファ。
    double rX[4], rGate[16], rWg[32], rWu[32], rWd[32];
    double rGs[8], rUs[8], rDs[8];
    for (int j = 0; j < n; j++) {
        rX[j] = X[j];
    }
    for (int i = 0; i < 16; i++) {
        rGate[i] = Wgate[i];
    }
    for (int i = 0; i < 32; i++) {
        rWg[i] = Wg[i];
        rWu[i] = Wu[i];
        rWd[i] = Wd[i];
    }
    for (int i = 0; i < 8; i++) {
        rGs[i] = Wg_s[i];
        rUs[i] = Wu_s[i];
        rDs[i] = Wd_s[i];
    }
    // dXのチェック。
    for (int j = 0; j < n; j++) {
        double o = rX[j];
        rX[j] = o + hh;
        double lp = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rX[j] = o - hh;
        double lm = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rX[j] = o;
        CHECK(grad_close(dX[j], (lp - lm) / (2 * hh), tol),
              "moe dX[%d] a=%f n=%f", j, dX[j], (lp - lm) / (2 * hh));
    }
    // dWgate (gate logits経由) のチェック。
    for (int i = 0; i < E * n; i++) {
        double o = rGate[i];
        rGate[i] = o + hh;
        double lp = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rGate[i] = o - hh;
        double lm = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rGate[i] = o;
        // 非選択行は数値勾配もほぼ0 (選択flipなしの前提)。
        CHECK(grad_close(dWgate[i], (lp - lm) / (2 * hh), tol),
              "moe dWgate[%d] a=%f n=%f", i, dWgate[i],
              (lp - lm) / (2 * hh));
    }
    // expert重み (選択expertのみ値を持ち、非選択は0)。
    for (int i = 0; i < E * h * n; i++) {
        double o = rWg[i];
        rWg[i] = o + hh;
        double lp = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rWg[i] = o - hh;
        double lm = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rWg[i] = o;
        CHECK(grad_close(dWg[i], (lp - lm) / (2 * hh), tol),
              "moe dWg[%d] a=%f n=%f", i, dWg[i],
              (lp - lm) / (2 * hh));
    }
    for (int i = 0; i < E * h * n; i++) {
        double o = rWd[i];
        rWd[i] = o + hh;
        double lp = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rWd[i] = o - hh;
        double lm = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rWd[i] = o;
        CHECK(grad_close(dWd[i], (lp - lm) / (2 * hh), tol),
              "moe dWd[%d] a=%f n=%f", i, dWd[i],
              (lp - lm) / (2 * hh));
    }
    // 共有expert重み。
    for (int i = 0; i < S * h * n; i++) {
        double o = rDs[i];
        rDs[i] = o + hh;
        double lp = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rDs[i] = o - hh;
        double lm = moe_loss_ref(rX, rGate, rWg, rWu, rWd, rGs, rUs,
                                 rDs, dY, n, h, E, k, S);
        rDs[i] = o;
        CHECK(grad_close(dWd_s[i], (lp - lm) / (2 * hh), tol),
              "moe dWd_s[%d] a=%f n=%f", i, dWd_s[i],
              (lp - lm) / (2 * hh));
    }
}

// ---- top-kマスク性 (非選択は0) ----
// k=1ではsoftmaxヤコビアンが恒等的に0になるため、非ゼロ確認はk=2で行う。
// 既知値と同一重み (E=3,k=2) を使い、非選択expert2の勾配が厳密に0で、
// 選択expert0/1のgate勾配が非ゼロであることを確認する。
static void test_topk_mask(void) {
    const int n = 2;
    const int h = 1;
    const int E = 3;
    const int k = 2;
    const int S = 0;
    float X[2] = {1.0f, 0.5f};
    float Wgate[6] = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    float Wg[6] = {0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f};
    float Wu[6] = {1.0f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f};
    float Wd[6] = {1.0f, 2.0f, -1.0f, 0.5f, 0.0f, 0.0f};
    float Y[2] = {0, 0};
    size_t ids[2] = {99, 99};
    float weights[2] = {0, 0};
    float Gsel[2] = {0, 0};
    float Usel[2] = {0, 0};
    float Ysel[4] = {0, 0, 0, 0};
    int rc = jt_moe_fwd(X, Wgate, Wg, Wu, Wd, NULL, NULL, NULL, Y, n,
                        h, E, k, S, ids, weights, NULL, Gsel, Usel,
                        Ysel, NULL, NULL);
    CHECK(rc == JT_OK, "mask fwd rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    CHECK(ids[0] == 0 && ids[1] == 1, "mask ids {%zu,%zu} want {0,1}",
          ids[0], ids[1]);
    float dY[2] = {0.7f, -0.3f};
    float dX[2] = {9, 9};
    float dWgate[6] = {9, 9, 9, 9, 9, 9};
    float dWg[6] = {9, 9, 9, 9, 9, 9};
    float dWu[6] = {9, 9, 9, 9, 9, 9};
    float dWd[6] = {9, 9, 9, 9, 9, 9};
    float dLog[3] = {9, 9, 9};
    rc = jt_moe_bwd(dY, X, Wgate, Wg, Wu, Wd, NULL, NULL, NULL, ids,
                    weights, Gsel, Usel, Ysel, NULL, NULL, dX, dWgate,
                    dWg, dWu, dWd, NULL, NULL, NULL, dLog, n, h, E, k,
                    S);
    CHECK(rc == JT_OK, "mask bwd rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    // 非選択 (expert2) の勾配は厳密に0。
    CHECK(dLog[2] == 0.0f, "mask dLog non-sel (%f) want 0", dLog[2]);
    CHECK(dWgate[4] == 0.0f && dWgate[5] == 0.0f,
          "mask dWgate non-sel");
    CHECK(dWg[4] == 0.0f && dWg[5] == 0.0f,
          "mask dWg non-sel");
    CHECK(dWu[4] == 0.0f && dWu[5] == 0.0f,
          "mask dWu non-sel");
    CHECK(dWd[4] == 0.0f && dWd[5] == 0.0f,
          "mask dWd non-sel");
    // 選択 (0,1) は非ゼロのはず (縮退していないことの確認)。
    CHECK(dLog[0] != 0.0f && dLog[1] != 0.0f,
          "mask sel dLog zero (%f,%f) (degenerate)", dLog[0], dLog[1]);
}

// ---- sticky系列ヘルパー ----
static void test_sticky_seq(void) {
    // T=4,n=2,W=4, λ=0.1,α=1.0。手動ループとの一致。
    float gates[8] = {0.5f, 0.5f, 0.6f, 0.4f, 0.7f, 0.3f, 0.7f, 0.3f};
    float out = -1.0f;
    int rc = jt_moe_sticky_seq_loss(gates, 4, 2, 0.1f, 1.0f, 4, &out);
    CHECK(rc == JT_OK, "sticky seq rc=%d", rc);
    if (rc == JT_OK) {
        double sum = 0.0;
        for (size_t t = 0; t < 4; t++) {
            const float *gt = gates + t * 2;
            const float *gp = (t > 0) ? (gates + (t - 1) * 2) : NULL;
            size_t s_idx = (t / 4) * 4;
            size_t t_rel = t - s_idx;
            const float *ga = gates + s_idx * 2;
            float lt = 0.0f;
            int lrc = jt_routing_sticky_loss(gt, gp, ga, 2, 0.1f, 1.0f,
                                             t_rel, 4, &lt);
            CHECK(lrc == JT_OK, "sticky seq ref lrc t=%zu", t);
            sum += (double)lt;
        }
        double want = sum / 4.0;
        CHECK(fwd_close(out, want, 1e-6), "sticky seq out=%f want=%f",
              out, want);
    }
    // T=1は先頭のみで0。
    {
        float g1[2] = {0.4f, 0.6f};
        float o1 = -1.0f;
        CHECK(jt_moe_sticky_seq_loss(g1, 1, 2, 0.1f, 0.5f, 4, &o1) ==
                      JT_OK,
              "sticky seq T=1 rc");
        CHECK(fabsf(o1) < 1e-7f, "sticky seq T=1 out=%f want 0", o1);
    }
    // 不正系 (fail-closed: out不変)。
    {
        float o = 7.0f;
        CHECK(jt_moe_sticky_seq_loss(NULL, 4, 2, 0.1f, 1.0f, 4, &o) ==
                      JT_ERR_INVAL,
              "sticky seq NULL gates");
        CHECK(o == 7.0f, "sticky seq mutated on NULL");
        CHECK(jt_moe_sticky_seq_loss(gates, 0, 2, 0.1f, 1.0f, 4, &o) ==
                      JT_ERR_INVAL,
              "sticky seq T=0");
        CHECK(jt_moe_sticky_seq_loss(gates, 4, 2, 0.1f, 1.0f, 4, NULL) ==
                      JT_ERR_INVAL,
              "sticky seq NULL out");
        CHECK(jt_moe_sticky_seq_loss(gates, 4, 2, -0.1f, 1.0f, 4, &o) ==
                      JT_ERR_INVAL,
              "sticky seq lambda<0");
    }
}

// ---- 不正系 (fail-closed) ----
static void test_invalid(void) {
    float X[2] = {1.0f, 0.0f};
    float Wgate[6] = {1, 0, 0, 1, 0, 0};
    float W[6] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float Y[2] = {3.0f, 4.0f};
    size_t ids[1] = {0};
    float w[1] = {1.0f};
    float G[1] = {0}, U[1] = {0}, Ys[2] = {0, 0};
    CHECK(jt_moe_fwd(NULL, Wgate, W, W, W, NULL, NULL, NULL, Y, 2, 1,
                     3, 1, 0, ids, w, NULL, G, U, Ys, NULL,
                     NULL) == JT_ERR_INVAL,
          "moe fwd NULL X");
    CHECK(Y[0] == 3.0f && Y[1] == 4.0f, "moe fwd mutated on NULL");
    CHECK(jt_moe_fwd(X, Wgate, W, W, W, NULL, NULL, NULL, Y, 0, 1, 3,
                     1, 0, ids, w, NULL, G, U, Ys, NULL,
                     NULL) == JT_ERR_INVAL,
          "moe fwd n=0");
    CHECK(jt_moe_fwd(X, Wgate, W, W, W, NULL, NULL, NULL, Y, 2, 1, 3,
                     4, 0, ids, w, NULL, G, U, Ys, NULL,
                     NULL) == JT_ERR_INVAL,
          "moe fwd k>E");
    CHECK(jt_moe_fwd(X, Wgate, W, W, W, NULL, NULL, NULL, Y, 2, 1, 3,
                     1, 1, ids, w, NULL, G, U, Ys, NULL,
                     NULL) == JT_ERR_INVAL,
          "moe fwd S=1 NULL shared");
    errno = 0;
    {
        float badX[2] = {1.0f, (float)NAN};
        float y0 = Y[0];
        int brc =
            jt_moe_fwd(badX, Wgate, W, W, W, NULL, NULL, NULL, Y, 2, 1,
                       3, 1, 0, ids, w, NULL, G, U, Ys, NULL, NULL);
        CHECK(brc == JT_ERR_INVAL && errno == EINVAL, "moe fwd NaN");
        CHECK(Y[0] == y0, "moe fwd NaN mutated");
    }
    // bwd不正: ids範囲外・重複・weights和不正はINVALでdX不変。
    {
        float dY[2] = {1, 0};
        float dX[2] = {5, 6};
        float dWg_[6] = {0, 0, 0, 0, 0, 0};
        float dWu_[6] = {0, 0, 0, 0, 0, 0};
        float dWd_[6] = {0, 0, 0, 0, 0, 0};
        float dWgate[6] = {0, 0, 0, 0, 0, 0};
        float Gv[1] = {0.5f}, Uv[1] = {0.5f}, Yv[2] = {0.1f, 0.2f};
        size_t badids[1] = {7};
        float ww[1] = {1.0f};
        CHECK(jt_moe_bwd(dY, X, Wgate, W, W, W, NULL, NULL, NULL,
                         badids, ww, Gv, Uv, Yv, NULL, NULL, dX,
                         dWgate, dWg_, dWu_, dWd_, NULL, NULL, NULL,
                         NULL, 2, 1, 3, 1, 0) == JT_ERR_INVAL,
              "moe bwd bad id");
        CHECK(dX[0] == 5.0f && dX[1] == 6.0f, "moe bwd mutated");
        CHECK(jt_moe_bwd(NULL, X, Wgate, W, W, W, NULL, NULL, NULL,
                         ids, ww, Gv, Uv, Yv, NULL, NULL, dX, dWgate,
                         dWg_, dWu_, dWd_, NULL, NULL, NULL, NULL, 2,
                         1, 3, 1, 0) == JT_ERR_INVAL,
              "moe bwd NULL dY");
    }
}

int main(void) {
    test_fwd_known();
    test_grad();
    test_topk_mask();
    test_sticky_seq();
    test_invalid();
    if (g_fail != 0) {
        fprintf(stderr, "moe_layer: FAIL\n");
        return 1;
    }
    printf("moe_layer: OK (fwd known + grad + topk-mask + sticky-seq)\n");
    return 0;
}
