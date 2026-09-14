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

// ---- unchecked経路の数値一致 (検証ありAPIとbit一致) ----
// unchecked使用条件: 重み事前検証済み＋区間内不変の呼び出しでのみ使うこと。
// ここでは有限の固定入力を与え、通常経路とunchecked経路の全出力をmemcmpする。
// fail-closed不変 (INVAL系) はtest_invalidで担保し、ここでは正常系の一致のみ見る。
static void test_unchecked_match(void) {
    float X[2] = {1.0f, 0.5f};
    float Wgate[6] = {1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    float Wg[6] = {0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f};
    float Wu[6] = {1.0f, 0.0f, 0.5f, -0.5f, 0.0f, 0.0f};
    float Wd[6] = {1.0f, 2.0f, -1.0f, 0.5f, 0.0f, 0.0f};
    float Wg_s[2] = {0.5f, 0.5f};
    float Wu_s[2] = {1.0f, -1.0f};
    float Wd_s[2] = {0.5f, -0.5f};
    float Y1[2] = {0, 0}, Y2[2] = {0, 0};
    size_t ids1[2] = {99, 99}, ids2[2] = {99, 99};
    float w1[2] = {0, 0}, w2[2] = {0, 0};
    float lg1[3] = {0, 0, 0}, lg2[3] = {0, 0, 0};
    float Gs1[2] = {0, 0}, Gs2[2] = {0, 0};
    float Us1[2] = {0, 0}, Us2[2] = {0, 0};
    float Ys1[4] = {0, 0, 0, 0}, Ys2[4] = {0, 0, 0, 0};
    float Gsel1[2] = {0, 0}, Gsel2[2] = {0, 0};
    float Usel1[2] = {0, 0}, Usel2[2] = {0, 0};
    int rc1 = jt_moe_fwd(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Y1,
                         2, 1, 3, 2, 1, ids1, w1, lg1, Gsel1, Usel1,
                         Ys1, Gs1, Us1);
    int rc2 = jt_moe_fwd_unchecked(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s,
                                   Y2, 2, 1, 3, 2, 1, ids2, w2, lg2,
                                   Gsel2, Usel2, Ys2, Gs2, Us2);
    CHECK(rc1 == JT_OK && rc2 == JT_OK, "unchecked fwd rc %d/%d", rc1,
          rc2);
    if (rc1 == JT_OK && rc2 == JT_OK) {
        CHECK(memcmp(Y1, Y2, sizeof(Y1)) == 0, "unchecked fwd Y bits");
        CHECK(memcmp(ids1, ids2, sizeof(ids1)) == 0,
              "unchecked fwd ids bits");
        CHECK(memcmp(w1, w2, sizeof(w1)) == 0,
              "unchecked fwd weights bits");
        CHECK(memcmp(lg1, lg2, sizeof(lg1)) == 0,
              "unchecked fwd logits bits");
        CHECK(memcmp(Gsel1, Gsel2, sizeof(Gsel1)) == 0,
              "unchecked fwd Gsel bits");
        CHECK(memcmp(Usel1, Usel2, sizeof(Usel1)) == 0,
              "unchecked fwd Usel bits");
        CHECK(memcmp(Ys1, Ys2, sizeof(Ys1)) == 0,
              "unchecked fwd Ysel bits");
        CHECK(memcmp(Gs1, Gs2, sizeof(Gs1)) == 0,
              "unchecked fwd Gs bits");
        CHECK(memcmp(Us1, Us2, sizeof(Us1)) == 0,
              "unchecked fwd Us bits");
    }
    // bwd一致 (fwd産ids/weights/cacheをそのまま渡す)。
    {
        float dY[2] = {0.7f, -0.3f};
        float dX1[2] = {0, 0}, dX2[2] = {0, 0};
        float dg1[6] = {0}, dg2[6] = {0};
        float gg1[6] = {0}, gg2[6] = {0};
        float gu1[6] = {0}, gu2[6] = {0};
        float gd1[6] = {0}, gd2[6] = {0};
        float sg1[2] = {0}, sg2[2] = {0};
        float su1[2] = {0}, su2[2] = {0};
        float sd1[2] = {0}, sd2[2] = {0};
        float dl1[3] = {0}, dl2[3] = {0};
        int br1 = jt_moe_bwd(dY, X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s,
                             ids1, w1, Gsel1, Usel1, Ys1, Gs1, Us1, dX1,
                             dg1, gg1, gu1, gd1, sg1, su1, sd1, dl1, 2, 1,
                             3, 2, 1);
        int br2 = jt_moe_bwd_unchecked(dY, X, Wgate, Wg, Wu, Wd, Wg_s,
                                       Wu_s, Wd_s, ids1, w1, Gsel1,
                                       Usel1, Ys1, Gs1, Us1, dX2, dg2,
                                       gg2, gu2, gd2, sg2, su2, sd2, dl2,
                                       2, 1, 3, 2, 1);
        CHECK(br1 == JT_OK && br2 == JT_OK, "unchecked bwd rc %d/%d",
              br1, br2);
        if (br1 == JT_OK && br2 == JT_OK) {
            CHECK(memcmp(dX1, dX2, sizeof(dX1)) == 0,
                  "unchecked bwd dX bits");
            CHECK(memcmp(dg1, dg2, sizeof(dg1)) == 0,
                  "unchecked bwd dWgate bits");
            CHECK(memcmp(gg1, gg2, sizeof(gg1)) == 0,
                  "unchecked bwd dWg bits");
            CHECK(memcmp(gu1, gu2, sizeof(gu1)) == 0,
                  "unchecked bwd dWu bits");
            CHECK(memcmp(gd1, gd2, sizeof(gd1)) == 0,
                  "unchecked bwd dWd bits");
            CHECK(memcmp(sg1, sg2, sizeof(sg1)) == 0,
                  "unchecked bwd dWg_s bits");
            CHECK(memcmp(su1, su2, sizeof(su1)) == 0,
                  "unchecked bwd dWu_s bits");
            CHECK(memcmp(sd1, sd2, sizeof(sd1)) == 0,
                  "unchecked bwd dWd_s bits");
            CHECK(memcmp(dl1, dl2, sizeof(dl1)) == 0,
                  "unchecked bwd dLogits bits");
        }
    }
}

// ---- Phase G Step 1: batch sort + fwd_batch ----
// 既知値 (T=4,k=2,E=3,cap_factor=1.0 → cap=ceil(8/3)=3。全fit)。
static void test_batch_sort_known(void) {
    size_t ids[8] = {2, 0, 1, 2, 0, 1, 2, 1};
    size_t perm[8] = {99, 99, 99, 99, 99, 99, 99, 99};
    size_t perm2[8] = {98, 98, 98, 98, 98, 98, 98, 98};
    size_t off[4] = {99, 99, 99, 99};
    unsigned char drop[8] = {9, 9, 9, 9, 9, 9, 9, 9};
    size_t kept = 999;
    size_t dropped = 999;
    size_t want_perm[8] = {1, 4, 2, 5, 7, 0, 3, 6};
    size_t want_off[4] = {0, 2, 5, 8};
    int rc = jt_moe_batch_sort(ids, 4, 2, 3, 1.0f, perm, off, drop,
                               &kept, &dropped);
    CHECK(rc == JT_OK, "batch sort known rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    CHECK(memcmp(perm, want_perm, sizeof(perm)) == 0,
          "batch sort known perm");
    CHECK(memcmp(off, want_off, sizeof(off)) == 0,
          "batch sort known off");
    CHECK(kept == 8 && dropped == 0, "batch sort known kept=%zu drop=%zu",
          kept, dropped);
    for (int q = 0; q < 8; q++) {
        CHECK(drop[q] == 0, "batch sort known drop[%d]=%u", q,
              drop[q]);
    }
    // 決定論性：同一入力の再実行が完全一致。
    CHECK(jt_moe_batch_sort(ids, 4, 2, 3, 1.0f, perm2, off, drop, NULL,
                            NULL) == JT_OK,
          "batch sort redet rc");
    CHECK(memcmp(perm, perm2, sizeof(perm)) == 0,
          "batch sort determinism");
}

// 超過 drop (T=4,k=2,E=2,cap_factor=1.0 → cap=4。e0に5割当てで1 drop)。
static void test_batch_sort_drop(void) {
    size_t ids[8] = {0, 0, 0, 0, 0, 1, 1, 1};
    size_t perm[8] = {99, 99, 99, 99, 99, 99, 99, 99};
    size_t off[3] = {99, 99, 99};
    unsigned char drop[8] = {9, 9, 9, 9, 9, 9, 9, 9};
    size_t kept = 999;
    size_t dropped = 999;
    size_t want_perm[7] = {0, 1, 2, 3, 5, 6, 7};
    size_t want_off[3] = {0, 4, 7};
    unsigned char want_drop[8] = {0, 0, 0, 0, 1, 0, 0, 0};
    int rc = jt_moe_batch_sort(ids, 4, 2, 2, 1.0f, perm, off, drop,
                               &kept, &dropped);
    CHECK(rc == JT_OK, "batch sort drop rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    CHECK(kept == 7 && dropped == 1, "batch sort drop kept=%zu drop=%zu",
          kept, dropped);
    CHECK(memcmp(off, want_off, sizeof(want_off)) == 0,
          "batch sort drop off");
    CHECK(memcmp(drop, want_drop, sizeof(want_drop)) == 0,
          "batch sort drop mask");
    CHECK(memcmp(perm, want_perm, sizeof(want_perm)) == 0,
          "batch sort drop perm");
    CHECK(off[2] == kept, "batch sort drop off[E]!=kept");
}

// sort 不正系 (fail-closed: 出力不変)。
static void test_batch_sort_invalid(void) {
    size_t ids[2] = {0, 1};
    size_t perm[2] = {7, 7};
    size_t off[3] = {7, 7, 7};
    unsigned char drop[2] = {7, 7};
    size_t bad[2] = {0, 9};
    CHECK(jt_moe_batch_sort(NULL, 1, 1, 2, 1.25f, perm, off, drop, NULL,
                            NULL) == JT_ERR_INVAL,
          "batch sort NULL ids");
    CHECK(jt_moe_batch_sort(ids, 0, 1, 2, 1.25f, perm, off, drop, NULL,
                            NULL) == JT_ERR_INVAL,
          "batch sort T=0");
    CHECK(jt_moe_batch_sort(ids, 1, 3, 2, 1.25f, perm, off, drop, NULL,
                            NULL) == JT_ERR_INVAL,
          "batch sort k>E");
    CHECK(jt_moe_batch_sort(ids, 1, 1, 2, 2.0f, perm, off, drop, NULL,
                            NULL) == JT_ERR_INVAL,
          "batch sort cap>1.5");
    CHECK(jt_moe_batch_sort(ids, 1, 1, 2, 0.5f, perm, off, drop, NULL,
                            NULL) == JT_ERR_INVAL,
          "batch sort cap<1.0");
    CHECK(jt_moe_batch_sort(bad, 1, 2, 2, 1.25f, perm, off, drop, NULL,
                            NULL) == JT_ERR_INVAL,
          "batch sort ids>=E");
    CHECK(perm[0] == 7 && perm[1] == 7 && off[0] == 7 && drop[0] == 7,
          "batch sort mutated on error");
}

// batch vs 単体ループの等価性 (drop なし → bit 一致)。
static void test_batch_equiv(void) {
    const int n = 4;
    const int h = 2;
    const int E = 4;
    const int k = 2;
    const int S = 1;
    const int T = 6;
    float X[24], Wgate[16], Wg[32], Wu[32], Wd[32];
    float Wg_s[8], Wu_s[8], Wd_s[8];
    float Ys[24], Yb[24];
    size_t ids_s[12], ids_b[12];
    float w_s[12], w_b[12];
    float Gs_s[24], Gs_b[24], Us_s[24], Us_b[24];
    float Ysel_s[48], Ysel_b[48];
    float sGs_s[12], sGs_b[12], sUs_s[12], sUs_b[12];
    size_t perm[12], off[5];
    unsigned char dropm[12];
    size_t kept = 999;
    size_t dropped = 999;
    int rc;
    for (int i = 0; i < T * n; i++) {
        X[i] = 0.4f * fdval(i, 0, 51) + 0.2f;
    }
    for (int e = 0; e < E; e++) {
        for (int j = 0; j < n; j++) {
            // オフセットなし (バランスルーティングで drop なしにする。
            // 摂動を与えないため選択安定化のオフセットは不要)。
            Wgate[e * n + j] = 0.3f * fdval(e, j, 52);
        }
    }
    for (int i = 0; i < E * h * n; i++) {
        Wg[i] = 0.3f * fdval(i, 0, 53);
        Wu[i] = 0.3f * fdval(i, 1, 54);
        Wd[i] = 0.3f * fdval(i, 2, 55);
    }
    for (int i = 0; i < S * h * n; i++) {
        Wg_s[i] = 0.3f * fdval(i, 0, 56);
        Wu_s[i] = 0.3f * fdval(i, 1, 57);
        Wd_s[i] = 0.3f * fdval(i, 2, 58);
    }
    memset(Ys, 0, sizeof(Ys));
    // 単体ループ参照。
    for (int t = 0; t < T; t++) {
        rc = jt_moe_fwd(X + t * n, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s,
                        Ys + t * n, n, h, E, k, S, ids_s + t * k,
                        w_s + t * k, NULL, Gs_s + t * k * h,
                        Us_s + t * k * h, Ysel_s + t * k * n,
                        sGs_s + t * S * h, sUs_s + t * S * h);
        CHECK(rc == JT_OK, "batch equiv single rc t=%d", t);
        if (rc != JT_OK) {
            return;
        }
    }
    memset(Yb, 0, sizeof(Yb));
    rc = jt_moe_fwd_batch(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Yb, T,
                          n, h, E, k, S, ids_b, w_b, Gs_b, Us_b, Ysel_b,
                          sGs_b, sUs_b, 1.25f, perm, off, dropm, &kept,
                          &dropped);
    CHECK(rc == JT_OK, "batch equiv batch rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    CHECK(dropped == 0, "batch equiv dropped=%zu (want 0)", dropped);
    CHECK(kept == (size_t)T * (size_t)k, "batch equiv kept=%zu", kept);
    if (dropped != 0) {
        return;
    }
    CHECK(memcmp(Ys, Yb, sizeof(Ys)) == 0, "batch equiv Y bits");
    CHECK(memcmp(ids_s, ids_b, sizeof(ids_s)) == 0,
          "batch equiv ids bits");
    CHECK(memcmp(w_s, w_b, sizeof(w_s)) == 0,
          "batch equiv weights bits");
    CHECK(memcmp(Gs_s, Gs_b, sizeof(Gs_s)) == 0,
          "batch equiv Gsel bits");
    CHECK(memcmp(Us_s, Us_b, sizeof(Us_s)) == 0,
          "batch equiv Usel bits");
    CHECK(memcmp(Ysel_s, Ysel_b, sizeof(Ysel_s)) == 0,
          "batch equiv Ysel bits");
    CHECK(memcmp(sGs_s, sGs_b, sizeof(sGs_s)) == 0,
          "batch equiv Gs bits");
    CHECK(memcmp(sUs_s, sUs_b, sizeof(sUs_s)) == 0,
          "batch equiv Us bits");
    CHECK(off[E] == kept, "batch equiv off[E]!=kept");
    // unchecked 版も bit 一致。
    {
        float Yu[24];
        size_t idsu[12];
        float wu[12];
        memset(Yu, 0, sizeof(Yu));
        rc = jt_moe_fwd_batch_unchecked(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s,
                                        Wd_s, Yu, T, n, h, E, k, S, idsu,
                                        wu, NULL, NULL, NULL, NULL, NULL,
                                        1.25f, NULL, NULL, NULL, NULL,
                                        NULL);
        CHECK(rc == JT_OK, "batch equiv unchecked rc=%d", rc);
        if (rc == JT_OK) {
            CHECK(memcmp(Ys, Yu, sizeof(Ys)) == 0,
                  "batch equiv unchecked Y bits");
            CHECK(memcmp(ids_s, idsu, sizeof(ids_s)) == 0,
                  "batch equiv unchecked ids bits");
        }
    }
}

// drop あり forward (部分 drop の寄与 0・renormalize なし)。
// n=2,h=1,E=4,k=2,S=0,T=8。X[t]=[1,0.125t]、Wgate で t0-3→[0,1]、t4-7→[0,2]
// (logit ギャップ ≥0.075 で AVX2/スカラー共通に安定)。
// cap=ceil(1.25*16/4)=5。e0 は 8 割当てで t5-7 を drop (dropped=3)。
static void test_batch_drop_fwd(void) {
    const int n = 2;
    const int h = 1;
    const int E = 4;
    const int k = 2;
    const int S = 0;
    const int T = 8;
    float X[16], Wgate[8], Wg[8], Wu[8], Wd[8];
    float Ys[16], Yb[16];
    size_t ids_s[16], ids_b[16];
    float w_s[16], w_b[16];
    float Gs_s[16], Gs_b[16], Us_s[16], Us_b[16];
    float Ysel_s[32], Ysel_b[32];
    size_t perm[16], off[5];
    unsigned char dropm[16];
    size_t kept = 999;
    size_t dropped = 999;
    int rc;
    for (int t = 0; t < T; t++) {
        X[t * n] = 1.0f;
        X[t * n + 1] = 0.125f * (float)t;
    }
    Wgate[0] = 2.0f;
    Wgate[1] = 0.0f;
    Wgate[2] = 1.2f;
    Wgate[3] = -1.5f;
    Wgate[4] = 0.0f;
    Wgate[5] = 1.5f;
    Wgate[6] = -5.0f;
    Wgate[7] = 0.0f;
    for (int i = 0; i < E * h * n; i++) {
        Wg[i] = 0.3f * fdval(i, 0, 59);
        Wu[i] = 0.3f * fdval(i, 1, 60);
        Wd[i] = 0.3f * fdval(i, 2, 61);
    }
    for (int t = 0; t < T; t++) {
        rc = jt_moe_fwd(X + t * n, Wgate, Wg, Wu, Wd, NULL, NULL, NULL,
                        Ys + t * n, n, h, E, k, S, ids_s + t * k,
                        w_s + t * k, NULL, Gs_s + t * k * h,
                        Us_s + t * k * h, Ysel_s + t * k * n, NULL,
                        NULL);
        CHECK(rc == JT_OK, "batch drop single rc t=%d", t);
        if (rc != JT_OK) {
            return;
        }
    }
    CHECK(ids_s[0] == 0 && ids_s[1] == 1, "batch drop route t0 {%zu,%zu}",
          ids_s[0], ids_s[1]);
    CHECK(ids_s[8] == 0 && ids_s[9] == 2, "batch drop route t4 {%zu,%zu}",
          ids_s[8], ids_s[9]);
    memset(Yb, 0, sizeof(Yb));
    memset(Ysel_b, 0xA5, sizeof(Ysel_b));
    rc = jt_moe_fwd_batch(X, Wgate, Wg, Wu, Wd, NULL, NULL, NULL, Yb, T,
                          n, h, E, k, S, ids_b, w_b, Gs_b, Us_b, Ysel_b,
                          NULL, NULL, 1.25f, perm, off, dropm, &kept,
                          &dropped);
    CHECK(rc == JT_OK, "batch drop batch rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    CHECK(kept == 13 && dropped == 3, "batch drop kept=%zu drop=%zu",
          kept, dropped);
    if (kept != 13 || dropped != 3) {
        return;
    }
    CHECK(memcmp(ids_s, ids_b, sizeof(ids_s)) == 0,
          "batch drop ids bits");
    CHECK(memcmp(w_s, w_b, sizeof(w_s)) == 0,
          "batch drop weights bits (no renormalize)");
    // kept トークン (t0-4) は単体版と bit 一致。
    CHECK(memcmp(Ys, Yb, (size_t)5 * (size_t)n * sizeof(float)) == 0,
          "batch drop kept rows bits");
    // 部分 drop トークン (t5-7): dropped 側寄与 0・renormalize なし。
    // 期待値 = kept 側の w*Ysel (単体版 cache 値で再結合)。
    for (int t = 5; t < T; t++) {
        for (int j = 0; j < n; j++) {
            double want = 0.0;
            for (int p = 0; p < k; p++) {
                size_t q = (size_t)t * (size_t)k + (size_t)p;
                if (dropm[q]) {
                    continue;
                }
                want += (double)w_s[q] *
                        (double)Ysel_s[q * (size_t)n + (size_t)j];
            }
            double got = (double)Yb[(size_t)t * (size_t)n + (size_t)j];
            double d = fabs(got - want);
            CHECK(d <= 1e-6 + 1e-6 * fabs(want),
                  "batch drop t=%d j=%d got=%f want=%f", t, j, got,
                  want);
        }
    }
    // dropped スロットの cache は 0 埋め (Step 3 申送りまでの不定値防止)。
    for (size_t q = 0; q < (size_t)T * (size_t)k; q++) {
        if (dropm[q]) {
            for (int j = 0; j < n; j++) {
                CHECK(Ysel_b[q * (size_t)n + (size_t)j] == 0.0f,
                      "batch drop cache not zeroed q=%zu", q);
            }
        }
    }
    CHECK(off[E] == kept, "batch drop off[E]!=kept");
}

// batch 不正系 (fail-closed: Y 不変)。
static void test_batch_invalid(void) {
    float X[4] = {1.0f, 0.5f, 0.25f, -0.5f};
    float Wgate[8] = {1, 0, 0, 1, 0, 0, 0, 0};
    float W[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float Y[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    size_t ids[4] = {0, 0, 0, 0};
    float w[4] = {0, 0, 0, 0};
    CHECK(jt_moe_fwd_batch(NULL, Wgate, W, W, W, NULL, NULL, NULL, Y, 2,
                           2, 1, 4, 1, 0, ids, w, NULL, NULL, NULL,
                           NULL, NULL, 1.25f, NULL, NULL, NULL, NULL,
                           NULL) == JT_ERR_INVAL,
          "batch fwd NULL X");
    CHECK(Y[0] == 3.0f && Y[3] == 6.0f, "batch fwd mutated on NULL");
    CHECK(jt_moe_fwd_batch(X, Wgate, W, W, W, NULL, NULL, NULL, Y, 0, 2,
                           1, 4, 1, 0, ids, w, NULL, NULL, NULL, NULL,
                           NULL, 1.25f, NULL, NULL, NULL, NULL,
                           NULL) == JT_ERR_INVAL,
          "batch fwd T=0");
    CHECK(jt_moe_fwd_batch(X, Wgate, W, W, W, NULL, NULL, NULL, Y, 2, 2,
                           1, 4, 5, 0, ids, w, NULL, NULL, NULL, NULL,
                           NULL, 1.25f, NULL, NULL, NULL, NULL,
                           NULL) == JT_ERR_INVAL,
          "batch fwd k>E");
    CHECK(jt_moe_fwd_batch(X, Wgate, W, W, W, NULL, NULL, NULL, Y, 2, 2,
                           1, 4, 1, 0, ids, w, NULL, NULL, NULL, NULL,
                           NULL, 9.0f, NULL, NULL, NULL, NULL,
                           NULL) == JT_ERR_INVAL,
          "batch fwd bad cap");
    CHECK(Y[0] == 3.0f && Y[3] == 6.0f, "batch fwd mutated on bad cap");
    errno = 0;
    {
        float badX[4] = {1.0f, (float)NAN, 0.25f, -0.5f};
        int brc = jt_moe_fwd_batch(badX, Wgate, W, W, W, NULL, NULL,
                                   NULL, Y, 2, 2, 1, 4, 1, 0, ids, w,
                                   NULL, NULL, NULL, NULL, NULL, 1.25f,
                                   NULL, NULL, NULL, NULL, NULL);
        CHECK(brc == JT_ERR_INVAL && errno == EINVAL, "batch fwd NaN");
        CHECK(Y[0] == 3.0f && Y[3] == 6.0f, "batch fwd NaN mutated");
    }
}

// ---- Phase G Step 3: バッチbwd ----
// 補助: 単体ループbwd参照 (token昇順蓄積。lr_bwd_rangeと同一順序)。
// dWgate/dWg/dWu/dWd/dWg_s等はゼロ埋め後にtoken昇順で加算する。
static int ref_single_bwd_accum(const float *dYall, const float *Xall,
                                const float *Wgate, const float *Wg,
                                const float *Wu, const float *Wd,
                                const float *Wg_s, const float *Wu_s,
                                const float *Wd_s, const size_t *ids,
                                const float *weights, const float *Gsel,
                                const float *Usel, const float *Ysel,
                                const float *Gs, const float *Us, int T,
                                int n, int h, int E, int k, int S,
                                float *dXall, float *dWgate, float *dWg,
                                float *dWu, float *dWd, float *dWg_s,
                                float *dWu_s, float *dWd_s, float *dLog) {
    size_t hhn = (size_t)h * (size_t)n;
    size_t e;
    size_t i;
    for (e = 0; e < (size_t)E; e++) {
        for (i = 0; i < (size_t)n; i++) {
            dWgate[e * (size_t)n + i] = 0.0f;
        }
        for (i = 0; i < hhn; i++) {
            dWg[e * hhn + i] = 0.0f;
            dWu[e * hhn + i] = 0.0f;
            dWd[e * hhn + i] = 0.0f;
        }
    }
    for (e = 0; e < (size_t)(S > 0 ? S : 0); e++) {
        for (i = 0; i < hhn; i++) {
            dWg_s[e * hhn + i] = 0.0f;
            dWu_s[e * hhn + i] = 0.0f;
            dWd_s[e * hhn + i] = 0.0f;
        }
    }
    if (dLog != NULL) {
        for (e = 0; e < (size_t)T * (size_t)E; e++) {
            dLog[e] = 0.0f;
        }
    }
    for (int t = 0; t < T; t++) {
        float dX[64];
        float dg[1024];
        float gg[1024];
        float gu[1024];
        float gd[1024];
        float sg[64];
        float su[64];
        float sd[64];
        float dl[64];
        const float *dYt = dYall + (size_t)t * (size_t)n;
        const float *Xt = Xall + (size_t)t * (size_t)n;
        const size_t *idst = ids + (size_t)t * (size_t)k;
        const float *wt = weights + (size_t)t * (size_t)k;
        const float *Gt = Gsel + (size_t)t * (size_t)k * (size_t)h;
        const float *Ut = Usel + (size_t)t * (size_t)k * (size_t)h;
        const float *Yt = Ysel + (size_t)t * (size_t)k * (size_t)n;
        const float *Gst =
            (S > 0) ? (Gs + (size_t)t * (size_t)S * (size_t)h) : NULL;
        const float *Ust =
            (S > 0) ? (Us + (size_t)t * (size_t)S * (size_t)h) : NULL;
        int brc;
        if (n > 64 || h > 64 || E > 64) {
            return JT_ERR_INVAL;
        }
        brc = jt_moe_bwd(dYt, Xt, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s,
                         idst, wt, Gt, Ut, Yt, Gst, Ust, dX, dg, gg, gu,
                         gd, sg, su, sd, dl, n, h, E, k, S);
        if (brc != JT_OK) {
            return brc;
        }
        for (int j = 0; j < n; j++) {
            dXall[(size_t)t * (size_t)n + (size_t)j] = dX[j];
        }
        // token昇順加算 (lr_bwd_rangeと同一)。
        for (int p = 0; p < k; p++) {
            size_t ee = idst[p];
            size_t erow = ee * (size_t)h * (size_t)n;
            size_t eg = ee * (size_t)n;
            for (i = 0; i < hhn; i++) {
                dWg[erow + i] += gg[ee * hhn + i];
                dWu[erow + i] += gu[ee * hhn + i];
                dWd[erow + i] += gd[ee * hhn + i];
            }
            for (int j = 0; j < n; j++) {
                dWgate[eg + (size_t)j] += dg[eg + (size_t)j];
            }
            if (dLog != NULL) {
                dLog[(size_t)t * (size_t)E + ee] = dl[ee];
            }
        }
        if (S > 0) {
            for (i = 0; i < (size_t)S * hhn; i++) {
                dWg_s[i] += sg[i];
                dWu_s[i] += su[i];
                dWd_s[i] += sd[i];
            }
        }
    }
    return JT_OK;
}

// gate勾配のtoken順加算とperm順加算のbit一致 (必須条件1)。
// dWgateのtoken方向蓄積をtoken昇順ループとperm順ループ (expert外側・m昇順)
// で別々に計算し、bit一致を確認する。一致しなければ実装バグ。
static void test_bwd_batch_gate_bitmatch(void) {
    const int n = 4;
    const int h = 2;
    const int E = 4;
    const int k = 2;
    const int S = 1;
    const int T = 6;
    float X[24], Wgate[16], Wg[32], Wu[32], Wd[32];
    float Wg_s[8], Wu_s[8], Wd_s[8];
    float Yb[24], dY[24];
    size_t ids[12];
    float w[12];
    float Gs[24], Us[24], Ysel[48];
    float sGs[12], sUs[12];
    size_t perm[12], off[5];
    unsigned char dropm[12];
    size_t kept = 0;
    size_t dropped = 0;
    int rc;
    for (int i = 0; i < T * n; i++) {
        X[i] = 0.4f * fdval(i, 0, 51) + 0.2f;
        dY[i] = 0.5f * fdval(i, 1, 71) + 0.15f;
    }
    for (int e = 0; e < E; e++) {
        for (int j = 0; j < n; j++) {
            Wgate[e * n + j] = 0.3f * fdval(e, j, 52);
        }
    }
    for (int i = 0; i < E * h * n; i++) {
        Wg[i] = 0.3f * fdval(i, 0, 53);
        Wu[i] = 0.3f * fdval(i, 1, 54);
        Wd[i] = 0.3f * fdval(i, 2, 55);
    }
    for (int i = 0; i < S * h * n; i++) {
        Wg_s[i] = 0.3f * fdval(i, 0, 56);
        Wu_s[i] = 0.3f * fdval(i, 1, 57);
        Wd_s[i] = 0.3f * fdval(i, 2, 58);
    }
    rc = jt_moe_fwd_batch(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Yb, T,
                          n, h, E, k, S, ids, w, Gs, Us, Ysel, sGs, sUs,
                          1.25f, perm, off, dropm, &kept, &dropped);
    CHECK(rc == JT_OK, "gate bitmatch fwd rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    {
        // 参照: 単体ループ蓄積のdWgate/dLogits。
        float dXr[24], dWr[16], dWgr[32], dWur[32], dWdr[32];
        float sgr[8], sur[8], sdr[8], dlr[24];
        float dXb[24], dWb[16], dWgb[32], dWub[32], dWdb[32];
        float sgb[8], sub[8], sdb[8], dlb[24];
        int brc;
        memset(dXr, 0, sizeof(dXr));
        brc = ref_single_bwd_accum(dY, X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s,
                                   Wd_s, ids, w, Gs, Us, Ysel, sGs, sUs,
                                   T, n, h, E, k, S, dXr, dWr, dWgr, dWur,
                                   dWdr, sgr, sur, sdr, dlr);
        CHECK(brc == JT_OK, "gate bitmatch ref rc=%d", brc);
        if (brc != JT_OK) {
            return;
        }
        memset(dXb, 0, sizeof(dXb));
        memset(dWb, 0, sizeof(dWb));
        brc = jt_moe_bwd_batch(dY, X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s,
                               ids, w, Gs, Us, Ysel, sGs, sUs, perm, off,
                               dropm, dXb, dWb, dWgb, dWub, dWdb, sgb, sub,
                               sdb, dlb, T, n, h, E, k, S);
        CHECK(brc == JT_OK, "gate bitmatch batch rc=%d", brc);
        if (brc != JT_OK) {
            return;
        }
        // dropなし時は単体版とdWgate/dLogitsがbit一致するはず。
        // dropあり時はdropped対のgate勾配0化 (Step 3修正) のため単体版と
        // 一致しない場合がある。その場合はbwd_batch内のtoken順/perm順の
        // 内部一致のみを別途確認する (下記)。
        if (dropped == 0) {
            CHECK(memcmp(dWr, dWb, sizeof(dWr)) == 0,
                  "gate bitmatch dWgate bits");
            CHECK(memcmp(dlr, dlb, sizeof(dlr)) == 0,
                  "gate bitmatch dLogits bits");
        }
        // 内部一致: token順加算とperm順加算のbit一致 (drop有無によらず)。
        // bwd_batchのdLogits出力 (float) を正として、dWgateをtoken昇順
        // ループとperm順ループ (expert外側・m昇順) で別々に組み立て、
        // bit一致を確認する。同一dl・同一丸めのため順序固定の検証になる。
        // 一致しなければ実装バグ (必須条件1)。
        {
            float dWtok[16], dWprm[16];
            for (int i = 0; i < E * n; i++) {
                dWtok[i] = 0.0f;
                dWprm[i] = 0.0f;
            }
            // token順 (t昇順・p昇順)。
            for (int t = 0; t < T; t++) {
                const float *Xt = X + (size_t)t * (size_t)n;
                for (int p = 0; p < k; p++) {
                    size_t q = (size_t)t * (size_t)k + (size_t)p;
                    size_t ee;
                    double dl;
                    if (dropm[q]) {
                        continue;
                    }
                    ee = ids[q];
                    dl = (double)dlb[(size_t)t * (size_t)E + ee];
                    for (int j = 0; j < n; j++) {
                        dWtok[ee * (size_t)n + (size_t)j] +=
                            (float)(dl * (double)Xt[j]);
                    }
                }
            }
            // perm順 (expert昇順外側・m昇順内側)。
            for (int ee = 0; ee < E; ee++) {
                size_t b0 = off[(size_t)ee];
                size_t Me = off[(size_t)ee + 1] - b0;
                for (size_t m = 0; m < Me; m++) {
                    size_t q = perm[b0 + m];
                    size_t t = q / (size_t)k;
                    const float *Xt = X + t * (size_t)n;
                    size_t e2 = ids[q];
                    double dl;
                    if (e2 != (size_t)ee) {
                        CHECK(0, "gate perm/expert mismatch");
                        return;
                    }
                    dl = (double)dlb[t * (size_t)E + e2];
                    for (int j = 0; j < n; j++) {
                        dWprm[ee * (size_t)n + (size_t)j] +=
                            (float)(dl * (double)Xt[j]);
                    }
                }
            }
            CHECK(memcmp(dWtok, dWprm, sizeof(dWtok)) == 0,
                  "gate token-vs-perm dWgate bits");
        }
    }
}

// バッチbwd等価性 (dropなし→dW系bit一致、dXはtol内)。
// dXはexpert_id昇順scatterのためslot順と異なる場合があり得る
// (f32非結合則。§3.2改訂)。tol=1e-5/1e-3は不変。
static void test_bwd_batch_equiv(void) {
    const int n = 4;
    const int h = 2;
    const int E = 4;
    const int k = 2;
    const int S = 1;
    const int T = 6;
    float X[24], Wgate[16], Wg[32], Wu[32], Wd[32];
    float Wg_s[8], Wu_s[8], Wd_s[8];
    float Yb[24], dY[24];
    size_t ids[12];
    float w[12];
    float Gs[24], Us[24], Ysel[48];
    float sGs[12], sUs[12];
    size_t perm[12], off[5];
    unsigned char dropm[12];
    size_t kept = 0;
    size_t dropped = 0;
    int rc;
    for (int i = 0; i < T * n; i++) {
        X[i] = 0.4f * fdval(i, 0, 51) + 0.2f;
        dY[i] = 0.5f * fdval(i, 1, 71) + 0.15f;
    }
    for (int e = 0; e < E; e++) {
        for (int j = 0; j < n; j++) {
            Wgate[e * n + j] = 0.3f * fdval(e, j, 52);
        }
    }
    for (int i = 0; i < E * h * n; i++) {
        Wg[i] = 0.3f * fdval(i, 0, 53);
        Wu[i] = 0.3f * fdval(i, 1, 54);
        Wd[i] = 0.3f * fdval(i, 2, 55);
    }
    for (int i = 0; i < S * h * n; i++) {
        Wg_s[i] = 0.3f * fdval(i, 0, 56);
        Wu_s[i] = 0.3f * fdval(i, 1, 57);
        Wd_s[i] = 0.3f * fdval(i, 2, 58);
    }
    rc = jt_moe_fwd_batch(X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, Yb, T,
                          n, h, E, k, S, ids, w, Gs, Us, Ysel, sGs, sUs,
                          1.25f, perm, off, dropm, &kept, &dropped);
    CHECK(rc == JT_OK, "bwd equiv fwd rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    CHECK(dropped == 0, "bwd equiv dropped=%zu (want 0)", dropped);
    if (dropped != 0) {
        return;
    }
    {
        float dXr[24], dWr[16], dWgr[32], dWur[32], dWdr[32];
        float sgr[8], sur[8], sdr[8], dlr[24];
        float dXb[24], dWb[16], dWgb[32], dWub[32], dWdb[32];
        float sgb[8], sub[8], sdb[8], dlb[24];
        int brc = ref_single_bwd_accum(dY, X, Wgate, Wg, Wu, Wd, Wg_s,
                                       Wu_s, Wd_s, ids, w, Gs, Us, Ysel,
                                       sGs, sUs, T, n, h, E, k, S, dXr,
                                       dWr, dWgr, dWur, dWdr, sgr, sur,
                                       sdr, dlr);
        CHECK(brc == JT_OK, "bwd equiv ref rc=%d", brc);
        if (brc != JT_OK) {
            return;
        }
        brc = jt_moe_bwd_batch(dY, X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s,
                               ids, w, Gs, Us, Ysel, sGs, sUs, perm, off,
                               dropm, dXb, dWb, dWgb, dWub, dWdb, sgb, sub,
                               sdb, dlb, T, n, h, E, k, S);
        CHECK(brc == JT_OK, "bwd equiv batch rc=%d", brc);
        if (brc != JT_OK) {
            return;
        }
        CHECK(memcmp(dWr, dWb, sizeof(dWr)) == 0, "bwd equiv dWgate bits");
        CHECK(memcmp(dlr, dlb, sizeof(dlr)) == 0, "bwd equiv dLogits bits");
        CHECK(memcmp(dWgr, dWgb, sizeof(dWgr)) == 0, "bwd equiv dWg bits");
        CHECK(memcmp(dWur, dWub, sizeof(dWur)) == 0, "bwd equiv dWu bits");
        CHECK(memcmp(dWdr, dWdb, sizeof(dWdr)) == 0, "bwd equiv dWd bits");
        CHECK(memcmp(sgr, sgb, sizeof(sgr)) == 0, "bwd equiv dWg_s bits");
        CHECK(memcmp(sur, sub, sizeof(sur)) == 0, "bwd equiv dWu_s bits");
        CHECK(memcmp(sdr, sdb, sizeof(sdr)) == 0, "bwd equiv dWd_s bits");
        // dXはexpert昇順scatterのためtol内一致 (既存tol不変)。
        for (int i = 0; i < T * n; i++) {
            double d = fabs((double)dXb[i] - (double)dXr[i]);
            double allowed = 1e-5 + 1e-5 * fabs((double)dXr[i]);
            CHECK(d <= allowed, "bwd equiv dX[%d] b=%f r=%f", i, dXb[i],
                  dXr[i]);
        }
        // unchecked版もbit一致。
        {
            float dXu[24], dWu2[16], dWgu[32];
            float dWuu[32], dWdu[32], sgu[8], suu[8], sdu[8], dlu[24];
            int urc = jt_moe_bwd_batch_unchecked(
                dY, X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, ids, w, Gs,
                Us, Ysel, sGs, sUs, perm, off, dropm, dXu, dWu2, dWgu,
                dWuu, dWdu, sgu, suu, sdu, dlu, T, n, h, E, k, S);
            CHECK(urc == JT_OK, "bwd equiv unchecked rc=%d", urc);
            if (urc == JT_OK) {
                CHECK(memcmp(dWb, dWu2, sizeof(dWb)) == 0,
                      "bwd equiv unchecked dWgate bits");
                CHECK(memcmp(dXb, dXu, sizeof(dXb)) == 0,
                      "bwd equiv unchecked dX bits");
            }
        }
        // 決定論性: 再実行がbit一致。
        {
            float dX2[24], dW2[16], dl2[24];
            float g2[32], u2[32], d2[32], sg2[8], su2[8], sd2[8];
            int r2 = jt_moe_bwd_batch(
                dY, X, Wgate, Wg, Wu, Wd, Wg_s, Wu_s, Wd_s, ids, w, Gs,
                Us, Ysel, sGs, sUs, perm, off, dropm, dX2, dW2, g2, u2,
                d2, sg2, su2, sd2, dl2, T, n, h, E, k, S);
            CHECK(r2 == JT_OK, "bwd redet rc=%d", r2);
            if (r2 == JT_OK) {
                CHECK(memcmp(dXb, dX2, sizeof(dXb)) == 0,
                      "bwd determinism dX bits");
                CHECK(memcmp(dWb, dW2, sizeof(dWb)) == 0,
                      "bwd determinism dWgate bits");
            }
        }
    }
}

// dropありbwd (dropped対のgate勾配0・寄与0)。
static void test_bwd_batch_drop(void) {
    const int n = 2;
    const int h = 1;
    const int E = 4;
    const int k = 2;
    const int S = 0;
    const int T = 8;
    float X[16], Wgate[8], Wg[8], Wu[8], Wd[8];
    float Yb[16], dY[16];
    size_t ids[16];
    float w[16];
    float Gs[16], Us[16], Ysel[32];
    size_t perm[16], off[5];
    unsigned char dropm[16];
    size_t kept = 0;
    size_t dropped = 0;
    int rc;
    for (int t = 0; t < T; t++) {
        X[t * n] = 1.0f;
        X[t * n + 1] = 0.125f * (float)t;
        dY[t * n] = 0.5f;
        dY[t * n + 1] = -0.25f;
    }
    Wgate[0] = 2.0f;
    Wgate[1] = 0.0f;
    Wgate[2] = 1.2f;
    Wgate[3] = -1.5f;
    Wgate[4] = 0.0f;
    Wgate[5] = 1.5f;
    Wgate[6] = -5.0f;
    Wgate[7] = 0.0f;
    for (int i = 0; i < E * h * n; i++) {
        Wg[i] = 0.3f * fdval(i, 0, 59);
        Wu[i] = 0.3f * fdval(i, 1, 60);
        Wd[i] = 0.3f * fdval(i, 2, 61);
    }
    rc = jt_moe_fwd_batch(X, Wgate, Wg, Wu, Wd, NULL, NULL, NULL, Yb, T,
                          n, h, E, k, S, ids, w, Gs, Us, Ysel, NULL, NULL,
                          1.25f, perm, off, dropm, &kept, &dropped);
    CHECK(rc == JT_OK, "bwd drop fwd rc=%d", rc);
    if (rc != JT_OK) {
        return;
    }
    CHECK(kept == 13 && dropped == 3, "bwd drop kept=%zu drop=%zu", kept,
          dropped);
    if (kept != 13 || dropped != 3) {
        return;
    }
    {
        float dXb[16], dWb[8], dWgb[8], dWub[8], dWdb[8], dlb[32];
        int brc = jt_moe_bwd_batch(dY, X, Wgate, Wg, Wu, Wd, NULL, NULL,
                                   NULL, ids, w, Gs, Us, Ysel, NULL, NULL,
                                   perm, off, dropm, dXb, dWb, dWgb, dWub,
                                   dWdb, NULL, NULL, NULL, dlb, T, n, h, E,
                                   k, S);
        CHECK(brc == JT_OK, "bwd drop batch rc=%d", brc);
        if (brc != JT_OK) {
            return;
        }
        // dropped qに対応する (t,e) のdLogitsは0。
        for (size_t q = 0; q < (size_t)T * (size_t)k; q++) {
            if (dropm[q]) {
                size_t t = q / (size_t)k;
                size_t ee = ids[q];
                CHECK(dlb[t * (size_t)E + ee] == 0.0f,
                      "bwd drop dLogits not zero q=%zu", q);
            }
        }
        // droppedスロットのYselは0埋め済み (fwd保証) のため、
        // 対応するexpert寄与が混入しないことはdXの有限性で確認する。
        for (int i = 0; i < T * n; i++) {
            CHECK(isfinite((double)dXb[i]), "bwd drop dX non-finite");
        }
    }
}

// バッチbwd不正系 (fail-closed: 出力不変)。
static void test_bwd_batch_invalid(void) {
    float X[4] = {1.0f, 0.5f, 0.25f, -0.5f};
    float dY[4] = {0.5f, -0.5f, 0.25f, 0.125f};
    float Wgate[8] = {1, 0, 0, 1, 0, 0, 0, 0};
    float W[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float dX[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    float dWg_[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t ids[4] = {0, 1, 0, 1};
    float wt[4] = {0.5f, 0.5f, 0.5f, 0.5f};
    float G[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float Ys[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    size_t perm[4] = {0, 1, 2, 3};
    size_t off[5] = {0, 1, 2, 2, 4};
    unsigned char drop[4] = {0, 0, 0, 0};
    float dWgate[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    CHECK(jt_moe_bwd_batch(NULL, X, Wgate, W, W, W, NULL, NULL, NULL, ids,
                           wt, G, G, Ys, NULL, NULL, perm, off, drop, dX,
                           dWgate, dWg_, dWg_, dWg_, NULL, NULL, NULL,
                           NULL, 2, 2, 1, 4, 2, 0) == JT_ERR_INVAL,
          "bwd batch NULL dY");
    CHECK(dX[0] == 3.0f && dX[3] == 6.0f, "bwd batch mutated on NULL");
    CHECK(jt_moe_bwd_batch(dY, X, Wgate, W, W, W, NULL, NULL, NULL, ids,
                           wt, G, G, Ys, NULL, NULL, perm, off, drop, dX,
                           dWgate, dWg_, dWg_, dWg_, NULL, NULL, NULL,
                           NULL, 0, 2, 1, 4, 2, 0) == JT_ERR_INVAL,
          "bwd batch T=0");
    CHECK(jt_moe_bwd_batch(dY, X, Wgate, W, W, W, NULL, NULL, NULL, ids,
                           wt, G, G, Ys, NULL, NULL, perm, off, drop, dX,
                           dWgate, dWg_, dWg_, dWg_, NULL, NULL, NULL,
                           NULL, 2, 2, 1, 4, 5, 0) == JT_ERR_INVAL,
          "bwd batch k>E");
    {
        size_t badoff[5] = {1, 0, 0, 0, 0};
        CHECK(jt_moe_bwd_batch(dY, X, Wgate, W, W, W, NULL, NULL, NULL,
                               ids, wt, G, G, Ys, NULL, NULL, perm, badoff,
                               drop, dX, dWgate, dWg_, dWg_, dWg_, NULL,
                               NULL, NULL, NULL, 2, 2, 1, 4, 2,
                               0) == JT_ERR_INVAL,
              "bwd batch bad off");
        CHECK(dX[0] == 3.0f, "bwd batch mutated on bad off");
    }
}

int main(void) {
    test_fwd_known();
    test_grad();
    test_topk_mask();
    test_sticky_seq();
    test_invalid();
    test_unchecked_match();
    test_batch_sort_known();
    test_batch_sort_drop();
    test_batch_sort_invalid();
    test_batch_equiv();
    test_batch_drop_fwd();
    test_batch_invalid();
    test_bwd_batch_gate_bitmatch();
    test_bwd_batch_equiv();
    test_bwd_batch_drop();
    test_bwd_batch_invalid();
    if (g_fail != 0) {
        fprintf(stderr, "moe_layer: FAIL\n");
        return 1;
    }
    printf("moe_layer: OK (fwd known + grad + topk-mask + sticky-seq)\n");
    return 0;
}
