// test_lowbit: routed低ビット副線の単体検証 (Stage 3b-2)。
// - fake-quant誤差分離 (INT2/INT4 vs fp32、INT8対照なしでも単体で上限確認)。
// - 真LUT vs fp32: テーブル量子化誤差の範囲内で一致 (bit一致は要求しない)。
// - fake-quant vs 真LUT: 同一レベル系のため近接 (構成上の等価性)。
// - fail-closed: NULL/不正bits/非有限/容量系。
// 成功時 exit 0 + "lowbit: OK"。
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/common.h"
#include "jimotono/lowbit.h"
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

// 決定論的パターン [-1,1]。
static float lpat(int a, int b, int salt) {
    int v = (a * 31 + b * 17 + salt * 13) % 11;  // 0..10
    return (float)(v - 5) / 5.0f;
}

static void test_mode_api(void) {
    CHECK(jt_lowbit_get_mode() == 0, "default mode 0");
    CHECK(jt_lowbit_set_mode(1) == JT_OK, "set 1");
    CHECK(jt_lowbit_get_mode() == 1, "get 1");
    CHECK(jt_lowbit_set_mode(2) == JT_OK, "set 2");
    CHECK(jt_lowbit_set_mode(0) == JT_OK, "set 0");
    CHECK(jt_lowbit_set_mode(3) == JT_ERR_INVAL, "set 3");
    CHECK(jt_lowbit_set_mode(-1) == JT_ERR_INVAL, "set -1");
    int b = 0;
    CHECK(jt_lowbit_bits_for(0, &b) == JT_OK && b == 2, "gate bits");
    CHECK(jt_lowbit_bits_for(1, &b) == JT_OK && b == 4, "down bits");
    CHECK(jt_lowbit_bits_for(0, NULL) == JT_ERR_INVAL, "bits null");
}

static void test_fakequant_err(void) {
    // N=256の滑らかパターン。INT4誤差 < INT2誤差、両方とも有界。
    enum { N = 256 };
    float w[N], fq[N];
    for (int i = 0; i < N; i++) {
        w[i] = lpat(i, 3, 7) * (1.0f + (float)(i % 5) * 0.1f);
    }
    double mae2 = -1.0, mae4 = -1.0;
    CHECK(jt_lowbit_fakequant(w, fq, N, 2) == JT_OK, "fq2 rc");
    {
        double s = 0.0;
        for (int i = 0; i < N; i++) {
            s += fabs((double)w[i] - (double)fq[i]);
        }
        mae2 = s / N;
    }
    CHECK(jt_lowbit_fakequant(w, fq, N, 4) == JT_OK, "fq4 rc");
    {
        double s = 0.0;
        for (int i = 0; i < N; i++) {
            s += fabs((double)w[i] - (double)fq[i]);
        }
        mae4 = s / N;
    }
    CHECK(mae4 < mae2, "mae4 %f < mae2 %f", mae4, mae2);
    CHECK(mae4 < 0.10, "mae4 bound %f", mae4);
    CHECK(mae2 < 0.35, "mae2 bound %f", mae2);
    // 量子化レベルの対称性: 全要素が奇対称グリッド上
    // ((2li-15)/15 の形。t*5 = 2li-15 は [-15,15] の奇整数)。
    {
        double maxa = 0.0;
        for (int i = 0; i < N; i++) {
            double a = fabs((double)w[i]);
            if (a > maxa) {
                maxa = a;
            }
        }
        for (int i = 0; i < N; i++) {
            double t = (double)fq[i] / maxa * 15.0;  // 2li-15 のはず
            double r = round(t);
            CHECK(fabs(t - r) < 1e-3, "fq4 grid i=%d t=%f", i, t);
            if (fabs(t - r) < 1e-3) {
                long li = (long)r;
                CHECK(li >= -15 && li <= 15 && (li & 1L) != 0L,
                      "fq4 odd i=%d li=%ld", i, li);
            }
        }
    }
    // fail-closed。
    CHECK(jt_lowbit_fakequant(NULL, fq, N, 4) == JT_ERR_INVAL, "fq null");
    CHECK(jt_lowbit_fakequant(w, NULL, N, 4) == JT_ERR_INVAL, "fq null2");
    CHECK(jt_lowbit_fakequant(w, fq, 0, 4) == JT_ERR_INVAL, "fq n0");
    CHECK(jt_lowbit_fakequant(w, fq, N, 3) == JT_ERR_INVAL, "fq bits3");
    {
        float bad[N];
        memcpy(bad, w, sizeof w);
        bad[7] = (float)NAN;
        CHECK(jt_lowbit_fakequant(bad, fq, N, 4) == JT_ERR_INVAL, "fq nan");
    }
}

static void test_lut_vs_fp32(void) {
    // M=4,H=8,K=64。真LUT vs fp32参照: テーブル量子化誤差の範囲で一致。
    // fake-quant vs 真LUT: 同一レベル系のため近接。
    enum { M = 4, H = 8, K = 64 };
    float X[M * K], W[K * H], Yl[M * H], Yr[M * H], Wfq[K * H], Yf[M * H];
    for (int i = 0; i < M * K; i++) {
        X[i] = lpat(i, 1, 11);
    }
    for (int i = 0; i < K * H; i++) {
        W[i] = lpat(i, 2, 13) * 0.5f;
    }
    // fp32参照。
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < H; j++) {
            double acc = 0.0;
            for (int t = 0; t < K; t++) {
                acc += (double)X[i * K + t] * (double)W[t * H + j];
            }
            Yr[i * H + j] = (float)acc;
        }
    }
    double denom = 0.0;
    for (int i = 0; i < M * H; i++) {
        denom += fabs((double)Yr[i]);
    }
    denom /= (M * H);
    // INT4真LUT。
    CHECK(jt_lowbit_gemm_lut(X, W, Yl, M, H, K, 4) == JT_OK, "lut4 rc");
    {
        double s = 0.0;
        for (int i = 0; i < M * H; i++) {
            s += fabs((double)Yl[i] - (double)Yr[i]);
        }
        double rel = s / (M * H) / (denom + 1e-6);
        CHECK(rel < 0.10, "lut4 rel %f", rel);
    }
    // INT2真LUT (誤差大だが有界)。
    CHECK(jt_lowbit_gemm_lut(X, W, Yl, M, H, K, 2) == JT_OK, "lut2 rc");
    {
        double s = 0.0;
        for (int i = 0; i < M * H; i++) {
            s += fabs((double)Yl[i] - (double)Yr[i]);
        }
        double rel = s / (M * H) / (denom + 1e-6);
        CHECK(rel < 0.45, "lut2 rel %f", rel);
    }
    // fake-quant(INT4)＋fp32 GEMM vs 真LUT(INT4): 近接。
    for (int j = 0; j < H; j++) {
        float col[K], fq[K];
        for (int t = 0; t < K; t++) {
            col[t] = W[t * H + j];
        }
        CHECK(jt_lowbit_fakequant(col, fq, K, 4) == JT_OK, "fq col");
        for (int t = 0; t < K; t++) {
            Wfq[t * H + j] = fq[t];
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < H; j++) {
            double acc = 0.0;
            for (int t = 0; t < K; t++) {
                acc += (double)X[i * K + t] * (double)Wfq[t * H + j];
            }
            Yf[i * H + j] = (float)acc;
        }
    }
    CHECK(jt_lowbit_gemm_lut(X, W, Yl, M, H, K, 4) == JT_OK, "lut4 rc2");
    {
        double s = 0.0;
        for (int i = 0; i < M * H; i++) {
            s += fabs((double)Yl[i] - (double)Yf[i]);
        }
        double rel = s / (M * H) / (denom + 1e-6);
        CHECK(rel < 0.10, "fq-vs-lut rel %f", rel);
    }
    // fail-closed / 例外。
    CHECK(jt_lowbit_gemm_lut(NULL, W, Yl, M, H, K, 4) == JT_ERR_INVAL,
          "lut null");
    CHECK(jt_lowbit_gemm_lut(X, W, Yl, M, H, K, 3) == JT_ERR_INVAL,
          "lut bits3");
    CHECK(jt_lowbit_gemm_lut(X, W, Yl, 0, H, K, 4) == JT_OK, "lut m0");
    {
        // K%32!=0 → fp32 fallback (bit一致級)。
        float X2[2 * 20], W2[20 * 3], Y2[2 * 3], R2[2 * 3];
        for (int i = 0; i < 40; i++) {
            X2[i] = lpat(i, 5, 17);
        }
        for (int i = 0; i < 60; i++) {
            W2[i] = lpat(i, 6, 19) * 0.5f;
        }
        CHECK(jt_lowbit_gemm_lut(X2, W2, Y2, 2, 3, 20, 4) == JT_OK,
              "lut fallback rc");
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                double acc = 0.0;
                for (int t = 0; t < 20; t++) {
                    acc += (double)X2[i * 20 + t] * (double)W2[t * 3 + j];
                }
                R2[i * 3 + j] = (float)acc;
            }
        }
        CHECK(memcmp(Y2, R2, sizeof Y2) == 0, "lut fallback exact");
    }
    {
        uint8_t idx[8];
        float ws = 0.0f;
        float col[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        CHECK(jt_lowbit_col_to_idx(col, 8, 4, idx, &ws) == JT_OK,
              "col idx rc");
        CHECK(ws == 8.0f, "col ws %f", ws);
        CHECK(jt_lowbit_col_to_idx(NULL, 8, 4, idx, &ws) == JT_ERR_INVAL,
              "col null");
        CHECK(jt_lowbit_col_to_idx(col, 7, 4, idx, &ws) == JT_ERR_INVAL,
              "col k4");
        CHECK(jt_lowbit_col_to_idx(col, 8, 3, idx, &ws) == JT_ERR_INVAL,
              "col bits3");
    }
}

// STE勾配一致: fake-quant順伝播＋fp32逆伝播の勾配は、
// fake-quant重みでのfp32逆伝播と同一 (別経路の量子化勾配なし)。
// すなわち勾配差は順伝播差の線形伝播に留まる (5倍以内をassert)。
static void test_ste_grad(void) {
    enum { N = 8, H = 4 };
    float X[N], Wg[H * N], Wu[H * N], Wd[H * N];
    float G[H], U[H], Y[N], Gq[H], Uq[H], Yq[N];
    float dY[N], dX[N], dXq[N];
    float dWg[H * N], dWu[H * N], dWd[H * N];
    float dWgq[H * N], dWuq[H * N], dWdq[H * N];
    float Wgq[H * N], Wuq[H * N], Wdq[H * N];
    for (int i = 0; i < N; i++) {
        X[i] = lpat(i, 1, 21) * 0.5f;
        dY[i] = lpat(i, 2, 23) * 0.5f;
    }
    for (int i = 0; i < H * N; i++) {
        Wg[i] = lpat(i, 3, 25) * 0.3f;
        Wu[i] = lpat(i, 4, 27) * 0.3f;
        Wd[i] = lpat(i, 5, 29) * 0.3f;
    }
    // fake-quant重み (gate/up INT2, down INT4)。
    for (int j = 0; j < H; j++) {
        float cg[N], cu[N], cd[H];
        for (int t = 0; t < N; t++) {
            cg[t] = Wg[(size_t)j * N + t];
            cu[t] = Wu[(size_t)j * N + t];
        }
        for (int t = 0; t < H; t++) {
            cd[t] = 0.0f;
        }
        CHECK(jt_lowbit_fakequant(cg, cg, N, 2) == JT_OK, "ste fqg");
        CHECK(jt_lowbit_fakequant(cu, cu, N, 2) == JT_OK, "ste fqu");
        for (int t = 0; t < N; t++) {
            Wgq[(size_t)j * N + t] = cg[t];
            Wuq[(size_t)j * N + t] = cu[t];
        }
    }
    for (int j = 0; j < N; j++) {
        float cd[H], cq[H];
        for (int t = 0; t < H; t++) {
            cd[t] = Wd[(size_t)t * N + j];
        }
        CHECK(jt_lowbit_fakequant(cd, cq, H, 4) == JT_OK, "ste fqd");
        for (int t = 0; t < H; t++) {
            Wdq[(size_t)t * N + j] = cq[t];
        }
    }
    CHECK(jt_swiglu_fwd(X, Wg, Wu, Wd, G, U, Y, N, H) == JT_OK, "ste fwd");
    CHECK(jt_swiglu_fwd(X, Wgq, Wuq, Wdq, Gq, Uq, Yq, N, H) == JT_OK,
          "ste fwdq");
    CHECK(jt_swiglu_bwd(dY, X, G, U, Wd, Wg, Wu, dX, dWg, dWu, dWd, N,
                        H) == JT_OK,
          "ste bwd");
    CHECK(jt_swiglu_bwd(dY, X, Gq, Uq, Wdq, Wgq, Wuq, dXq, dWgq, dWuq,
                        dWdq, N, H) == JT_OK,
          "ste bwdq");
    // 順伝播の相対差と勾配の相対差を比較 (勾配差≦5×順伝播差)。
    {
        double fs = 0.0, gs = 0.0, fn = 0.0;
        for (int i = 0; i < N; i++) {
            fs += fabs((double)Y[i] - (double)Yq[i]);
            fn += fabs((double)Y[i]);
        }
        for (int i = 0; i < H * N; i++) {
            gs += fabs((double)dWg[i] - (double)dWgq[i]);
            gs += fabs((double)dWu[i] - (double)dWuq[i]);
            gs += fabs((double)dWd[i] - (double)dWdq[i]);
            fn += 0.0;
        }
        double frel = fs / N / (fn / N + 1e-9);
        double gscale = 0.0;
        for (int i = 0; i < H * N; i++) {
            gscale += fabs((double)dWg[i]) + fabs((double)dWu[i]) +
                      fabs((double)dWd[i]);
        }
        double grel = gs / (gscale + 1e-9);
        CHECK(grel <= 5.0 * frel + 1e-6, "ste grad %f vs fwd %f", grel,
              frel);
    }
}

int main(void) {
    test_mode_api();
    test_fakequant_err();
    test_lut_vs_fp32();
    test_ste_grad();
    if (g_fail == 0) {
        printf("lowbit: OK\n");
        return 0;
    }
    fprintf(stderr, "lowbit: %d FAIL(s)\n", g_fail);
    return 1;
}
