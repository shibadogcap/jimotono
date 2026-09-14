// test_w8a8: Stage 3 W8A8 INT8経路の単体検証 (ctest登録)。
// - フラグ既定OFF (fp32並存)・set/get・不正系
// - per-tensor/per-channel量子化往復 (MAE<0.02。mixed_lutのINT8基準と同一)
// - int8 GEMM vs fp32参照 (相対差2%以内。量子化ノイズの範囲)
// - fake-quant往復・不正系 (fail-closed: 出力不変)
// - 既存fp32核 (jt_gemm_mat_f32) の疎通 (回帰なしのAPI確認)
// 成功時 exit 0 + "w8a8: OK"。C11・errnoベース。AVX-512不使用。
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/common.h"
#include "jimotono/moe_gemm.h"
#include "jimotono/w8a8.h"

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

static float tval(size_t i, float amp, unsigned salt) {
    unsigned u = (unsigned)((i * 2654435761u + salt * 40503u) >> 9) % 2001u;
    return (((float)u / 1000.0f) - 1.0f) * amp;
}

static void test_flag(void) {
    int e = -1;
    // 既定はOFF (fp32)。テスト開始時に明示OFFへ戻す (順序依存の排除)。
    CHECK(jt_w8a8_set_enabled(0) == JT_OK, "flag reset");
    CHECK(jt_w8a8_is_enabled() == 0, "flag default OFF");
    CHECK(jt_w8a8_get_enabled(&e) == JT_OK && e == 0, "flag get OFF");
    CHECK(jt_w8a8_set_enabled(1) == JT_OK, "flag set ON");
    CHECK(jt_w8a8_is_enabled() == 1, "flag is ON");
    CHECK(jt_w8a8_set_enabled(0) == JT_OK, "flag set OFF");
    CHECK(jt_w8a8_is_enabled() == 0, "flag is OFF");
    CHECK(jt_w8a8_set_enabled(2) == JT_ERR_INVAL, "flag bad val");
    CHECK(jt_w8a8_set_enabled(-1) == JT_ERR_INVAL, "flag neg");
    CHECK(jt_w8a8_get_enabled(NULL) == JT_ERR_INVAL, "flag NULL");
    CHECK(jt_w8a8_is_enabled() == 0, "flag unchanged after bad");
}

static void test_per_tensor(void) {
    const size_t n = 256;
    float *src = (float *)malloc(n * sizeof(float));
    int8_t *q = (int8_t *)malloc(n * sizeof(int8_t));
    float *dq = (float *)malloc(n * sizeof(float));
    float sc = 0.0f;
    CHECK(src && q && dq, "tensor alloc");
    if (!src || !q || !dq) {
        goto done;
    }
    for (size_t i = 0; i < n; i++) {
        src[i] = tval(i, 1.0f, 7u);
    }
    CHECK(jt_w8a8_quant_per_tensor(src, n, q, &sc) == JT_OK, "tensor q rc");
    CHECK(sc > 0.0f && isfinite((double)sc), "tensor scale=%f", sc);
    CHECK(jt_w8a8_dequant_per_tensor(q, sc, n, dq) == JT_OK, "tensor dq rc");
    {
        double acc = 0.0;
        for (size_t i = 0; i < n; i++) {
            acc += fabs((double)src[i] - (double)dq[i]);
        }
        CHECK(acc / (double)n < 0.02, "tensor mae=%f", acc / (double)n);
        printf("w8a8 per-tensor mae=%.5f scale=%.6f\n", acc / (double)n,
               (double)sc);
    }
    // ゼロ行列: scale=1.0, q=0。
    {
        float z[4] = {0, 0, 0, 0};
        int8_t zq[4] = {9, 9, 9, 9};
        float zs = 0.0f;
        CHECK(jt_w8a8_quant_per_tensor(z, 4, zq, &zs) == JT_OK, "zero q");
        CHECK(zs == 1.0f && zq[0] == 0 && zq[3] == 0, "zero val");
    }
    // fail-closed: NaN・NULL・n=0・不正scale・q値域外 (-128) はINVAL。
    {
        float bad[4] = {0.1f, 0.2f, (float)NAN, 0.4f};
        int8_t qq[4] = {0, 0, 0, 0};
        float s2 = 1.0f;
        float dd[4] = {0, 0, 0, 0};
        float keep = dd[0];
        CHECK(jt_w8a8_quant_per_tensor(bad, 4, qq, &s2) == JT_ERR_INVAL,
              "tensor nan");
        CHECK(jt_w8a8_quant_per_tensor(NULL, 4, qq, &s2) == JT_ERR_INVAL,
              "tensor NULL src");
        CHECK(jt_w8a8_quant_per_tensor(src, 0, qq, &s2) == JT_ERR_INVAL,
              "tensor n=0");
        CHECK(jt_w8a8_quant_per_tensor(src, 4, qq, NULL) == JT_ERR_INVAL,
              "tensor NULL scale");
        qq[0] = -128;  // 対称範囲外 (許容は[-127,127])
        CHECK(jt_w8a8_dequant_per_tensor(qq, 0.01f, 4, dd) == JT_ERR_INVAL,
              "tensor qrange -128");
        CHECK(dd[0] == keep, "tensor mutated on qrange");
        CHECK(jt_w8a8_dequant_per_tensor(q, 0.0f, n, dq) == JT_ERR_INVAL,
              "tensor scale=0");
        CHECK(jt_w8a8_dequant_per_tensor(q, (float)NAN, n, dq) ==
                      JT_ERR_INVAL,
              "tensor nan scale");
        CHECK(jt_w8a8_dequant_per_tensor(NULL, sc, n, dq) == JT_ERR_INVAL,
              "tensor NULL q");
    }
done:
    free(src);
    free(q);
    free(dq);
}

static void test_per_channel(void) {
    const size_t rows = 8, cols = 32;
    const size_t n = rows * cols;
    float *src = (float *)malloc(n * sizeof(float));
    int8_t *q = (int8_t *)malloc(n * sizeof(int8_t));
    float *dq = (float *)malloc(n * sizeof(float));
    float *sc = (float *)malloc(rows * sizeof(float));
    CHECK(src && q && dq && sc, "channel alloc");
    if (!src || !q || !dq || !sc) {
        goto done;
    }
    for (size_t i = 0; i < n; i++) {
        // 行毎に振幅を変える (per-channelの効果を確認)。
        float amp = 0.25f * (float)(1 + (i / cols) % 4);
        src[i] = tval(i, amp, 21u);
    }
    CHECK(jt_w8a8_quant_per_channel(src, rows, cols, q, sc) == JT_OK,
          "channel q rc");
    for (size_t r = 0; r < rows; r++) {
        CHECK(sc[r] > 0.0f && isfinite((double)sc[r]), "ch scale r=%zu", r);
    }
    CHECK(jt_w8a8_dequant_per_channel(q, sc, rows, cols, dq) == JT_OK,
          "channel dq rc");
    {
        double acc = 0.0;
        for (size_t i = 0; i < n; i++) {
            acc += fabs((double)src[i] - (double)dq[i]);
        }
        CHECK(acc / (double)n < 0.02, "channel mae=%f", acc / (double)n);
        printf("w8a8 per-channel mae=%.5f\n", acc / (double)n);
    }
    // fail-closed。
    {
        float keep = dq[0];
        int8_t qb[4] = {0, 0, 0, 0};
        float sb[1] = {0.01f};
        float db[4] = {0, 0, 0, 0};
        CHECK(jt_w8a8_quant_per_channel(NULL, 2, 2, qb, sb) == JT_ERR_INVAL,
              "ch NULL src");
        CHECK(jt_w8a8_quant_per_channel(src, 0, cols, q, sc) == JT_ERR_INVAL,
              "ch rows=0");
        CHECK(jt_w8a8_quant_per_channel(src, rows, 0, q, sc) == JT_ERR_INVAL,
              "ch cols=0");
        sb[0] = (float)NAN;
        CHECK(jt_w8a8_dequant_per_channel(qb, sb, 1, 4, db) == JT_ERR_INVAL,
              "ch nan scale");
        qb[0] = -128;
        sb[0] = 0.01f;
        CHECK(jt_w8a8_dequant_per_channel(qb, sb, 1, 4, db) == JT_ERR_INVAL,
              "ch qrange");
        CHECK(dq[0] == keep, "ch mutated check base");
    }
done:
    free(src);
    free(q);
    free(dq);
    free(sc);
}

// fp32参照GEMM (k逐次。moe_gemmと同一順序)。
static void ref_gemm(const float *A, const float *B, float *C, int M, int N,
                     int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                acc += (double)A[(size_t)m * (size_t)K + (size_t)k] *
                       (double)B[(size_t)k * (size_t)N + (size_t)n];
            }
            C[(size_t)m * (size_t)N + (size_t)n] = (float)acc;
        }
    }
}

static int gemm_close(float got, double want) {
    double d = fabs((double)got - want);
    double allowed = 0.02 * fabs(want) + 0.02;
    return d <= allowed;
}

static void test_gemm_tensor(void) {
    const int M = 4, N = 8, K = 16;
    float *A = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    float *B = (float *)malloc((size_t)K * (size_t)N * sizeof(float));
    float *Cref = (float *)malloc((size_t)M * (size_t)N * sizeof(float));
    float *Cq = (float *)malloc((size_t)M * (size_t)N * sizeof(float));
    int8_t *Aq = (int8_t *)malloc((size_t)M * (size_t)K * sizeof(int8_t));
    int8_t *Bq = (int8_t *)malloc((size_t)K * (size_t)N * sizeof(int8_t));
    float sA = 0, sB = 0;
    CHECK(A && B && Cref && Cq && Aq && Bq, "gemm alloc");
    if (!A || !B || !Cref || !Cq || !Aq || !Bq) {
        goto done;
    }
    for (int i = 0; i < M * K; i++) {
        A[i] = tval((size_t)i, 1.0f, 31u);
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = tval((size_t)i, 1.0f, 37u);
    }
    ref_gemm(A, B, Cref, M, N, K);
    CHECK(jt_w8a8_quant_per_tensor(A, (size_t)M * (size_t)K, Aq, &sA) ==
                  JT_OK,
          "gemm Aq");
    CHECK(jt_w8a8_quant_per_tensor(B, (size_t)K * (size_t)N, Bq, &sB) ==
                  JT_OK,
          "gemm Bq");
    memset(Cq, 0xA5, (size_t)M * (size_t)N * sizeof(float));
    CHECK(jt_w8a8_gemm_per_tensor(Aq, Bq, sA, sB, Cq, M, N, K) == JT_OK,
          "gemm rc");
    for (int i = 0; i < M * N; i++) {
        CHECK(gemm_close(Cq[i], (double)Cref[i]), "gemm[%d] q=%f ref=%f", i,
              Cq[i], Cref[i]);
    }
    {
        double acc = 0.0, denom = 0.0;
        for (int i = 0; i < M * N; i++) {
            acc += fabs((double)Cq[i] - (double)Cref[i]);
            denom += fabs((double)Cref[i]);
        }
        printf("w8a8 gemm-tensor mare=%.5f\n", acc / denom);
    }
    // M==0は起動スキップ (C不変でJT_OK)。
    {
        float keep = Cq[0];
        CHECK(jt_w8a8_gemm_per_tensor(Aq, Bq, sA, sB, Cq, 0, N, K) == JT_OK,
              "gemm M=0");
        CHECK(Cq[0] == keep, "gemm M=0 mutated");
        CHECK(jt_w8a8_gemm_per_tensor(Aq, Bq, sA, sB, Cq, 0, 0, K) ==
                      JT_ERR_INVAL,
              "gemm M=0 N=0");
    }
    // 不正系 (fail-closed: C不変)。
    {
        float keep = Cq[0];
        CHECK(jt_w8a8_gemm_per_tensor(NULL, Bq, sA, sB, Cq, M, N, K) ==
                      JT_ERR_INVAL,
              "gemm NULL A");
        CHECK(Cq[0] == keep, "gemm mutated on NULL");
        CHECK(jt_w8a8_gemm_per_tensor(Aq, Bq, 0.0f, sB, Cq, M, N, K) ==
                      JT_ERR_INVAL,
              "gemm sA=0");
        CHECK(jt_w8a8_gemm_per_tensor(Aq, Bq, sA, (float)NAN, Cq, M, N,
                                      K) == JT_ERR_INVAL,
              "gemm nan sB");
        CHECK(jt_w8a8_gemm_per_tensor(Aq, Bq, sA, sB, Cq, -1, N, K) ==
                      JT_ERR_INVAL,
              "gemm M<0");
        Aq[0] = -128;
        CHECK(jt_w8a8_gemm_per_tensor(Aq, Bq, sA, sB, Cq, M, N, K) ==
                      JT_ERR_INVAL,
              "gemm qrange A");
        CHECK(Cq[0] == keep, "gemm mutated on qrange");
        Aq[0] = 0;
    }
done:
    free(A);
    free(B);
    free(Cref);
    free(Cq);
    free(Aq);
    free(Bq);
}

static void test_gemm_channel(void) {
    const int M = 4, N = 8, K = 16;
    float *A = (float *)malloc((size_t)M * (size_t)K * sizeof(float));
    // Bは転置形 [N][K] で行量子化し、GEMM用 [K][N] に戻す (列scale対応)。
    float *BT = (float *)malloc((size_t)N * (size_t)K * sizeof(float));
    float *B = (float *)malloc((size_t)K * (size_t)N * sizeof(float));
    float *Cref = (float *)malloc((size_t)M * (size_t)N * sizeof(float));
    float *Cq = (float *)malloc((size_t)M * (size_t)N * sizeof(float));
    int8_t *Aq = (int8_t *)malloc((size_t)M * (size_t)K * sizeof(int8_t));
    int8_t *BTq = (int8_t *)malloc((size_t)N * (size_t)K * sizeof(int8_t));
    int8_t *Bq = (int8_t *)malloc((size_t)K * (size_t)N * sizeof(int8_t));
    float sA = 0;
    float *sBcol = (float *)malloc((size_t)N * sizeof(float));
    CHECK(A && BT && B && Cref && Cq && Aq && BTq && Bq && sBcol,
          "gemmc alloc");
    if (!A || !BT || !B || !Cref || !Cq || !Aq || !BTq || !Bq || !sBcol) {
        goto done;
    }
    for (int i = 0; i < M * K; i++) {
        A[i] = tval((size_t)i, 1.0f, 41u);
    }
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            float v = tval((size_t)(k * N + n), 1.0f, 43u);
            B[(size_t)k * (size_t)N + (size_t)n] = v;
            BT[(size_t)n * (size_t)K + (size_t)k] = v;
        }
    }
    ref_gemm(A, B, Cref, M, N, K);
    CHECK(jt_w8a8_quant_per_tensor(A, (size_t)M * (size_t)K, Aq, &sA) ==
                  JT_OK,
          "gemmc Aq");
    CHECK(jt_w8a8_quant_per_channel(BT, (size_t)N, (size_t)K, BTq, sBcol) ==
                  JT_OK,
          "gemmc Bq rows");
    // BTq [N][K] → Bq [K][N] 転置 (exact)。
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            Bq[(size_t)k * (size_t)N + (size_t)n] =
                BTq[(size_t)n * (size_t)K + (size_t)k];
        }
    }
    memset(Cq, 0, (size_t)M * (size_t)N * sizeof(float));
    CHECK(jt_w8a8_gemm_per_channel(Aq, Bq, sA, sBcol, Cq, M, N, K) == JT_OK,
          "gemmc rc");
    for (int i = 0; i < M * N; i++) {
        CHECK(gemm_close(Cq[i], (double)Cref[i]), "gemmc[%d] q=%f ref=%f",
              i, Cq[i], Cref[i]);
    }
    {
        double acc = 0.0, denom = 0.0;
        for (int i = 0; i < M * N; i++) {
            acc += fabs((double)Cq[i] - (double)Cref[i]);
            denom += fabs((double)Cref[i]);
        }
        printf("w8a8 gemm-channel mare=%.5f\n", acc / denom);
    }
    // sB_col==NULLはINVAL (per-tensor版を使うこと)。
    CHECK(jt_w8a8_gemm_per_channel(Aq, Bq, sA, NULL, Cq, M, N, K) ==
                  JT_ERR_INVAL,
          "gemmc NULL col");
done:
    free(A);
    free(BT);
    free(B);
    free(Cref);
    free(Cq);
    free(Aq);
    free(BTq);
    free(Bq);
    free(sBcol);
}

static void test_fakequant(void) {
    float src[8] = {0.1f, -0.5f, 0.9f, -0.05f, 0.0f, 0.33f, -0.77f, 0.02f};
    float dst[8] = {0};
    CHECK(jt_w8a8_fakequant_per_tensor(src, 8, dst) == JT_OK, "fq rc");
    {
        double acc = 0.0;
        for (int i = 0; i < 8; i++) {
            acc += fabs((double)src[i] - (double)dst[i]);
        }
        CHECK(acc / 8.0 < 0.02, "fq mae=%f", acc / 8.0);
    }
    CHECK(jt_w8a8_fakequant_per_channel(src, 2, 4, dst) == JT_OK,
          "fq ch rc");
    {
        float bad[2] = {0.1f, (float)NAN};
        CHECK(jt_w8a8_fakequant_per_tensor(bad, 2, dst) == JT_ERR_INVAL,
              "fq nan");
        CHECK(jt_w8a8_fakequant_per_tensor(NULL, 2, dst) == JT_ERR_INVAL,
              "fq NULL");
        CHECK(jt_w8a8_fakequant_per_tensor(src, 0, dst) == JT_ERR_INVAL,
              "fq n=0");
        CHECK(jt_w8a8_fakequant_per_channel(NULL, 2, 4, dst) ==
                      JT_ERR_INVAL,
              "fq ch NULL");
    }
}

static void test_fp32_coexist(void) {
    // 既存fp32核の疎通 (2x2既知値。C = A·B)。
    float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float C[4] = {0, 0, 0, 0};
    CHECK(jt_gemm_mat_f32(A, B, C, 2, 2, 2) == JT_OK, "fp32 coexist rc");
    CHECK(fabs((double)C[0] - 19.0) < 1e-5, "fp32 C00=%f", C[0]);
    CHECK(fabs((double)C[1] - 22.0) < 1e-5, "fp32 C01=%f", C[1]);
    CHECK(fabs((double)C[2] - 43.0) < 1e-5, "fp32 C10=%f", C[2]);
    CHECK(fabs((double)C[3] - 50.0) < 1e-5, "fp32 C11=%f", C[3]);
}

int main(void) {
    test_flag();
    test_per_tensor();
    test_per_channel();
    test_gemm_tensor();
    test_gemm_channel();
    test_fakequant();
    test_fp32_coexist();
    // フラグを既定OFFに戻して終了 (他テストへの漏洩防止)。
    jt_w8a8_set_enabled(0);
    if (g_fail != 0) {
        fprintf(stderr, "w8a8: FAIL\n");
        return 1;
    }
    printf("w8a8: OK (flag/quant/dequant/gemm/fakequant/fp32-coexist)\n");
    return 0;
}
