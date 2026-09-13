// bench+test: GDN-2 デコード逐次核の数値検証 + 異常系 (Phase 1)。
// 検証式 (papers.md §2):
//   S_t = (I - k_t (b_t⊙k_t)^T) D_t S_{t-1} + k_t (w_t⊙v_t)^T, o_t = S_t^T q_t
// 方針: 倍精度リファレンス (double) と単精度カーネル出力を tol=2e-5 で比較。
// 成功時 exit 0 + "gdn2: OK"、失敗時 exit 1 + stderr。
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/common.h"
#include "jimotono/gdn2.h"

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

// ---- 倍精度リファレンス (テスト内のみ) ----
static void ref_l2norm(const double *x, double *y, int n) {
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

// S: [dk][dv] row-major double。入出力。
static void ref_step(double *S, double *o, const double *q, const double *k,
                     const double *v, const double *b, const double *w,
                     const double *alpha, int dk, int dv) {
    double qn[1024];
    double kn[1024];
    double c[1024];
    double vw[1024];
    ref_l2norm(q, qn, dk);
    ref_l2norm(k, kn, dk);
    for (int i = 0; i < dk; i++) {  // S' = D S
        for (int j = 0; j < dv; j++) {
            S[i * dv + j] *= alpha[i];
        }
    }
    for (int j = 0; j < dv; j++) {  // c^T = (b⊙k)^T S'
        c[j] = 0.0;
        for (int i = 0; i < dk; i++) {
            c[j] += (b[i] * kn[i]) * S[i * dv + j];
        }
    }
    for (int i = 0; i < dk; i++) {  // S'' = S' - k c^T
        for (int j = 0; j < dv; j++) {
            S[i * dv + j] -= kn[i] * c[j];
        }
    }
    for (int j = 0; j < dv; j++) {  // vw = w⊙v
        vw[j] = w[j] * v[j];
    }
    for (int i = 0; i < dk; i++) {  // S = S'' + k vw^T
        for (int j = 0; j < dv; j++) {
            S[i * dv + j] += kn[i] * vw[j];
        }
    }
    for (int j = 0; j < dv; j++) {  // o = S^T q
        o[j] = 0.0;
        for (int i = 0; i < dk; i++) {
            o[j] += qn[i] * S[i * dv + j];
        }
    }
}

static float fd(int i, int j, int s) {
    // 決定論的ダミー値 (-1..1 付近)。s で系列を変える。
    double v = (double)(((i * 31 + j * 17 + s * 13) % 11) - 5) / 5.0;
    return (float)v;
}

// 1ケースの数値比較。S0 初期値・入力を決定論的に生成。
static void test_numeric_case(int dk, int dv, int seed) {
    float *S = NULL;
    CHECK(jt_gdn2_state_alloc(&S, dk, dv) == JT_OK && S != NULL,
          "state_alloc dk=%d dv=%d", dk, dv);
    if (S == NULL) {
        return;
    }
    CHECK(((uintptr_t)S % 64u) == 0, "state not 64B aligned");
    size_t need = 0;
    CHECK(jt_gdn2_scratch_floats(dk, dv, &need) == JT_OK, "scratch size");
    float *scratch = (float *)malloc(need * sizeof(float));
    float *o = (float *)malloc((size_t)dv * sizeof(float));
    CHECK(scratch != NULL && o != NULL, "malloc scratch/o");
    if (scratch == NULL || o == NULL) {
        free(scratch);
        free(o);
        jt_gdn2_state_free(S);
        return;
    }
    float *q = (float *)malloc((size_t)dk * sizeof(float));
    float *k = (float *)malloc((size_t)dk * sizeof(float));
    float *b = (float *)malloc((size_t)dk * sizeof(float));
    float *alpha = (float *)malloc((size_t)dk * sizeof(float));
    float *v = (float *)malloc((size_t)dv * sizeof(float));
    float *w = (float *)malloc((size_t)dv * sizeof(float));
    double *rS = (double *)malloc((size_t)dk * (size_t)dv * sizeof(double));
    double *ro = (double *)malloc((size_t)dv * sizeof(double));
    double *rq = (double *)malloc((size_t)dk * sizeof(double));
    double *rk = (double *)malloc((size_t)dk * sizeof(double));
    double *rb = (double *)malloc((size_t)dk * sizeof(double));
    double *ra = (double *)malloc((size_t)dk * sizeof(double));
    double *rv = (double *)malloc((size_t)dv * sizeof(double));
    double *rw = (double *)malloc((size_t)dv * sizeof(double));
    CHECK(q && k && b && alpha && v && w && rS && ro && rq && rk && rb && ra &&
              rv && rw,
          "malloc vecs");
    if (!(q && k && b && alpha && v && w && rS && ro && rq && rk && rb && ra &&
          rv && rw)) {
        goto free_all;
    }
    for (int i = 0; i < dk; i++) {
        q[i] = fd(i, 0, seed) + 0.37f;
        k[i] = fd(i, 1, seed + 1) - 0.23f;
        b[i] = 0.5f + 0.1f * fd(i, 2, seed);      // erase: 0.4..0.6
        alpha[i] = 0.9f + 0.05f * fd(i, 3, seed);  // decay: 0.85..0.95
        rq[i] = (double)q[i];
        rk[i] = (double)k[i];
        rb[i] = (double)b[i];
        ra[i] = (double)alpha[i];
    }
    for (int j = 0; j < dv; j++) {
        v[j] = fd(0, j, seed + 2) + 0.11f;
        w[j] = 0.6f + 0.1f * fd(1, j, seed);  // write 側
        rv[j] = (double)v[j];
        rw[j] = (double)w[j];
    }
    for (int i = 0; i < dk; i++) {
        for (int j = 0; j < dv; j++) {
            float s0 = 0.25f * fd(i, j, seed + 3);
            S[i * dv + j] = s0;
            rS[i * dv + j] = (double)s0;
        }
    }
    int rc = jt_gdn2_decode_step(S, o, q, k, v, b, w, alpha, dk, dv, scratch,
                                 need);
    CHECK(rc == JT_OK, "decode_step rc=%d dk=%d dv=%d", rc, dk, dv);
    ref_step(rS, ro, rq, rk, rv, rb, rw, ra, dk, dv);
    const double tol = 2e-5;
    for (int j = 0; j < dv; j++) {
        double d = fabs((double)o[j] - ro[j]);
        CHECK(d <= tol, "o[%d] diff=%g (got %f want %f) dk=%d dv=%d", j, d,
              o[j], ro[j], dk, dv);
    }
    for (int i = 0; i < dk * dv; i++) {
        double d = fabs((double)S[i] - rS[i]);
        CHECK(d <= tol, "S[%d] diff=%g dk=%d dv=%d", i, d, dk, dv);
    }
free_all:
    free(q);
    free(k);
    free(b);
    free(alpha);
    free(v);
    free(w);
    free(rS);
    free(ro);
    free(rq);
    free(rk);
    free(rb);
    free(ra);
    free(rv);
    free(rw);
    free(scratch);
    free(o);
    jt_gdn2_state_free(S);
}

// 3ステップ連続更新のリファレンス一致 (state 持ち越し)。
static void test_numeric_sequence(void) {
    const int dk = 4, dv = 6;
    float *S = NULL;
    CHECK(jt_gdn2_state_alloc(&S, dk, dv) == JT_OK, "seq alloc");
    if (S == NULL) {
        return;
    }
    size_t need = 0;
    CHECK(jt_gdn2_scratch_floats(dk, dv, &need) == JT_OK, "seq scratch");
    float *scratch = (float *)malloc(need * sizeof(float));
    double *rS = (double *)calloc((size_t)dk * (size_t)dv, sizeof(double));
    CHECK(scratch != NULL && rS != NULL, "seq malloc");
    if (scratch == NULL || rS == NULL) {
        free(scratch);
        free(rS);
        jt_gdn2_state_free(S);
        return;
    }
    CHECK(jt_gdn2_state_reset(S, dk, dv) == JT_OK, "seq reset");
    const double tol = 2e-5;
    for (int t = 0; t < 3; t++) {
        float q[4], k[4], b[4], alpha[4], v[6], w[6], o[6];
        double rq[4], rk[4], rb[4], ra[4], rv[6], rw[6], ro[6];
        for (int i = 0; i < dk; i++) {
            q[i] = fd(i, t, 100) + 0.31f;
            k[i] = fd(i, t, 200) - 0.17f;
            b[i] = 0.55f;
            alpha[i] = 0.93f;
            rq[i] = q[i];
            rk[i] = k[i];
            rb[i] = b[i];
            ra[i] = alpha[i];
        }
        for (int j = 0; j < dv; j++) {
            v[j] = fd(t, j, 300) + 0.07f;
            w[j] = 0.65f;
            rv[j] = v[j];
            rw[j] = w[j];
        }
        int rc = jt_gdn2_decode_step(S, o, q, k, v, b, w, alpha, dk, dv,
                                     scratch, need);
        CHECK(rc == JT_OK, "seq step %d rc=%d", t, rc);
        ref_step(rS, ro, rq, rk, rv, rb, rw, ra, dk, dv);
        for (int j = 0; j < dv; j++) {
            double d = fabs((double)o[j] - ro[j]);
            CHECK(d <= tol, "seq t=%d o[%d] diff=%g", t, j, d);
        }
    }
    free(scratch);
    free(rS);
    jt_gdn2_state_free(S);
}

// KDA 帰着: b=w=beta*1, alpha=gamma*1 でスカラーゲートと一致すること。
static void test_kda_reduction(void) {
    const int dk = 4, dv = 4;
    float *S = NULL;
    CHECK(jt_gdn2_state_alloc(&S, dk, dv) == JT_OK, "kda alloc");
    if (S == NULL) {
        return;
    }
    size_t need = 0;
    CHECK(jt_gdn2_scratch_floats(dk, dv, &need) == JT_OK, "kda scratch");
    float *scratch = (float *)malloc(need * sizeof(float));
    float o[4];
    CHECK(scratch != NULL, "kda malloc");
    if (scratch == NULL) {
        jt_gdn2_state_free(S);
        return;
    }
    for (int i = 0; i < dk * dv; i++) {
        S[i] = 0.1f * (float)(i + 1);
    }
    float q[4] = {0.5f, -0.5f, 0.25f, 0.75f};
    float k[4] = {0.3f, 0.8f, -0.2f, 0.1f};
    float v[4] = {1.0f, -1.0f, 0.5f, 0.25f};
    float beta = 0.7f, gamma = 0.9f;
    float b[4] = {beta, beta, beta, beta};
    float w[4] = {beta, beta, beta, beta};
    float a[4] = {gamma, gamma, gamma, gamma};
    double rS[16], ro[4], rq[4], rk[4], rv[4], rb[4], rw[4], ra[4];
    for (int i = 0; i < 16; i++) {
        rS[i] = (double)S[i];
    }
    for (int i = 0; i < 4; i++) {
        rq[i] = q[i];
        rk[i] = k[i];
        rb[i] = beta;
        rw[i] = beta;
        ra[i] = gamma;
        rv[i] = v[i];
    }
    int rc =
        jt_gdn2_decode_step(S, o, q, k, v, b, w, a, dk, dv, scratch, need);
    CHECK(rc == JT_OK, "kda rc=%d", rc);
    ref_step(rS, ro, rq, rk, rv, rb, rw, ra, dk, dv);
    for (int j = 0; j < dv; j++) {
        CHECK(fabs((double)o[j] - ro[j]) <= 2e-5, "kda o[%d]", j);
    }
    free(scratch);
    jt_gdn2_state_free(S);
}

static void test_l2norm(void) {
    float y[4];
    float x[4] = {3.0f, 4.0f, 0.0f, 0.0f};
    CHECK(jt_gdn2_l2norm(x, y, 4, 1e-6f) == JT_OK, "l2norm ok");
    CHECK(fabsf(y[0] - 0.6f) < 1e-5f && fabsf(y[1] - 0.8f) < 1e-5f,
          "l2norm val %f %f", y[0], y[1]);
    float z[3] = {0.0f, 0.0f, 0.0f};
    float yz[3] = {9.0f, 9.0f, 9.0f};
    CHECK(jt_gdn2_l2norm(z, yz, 3, 1e-6f) == JT_OK, "l2norm zero ok");
    CHECK(yz[0] == 0.0f && yz[1] == 0.0f && yz[2] == 0.0f,
          "l2norm zero guard");
    CHECK(jt_gdn2_l2norm(NULL, y, 4, 1e-6f) == JT_ERR_INVAL, "l2norm NULL x");
    CHECK(jt_gdn2_l2norm(x, NULL, 4, 1e-6f) == JT_ERR_INVAL, "l2norm NULL y");
    CHECK(jt_gdn2_l2norm(x, y, 0, 1e-6f) == JT_ERR_INVAL, "l2norm n=0");
    CHECK(jt_gdn2_l2norm(x, y, 4, 0.0f) == JT_ERR_INVAL, "l2norm eps=0");
}

static void test_state_helpers(void) {
    size_t n = 0;
    CHECK(jt_gdn2_state_bytes(4, 8, &n) == JT_OK && n == 128, "state_bytes");
    CHECK(jt_gdn2_state_bytes(0, 8, &n) == JT_ERR_INVAL, "state_bytes dk=0");
    CHECK(jt_gdn2_state_bytes(4, 8, NULL) == JT_ERR_INVAL,
          "state_bytes NULL");
    CHECK(jt_gdn2_state_bytes(2048, 8, &n) == JT_ERR_INVAL, "state_bytes big");
    CHECK(jt_gdn2_scratch_floats(4, 8, &n) == JT_OK && n == 24,
          "scratch=%zu", n);
    CHECK(jt_gdn2_scratch_floats(-1, 8, &n) == JT_ERR_INVAL,
          "scratch bad dims");
    float *S = NULL;
    CHECK(jt_gdn2_state_alloc(NULL, 4, 4) == JT_ERR_INVAL, "alloc NULL out");
    CHECK(jt_gdn2_state_alloc(&S, 0, 4) == JT_ERR_INVAL, "alloc dk=0");
    CHECK(jt_gdn2_state_alloc(&S, 4, 4) == JT_OK && S != NULL, "alloc ok");
    if (S != NULL) {
        S[0] = 1.0f;
        CHECK(jt_gdn2_state_reset(S, 4, 4) == JT_OK, "reset ok");
        CHECK(S[0] == 0.0f, "reset zero");
        CHECK(jt_gdn2_state_reset(NULL, 4, 4) == JT_ERR_INVAL,
              "reset NULL");
        jt_gdn2_state_free(S);
    }
    jt_gdn2_state_free(NULL);  // 無害
}

static void test_invalid_decode(void) {
    float *S = NULL;
    CHECK(jt_gdn2_state_alloc(&S, 4, 4) == JT_OK, "inv alloc");
    if (S == NULL) {
        return;
    }
    size_t need = 0;
    CHECK(jt_gdn2_scratch_floats(4, 4, &need) == JT_OK, "inv scratch");
    float *scratch = (float *)malloc(need * sizeof(float));
    float q[4] = {1, 0, 0, 0}, k[4] = {0, 1, 0, 0}, v[4] = {1, 1, 1, 1},
          b[4] = {1, 1, 1, 1}, w[4] = {1, 1, 1, 1}, a[4] = {1, 1, 1, 1},
          o[4] = {0, 0, 0, 0};
    CHECK(scratch != NULL, "inv malloc");
    if (scratch == NULL) {
        jt_gdn2_state_free(S);
        return;
    }
    errno = 0;
    CHECK(jt_gdn2_decode_step(NULL, o, q, k, v, b, w, a, 4, 4, scratch,
                              need) == JT_ERR_INVAL,
          "NULL S");
    CHECK(jt_gdn2_decode_step(S, NULL, q, k, v, b, w, a, 4, 4, scratch,
                              need) == JT_ERR_INVAL,
          "NULL o");
    CHECK(jt_gdn2_decode_step(S, o, NULL, k, v, b, w, a, 4, 4, scratch,
                              need) == JT_ERR_INVAL,
          "NULL q");
    CHECK(jt_gdn2_decode_step(S, o, q, k, v, b, w, a, 0, 4, scratch,
                              need) == JT_ERR_INVAL,
          "dk=0");
    CHECK(jt_gdn2_decode_step(S, o, q, k, v, b, w, a, 4, 4, scratch,
                              need - 1) == JT_ERR_INVAL,
          "scratch short");
    // 非整列 S は JT_ERR_ALIGN。
    {
        void *raw = malloc(4 * 4 * sizeof(float) + 64);
        CHECK(raw != NULL, "raw malloc");
        if (raw != NULL) {
            float *mis = (float *)((char *)raw + 1);  // 1B ずらし
            int rc = jt_gdn2_decode_step(mis, o, q, k, v, b, w, a, 4, 4,
                                         scratch, need);
            CHECK(rc == JT_ERR_ALIGN, "misalign rc=%d", rc);
            free(raw);
        }
    }
    // ゼロ q/k は NaN を出さず o=0。
    {
        float zq[4] = {0, 0, 0, 0}, zk[4] = {0, 0, 0, 0};
        float zo[4] = {9, 9, 9, 9};
        int rc = jt_gdn2_decode_step(S, zo, zq, zk, v, b, w, a, 4, 4,
                                     scratch, need);
        CHECK(rc == JT_OK, "zero qk rc=%d", rc);
        for (int j = 0; j < 4; j++) {
            CHECK(zo[j] == 0.0f, "zero qk o[%d]=%f", j, zo[j]);
        }
    }
    // MAJOR-2: 非有限・範囲外は JT_ERR_INVAL で拒否し S/o 不変 (fail-closed)。
    {
        float tq[4], tk[4], tv[4], tb[4], tw[4], ta[4], to[4];
        float Sref[16];
        float oref[4] = {7.0f, 8.0f, 9.0f, 10.0f};
        int rc;
        for (int i = 0; i < 4; i++) {
            tq[i] = q[i];
            tk[i] = k[i];
            tb[i] = b[i];
            ta[i] = 0.9f;
            tv[i] = v[i];
            tw[i] = w[i];
        }
        for (int i = 0; i < 16; i++) {
            S[i] = 0.25f * (float)(i + 1);
        }
        memcpy(Sref, S, sizeof(Sref));
        memcpy(to, oref, sizeof(to));
        // q NaN → 拒否。
        tq[0] = NAN;
        rc = jt_gdn2_decode_step(S, to, tq, tk, tv, tb, tw, ta, 4, 4,
                                 scratch, need);
        CHECK(rc == JT_ERR_INVAL, "nan q rc=%d", rc);
        CHECK(memcmp(S, Sref, sizeof(Sref)) == 0, "nan q S mutated");
        CHECK(memcmp(to, oref, sizeof(oref)) == 0, "nan q o mutated");
        tq[0] = q[0];
        // k Inf → 拒否。
        memcpy(Sref, S, sizeof(Sref));
        memcpy(to, oref, sizeof(to));
        tk[1] = INFINITY;
        rc = jt_gdn2_decode_step(S, to, tq, tk, tv, tb, tw, ta, 4, 4,
                                 scratch, need);
        CHECK(rc == JT_ERR_INVAL, "inf k rc=%d", rc);
        CHECK(memcmp(S, Sref, sizeof(Sref)) == 0, "inf k S mutated");
        CHECK(memcmp(to, oref, sizeof(oref)) == 0, "inf k o mutated");
        tk[1] = k[1];
        // v NaN → 拒否。
        memcpy(Sref, S, sizeof(Sref));
        memcpy(to, oref, sizeof(to));
        tv[2] = NAN;
        rc = jt_gdn2_decode_step(S, to, tq, tk, tv, tb, tw, ta, 4, 4,
                                 scratch, need);
        CHECK(rc == JT_ERR_INVAL, "nan v rc=%d", rc);
        CHECK(memcmp(S, Sref, sizeof(Sref)) == 0, "nan v S mutated");
        CHECK(memcmp(to, oref, sizeof(oref)) == 0, "nan v o mutated");
        tv[2] = v[2];
        // b Inf → 拒否。
        memcpy(Sref, S, sizeof(Sref));
        memcpy(to, oref, sizeof(to));
        tb[0] = INFINITY;
        rc = jt_gdn2_decode_step(S, to, tq, tk, tv, tb, tw, ta, 4, 4,
                                 scratch, need);
        CHECK(rc == JT_ERR_INVAL, "inf b rc=%d", rc);
        CHECK(memcmp(S, Sref, sizeof(Sref)) == 0, "inf b S mutated");
        CHECK(memcmp(to, oref, sizeof(oref)) == 0, "inf b o mutated");
        tb[0] = b[0];
        // w NaN → 拒否。
        memcpy(Sref, S, sizeof(Sref));
        memcpy(to, oref, sizeof(to));
        tw[3] = NAN;
        rc = jt_gdn2_decode_step(S, to, tq, tk, tv, tb, tw, ta, 4, 4,
                                 scratch, need);
        CHECK(rc == JT_ERR_INVAL, "nan w rc=%d", rc);
        CHECK(memcmp(S, Sref, sizeof(Sref)) == 0, "nan w S mutated");
        CHECK(memcmp(to, oref, sizeof(oref)) == 0, "nan w o mutated");
        tw[3] = w[3];
        // alpha NaN → 拒否。
        memcpy(Sref, S, sizeof(Sref));
        memcpy(to, oref, sizeof(to));
        ta[0] = NAN;
        rc = jt_gdn2_decode_step(S, to, tq, tk, tv, tb, tw, ta, 4, 4,
                                 scratch, need);
        CHECK(rc == JT_ERR_INVAL, "nan alpha rc=%d", rc);
        CHECK(memcmp(S, Sref, sizeof(Sref)) == 0, "nan alpha S mutated");
        CHECK(memcmp(to, oref, sizeof(oref)) == 0, "nan alpha o mutated");
        ta[0] = 0.9f;
        // alpha=2 (範囲外) → 拒否。
        memcpy(Sref, S, sizeof(Sref));
        memcpy(to, oref, sizeof(to));
        ta[1] = 2.0f;
        rc = jt_gdn2_decode_step(S, to, tq, tk, tv, tb, tw, ta, 4, 4,
                                 scratch, need);
        CHECK(rc == JT_ERR_INVAL, "alpha=2 rc=%d", rc);
        CHECK(memcmp(S, Sref, sizeof(Sref)) == 0, "alpha=2 S mutated");
        CHECK(memcmp(to, oref, sizeof(oref)) == 0, "alpha=2 o mutated");
        ta[1] = 0.9f;
        // alpha<0 (範囲外) → 拒否。
        memcpy(Sref, S, sizeof(Sref));
        memcpy(to, oref, sizeof(to));
        ta[2] = -0.25f;
        rc = jt_gdn2_decode_step(S, to, tq, tk, tv, tb, tw, ta, 4, 4,
                                 scratch, need);
        CHECK(rc == JT_ERR_INVAL, "alpha<0 rc=%d", rc);
        CHECK(memcmp(S, Sref, sizeof(Sref)) == 0, "alpha<0 S mutated");
        CHECK(memcmp(to, oref, sizeof(oref)) == 0, "alpha<0 o mutated");
        ta[2] = 0.9f;
        // alpha Inf → 拒否。
        memcpy(Sref, S, sizeof(Sref));
        memcpy(to, oref, sizeof(to));
        ta[3] = INFINITY;
        rc = jt_gdn2_decode_step(S, to, tq, tk, tv, tb, tw, ta, 4, 4,
                                 scratch, need);
        CHECK(rc == JT_ERR_INVAL, "inf alpha rc=%d", rc);
        CHECK(memcmp(S, Sref, sizeof(Sref)) == 0, "inf alpha S mutated");
        CHECK(memcmp(to, oref, sizeof(oref)) == 0, "inf alpha o mutated");
    }
    free(scratch);
    jt_gdn2_state_free(S);
}

static void test_prefill_stub(void) {
    float *S = NULL;
    CHECK(jt_gdn2_state_alloc(&S, 4, 4) == JT_OK, "pre alloc");
    if (S == NULL) {
        return;
    }
    float Out[16] = {0}, Q[16] = {0}, K[16] = {0}, V[16] = {0}, B[16] = {0},
          W[16] = {0}, A[16] = {0}, sc[8] = {0};
    errno = 0;
    int rc16 = jt_gdn2_prefill_chunk16(S, Out, Q, K, V, B, W, A, 4, 4, sc,
                                       8);
    CHECK(rc16 == JT_ERR_NOSUP && errno == ENOSYS, "chunk16 stub rc=%d e=%d",
          rc16, errno);
    errno = 0;
    int rc32 = jt_gdn2_prefill_chunk32(S, Out, Q, K, V, B, W, A, 4, 4, sc,
                                       8);
    CHECK(rc32 == JT_ERR_NOSUP && errno == ENOSYS, "chunk32 stub rc=%d e=%d",
          rc32, errno);
    errno = 0;
    CHECK(jt_gdn2_prefill_chunk16(NULL, Out, Q, K, V, B, W, A, 4, 4, sc,
                                  8) == JT_ERR_INVAL && errno == EINVAL,
          "chunk16 NULL");
    errno = 0;
    CHECK(jt_gdn2_prefill_chunk16(S, Out, Q, K, V, B, W, A, 0, 4, sc,
                                  8) == JT_ERR_INVAL && errno == EINVAL,
          "chunk16 bad dims");
    jt_gdn2_state_free(S);
}

int main(void) {
    test_state_helpers();
    test_l2norm();
    // 4分割境界の両側: 4の倍数 / 非倍数 / 小次元。
    test_numeric_case(4, 8, 1);
    test_numeric_case(5, 7, 2);
    test_numeric_case(2, 3, 3);
    test_numeric_case(8, 8, 4);
    test_numeric_sequence();
    test_kda_reduction();
    test_invalid_decode();
    test_prefill_stub();
    if (g_fail != 0) {
        fprintf(stderr, "gdn2: FAIL\n");
        return 1;
    }
    printf("gdn2: OK (decode numeric + invalid + prefill-stub)\n");
    return 0;
}

