// bench+test: T-MAC LUT kernel correctness + microbench (Phase 1).
// cases: bit-width {1,2,3,4} x act {dynamic int8 LUT}; metric: us/call.
// 正常系+異常系+アライメントを検証 (成功時 exit 0)、末尾に簡易benchを出力。
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "jimotono/arena.h"
#include "jimotono/common.h"
#include "jimotono/tmac.h"

static int g_fail = 0;

static void fail_at(int line, const char *msg) {
    printf("FAIL line %d: %s\n", line, msg);
    g_fail = 1;
}

#define CHECK(cond, msg)              \
    do {                              \
        if (!(cond)) {                \
            fail_at(__LINE__, (msg)); \
        }                             \
    } while (0)

// 1ブロック(32 acts)x1プレーンの厳密結合値 (量子化なし参照)。
static double exact_plane_sum(const float *restrict abase, const uint8_t *restrict plane) {
    double s = 0.0;
    for (size_t gi = 0; gi < 8u; gi++) {
        unsigned p = (unsigned)(plane[gi] & 0x0Fu);
        for (unsigned j = 0u; j < 4u; j++) {
            double a = (double)abase[gi * 4u + (size_t)j];
            s += (((p >> j) & 1u) != 0u) ? a : -a;
        }
    }
    return s;
}

static void test_sizes(void) {
    size_t v = 0;
    CHECK(jt_tmac_qlut_bytes(1u, 32u, &v) == JT_OK && v == 64u, "qlut_bytes(1,32)==64");
    CHECK(jt_tmac_param_bytes(1u, 32u, &v) == JT_OK && v == 4u, "param_bytes(1,32)==4");
    CHECK(jt_tmac_qlut_bytes(2u, 64u, &v) == JT_OK && v == 256u, "qlut_bytes(2,64)==256");
    CHECK(jt_tmac_param_bytes(2u, 64u, &v) == JT_OK && v == 16u, "param_bytes(2,64)==16");
    CHECK(jt_tmac_qlut_bytes(0u, 32u, &v) == JT_ERR_INVAL, "qlut n==0");
    CHECK(jt_tmac_qlut_bytes(1u, 0u, &v) == JT_ERR_INVAL, "qlut k==0");
    CHECK(jt_tmac_qlut_bytes(1u, 30u, &v) == JT_ERR_INVAL, "qlut K%4");
    CHECK(jt_tmac_param_bytes(1u, 20u, &v) == JT_ERR_INVAL, "param K%32");
    CHECK(jt_tmac_qlut_bytes(1u, 32u, NULL) == JT_ERR_INVAL, "qlut NULL out");
    CHECK(jt_tmac_param_bytes(1u, 32u, NULL) == JT_ERR_INVAL, "param NULL out");
    CHECK(jt_tmac_qlut_bytes(SIZE_MAX, 4u, &v) == JT_ERR_NOMEM, "qlut overflow");
    CHECK(jt_tmac_param_bytes(SIZE_MAX, 32u, &v) == JT_ERR_NOMEM, "param overflow");
}

static void test_ctor_zeros(void) {
    float act[32] = {0.0f};
    int8_t q[64];
    float sc[1] = {0.0f};
    float bi[1] = {0.0f};
    memset(q, 0xAA, sizeof q);
    CHECK(jt_tmac_lut_ctor(act, 1u, 32u, q, sc, bi) == JT_OK, "ctor zeros rc");
    CHECK(sc[0] == 1.0f, "ctor zeros scale");
    CHECK(bi[0] == 0.0f, "ctor zeros bias");
    for (size_t i = 0; i < 64u; i++) {
        if (q[i] != 0) {
            CHECK(0, "ctor zeros qlut");
            break;
        }
    }
}

static void test_ctor_exact(void) {
    float act[32];
    int8_t q[64];
    float sc[1];
    float bi[1];
    int8_t exp0[8] = {-127, 127, -127, 127, -127, 127, -127, 127};
    memset(act, 0, sizeof act);
    memset(q, 0xAA, sizeof q);
    act[0] = 8.0f;
    CHECK(jt_tmac_lut_ctor(act, 1u, 32u, q, sc, bi) == JT_OK, "ctor exact rc");
    CHECK(sc[0] == (float)(8.0 / 127.0), "ctor exact scale");
    CHECK(bi[0] == 8.0f, "ctor exact bias");
    CHECK(memcmp(q, exp0, 8u) == 0, "ctor exact q group0");
    for (size_t i = 8u; i < 64u; i++) {
        if (q[i] != 0) {
            CHECK(0, "ctor exact q rest zero");
            break;
        }
    }
}

static void test_mirror(void) {
    int8_t q[64];
    uint8_t idx[8];
    float sc[1] = {1.0f};
    float bi[1] = {0.0f};
    float out = 0.0f;
    for (size_t i = 0; i < 64u; i++) {
        q[i] = (int8_t)(i + 1u);
    }
    memset(idx, 15, sizeof idx);
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 1, 1.0f, &out) == JT_OK && out == 144.0f,
          "mirror p=15");
    memset(idx, 0, sizeof idx);
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 1, 1.0f, &out) == JT_OK && out == -144.0f,
          "mirror p=0");
    memset(idx, 8, sizeof idx);
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 1, 1.0f, &out) == JT_OK && out == 116.0f,
          "mirror p=8");
    memset(idx, 7, sizeof idx);
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 1, 1.0f, &out) == JT_OK && out == -116.0f,
          "mirror p=7");
    // 上位nibbleは無視される (0xA0 | 0 == p=0 と等価)。
    for (size_t i = 0; i < 8u; i++) {
        idx[i] = (uint8_t)0xA0u;
    }
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 1, 1.0f, &out) == JT_OK && out == -144.0f,
          "idx high nibble ignored");
}

static void test_bits(void) {
    int8_t q[64];
    uint8_t idx[4 * 8];
    float sc[1] = {1.0f};
    float bi[1] = {0.0f};
    float out = 0.0f;
    for (size_t i = 0; i < 64u; i++) {
        q[i] = (int8_t)(i + 1u);
    }
    // bits=2: plane0 all 15 (acc 288), plane1 all 8 (acc 232) -> 144 + 116*2 = 376。
    for (size_t i = 0; i < 8u; i++) {
        idx[i] = 15u;
        idx[8u + i] = 8u;
    }
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 2, 1.0f, &out) == JT_OK && out == 376.0f,
          "bits=2 weighting");
    // bits=3: planes 15,14,13 -> (288 + 280*2 + 272*4)/2 = 968。
    for (size_t i = 0; i < 8u; i++) {
        idx[i] = 15u;
        idx[8u + i] = 14u;
        idx[16u + i] = 13u;
    }
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 3, 1.0f, &out) == JT_OK && out == 968.0f,
          "bits=3 weighting");
    // bits=4: planes 15,14,13,12 -> (288 + 280*2 + 272*4 + 264*8)/2 = 2024。
    for (size_t i = 0; i < 8u; i++) {
        idx[i] = 15u;
        idx[8u + i] = 14u;
        idx[16u + i] = 13u;
        idx[24u + i] = 12u;
    }
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 4, 1.0f, &out) == JT_OK && out == 2024.0f,
          "bits=4 weighting");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 4, 0.5f, &out) == JT_OK && out == 1012.0f,
          "w_scale");
    // bias=4: 各planeに +2、重み合計15 -> 2024 + 30 = 2054。
    bi[0] = 4.0f;
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 4, 1.0f, &out) == JT_OK && out == 2054.0f,
          "bias");
}

static void test_roundtrip(void) {
    enum { TN = 1, TK = 64, TNG = 16, TNB = 2, TBITS = 4 };
    float act[TN * TK];
    int8_t q[TNG * 8];
    float sc[TNB];
    float bi[TNB];
    uint8_t idx[TBITS * TNG];
    float w = 1.5f;
    float out = 0.0f;
    for (size_t i = 0; i < (size_t)(TN * TK); i++) {
        int v = (int)((i * 37u + 11u) % 13u) - 6;
        act[i] = (float)v * 0.25f;
    }
    for (size_t i = 0; i < (size_t)(TBITS * TNG); i++) {
        idx[i] = (uint8_t)((i * 5u + 3u) % 16u);
    }
    CHECK(jt_tmac_lut_ctor(act, (size_t)TN, (size_t)TK, q, sc, bi) == JT_OK, "roundtrip ctor");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, (size_t)TNG, (size_t)TNB, TBITS, w, &out) == JT_OK,
          "roundtrip lookup");
    if (g_fail != 0) {
        return;
    }
    {
        double exact = 0.0;
        double bound = 0.0;
        for (size_t bb = 0; bb < (size_t)TNB; bb++) {
            const float *abase = act + bb * 32u;
            double bsum = 0.0;
            for (size_t i = 0; i < 32u; i++) {
                bsum += (double)abase[i];
            }
            CHECK(bi[bb] == (float)bsum, "roundtrip bias");
            for (int b = 0; b < TBITS; b++) {
                const uint8_t *plane = idx + (size_t)b * (size_t)TNG + bb * 8u;
                double f = exact_plane_sum(abase, plane);
                double pw = (double)(1 << b);
                exact += ((f + bsum) * 0.5) * pw;
                bound += (2.0 * (double)sc[bb]) * pw;
            }
        }
        exact *= (double)w;
        bound *= ((double)w >= 0.0) ? (double)w : -(double)w;
        {
            double diff = (double)out - exact;
            double ad = (diff >= 0.0) ? diff : -diff;
            double ae = (exact >= 0.0) ? exact : -exact;
            double tol = bound + 1e-4 * ae + 1e-3;
            if (!(ad <= tol)) {
                char buf[160];
                snprintf(buf, sizeof buf, "roundtrip bound ad=%g tol=%g", ad, tol);
                fail_at(__LINE__, buf);
            }
        }
    }
}

static void test_roundtrip_bits3(void) {
    enum { TN3 = 1, TK3 = 64, TNG3 = 16, TNB3 = 2, TBITS3 = 3 };
    float act[TN3 * TK3];
    int8_t q[TNG3 * 8];
    float sc[TNB3];
    float bi[TNB3];
    uint8_t idx[TBITS3 * TNG3];
    float w = 1.5f;
    float out = 0.0f;
    for (size_t i = 0; i < (size_t)(TN3 * TK3); i++) {
        int v = (int)((i * 37u + 11u) % 13u) - 6;
        act[i] = (float)v * 0.25f;
    }
    for (size_t i = 0; i < (size_t)(TBITS3 * TNG3); i++) {
        idx[i] = (uint8_t)((i * 5u + 3u) % 16u);
    }
    CHECK(jt_tmac_lut_ctor(act, (size_t)TN3, (size_t)TK3, q, sc, bi) == JT_OK, "roundtrip3 ctor");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, (size_t)TNG3, (size_t)TNB3, TBITS3, w, &out)
              == JT_OK,
          "roundtrip3 lookup");
    if (g_fail != 0) {
        return;
    }
    {
        double exact = 0.0;
        double bound = 0.0;
        for (size_t bb = 0; bb < (size_t)TNB3; bb++) {
            const float *abase = act + bb * 32u;
            double bsum = 0.0;
            for (size_t i = 0; i < 32u; i++) {
                bsum += (double)abase[i];
            }
            CHECK(bi[bb] == (float)bsum, "roundtrip3 bias");
            for (int b = 0; b < TBITS3; b++) {
                const uint8_t *plane = idx + (size_t)b * (size_t)TNG3 + bb * 8u;
                double f = exact_plane_sum(abase, plane);
                double pw = (double)(1 << b);
                exact += ((f + bsum) * 0.5) * pw;
                bound += (2.0 * (double)sc[bb]) * pw;
            }
        }
        exact *= (double)w;
        bound *= ((double)w >= 0.0) ? (double)w : -(double)w;
        {
            double diff = (double)out - exact;
            double ad = (diff >= 0.0) ? diff : -diff;
            double ae = (exact >= 0.0) ? exact : -exact;
            double tol = bound + 1e-4 * ae + 1e-3;
            if (!(ad <= tol)) {
                char buf[160];
                snprintf(buf, sizeof buf, "roundtrip3 bound ad=%g tol=%g", ad, tol);
                fail_at(__LINE__, buf);
            }
        }
    }
}

static void test_errors(void) {
    float act[32] = {0.0f};
    int8_t q[64];
    float sc[1] = {1.0f};
    float bi[1] = {0.0f};
    float out = 0.0f;
    uint8_t idx[8] = {0u};
    float a20[20] = {0.0f};
    int8_t q20[40];
    float s20[1];
    float b20[1];
    float a36[36] = {0.0f};
    int8_t q36[72];
    float s36[2];
    float b36[2];
    CHECK(jt_tmac_lut_ctor(NULL, 1u, 32u, q, sc, bi) == JT_ERR_INVAL, "ctor NULL act");
    CHECK(jt_tmac_lut_ctor(act, 1u, 32u, NULL, sc, bi) == JT_ERR_INVAL, "ctor NULL qlut");
    CHECK(jt_tmac_lut_ctor(act, 1u, 32u, q, NULL, bi) == JT_ERR_INVAL, "ctor NULL scales");
    CHECK(jt_tmac_lut_ctor(act, 1u, 32u, q, sc, NULL) == JT_ERR_INVAL, "ctor NULL biases");
    CHECK(jt_tmac_lut_ctor(act, 0u, 32u, q, sc, bi) == JT_ERR_INVAL, "ctor n==0");
    CHECK(jt_tmac_lut_ctor(act, 1u, 0u, q, sc, bi) == JT_ERR_INVAL, "ctor k==0");
    CHECK(jt_tmac_lut_ctor(a20, 1u, 20u, q20, s20, b20) == JT_ERR_INVAL, "ctor K%32 (20)");
    CHECK(jt_tmac_lut_ctor(a36, 1u, 36u, q36, s36, b36) == JT_ERR_INVAL, "ctor K%32 (36)");
    act[0] = (float)NAN;
    CHECK(jt_tmac_lut_ctor(act, 1u, 32u, q, sc, bi) == JT_ERR_INVAL, "ctor NaN");
    act[0] = (float)INFINITY;
    CHECK(jt_tmac_lut_ctor(act, 1u, 32u, q, sc, bi) == JT_ERR_INVAL, "ctor Inf");
    act[0] = 0.0f;
    CHECK(jt_tmac_lookup_accum(NULL, idx, sc, bi, 8u, 1u, 1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup NULL qlut");
    CHECK(jt_tmac_lookup_accum(q, NULL, sc, bi, 8u, 1u, 1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup NULL idx");
    CHECK(jt_tmac_lookup_accum(q, idx, NULL, bi, 8u, 1u, 1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup NULL scales");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, NULL, 8u, 1u, 1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup NULL biases");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 1, 1.0f, NULL) == JT_ERR_INVAL,
          "lookup NULL out");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 0u, 1u, 1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup ngroups==0");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 0u, 1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup nblocks==0");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 7u, 1u, 1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup ngroups!=8*nblocks");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 0, 1.0f, &out) == JT_ERR_INVAL,
          "lookup bits==0");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 5, 1.0f, &out) == JT_ERR_INVAL,
          "lookup bits==5");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, -1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup bits==-1");
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 1, (float)NAN, &out) == JT_ERR_INVAL,
          "lookup NaN w_scale");
    sc[0] = (float)NAN;
    bi[0] = 0.0f;
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup NaN scale");
    sc[0] = 1.0f;
    bi[0] = (float)INFINITY;
    CHECK(jt_tmac_lookup_accum(q, idx, sc, bi, 8u, 1u, 1, 1.0f, &out) == JT_ERR_INVAL,
          "lookup Inf bias");
}

static void test_arena_align(void) {
    jt_arena_t ar;
    float *act = NULL;
    int8_t *qq = NULL;
    float *ss = NULL;
    float *bb = NULL;
    uint8_t idx[32];
    float out = 0.0f;
    CHECK(jt_arena_init(&ar, 4096u) == JT_OK, "arena init");
    if (g_fail != 0) {
        return;
    }
    act = (float *)jt_arena_alloc(&ar, 32u * sizeof(float));
    qq = (int8_t *)jt_arena_alloc(&ar, 64u * sizeof(int8_t));
    ss = (float *)jt_arena_alloc(&ar, (size_t)sizeof(float));
    bb = (float *)jt_arena_alloc(&ar, (size_t)sizeof(float));
    CHECK(act != NULL && qq != NULL && ss != NULL && bb != NULL, "arena alloc");
    if (act == NULL || qq == NULL || ss == NULL || bb == NULL) {
        jt_arena_fini(&ar);
        return;
    }
    CHECK(((uintptr_t)act % 64u) == 0u, "arena act 64B");
    CHECK(((uintptr_t)qq % 64u) == 0u, "arena qlut 64B");
    CHECK(((uintptr_t)ss % 64u) == 0u, "arena scales 64B");
    CHECK(((uintptr_t)bb % 64u) == 0u, "arena biases 64B");
    for (size_t i = 0; i < 32u; i++) {
        act[i] = (float)((int)(i % 7u) - 3) * 0.5f;
    }
    for (size_t i = 0; i < 32u; i++) {
        idx[i] = (uint8_t)(i % 16u);
    }
    CHECK(jt_tmac_lut_ctor(act, 1u, 32u, qq, ss, bb) == JT_OK, "arena ctor");
    CHECK(jt_tmac_lookup_accum(qq, idx, ss, bb, 8u, 1u, 4, 1.0f, &out) == JT_OK, "arena lookup");
    CHECK(isfinite((double)out) != 0, "arena out finite");
    jt_arena_fini(&ar);
}

static void run_bench(void) {
    enum { BK = 512, BNG = 128, BNB = 16 };
    static float bact[BK];
    static int8_t bq[BNG * 8];
    static float bsc[BNB];
    static float bbi[BNB];
    static uint8_t bidx[4 * BNG];
    const long iters = 2000L;
    const int bits_list[4] = {1, 2, 3, 4};
    for (size_t i = 0; i < (size_t)BK; i++) {
        bact[i] = (float)((int)(i % 9u) - 4) * 0.25f;
    }
    for (size_t i = 0; i < (size_t)(4 * BNG); i++) {
        bidx[i] = (uint8_t)((i * 7u + 1u) % 16u);
    }
    if (jt_tmac_lut_ctor(bact, 1u, (size_t)BK, bq, bsc, bbi) != JT_OK) {
        fail_at(__LINE__, "bench ctor");
        return;
    }
    for (size_t li = 0; li < 4u; li++) {
        int bits = bits_list[li];
        clock_t t0 = clock();
        double sink = 0.0;
        for (long it = 0; it < iters; it++) {
            float o = 0.0f;
            if (jt_tmac_lookup_accum(bq, bidx, bsc, bbi, (size_t)BNG, (size_t)BNB, bits, 1.0f, &o)
                != JT_OK) {
                fail_at(__LINE__, "bench lookup");
                return;
            }
            sink += (double)o;
        }
        {
            clock_t t1 = clock();
            double sec = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;
            double us = sec * 1e6 / (double)iters;
            printf("bench tmac K=%d bits=%d: %.3f us/call (sink=%.3f)\n", BK, bits, us, sink);
        }
    }
}

int main(void) {
    test_sizes();
    test_ctor_zeros();
    test_ctor_exact();
    test_mirror();
    test_bits();
    test_roundtrip();
    test_roundtrip_bits3();
    test_errors();
    test_arena_align();
    if (g_fail != 0) {
        printf("tmac: FAIL\n");
        return 1;
    }
    printf("tmac: OK\n");
    run_bench();
    if (g_fail != 0) {
        printf("tmac: FAIL\n");
        return 1;
    }
    return 0;
}

