// test: optim8収束スモーク + esmoe往復一致 (Phase 2足場)。
// - optim8: 二次関数最小化で損失減少 (AdamW-lite + int8状態でも収束)。
// - quantize/dequantize: 往復誤差有界 + 不正入力拒否。
// - esmoe: 書き込み→offload→読み戻し一致 + evict/prefetch/rows/stats。
// 成功時 exit 0、失敗時 exit 1 + stderr。
// C11、errnoベース (AGENTS.MD 7.1)。
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/esmoe.h"
#include "jimotono/optim8.h"

static int fails = 0;

#define CHECK(cond, ...)                                   \
    do {                                                   \
        if (!(cond)) {                                     \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                  \
            fprintf(stderr, "\n");                         \
            fails++;                                       \
        }                                                  \
    } while (0)

#define JT_NP 32
#define JT_STEPS 500

static float quad_loss(const float *restrict p, const float *restrict tgt,
                       size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)p[i] - (double)tgt[i];
        s += d * d;
    }
    return (float)s;
}

static void test_quant_roundtrip(void) {
    float src[128];
    for (int i = 0; i < 128; i++) {
        // 決定的パターン [-1,1)。ゼロブロック境界も含む。
        src[i] = (float)((i % 64) - 32) / 32.0f;
    }
    src[64] = 0.0f;
    int8_t q[128] = {0};
    float sc[2] = {0.0f, 0.0f};
    float dst[128] = {0.0f};
    int rc = jt_optim8_quantize(src, 128, q, sc, 2);
    CHECK(rc == JT_OK, "quant rc=%d", rc);
    rc = jt_optim8_dequantize(q, sc, 2, 128, dst);
    CHECK(rc == JT_OK, "dequant rc=%d", rc);
    if (rc == JT_OK) {
        float worst = 0.0f;
        for (int i = 0; i < 128; i++) {
            float e = fabsf(dst[i] - src[i]);
            worst = (e > worst) ? e : worst;
        }
        // scale≈1/127≈0.0079、半ステップ≈0.004。余裕を見て0.02。
        CHECK(worst < 0.02f, "roundtrip worst=%f want <0.02", worst);
    }
    // ゼロブロック: scale=1, q=0。
    {
        float z[64] = {0.0f};
        int8_t qz[64] = {0};
        float sz[1] = {0.0f};
        CHECK(jt_optim8_quantize(z, 64, qz, sz, 1) == JT_OK, "zero quant rc");
        CHECK(sz[0] == 1.0f, "zero scale=%f want 1.0", sz[0]);
    }
    // 不正入力。
    CHECK(jt_optim8_quantize(NULL, 128, q, sc, 2) == JT_ERR_INVAL,
          "quant NULL src");
    CHECK(jt_optim8_quantize(src, 0, q, sc, 0) == JT_ERR_INVAL,
          "quant n==0");
    CHECK(jt_optim8_quantize(src, 128, q, sc, 1) == JT_ERR_INVAL,
          "quant bad nblocks");
    CHECK(jt_optim8_dequantize(NULL, sc, 2, 128, dst) == JT_ERR_INVAL,
          "dequant NULL");
    CHECK(jt_optim8_dequantize(q, sc, 1, 128, dst) == JT_ERR_INVAL,
          "dequant bad nblocks");
    {
        float nan[64];
        for (int i = 0; i < 64; i++) {
            nan[i] = 0.0f;
        }
        nan[3] = NAN;
        CHECK(jt_optim8_quantize(nan, 64, q, sc, 1) == JT_ERR_INVAL,
              "quant NaN should fail");
    }
}

static void test_optim_converge(void) {
    float param[JT_NP];
    float tgt[JT_NP];
    float grad[JT_NP];
    for (int i = 0; i < JT_NP; i++) {
        param[i] = 0.0f;
        tgt[i] = 1.0f;
    }
    jt_optim8_cfg_t cfg;
    jt_optim8_cfg_default(&cfg);
    cfg.lr = 0.05f;
    jt_optim8_t opt;
    memset(&opt, 0, sizeof(opt));
    CHECK(jt_optim8_init(&opt, JT_NP, &cfg) == JT_OK, "optim init rc");
    float init_loss = quad_loss(param, tgt, JT_NP);
    for (int s = 0; s < JT_STEPS; s++) {
        for (int i = 0; i < JT_NP; i++) {
            grad[i] = 2.0f * (param[i] - tgt[i]);
        }
        int rc = jt_optim8_step(&opt, param, grad, JT_NP);
        if (rc != JT_OK) {
            CHECK(0, "optim step %d rc=%d", s, rc);
            break;
        }
    }
    float final_loss = quad_loss(param, tgt, JT_NP);
    CHECK(final_loss < init_loss * 0.1f,
          "converge final=%f init=%f (want <10%%)", final_loss, init_loss);
    // 不正入力 (状態を汚さないこと: rcのみ確認)。
    CHECK(jt_optim8_step(NULL, param, grad, JT_NP) == JT_ERR_INVAL,
          "step NULL opt");
    CHECK(jt_optim8_step(&opt, NULL, grad, JT_NP) == JT_ERR_INVAL,
          "step NULL param");
    CHECK(jt_optim8_step(&opt, param, NULL, JT_NP) == JT_ERR_INVAL,
          "step NULL grad");
    CHECK(jt_optim8_step(&opt, param, grad, JT_NP + 1) == JT_ERR_INVAL,
          "step n mismatch");
    {
        float bad[JT_NP];
        memcpy(bad, grad, sizeof(bad));
        bad[0] = INFINITY;
        CHECK(jt_optim8_step(&opt, param, bad, JT_NP) == JT_ERR_INVAL,
              "step INF grad");
    }
    jt_optim8_fini(&opt);
    // 二重fini安全。
    jt_optim8_fini(&opt);
    // 不正init。
    {
        jt_optim8_t o2;
        memset(&o2, 0, sizeof(o2));
        CHECK(jt_optim8_init(NULL, JT_NP, &cfg) == JT_ERR_INVAL,
              "init NULL opt");
        CHECK(jt_optim8_init(&o2, 0, &cfg) == JT_ERR_INVAL, "init n==0");
        jt_optim8_cfg_t bad = cfg;
        bad.lr = -1.0f;
        CHECK(jt_optim8_init(&o2, JT_NP, &bad) == JT_ERR_INVAL,
              "init bad lr");
    }
}

static void test_esmoe_roundtrip(void) {
    jt_esmoe_cfg_t cfg;
    cfg.n_experts = 4;
    cfg.expert_bytes = 64;
    cfg.row_bytes = 16;  // 4行/expert
    jt_esmoe_t es;
    memset(&es, 0, sizeof(es));
    CHECK(jt_esmoe_init(&es, &cfg) == JT_OK, "esmoe init rc=%d", errno);

    unsigned char pat[4][64];
    for (int e = 0; e < 4; e++) {
        for (int j = 0; j < 64; j++) {
            pat[e][j] = (unsigned char)((e * 64 + j) & 0xFF);
        }
        CHECK(jt_esmoe_register(&es, (uint32_t)e, pat[e], 64) == JT_OK,
              "register e=%d", e);
    }
    // prefetch {1,3} → load一致。
    {
        uint32_t ids[2] = {1, 3};
        CHECK(jt_esmoe_prefetch(&es, ids, 2) == JT_OK, "prefetch rc");
    }
    for (int e = 0; e < 4; e++) {
        unsigned char out[64] = {0};
        CHECK(jt_esmoe_load(&es, (uint32_t)e, out, 64) == JT_OK,
              "load e=%d", e);
        CHECK(memcmp(out, pat[e], 64) == 0, "roundtrip e=%d mismatch", e);
    }
    // evict→再loadでも一致 (SSD往復)。
    CHECK(jt_esmoe_evict(&es, 1) == JT_OK, "evict rc");
    {
        unsigned char out[64] = {0};
        CHECK(jt_esmoe_load(&es, 1, out, 64) == JT_OK, "reload rc");
        CHECK(memcmp(out, pat[1], 64) == 0, "reload mismatch");
    }
    // rows読み: expert=2, rows[1,3) = bytes[16,48)。
    {
        unsigned char rows[32] = {0};
        CHECK(jt_esmoe_read_rows(&es, 2, 1, 2, rows) == JT_OK,
              "read_rows rc");
        CHECK(memcmp(rows, pat[2] + 16, 32) == 0, "read_rows mismatch");
    }
    // stats: 読みが発生し、bytes>0。
    {
        jt_esmoe_stats_t st;
        memset(&st, 0, sizeof(st));
        CHECK(jt_esmoe_stats(&es, &st) == JT_OK, "stats rc");
        CHECK(st.loads > 0, "stats loads=%llu want >0",
              (unsigned long long)st.loads);
        CHECK(st.bytes_read > 0, "stats bytes=%llu want >0",
              (unsigned long long)st.bytes_read);
        CHECK(st.prefetches >= 1, "stats prefetches=%llu want >=1",
              (unsigned long long)st.prefetches);
    }
    // n==0 prefetchはOK。
    CHECK(jt_esmoe_prefetch(&es, NULL, 0) == JT_OK, "prefetch n==0");
    // 不正入力。
    {
        unsigned char out[64] = {0};
        CHECK(jt_esmoe_register(&es, 99, pat[0], 64) == JT_ERR_INVAL,
              "register bad id");
        CHECK(jt_esmoe_register(&es, 0, pat[0], 63) == JT_ERR_INVAL,
              "register bad len");
        CHECK(jt_esmoe_load(&es, 99, out, 64) == JT_ERR_INVAL,
              "load bad id");
        CHECK(jt_esmoe_load(&es, 0, out, 63) == JT_ERR_INVAL,
              "load bad len");
        CHECK(jt_esmoe_evict(&es, 99) == JT_ERR_INVAL, "evict bad id");
        CHECK(jt_esmoe_read_rows(&es, 2, 3, 2, out) == JT_ERR_INVAL,
              "rows overflow");
        CHECK(jt_esmoe_read_rows(&es, 2, 0, 0, out) == JT_ERR_INVAL,
              "rows count 0");
        CHECK(jt_esmoe_stats(&es, NULL) == JT_ERR_INVAL, "stats NULL");
        CHECK(jt_esmoe_stats(NULL, NULL) == JT_ERR_INVAL, "stats NULL es");
    }
    jt_esmoe_fini(&es);
    jt_esmoe_fini(&es);  // 二重fini安全
    // 不正init。
    {
        jt_esmoe_t e2;
        memset(&e2, 0, sizeof(e2));
        jt_esmoe_cfg_t bad = cfg;
        bad.n_experts = 0;
        CHECK(jt_esmoe_init(&e2, &bad) == JT_ERR_INVAL, "init n==0");
        CHECK(jt_esmoe_init(NULL, &cfg) == JT_ERR_INVAL, "init NULL");
        CHECK(jt_esmoe_init(&e2, NULL) == JT_ERR_INVAL, "init NULL cfg");
        bad = cfg;
        bad.row_bytes = 24;  // 64割り切れずINVAL
        CHECK(jt_esmoe_init(&e2, &bad) == JT_ERR_INVAL, "init bad rows");
    }
}

int main(void) {
    CHECK(jt_optim8_nblocks(1) == 1, "nblocks(1)==1");
    CHECK(jt_optim8_nblocks(64) == 1, "nblocks(64)==1");
    CHECK(jt_optim8_nblocks(65) == 2, "nblocks(65)==2");
    test_quant_roundtrip();
    test_optim_converge();
    test_esmoe_roundtrip();
    if (fails == 0) {
        printf("optim_offload: OK\n");
        return 0;
    }
    fprintf(stderr, "optim_offload: %d FAIL(s)\n", fails);
    return 1;
}
