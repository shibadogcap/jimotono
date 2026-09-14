// test_mixed_lut: 混合精度ポリシー + LUT-aware足場の検証。
// - ビット幅ポリシー表 (shared=8, down=4, gate/up=2, attn=8, emb=8)
// - 量子化往復誤差 (平均絶対誤差: INT8<0.02, INT4<0.1, INT2<0.3)
//   ※INT2は2bitグリッドが粗いためmaxではなくmeanで評価する (理由は下記)。
// - STEスモーク (微小ステップで損失が発散しない)
// - エクスポート記述子の一貫性 (再計算一致・同ページ配置)
// 成功時 exit 0 + "mixed_lut: OK"。
#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/common.h"
#include "jimotono/lut_aware.h"
#include "jimotono/mixed_prec.h"

#if defined(_WIN32)
#include <malloc.h>
// MSVCにaligned_allocがないため_aligned_mallocで代替。
static void *td_aligned_alloc(size_t align, size_t size) {
    return _aligned_malloc(size, align);
}
static void td_aligned_free(void *p) {
    _aligned_free(p);
}
#else
static void *td_aligned_alloc(size_t align, size_t size) {
    return aligned_alloc(align, size);
}
static void td_aligned_free(void *p) {
    free(p);
}
#endif

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

// 決定的な疑似データ ([-amp, amp])。
static float test_val(size_t i, float amp, unsigned salt) {
    unsigned u = (unsigned)((i * 2654435761u + salt * 40503u) >> 9) % 2001u;
    return (((float)u / 1000.0f) - 1.0f) * amp;
}

static void test_policy(void) {
    int b = 0;
    CHECK(jt_mp_bits_for(JT_MP_SHARED, &b) == JT_OK && b == 8, "shared=8 got=%d", b);
    CHECK(jt_mp_bits_for(JT_MP_ROUTING_DOWN, &b) == JT_OK && b == 4, "down=4 got=%d", b);
    CHECK(jt_mp_bits_for(JT_MP_ROUTING_GATE, &b) == JT_OK && b == 2, "gate=2 got=%d", b);
    CHECK(jt_mp_bits_for(JT_MP_ROUTING_UP, &b) == JT_OK && b == 2, "up=2 got=%d", b);
    CHECK(jt_mp_bits_for(JT_MP_ATTN, &b) == JT_OK && b == 8, "attn=8 got=%d", b);
    CHECK(jt_mp_bits_for(JT_MP_EMB, &b) == JT_OK && b == 8, "emb=8 got=%d", b);
    CHECK(jt_mp_bits_for(JT_MP_LAYER_COUNT, &b) == JT_ERR_INVAL, "bad layer");
    CHECK(jt_mp_bits_for(JT_MP_SHARED, NULL) == JT_ERR_INVAL, "NULL out");
    {
        int q = 0;
        CHECK(jt_mp_qmax_for(8, &q) == JT_OK && q == 127, "qmax8");
        CHECK(jt_mp_qmax_for(4, &q) == JT_OK && q == 7, "qmax4");
        CHECK(jt_mp_qmax_for(2, &q) == JT_OK && q == 1, "qmax2");
        CHECK(jt_mp_qmax_for(3, &q) == JT_ERR_INVAL, "qmax3 inval");
    }
    {
        size_t nb = 0;
        CHECK(jt_mp_nblocks(64, 32, &nb) == JT_OK && nb == 2, "nblocks");
        CHECK(jt_mp_nblocks(0, 32, &nb) == JT_ERR_INVAL, "nblocks n=0");
        CHECK(jt_mp_nblocks(64, 0, &nb) == JT_ERR_INVAL, "nblocks b=0");
    }
    // FRI-MxMoE風API: 高感度→上げ提案、低感度→下げ提案、不正→INVAL。
    {
        jt_mp_sensitivity_t hi = {2.0f, 2.0f, 2.0f};
        jt_mp_sensitivity_t lo = {0.01f, 0.01f, 0.01f};
        jt_mp_sensitivity_t bad = {(float)NAN, 0.0f, 0.0f};
        float s = 0.0f;
        int pb = 0;
        CHECK(jt_mp_sensitivity_score(&hi, &s) == JT_OK && s > 2.0f, "hi score=%f", s);
        CHECK(jt_mp_sensitivity_score(&lo, &s) == JT_OK && s < 0.5f, "lo score=%f", s);
        CHECK(jt_mp_sensitivity_score(&bad, &s) == JT_ERR_INVAL, "nan score");
        CHECK(jt_mp_sensitivity_score(NULL, &s) == JT_ERR_INVAL, "NULL st");
        // down(4bit)基準: 高感度→8、低感度→2。
        CHECK(jt_mp_propose_bits(JT_MP_ROUTING_DOWN, &hi, &pb) == JT_OK && pb == 8,
              "propose up=%d", pb);
        CHECK(jt_mp_propose_bits(JT_MP_ROUTING_DOWN, &lo, &pb) == JT_OK && pb == 2,
              "propose down=%d", pb);
        CHECK(jt_mp_propose_bits(JT_MP_ROUTING_DOWN, &bad, &pb) == JT_ERR_INVAL,
              "propose bad");
        CHECK(jt_mp_propose_bits(JT_MP_LAYER_COUNT, &hi, &pb) == JT_ERR_INVAL,
              "propose bad layer");
    }
}

// bits量子化の往復平均絶対誤差を返す。失敗時は-1。
static double roundtrip_mae(int bits, size_t n, size_t block, float amp) {
    float *src = (float *)malloc(n * sizeof(float));
    int8_t *q = (int8_t *)malloc(n * sizeof(int8_t));
    float *dq = (float *)malloc(n * sizeof(float));
    size_t nb = 0;
    double mae = -1.0;
    if (!src || !q || !dq) {
        free(src);
        free(q);
        free(dq);
        return -1.0;
    }
    if (jt_mp_nblocks(n, block, &nb) != JT_OK) {
        goto done;
    }
    {
        float *sc = (float *)malloc(nb * sizeof(float));
        if (!sc) {
            goto done;
        }
        for (size_t i = 0; i < n; i++) {
            src[i] = test_val(i, amp, 7u);
        }
        if (jt_mp_quantize(src, n, bits, block, q, sc, nb) != JT_OK) {
            free(sc);
            goto done;
        }
        if (jt_mp_dequantize(q, sc, n, bits, block, nb, dq) != JT_OK) {
            free(sc);
            goto done;
        }
        free(sc);
    }
    {
        double acc = 0.0;
        for (size_t i = 0; i < n; i++) {
            acc += fabs((double)src[i] - (double)dq[i]);
        }
        mae = acc / (double)n;
    }
done:
    free(src);
    free(q);
    free(dq);
    return mae;
}

static void test_roundtrip(void) {
    const size_t n = 256;
    const size_t block = 32;
    double e8 = roundtrip_mae(8, n, block, 1.0f);
    double e4 = roundtrip_mae(4, n, block, 1.0f);
    // INT2はグリッドが {-2,-1,0,1}*scale と粗く、一様データのmax誤差は
    // 0.5*scaleに達するため、粗い上限は平均誤差で評価する。
    double e2 = roundtrip_mae(2, n, block, 1.0f);
    CHECK(e8 >= 0.0 && e8 < 0.02, "INT8 mae=%f", e8);
    CHECK(e4 >= 0.0 && e4 < 0.10, "INT4 mae=%f", e4);
    CHECK(e2 >= 0.0 && e2 < 0.30, "INT2 mae=%f", e2);
    printf("roundtrip mae: int8=%.5f int4=%.5f int2=%.5f\n", e8, e4, e2);
    // fail-closed: NaN混入・不正bits・q破損はINVAL。
    {
        float src[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        int8_t q[4] = {0};
        float sc[1] = {0};
        float dq[4] = {0};
        src[2] = (float)NAN;
        CHECK(jt_mp_quantize(src, 4, 8, 4, q, sc, 1) == JT_ERR_INVAL, "quant nan");
        src[2] = 0.3f;
        CHECK(jt_mp_quantize(src, 4, 3, 4, q, sc, 1) == JT_ERR_INVAL, "quant bits=3");
        CHECK(jt_mp_quantize(src, 4, 8, 4, q, sc, 2) == JT_ERR_INVAL, "quant nblocks");
        CHECK(jt_mp_quantize(NULL, 4, 8, 4, q, sc, 1) == JT_ERR_INVAL, "quant NULL");
        CHECK(jt_mp_quantize(src, 4, 8, 4, q, sc, 1) == JT_OK, "quant ok");
        CHECK(jt_mp_dequantize(q, sc, 4, 8, 4, 1, dq) == JT_OK, "dequant ok");
        q[0] = 100; // INT8 qmax=127内だが別bits文脈では破損扱いになる例
        CHECK(jt_mp_dequantize(q, sc, 4, 2, 4, 1, dq) == JT_ERR_INVAL,
              "dequant qrange bits=2");
        sc[0] = (float)NAN;
        q[0] = 0;
        CHECK(jt_mp_dequantize(q, sc, 4, 8, 4, 1, dq) == JT_ERR_INVAL,
              "dequant nan scale");
    }
}

static void test_ste_smoke(void) {
    // 微小線形回帰: X=[1,0.5], W=[0.5,-0.25], b=[0.1], target=[0.8]。
    // fwd→STE素通し→微小SGD 1歩で損失が発散しないこと。
    float X[2] = {1.0f, 0.5f};
    float W[2] = {0.5f, -0.25f};
    float b[1] = {0.1f};
    float Y[1] = {0};
    float target[1] = {0.8f};
    jt_lut_err_t err;
    const float lr = 1e-3f;
    double loss0 = 0.0, loss1 = 0.0;
    CHECK(jt_lut_dense_fwd(X, W, b, Y, 1, 1, 2, 4, 2, &err) == JT_OK, "ste fwd rc");
    CHECK(isfinite((double)err.mse) && isfinite((double)err.max_abs_err), "ste err finite");
    loss0 = (double)(Y[0] - target[0]) * (double)(Y[0] - target[0]);
    {
        // dL/dY = 2(Y-t)。STEで dL/dW相当へ素通しできること。
        float up[1] = {(float)(2.0 * (double)(Y[0] - target[0]))};
        float dn[1] = {0};
        CHECK(jt_lut_ste_pass(up, dn, 1) == JT_OK && dn[0] == up[0], "ste pass");
        CHECK(jt_lut_ste_pass(up, NULL, 1) == JT_ERR_INVAL, "ste NULL");
        {
            float bad[1] = {(float)NAN};
            CHECK(jt_lut_ste_pass(bad, dn, 1) == JT_ERR_INVAL, "ste nan");
        }
        // 微小ステップ (W[0]のみ更新する簡易スモーク)。
        W[0] -= lr * dn[0] * X[0];
        W[1] -= lr * dn[0] * X[1];
    }
    CHECK(jt_lut_dense_fwd(X, W, b, Y, 1, 1, 2, 4, 2, NULL) == JT_OK, "ste fwd2");
    loss1 = (double)(Y[0] - target[0]) * (double)(Y[0] - target[0]);
    CHECK(isfinite(loss1) && loss1 <= loss0 + 1e-6, "ste loss0=%f loss1=%f", loss0,
          loss1);
    printf("ste smoke: loss0=%.6f loss1=%.6f mse=%.6f\n", loss0, loss1, (double)err.mse);
    // 不正系。
    CHECK(jt_lut_dense_fwd(NULL, W, b, Y, 1, 1, 2, 4, 2, NULL) == JT_ERR_INVAL,
          "fwd NULL X");
    CHECK(jt_lut_dense_fwd(X, W, b, Y, 0, 1, 2, 4, 2, NULL) == JT_ERR_INVAL,
          "fwd m=0");
    CHECK(jt_lut_dense_fwd(X, W, b, Y, 1, 1, 2, 3, 2, NULL) == JT_ERR_INVAL,
          "fwd bits=3");
    {
        float Xn[2] = {1.0f, (float)NAN};
        jt_lut_err_t e2;
        CHECK(jt_lut_dense_fwd(Xn, W, b, Y, 1, 1, 2, 4, 2, &e2) == JT_ERR_INVAL,
              "fwd nan X");
    }
}

static void test_export(void) {
    jt_lut_export_desc_t a, c;
    CHECK(jt_lut_export_desc(64, 128, 4, 32, &a) == JT_OK, "export rc");
    CHECK(a.magic == JT_LUT_EXPORT_MAGIC, "export magic");
    CHECK(a.bits == 4 && a.rows == 64 && a.cols == 128 && a.block_len == 32,
          "export dims");
    // table=64*128*4/8=4096, scales=64*4*4=1024,
    // scale_off=align64(4096)=4096, page=align64(5120)=5120 (>4K → same_page=0)。
    CHECK(a.table_bytes == 4096u, "export table=%llu", (unsigned long long)a.table_bytes);
    CHECK(a.scale_bytes == 1024u, "export scale=%llu", (unsigned long long)a.scale_bytes);
    CHECK(a.table_offset == 0u, "export toff");
    CHECK(a.scale_offset == 4096u, "export soff=%llu",
          (unsigned long long)a.scale_offset);
    CHECK(a.page_bytes == 5120u, "export page=%llu", (unsigned long long)a.page_bytes);
    CHECK(a.same_page == 0, "export same_page=%d", a.same_page);
    // 小規模は同ページに収まる。
    CHECK(jt_lut_export_desc(8, 32, 4, 32, &c) == JT_OK, "export small rc");
    CHECK(c.table_bytes == 128u && c.scale_bytes == 32u, "export small sizes");
    CHECK(c.page_bytes == 192u && c.same_page == 1, "export small page=%llu sp=%d",
          (unsigned long long)c.page_bytes, c.same_page);
    // 決定性 (同一入力→同一記述子)。
    {
        jt_lut_export_desc_t b2;
        CHECK(jt_lut_export_desc(64, 128, 4, 32, &b2) == JT_OK, "export rc2");
        CHECK(memcmp(&a, &b2, sizeof(a)) == 0, "export deterministic");
    }
    // 不正系。
    CHECK(jt_lut_export_desc(0, 128, 4, 32, &a) == JT_ERR_INVAL, "export rows=0");
    CHECK(jt_lut_export_desc(64, 128, 3, 32, &a) == JT_ERR_INVAL, "export bits=3");
    CHECK(jt_lut_export_desc(64, 128, 4, 0, &a) == JT_ERR_INVAL, "export block=0");
    CHECK(jt_lut_export_desc(64, 128, 4, 32, NULL) == JT_ERR_INVAL, "export NULL");
}

// テスト内参照import: exportバッファ→(q,scales)復元→fp32逆量子化。
// packing仕様はsrc/lut_aware.cの注記と同一 (LE、row-major)。
static int test_import_ref(const unsigned char *buf, const jt_lut_export_desc_t *d,
                           int8_t *q_out, float *s_out, float *dq_out) {
    size_t elems = (size_t)d->rows * (size_t)d->cols;
    size_t colblocks = ((size_t)d->cols + (size_t)d->block_len - 1u) / (size_t)d->block_len;
    size_t nscales = (size_t)d->rows * colblocks;
    if (!buf || !d || !q_out || !s_out || !dq_out) {
        return -1;
    }
    memcpy(s_out, buf + (size_t)d->scale_offset, (size_t)d->scale_bytes);
    if (d->bits == 8) {
        for (size_t i = 0; i < elems; i++) {
            q_out[i] = (int8_t)buf[i];
        }
    } else if (d->bits == 4) {
        for (size_t i = 0; i < elems; i++) {
            unsigned char b = buf[i / 2u];
            unsigned nib = ((i % 2u) == 0u) ? (b & 0xFu) : ((b >> 4) & 0xFu);
            int v = (int)nib;
            if (v >= 8) {
                v -= 16;
            }
            q_out[i] = (int8_t)v;
        }
    } else if (d->bits == 2) {
        for (size_t i = 0; i < elems; i++) {
            unsigned char b = buf[i / 4u];
            unsigned v = ((unsigned)b >> ((unsigned)(i % 4u) * 2u)) & 0x3u;
            int sv = (int)v;
            if (sv >= 2) {
                sv -= 4;
            }
            q_out[i] = (int8_t)sv;
        }
    } else {
        return -1;
    }
    for (size_t r = 0; r < d->rows; r++) {
        for (size_t c = 0; c < d->cols; c++) {
            size_t cb = c / (size_t)d->block_len;
            dq_out[r * d->cols + c] =
                (float)((double)q_out[r * d->cols + c] * (double)s_out[r * colblocks + cb]);
        }
    }
    (void)nscales;
    return 0;
}

static void test_export_binary_one(uint32_t rows, uint32_t cols, int bits, uint32_t block,
                                   unsigned salt) {
    jt_lut_export_desc_t d;
    size_t elems = (size_t)rows * (size_t)cols;
    size_t colblocks = ((size_t)cols + (size_t)block - 1u) / (size_t)block;
    size_t nscales = (size_t)rows * colblocks;
    float *W = NULL;
    float *dq_direct = NULL;
    float *dq_rt = NULL;
    float *sc_direct = NULL;
    float *sc_rt = NULL;
    int8_t *q_direct = NULL;
    int8_t *q_rt = NULL;
    unsigned char *buf = NULL;
    size_t wrote = 0;

    CHECK(jt_lut_export_desc(rows, cols, bits, block, &d) == JT_OK, "bin desc %ux%u b%d blk%u",
          rows, cols, bits, block);
    CHECK(d.table_offset == 0u, "bin toff");
    CHECK((d.scale_offset % JT_LUT_ALIGN) == 0u, "bin soff align=%llu",
          (unsigned long long)d.scale_offset);
    CHECK((d.page_bytes % JT_LUT_ALIGN) == 0u, "bin page align=%llu",
          (unsigned long long)d.page_bytes);
    CHECK(d.same_page == ((d.page_bytes <= JT_LUT_PAGE_SIZE) ? 1 : 0), "bin same_page");

    W = (float *)malloc(elems * sizeof(float));
    dq_direct = (float *)malloc(elems * sizeof(float));
    dq_rt = (float *)malloc(elems * sizeof(float));
    sc_direct = (float *)malloc(nscales * sizeof(float));
    sc_rt = (float *)malloc(nscales * sizeof(float));
    q_direct = (int8_t *)malloc(elems * sizeof(int8_t));
    q_rt = (int8_t *)malloc(elems * sizeof(int8_t));
    CHECK(W && dq_direct && dq_rt && sc_direct && sc_rt && q_direct && q_rt, "bin alloc");
    if (!W || !dq_direct || !dq_rt || !sc_direct || !sc_rt || !q_direct || !q_rt) {
        goto done;
    }
    for (size_t i = 0; i < elems; i++) {
        W[i] = test_val(i, 1.0f, salt);
    }
    // 直接の行単位量子化→逆量子化 (fp32等価値の期待値)。
    for (uint32_t r = 0; r < rows; r++) {
        CHECK(jt_mp_quantize(W + (size_t)r * cols, cols, bits, block,
                             q_direct + (size_t)r * cols,
                             sc_direct + (size_t)r * colblocks, colblocks) == JT_OK,
              "bin direct q r=%u", r);
    }
    for (uint32_t r = 0; r < rows; r++) {
        CHECK(jt_mp_dequantize(q_direct + (size_t)r * cols, sc_direct + (size_t)r * colblocks,
                               cols, bits, block, colblocks,
                               dq_direct + (size_t)r * cols) == JT_OK,
              "bin direct dq r=%u", r);
    }
    // export (64B整列バッファ。page_bytesは64の倍数のためaligned_alloc可)。
    buf = (unsigned char *)td_aligned_alloc(JT_LUT_ALIGN, (size_t)d.page_bytes);
    CHECK(buf != NULL, "bin buf alloc page=%llu", (unsigned long long)d.page_bytes);
    if (!buf) {
        goto done;
    }
    CHECK(((uintptr_t)buf % JT_LUT_ALIGN) == 0u, "bin buf aligned");
    memset(buf, 0xA5, (size_t)d.page_bytes);
    // ゼロ埋め前のゴミでパディング検査ができるよう、export前に0xA5で汚す。
    // export成功時はパディングがゼロ化されるはず。
    CHECK(jt_lut_export_binary(W, rows, cols, bits, block, buf, (size_t)d.page_bytes,
                               &wrote) == JT_OK,
          "bin export %ux%u b%d blk%u", rows, cols, bits, block);
    CHECK(wrote == (size_t)d.page_bytes, "bin wrote=%zu want=%llu", wrote,
          (unsigned long long)d.page_bytes);
    // パディングゼロ検査。
    for (size_t i = (size_t)d.table_bytes; i < (size_t)d.scale_offset; i++) {
        if (buf[i] != 0) {
            CHECK(0, "bin table pad[%zu]=%02x", i, buf[i]);
            break;
        }
    }
    for (size_t i = (size_t)d.scale_offset + (size_t)d.scale_bytes; i < (size_t)d.page_bytes;
         i++) {
        if (buf[i] != 0) {
            CHECK(0, "bin tail pad[%zu]=%02x", i, buf[i]);
            break;
        }
    }
    // import→fp32等価値の一致 (q/scales完全一致 + dq bit-exact)。
    CHECK(test_import_ref(buf, &d, q_rt, sc_rt, dq_rt) == 0, "bin import ref");
    CHECK(memcmp(q_direct, q_rt, elems) == 0, "bin q match %ux%u b%d", rows, cols, bits);
    CHECK(memcmp(sc_direct, sc_rt, nscales * sizeof(float)) == 0, "bin scales match");
    for (size_t i = 0; i < elems; i++) {
        if (dq_direct[i] != dq_rt[i]) {
            CHECK(0, "bin dq[%zu] direct=%a rt=%a", i, (double)dq_direct[i],
                  (double)dq_rt[i]);
            break;
        }
        if (!isfinite((double)dq_rt[i])) {
            CHECK(0, "bin dq non-finite");
            break;
        }
    }
    // out_written=NULLでも成功すること。
    CHECK(jt_lut_export_binary(W, rows, cols, bits, block, buf, (size_t)d.page_bytes,
                               NULL) == JT_OK,
          "bin null-written ok");
done:
    free(W);
    free(dq_direct);
    free(dq_rt);
    free(sc_direct);
    free(sc_rt);
    free(q_direct);
    free(q_rt);
    td_aligned_free(buf);
}

static void test_export_binary(void) {
    // ラウンドトリップ行列: 8/4/2bit × 整除/非整除 × same_page内外。
    test_export_binary_one(2, 8, 8, 4, 11u);
    test_export_binary_one(4, 8, 4, 4, 12u);
    test_export_binary_one(2, 7, 4, 4, 13u); // cols非整除・奇数table
    test_export_binary_one(4, 9, 2, 4, 14u); // 2bit端数
    test_export_binary_one(8, 32, 4, 32, 15u); // same_page=1
    test_export_binary_one(2, 2, 4, 2, 16u); // 極小
    // fail-closed: NULL/サイズ不足/不整列/不正dims/非有限はINVAL。
    {
        jt_lut_export_desc_t d;
        float W[4] = {0.1f, 0.2f, 0.3f, 0.4f};
        unsigned char *buf = NULL;
        size_t wrote = 0xDEADu;
        CHECK(jt_lut_export_desc(2, 2, 4, 2, &d) == JT_OK, "bin fail desc");
        buf = (unsigned char *)td_aligned_alloc(JT_LUT_ALIGN, (size_t)d.page_bytes);
        CHECK(buf != NULL, "bin fail alloc");
        if (buf) {
            memset(buf, 0xA5, (size_t)d.page_bytes);
            CHECK(jt_lut_export_binary(NULL, 2, 2, 4, 2, buf, (size_t)d.page_bytes,
                                       &wrote) == JT_ERR_INVAL,
                  "bin NULL W");
            CHECK(wrote == 0, "bin NULL W wrote=%zu", wrote);
            wrote = 0xDEADu;
            CHECK(jt_lut_export_binary(W, 2, 2, 4, 2, NULL, 0, &wrote) == JT_ERR_INVAL,
                  "bin NULL out");
            CHECK(wrote == 0, "bin NULL out wrote");
            wrote = 0;
            CHECK(jt_lut_export_binary(W, 2, 2, 4, 2, buf, (size_t)d.page_bytes - 1u,
                                       &wrote) == JT_ERR_INVAL,
                  "bin short cap");
            CHECK(wrote == 0, "bin short wrote");
            // 不整列 (align64+1)。
            {
                unsigned char *mis = buf + 1;
                errno = 0;
                wrote = 0;
                CHECK(jt_lut_export_binary(W, 2, 2, 4, 2, mis, (size_t)d.page_bytes,
                                           &wrote) == JT_ERR_INVAL,
                      "bin misalign");
                CHECK(errno == EINVAL, "bin misalign errno=%d", errno);
                CHECK(wrote == 0, "bin misalign wrote");
            }
            CHECK(jt_lut_export_binary(W, 0, 2, 4, 2, buf, (size_t)d.page_bytes,
                                       &wrote) == JT_ERR_INVAL,
                  "bin rows=0");
            CHECK(jt_lut_export_binary(W, 2, 2, 3, 2, buf, (size_t)d.page_bytes,
                                       &wrote) == JT_ERR_INVAL,
                  "bin bits=3");
            CHECK(jt_lut_export_binary(W, 2, 2, 4, 0, buf, (size_t)d.page_bytes,
                                       &wrote) == JT_ERR_INVAL,
                  "bin block=0");
            // 非有限Wは拒否しoutを汚さない (0xA5のまま)。
            {
                float Wn[4] = {0.1f, 0.2f, (float)NAN, 0.4f};
                memset(buf, 0xA5, (size_t)d.page_bytes);
                wrote = 0;
                CHECK(jt_lut_export_binary(Wn, 2, 2, 4, 2, buf, (size_t)d.page_bytes,
                                           &wrote) == JT_ERR_INVAL,
                      "bin nan W");
                CHECK(wrote == 0, "bin nan wrote");
                CHECK(buf[0] == 0xA5, "bin nan untouched=%02x", buf[0]);
            }
            td_aligned_free(buf);
        }
    }
}

int main(void) {
    test_policy();
    test_roundtrip();
    test_ste_smoke();
    test_export();
    test_export_binary();
    if (g_fail != 0) {
        fprintf(stderr, "mixed_lut: FAIL\n");
        return 1;
    }
    printf("mixed_lut: OK (policy/roundtrip/ste/export)\n");
    return 0;
}
