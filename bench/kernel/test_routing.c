// test: MoEルーティングの正しさ検証 (Phase 1)。
// - topk正しさ: 既知logitsで期待ID一致 + softmax合計1 + tie-break確定性
// - sticky loss: 同一gate連続でloss≈0、切替で>0、窓先頭でhard=0
// 成功時 exit 0、失敗時 exit 1 + stderr。
// 実ベンチ化は任意のため本ファイルはtest専用 (bench harnessはSTUB扱い):
//   cases: n={8,512,4096} k={1,8,16} metric=us/call (Phase 1では未計測)
#include <math.h>
#include <stdio.h>

#include "jimotono/routing.h"

static int fails = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            fails++;                                                  \
        }                                                             \
    } while (0)

static int feq(float a, float b, float tol) {
    return fabsf(a - b) <= tol;
}

int main(void) {
    // ---- topk: 既知logits ----
    {
        float logits[5] = {0.1f, 2.0f, 0.5f, 3.0f, 1.0f};
        size_t ids[5] = {0};
        float w[5] = {0.0f};
        CHECK(jt_routing_topk(logits, 5, 2, ids, w) == JT_OK, "topk k=2 rc");
        CHECK(ids[0] == 3 && ids[1] == 1, "topk k=2 ids {3,1}");
        // softmax([3,2]) = [0.7310586, 0.2689414]
        CHECK(feq(w[0], 0.7310586f, 1e-5f), "topk k=2 w0");
        CHECK(feq(w[1], 0.2689414f, 1e-5f), "topk k=2 w1");
        CHECK(feq(w[0] + w[1], 1.0f, 1e-6f), "topk k=2 sum==1");
    }
    // ---- topk: k=1 ----
    {
        float logits[5] = {0.1f, 2.0f, 0.5f, 3.0f, 1.0f};
        size_t ids[1] = {0};
        float w[1] = {0.0f};
        CHECK(jt_routing_topk(logits, 5, 1, ids, w) == JT_OK, "topk k=1 rc");
        CHECK(ids[0] == 3, "topk k=1 id==3");
        CHECK(feq(w[0], 1.0f, 1e-6f), "topk k=1 w==1");
    }
    // ---- topk: k=n (全選択・降順) ----
    {
        float logits[5] = {0.1f, 2.0f, 0.5f, 3.0f, 1.0f};
        size_t ids[5] = {0};
        float w[5] = {0.0f};
        CHECK(jt_routing_topk(logits, 5, 5, ids, w) == JT_OK, "topk k=n rc");
        CHECK(ids[0] == 3 && ids[1] == 1 && ids[2] == 4 && ids[3] == 2 && ids[4] == 0,
              "topk k=n order {3,1,4,2,0}");
        float s = w[0] + w[1] + w[2] + w[3] + w[4];
        CHECK(feq(s, 1.0f, 1e-6f), "topk k=n sum==1");
    }
    // ---- topk: tieは小さいindex優先 ----
    {
        float logits[3] = {1.0f, 1.0f, 1.0f};
        size_t ids[2] = {0};
        float w[2] = {0.0f};
        CHECK(jt_routing_topk(logits, 3, 2, ids, w) == JT_OK, "topk tie rc");
        CHECK(ids[0] == 0 && ids[1] == 1, "topk tie ids {0,1}");
        CHECK(feq(w[0], 0.5f, 1e-6f) && feq(w[1], 0.5f, 1e-6f), "topk tie w==0.5");
    }
    // ---- topk: 不正入力 ----
    {
        float logits[2] = {0.0f, 1.0f};
        size_t ids[2] = {0};
        float w[2] = {0.0f};
        CHECK(jt_routing_topk(NULL, 2, 1, ids, w) == JT_ERR_INVAL, "topk NULL logits");
        CHECK(jt_routing_topk(logits, 2, 1, NULL, w) == JT_ERR_INVAL, "topk NULL ids");
        CHECK(jt_routing_topk(logits, 2, 1, ids, NULL) == JT_ERR_INVAL, "topk NULL w");
        CHECK(jt_routing_topk(logits, 0, 1, ids, w) == JT_ERR_INVAL, "topk n==0");
        CHECK(jt_routing_topk(logits, 2, 0, ids, w) == JT_ERR_INVAL, "topk k==0");
        CHECK(jt_routing_topk(logits, 2, 3, ids, w) == JT_ERR_INVAL, "topk k>n");
    }
    // ---- sticky: 同一gate連続でloss≈0 (restrictのため別バッファ) ----
    {
        float gt[4] = {0.5f, 0.3f, 0.1f, 0.1f};
        float gp[4] = {0.5f, 0.3f, 0.1f, 0.1f};
        float ga[4] = {0.5f, 0.3f, 0.1f, 0.1f};
        float loss = -1.0f;
        CHECK(jt_routing_sticky_loss(gt, gp, ga, 4, 0.1f, 0.5f, 2, 4, &loss) == JT_OK,
              "sticky same rc");
        CHECK(fabsf(loss) < 1e-7f, "sticky same loss≈0");
    }
    // ---- sticky: 切替で>0 (soft項のみ) ----
    {
        float cur[4] = {0.7f, 0.1f, 0.1f, 0.1f};
        float prev[4] = {0.1f, 0.7f, 0.1f, 0.1f};
        float loss = 0.0f;
        // L_cons = 0.36+0.36 = 0.72, λ=0.1 → 0.072
        CHECK(jt_routing_sticky_loss(cur, prev, NULL, 4, 0.1f, 0.0f, 1, 4, &loss) == JT_OK,
              "sticky switch rc");
        CHECK(loss > 0.0f, "sticky switch loss>0");
        CHECK(feq(loss, 0.072f, 1e-6f), "sticky switch loss==0.072");
    }
    // ---- sticky: hard項 (窓末ランプ) ----
    {
        float cur[2] = {1.0f, 0.0f};
        float anchor[2] = {0.0f, 1.0f};
        float loss = 0.0f;
        // ||diff||^2=2, ramp=3/4 → α=1.0で1.5
        CHECK(jt_routing_sticky_loss(cur, NULL, anchor, 2, 0.0f, 1.0f, 3, 4, &loss) == JT_OK,
              "sticky hard rc");
        CHECK(feq(loss, 1.5f, 1e-6f), "sticky hard loss==1.5");
    }
    // ---- sticky: 窓先頭 (t_rel==0) ではhard=0 ----
    {
        float cur[2] = {1.0f, 0.0f};
        float anchor[2] = {0.0f, 1.0f};
        float loss = -1.0f;
        CHECK(jt_routing_sticky_loss(cur, NULL, anchor, 2, 0.1f, 1.0f, 0, 4, &loss) == JT_OK,
              "sticky head rc");
        CHECK(fabsf(loss) < 1e-7f, "sticky head loss≈0");
    }
    // ---- sticky: 不正入力 (restrictのため別バッファ) ----
    {
        float gt[2] = {0.5f, 0.5f};
        float gp[2] = {0.5f, 0.5f};
        float ga[2] = {0.5f, 0.5f};
        float loss = 0.0f;
        CHECK(jt_routing_sticky_loss(NULL, gp, ga, 2, 0.1f, 0.5f, 1, 4, &loss) == JT_ERR_INVAL,
              "sticky NULL g_t");
        CHECK(jt_routing_sticky_loss(gt, gp, ga, 2, 0.1f, 0.5f, 1, 4, NULL) == JT_ERR_INVAL,
              "sticky NULL out");
        CHECK(jt_routing_sticky_loss(gt, gp, ga, 0, 0.1f, 0.5f, 1, 4, &loss) == JT_ERR_INVAL,
              "sticky n==0");
        CHECK(jt_routing_sticky_loss(gt, gp, ga, 2, -0.1f, 0.5f, 1, 4, &loss) == JT_ERR_INVAL,
              "sticky lambda<0");
        CHECK(jt_routing_sticky_loss(gt, gp, ga, 2, 0.1f, -0.5f, 1, 4, &loss) == JT_ERR_INVAL,
              "sticky alpha<0");
        CHECK(jt_routing_sticky_loss(gt, gp, ga, 2, 0.1f, 0.5f, 1, 0, &loss) == JT_ERR_INVAL,
              "sticky w==0");
        CHECK(jt_routing_sticky_loss(gt, gp, ga, 2, 0.1f, 0.5f, 4, 4, &loss) == JT_ERR_INVAL,
              "sticky t_rel>=w");
    }

    // ---- topk: NaN拒否 / -INFマスク許容 / +INF拒否 ----
    {
        float nanl[4] = {1.0f, NAN, 0.5f, 0.25f};
        size_t nids[2] = {0};
        float nw[2] = {0.0f};
        CHECK(jt_routing_topk(nanl, 4, 2, nids, nw) == JT_ERR_INVAL, "topk NaN");
        float mask[4] = {2.0f, 1.0f, -INFINITY, -INFINITY};
        CHECK(jt_routing_topk(mask, 4, 2, nids, nw) == JT_OK, "topk -INF mask rc");
        CHECK(nids[0] == 0 && nids[1] == 1, "topk -INF mask ids");
        CHECK(feq(nw[0] + nw[1], 1.0f, 1e-6f), "topk -INF mask sum==1");
        float infl[3] = {INFINITY, 1.0f, 0.0f};
        CHECK(jt_routing_topk(infl, 3, 2, nids, nw) == JT_ERR_INVAL, "topk +INF");
    }
    // ---- topk: 大規模 (n=512,k=8) 降順+合計1 ----
    {
        float bigl[512];
        size_t bigids[8] = {0};
        float bigw[8] = {0.0f};
        for (size_t i = 0; i < 512; i++) {
            bigl[i] = (float)(i % 64);
        }
        CHECK(jt_routing_topk(bigl, 512, 8, bigids, bigw) == JT_OK, "topk 512x8 rc");
        // 値63のindex {63,127,...,511} が小index優先で並ぶ
        CHECK(bigids[0] == 63 && bigids[7] == 511, "topk 512x8 ids");
        float s = 0.0f;
        for (size_t p = 0; p < 8; p++) {
            s += bigw[p];
        }
        CHECK(feq(s, 1.0f, 1e-5f), "topk 512x8 sum==1");
        CHECK(feq(bigw[0], 0.125f, 1e-6f), "topk 512x8 uniform w");
    }
    // ---- topk: 大規模 (n=4096,k=16) ----
    {
        float hugel[4096];
        size_t hugeids[16] = {0};
        float hugew[16] = {0.0f};
        for (size_t i = 0; i < 4096; i++) {
            hugel[i] = (float)(i % 256);
        }
        CHECK(jt_routing_topk(hugel, 4096, 16, hugeids, hugew) == JT_OK,
              "topk 4096x16 rc");
        CHECK(hugeids[0] == 255 && hugeids[15] == 4095, "topk 4096x16 ids");
        double s = 0.0;
        for (size_t p = 0; p < 16; p++) {
            s += hugew[p];
        }
        CHECK(fabs(s - 1.0) < 1e-5, "topk 4096x16 sum==1");
    }
    // ---- sticky: 非有限hyperparam/gate拒否 ----
    {
        float sgt[2] = {0.5f, 0.5f};
        float sgp[2] = {0.5f, 0.5f};
        float sga[2] = {0.5f, 0.5f};
        float loss = 0.0f;
        CHECK(jt_routing_sticky_loss(sgt, sgp, sga, 2, INFINITY, 0.5f, 1, 4, &loss) ==
                  JT_ERR_INVAL,
              "sticky lambda INF");
        CHECK(jt_routing_sticky_loss(sgt, sgp, sga, 2, 0.1f, INFINITY, 1, 4, &loss) ==
                  JT_ERR_INVAL,
              "sticky alpha INF");
        float nan[2] = {NAN, 0.5f};
        CHECK(jt_routing_sticky_loss(nan, sgp, sga, 2, 0.1f, 0.5f, 1, 4, &loss) ==
                  JT_ERR_INVAL,
              "sticky NaN gate");
        float inf[2] = {INFINITY, 0.0f};
        CHECK(jt_routing_sticky_loss(inf, sgp, sga, 2, 0.1f, 0.5f, 1, 4, &loss) ==
                  JT_ERR_INVAL,
              "sticky INF gate");
    }

    if (fails == 0) {
        printf("routing: OK\n");
        return 0;
    }
    fprintf(stderr, "routing: %d FAILs\n", fails);
    return 1;
}
