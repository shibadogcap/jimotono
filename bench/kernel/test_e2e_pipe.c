// test_e2e_pipe: 小規模E2E学習パイプラインのスモークテスト (Phase 2)。
//
// 目的 (ROADMAP Phase2完了条件「総1BクラスのMoEで学習パイプラインを検証」の
// 小規模代替): フル1Bは不可のため、2層・hidden=8相当・experts=4・top相当の
// 合成回帰タスクで forward→backward→optim更新→checkpoint再計算→offload往復
// のループが回り、損失が明確に減少すること (初期比50%以上減少) をassertする。
//
// ループ構成 (1ステップ = 全バッチ平均勾配で1更新):
// forward : gdn2_decode_step (S_prev=0の単一トークン) → jt_rmsnorm_fwd →
//             jt_swiglu_fwd → 線形ヘッド (pred = dot(y2,Wo)+bo)
//   loss    : MSE + StickyMoE損失の加算 (実MoEルーティングは使わない)
//   backward: 線形ヘッド手計算 → jt_swiglu_bwd → jt_rmsnorm_bwd →
//             jt_gdn2_decode_bwd (dW_gdnのみ更新に使用、他は勾配疎通確認)
//   update  : jt_optim8_step (全パラメータを1本のflatベクトルで管理)
//   ckpt    : 区間境界保存 (memcpy) + jt_ckpt_recompute_range実再計算
//             (jt_ckpt_layer_fn経由で保存境界からswiglu+headを再実行し一致確認)
//   esmoe   : N step毎にWdスライスを4 expertとしてregister→evict→prefetch→
//             load往復一致確認
//
// 合成タスク: 固定データ X[K][n] に対し t = dot(X, w_star) の線形ターゲット。
// 決定論的生成 (乱数なし) のため再現性あり。フルバッチ平均で安定減少する。
//
// 実行時間: 60 steps × 8 samples × 微小次元で0.1秒級。長時間学習にしないこと。
//
// TODO(1B級への外挿):
// - 次元: n=8/h=8/dk=4/dv=8 → n=2048級、層数2 → 数十層、expert=4 → 数千、
//   top-k実ルーティング (jt_routing_topk) + gateへの逆伝播を有効化する。
// - StickyMoE損失は現在スカラー加算のみ。1B級では系列平均1/(T-1)・全層平均・
//   L_CE・μL_bal合算を呼び出し側で行い (routing.hコメント準拠)、gate分布への
//   勾配をMoE重みに流す必要がある。
// - checkpointは現在2層2区間の形のみ。1B級では O(√n) 区間 (jt_ckpt_num_segments)
//   の境界のみ保存し、中間活性はbackward時に再計算する (実forward結合済みのため
//   層forward本体をjt_ckpt_layer_fnとして渡す形で拡張する)。
// - esmoeは現在Wd一部のみ。1B級では全expert重みをSSD backing + 行単位読み
//   (read_rows/Delta Prefetching) + io_uring/O_DIRECT非同期化で賄う。
// - 精度: 学習時はHGQ-LUT流儀でLUT-Denseを通常テンソル演算で学習し、推論時に
//   LUTへコンパイルする (AGENTS.MD 3.3)。本テストはfp32スカラー核の疎通が目的。
//
// 成功時 exit 0 + "e2e_pipe: OK"、失敗時 exit 1 + stderr。C11・errnoベース。
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/checkpoint.h"
#include "jimotono/common.h"
#include "jimotono/esmoe.h"
#include "jimotono/gdn2.h"
#include "jimotono/optim8.h"
#include "jimotono/routing.h"
#include "jimotono/train_bwd.h"

// ---- 小規模構成 (1B級の縮小代替) ----
#define EP_N 8       // model dim (= dv)
#define EP_H 8       // swiglu hidden
#define EP_DK 4      // gdn2 dk
#define EP_DV 8      // gdn2 dv (= EP_N)
#define EP_KDATA 8   // 合成データ数
#define EP_STEPS 60  // 学習ステップ (数十)
#define EP_ESMOE_EVERY 10
#define EP_NLAYERS 2
#define EP_EPS 1e-6f

// flat param layout
#define EP_OFF_RMS 0
#define EP_OFF_WG (EP_OFF_RMS + EP_N)
#define EP_OFF_WU (EP_OFF_WG + EP_H * EP_N)
#define EP_OFF_WD (EP_OFF_WU + EP_H * EP_N)
#define EP_OFF_WO (EP_OFF_WD + EP_H * EP_N)
#define EP_OFF_BO (EP_OFF_WO + EP_N)
#define EP_OFF_WGDN (EP_OFF_BO + 1)
#define EP_NP (EP_OFF_WGDN + EP_DV)

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
static float epat(int a, int b, int salt) {
    int v = (a * 31 + b * 17 + salt * 13) % 11;  // 0..10
    return (float)(v - 5) / 5.0f;
}

static float esilu(float z) {
    return z / (1.0f + expf(-z));
}

typedef struct {
    const float *Wg;
    const float *Wu;
    const float *Wd;
    float saved[EP_N];
    float y2[EP_N];
    int calls;
    int segs[4];
    int layers[4];
} eckpt_ctx_t;

// 新API経由の再計算層: seg0/layer0は境界保存の検証のみ、seg1/layer1は保存境界
// (y1) からswiglu+headを再実行しy2を出す。重みは呼び出し側所有 (ctx経由)。
static int eckpt_layer(int seg_idx, int layer, void *vctx) {
    eckpt_ctx_t *c = (eckpt_ctx_t *)vctx;
    if (c == NULL) {
        return JT_ERR_INVAL;
    }
    if (layer < 0 || layer >= EP_NLAYERS) {
        return JT_ERR_INVAL;
    }
    if (c->calls >= 0 && c->calls < 4) {
        c->segs[c->calls] = seg_idx;
        c->layers[c->calls] = layer;
    }
    c->calls++;
    if (layer == 0) {
        for (int j = 0; j < EP_N; j++) {
            if (!isfinite(c->saved[j])) {
                return JT_ERR_INVAL;
            }
        }
        return JT_OK;
    }
    if (c->Wg == NULL || c->Wu == NULL || c->Wd == NULL) {
        return JT_ERR_INVAL;
    }
    {
        float Gc2[EP_H], Uc2[EP_H];
        for (int i = 0; i < EP_H; i++) {
            double g = 0.0, u = 0.0;
            for (int j = 0; j < EP_N; j++) {
                g += (double)c->saved[j] * (double)c->Wg[i * EP_N + j];
                u += (double)c->saved[j] * (double)c->Wu[i * EP_N + j];
            }
            Gc2[i] = (float)g;
            Uc2[i] = (float)u;
        }
        for (int j = 0; j < EP_N; j++) {
            double acc = 0.0;
            for (int i = 0; i < EP_H; i++) {
                acc += (double)(esilu(Gc2[i]) * Uc2[i]) *
                       (double)c->Wd[i * EP_N + j];
            }
            c->y2[j] = (float)acc;
        }
    }
    return JT_OK;
}

int main(void) {
    // ---- 合成データ (固定) ----
    float X[EP_KDATA][EP_N];
    float T[EP_KDATA];
    float w_star[EP_N];
    for (int j = 0; j < EP_N; j++) {
        w_star[j] = 0.5f * epat(j, 0, 101);
    }
    for (int k = 0; k < EP_KDATA; k++) {
        double acc = 0.0;
        for (int j = 0; j < EP_N; j++) {
            X[k][j] = epat(k, j, 7);
            acc += (double)X[k][j] * (double)w_star[j];
        }
        T[k] = (float)acc;
    }

    // ---- gdn2固定入力 ----
    float gk[EP_DK], gb[EP_DK], gal[EP_DK];
    for (int i = 0; i < EP_DK; i++) {
        gk[i] = epat(i, 1, 12) - 0.23f;
        gb[i] = 0.5f + 0.1f * epat(i, 2, 13);
        gal[i] = 0.9f;
    }

    // ---- パラメータ初期化 ----
    float P[EP_NP];
    float G[EP_NP];
    for (int j = 0; j < EP_N; j++) {
        P[EP_OFF_RMS + j] = 1.0f;
        P[EP_OFF_WO + j] = 0.0f;
    }
    P[EP_OFF_BO] = 0.0f;
    for (int i = 0; i < EP_H; i++) {
        for (int j = 0; j < EP_N; j++) {
            P[EP_OFF_WG + i * EP_N + j] = 0.3f * epat(i, j, 23);
            P[EP_OFF_WU + i * EP_N + j] = 0.3f * epat(i, j, 24);
            P[EP_OFF_WD + i * EP_N + j] = 0.3f * epat(i, j, 25);
        }
    }
    for (int j = 0; j < EP_DV; j++) {
        P[EP_OFF_WGDN + j] = 0.6f;
    }

    // ---- optim8 ----
    jt_optim8_cfg_t cfg;
    jt_optim8_cfg_default(&cfg);
    cfg.lr = 1e-3f;
    jt_optim8_t opt;
    memset(&opt, 0, sizeof(opt));
    CHECK(jt_optim8_init(&opt, EP_NP, &cfg) == JT_OK, "optim init");
    if (g_fail) {
        return 1;
    }

    // ---- gdn2 scratch/state ----
    size_t fneed = 0, bneed = 0;
    CHECK(jt_gdn2_scratch_floats(EP_DK, EP_DV, &fneed) == JT_OK, "fscratch");
    CHECK(jt_gdn2_decode_bwd_scratch_floats(EP_DK, EP_DV, &bneed) == JT_OK,
          "bscratch");
    float *fsc = (float *)malloc((fneed ? fneed : 1) * sizeof(float));
    float *bsc = (float *)malloc((bneed ? bneed : 1) * sizeof(float));
    CHECK(fsc && bsc, "scratch malloc");
    float *S_work = NULL;
    float *S_zero = NULL;
    CHECK(jt_gdn2_state_alloc(&S_work, EP_DK, EP_DV) == JT_OK && S_work,
          "S_work alloc");
    CHECK(jt_gdn2_state_alloc(&S_zero, EP_DK, EP_DV) == JT_OK && S_zero,
          "S_zero alloc");
    if (!fsc || !bsc || !S_work || !S_zero) {
        free(fsc);
        free(bsc);
        jt_gdn2_state_free(S_work);
        jt_gdn2_state_free(S_zero);
        jt_optim8_fini(&opt);
        return 1;
    }
    CHECK(jt_gdn2_state_reset(S_zero, EP_DK, EP_DV) == JT_OK, "S_zero reset");

    // ---- checkpoint計画 (2層) ----
    int n_seg = 0;
    CHECK(jt_ckpt_num_segments(EP_NLAYERS, &n_seg) == JT_OK, "ckpt nseg");
    int bounds[8] = {0};
    CHECK(jt_ckpt_boundaries(EP_NLAYERS, n_seg, bounds,
                             (size_t)(n_seg + 1)) == JT_OK,
          "ckpt bounds n_seg=%d", n_seg);
    jt_ckpt_plan_t plan = {EP_NLAYERS, n_seg, bounds};
    CHECK(jt_ckpt_plan_init(&plan) == JT_OK, "ckpt plan");
    // 境界保存バッファ (layer0出力 y1相当 [N])。
    float ckpt_saved[EP_N] = {0.0f};
    int ckpt_calls = 0;

    // ---- esmoe (Wd=64 floats=256Bを4 expert×64Bに分割) ----
    jt_esmoe_cfg_t ecfg;
    ecfg.n_experts = 4;
    ecfg.expert_bytes = 64;
    ecfg.row_bytes = 16;
    jt_esmoe_t es;
    memset(&es, 0, sizeof(es));
    CHECK(jt_esmoe_init(&es, &ecfg) == JT_OK, "esmoe init");
    int esmoe_checks = 0;

    // ---- routing用ダミーgate (偶奇で切替→soft損失が0/小を交互) ----
    float gate_even[4] = {0.25f, 0.25f, 0.25f, 0.25f};
    float gate_odd[4] = {0.30f, 0.20f, 0.25f, 0.25f};
    float gate_prev[4];
    memcpy(gate_prev, gate_even, sizeof(gate_prev));

    // forwardヘルパー (1サンプル分。y2/G/Uも返す)。
    // NOTE: Pスライスを読むためマクロではなくインライン展開で記述。
    float loss_hist[EP_STEPS + 1];
    float init_loss = 0.0f;

    for (int step = 0; step <= EP_STEPS; step++) {
        double loss_sum = 0.0;
        // grad累積ゼロ化 (step==EP_STEPSは評価のみで更新しない)。
        for (int i = 0; i < EP_NP; i++) {
            G[i] = 0.0f;
        }
        // 先頭サンプルの再計算照合用。
        float y2_ref[EP_N] = {0.0f};
        float y1_saved0[EP_N] = {0.0f};
        int y1_has0 = 0;

        for (int k = 0; k < EP_KDATA; k++) {
            const float *Wr = &P[EP_OFF_RMS];
            const float *Wg = &P[EP_OFF_WG];
            const float *Wu = &P[EP_OFF_WU];
            const float *Wd = &P[EP_OFF_WD];
            const float *Wo = &P[EP_OFF_WO];
            float bo = P[EP_OFF_BO];
            const float *wgdn = &P[EP_OFF_WGDN];

            float q[EP_DK], v[EP_DV];
            for (int i = 0; i < EP_DK; i++) {
                q[i] = X[k][i];
            }
            for (int j = 0; j < EP_DV; j++) {
                v[j] = X[k][j];
            }
            // gdn2 forward (S_prev=0)。
            CHECK(jt_gdn2_state_reset(S_work, EP_DK, EP_DV) == JT_OK,
                  "S reset s=%d k=%d", step, k);
            float o[EP_DV];
            int rc = jt_gdn2_decode_step(S_work, o, q, gk, v, gb, wgdn,
                                         gal, EP_DK, EP_DV, fsc, fneed);
            CHECK(rc == JT_OK, "gdn2 fwd rc=%d s=%d k=%d", rc, step, k);
            if (rc != JT_OK) {
                goto fail;
            }
            // rmsnorm forward (ライブラリ核)。
            float y1[EP_N];
            rc = jt_rmsnorm_fwd(o, Wr, y1, EP_N, EP_EPS);
            CHECK(rc == JT_OK, "rms fwd rc=%d s=%d k=%d", rc, step, k);
            if (rc != JT_OK) {
                goto fail;
            }
            // swiglu forward (ライブラリ核。G/Uはbwd用cached中間値)。
            float Gc[EP_H], Uc[EP_H];
            float y2[EP_N];
            rc = jt_swiglu_fwd(y1, Wg, Wu, Wd, Gc, Uc, y2, EP_N, EP_H);
            CHECK(rc == JT_OK, "swiglu fwd rc=%d s=%d k=%d", rc, step,
                  k);
            if (rc != JT_OK) {
                goto fail;
            }
            double pred = (double)bo;
            for (int j = 0; j < EP_N; j++) {
                pred += (double)y2[j] * (double)Wo[j];
            }
            double diff = pred - (double)T[k];
            loss_sum += diff * diff;

            if (k == 0) {
                memcpy(y2_ref, y2, sizeof(y2_ref));
                memcpy(y1_saved0, y1, sizeof(y1_saved0));
                y1_has0 = 1;
            }

            if (step < EP_STEPS) {
                float d = (float)(2.0 * diff / (double)EP_KDATA);
                // head grad。
                float dy2[EP_N];
                for (int j = 0; j < EP_N; j++) {
                    G[EP_OFF_WO + j] += d * y2[j];
                    dy2[j] = d * Wo[j];
                }
                G[EP_OFF_BO] += d;
                // swiglu bwd。
                float dy1[EP_N], dWg[EP_H * EP_N], dWu[EP_H * EP_N],
                    dWd[EP_H * EP_N];
                rc = jt_swiglu_bwd(dy2, y1, Gc, Uc, Wd, Wg, Wu, dy1,
                                   dWg, dWu, dWd, EP_N, EP_H);
                CHECK(rc == JT_OK, "swiglu bwd rc=%d s=%d k=%d", rc,
                      step, k);
                if (rc != JT_OK) {
                    goto fail;
                }
                for (int i = 0; i < EP_H * EP_N; i++) {
                    G[EP_OFF_WG + i] += dWg[i];
                    G[EP_OFF_WU + i] += dWu[i];
                    G[EP_OFF_WD + i] += dWd[i];
                }
                // rmsnorm bwd。
                float dO[EP_DV], dWr[EP_N];
                rc = jt_rmsnorm_bwd(dy1, o, Wr, dO, dWr, EP_N, EP_EPS);
                CHECK(rc == JT_OK, "rms bwd rc=%d s=%d k=%d", rc, step,
                      k);
                if (rc != JT_OK) {
                    goto fail;
                }
                for (int j = 0; j < EP_N; j++) {
                    G[EP_OFF_RMS + j] += dWr[j];
                }
                // gdn2 bwd (dS_next=0の単一ステップ)。
                float dSn[EP_DK * EP_DV] = {0.0f};
                float dQ[EP_DK], dK[EP_DK], dV[EP_DV], dB[EP_DK],
                    dWgdn[EP_DV], dA[EP_DK], dSp[EP_DK * EP_DV];
                rc = jt_gdn2_decode_bwd(
                    S_zero, q, gk, v, gb, wgdn, gal, dO, dSn, dQ,
                    dK, dV, dB, dWgdn, dA, dSp, EP_DK, EP_DV, bsc,
                    bneed);
                CHECK(rc == JT_OK, "gdn2 bwd rc=%d s=%d k=%d", rc,
                      step, k);
                if (rc != JT_OK) {
                    goto fail;
                }
                for (int j = 0; j < EP_DV; j++) {
                    G[EP_OFF_WGDN + j] += dWgdn[j];
                }
            }
        }

        float mse = (float)(loss_sum / (double)EP_KDATA);
        // routing損失を加算 (実ルーティングなし・勾配なし)。
        const float *gcur = (step % 2 == 0) ? gate_even : gate_odd;
        float rloss = 0.0f;
        int rrc = jt_routing_sticky_loss(gcur, (step == 0) ? NULL : gate_prev,
                                         NULL, 4, 0.05f, 0.0f,
                                         (size_t)(step % 4), 4, &rloss);
        CHECK(rrc == JT_OK, "routing loss rc=%d s=%d", rrc, step);
        memcpy(gate_prev, gcur, sizeof(gate_prev));
        float total = mse + rloss;
        loss_hist[step] = total;
        if (step == 0) {
            init_loss = total;
        }

        // ---- checkpoint: 境界保存 + 新API経由の再計算照合 ----
        // 先頭サンプルy1を境界として保存し、jt_ckpt_recompute_range経由で
        // seg1 (swiglu+head) を再実行してy2一致を確認する。seg0は境界検証のみ。
        if (y1_has0) {
            memcpy(ckpt_saved, y1_saved0, sizeof(ckpt_saved));
            // 手動再計算: 保存境界からswiglu+headを再実行しy2一致を確認
            // (ライブラリforward核を使用)。
            const float *Wg = &P[EP_OFF_WG];
            const float *Wu = &P[EP_OFF_WU];
            const float *Wd = &P[EP_OFF_WD];
            float Gc2[EP_H], Uc2[EP_H];
            float y2b[EP_N];
            int frc =
                jt_swiglu_fwd(ckpt_saved, Wg, Wu, Wd, Gc2, Uc2, y2b,
                              EP_N, EP_H);
            CHECK(frc == JT_OK, "ckpt fwd recompute rc=%d s=%d", frc,
                  step);
            eckpt_ctx_t ectx;
            float worst = 0.0f;
            int seg = step % n_seg;
            memset(&ectx, 0, sizeof(ectx));
            memcpy(ckpt_saved, y1_saved0, sizeof(ckpt_saved));
            memcpy(ectx.saved, ckpt_saved, sizeof(ectx.saved));
            ectx.Wg = &P[EP_OFF_WG];
            ectx.Wu = &P[EP_OFF_WU];
            ectx.Wd = &P[EP_OFF_WD];
            // 当該区間の再実行 (境界→再計算→一致)。
            if (seg == 0) {
                CHECK(jt_ckpt_recompute_range(&plan, 0, eckpt_layer,
                                              &ectx) == JT_OK,
                      "ckpt recomp seg0 rc s=%d", step);
                CHECK(ectx.calls == 1 && ectx.layers[0] == 0,
                      "ckpt seg0 trace s=%d", step);
            } else {
                CHECK(jt_ckpt_recompute_range(&plan, 1, eckpt_layer,
                                              &ectx) == JT_OK,
                      "ckpt recomp seg1 rc s=%d", step);
                CHECK(ectx.calls == 1 && ectx.layers[0] == 1,
                      "ckpt seg1 trace s=%d", step);
                for (int j = 0; j < EP_N; j++) {
                    float e = fabsf(ectx.y2[j] - y2_ref[j]);
                    worst = (e > worst) ? e : worst;
                }
                CHECK(worst < 1e-5f,
                      "ckpt recompute mismatch worst=%f s=%d", worst,
                      step);
            }
            if (step == 0) {
                // 不正区間はINVAL (正常系JT_OKと区別可能)。
                CHECK(jt_ckpt_recompute_range(&plan, n_seg, eckpt_layer,
                                              &ectx) == JT_ERR_INVAL,
                      "ckpt bad seg s=%d", step);
            }
            ckpt_calls++;
        }

        // ---- esmoe: N step毎にWdスライスのoffload往復 ----
        if (step % EP_ESMOE_EVERY == 0) {
            const unsigned char *wdb =
                (const unsigned char *)&P[EP_OFF_WD];
            for (uint32_t e = 0; e < 4; e++) {
                CHECK(jt_esmoe_register(&es, e, wdb + e * 64, 64) == JT_OK,
                      "esmoe reg e=%u s=%d", e, step);
            }
            CHECK(jt_esmoe_evict(&es, 1) == JT_OK, "esmoe evict s=%d",
                  step);
            {
                uint32_t ids[2] = {1, 3};
                CHECK(jt_esmoe_prefetch(&es, ids, 2) == JT_OK,
                      "esmoe prefetch s=%d", step);
            }
            for (uint32_t e = 0; e < 4; e++) {
                unsigned char out[64] = {0};
                CHECK(jt_esmoe_load(&es, e, out, 64) == JT_OK,
                      "esmoe load e=%u s=%d", e, step);
                CHECK(memcmp(out, wdb + e * 64, 64) == 0,
                      "esmoe roundtrip e=%u s=%d", e, step);
            }
            esmoe_checks++;
        }

        if (step < EP_STEPS) {
            // 非有限ガード (fail-closed: 更新前に検出)。
            for (int i = 0; i < EP_NP; i++) {
                if (!isfinite(G[i])) {
                    CHECK(0, "non-finite grad G[%d]=%f s=%d", i, G[i],
                          step);
                    goto fail;
                }
            }
            CHECK(jt_optim8_step(&opt, P, G, EP_NP) == JT_OK,
                  "optim step s=%d", step);
            if (g_fail) {
                goto fail;
            }
        }
        if (step % 10 == 0 || step == EP_STEPS) {
            printf("e2e_pipe: step=%d loss=%.6f (mse=%.6f rout=%.6f)\n",
                   step, total, mse, rloss);
        }
    }

    {
        float final_loss = loss_hist[EP_STEPS];
        printf("e2e_pipe: init=%.6f final=%.6f ratio=%.3f ckpt_calls=%d "
               "esmoe_checks=%d\n",
               init_loss, final_loss,
               (init_loss > 0.0f) ? (double)final_loss / (double)init_loss
                                  : -1.0,
               ckpt_calls, esmoe_checks);
        CHECK(isfinite(init_loss) && isfinite(final_loss), "loss finite");
        CHECK(init_loss > 0.0f, "init_loss=%f want >0", init_loss);
        CHECK(final_loss < init_loss * 0.5f,
              "loss not halved: final=%f init=%f", final_loss, init_loss);
        CHECK(ckpt_calls == EP_STEPS + 1, "ckpt_calls=%d", ckpt_calls);
        CHECK(esmoe_checks == EP_STEPS / EP_ESMOE_EVERY + 1,
              "esmoe_checks=%d", esmoe_checks);
    }

fail:
    jt_esmoe_fini(&es);
    jt_gdn2_state_free(S_work);
    jt_gdn2_state_free(S_zero);
    free(fsc);
    free(bsc);
    jt_optim8_fini(&opt);
    if (g_fail != 0) {
        fprintf(stderr, "e2e_pipe: FAIL\n");
        return 1;
    }
    printf("e2e_pipe: OK (2layer tiny-MoE-alt, %d steps, loss halved)\n",
           EP_STEPS);
    return 0;
}
