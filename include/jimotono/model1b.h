#pragma once
#ifndef JIMOTONO_MODEL1B_H
#define JIMOTONO_MODEL1B_H
// JIMOTONO 1B級MoE構成の定義 + パラメータ会計/メモリ予算の純粋関数 (Phase 2)。
// 背景: P2最終レビューMAJOR-3 (1B級未検証)。ARCHITECTURE.MD §2 /
//   AGENTS.MD §2.2混合精度 / §4学習戦略に準拠し、総10〜20B目標の
//   スケールダウン検証構成 (1B級) をコードとして固定する。
//
// ---- 既定構成 (jt_model1b_default) ----
//   layers=24, d_model=1024, experts/layer=64, expert_hidden=128,
//   shared=2, top-k=4, vocab=48588 (llm-jp-tokenizer v2.2),
//   GDN-2 blocks (linear_every=6 → 20線形 + 4フル, 線形:フル=5:1。
//   AGENTS.MD §2.1の4:1〜7:1帯内), tied embeddings。
//   task例の hidden=256→128 縮小は意図的 (d_arch概算許容の範囲):
//   top-k=4/shared=2を維持したまま hidden=256 (expert 0.79M) では
//   active_body が約140Mとなり100M制約を超過するため。
//
// ---- パラメータ計算式 (fp換算・量子化前) ----
//   expert      = 3*d*h            (SwiGLU gate/up/down。各d*h)
//   routing     = L*E*expert
//   shared      = L*S*expert       (常時発火・dense常駐)
//   attn_lin    = d*d + d          (GDN-2線形層概算: fused q/out射影 d*d +
//                                  ゲート/正規化 d。KV射影は固定長ランニング
//                                  ステートへの畳み込みで不要とみなす概算。
//                                  ARCHITECTURE.MD §1「KVキャッシュ不要」)
//   attn_full   = 4*d*d + d        (フル注意層: Q,K,V,O + ゲート)
//   n_full      = L / linear_every (切り捨て。24/6=4), n_lin = L - n_full
//   attn        = n_lin*attn_lin + n_full*attn_full
//   emb         = vocab*d          (tied: totalに1回のみ計上)
//   router      = L*d*E            (層毎ゲート d→E)
//   norms       = L*2*d            (RMSNorm×2/層: attention前・MoE前)
//   total       = routing+shared+attn+emb+router+norms
//   active_body = attn + shared + L*K*expert + router + norms
//     (Transformer bodyの発火分。tied emb/headの vocab*d は全トークン共通の
//      定数項としてtotalに計上し、active比較(成功基準アクティブ100M)からは
//      除外する。DeepSeek式のactive表記と同方針)
//   既定値での検算:
//     expert=393,216 / routing=603,979,776 / shared=18,874,368 /
//     attn=37,773,312 (20*1,049,600+4*4,195,328) / emb=49,754,112 /
//     router=1,572,864 / norms=49,152 /
//     total=712,003,584 (約0.71B。0.5B〜2B帯内=1B級) /
//     active_body=96,018,432 (約96M <100M)
//
// ---- 1B到達の拡張式 (コメント設計) ----
//   E=64固定では total/active比 ≈ E/(S+K) = 10.7 のため active<100Mのまま
//   1B超には attn を削る余地がほぼない。1B超が必要なら E=96 (総expert 2304,
//   ARCH §2の2000〜4000帯) とし total ≈ 0.80B級へ (activeはEにほぼ不変)。
//
// ---- 混合精度バイト会計 (mixed_precポリシー参照) ----
//   ビット幅は jt_mp_bits_for() から取得する (AGENTS.MD §2.2):
//   shared=INT8 / routing down=INT4 / gate,up=INT2 / attn=INT8 / emb=INT8。
//   weight_bytes = params*bits/8、scale_bytes = ceil(params/32)*4
//   (per-block fp32スケール。JT_MP_DEFAULT_BLOCK=32)。
//   dense常駐 = shared+attn+emb+router+norms (router/normsはfp32換算=4B扱い。
//   小規模のため量子化対象外と明示)。SSD = routing experts (gate/up/down別)。
//
// ---- 学習時メモリ予算 (AGENTS.MD §4.1) ----
//   前提: 勾配チェックポインティング (jt_ckpt_*再利用。O(√n)) +
//   8bitオプティマイザ (2.125B/param ≈ 1/3.8。optim8.h注記) +
//   ES-MoEオフロード (offload_routing=1でrouting expertの重みfp32 masterと
//   optim状態をSSDに置き、RAMにはdense常駐分のみ)。
//   peak = weights_ram + grads_ram + optim_ram + act_peak + prefetch + misc。
//   16GBに収まることをテストがassert (タスク要件3)。
//   N100実機11GBの可否はテスト側でコメント判定 (要件3)。
//
// 規約: C11, restrict, errnoベース (AGENTS.MD 7.1)。mallocなし純粋関数。
//   積オーバーフローは JT_ERR_NOMEM。リトルエンディアン前提 (common.h)。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// 既定構成値 (上記コメントの検算と一致すること。テストが固定値検証する)。
#define JT_MODEL1B_LAYERS 24
#define JT_MODEL1B_D_MODEL 1024
#define JT_MODEL1B_EXPERTS 64
#define JT_MODEL1B_EXPERT_HIDDEN 128
#define JT_MODEL1B_SHARED 2
#define JT_MODEL1B_TOP_K 4
#define JT_MODEL1B_VOCAB 48588
#define JT_MODEL1B_LINEAR_EVERY 6

// 量子化ブロック長・スケール幅 (mixed_precと整合)。
#define JT_MODEL1B_QBLOCK 32
#define JT_MODEL1B_SCALE_BYTES 4

// 8bitオプティマイザ状態 2.125B/param = 17/8 (optim8.h注記: 1/3.8)。
#define JT_MODEL1B_OPTIM_NUM 17
#define JT_MODEL1B_OPTIM_DEN 8

// 学習時misc固定分 (dataloader+workspace概算)。
#define JT_MODEL1B_MISC_BYTES (134217728ULL) // 128MiB

typedef struct jt_model1b_config {
    int32_t n_layers;      // >0 (既定24)
    int32_t d_model;       // >0 (既定1024)
    int32_t n_experts;     // >0 /層 (既定64)
    int32_t expert_hidden; // >0 (既定128)
    int32_t n_shared;      // >=0 (既定2)
    int32_t top_k;         // 1..n_experts (既定4)
    int32_t vocab;         // >0 (既定48588)
    int32_t linear_every;  // >=1 (既定6。n_full=floor(L/every))
} jt_model1b_config_t;

// 既定構成を格納 (NULLはINVAL)。
int jt_model1b_default(jt_model1b_config_t *restrict out);

// 構成検査 (範囲外はINVAL, errno=EINVAL)。
int jt_model1b_validate(const jt_model1b_config_t *restrict cfg);

// パラメータ会計 (fp換算カウント。上記計算式)。
typedef struct jt_model1b_counts {
    uint64_t expert_params;   // 1 expertあたり (3*d*h)
    uint64_t routing_total;   // L*E*expert (SSDオフロード対象)
    uint64_t shared_total;    // L*S*expert (dense常駐)
    uint64_t attn_total;      // n_lin*lin + n_full*full
    uint64_t emb_total;       // vocab*d (tied, 1回計上)
    uint64_t router_total;    // L*d*E
    uint64_t norm_total;      // L*2*d
    uint64_t total;           // 上記合計
    uint64_t active_body;     // attn+shared+L*K*expert+router+norms
    uint64_t active_moe;      // shared + L*K*expert (内数・参考)
    int32_t n_full;           // フル注意層数
    int32_t n_lin;            // 線形(GDN-2)層数
} jt_model1b_counts_t;

// 戻り値: JT_OK / JT_ERR_INVAL (不正cfg/NULL) / JT_ERR_NOMEM (積溢れ)。
int jt_model1b_counts(const jt_model1b_config_t *restrict cfg,
                      jt_model1b_counts_t *restrict out);

// 量子化後バイト会計 (mixed_precポリシーのビット幅を参照)。
typedef struct jt_model1b_bytes {
    uint64_t dense_weight_int8; // dense重みバイト (INT8換算。router/normsはfp32)
    uint64_t dense_scale_int8;  // denseスケールバイト (ceil(p/32)*4)
    uint64_t dense_total_int8;  // 上記合計
    uint64_t dense_weight_int4; // dense重みバイト (INT4換算。成功基準2GB判定用)
    uint64_t dense_scale_int4;  // denseスケールバイト (INT4時もfp32スケール)
    uint64_t dense_total_int4;  // 上記合計
    uint64_t ssd_weight;        // routing expert重み (gate/up=INT2, down=INT4)
    uint64_t ssd_scale;         // routing expertスケール (fp32/block)
    uint64_t ssd_total;         // 上記合計
} jt_model1b_bytes_t;

// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM。
int jt_model1b_bytes(const jt_model1b_config_t *restrict cfg,
                     jt_model1b_bytes_t *restrict out);

// 学習構成。
typedef struct jt_model1b_train_cfg {
    int32_t batch;          // >0 (既定8)
    int32_t seq;            // >0 (既定2048)
    int32_t elem_bytes;     // >0 (既定4=fp32活性)
    int32_t offload_routing;// 0/1 (既定1: expert重み+optimをSSD)
    int32_t n_seg;          // 0=auto(ceil(sqrt(L)))、>0で指定
} jt_model1b_train_cfg_t;

void jt_model1b_train_default(jt_model1b_train_cfg_t *restrict out);

// 学習時ピーク見積り。
typedef struct jt_model1b_train_mem {
    uint64_t resident_params; // RAM常駐param数 (offload時=total-routing)
    uint64_t weights_ram;     // fp32 master (resident*4)
    uint64_t grads_ram;       // resident*4 + offload時は1層分expert grads加算
    uint64_t optim_ram;       // resident*17/8 (offload experts分はSSD)
    uint64_t act_full;        // 全活性 (参考。jt_ckpt_mem_estimateと一致)
    uint64_t act_stored;      // 境界保存分
    uint64_t act_peak;        // stored + 最長区間再計算分
    uint64_t prefetch_buf;    // 投機先読みバッファ (2*K experts×fp32)
    uint64_t misc_ram;        // JT_MODEL1B_MISC_BYTES
    uint64_t peak_total;      // 上記合計 (16GB判定の母数)
    int32_t n_seg_used;       // 使用区間数
} jt_model1b_train_mem_t;

// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM。
int jt_model1b_train_mem(const jt_model1b_config_t *restrict cfg,
                         const jt_model1b_train_cfg_t *restrict tcfg,
                         jt_model1b_train_mem_t *restrict out);

#ifdef __cplusplus
}
#endif

#endif // JIMOTONO_MODEL1B_H
