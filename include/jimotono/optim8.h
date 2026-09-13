#pragma once
#ifndef JIMOTONO_OPTIM8_H
#define JIMOTONO_OPTIM8_H
// 8bitオプティマイザ足場 (Phase 2, bitsandbytes-CPU流儀)。
// AGENTS.MD 4.1: オプティマイザ状態を1/3.8に (fp32 m/v → int8+scale)。
// 規約: C11, restrict, errnoベース + goto cleanup (実装側)。ARENA不要
// (malloc/free)。OOM時は JT_ERR_NOMEM (common.h)。
//
// 方式 (scaffold決め打ち):
// - fp32→int8 ブロック量子化。ブロック64 (JT_OPTIM8_BLOCK)。
// - スケールは素朴fp32 (scale = max_abs/127, 対称量子化)。
//   TODO(FP4化): E4M3的スケール / E2M1状態化は将来。現状のfp32スケールは
//   オーバーヘッドとして計上し、per-block E4M3共有化で削減する (DESIGN 5.1-5
//   「LUTとスケールを重みと同じページに」の学習側適用)。
// - 状態は m/v (Adam一次・二次モーメント) を int8+scale で保持。
//   SGD-momentum単状態は本足場の対象外 (必要になれば別structで追加)。
// - 更新式は AdamW-lite (decoupled weight decay + bias補正付き)。
//
// 精度メモリ見積: fp32 m/v = 8B/param → int8 m/v + fp32 scale/blk =
//   2B/param + 8B/64param ≈ 2.125B/param (約1/3.8)。
//
// 禁止: llama.cpp/vLLM/SGLang/PyTorchリンクなし (AGENTS.MD 7.2)。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// 量子化ブロック長 (bitsandbytes流儀の小ブロック。64で固定)。
#define JT_OPTIM8_BLOCK 64

typedef struct jt_optim8_cfg {
    float lr;            // >0
    float beta1;         // (0,1), 既定0.9
    float beta2;         // (0,1), 既定0.999
    float eps;           // >0, 既定1e-8
    float weight_decay;  // >=0, 既定0 (AdamW decoupled)
} jt_optim8_cfg_t;

typedef struct jt_optim8 {
    size_t n;        // パラメータ数
    size_t nblocks;  // (n+63)/64
    float lr;
    float beta1;
    float beta2;
    float eps;
    float wd;
    uint64_t step;   // 更新回数 (bias補正 t=step, init 0)
    int8_t *q_m;     // [n] int8一次モーメント
    int8_t *q_v;     // [n] int8二次モーメント (非負値域のみ使用)
    float *s_m;      // [nblocks] fp32スケール
    float *s_v;      // [nblocks] fp32スケール
} jt_optim8_t;

// 既定cfg (lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0)。
void jt_optim8_cfg_default(jt_optim8_cfg_t *restrict cfg);

// nから必要ブロック数を返す ((n+63)/64)。
size_t jt_optim8_nblocks(size_t n);

// 初期化 (m=v=0 → q=0, scale=1.0)。cfg==NULLで既定値。
// 戻り値: JT_OK / JT_ERR_INVAL (NULL opt, n==0, 不正hyperparam, 非有限値)
//          JT_ERR_NOMEM (malloc失敗, errno=ENOMEM)
int jt_optim8_init(jt_optim8_t *restrict opt, size_t n,
                   const jt_optim8_cfg_t *restrict cfg);

// 解放 (二重fini安全: 内部NULLなら何もしない。opt自体NULLも安全)。
void jt_optim8_fini(jt_optim8_t *restrict opt);

// 1ブロック列量子化: src[0..n) → dst[0..n) + scales[0..nblocks)。
// nblocksは jt_optim8_nblocks(n) と一致すること。末尾ブロックは実要素のみで
// maxを取る。ゼロブロックは scale=1.0, q=0。
// 戻り値: JT_OK / JT_ERR_INVAL (NULL, n==0, nblocks不一致, 非有限src,
//          errno=EINVAL)
int jt_optim8_quantize(const float *restrict src, size_t n,
                       int8_t *restrict dst, float *restrict scales,
                       size_t nblocks);

// 逆量子化: dst[i] = (float)src[i] * scales[i/64]。
// 戻り値: JT_OK / JT_ERR_INVAL (NULL, n==0, nblocks不一致, 非有限scale,
//          errno=EINVAL)
int jt_optim8_dequantize(const int8_t *restrict src,
                         const float *restrict scales, size_t nblocks,
                         size_t n, float *restrict dst);

// AdamW-lite 1ステップ (bias補正付き, decoupled WD):
//   m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g*g
//   mh = m/(1-b1^t) ; vh = v/(1-b2^t)
//   p -= lr * (mh/(sqrt(vh)+eps) + wd*p)
// 状態m/vはint8+scaleで往復する (ブロック64量子化誤差ありのscaffold)。
// nはopt->nと一致すること。
// 戻り値: JT_OK / JT_ERR_INVAL (NULL, n==0/不一致, 非有限grad,
//          errno=EINVAL)
int jt_optim8_step(jt_optim8_t *restrict opt, float *restrict param,
                   const float *restrict grad, size_t n);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_OPTIM8_H
