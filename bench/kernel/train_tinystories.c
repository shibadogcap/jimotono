// train_tinystories: バイトレベル次トークン予測の小規模言語モデル学習
//   (実行バイナリ、ctest非登録)。Phase G以降の評価用スケーリング曲線担当。
//   TinyStories先頭サブセット (約1.8万話・16MB) による実言語タスクで、
//   容量(steps)増加に対しtrain/val CEが下がり続けることを確認する。
//
// 背景・選定理由:
//   * TinyStories (roneneldan/TinyStories, CDLA-Sharing-1.0) は到達確認済み。
//     フルは train 1.92GB・212万話・全体7.62GBのため、先頭16MiBのみを
//     Range取得し末尾を<|endoftext|>境界で切断 (18544話・16776860B、決定論的)。
//     5000 steps x batch 64 = 32万ペア、親の50000 steps x 256 = 1280万ペア
//     のいずれにも足りる量 (総ペア約1600万)。
//   * B(日本語ツールコール)/C(コード)は調査のみ・DL未実行 (親判断待ち):
//     B候補=Tonari-no-usagi版(500件・164kB・CC-BY-NC-4.0、フルはGumroad有償)、
//     DataPilot版(7000件・20.8MB・odc-by)、nappa0326 glaive-ja訳
//     (112960件・232MB)。C候補=SwallowCode-v2 (49.8Bトークン・Apache-2.0だが
//     stage5約954GBで16GB/1hに不適合、parquet先頭切り出し要・未実行)。
//   * トークナイザーはバイトレベル256語彙+特殊2 (V=258既定) に決定。
//     GPT-2 BPEファイル自体は到達可能
//     (encoder.json 1042301B・openaipublic、vocab.bpeも同所) だったが採用せず:
//     (1) data_packのJT_DP_VOCAB_SIZE=48588がpack内ID<48588を強制し、
//     50257語彙はdata_pack経路に乗らない (data_pack.c改変は既存テスト破壊の
//     ため不可)。(2) BPEマージのC実装は1h-CPU枠外。(3) バイト列でも実言語の
//     スケーリング曲線は観測可能。
//   * Phase F6: V=48588 (llm-jp-tokenizer v2.2 Unigram) 経路を追加。
//     TinyStoriesテキストをbpe C実装 (jt_bpe_encode) で物語単位にトークン化し、
//     data_pack (.jtdp) 経由または直接で学習する。--vocab 48588 +
//     --bpe-vocab data/llmjp-v22.jtvocab (scripts/bpe_make_vocab.py生成)。
//     BPE経路の系列は生BPE ID列 (BOS/EOS付加なし。ID 0/1は実ピースのため
//     衝突回避。物語境界は系列分割で保持)。V=258経路は不変。
//   * 旧--vocab 50257 (構造対応のみ) は廃止。指定時は使用不可エラー。
//     (50257はJT_DP_VOCAB_SIZE=48588を超えるIDを含むためpack経路に乗らない)
//
// 構成: byte embedding (学習対象) → L層の小規模MoE (jt_moe_fwd/bwd再利用、
//   d=64既定・E=8・top-2・S=1・H=32、残差接続) → 線形head＋softmax CE。
// CEのforward/backwardは本ファイル内で新規実装 (C11・fail-closed)。
// データ: 物語単位で分割 (i%10==9をval、残りをtrain、決定論的。namesと同理由で
//   等間隔抽出)。valはforwardのみ (勾配計算・更新なし)。
// data_pack経路: --pack PATH (既定data/tinystories16M.jtdp) が開ければ
//   jt_dp_open経由で読む (正規経路)。なければテキスト直接読みにフォールバック
//   (その旨をstdoutに明記)。--write-pack PATHでテキスト→.jtdp変換
//   (jt_dp_writer使用)。V=258のbyte-fallback IDは最大257のため
//   JT_DP_VOCAB_SIZE制限 (<48588) を満たす。V=48588のBPE IDは語彙由来で
//   <48588 (bpe_make_vocab.pyがassert)。V=258/48588いずれでもpack経路が有効。
// CE_per_byte正規化 (RULE.MD): トークン数の異なる語彙をtoks/sで比較禁止。
//   比較軸は同一wall-clock・同一バイト数。valはCE_per_token × tokens/bytesを
//   val_ce_pb列として記録 (tokens/bytesは当該Vの実測値 =
//   (train+valペア数)/テキストバイト数)。pack経路時は--dataのファイルサイズを
//   分母に用いる (同コーパス前提。取得不可時は0→val_ce_pbはn/a表示)。
// 早期停止: --patience P (既定100・steps単位)。最良val更新からP steps経過で停止。
//   加えて直近3回のval評価値が厳密に単調増加 (移動平均窓=1の3点連続上昇) したら停止。
//   --patience 0で無効化。
// 規約: C11, restrict積極使用, errnoベース + goto cleanup (AGENTS.MD 7.1)。
// 単一スレッド・決定論的 (同一引数で同一結果)。既存API・テストに不変。
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "jimotono/common.h"
#include "jimotono/bpe.h"
#include "jimotono/data_pack.h"
#include "jimotono/moe_layer.h"
#include "jimotono/routing.h"
#include "jimotono/optim8.h"

#define TS_BOS 0
#define TS_EOS 1
#define TS_BYTE_OFF 2 /* byte b -> 2+b (2..257)。V=258経路のみ */
#define TS_VOCAB_SMALL 258
#define TS_VOCAB_LLJ 48588 /* llm-jp-tokenizer v2.2 Unigram (bpe C実装で符号化) */
#define TS_BPE_VOCAB_DEFAULT "data/llmjp-v22.jtvocab"

#define TS_E 8
#define TS_K 2
#define TS_S 1
#define TS_H 32

#define TS_DELIM "<|endoftext|>"
#define TS_DELIM_LEN 13

// Phase G G2改訂: --moe-batch でバッチ経路を使用。既定0=単体ループ。
// 同一データ順・同一初期化で single vs batch を比較する。
// T=1時は per-token batch API、T>1時（--seq/--batchでT=seq*batch）は
// jt_moe_fwd_batch/bwd_batch にTトークンを一括投入する本番経路。
// capは1.5維持（2.0での隠蔽禁止）。
static int g_ts_moe_batch = 0;
#define TS_CAP_FACTOR 1.5f
// G2fix: seq×batchデータローダー拡張（既定seq=1で従来互換）。
// 例: --seq 128 --batch 4 でT=512。現T=1固定の制約を解除する。
static long g_ts_seq = 1;
// load-balancing補助損失重みμ（既定0.01、調整範囲0.01〜0.1。0で無効化可）。
static float g_ts_aux_w = 0.01f;

// ---- 決定論的ハッシュ乱数 (train_names/train_proxy100mと同流儀) ----
static uint64_t ts_hash64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// ---- softmax交差エントロピー (新規実装、fail-closed。Vは実行時可変) ----
static int ts_ce_fwd(const float *restrict logits, int v, int target,
                     float *restrict loss_out, float *restrict probs_out) {
    double mx = 0.0;
    double sum = 0.0;
    double loss = 0.0;
    int i = 0;
    if (logits == NULL || loss_out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (v < 2 || v > 65536 || target < 0 || target >= v) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    mx = (double)logits[0];
    for (i = 1; i < v; i++) {
        double x = (double)logits[i];
        if (!isfinite(x)) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        if (x > mx) {
            mx = x;
        }
    }
    if (!isfinite((double)logits[0])) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (i = 0; i < v; i++) {
        sum += exp((double)logits[i] - mx);
    }
    if (!(sum > 0.0) || !isfinite(sum)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    loss = log(sum) - ((double)logits[target] - mx);
    if (!isfinite(loss) || loss < 0.0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (probs_out != NULL) {
        for (i = 0; i < v; i++) {
            probs_out[i] = (float)(exp((double)logits[i] - mx) / sum);
        }
    }
    *loss_out = (float)loss;
    return JT_OK;
}

static int ts_ce_bwd(const float *restrict probs, int v, int target,
                     float scale, float *restrict dlogits_out) {
    int i = 0;
    if (probs == NULL || dlogits_out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (v < 2 || v > 65536 || target < 0 || target >= v) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (!isfinite((double)scale)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (i = 0; i < v; i++) {
        double p = (double)probs[i];
        if (!isfinite(p) || p < 0.0) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    for (i = 0; i < v; i++) {
        double p = (double)probs[i];
        double dd = p - ((i == target) ? 1.0 : 0.0);
        dlogits_out[i] = (float)((double)scale * dd);
    }
    return JT_OK;
}

static double ts_now_sec(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void ts_usage(const char *prog) {
    fprintf(stderr,
            "usage: %s [--data txt] [--pack jtdp] [--write-pack out.jtdp] "
            "[--steps N] [--time SECS] [--lr LR] [--batch B] [--d DIM] "
            "[--layers L] [--val-every K] [--patience P] [--vocab V] "
            "[--bpe-vocab PATH] [--max-stories N] [--max-val-pairs N] "
            "[--moe-batch] [--seq S] [--aux-weight W]\n"
            "  defaults: data=data/tinystories_head16M.txt "
            "pack=data/tinystories16M.jtdp steps=500 time=3600 lr=3e-4 "
            "batch=64 d=64 layers=2 val-every=100 patience=100 vocab=258 "
            "bpe-vocab=" TS_BPE_VOCAB_DEFAULT " "
            "max-stories=0(all) max-val-pairs=20000 seq=1 aux-weight=0.01\n"
            "  --vocab: 258 (byte+special) or 48588 (llm-jp v2.2 Unigram, "
            "requires --bpe-vocab)\n"
            "  --patience 0 disables early stopping\n"
            "  --max-val-pairs 0 evaluates full val set\n"
            "  --moe-batch: Phase G batch dispatch (cap=1.5; T=seq*batch)\n"
            "  --seq S: tokens per sequence (1..512). T_step=seq*batch (<=1024)\n"
            "  --aux-weight W: load-balancing L_aux weight (0..0.1, default "
            "0.01; tunable 0.01-0.1)\n",
            prog);
}

// ---- パラメータ配置 (単一Pバッファ。Gも同一配置。Vは実行時可変) ----
//   [emb V*d][layer0][layer1]...[Wo V*d][bo V]
typedef struct ts_layout {
    size_t off_emb;
    size_t layer_stride;
    size_t off_wgate;
    size_t off_wg;
    size_t off_wu;
    size_t off_wd;
    size_t off_wgs;
    size_t off_wus;
    size_t off_wds;
    size_t off_head_wo;
    size_t off_head_bo;
    size_t n_total;
    int n_layers;
    int d;
    int v;
} ts_layout_t;

static void ts_build_layout(ts_layout_t *lt, int n_layers, int d, int v) {
    size_t e = (size_t)TS_E;
    size_t h = (size_t)TS_H;
    size_t dd = (size_t)d;
    size_t s = (size_t)TS_S;
    size_t vv = (size_t)v;
    size_t routed = e * h * dd;
    size_t shared = s * h * dd;
    memset(lt, 0, sizeof(*lt));
    lt->n_layers = n_layers;
    lt->d = d;
    lt->v = v;
    lt->off_emb = 0;
    lt->off_wgate = 0;
    lt->off_wg = e * dd;
    lt->off_wu = e * dd + routed;
    lt->off_wd = e * dd + (size_t)2 * routed;
    lt->off_wgs = e * dd + (size_t)3 * routed;
    lt->off_wus = e * dd + (size_t)3 * routed + shared;
    lt->off_wds = e * dd + (size_t)3 * routed + (size_t)2 * shared;
    lt->layer_stride = e * dd + (size_t)3 * routed + (size_t)3 * shared;
    lt->off_head_wo = vv * dd + (size_t)n_layers * lt->layer_stride;
    lt->off_head_bo = vv * dd + (size_t)n_layers * lt->layer_stride + vv * dd;
    lt->n_total = vv * dd + (size_t)n_layers * lt->layer_stride + vv * dd + vv;
}

typedef struct ts_model {
    ts_layout_t lt;
    float *P;
    float *G;
    jt_optim8_t opt;
    int opt_inited;
} ts_model_t;

static int ts_fwd_one(const ts_model_t *restrict m, int x,
                      float *restrict acts, size_t *restrict cids,
                      float *restrict cw, float *restrict cgsel,
                      float *restrict cusel, float *restrict cysel,
                      float *restrict cgs, float *restrict cus,
                      float *restrict logits, float *restrict probs,
                      float *restrict loss_out, int y) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    const float *P = m->P;
    const float *emb = P + m->lt.off_emb;
    const float *Wo = P + m->lt.off_head_wo;
    const float *bo = P + m->lt.off_head_bo;
    float M[256];
    int l = 0;
    int j = 0;
    int v = 0;
    if (d > 256 || x < 0 || x >= V) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (j = 0; j < d; j++) {
        acts[j] = emb[(size_t)x * (size_t)d + (size_t)j];
    }
    for (l = 0; l < L; l++) {
        const float *base =
            P + m->lt.off_emb + (size_t)V * (size_t)d +
            (size_t)l * m->lt.layer_stride;
        const float *Wgate = base + m->lt.off_wgate;
        const float *Wg = base + m->lt.off_wg;
        const float *Wu = base + m->lt.off_wu;
        const float *Wd = base + m->lt.off_wd;
        const float *Wgs = base + m->lt.off_wgs;
        const float *Wus = base + m->lt.off_wus;
        const float *Wds = base + m->lt.off_wds;
        const float *xin = acts + (size_t)l * (size_t)d;
        float *xout = acts + (size_t)(l + 1) * (size_t)d;
        size_t *ids = cids + (size_t)l * (size_t)TS_K;
        float *w = cw + (size_t)l * (size_t)TS_K;
        float *Gsel = cgsel + (size_t)l * (size_t)TS_K * (size_t)TS_H;
        float *Usel = cusel + (size_t)l * (size_t)TS_K * (size_t)TS_H;
        float *Ysel = cysel + (size_t)l * (size_t)TS_K * (size_t)d;
        float *Gs = cgs + (size_t)l * (size_t)TS_S * (size_t)TS_H;
        float *Us = cus + (size_t)l * (size_t)TS_S * (size_t)TS_H;
        int rc = jt_moe_fwd(xin, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds, M, d,
                            TS_H, TS_E, TS_K, TS_S, ids, w, NULL, Gsel,
                            Usel, Ysel, Gs, Us);
        if (rc != JT_OK) {
            return rc;
        }
        for (j = 0; j < d; j++) {
            double t = (double)xin[j] + (double)M[j];
            if (!isfinite(t)) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
            xout[j] = (float)t;
        }
    }
    {
        const float *hl = acts + (size_t)L * (size_t)d;
        for (v = 0; v < V; v++) {
            double acc = (double)bo[v];
            for (j = 0; j < d; j++) {
                acc += (double)Wo[(size_t)v * (size_t)d + (size_t)j] *
                       (double)hl[j];
            }
            if (!isfinite(acc)) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
            logits[v] = (float)acc;
        }
    }
    return ts_ce_fwd(logits, V, y, loss_out, probs);
}

static int ts_bwd_one(ts_model_t *restrict m, int x,
                      const float *restrict acts,
                      const size_t *restrict cids, const float *restrict cw,
                      const float *restrict cgsel, const float *restrict cusel,
                      const float *restrict cysel, const float *restrict cgs,
                      const float *restrict cus,
                      const float *restrict probs, float *restrict dlog,
                      float *restrict dh, float *restrict dX,
                      float *restrict tW, int y, float dlog_scale) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    float *G = m->G;
    const float *P = m->P;
    float *Gemb = G + m->lt.off_emb;
    float *GWo = G + m->lt.off_head_wo;
    float *Gbo = G + m->lt.off_head_bo;
    const float *Wo = P + m->lt.off_head_wo;
    int l = 0;
    int j = 0;
    int v = 0;
    if (ts_ce_bwd(probs, V, y, dlog_scale, dlog) != JT_OK) {
        return JT_ERR_INVAL;
    }
    {
        const float *hl = acts + (size_t)L * (size_t)d;
        for (j = 0; j < d; j++) {
            dh[j] = 0.0f;
        }
        for (v = 0; v < V; v++) {
            float g = dlog[v];
            Gbo[v] += g;
            for (j = 0; j < d; j++) {
                GWo[(size_t)v * (size_t)d + (size_t)j] += g * hl[j];
                dh[j] += g * Wo[(size_t)v * (size_t)d + (size_t)j];
            }
        }
    }
    for (l = L - 1; l >= 0; l--) {
        const float *base =
            P + m->lt.off_emb + (size_t)V * (size_t)d +
            (size_t)l * m->lt.layer_stride;
        float *gbase = G + m->lt.off_emb +
                       (size_t)V * (size_t)d +
                       (size_t)l * m->lt.layer_stride;
        const float *Wgate = base + m->lt.off_wgate;
        const float *Wg = base + m->lt.off_wg;
        const float *Wu = base + m->lt.off_wu;
        const float *Wd = base + m->lt.off_wd;
        const float *Wgs = base + m->lt.off_wgs;
        const float *Wus = base + m->lt.off_wus;
        const float *Wds = base + m->lt.off_wds;
        const float *xin = acts + (size_t)l * (size_t)d;
        const size_t *ids = cids + (size_t)l * (size_t)TS_K;
        const float *w = cw + (size_t)l * (size_t)TS_K;
        const float *Gsel = cgsel + (size_t)l * (size_t)TS_K * (size_t)TS_H;
        const float *Usel = cusel + (size_t)l * (size_t)TS_K * (size_t)TS_H;
        const float *Ysel = cysel + (size_t)l * (size_t)TS_K * (size_t)d;
        const float *Gs = cgs + (size_t)l * (size_t)TS_S * (size_t)TS_H;
        const float *Us = cus + (size_t)l * (size_t)TS_S * (size_t)TS_H;
        float *tWgate = tW + m->lt.off_wgate;
        float *tWg = tW + m->lt.off_wg;
        float *tWu = tW + m->lt.off_wu;
        float *tWd = tW + m->lt.off_wd;
        float *tWgs = tW + m->lt.off_wgs;
        float *tWus = tW + m->lt.off_wus;
        float *tWds = tW + m->lt.off_wds;
        size_t z = 0;
        int rc = jt_moe_bwd(dh, xin, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds,
                            ids, w, Gsel, Usel, Ysel, Gs, Us, dX, tWgate,
                            tWg, tWu, tWd, tWgs, tWus, tWds, NULL, d, TS_H,
                            TS_E, TS_K, TS_S);
        if (rc != JT_OK) {
            return rc;
        }
        for (z = 0; z < m->lt.layer_stride; z++) {
            gbase[z] += tW[z];
        }
        for (j = 0; j < d; j++) {
            dh[j] += dX[j];
        }
    }
    for (j = 0; j < d; j++) {
        Gemb[(size_t)x * (size_t)d + (size_t)j] += dh[j];
    }
    return JT_OK;
}

// ---- Phase G G2改訂: バッチ経路 (T=1 per-token batch API) ----
// 単体版と同一レイアウトの cache を使い、MoE層のみ jt_moe_fwd_batch /
// jt_moe_bwd_batch (T=1, cap=1.5) に置換する。T=1・E=8・k=2では
// cap=ceil(1.5*2/8)=1、各ペアは distinct expert のため drop は通常0
// (H1分離：drop 0なら残差はH2)。renormalizeはライブラリ側で実施。
// perm/off/drop は層別に呼び出し側確保域へ写す (bwdで同一perm再利用、再ソート禁止)。
// fail-closed: 検証失敗時は acts/Y 更新前に拒否 (Y不変は下位APIが保証)。
static int ts_fwd_one_batch(const ts_model_t *restrict m, int x,
                            float *restrict acts, size_t *restrict cids,
                            float *restrict cw, float *restrict cgsel,
                            float *restrict cusel, float *restrict cysel,
                            float *restrict cgs, float *restrict cus,
                            size_t *restrict bperm, size_t *restrict boff,
                            unsigned char *restrict bdrop,
                            size_t *restrict out_dropped,
                            float *restrict logits, float *restrict probs,
                            float *restrict loss_out, int y) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    const float *P = m->P;
    const float *emb = P + m->lt.off_emb;
    const float *Wo = P + m->lt.off_head_wo;
    const float *bo = P + m->lt.off_head_bo;
    float M[256];
    size_t dropped_total = 0;
    int l = 0;
    int j = 0;
    int v = 0;
    if (d > 256 || x < 0 || x >= V) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (j = 0; j < d; j++) {
        acts[j] = emb[(size_t)x * (size_t)d + (size_t)j];
    }
    for (l = 0; l < L; l++) {
        const float *base =
            P + m->lt.off_emb + (size_t)V * (size_t)d +
            (size_t)l * m->lt.layer_stride;
        const float *Wgate = base + m->lt.off_wgate;
        const float *Wg = base + m->lt.off_wg;
        const float *Wu = base + m->lt.off_wu;
        const float *Wd = base + m->lt.off_wd;
        const float *Wgs = base + m->lt.off_wgs;
        const float *Wus = base + m->lt.off_wus;
        const float *Wds = base + m->lt.off_wds;
        const float *xin = acts + (size_t)l * (size_t)d;
        size_t *ids = cids + (size_t)l * (size_t)TS_K;
        float *w = cw + (size_t)l * (size_t)TS_K;
        float *Gsel = cgsel + (size_t)l * (size_t)TS_K * (size_t)TS_H;
        float *Usel = cusel + (size_t)l * (size_t)TS_K * (size_t)TS_H;
        float *Ysel = cysel + (size_t)l * (size_t)TS_K * (size_t)d;
        float *Gs = cgs + (size_t)l * (size_t)TS_S * (size_t)TS_H;
        float *Us = cus + (size_t)l * (size_t)TS_S * (size_t)TS_H;
        size_t perm_tmp[TS_K];
        size_t off_tmp[TS_E + 1];
        unsigned char drop_tmp[TS_K];
        size_t kept = 0;
        size_t dropped = 0;
        int rc = jt_moe_fwd_batch(xin, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds,
                                  M, 1, d, TS_H, TS_E, TS_K, TS_S, ids, w,
                                  Gsel, Usel, Ysel, Gs, Us, TS_CAP_FACTOR,
                                  perm_tmp, off_tmp, drop_tmp, &kept,
                                  &dropped);
        if (rc != JT_OK) {
            return rc;
        }
        dropped_total += dropped;
        if (bperm != NULL) {
            size_t *bp = bperm + (size_t)l * (size_t)TS_K;
            for (size_t i = 0; i < kept && i < (size_t)TS_K; i++) {
                bp[i] = perm_tmp[i];
            }
            for (size_t i = kept; i < (size_t)TS_K; i++) {
                bp[i] = 0;
            }
        }
        if (boff != NULL) {
            size_t *bop = boff + (size_t)l * ((size_t)TS_E + 1);
            for (int e = 0; e <= TS_E; e++) {
                bop[(size_t)e] = off_tmp[(size_t)e];
            }
        }
        if (bdrop != NULL) {
            unsigned char *bd = bdrop + (size_t)l * (size_t)TS_K;
            for (int p = 0; p < TS_K; p++) {
                bd[(size_t)p] = drop_tmp[(size_t)p];
            }
        }
        for (j = 0; j < d; j++) {
            double t = (double)xin[j] + (double)M[j];
            if (!isfinite(t)) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
            (acts + (size_t)(l + 1) * (size_t)d)[j] = (float)t;
        }
    }
    {
        const float *hl = acts + (size_t)L * (size_t)d;
        for (v = 0; v < V; v++) {
            double acc = (double)bo[v];
            for (j = 0; j < d; j++) {
                acc += (double)Wo[(size_t)v * (size_t)d + (size_t)j] *
                       (double)hl[j];
            }
            if (!isfinite(acc)) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
            logits[v] = (float)acc;
        }
    }
    if (out_dropped != NULL) {
        *out_dropped = dropped_total;
    }
    return ts_ce_fwd(logits, V, y, loss_out, probs);
}

static int ts_bwd_one_batch(ts_model_t *restrict m, int x,
                            const float *restrict acts,
                            const size_t *restrict cids,
                            const float *restrict cw,
                            const float *restrict cgsel,
                            const float *restrict cusel,
                            const float *restrict cysel,
                            const float *restrict cgs,
                            const float *restrict cus,
                            const size_t *restrict bperm,
                            const size_t *restrict boff,
                            const unsigned char *restrict bdrop,
                            const float *restrict probs, float *restrict dlog,
                            float *restrict dh, float *restrict dX,
                            float *restrict tW, int y, float dlog_scale) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    float *G = m->G;
    const float *P = m->P;
    float *Gemb = G + m->lt.off_emb;
    float *GWo = G + m->lt.off_head_wo;
    float *Gbo = G + m->lt.off_head_bo;
    const float *Wo = P + m->lt.off_head_wo;
    int l = 0;
    int j = 0;
    int v = 0;
    if (bperm == NULL || boff == NULL || bdrop == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (ts_ce_bwd(probs, V, y, dlog_scale, dlog) != JT_OK) {
        return JT_ERR_INVAL;
    }
    {
        const float *hl = acts + (size_t)L * (size_t)d;
        for (j = 0; j < d; j++) {
            dh[j] = 0.0f;
        }
        for (v = 0; v < V; v++) {
            float g = dlog[v];
            Gbo[v] += g;
            for (j = 0; j < d; j++) {
                GWo[(size_t)v * (size_t)d + (size_t)j] += g * hl[j];
                dh[j] += g * Wo[(size_t)v * (size_t)d + (size_t)j];
            }
        }
    }
    for (l = L - 1; l >= 0; l--) {
        const float *base =
            P + m->lt.off_emb + (size_t)V * (size_t)d +
            (size_t)l * m->lt.layer_stride;
        float *gbase = G + m->lt.off_emb +
                       (size_t)V * (size_t)d +
                       (size_t)l * m->lt.layer_stride;
        const float *Wgate = base + m->lt.off_wgate;
        const float *Wg = base + m->lt.off_wg;
        const float *Wu = base + m->lt.off_wu;
        const float *Wd = base + m->lt.off_wd;
        const float *Wgs = base + m->lt.off_wgs;
        const float *Wus = base + m->lt.off_wus;
        const float *Wds = base + m->lt.off_wds;
        const float *xin = acts + (size_t)l * (size_t)d;
        const size_t *ids = cids + (size_t)l * (size_t)TS_K;
        const float *w = cw + (size_t)l * (size_t)TS_K;
        const float *Gsel = cgsel + (size_t)l * (size_t)TS_K * (size_t)TS_H;
        const float *Usel = cusel + (size_t)l * (size_t)TS_K * (size_t)TS_H;
        const float *Ysel = cysel + (size_t)l * (size_t)TS_K * (size_t)d;
        const float *Gs = cgs + (size_t)l * (size_t)TS_S * (size_t)TS_H;
        const float *Us = cus + (size_t)l * (size_t)TS_S * (size_t)TS_H;
        const size_t *perm = bperm + (size_t)l * (size_t)TS_K;
        const size_t *off = boff + (size_t)l * ((size_t)TS_E + 1);
        const unsigned char *dropm = bdrop + (size_t)l * (size_t)TS_K;
        float *tWgate = tW + m->lt.off_wgate;
        float *tWg = tW + m->lt.off_wg;
        float *tWu = tW + m->lt.off_wu;
        float *tWd = tW + m->lt.off_wd;
        float *tWgs = tW + m->lt.off_wgs;
        float *tWus = tW + m->lt.off_wus;
        float *tWds = tW + m->lt.off_wds;
        size_t z = 0;
        int rc = jt_moe_bwd_batch(dh, xin, Wgate, Wg, Wu, Wd, Wgs, Wus,
                                  Wds, ids, w, Gsel, Usel, Ysel, Gs, Us,
                                  perm, off, dropm, dX, tWgate, tWg, tWu,
                                  tWd, tWgs, tWus, tWds, NULL, 1, d, TS_H,
                                  TS_E, TS_K, TS_S);
        if (rc != JT_OK) {
            return rc;
        }
        for (z = 0; z < m->lt.layer_stride; z++) {
            gbase[z] += tW[z];
        }
        for (j = 0; j < d; j++) {
            dh[j] += dX[j];
        }
    }
    for (j = 0; j < d; j++) {
        Gemb[(size_t)x * (size_t)d + (size_t)j] += dh[j];
    }
    return JT_OK;
}

// ---- G2fix: T=seq*batch一括経路 (layout [L+1][T][d]、fail-closed) ----
// actsT: [(L+1)*T*d]、cidsT/cwT: [L*T*K]、cgselT/cuselT: [L*T*K*H]、
// cyselT: [L*T*K*d]、cgsT/cusT: [L*T*S*H]。単体版はトークン毎jt_moe_fwd、
// バッチ版は層毎jt_moe_fwd_batch(T)。head/CEは両者ともトークン毎共通核。
// permT/offT/dropTはバッチ版のみ使用（単体版ではNULL）。
static int ts_fwd_T_single(const ts_model_t *restrict m, const int *restrict TX,
                           int T, float *restrict actsT,
                           size_t *restrict cidsT, float *restrict cwT,
                           float *restrict cgselT, float *restrict cuselT,
                           float *restrict cyselT, float *restrict cgsT,
                           float *restrict cusT, float *restrict logits_tmp,
                           float *restrict probsT, double *restrict loss_sum) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    const float *P = m->P;
    const float *emb = P + m->lt.off_emb;
    const float *Wo = P + m->lt.off_head_wo;
    const float *bo = P + m->lt.off_head_bo;
    double sum = 0.0;
    if (m == NULL || TX == NULL || actsT == NULL || cidsT == NULL ||
        cwT == NULL || cgselT == NULL || cuselT == NULL || cyselT == NULL ||
        cgsT == NULL || cusT == NULL || logits_tmp == NULL ||
        probsT == NULL || loss_sum == NULL || T <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (d > 256) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (int t = 0; t < T; t++) {
        int x = TX[t];
        float *a0;
        if (x < 0 || x >= V) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        a0 = actsT + (size_t)t * (size_t)d;
        for (int j = 0; j < d; j++) {
            a0[j] = emb[(size_t)x * (size_t)d + (size_t)j];
        }
    }
    for (int l = 0; l < L; l++) {
        const float *base = P + m->lt.off_emb + (size_t)V * (size_t)d +
                            (size_t)l * m->lt.layer_stride;
        const float *Wgate = base + m->lt.off_wgate;
        const float *Wg = base + m->lt.off_wg;
        const float *Wu = base + m->lt.off_wu;
        const float *Wd = base + m->lt.off_wd;
        const float *Wgs = base + m->lt.off_wgs;
        const float *Wus = base + m->lt.off_wus;
        const float *Wds = base + m->lt.off_wds;
        for (int t = 0; t < T; t++) {
            const float *xin =
                actsT + ((size_t)l * (size_t)T + (size_t)t) * (size_t)d;
            float *xout =
                actsT + ((size_t)(l + 1) * (size_t)T + (size_t)t) * (size_t)d;
            size_t *ids =
                cidsT + ((size_t)l * (size_t)T + (size_t)t) * (size_t)TS_K;
            float *w =
                cwT + ((size_t)l * (size_t)T + (size_t)t) * (size_t)TS_K;
            float *Gsel =
                cgselT + ((size_t)l * (size_t)T + (size_t)t) *
                             (size_t)TS_K * (size_t)TS_H;
            float *Usel =
                cuselT + ((size_t)l * (size_t)T + (size_t)t) *
                             (size_t)TS_K * (size_t)TS_H;
            float *Ysel =
                cyselT +
                ((size_t)l * (size_t)T + (size_t)t) * (size_t)TS_K * (size_t)d;
            float *Gs = cgsT + ((size_t)l * (size_t)T + (size_t)t) *
                                   (size_t)TS_S * (size_t)TS_H;
            float *Us = cusT + ((size_t)l * (size_t)T + (size_t)t) *
                                   (size_t)TS_S * (size_t)TS_H;
            float M[256];
            int rc = jt_moe_fwd(xin, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds, M, d,
                                TS_H, TS_E, TS_K, TS_S, ids, w, NULL, Gsel,
                                Usel, Ysel, Gs, Us);
            if (rc != JT_OK) {
                return rc;
            }
            for (int j = 0; j < d; j++) {
                double v = (double)xin[j] + (double)M[j];
                if (!isfinite(v)) {
                    errno = EINVAL;
                    return JT_ERR_INVAL;
                }
                xout[j] = (float)v;
            }
        }
    }
    // head/CEはts_head_ce_fwdで行う（本関数はMoE＋残差まで）。
    (void)P;
    (void)Wo;
    (void)bo;
    (void)logits_tmp;
    (void)probsT;
    sum = 0.0;
    *loss_sum = sum;
    return JT_OK;
}

// head+CE一括（fwd）：hlT=[T][d]相当（actsTの最終層）、TY[T]。
// loss_sumに合計、probsT[T][V]に分布を格納。fail-closed。
static int ts_head_ce_fwd(const ts_model_t *restrict m,
                          const float *restrict actsT, int T,
                          const int *restrict TY, float *restrict logits_tmp,
                          float *restrict probsT,
                          double *restrict loss_sum) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    const float *P = m->P;
    const float *Wo = P + m->lt.off_head_wo;
    const float *bo = P + m->lt.off_head_bo;
    double sum = 0.0;
    if (m == NULL || actsT == NULL || TY == NULL || logits_tmp == NULL ||
        probsT == NULL || loss_sum == NULL || T <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (int t = 0; t < T; t++) {
        const float *hl =
            actsT + ((size_t)L * (size_t)T + (size_t)t) * (size_t)d;
        float *prow = probsT + (size_t)t * (size_t)V;
        float loss = 0.0f;
        int y = TY[t];
        if (y < 0 || y >= V) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        for (int v = 0; v < V; v++) {
            double acc = (double)bo[v];
            for (int j = 0; j < d; j++) {
                acc += (double)Wo[(size_t)v * (size_t)d + (size_t)j] *
                       (double)hl[j];
            }
            if (!isfinite(acc)) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
            logits_tmp[v] = (float)acc;
        }
        if (ts_ce_fwd(logits_tmp, V, y, &loss, prow) != JT_OK) {
            return JT_ERR_INVAL;
        }
        sum += (double)loss;
    }
    *loss_sum = sum;
    return JT_OK;
}

// T一括バッチfwd（層毎jt_moe_fwd_batch）。permT/offT/dropTを保存。
// out_dropped/renorm_fire（層×トークン単位の発火数）を返す。
static int ts_fwd_T_batch(const ts_model_t *restrict m,
                          const int *restrict TX, int T, float *restrict actsT,
                          size_t *restrict cidsT, float *restrict cwT,
                          float *restrict cgselT, float *restrict cuselT,
                          float *restrict cyselT, float *restrict cgsT,
                          float *restrict cusT, size_t *restrict permT,
                          size_t *restrict offT, unsigned char *restrict dropT,
                          size_t *restrict out_dropped,
                          size_t *restrict out_renorm) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    const float *P = m->P;
    const float *emb = P + m->lt.off_emb;
    size_t dropped_total = 0;
    size_t renorm_total = 0;
    if (m == NULL || TX == NULL || actsT == NULL || cidsT == NULL ||
        cwT == NULL || cgselT == NULL || cuselT == NULL || cyselT == NULL ||
        cgsT == NULL || cusT == NULL || permT == NULL || offT == NULL ||
        dropT == NULL || T <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (int t = 0; t < T; t++) {
        int x = TX[t];
        float *a0;
        if (x < 0 || x >= V) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        a0 = actsT + (size_t)t * (size_t)d;
        for (int j = 0; j < d; j++) {
            a0[j] = emb[(size_t)x * (size_t)d + (size_t)j];
        }
    }
    for (int l = 0; l < L; l++) {
        const float *base = P + m->lt.off_emb + (size_t)V * (size_t)d +
                            (size_t)l * m->lt.layer_stride;
        const float *Wgate = base + m->lt.off_wgate;
        const float *Wg = base + m->lt.off_wg;
        const float *Wu = base + m->lt.off_wu;
        const float *Wd = base + m->lt.off_wd;
        const float *Wgs = base + m->lt.off_wgs;
        const float *Wus = base + m->lt.off_wus;
        const float *Wds = base + m->lt.off_wds;
        const float *Xb = actsT + (size_t)l * (size_t)T * (size_t)d;
        float *Yb = actsT + (size_t)(l + 1) * (size_t)T * (size_t)d;
        size_t *ids = cidsT + (size_t)l * (size_t)T * (size_t)TS_K;
        float *w = cwT + (size_t)l * (size_t)T * (size_t)TS_K;
        float *Gsel =
            cgselT + (size_t)l * (size_t)T * (size_t)TS_K * (size_t)TS_H;
        float *Usel =
            cuselT + (size_t)l * (size_t)T * (size_t)TS_K * (size_t)TS_H;
        float *Ysel =
            cyselT + (size_t)l * (size_t)T * (size_t)TS_K * (size_t)d;
        float *Gs = cgsT + (size_t)l * (size_t)T * (size_t)TS_S * (size_t)TS_H;
        float *Us = cusT + (size_t)l * (size_t)T * (size_t)TS_S * (size_t)TS_H;
        size_t *perm = permT + (size_t)l * (size_t)T * (size_t)TS_K;
        size_t *off = offT + (size_t)l * ((size_t)TS_E + 1);
        unsigned char *dropm = dropT + (size_t)l * (size_t)T * (size_t)TS_K;
        size_t kept = 0;
        size_t dropped = 0;
        // jt_moe_fwd_batchはYbへMoE出力のみ書き込む。残差は後段で加算する。
        int rc = jt_moe_fwd_batch(Xb, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds, Yb,
                                  T, d, TS_H, TS_E, TS_K, TS_S, ids, w,
                                  Gsel, Usel, Ysel, Gs, Us, TS_CAP_FACTOR,
                                  perm, off, dropm, &kept, &dropped);
        if (rc != JT_OK) {
            return rc;
        }
        // 残差加算：acts[l+1] = acts[l] + MoE(Yb)
        {
            // jt_moe_fwd_batchは第9引数Mb...ではなくYbへ出力済み。
            // Ybは現時点でMoE出力のみ。残差を加算する。
            for (int t = 0; t < T; t++) {
                const float *xin = Xb + (size_t)t * (size_t)d;
                float *xout = Yb + (size_t)t * (size_t)d;
                for (int j = 0; j < d; j++) {
                    double v = (double)xin[j] + (double)xout[j];
                    if (!isfinite(v)) {
                        errno = EINVAL;
                        return JT_ERR_INVAL;
                    }
                    xout[j] = (float)v;
                }
            }
        }
        dropped_total += dropped;
        // renormalize発火：dropあり＆kept残存トークン数
        for (int t = 0; t < T; t++) {
            int nd = 0;
            int nk = 0;
            for (int p = 0; p < TS_K; p++) {
                size_t q = (size_t)t * (size_t)TS_K + (size_t)p;
                if (dropm[q]) {
                    nd++;
                } else {
                    nk++;
                }
            }
            if (nd > 0 && nk > 0) {
                renorm_total++;
            }
        }
    }
    if (out_dropped != NULL) {
        *out_dropped = dropped_total;
    }
    if (out_renorm != NULL) {
        *out_renorm = renorm_total;
    }
    return JT_OK;
}

// T一括逆伝播：単体版（トークン毎jt_moe_bwd）。m->Gへ加算（呼出し側で0埋め済み）。
// probsT[T][V]、dhT[T][d]・dX1[d]・dlog[V]・tW[stride]は作業域。
static int ts_bwd_T_single(ts_model_t *restrict m, const int *restrict TX,
                           const float *restrict actsT, int T,
                           const size_t *restrict cidsT,
                           const float *restrict cwT,
                           const float *restrict cgselT,
                           const float *restrict cuselT,
                           const float *restrict cyselT,
                           const float *restrict cgsT,
                           const float *restrict cusT,
                           const float *restrict probsT,
                           const int *restrict TY, float *restrict dlog,
                           float *restrict dhT, float *restrict dX1,
                           float *restrict tW, float dscale) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    float *G = m->G;
    const float *P = m->P;
    float *Gemb = G + m->lt.off_emb;
    float *GWo = G + m->lt.off_head_wo;
    float *Gbo = G + m->lt.off_head_bo;
    const float *Wo = P + m->lt.off_head_wo;
    if (m == NULL || TX == NULL || actsT == NULL || cidsT == NULL ||
        cwT == NULL || cgselT == NULL || cuselT == NULL || cyselT == NULL ||
        cgsT == NULL || cusT == NULL || probsT == NULL || TY == NULL ||
        dlog == NULL || dhT == NULL || dX1 == NULL || tW == NULL || T <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (int t = 0; t < T; t++) {
        const float *hl =
            actsT + ((size_t)L * (size_t)T + (size_t)t) * (size_t)d;
        const float *prow = probsT + (size_t)t * (size_t)V;
        float *dh = dhT + (size_t)t * (size_t)d;
        int y = TY[t];
        if (ts_ce_bwd(prow, V, y, dscale, dlog) != JT_OK) {
            return JT_ERR_INVAL;
        }
        for (int j = 0; j < d; j++) {
            dh[j] = 0.0f;
        }
        for (int v = 0; v < V; v++) {
            float g = dlog[v];
            Gbo[v] += g;
            for (int j = 0; j < d; j++) {
                GWo[(size_t)v * (size_t)d + (size_t)j] += g * hl[j];
                dh[j] += g * Wo[(size_t)v * (size_t)d + (size_t)j];
            }
        }
    }
    for (int l = L - 1; l >= 0; l--) {
        const float *base = P + m->lt.off_emb + (size_t)V * (size_t)d +
                            (size_t)l * m->lt.layer_stride;
        float *gbase = G + m->lt.off_emb + (size_t)V * (size_t)d +
                       (size_t)l * m->lt.layer_stride;
        const float *Wgate = base + m->lt.off_wgate;
        const float *Wg = base + m->lt.off_wg;
        const float *Wu = base + m->lt.off_wu;
        const float *Wd = base + m->lt.off_wd;
        const float *Wgs = base + m->lt.off_wgs;
        const float *Wus = base + m->lt.off_wus;
        const float *Wds = base + m->lt.off_wds;
        float *tWgate = tW + m->lt.off_wgate;
        float *tWg = tW + m->lt.off_wg;
        float *tWu = tW + m->lt.off_wu;
        float *tWd = tW + m->lt.off_wd;
        float *tWgs = tW + m->lt.off_wgs;
        float *tWus = tW + m->lt.off_wus;
        float *tWds = tW + m->lt.off_wds;
        for (int t = 0; t < T; t++) {
            const float *xin =
                actsT + ((size_t)l * (size_t)T + (size_t)t) * (size_t)d;
            float *dh = dhT + (size_t)t * (size_t)d;
            const size_t *ids =
                cidsT + ((size_t)l * (size_t)T + (size_t)t) * (size_t)TS_K;
            const float *w =
                cwT + ((size_t)l * (size_t)T + (size_t)t) * (size_t)TS_K;
            const float *Gsel =
                cgselT + ((size_t)l * (size_t)T + (size_t)t) *
                             (size_t)TS_K * (size_t)TS_H;
            const float *Usel =
                cuselT + ((size_t)l * (size_t)T + (size_t)t) *
                             (size_t)TS_K * (size_t)TS_H;
            const float *Ysel =
                cyselT +
                ((size_t)l * (size_t)T + (size_t)t) * (size_t)TS_K * (size_t)d;
            const float *Gs = cgsT + ((size_t)l * (size_t)T + (size_t)t) *
                                         (size_t)TS_S * (size_t)TS_H;
            const float *Us = cusT + ((size_t)l * (size_t)T + (size_t)t) *
                                         (size_t)TS_S * (size_t)TS_H;
            int rc = jt_moe_bwd(dh, xin, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds,
                                ids, w, Gsel, Usel, Ysel, Gs, Us, dX1,
                                tWgate, tWg, tWu, tWd, tWgs, tWus, tWds,
                                NULL, d, TS_H, TS_E, TS_K, TS_S);
            if (rc != JT_OK) {
                return rc;
            }
            for (size_t z = 0; z < m->lt.layer_stride; z++) {
                gbase[z] += tW[z];
            }
            for (int j = 0; j < d; j++) {
                dh[j] += dX1[j];
            }
        }
    }
    for (int t = 0; t < T; t++) {
        int x = TX[t];
        float *dh = dhT + (size_t)t * (size_t)d;
        for (int j = 0; j < d; j++) {
            Gemb[(size_t)x * (size_t)d + (size_t)j] += dh[j];
        }
    }
    return JT_OK;
}

// T一括逆伝播：バッチ版（層毎jt_moe_bwd_batch、同一perm再利用）。
static int ts_bwd_T_batch(ts_model_t *restrict m,
                          const int *restrict TX,
                          const float *restrict actsT, int T,
                          const size_t *restrict cidsT,
                          const float *restrict cwT,
                          const float *restrict cgselT,
                          const float *restrict cuselT,
                          const float *restrict cyselT,
                          const float *restrict cgsT,
                          const float *restrict cusT,
                          const size_t *restrict permT,
                          const size_t *restrict offT,
                          const unsigned char *restrict dropT,
                          const float *restrict probsT,
                          const int *restrict TY, float *restrict dlog,
                          float *restrict dhT, float *restrict dXb,
                          float dscale) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    float *G = m->G;
    const float *P = m->P;
    float *Gemb = G + m->lt.off_emb;
    float *GWo = G + m->lt.off_head_wo;
    float *Gbo = G + m->lt.off_head_bo;
    const float *Wo = P + m->lt.off_head_wo;
    if (m == NULL || TX == NULL || actsT == NULL || cidsT == NULL ||
        cwT == NULL || cgselT == NULL || cuselT == NULL || cyselT == NULL ||
        cgsT == NULL || cusT == NULL || permT == NULL || offT == NULL ||
        dropT == NULL || probsT == NULL || TY == NULL || dlog == NULL ||
        dhT == NULL || dXb == NULL || T <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (int t = 0; t < T; t++) {
        const float *hl =
            actsT + ((size_t)L * (size_t)T + (size_t)t) * (size_t)d;
        const float *prow = probsT + (size_t)t * (size_t)V;
        float *dh = dhT + (size_t)t * (size_t)d;
        int y = TY[t];
        if (ts_ce_bwd(prow, V, y, dscale, dlog) != JT_OK) {
            return JT_ERR_INVAL;
        }
        for (int j = 0; j < d; j++) {
            dh[j] = 0.0f;
        }
        for (int v = 0; v < V; v++) {
            float g = dlog[v];
            Gbo[v] += g;
            for (int j = 0; j < d; j++) {
                GWo[(size_t)v * (size_t)d + (size_t)j] += g * hl[j];
                dh[j] += g * Wo[(size_t)v * (size_t)d + (size_t)j];
            }
        }
    }
    for (int l = L - 1; l >= 0; l--) {
        const float *base = P + m->lt.off_emb + (size_t)V * (size_t)d +
                            (size_t)l * m->lt.layer_stride;
        float *gbase = G + m->lt.off_emb + (size_t)V * (size_t)d +
                       (size_t)l * m->lt.layer_stride;
        const float *Wgate = base + m->lt.off_wgate;
        const float *Wg = base + m->lt.off_wg;
        const float *Wu = base + m->lt.off_wu;
        const float *Wd = base + m->lt.off_wd;
        const float *Wgs = base + m->lt.off_wgs;
        const float *Wus = base + m->lt.off_wus;
        const float *Wds = base + m->lt.off_wds;
        const float *Xb = actsT + (size_t)l * (size_t)T * (size_t)d;
        const size_t *ids = cidsT + (size_t)l * (size_t)T * (size_t)TS_K;
        const float *w = cwT + (size_t)l * (size_t)T * (size_t)TS_K;
        const float *Gsel =
            cgselT + (size_t)l * (size_t)T * (size_t)TS_K * (size_t)TS_H;
        const float *Usel =
            cuselT + (size_t)l * (size_t)T * (size_t)TS_K * (size_t)TS_H;
        const float *Ysel =
            cyselT + (size_t)l * (size_t)T * (size_t)TS_K * (size_t)d;
        const float *Gs = cgsT + (size_t)l * (size_t)T * (size_t)TS_S * (size_t)TS_H;
        const float *Us = cusT + (size_t)l * (size_t)T * (size_t)TS_S * (size_t)TS_H;
        const size_t *perm = permT + (size_t)l * (size_t)T * (size_t)TS_K;
        const size_t *off = offT + (size_t)l * ((size_t)TS_E + 1);
        const unsigned char *dropm =
            dropT + (size_t)l * (size_t)T * (size_t)TS_K;
        float *gWgate = gbase + m->lt.off_wgate;
        float *gWg = gbase + m->lt.off_wg;
        float *gWu = gbase + m->lt.off_wu;
        float *gWd = gbase + m->lt.off_wd;
        float *gWgs = gbase + m->lt.off_wgs;
        float *gWus = gbase + m->lt.off_wus;
        float *gWds = gbase + m->lt.off_wds;
        int rc = jt_moe_bwd_batch(dhT, Xb, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds,
                                  ids, w, Gsel, Usel, Ysel, Gs, Us, perm,
                                  off, dropm, dXb, gWgate, gWg, gWu, gWd,
                                  gWgs, gWus, gWds, NULL, T, d, TS_H, TS_E,
                                  TS_K, TS_S);
        if (rc != JT_OK) {
            return rc;
        }
        for (int i = 0; i < T * d; i++) {
            dhT[i] += dXb[i];
        }
    }
    for (int t = 0; t < T; t++) {
        int x = TX[t];
        float *dh = dhT + (size_t)t * (size_t)d;
        for (int j = 0; j < d; j++) {
            Gemb[(size_t)x * (size_t)d + (size_t)j] += dh[j];
        }
    }
    return JT_OK;
}

// 層平均のL_aux＋エントロピー集計（droppedは重み0として評価）。
// aux_mean/ent_meanに層平均を返す。fail-closed。
static int ts_aux_stats(const size_t *restrict cidsT,
                        const float *restrict cwT,
                        const unsigned char *restrict dropT, int use_drop,
                        int L, int T, float *restrict aux_mean,
                        float *restrict ent_mean) {
    double sa = 0.0;
    double se = 0.0;
    if (cidsT == NULL || cwT == NULL || aux_mean == NULL ||
        ent_mean == NULL || L <= 0 || T <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (use_drop && dropT == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (int l = 0; l < L; l++) {
        const size_t *ids = cidsT + (size_t)l * (size_t)T * (size_t)TS_K;
        const float *w = cwT + (size_t)l * (size_t)T * (size_t)TS_K;
        float a = 0.0f;
        float e = 0.0f;
        int rc;
        if (!use_drop) {
            rc = jt_routing_balance_loss(ids, w, (size_t)T, (size_t)TS_K,
                                         (size_t)TS_E, &a, &e);
        } else {
            const unsigned char *dm =
                dropT + (size_t)l * (size_t)T * (size_t)TS_K;
            size_t Tk = (size_t)T * (size_t)TS_K;
            float *wf = (float *)malloc(Tk * sizeof(float));
            if (wf == NULL) {
                errno = ENOMEM;
                return JT_ERR_INVAL;
            }
            for (size_t q = 0; q < Tk; q++) {
                wf[q] = dm[q] ? 0.0f : w[q];
            }
            rc = jt_routing_balance_loss(ids, wf, (size_t)T, (size_t)TS_K,
                                         (size_t)TS_E, &a, &e);
            free(wf);
        }
        if (rc != JT_OK) {
            return rc;
        }
        sa += (double)a;
        se += (double)e;
    }
    *aux_mean = (float)(sa / (double)L);
    *ent_mean = (float)(se / (double)L);
    return JT_OK;
}

// L_aux勾配のWgateへの加算（f定数近似。dL/dw=μ*E*f_e/(T*K)経由のヤコビアン）。
// m->GのWgate部へ加算する。fail-closed。
static int ts_aux_grad_apply(ts_model_t *restrict m,
                             const float *restrict actsT,
                             const size_t *restrict cidsT,
                             const float *restrict cwT,
                             const unsigned char *restrict dropT,
                             int use_drop, int T, float aux_w) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    int V = m->lt.v;
    float *G = m->G;
    if (m == NULL || actsT == NULL || cidsT == NULL || cwT == NULL || T <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (!(aux_w >= 0.0f) || !isfinite((double)aux_w)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (aux_w == 0.0f) {
        return JT_OK;
    }
    if (use_drop && dropT == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (int l = 0; l < L; l++) {
        const size_t *ids = cidsT + (size_t)l * (size_t)T * (size_t)TS_K;
        const float *w = cwT + (size_t)l * (size_t)T * (size_t)TS_K;
        const unsigned char *dm =
            use_drop ? (dropT + (size_t)l * (size_t)T * (size_t)TS_K) : NULL;
        size_t cnt[TS_E];
        double f[TS_E];
        size_t Tk = (size_t)T * (size_t)TS_K;
        float *gbase = G + m->lt.off_emb + (size_t)V * (size_t)d +
                       (size_t)l * m->lt.layer_stride;
        float *gWgate = gbase + m->lt.off_wgate;
        for (int e = 0; e < TS_E; e++) {
            cnt[e] = 0;
        }
        for (size_t q = 0; q < Tk; q++) {
            if (dm != NULL && dm[q]) {
                continue;
            }
            if (ids[q] < (size_t)TS_E) {
                cnt[ids[q]]++;
            }
        }
        for (int e = 0; e < TS_E; e++) {
            f[e] = (double)cnt[e] / (double)Tk;
        }
        for (int t = 0; t < T; t++) {
            const float *xin =
                actsT + ((size_t)l * (size_t)T + (size_t)t) * (size_t)d;
            double gval[TS_K];
            double s = 0.0;
            for (int p = 0; p < TS_K; p++) {
                size_t q = (size_t)t * (size_t)TS_K + (size_t)p;
                size_t e = ids[q];
                if (dm != NULL && dm[q]) {
                    gval[p] = 0.0;
                    continue;
                }
                if (e >= (size_t)TS_E) {
                    errno = EINVAL;
                    return JT_ERR_INVAL;
                }
                // 合計G基準（CEは合計のため）に合わせ、平均損失の勾配にTを掛ける：
                // g = T*μ*E*f_e/(T*K) = μ*E*f_e/K。
                gval[p] =
                    (double)aux_w * (double)TS_E * f[e] / (double)TS_K;
            }
            for (int p = 0; p < TS_K; p++) {
                size_t q = (size_t)t * (size_t)TS_K + (size_t)p;
                if (dm != NULL && dm[q]) {
                    continue;
                }
                s += (double)w[q] * gval[p];
            }
            for (int p = 0; p < TS_K; p++) {
                size_t q = (size_t)t * (size_t)TS_K + (size_t)p;
                size_t e = ids[q];
                double dl;
                if (dm != NULL && dm[q]) {
                    continue;
                }
                dl = (double)w[q] * (gval[p] - s);
                if (!isfinite(dl)) {
                    errno = EINVAL;
                    return JT_ERR_INVAL;
                }
                for (int j = 0; j < d; j++) {
                    gWgate[e * (size_t)d + (size_t)j] +=
                        (float)(dl * (double)xin[j]);
                }
            }
        }
    }
    return JT_OK;
}

// ---- テキスト読み: <|endoftext|>区切りの物語を [BOS, bytes..., EOS] に変換 ----
// seqs_out/lens_outは呼び出し側でfree (各行malloc)。戻り値は物語数。失敗時-1。
// max_stories>0時は先頭max_stories話のみ (決定論的)。
// V=258専用 (V=258経路は不変)。V=48588はts_read_text_bpeを使用。
static long ts_read_text(const char *restrict path, int vocab,
                         uint32_t ***restrict seqs_out,
                         size_t **restrict lens_out, long max_stories,
                         size_t *restrict nbytes_out) {
    FILE *f = NULL;
    uint8_t *buf = NULL;
    long fsz = 0;
    uint32_t **seqs = NULL;
    size_t *lens = NULL;
    size_t n = 0;
    size_t cap = 0;
    size_t start = 0;
    size_t total = 0;
    if (path == NULL || seqs_out == NULL || lens_out == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (vocab != TS_VOCAB_SMALL) {
        errno = EINVAL;
        return -1;
    }
    f = fopen(path, "rb");
    if (f == NULL) {
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (fsz = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        goto fail;
    }
    if (fsz == 0 || fsz > 512L * 1024L * 1024L) {
        goto fail;
    }
    buf = (uint8_t *)malloc((size_t)fsz);
    if (buf == NULL) {
        goto fail;
    }
    if (fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) {
        goto fail;
    }
    fclose(f);
    f = NULL;
    total = (size_t)fsz;
    while (start < total) {
        size_t end = start;
        size_t s0 = start; /* 物語先頭 (前進前に保存) */
        size_t blen = 0;
        uint32_t *sq = NULL;
        size_t p = 0;
        /* 次の区切りを探索。末尾に区切りが無くても最終話として採用 */
        while (end + TS_DELIM_LEN <= total &&
               memcmp(buf + end, TS_DELIM, TS_DELIM_LEN) != 0) {
            end++;
        }
        if (end + TS_DELIM_LEN <= total) {
            blen = end - start;
            end += TS_DELIM_LEN;
        } else {
            blen = total - start;
            end = total;
        }
        start = end;
        if (blen == 0) {
            continue; /* 空断片は捨てる */
        }
        if (max_stories > 0 && n >= (size_t)max_stories) {
            break;
        }
        sq = (uint32_t *)malloc((blen + 2) * sizeof(uint32_t));
        if (sq == NULL) {
            goto fail;
        }
        sq[0] = (uint32_t)TS_BOS;
        for (p = 0; p < blen; p++) {
            sq[1 + p] = (uint32_t)(TS_BYTE_OFF + buf[s0 + p]);
        }
        sq[blen + 1] = (uint32_t)TS_EOS;
        if (n == cap) {
            size_t ncap = (cap == 0) ? 4096 : cap * 2;
            uint32_t **ns = NULL;
            size_t *nl = NULL;
            if (ncap > (size_t)4000000) {
                free(sq);
                goto fail;
            }
            ns = (uint32_t **)realloc(seqs, ncap * sizeof(uint32_t *));
            nl = (size_t *)realloc(lens, ncap * sizeof(size_t));
            if (ns == NULL || nl == NULL) {
                if (ns != NULL && ns != seqs) {
                    seqs = ns;
                }
                if (nl != NULL && nl != lens) {
                    lens = nl;
                }
                free(sq);
                goto fail;
            }
            seqs = ns;
            lens = nl;
            cap = ncap;
        }
        seqs[n] = sq;
        lens[n] = blen + 2;
        n++;
    }
    free(buf);
    if (nbytes_out != NULL) {
        *nbytes_out = total;
    }
    *seqs_out = seqs;
    *lens_out = lens;
    return (long)n;
fail: {
    size_t i = 0;
    int se = errno;
    if (f != NULL) {
        fclose(f);
    }
    free(buf);
    for (i = 0; i < n; i++) {
        free(seqs[i]);
    }
    free(seqs);
    free(lens);
    errno = (se != 0) ? se : ENOMEM;
    return -1;
}
}

// ---- ファイルサイズ (CE_per_byte分母用。失敗時0) ----
static size_t ts_file_size(const char *restrict path) {
    FILE *f = NULL;
    long n = 0;
    if (path == NULL) {
        return 0;
    }
    f = fopen(path, "rb");
    if (f == NULL) {
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (n = ftell(f)) < 0) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return (size_t)n;
}

// ---- BPEテキスト読み (V=48588専用): 物語バイト列をjt_bpe_encodeで符号化 ----
// 区切り探索・max_stories先頭制限はts_read_textと同一 (決定論的)。
// 系列は生BPE ID列 (BOS/EOSなし)。符号化失敗の物語は数えて捨てる
// (skipped_out、NULL可。TinyStories英語ASCIIでは0になる想定)。
// seqs_out/lens_outは呼び出し側でfree。戻り値は物語数。失敗時-1。
static long ts_read_text_bpe(const char *restrict path,
                             const char *restrict bpe_vocab_path,
                             uint32_t ***restrict seqs_out,
                             size_t **restrict lens_out, long max_stories,
                             size_t *restrict nbytes_out,
                             long *restrict skipped_out) {
    FILE *f = NULL;
    uint8_t *buf = NULL;
    long fsz = 0;
    jt_bpe_t bpe;
    uint32_t **seqs = NULL;
    size_t *lens = NULL;
    size_t n = 0;
    size_t cap = 0;
    size_t start = 0;
    size_t total = 0;
    long skipped = 0;
    memset(&bpe, 0, sizeof(bpe));
    if (path == NULL || bpe_vocab_path == NULL || seqs_out == NULL ||
        lens_out == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (jt_bpe_open(&bpe, bpe_vocab_path) != JT_OK) {
        return -1; /* errnoはbpe側で設定済み */
    }
    f = fopen(path, "rb");
    if (f == NULL) {
        jt_bpe_close(&bpe);
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (fsz = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        goto fail;
    }
    if (fsz == 0 || fsz > 512L * 1024L * 1024L) {
        goto fail;
    }
    buf = (uint8_t *)malloc((size_t)fsz);
    if (buf == NULL) {
        goto fail;
    }
    if (fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) {
        goto fail;
    }
    fclose(f);
    f = NULL;
    total = (size_t)fsz;
    while (start < total) {
        size_t end = start;
        size_t s0 = start;
        size_t blen = 0;
        uint32_t *tmp = NULL;
        uint32_t *sq = NULL;
        size_t cap_tok = 0;
        size_t ntok = 0;
        int erc = 0;
        while (end + TS_DELIM_LEN <= total &&
               memcmp(buf + end, TS_DELIM, TS_DELIM_LEN) != 0) {
            end++;
        }
        if (end + TS_DELIM_LEN <= total) {
            blen = end - start;
            end += TS_DELIM_LEN;
        } else {
            blen = total - start;
            end = total;
        }
        start = end;
        if (blen == 0) {
            continue;
        }
        if (max_stories > 0 && n >= (size_t)max_stories) {
            break;
        }
        /* トークン上限は正規化後バイト長以下。blen*3+8 (オーバーフロー守衛) */
        if (blen > (SIZE_MAX - 8u) / 3u) {
            goto fail;
        }
        cap_tok = blen * 3u + 8u;
        if (cap_tok / sizeof(uint32_t) > SIZE_MAX / sizeof(uint32_t)) {
            goto fail;
        }
        tmp = (uint32_t *)malloc(cap_tok * sizeof(uint32_t));
        if (tmp == NULL) {
            goto fail;
        }
        erc = jt_bpe_encode(&bpe, (const char *)(buf + s0), blen, tmp,
                            cap_tok, &ntok);
        if (erc != JT_OK) {
            free(tmp);
            skipped++; /* 不正UTF-8等は当該話のみ捨てる (件数を報告) */
            continue;
        }
        sq = (uint32_t *)malloc((ntok == 0 ? 1 : ntok) * sizeof(uint32_t));
        if (sq == NULL) {
            free(tmp);
            goto fail;
        }
        if (ntok > 0) {
            memcpy(sq, tmp, ntok * sizeof(uint32_t));
        }
        free(tmp);
        if (n == cap) {
            size_t ncap = (cap == 0) ? 4096 : cap * 2;
            uint32_t **ns = NULL;
            size_t *nl = NULL;
            if (ncap > (size_t)4000000) {
                free(sq);
                goto fail;
            }
            ns = (uint32_t **)realloc(seqs, ncap * sizeof(uint32_t *));
            nl = (size_t *)realloc(lens, ncap * sizeof(size_t));
            if (ns == NULL || nl == NULL) {
                if (ns != NULL && ns != seqs) {
                    seqs = ns;
                }
                if (nl != NULL && nl != lens) {
                    lens = nl;
                }
                free(sq);
                goto fail;
            }
            seqs = ns;
            lens = nl;
            cap = ncap;
        }
        seqs[n] = sq;
        lens[n] = ntok;
        n++;
    }
    free(buf);
    jt_bpe_close(&bpe);
    if (nbytes_out != NULL) {
        *nbytes_out = total;
    }
    if (skipped_out != NULL) {
        *skipped_out = skipped;
    }
    *seqs_out = seqs;
    *lens_out = lens;
    return (long)n;
fail: {
    size_t i = 0;
    int se = errno;
    if (f != NULL) {
        fclose(f);
    }
    free(buf);
    for (i = 0; i < n; i++) {
        free(seqs[i]);
    }
    free(seqs);
    free(lens);
    jt_bpe_close(&bpe);
    errno = (se != 0) ? se : ENOMEM;
    return -1;
}
}

// .jtdp読み (data_pack正規経路)。ID>=vocabは拒否 (fail-closed)。
// V=258のbyte-fallback IDは最大257のためjt_dp_openの<48588検証も通過する。
static long ts_read_pack(const char *restrict path, int vocab,
                         uint32_t ***restrict seqs_out,
                         size_t **restrict lens_out) {
    jt_dp_reader_t r;
    uint64_t nseq = 0;
    uint32_t **seqs = NULL;
    size_t *lens = NULL;
    uint64_t i = 0;
    memset(&r, 0, sizeof(r));
    if (path == NULL || seqs_out == NULL || lens_out == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (vocab != TS_VOCAB_SMALL && vocab != TS_VOCAB_LLJ) {
        errno = EINVAL;
        return -1;
    }
    if (jt_dp_open(path, &r) != JT_OK) {
        return -1;
    }
    if (jt_dp_nseq(&r, &nseq) != JT_OK || nseq == 0 || nseq > 4000000ULL) {
        jt_dp_close(&r);
        errno = EINVAL;
        return -1;
    }
    seqs = (uint32_t **)malloc((size_t)nseq * sizeof(uint32_t *));
    lens = (size_t *)malloc((size_t)nseq * sizeof(size_t));
    if (seqs == NULL || lens == NULL) {
        free(seqs);
        free(lens);
        jt_dp_close(&r);
        errno = ENOMEM;
        return -1;
    }
    for (i = 0; i < nseq; i++) {
        seqs[i] = NULL;
        lens[i] = 0;
    }
    for (i = 0; i < nseq; i++) {
        const uint32_t *ptr = NULL;
        uint64_t slen = 0;
        uint32_t *cp = NULL;
        uint64_t k = 0;
        if (jt_dp_seq_ptr(&r, i, &ptr, &slen) != JT_OK || slen == 0 ||
            slen > 1048576ULL || ptr == NULL) {
            goto fail;
        }
        cp = (uint32_t *)malloc((size_t)slen * sizeof(uint32_t));
        if (cp == NULL) {
            goto fail;
        }
        for (k = 0; k < slen; k++) {
            if ((int)ptr[k] < 0 || ptr[k] >= (uint32_t)vocab) {
                free(cp);
                goto fail; /* 本語彙外IDは拒否 */
            }
            cp[k] = ptr[k];
        }
        seqs[i] = cp;
        lens[i] = (size_t)slen;
    }
    jt_dp_close(&r);
    *seqs_out = seqs;
    *lens_out = lens;
    return (long)nseq;
fail: {
    uint64_t j = 0;
    int se = errno;
    for (j = 0; j < nseq; j++) {
        free(seqs[j]);
    }
    free(seqs);
    free(lens);
    jt_dp_close(&r);
    errno = (se != 0) ? se : EINVAL;
    return -1;
}
}

static void ts_free_seqs(uint32_t **seqs, size_t *lens, long n) {
    long i = 0;
    (void)lens;
    if (seqs == NULL) {
        return;
    }
    for (i = 0; i < n; i++) {
        free(seqs[i]);
    }
    free(seqs);
    free(lens);
}

// val評価: cap<=0またはcap>=nで全件、そうでなければ等間隔stride抽出 (決定論的)。
static int ts_eval_val(const ts_model_t *restrict m, const int *restrict vx,
                       const int *restrict vy, long n_val, long cap,
                       float *restrict acts, size_t *restrict cids,
                       float *restrict cw, float *restrict cgsel,
                       float *restrict cusel, float *restrict cysel,
                       float *restrict cgs, float *restrict cus,
                       float *restrict logits, float *restrict probs,
                       double *restrict out_ce) {
    long cnt = 0;
    long i = 0;
    double sv = 0.0;
    long n = 0;
    if (n_val <= 0 || out_ce == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    n = (cap > 0 && cap < n_val) ? cap : n_val;
    for (i = 0; i < n; i++) {
        long idx = (cap > 0 && cap < n_val) ? (long)((uint64_t)i *
                                                    (uint64_t)n_val /
                                                    (uint64_t)n)
                                            : i;
        float loss = 0.0f;
        int rc = ts_fwd_one(m, vx[idx], acts, cids, cw, cgsel, cusel, cysel,
                            cgs, cus, logits, probs, &loss, vy[idx]);
        if (rc != JT_OK) {
            return rc;
        }
        sv += (double)loss;
        cnt++;
    }
    *out_ce = sv / (double)cnt;
    return JT_OK;
}

int main(int argc, char **argv) {
    int rc_all = 1;
    const char *data_path = "data/tinystories_head16M.txt";
    const char *pack_path = "data/tinystories16M.jtdp";
    const char *write_pack = NULL;
    int use_pack_opt = 0;
    long max_steps = 500;
    double time_limit = 3600.0;
    float lr = 3e-4f;  //既定: 1e-3は1645 stepで発散実績のため3e-4
    long batch = 64;
    int d = 64;
    int n_layers = 2;
    long val_every = 100;
    long patience = 100; /* steps単位。0=無効 */
    int vocab = TS_VOCAB_SMALL;
    const char *bpe_vocab_path = TS_BPE_VOCAB_DEFAULT;
    long max_stories = 0; /* 0=全話 */
    long max_val_pairs = 20000; /* 0=全val */
    uint32_t **seqs = NULL;
    size_t *lens = NULL;
    long n_stories = 0;
    long n_train_stories = 0;
    size_t nbytes = 0;
    int *tx = NULL;
    int *ty = NULL;
    long n_train_pairs = 0;
    int *vx = NULL;
    int *vy = NULL;
    long n_val_pairs = 0;
    const char *data_source = "?";
    ts_model_t m;
    float *acts = NULL;
    size_t *cids = NULL;
    float *cw = NULL;
    float *cgsel = NULL;
    float *cusel = NULL;
    float *cysel = NULL;
    float *cgs = NULL;
    float *cus = NULL;
    float *logits = NULL;
    float *probs = NULL;
    float *dlog = NULL;
    float *dh = NULL;
    float *dX = NULL;
    float *tW = NULL;
    size_t *bperm = NULL;
    size_t *boff = NULL;
    unsigned char *bdrop = NULL;
    // G2fix T一括用（seq>1時のみ確保）
    int Tstep = 0;
    int *TX = NULL;
    int *TY = NULL;
    float *actsT = NULL;
    size_t *cidsT = NULL;
    float *cwT = NULL;
    float *cgselT = NULL;
    float *cuselT = NULL;
    float *cyselT = NULL;
    float *cgsT = NULL;
    float *cusT = NULL;
    size_t *permT = NULL;
    size_t *offT = NULL;
    unsigned char *dropT = NULL;
    float *probsT = NULL;
    float *dhT = NULL;
    float *dXb = NULL;
    // legacy seq==1用ステップaux集計域
    size_t *step_ids = NULL;
    float *step_w = NULL;
    double t0 = 0.0;
    long step = 0;
    long steps_done = 0;
    double tok_per_byte = 0.0; /* (train+valペア数)/テキストバイト数 (当該V実測) */
    double init_train_ce = 0.0;
    double init_val_ce = 0.0;
    double last_train_ce = 0.0;
    double last_val_ce = 0.0;
    double best_val_ce = 0.0;
    long best_val_step = 0;
    double prev_vals[3];
    int n_prev = 0;
    int stopped_early = 0;
    const char *stop_reason = "-";
    int i = 0;

    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    memset(&m, 0, sizeof(m));
    prev_vals[0] = prev_vals[1] = prev_vals[2] = 0.0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_path = argv[++i];
        } else if (strcmp(argv[i], "--pack") == 0 && i + 1 < argc) {
            pack_path = argv[++i];
            use_pack_opt = 1;
        } else if (strcmp(argv[i], "--write-pack") == 0 && i + 1 < argc) {
            write_pack = argv[++i];
        } else if ((strcmp(argv[i], "--steps") == 0 ||
                    strcmp(argv[i], "--max-steps") == 0) &&
                   i + 1 < argc) {
            max_steps = atol(argv[++i]);
        } else if ((strcmp(argv[i], "--time") == 0 ||
                    strcmp(argv[i], "--time-limit") == 0) &&
                   i + 1 < argc) {
            time_limit = atof(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch = atol(argv[++i]);
        } else if (strcmp(argv[i], "--d") == 0 && i + 1 < argc) {
            d = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            n_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--val-every") == 0 && i + 1 < argc) {
            val_every = atol(argv[++i]);
        } else if (strcmp(argv[i], "--patience") == 0 && i + 1 < argc) {
            patience = atol(argv[++i]);
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--bpe-vocab") == 0 && i + 1 < argc) {
            bpe_vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--max-stories") == 0 && i + 1 < argc) {
            max_stories = atol(argv[++i]);
        } else if (strcmp(argv[i], "--max-val-pairs") == 0 && i + 1 < argc) {
            max_val_pairs = atol(argv[++i]);
        } else if (strcmp(argv[i], "--moe-batch") == 0) {
            g_ts_moe_batch = 1;
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            g_ts_seq = atol(argv[++i]);
        } else if (strcmp(argv[i], "--aux-weight") == 0 && i + 1 < argc) {
            g_ts_aux_w = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 ||
                   strcmp(argv[i], "--help") == 0) {
            ts_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "train_tinystories: unknown arg '%s'\n", argv[i]);
            ts_usage(argv[0]);
            errno = EINVAL;
            goto cleanup;
        }
    }
    if (vocab == 50257) {
        fprintf(stderr,
                "train_tinystories: --vocab 50257 is retired (JT_DP_VOCAB_SIZE "
                "exceeded); use --vocab 48588 --bpe-vocab <jtvocab>\n");
        errno = EINVAL;
        goto cleanup;
    }
    if (max_steps <= 0 || !(time_limit > 0.0) || !(lr > 0.0f) ||
        !isfinite(lr) || batch <= 0 || batch > 4096 || d < 8 || d > 256 ||
        n_layers < 1 || n_layers > 4 || val_every <= 0 || patience < 0 ||
        (vocab != TS_VOCAB_SMALL && vocab != TS_VOCAB_LLJ) ||
        max_stories < 0 || max_val_pairs < 0 || g_ts_seq < 1 ||
        g_ts_seq > 512 || !(g_ts_aux_w >= 0.0f) ||
        !(g_ts_aux_w <= 0.1f) || !isfinite((double)g_ts_aux_w)) {
        fprintf(stderr, "train_tinystories: invalid limits/args\n");
        errno = EINVAL;
        goto cleanup;
    }
    if (g_ts_seq * batch > 1024) {
        fprintf(stderr, "train_tinystories: seq*batch exceeds 1024\n");
        errno = EINVAL;
        goto cleanup;
    }

    /* ---- --write-pack: テキスト -> .jtdp変換のみ ---- */
    if (write_pack != NULL) {
        jt_dp_writer_t w;
        long wn = 0;
        long k = 0;
        long skipped = 0;
        memset(&w, 0, sizeof(w));
        if (vocab == TS_VOCAB_LLJ) {
            n_stories = ts_read_text_bpe(data_path, bpe_vocab_path, &seqs,
                                         &lens, max_stories, &nbytes,
                                         &skipped);
        } else {
            n_stories = ts_read_text(data_path, vocab, &seqs, &lens,
                                     max_stories, &nbytes);
        }
        if (n_stories <= 0) {
            fprintf(stderr, "train_tinystories: cannot read '%s': %s\n",
                    data_path, strerror(errno));
            goto cleanup;
        }
        if (jt_dp_writer_open(&w, write_pack) != JT_OK) {
            fprintf(stderr, "train_tinystories: cannot open '%s': %s\n",
                    write_pack, strerror(errno));
            goto cleanup;
        }
        wn = n_stories;
        for (k = 0; k < wn; k++) {
            if (jt_dp_writer_add(&w, seqs[k], lens[k]) != JT_OK) {
                fprintf(stderr,
                        "train_tinystories: writer_add failed at %ld: %s\n",
                        k, strerror(errno));
                goto cleanup;
            }
        }
        if (jt_dp_writer_close(&w) != JT_OK) {
            fprintf(stderr, "train_tinystories: writer_close failed: %s\n",
                    strerror(errno));
            goto cleanup;
        }
        printf("train_ts: write-pack nseq=%ld bytes=%zu skipped=%ld source=%s "
               "dest=%s vocab=%d\n",
               wn, nbytes, skipped, data_path, write_pack, vocab);
        ts_free_seqs(seqs, lens, n_stories);
        return 0;
    }

    /* ---- データ読み: data_pack正規経路を優先、不可時は直接読み ---- */
    n_stories = ts_read_pack(pack_path, vocab, &seqs, &lens);
    if (n_stories > 0) {
        data_source = "pack";
        /* pack内にバイト数記録なしのため、分母は--data実ファイルから参照 */
        nbytes = ts_file_size(data_path);
    } else if (use_pack_opt) {
        fprintf(stderr, "train_tinystories: cannot open pack '%s': %s\n",
                pack_path, strerror(errno));
        goto cleanup;
    } else {
        int se = errno;
        long skipped = 0;
        if (vocab == TS_VOCAB_LLJ) {
            n_stories = ts_read_text_bpe(data_path, bpe_vocab_path, &seqs,
                                         &lens, max_stories, &nbytes,
                                         &skipped);
        } else {
            n_stories = ts_read_text(data_path, vocab, &seqs, &lens,
                                     max_stories, &nbytes);
        }
        if (n_stories <= 0) {
            fprintf(stderr, "train_tinystories: cannot read '%s': %s\n",
                    data_path, strerror(errno));
            goto cleanup;
        }
        data_source = "direct(txt)";
        if (vocab == TS_VOCAB_LLJ) {
            printf("train_ts: bpe encode stories=%ld skipped=%ld "
                   "vocab_file=%s\n",
                   n_stories, skipped, bpe_vocab_path);
        }
        (void)se;
    }

    /* ---- 物語単位で分割: i%10==9をval、残りをtrain (決定論的) ---- */
    n_train_stories = n_stories - (n_stories + 9L) / 10L;
    if (n_train_stories < 1 || n_stories - n_train_stories < 1) {
        fprintf(stderr, "train_tinystories: too few stories (%ld)\n",
                n_stories);
        errno = EINVAL;
        goto cleanup;
    }
    {
        long k = 0;
        long cap_t = 0;
        long cap_v = 0;
        long t = 0;
        long q = 0;
        for (k = 0; k < n_stories; k++) {
            if (lens[k] > 1048576) {
                fprintf(stderr, "train_tinystories: story %ld too long\n",
                        k);
                errno = EINVAL;
                goto cleanup;
            }
            if (k % 10 == 9) {
                cap_v += (long)lens[k];
            } else {
                cap_t += (long)lens[k];
            }
        }
        if (cap_t <= 0 || cap_v <= 0 || cap_t > (1L << 26) ||
            cap_v > (1L << 26)) {
            fprintf(stderr, "train_tinystories: pair count out of range\n");
            errno = EINVAL;
            goto cleanup;
        }
        tx = (int *)malloc((size_t)cap_t * sizeof(int));
        ty = (int *)malloc((size_t)cap_t * sizeof(int));
        vx = (int *)malloc((size_t)cap_v * sizeof(int));
        vy = (int *)malloc((size_t)cap_v * sizeof(int));
        if (tx == NULL || ty == NULL || vx == NULL || vy == NULL) {
            fprintf(stderr, "train_tinystories: OOM\n");
            errno = ENOMEM;
            goto cleanup;
        }
        for (k = 0; k < n_stories; k++) {
            size_t s = 0;
            if (k % 10 == 9) {
                for (s = 0; s + 1 < lens[k]; s++) {
                    vx[q] = (int)seqs[k][s];
                    vy[q] = (int)seqs[k][s + 1];
                    q++;
                }
            } else {
                for (s = 0; s + 1 < lens[k]; s++) {
                    tx[t] = (int)seqs[k][s];
                    ty[t] = (int)seqs[k][s + 1];
                    t++;
                }
            }
        }
        n_train_pairs = t;
        n_val_pairs = q;
        n_train_stories = 0;
        for (k = 0; k < n_stories; k++) {
            if (k % 10 != 9) {
                n_train_stories++;
            }
        }
    }
    printf("train_ts: stories=%ld train_stories=%ld val_stories=%ld "
           "train_pairs=%ld val_pairs=%ld vocab=%d source=%s\n",
           n_stories, n_train_stories, n_stories - n_train_stories,
           n_train_pairs, n_val_pairs, vocab, data_source);
    /* CE_per_byte正規化の分母 (RULE.MD)。pack時は--data参照値 (0ならn/a) */
    if (nbytes > 0) {
        tok_per_byte =
            (double)(n_train_pairs + n_val_pairs) / (double)nbytes;
    }
    printf("train_ts: text_bytes=%zu tok_per_byte=%.6f (vocab=%d%s)\n",
           nbytes, tok_per_byte, vocab,
           strcmp(data_source, "pack") == 0
               ? "; pack assumes pack content == --data file"
               : "");
    if (strcmp(data_source, "direct(txt)") == 0) {
        printf("train_ts: note: pack '%s' unavailable; using direct "
               "txt read (fallback). Run --write-pack to enable data_pack "
               "route.\n",
               pack_path);
    }
    fflush(stdout);
    ts_free_seqs(seqs, lens, n_stories);
    seqs = NULL;
    lens = NULL;

    /* ---- モデル確保・初期化 (train_namesと同流儀の決定論的初期化) ---- */
    ts_build_layout(&m.lt, n_layers, d, vocab);
    m.P = (float *)malloc(m.lt.n_total * sizeof(float));
    m.G = (float *)malloc(m.lt.n_total * sizeof(float));
    if (m.P == NULL || m.G == NULL) {
        fprintf(stderr, "train_tinystories: OOM\n");
        errno = ENOMEM;
        goto cleanup;
    }
    {
        size_t n = m.lt.n_total;
        size_t p = 0;
        size_t emb_n = (size_t)vocab * (size_t)d;
        uint64_t key = 0x1234ULL;
        for (p = 0; p < n; p++) {
            double u = (double)(ts_hash64(key * 2ULL + 0x9e3779b9ULL +
                                         (uint64_t)p) >>
                                11) *
                       (1.0 / 9007199254740992.0);
            double sc = 0.125;
            if (p < emb_n) {
                sc = 1.0;
            } else if (p < emb_n + (size_t)n_layers * m.lt.layer_stride) {
                size_t rel = (p - emb_n) % m.lt.layer_stride;
                if (rel < (size_t)TS_E * (size_t)d) {
                    sc = 0.05;
                } else {
                    sc = 0.125;
                }
            } else {
                sc = 0.125;
            }
            m.P[p] = (float)((u * 2.0 - 1.0) * sc);
        }
        for (p = 0; p < (size_t)vocab; p++) {
            m.P[m.lt.off_head_bo + p] = 0.0f;
        }
    }
    {
        jt_optim8_cfg_t cfg;
        jt_optim8_cfg_default(&cfg);
        cfg.lr = lr;
        cfg.eps = 1e-4f;
        if (jt_optim8_init(&m.opt, m.lt.n_total, &cfg) != JT_OK) {
            fprintf(stderr, "train_tinystories: optim init failed\n");
            goto cleanup;
        }
        m.opt_inited = 1;
    }
    printf("train_ts: layers=%d d=%d E=%d K=%d S=%d H=%d V=%d "
           "params=%zu lr=%.5f batch=%ld seq=%ld T=%ld patience=%ld "
           "moe_batch=%d cap=%.2f aux_w=%.4f\n",
           n_layers, d, TS_E, TS_K, TS_S, TS_H, vocab, m.lt.n_total, lr,
           batch, g_ts_seq, g_ts_seq * batch, patience, g_ts_moe_batch,
           (double)TS_CAP_FACTOR, (double)g_ts_aux_w);
    /* 常駐見積り (d=64維持。V=48588でemb/head各64*48588*4B≈12.4MB。許容内) */
    {
        size_t n = m.lt.n_total;
        size_t nblocks = (n + 63u) / 64u;
        double p_mb = (double)n * 4.0 / 1048576.0;
        double g_mb = (double)n * 4.0 / 1048576.0;
        double o_mb =
            ((double)n * 2.0 + (double)nblocks * 8.0) / 1048576.0;
        double emb_mb =
            (double)(size_t)vocab * (size_t)d * 4.0 / 1048576.0;
        printf("train_ts: mem_est P=%.1fMB G=%.1fMB optim8=%.1fMB "
               "total=%.1fMB (emb=%.1fMB head=%.1fMB)\n",
               p_mb, g_mb, o_mb, p_mb + g_mb + o_mb, emb_mb, emb_mb);
        fflush(stdout);
    }
    acts =
        (float *)malloc((size_t)(n_layers + 1) * (size_t)d * sizeof(float));
    cids = (size_t *)malloc((size_t)n_layers * (size_t)TS_K * sizeof(size_t));
    cw = (float *)malloc((size_t)n_layers * (size_t)TS_K * sizeof(float));
    cgsel =
        (float *)malloc((size_t)n_layers * (size_t)TS_K * (size_t)TS_H *
                        sizeof(float));
    cusel =
        (float *)malloc((size_t)n_layers * (size_t)TS_K * (size_t)TS_H *
                        sizeof(float));
    cysel = (float *)malloc((size_t)n_layers * (size_t)TS_K * (size_t)d *
                            sizeof(float));
    cgs = (float *)malloc((size_t)n_layers * (size_t)TS_S * (size_t)TS_H *
                          sizeof(float));
    cus = (float *)malloc((size_t)n_layers * (size_t)TS_S * (size_t)TS_H *
                          sizeof(float));
    logits = (float *)malloc((size_t)vocab * sizeof(float));
    probs = (float *)malloc((size_t)vocab * sizeof(float));
    dlog = (float *)malloc((size_t)vocab * sizeof(float));
    dh = (float *)malloc((size_t)d * sizeof(float));
    dX = (float *)malloc((size_t)d * sizeof(float));
    tW = (float *)malloc(m.lt.layer_stride * sizeof(float));
    if (g_ts_moe_batch) {
        bperm = (size_t *)malloc((size_t)n_layers * (size_t)TS_K *
                                 sizeof(size_t));
        boff = (size_t *)malloc((size_t)n_layers * ((size_t)TS_E + 1) *
                                sizeof(size_t));
        bdrop = (unsigned char *)malloc((size_t)n_layers * (size_t)TS_K *
                                        sizeof(unsigned char));
    }
    if (acts == NULL || cids == NULL || cw == NULL || cgsel == NULL ||
        cusel == NULL || cysel == NULL || cgs == NULL || cus == NULL ||
        logits == NULL || probs == NULL || dlog == NULL || dh == NULL ||
        dX == NULL || tW == NULL ||
        (g_ts_moe_batch && (bperm == NULL || boff == NULL || bdrop == NULL))) {
        fprintf(stderr, "train_tinystories: OOM\n");
        errno = ENOMEM;
        goto cleanup;
    }
    Tstep = (int)(g_ts_seq * batch);
    if (g_ts_seq > 1) {
        size_t Tt = (size_t)Tstep;
        size_t Lt = (size_t)n_layers;
        if ((size_t)Tstep * (size_t)vocab > (size_t)16777216) {
            fprintf(stderr,
                    "train_tinystories: T*V too large (%d*%d); reduce "
                    "seq/batch for vocab %d\n",
                    Tstep, vocab, vocab);
            errno = EINVAL;
            goto cleanup;
        }
        TX = (int *)malloc(Tt * sizeof(int));
        TY = (int *)malloc(Tt * sizeof(int));
        actsT = (float *)malloc((Lt + 1) * Tt * (size_t)d * sizeof(float));
        cidsT = (size_t *)malloc(Lt * Tt * (size_t)TS_K * sizeof(size_t));
        cwT = (float *)malloc(Lt * Tt * (size_t)TS_K * sizeof(float));
        cgselT = (float *)malloc(Lt * Tt * (size_t)TS_K * (size_t)TS_H *
                                sizeof(float));
        cuselT = (float *)malloc(Lt * Tt * (size_t)TS_K * (size_t)TS_H *
                                sizeof(float));
        cyselT = (float *)malloc(Lt * Tt * (size_t)TS_K * (size_t)d *
                                sizeof(float));
        cgsT = (float *)malloc(Lt * Tt * (size_t)TS_S * (size_t)TS_H *
                              sizeof(float));
        cusT = (float *)malloc(Lt * Tt * (size_t)TS_S * (size_t)TS_H *
                              sizeof(float));
        permT = (size_t *)malloc(Lt * Tt * (size_t)TS_K * sizeof(size_t));
        offT = (size_t *)malloc(Lt * ((size_t)TS_E + 1) * sizeof(size_t));
        dropT = (unsigned char *)malloc(Lt * Tt * (size_t)TS_K *
                                       sizeof(unsigned char));
        probsT = (float *)malloc(Tt * (size_t)vocab * sizeof(float));
        dhT = (float *)malloc(Tt * (size_t)d * sizeof(float));
        dXb = (float *)malloc(Tt * (size_t)d * sizeof(float));
        if (TX == NULL || TY == NULL || actsT == NULL || cidsT == NULL ||
            cwT == NULL || cgselT == NULL || cuselT == NULL ||
            cyselT == NULL || cgsT == NULL || cusT == NULL || permT == NULL ||
            offT == NULL || dropT == NULL || probsT == NULL || dhT == NULL ||
            dXb == NULL) {
            fprintf(stderr, "train_tinystories: OOM (T buffers)\n");
            errno = ENOMEM;
            goto cleanup;
        }
    } else {
        // legacy seq==1用：層別ステップ集計（batch*L*K）
        size_t nse = (size_t)n_layers * (size_t)batch * (size_t)TS_K;
        step_ids = (size_t *)malloc(nse * sizeof(size_t));
        step_w = (float *)malloc(nse * sizeof(float));
        if (step_ids == NULL || step_w == NULL) {
            fprintf(stderr, "train_tinystories: OOM (step aux)\n");
            errno = ENOMEM;
            goto cleanup;
        }
    }

    /* ---- 初期CE (train先頭batch / val stride抽出) ---- */
    {
        double st = 0.0;
        long b = 0;
        long nb = (batch < n_train_pairs) ? batch : n_train_pairs;
        for (b = 0; b < nb; b++) {
            float loss = 0.0f;
            if (ts_fwd_one(&m, tx[b], acts, cids, cw, cgsel, cusel, cysel,
                           cgs, cus, logits, probs, &loss, ty[b]) != JT_OK) {
                fprintf(stderr, "train_tinystories: init fwd failed\n");
                goto cleanup;
            }
            st += (double)loss;
        }
        init_train_ce = st / (double)nb;
        if (ts_eval_val(&m, vx, vy, n_val_pairs, max_val_pairs, acts, cids,
                        cw, cgsel, cusel, cysel, cgs, cus, logits, probs,
                        &init_val_ce) != JT_OK) {
            fprintf(stderr, "train_tinystories: init val failed\n");
            goto cleanup;
        }
        last_train_ce = init_train_ce;
        last_val_ce = init_val_ce;
        best_val_ce = init_val_ce;
        best_val_step = 0;
        printf("train_ts: init train_ce=%.4f val_ce=%.4f val_ce_pb=%.6f "
               "(tok_per_byte=%.6f val_cap=%ld/%ld)\n",
               init_train_ce, init_val_ce, init_val_ce * tok_per_byte,
               tok_per_byte,
               (max_val_pairs > 0 && max_val_pairs < n_val_pairs)
                   ? max_val_pairs
                   : n_val_pairs,
               n_val_pairs);
        fflush(stdout);
    }

    /* ---- 学習ループ (合計勾配更新・fail-closed、train_namesと同流儀) ---- */
    /* G2用に毎ステップ train_ce/gnorm/drop/aux/entropy/renorm を stderr へ1行記録。 */
    t0 = ts_now_sec();
    if (g_ts_seq > 1) {
        // G2fix T一括路：T=seq*batchトークン/ステップを同一TX/TYでsingle/batch比較。
        // サンプリングはbatch要素毎にseq連続スパン（決定論的ハッシュ）。
        for (step = 0; step < max_steps; step++) {
            double sum = 0.0;
            double loss_mean = 0.0;
            float scale = 1.0f;
            size_t p = 0;
            size_t step_dropped = 0;
            size_t step_renorm = 0;
            float aux_m = 0.0f;
            float ent_m = 0.0f;
            double gnorm = 0.0;
            double dummy = 0.0;
            int rc = JT_OK;
            if (ts_now_sec() - t0 > time_limit) {
                printf("train_ts: time limit (%.0fs), stop at step=%ld\n",
                       time_limit, step);
                break;
            }
            if ((long)n_train_pairs < g_ts_seq) {
                fprintf(stderr, "train_tinystories: too few pairs for seq\n");
                goto cleanup;
            }
            for (long b = 0; b < batch; b++) {
                uint64_t key =
                    (uint64_t)(step * batch + b) * 0x9e3779b97f4a7c15ULL +
                    0x123456789abcdefULL;
                long span_max = n_train_pairs - g_ts_seq;
                long s0 = (long)(ts_hash64(key) % (uint64_t)(span_max + 1));
                for (long s = 0; s < g_ts_seq; s++) {
                    long tt = b * g_ts_seq + s;
                    TX[tt] = tx[s0 + s];
                    TY[tt] = ty[s0 + s];
                }
            }
            memset(m.G, 0, m.lt.n_total * sizeof(float));
            if (!g_ts_moe_batch) {
                rc = ts_fwd_T_single(&m, TX, Tstep, actsT, cidsT, cwT,
                                     cgselT, cuselT, cyselT, cgsT, cusT,
                                     logits, probsT, &dummy);
                if (rc != JT_OK) {
                    fprintf(stderr,
                            "train_tinystories: fwd_T failed at step=%ld\n",
                            step);
                    goto cleanup;
                }
            } else {
                rc = ts_fwd_T_batch(&m, TX, Tstep, actsT, cidsT, cwT,
                                    cgselT, cuselT, cyselT, cgsT, cusT,
                                    permT, offT, dropT, &step_dropped,
                                    &step_renorm);
                if (rc != JT_OK) {
                    fprintf(stderr,
                            "train_tinystories: fwd_T failed at step=%ld\n",
                            step);
                    goto cleanup;
                }
            }
            rc = ts_head_ce_fwd(&m, actsT, Tstep, TY, logits, probsT, &sum);
            if (rc != JT_OK) {
                fprintf(stderr,
                        "train_tinystories: head failed at step=%ld\n", step);
                goto cleanup;
            }
            loss_mean = sum / (double)Tstep;
            if (!g_ts_moe_batch) {
                rc = ts_bwd_T_single(&m, TX, actsT, Tstep, cidsT, cwT,
                                     cgselT, cuselT, cyselT, cgsT, cusT,
                                     probsT, TY, dlog, dhT, dX, tW, scale);
            } else {
                rc = ts_bwd_T_batch(&m, TX, actsT, Tstep, cidsT, cwT,
                                    cgselT, cuselT, cyselT, cgsT, cusT,
                                    permT, offT, dropT, probsT, TY, dlog,
                                    dhT, dXb, scale);
            }
            if (rc != JT_OK) {
                fprintf(stderr,
                        "train_tinystories: bwd_T failed at step=%ld\n",
                        step);
                goto cleanup;
            }
            rc = ts_aux_stats(cidsT, cwT, dropT, g_ts_moe_batch, n_layers,
                              Tstep, &aux_m, &ent_m);
            if (rc != JT_OK) {
                fprintf(stderr,
                        "train_tinystories: aux failed at step=%ld\n", step);
                goto cleanup;
            }
            rc = ts_aux_grad_apply(&m, actsT, cidsT, cwT, dropT,
                                   g_ts_moe_batch, Tstep, g_ts_aux_w);
            if (rc != JT_OK) {
                fprintf(stderr,
                        "train_tinystories: aux_grad failed at step=%ld\n",
                        step);
                goto cleanup;
            }
            last_train_ce = loss_mean + (double)g_ts_aux_w * (double)aux_m;
            {
                double ss = 0.0;
                for (p = 0; p < m.lt.n_total; p++) {
                    double g = (double)m.G[p];
                    if (!isfinite(g)) {
                        fprintf(stderr,
                                "train_tinystories: non-finite grad at "
                                "step=%ld\n",
                                step);
                        errno = EINVAL;
                        goto cleanup;
                    }
                    ss += g * g;
                }
                gnorm = sqrt(ss);
            }
            {
                size_t denom =
                    (size_t)Tstep * (size_t)n_layers * (size_t)TS_K;
                double dr = (denom > 0) ? (100.0 * (double)step_dropped /
                                           (double)denom)
                                        : 0.0;
                size_t rdenom = (size_t)Tstep * (size_t)n_layers;
                double rr = (rdenom > 0) ? (100.0 * (double)step_renorm /
                                            (double)rdenom)
                                         : 0.0;
                fprintf(stderr,
                        "train_ts_step: step=%ld train_ce=%.6f "
                        "ce=%.6f aux=%.6f ent=%.6f gnorm=%.6f "
                        "dropped=%zu/%zu (%.4f%%) renorm=%zu/%zu (%.4f%%) "
                        "moe_batch=%d T=%d\n",
                        step + 1, last_train_ce, loss_mean, aux_m, ent_m,
                        gnorm, step_dropped, denom, dr, step_renorm, rdenom,
                        rr, g_ts_moe_batch, Tstep);
            }
            if (jt_optim8_step(&m.opt, m.P, m.G, m.lt.n_total) != JT_OK) {
                fprintf(stderr,
                        "train_tinystories: optim failed at step=%ld\n",
                        step);
                goto cleanup;
            }
            steps_done = step + 1;
            if ((step + 1) % val_every == 0 || step + 1 == max_steps) {
                double ev = 0.0;
                int vrc = ts_eval_val(&m, vx, vy, n_val_pairs, max_val_pairs,
                                      acts, cids, cw, cgsel, cusel, cysel,
                                      cgs, cus, logits, probs, &ev);
                if (vrc != JT_OK) {
                    fprintf(stderr,
                            "train_tinystories: val failed at step=%ld\n",
                            step);
                    goto cleanup;
                }
                last_val_ce = ev;
                if (ev < best_val_ce - 1e-9) {
                    best_val_ce = ev;
                    best_val_step = step + 1;
                }
                prev_vals[0] = prev_vals[1];
                prev_vals[1] = prev_vals[2];
                prev_vals[2] = ev;
                if (n_prev < 3) {
                    n_prev++;
                }
                printf("train_ts: step=%ld train_ce=%.4f val_ce=%.4f "
                       "val_ce_pb=%.6f best=%.4f@%ld elapsed=%.1fs\n",
                       step + 1, last_train_ce, last_val_ce,
                       last_val_ce * tok_per_byte, best_val_ce,
                       best_val_step, ts_now_sec() - t0);
                fflush(stdout);
                if (patience > 0 && n_prev >= 3 &&
                    prev_vals[0] < prev_vals[1] &&
                    prev_vals[1] < prev_vals[2]) {
                    stopped_early = 1;
                    stop_reason = "val-rise-x3";
                    printf("train_ts: early stop (%s) at step=%ld\n",
                           stop_reason, step + 1);
                    break;
                }
                if (patience > 0 && (step + 1) - best_val_step >= patience) {
                    stopped_early = 1;
                    stop_reason = "patience";
                    printf("train_ts: early stop (%s=%ld) at step=%ld\n",
                           stop_reason, patience, step + 1);
                    break;
                }
            }
        }
    } else {
    for (step = 0; step < max_steps; step++) {
        double sum = 0.0;
        long b = 0;
        float scale = 1.0f;
        size_t p = 0;
        size_t step_dropped = 0;
        double gnorm = 0.0;
        if (ts_now_sec() - t0 > time_limit) {
            printf("train_ts: time limit (%.0fs), stop at step=%ld\n",
                   time_limit, step);
            break;
        }
        memset(m.G, 0, m.lt.n_total * sizeof(float));
        for (b = 0; b < batch; b++) {
            uint64_t key =
                (uint64_t)(step * batch + b) * 0x9e3779b97f4a7c15ULL +
                0x123456789abcdefULL;
            long idx = (long)(ts_hash64(key) % (uint64_t)n_train_pairs);
            float loss = 0.0f;
            int rc;
            if (g_ts_moe_batch) {
                size_t bd = 0;
                rc = ts_fwd_one_batch(&m, tx[idx], acts, cids, cw, cgsel,
                                      cusel, cysel, cgs, cus, bperm, boff,
                                      bdrop, &bd, logits, probs, &loss,
                                      ty[idx]);
                if (rc != JT_OK) {
                    fprintf(stderr,
                            "train_tinystories: fwd failed at step=%ld\n",
                            step);
                    goto cleanup;
                }
                step_dropped += bd;
            } else {
                rc = ts_fwd_one(&m, tx[idx], acts, cids, cw, cgsel, cusel,
                                cysel, cgs, cus, logits, probs, &loss,
                                ty[idx]);
                if (rc != JT_OK) {
                    fprintf(stderr,
                            "train_tinystories: fwd failed at step=%ld\n",
                            step);
                    goto cleanup;
                }
            }
            // seq==1 legacy用ステップ集計（層平均aux/entropy用）
            for (int _l = 0; _l < n_layers; _l++) {
                for (int _p = 0; _p < TS_K; _p++) {
                    size_t _s =
                        ((size_t)_l * (size_t)batch + (size_t)b) *
                            (size_t)TS_K +
                        (size_t)_p;
                    step_ids[_s] =
                        cids[(size_t)_l * (size_t)TS_K + (size_t)_p];
                    step_w[_s] = cw[(size_t)_l * (size_t)TS_K + (size_t)_p];
                }
            }
            sum += (double)loss;
            if (g_ts_moe_batch) {
                rc = ts_bwd_one_batch(&m, tx[idx], acts, cids, cw, cgsel,
                                      cusel, cysel, cgs, cus, bperm, boff,
                                      bdrop, probs, dlog, dh, dX, tW,
                                      ty[idx], scale);
            } else {
                rc = ts_bwd_one(&m, tx[idx], acts, cids, cw, cgsel, cusel,
                                cysel, cgs, cus, probs, dlog, dh, dX, tW,
                                ty[idx], scale);
            }
            if (rc != JT_OK) {
                fprintf(stderr, "train_tinystories: bwd failed at step=%ld\n",
                        step);
                goto cleanup;
            }
        }
        {
            // seq==1 legacyはauxログのみ（gradはT路のみ適用。G2はT路で比較）。
            float aux_m = 0.0f;
            float ent_m = 0.0f;
            double ce_mean = sum / (double)batch;
            if (ts_aux_stats(step_ids, step_w, NULL, 0, n_layers,
                             (int)batch, &aux_m, &ent_m) != JT_OK) {
                fprintf(stderr,
                        "train_tinystories: aux failed at step=%ld\n", step);
                goto cleanup;
            }
            last_train_ce = ce_mean + (double)g_ts_aux_w * (double)aux_m;
            {
                double ss = 0.0;
                for (p = 0; p < m.lt.n_total; p++) {
                    double g = (double)m.G[p];
                    if (!isfinite(g)) {
                        fprintf(stderr,
                                "train_tinystories: non-finite grad at "
                                "step=%ld\n",
                                step);
                        errno = EINVAL;
                        goto cleanup;
                    }
                    ss += g * g;
                }
                gnorm = sqrt(ss);
            }
            {
                size_t denom =
                    (size_t)batch * (size_t)n_layers * (size_t)TS_K;
                double dr = (denom > 0)
                                ? (100.0 * (double)step_dropped / (double)denom)
                                : 0.0;
                fprintf(stderr,
                        "train_ts_step: step=%ld train_ce=%.6f ce=%.6f "
                        "aux=%.6f ent=%.6f gnorm=%.6f dropped=%zu/%zu "
                        "(%.4f%%) renorm=0 moe_batch=%d T=%ld\n",
                        step + 1, last_train_ce, ce_mean, aux_m, ent_m,
                        gnorm, step_dropped, denom, dr, g_ts_moe_batch,
                        batch);
            }
        }
        if (jt_optim8_step(&m.opt, m.P, m.G, m.lt.n_total) != JT_OK) {
            fprintf(stderr, "train_tinystories: optim failed at step=%ld\n",
                    step);
            goto cleanup;
        }
        steps_done = step + 1;
        if ((step + 1) % val_every == 0 || step + 1 == max_steps) {
            double ev = 0.0;
            int rc = ts_eval_val(&m, vx, vy, n_val_pairs, max_val_pairs,
                                 acts, cids, cw, cgsel, cusel, cysel, cgs,
                                 cus, logits, probs, &ev);
            if (rc != JT_OK) {
                fprintf(stderr, "train_tinystories: val failed at step=%ld\n",
                        step);
                goto cleanup;
            }
            last_val_ce = ev;
            if (ev < best_val_ce - 1e-9) {
                best_val_ce = ev;
                best_val_step = step + 1;
            }
            /* 3点連続上昇リングを更新 */
            prev_vals[0] = prev_vals[1];
            prev_vals[1] = prev_vals[2];
            prev_vals[2] = ev;
            if (n_prev < 3) {
                n_prev++;
            }
            printf("train_ts: step=%ld train_ce=%.4f val_ce=%.4f "
                   "val_ce_pb=%.6f best=%.4f@%ld elapsed=%.1fs\n",
                   step + 1, last_train_ce, last_val_ce,
                   last_val_ce * tok_per_byte, best_val_ce, best_val_step,
                   ts_now_sec() - t0);
            fflush(stdout);
            if (patience > 0 && n_prev >= 3 && prev_vals[0] < prev_vals[1] &&
                prev_vals[1] < prev_vals[2]) {
                stopped_early = 1;
                stop_reason = "val-rise-x3";
                printf("train_ts: early stop (%s) at step=%ld\n",
                       stop_reason, step + 1);
                break;
            }
            if (patience > 0 && (step + 1) - best_val_step >= patience) {
                stopped_early = 1;
                stop_reason = "patience";
                printf("train_ts: early stop (%s=%ld) at step=%ld\n",
                       stop_reason, patience, step + 1);
                break;
            }
        }
    }
    } // else (seq==1 legacy) 終端

    {
        double el = ts_now_sec() - t0;
        long toks_per_step =
            (g_ts_seq > 1) ? (long)(g_ts_seq * batch) : batch;
        double toks = (double)steps_done * (double)toks_per_step;
        printf("train_ts: done steps=%ld train_ce: %.4f -> %.4f "
               "val_ce: %.4f -> %.4f val_ce_pb: %.6f -> %.6f "
               "(tok_per_byte=%.6f ratio %.3f) early=%d(%s) "
               "tokens=%.0f elapsed=%.1fs toks_per_s=%.1f "
               "(toks/s cross-vocab compare PROHIBITED per RULE.MD)\n",
               steps_done, init_train_ce, last_train_ce, init_val_ce,
               last_val_ce, init_val_ce * tok_per_byte,
               last_val_ce * tok_per_byte, tok_per_byte,
               (init_val_ce > 0.0) ? last_val_ce / init_val_ce : 0.0,
               stopped_early, stop_reason, toks, el,
               (el > 0.0) ? toks / el : 0.0);
        fflush(stdout);
    }
    rc_all = 0;

cleanup:
    if (seqs != NULL) {
        ts_free_seqs(seqs, lens, n_stories);
    }
    free(tx);
    free(ty);
    free(vx);
    free(vy);
    if (m.opt_inited) {
        jt_optim8_fini(&m.opt);
    }
    free(m.P);
    free(m.G);
    free(acts);
    free(cids);
    free(cw);
    free(cgsel);
    free(cusel);
    free(cysel);
    free(cgs);
    free(cus);
    free(logits);
    free(probs);
    free(dlog);
    free(dh);
    free(dX);
    free(tW);
    free(bperm);
    free(boff);
    free(bdrop);
    free(TX);
    free(TY);
    free(actsT);
    free(cidsT);
    free(cwT);
    free(cgselT);
    free(cuselT);
    free(cyselT);
    free(cgsT);
    free(cusT);
    free(permT);
    free(offT);
    free(dropT);
    free(probsT);
    free(dhT);
    free(dXb);
    free(step_ids);
    free(step_w);
    return rc_all;
}
