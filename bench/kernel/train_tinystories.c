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
//     スケーリング曲線は観測可能。--vocab 50257は構造対応のみ
//     (次元適応、encoderはbyte->2+b同一、258行以上は未使用・将来のBPE導入 reserve)。
//
// 構成: byte embedding (学習対象) → L層の小規模MoE (jt_moe_fwd/bwd再利用、
//   d=64既定・E=8・top-2・S=1・H=32、残差接続) → 線形head＋softmax CE。
// CEのforward/backwardは本ファイル内で新規実装 (C11・fail-closed)。
// データ: 物語単位で分割 (i%10==9をval、残りをtrain、決定論的。namesと同理由で
//   等間隔抽出)。valはforwardのみ (勾配計算・更新なし)。
// data_pack経路: --pack PATH (既定data/tinystories16M.jtdp) が開ければ
//   jt_dp_open経由で読む (正規経路)。なければテキスト直接読みにフォールバック
//   (その旨をstdoutに明記)。--write-pack PATHでテキスト→.jtdp変換
//   (jt_dp_writer使用)。byte-fallbackのIDは最大257のためJT_DP_VOCAB_SIZE制限
//   (<48588) を満たし、V=258/50257いずれでもpack経路が有効。
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
#include "jimotono/data_pack.h"
#include "jimotono/moe_layer.h"
#include "jimotono/optim8.h"

#define TS_BOS 0
#define TS_EOS 1
#define TS_BYTE_OFF 2 /* byte b -> 2+b (2..257) */
#define TS_VOCAB_SMALL 258
#define TS_VOCAB_BPE 50257

#define TS_E 8
#define TS_K 2
#define TS_S 1
#define TS_H 32

#define TS_DELIM "<|endoftext|>"
#define TS_DELIM_LEN 13

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
            "[--max-stories N] [--max-val-pairs N]\n"
            "  defaults: data=data/tinystories_head16M.txt "
            "pack=data/tinystories16M.jtdp steps=500 time=3600 lr=3e-4 "
            "batch=64 d=64 layers=2 val-every=100 patience=100 vocab=258 "
            "max-stories=0(all) max-val-pairs=20000\n"
            "  --vocab: 258 (byte+special) or 50257 (structural only) \n"
            "  --patience 0 disables early stopping\n"
            "  --max-val-pairs 0 evaluates full val set\n",
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

// ---- テキスト読み: <|endoftext|>区切りの物語を [BOS, bytes..., EOS] に変換 ----
// seqs_out/lens_outは呼び出し側でfree (各行malloc)。戻り値は物語数。失敗時-1。
// max_stories>0時は先頭max_stories話のみ (決定論的)。
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
    if (vocab != TS_VOCAB_SMALL && vocab != TS_VOCAB_BPE) {
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

// .jtdp読み (data_pack正規経路)。ID>=vocabは拒否 (fail-closed)。
// byte-fallbackのIDは最大257のためjt_dp_openの<48588検証も通過する。
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
    if (vocab != TS_VOCAB_SMALL && vocab != TS_VOCAB_BPE) {
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
    double t0 = 0.0;
    long step = 0;
    long steps_done = 0;
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
        } else if (strcmp(argv[i], "--max-stories") == 0 && i + 1 < argc) {
            max_stories = atol(argv[++i]);
        } else if (strcmp(argv[i], "--max-val-pairs") == 0 && i + 1 < argc) {
            max_val_pairs = atol(argv[++i]);
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
    if (max_steps <= 0 || !(time_limit > 0.0) || !(lr > 0.0f) ||
        !isfinite(lr) || batch <= 0 || batch > 4096 || d < 8 || d > 256 ||
        n_layers < 1 || n_layers > 4 || val_every <= 0 || patience < 0 ||
        (vocab != TS_VOCAB_SMALL && vocab != TS_VOCAB_BPE) ||
        max_stories < 0 || max_val_pairs < 0) {
        fprintf(stderr, "train_tinystories: invalid limits/args\n");
        errno = EINVAL;
        goto cleanup;
    }

    /* ---- --write-pack: テキスト -> .jtdp変換のみ ---- */
    if (write_pack != NULL) {
        jt_dp_writer_t w;
        long wn = 0;
        long k = 0;
        memset(&w, 0, sizeof(w));
        n_stories = ts_read_text(data_path, vocab, &seqs, &lens, max_stories,
                                 &nbytes);
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
        printf("train_ts: write-pack nseq=%ld bytes=%zu source=%s dest=%s\n",
               wn, nbytes, data_path, write_pack);
        ts_free_seqs(seqs, lens, n_stories);
        return 0;
    }

    /* ---- データ読み: data_pack正規経路を優先、不可時は直接読み ---- */
    n_stories = ts_read_pack(pack_path, vocab, &seqs, &lens);
    if (n_stories > 0) {
        data_source = "pack";
    } else if (use_pack_opt) {
        fprintf(stderr, "train_tinystories: cannot open pack '%s': %s\n",
                pack_path, strerror(errno));
        goto cleanup;
    } else {
        int se = errno;
        n_stories = ts_read_text(data_path, vocab, &seqs, &lens, max_stories,
                                 &nbytes);
        if (n_stories <= 0) {
            fprintf(stderr, "train_tinystories: cannot read '%s': %s\n",
                    data_path, strerror(errno));
            goto cleanup;
        }
        data_source = "direct(txt)";
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
           "params=%zu lr=%.5f batch=%ld patience=%ld\n",
           n_layers, d, TS_E, TS_K, TS_S, TS_H, vocab, m.lt.n_total, lr,
           batch, patience);
    fflush(stdout);
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
    if (acts == NULL || cids == NULL || cw == NULL || cgsel == NULL ||
        cusel == NULL || cysel == NULL || cgs == NULL || cus == NULL ||
        logits == NULL || probs == NULL || dlog == NULL || dh == NULL ||
        dX == NULL || tW == NULL) {
        fprintf(stderr, "train_tinystories: OOM\n");
        errno = ENOMEM;
        goto cleanup;
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
        printf("train_ts: init train_ce=%.4f val_ce=%.4f (val_cap=%ld/%ld)\n",
               init_train_ce, init_val_ce,
               (max_val_pairs > 0 && max_val_pairs < n_val_pairs)
                   ? max_val_pairs
                   : n_val_pairs,
               n_val_pairs);
        fflush(stdout);
    }

    /* ---- 学習ループ (合計勾配更新・fail-closed、train_namesと同流儀) ---- */
    t0 = ts_now_sec();
    for (step = 0; step < max_steps; step++) {
        double sum = 0.0;
        long b = 0;
        float scale = 1.0f;
        size_t p = 0;
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
            int rc = ts_fwd_one(&m, tx[idx], acts, cids, cw, cgsel, cusel,
                                cysel, cgs, cus, logits, probs, &loss,
                                ty[idx]);
            if (rc != JT_OK) {
                fprintf(stderr, "train_tinystories: fwd failed at step=%ld\n",
                        step);
                goto cleanup;
            }
            sum += (double)loss;
            rc = ts_bwd_one(&m, tx[idx], acts, cids, cw, cgsel, cusel,
                            cysel, cgs, cus, probs, dlog, dh, dX, tW,
                            ty[idx], scale);
            if (rc != JT_OK) {
                fprintf(stderr, "train_tinystories: bwd failed at step=%ld\n",
                        step);
                goto cleanup;
            }
        }
        last_train_ce = sum / (double)batch;
        for (p = 0; p < m.lt.n_total; p++) {
            double g = (double)m.G[p];
            if (!isfinite(g)) {
                fprintf(stderr,
                        "train_tinystories: non-finite grad at step=%ld\n",
                        step);
                errno = EINVAL;
                goto cleanup;
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
                   "best=%.4f@%ld elapsed=%.1fs\n",
                   step + 1, last_train_ce, last_val_ce, best_val_ce,
                   best_val_step, ts_now_sec() - t0);
            fflush(stdout);
            if (n_prev >= 3 && prev_vals[0] < prev_vals[1] &&
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

    {
        double el = ts_now_sec() - t0;
        double toks = (double)steps_done * (double)batch;
        printf("train_ts: done steps=%ld train_ce: %.4f -> %.4f "
               "val_ce: %.4f -> %.4f (ratio %.3f) early=%d(%s) "
               "tokens=%.0f elapsed=%.1fs toks_per_s=%.1f\n",
               steps_done, init_train_ce, last_train_ce, init_val_ce,
               last_val_ce,
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
    return rc_all;
}
