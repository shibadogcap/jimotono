// train_names: 文字レベル次文字予測の小規模言語モデル学習 (実行バイナリ、
//   ctest非登録)。Phase F4・実データ担当。連続値合成回帰 (train_proxy100m)
//   ではなく、makemore names.txt (約32k英語名) による言語モデリングで
//   train/val CEの双方が下がることを確認する。
//
// 語彙: V=28 (0=BOS, 1=EOS, 2..27='a'..'z')。会計上の48588は使わない。
// 構成: 文字embedding (学習対象) → L層の小規模MoE (jt_moe_fwd/bwd再利用、
//   d=64既定・E=8・top-2・S=1・H=32、残差接続) → 線形head＋softmax CE。
// CEのforward/backwardは本ファイル内で新規実装 (C11・fail-closed)。
// データ: 名前単位で分割 (先頭90%=train、末尾10%=val、決定論的)。
//   valはforwardのみ (勾配計算・更新なし)。
// data_pack経路: --pack PATH (既定data/names.jtdp) が開ければjt_dp_open経由で
//   読む (正規経路)。なければnames.txt直接読みにフォールバックする (その旨を
//   stdoutに明記)。--write-pack PATHでnames.txt→.jtdp変換 (jt_dp_writer使用)。
//   .jtdp内のIDは本語彙 (<28) のみを受け付け、範囲外はfail-closed (exit 1)。
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

#define TN_VOCAB 28
#define TN_BOS 0
#define TN_EOS 1
#define TN_A_OFF 2 /* 'a' -> 2 ... 'z' -> 27 */

#define TN_E 8
#define TN_K 2
#define TN_S 1
#define TN_H 32

// ---- 決定論的ハッシュ乱数 (train_proxy100mと同流儀のsplitmix64) ----
static uint64_t tn_hash64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static double tn_u01(uint64_t key) {
    uint64_t h = tn_hash64(key);
    return (double)(h >> 11) * (1.0 / 9007199254740992.0);
}

// ---- softmax交差エントロピー (新規実装、fail-closed) ----
// ce_fwd: loss = -log softmax(logits)[target]。probs_out!=NULL時はsoftmax全量。
//   失敗時はloss_out/probs_outを更新せずJT_ERR_INVAL (errno=EINVAL)。
static int tn_ce_fwd(const float *restrict logits, int v, int target,
                     float *restrict loss_out, float *restrict probs_out) {
    double mx = 0.0;
    double sum = 0.0;
    double loss = 0.0;
    int i = 0;
    if (logits == NULL || loss_out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (v < 2 || v > 4096 || target < 0 || target >= v) {
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

// ce_bwd: dlogits[i] = scale * (probs[i] - (i==target))。
//   失敗時はdlogits_outを更新せずJT_ERR_INVAL (errno=EINVAL)。
static int tn_ce_bwd(const float *restrict probs, int v, int target,
                     float scale, float *restrict dlogits_out) {
    int i = 0;
    if (probs == NULL || dlogits_out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (v < 2 || v > 4096 || target < 0 || target >= v) {
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
        double d = p - ((i == target) ? 1.0 : 0.0);
        dlogits_out[i] = (float)((double)scale * d);
    }
    return JT_OK;
}

static double tn_now_sec(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void tn_usage(const char *prog) {
    fprintf(stderr,
            "usage: %s [--data names.txt] [--pack names.jtdp] "
            "[--write-pack out.jtdp] [--steps N] [--time SECS] [--lr LR] "
            "[--batch B] [--d DIM] [--layers L] [--val-every K] "
            "[--sample N] [--temp T] [--seed S] [--maxlen M]\n"
            "  defaults: data=data/names.txt pack=data/names.jtdp steps=500 "
            "time=1200 lr=1e-3 batch=256 d=64 layers=2 val-every=25 "
            "sample=0 temp=0 seed=0x53414D50 maxlen=20\n"
            "  --write-pack: names.txt -> .jtdp変換のみ行い終了 "
            "(data_pack正規経路)\n"
            "  --sample N: 学習完了後に最終重みでN個の名前を生成 (forwardのみ)\n"
            "  --temp T: 0でgreedy(argmax)、T>0で温度サンプリング (決定論的)\n",
            prog);
}

// ---- パラメータ配置 (単一Pバッファ。Gも同一配置) ----
//   [emb V*d][layer0][layer1]...[Wo V*d][bo V]
//   layer: [Wgate E*d][Wg E*H*d][Wu ..][Wd ..][Wgs S*H*d][Wus ..][Wds ..]
typedef struct tn_layout {
    size_t off_emb;
    size_t layer_stride;
    size_t off_wgate; /* 層内相対 */
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
} tn_layout_t;

static void tn_build_layout(tn_layout_t *lt, int n_layers, int d) {
    size_t e = (size_t)TN_E;
    size_t h = (size_t)TN_H;
    size_t dd = (size_t)d;
    size_t s = (size_t)TN_S;
    size_t v = (size_t)TN_VOCAB;
    size_t routed = e * h * dd;
    size_t shared = s * h * dd;
    memset(lt, 0, sizeof(*lt));
    lt->n_layers = n_layers;
    lt->d = d;
    lt->off_emb = 0;
    lt->off_wgate = 0;
    lt->off_wg = e * dd;
    lt->off_wu = e * dd + routed;
    lt->off_wd = e * dd + (size_t)2 * routed;
    lt->off_wgs = e * dd + (size_t)3 * routed;
    lt->off_wus = e * dd + (size_t)3 * routed + shared;
    lt->off_wds = e * dd + (size_t)3 * routed + (size_t)2 * shared;
    lt->layer_stride = e * dd + (size_t)3 * routed + (size_t)3 * shared;
    lt->off_head_wo = v * dd + (size_t)n_layers * lt->layer_stride;
    lt->off_head_bo = v * dd + (size_t)n_layers * lt->layer_stride + v * dd;
    lt->n_total = v * dd + (size_t)n_layers * lt->layer_stride + v * dd + v;
}

// ---- 学習モデル本体 ----
typedef struct tn_model {
    tn_layout_t lt;
    float *P;
    float *G;
    jt_optim8_t opt;
    int opt_inited;
} tn_model_t;

// 1サンプル分のforward。actsは[(L+1)*d]作業域 (層入力の保存)。
// caches: ids[L*K], w[L*K], Gsel[L*K*H], Usel[L*K*H], Ysel[L*K*d],
//   Gs[L*S*H], Us[L*S*H]。logits[V], probs[V]出力。loss_out出力。
static int tn_fwd_one(const tn_model_t *restrict m, int x,
                      float *restrict acts, size_t *restrict cids,
                      float *restrict cw, float *restrict cgsel,
                      float *restrict cusel, float *restrict cysel,
                      float *restrict cgs, float *restrict cus,
                      float *restrict logits, float *restrict probs,
                      float *restrict loss_out, int y) {
    int L = m->lt.n_layers;
    int d = m->lt.d;
    const float *P = m->P;
    const float *emb = P + m->lt.off_emb;
    const float *Wo = P + m->lt.off_head_wo;
    const float *bo = P + m->lt.off_head_bo;
    float M[256];
    int l = 0;
    int j = 0;
    int v = 0;
    if (d > 256) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (j = 0; j < d; j++) {
        acts[j] = emb[(size_t)x * (size_t)d + (size_t)j];
    }
    for (l = 0; l < L; l++) {
        const float *base =
            P + m->lt.off_emb + (size_t)TN_VOCAB * (size_t)d +
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
        size_t *ids = cids + (size_t)l * (size_t)TN_K;
        float *w = cw + (size_t)l * (size_t)TN_K;
        float *Gsel = cgsel + (size_t)l * (size_t)TN_K * (size_t)TN_H;
        float *Usel = cusel + (size_t)l * (size_t)TN_K * (size_t)TN_H;
        float *Ysel = cysel + (size_t)l * (size_t)TN_K * (size_t)d;
        float *Gs = cgs + (size_t)l * (size_t)TN_S * (size_t)TN_H;
        float *Us = cus + (size_t)l * (size_t)TN_S * (size_t)TN_H;
        int rc = jt_moe_fwd(xin, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds, M, d,
                            TN_H, TN_E, TN_K, TN_S, ids, w, NULL, Gsel,
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
        for (v = 0; v < TN_VOCAB; v++) {
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
    return tn_ce_fwd(logits, TN_VOCAB, y, loss_out, probs);
}

// 1サンプル分のbackward (forward済みのacts/caches/probsを使用)。
// Gへの加算とtW作業域 (layer_stride長) を使う。dlog_scale=1/B。
static int tn_bwd_one(tn_model_t *restrict m, int x,
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
    float *G = m->G;
    const float *P = m->P;
    float *Gemb = G + m->lt.off_emb;
    float *GWo = G + m->lt.off_head_wo;
    float *Gbo = G + m->lt.off_head_bo;
    const float *Wo = P + m->lt.off_head_wo;
    int l = 0;
    int j = 0;
    int v = 0;
    if (tn_ce_bwd(probs, TN_VOCAB, y, dlog_scale, dlog) != JT_OK) {
        return JT_ERR_INVAL;
    }
    {
        const float *hl = acts + (size_t)L * (size_t)d;
        for (j = 0; j < d; j++) {
            dh[j] = 0.0f;
        }
        for (v = 0; v < TN_VOCAB; v++) {
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
            P + m->lt.off_emb + (size_t)TN_VOCAB * (size_t)d +
            (size_t)l * m->lt.layer_stride;
        float *gbase = G + m->lt.off_emb +
                       (size_t)TN_VOCAB * (size_t)d +
                       (size_t)l * m->lt.layer_stride;
        const float *Wgate = base + m->lt.off_wgate;
        const float *Wg = base + m->lt.off_wg;
        const float *Wu = base + m->lt.off_wu;
        const float *Wd = base + m->lt.off_wd;
        const float *Wgs = base + m->lt.off_wgs;
        const float *Wus = base + m->lt.off_wus;
        const float *Wds = base + m->lt.off_wds;
        const float *xin = acts + (size_t)l * (size_t)d;
        const size_t *ids = cids + (size_t)l * (size_t)TN_K;
        const float *w = cw + (size_t)l * (size_t)TN_K;
        const float *Gsel = cgsel + (size_t)l * (size_t)TN_K * (size_t)TN_H;
        const float *Usel = cusel + (size_t)l * (size_t)TN_K * (size_t)TN_H;
        const float *Ysel = cysel + (size_t)l * (size_t)TN_K * (size_t)d;
        const float *Gs = cgs + (size_t)l * (size_t)TN_S * (size_t)TN_H;
        const float *Us = cus + (size_t)l * (size_t)TN_S * (size_t)TN_H;
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
                            tWg, tWu, tWd, tWgs, tWus, tWds, NULL, d, TN_H,
                            TN_E, TN_K, TN_S);
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

// ---- データ読み ----
// names.txt直接読み: 各行[a-z]+を [BOS, letters..., EOS] のID列に変換。
// seqs_outは呼び出し側でfree (各行malloc)。戻り値は名前数。失敗時は-1。
static long tn_read_names_txt(const char *restrict path,
                              uint32_t ***restrict seqs_out,
                              size_t **restrict lens_out) {
    FILE *f = NULL;
    uint32_t **seqs = NULL;
    size_t *lens = NULL;
    size_t n = 0;
    size_t cap = 0;
    char line[256];
    if (path == NULL || seqs_out == NULL || lens_out == NULL) {
        errno = EINVAL;
        return -1;
    }
    f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), f) != NULL) {
        size_t len = strlen(line);
        size_t k = 0;
        size_t nlets = 0;
        uint32_t *sq = NULL;
        size_t p = 0;
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            len--;
        }
        line[len] = '\0';
        if (len == 0 || len > 64) {
            continue; /* 空行スキップ。64超は異常行として捨てる */
        }
        for (k = 0; k < len; k++) {
            if (line[k] < 'a' || line[k] > 'z') {
                break;
            }
        }
        if (k != len) {
            continue; /* 非a-z行は捨てる */
        }
        nlets = len;
        sq = (uint32_t *)malloc((nlets + 2) * sizeof(uint32_t));
        if (sq == NULL) {
            goto fail;
        }
        sq[0] = (uint32_t)TN_BOS;
        for (p = 0; p < nlets; p++) {
            sq[1 + p] = (uint32_t)(TN_A_OFF + (line[p] - 'a'));
        }
        sq[nlets + 1] = (uint32_t)TN_EOS;
        if (n == cap) {
            size_t ncap = (cap == 0) ? 4096 : cap * 2;
            uint32_t **ns = NULL;
            size_t *nl = NULL;
            if (ncap > (size_t)1000000) {
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
        lens[n] = nlets + 2;
        n++;
    }
    if (ferror(f)) {
        goto fail;
    }
    fclose(f);
    *seqs_out = seqs;
    *lens_out = lens;
    return (long)n;
fail: {
    size_t i = 0;
    int se = errno;
    if (f != NULL) {
        fclose(f);
    }
    for (i = 0; i < n; i++) {
        free(seqs[i]);
    }
    free(seqs);
    free(lens);
    errno = (se != 0) ? se : ENOMEM;
    return -1;
}
}

// .jtdp読み (data_pack正規経路)。ID>=TN_VOCABは拒否 (fail-closed)。
static long tn_read_pack(const char *restrict path,
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
    if (jt_dp_open(path, &r) != JT_OK) {
        return -1;
    }
    if (jt_dp_nseq(&r, &nseq) != JT_OK || nseq == 0 || nseq > 1000000ULL) {
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
            slen > 128ULL || ptr == NULL) {
            goto fail;
        }
        cp = (uint32_t *)malloc((size_t)slen * sizeof(uint32_t));
        if (cp == NULL) {
            goto fail;
        }
        for (k = 0; k < slen; k++) {
            if (ptr[k] >= (uint32_t)TN_VOCAB) {
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

static void tn_free_seqs(uint32_t **seqs, size_t *lens, long n) {
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

int main(int argc, char **argv) {
    int rc_all = 1;
    const char *data_path = "data/names.txt";
    const char *pack_path = "data/names.jtdp";
    const char *write_pack = NULL;
    int use_pack_opt = 0; /* --pack明示時のみ必須化 */
    long max_steps = 500;
    double time_limit = 1200.0;
    float lr = 1e-3f;
    long batch = 256;
    int d = 64;
    int n_layers = 2;
    long val_every = 25;
    long n_sample = 0; /* --sample N: 学習後にN個生成 (0=生成なし) */
    double temp = 0.0; /* --temp T: 0=greedy、>0で温度サンプリング */
    uint64_t seed = 0x53414D50ULL; /* 決定論的既定seed */
    int max_len = 20;  /* 1名前あたり最大文字数 */
    uint32_t **seqs = NULL;
    size_t *lens = NULL;
    long n_names = 0;
    long n_train_names = 0;
    int *tx = NULL; /* train pairs */
    int *ty = NULL;
    long n_train_pairs = 0;
    int *vx = NULL; /* val pairs */
    int *vy = NULL;
    long n_val_pairs = 0;
    const char *data_source = "?";
    tn_model_t m;
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
    float init_train_ce = 0.0f;
    float init_val_ce = 0.0f;
    float last_train_ce = 0.0f;
    float last_val_ce = 0.0f;
    int i = 0;

    memset(&m, 0, sizeof(m));

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
        } else if (strcmp(argv[i], "--sample") == 0 && i + 1 < argc) {
            n_sample = atol(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temp = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 0);
        } else if ((strcmp(argv[i], "--maxlen") == 0 ||
                    strcmp(argv[i], "--max-len") == 0) &&
                   i + 1 < argc) {
            max_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 ||
                   strcmp(argv[i], "--help") == 0) {
            tn_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "train_names: unknown arg '%s'\n", argv[i]);
            tn_usage(argv[0]);
            errno = EINVAL;
            goto cleanup;
        }
    }
    if (max_steps <= 0 || !(time_limit > 0.0) || !(lr > 0.0f) ||
        !isfinite(lr) || batch <= 0 || batch > 4096 || d < 8 || d > 256 ||
        n_layers < 1 || n_layers > 4 || val_every <= 0 || n_sample < 0 ||
        n_sample > 100000 || !isfinite(temp) || temp < 0.0 || temp > 5.0 ||
        max_len < 1 || max_len > 64) {
        fprintf(stderr, "train_names: invalid limits/args\n");
        errno = EINVAL;
        goto cleanup;
    }

    /* ---- --write-pack: names.txt -> .jtdp変換のみ ---- */
    if (write_pack != NULL) {
        jt_dp_writer_t w;
        long wn = 0;
        long k = 0;
        memset(&w, 0, sizeof(w));
        n_names = tn_read_names_txt(data_path, &seqs, &lens);
        if (n_names <= 0) {
            fprintf(stderr, "train_names: cannot read '%s': %s\n",
                    data_path, strerror(errno));
            goto cleanup;
        }
        if (jt_dp_writer_open(&w, write_pack) != JT_OK) {
            fprintf(stderr, "train_names: cannot open '%s': %s\n",
                    write_pack, strerror(errno));
            goto cleanup;
        }
        wn = n_names;
        for (k = 0; k < wn; k++) {
            if (jt_dp_writer_add(&w, seqs[k], lens[k]) != JT_OK) {
                fprintf(stderr, "train_names: writer_add failed at %ld: %s\n",
                        k, strerror(errno));
                goto cleanup;
            }
        }
        if (jt_dp_writer_close(&w) != JT_OK) {
            fprintf(stderr, "train_names: writer_close failed: %s\n",
                    strerror(errno));
            goto cleanup;
        }
        printf("train_names: write-pack nseq=%ld source=%s dest=%s\n", wn,
               data_path, write_pack);
        tn_free_seqs(seqs, lens, n_names);
        return 0;
    }

    /* ---- データ読み: data_pack正規経路を優先、不可時は直接読み ---- */
    n_names = tn_read_pack(pack_path, &seqs, &lens);
    if (n_names > 0) {
        data_source = "pack";
    } else if (use_pack_opt) {
        fprintf(stderr, "train_names: cannot open pack '%s': %s\n",
                pack_path, strerror(errno));
        goto cleanup;
    } else {
        int se = errno;
        n_names = tn_read_names_txt(data_path, &seqs, &lens);
        if (n_names <= 0) {
            fprintf(stderr, "train_names: cannot read '%s': %s\n",
                    data_path, strerror(errno));
            goto cleanup;
        }
        data_source = "direct(txt)";
        (void)se;
    }

    /* ---- 名前単位で分割: i%10==9をval、残りをtrain (決定論的) ----
       末尾10%切りだとnames.txtが字母順ソートのためvalがx/y/z始まりに偏る
       (分布シフト混入)。等間隔抽出でtrain/valの分布を一致させる。 */
    n_train_names = n_names - (n_names + 9L) / 10L;
    if (n_train_names < 1 || n_names - n_train_names < 1) {
        fprintf(stderr, "train_names: too few names (%ld)\n", n_names);
        errno = EINVAL;
        goto cleanup;
    }
    {
        long k = 0;
        long cap_t = 0;
        long cap_v = 0;
        long t = 0;
        long q = 0;
        for (k = 0; k < n_names; k++) {
            if (k % 10 == 9) {
                cap_v += (long)lens[k];
            } else {
                cap_t += (long)lens[k];
            }
        }
        tx = (int *)malloc((size_t)cap_t * sizeof(int));
        ty = (int *)malloc((size_t)cap_t * sizeof(int));
        vx = (int *)malloc((size_t)cap_v * sizeof(int));
        vy = (int *)malloc((size_t)cap_v * sizeof(int));
        if (tx == NULL || ty == NULL || vx == NULL || vy == NULL) {
            fprintf(stderr, "train_names: OOM\n");
            errno = ENOMEM;
            goto cleanup;
        }
        for (k = 0; k < n_names; k++) {
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
        n_train_names = 0;
        for (k = 0; k < n_names; k++) {
            if (k % 10 != 9) {
                n_train_names++;
            }
        }
    }
    printf("train_names: names=%ld train_names=%ld val_names=%ld "
           "train_pairs=%ld val_pairs=%ld source=%s\n",
           n_names, n_train_names, n_names - n_train_names, n_train_pairs,
           n_val_pairs, data_source);
    if (strcmp(data_source, "direct(txt)") == 0) {
        printf("train_names: note: pack '%s' unavailable; using direct "
               "txt read (fallback). Run --write-pack to enable data_pack "
               "route.\n",
               pack_path);
    }
    fflush(stdout);
    tn_free_seqs(seqs, lens, n_names);
    seqs = NULL;
    lens = NULL;

    /* ---- モデル確保・初期化 ---- */
    tn_build_layout(&m.lt, n_layers, d);
    m.P = (float *)malloc(m.lt.n_total * sizeof(float));
    m.G = (float *)malloc(m.lt.n_total * sizeof(float));
    if (m.P == NULL || m.G == NULL) {
        fprintf(stderr, "train_names: OOM\n");
        errno = ENOMEM;
        goto cleanup;
    }
    {
        /* 決定論的初期化: emb/MoE/head小一様。headゼロだと初手で深層に
           勾配が流れないためheadも小乱数 (初期CE≒ln28)。 */
        size_t n = m.lt.n_total;
        size_t p = 0;
        size_t emb_n = (size_t)TN_VOCAB * (size_t)d;
        uint64_t key = 0x1234ULL;
        for (p = 0; p < n; p++) {
            double u = tn_u01(key * 2ULL + 0x9e3779b9ULL + (uint64_t)p);
            double sc = 0.125;
            if (p < emb_n) {
                sc = 1.0; /* embは±1 */
            } else if (p < emb_n + (size_t)n_layers * m.lt.layer_stride) {
                size_t rel =
                    (p - emb_n) % m.lt.layer_stride;
                if (rel < (size_t)TN_E * (size_t)d) {
                    sc = 0.05; /* gateは小さく (初期均等routing) */
                } else {
                    sc = 0.125; /* expert */
                }
            } else {
                sc = 0.125; /* head */
            }
            m.P[p] = (float)((u * 2.0 - 1.0) * sc);
        }
        /* head biasは0 */
        for (p = 0; p < (size_t)TN_VOCAB; p++) {
            m.P[m.lt.off_head_bo + p] = 0.0f;
        }
    }
    {
        jt_optim8_cfg_t cfg;
        jt_optim8_cfg_default(&cfg);
        cfg.lr = lr;
        /* int8量子化状態では疎な勾配 (稀な文字のemb行・非選択expert行) の
           vが0丸めされ mh/eps が発散しうる。eps=1e-4で更新幅に上限を設け、
           既定1e-8の発散モードを抑える (Adamの標準的な安定化knob)。 */
        cfg.eps = 1e-4f;
        if (jt_optim8_init(&m.opt, m.lt.n_total, &cfg) != JT_OK) {
            fprintf(stderr, "train_names: optim init failed\n");
            goto cleanup;
        }
        m.opt_inited = 1;
    }
    printf("train_names: layers=%d d=%d E=%d K=%d S=%d H=%d V=%d "
           "params=%zu lr=%.5f batch=%ld\n",
           n_layers, d, TN_E, TN_K, TN_S, TN_H, TN_VOCAB, m.lt.n_total, lr,
           batch);
    fflush(stdout);
    acts = (float *)malloc((size_t)(n_layers + 1) * (size_t)d * sizeof(float));
    cids = (size_t *)malloc((size_t)n_layers * (size_t)TN_K * sizeof(size_t));
    cw = (float *)malloc((size_t)n_layers * (size_t)TN_K * sizeof(float));
    cgsel =
        (float *)malloc((size_t)n_layers * (size_t)TN_K * (size_t)TN_H *
                        sizeof(float));
    cusel =
        (float *)malloc((size_t)n_layers * (size_t)TN_K * (size_t)TN_H *
                        sizeof(float));
    cysel = (float *)malloc((size_t)n_layers * (size_t)TN_K * (size_t)d *
                            sizeof(float));
    cgs = (float *)malloc((size_t)n_layers * (size_t)TN_S * (size_t)TN_H *
                          sizeof(float));
    cus = (float *)malloc((size_t)n_layers * (size_t)TN_S * (size_t)TN_H *
                          sizeof(float));
    logits = (float *)malloc((size_t)TN_VOCAB * sizeof(float));
    probs = (float *)malloc((size_t)TN_VOCAB * sizeof(float));
    dlog = (float *)malloc((size_t)TN_VOCAB * sizeof(float));
    dh = (float *)malloc((size_t)d * sizeof(float));
    dX = (float *)malloc((size_t)d * sizeof(float));
    tW = (float *)malloc(m.lt.layer_stride * sizeof(float));
    if (acts == NULL || cids == NULL || cw == NULL || cgsel == NULL ||
        cusel == NULL || cysel == NULL || cgs == NULL || cus == NULL ||
        logits == NULL || probs == NULL || dlog == NULL || dh == NULL ||
        dX == NULL || tW == NULL) {
        fprintf(stderr, "train_names: OOM\n");
        errno = ENOMEM;
        goto cleanup;
    }

    /* ---- 初期CE (train先頭batch / val全体) ---- */
    {
        double st = 0.0;
        double sv = 0.0;
        long b = 0;
        long nb = (batch < n_train_pairs) ? batch : n_train_pairs;
        for (b = 0; b < nb; b++) {
            float loss = 0.0f;
            if (tn_fwd_one(&m, tx[b], acts, cids, cw, cgsel, cusel, cysel,
                           cgs, cus, logits, probs, &loss, ty[b]) != JT_OK) {
                fprintf(stderr, "train_names: init fwd failed\n");
                goto cleanup;
            }
            st += (double)loss;
        }
        for (b = 0; b < n_val_pairs; b++) {
            float loss = 0.0f;
            if (tn_fwd_one(&m, vx[b], acts, cids, cw, cgsel, cusel, cysel,
                           cgs, cus, logits, probs, &loss, vy[b]) != JT_OK) {
                fprintf(stderr, "train_names: init val failed\n");
                goto cleanup;
            }
            sv += (double)loss;
        }
        init_train_ce = (float)(st / (double)nb);
        init_val_ce = (float)(sv / (double)n_val_pairs);
        last_train_ce = init_train_ce;
        last_val_ce = init_val_ce;
        printf("train_names: init train_ce=%.4f val_ce=%.4f\n",
               init_train_ce, init_val_ce);
        fflush(stdout);
    }

    /* ---- 学習ループ ---- */
    t0 = tn_now_sec();
    for (step = 0; step < max_steps; step++) {
        double sum = 0.0;
        long b = 0;
        /* batch平均ではなくbatch合計の勾配で更新する。Adamはスケール不変の
           ため方向は等価だが、int8量子化状態 (optim8) が0潰れせず健全に
           保たれる。平均勾配 (1/B倍) では状態が0に丸まり mh/eps が発散する
           ため本設定は必須。 */
        float scale = 1.0f;
        size_t p = 0;
        if (tn_now_sec() - t0 > time_limit) {
            printf("train_names: time limit (%.0fs), stop at step=%ld\n",
                   time_limit, step);
            break;
        }
        memset(m.G, 0, m.lt.n_total * sizeof(float));
        for (b = 0; b < batch; b++) {
            /* 決定論的サンプリング (step由来キー。valと無関係) */
            uint64_t key =
                (uint64_t)(step * batch + b) * 0x9e3779b97f4a7c15ULL +
                0x123456789abcdefULL;
            long idx = (long)(tn_hash64(key) % (uint64_t)n_train_pairs);
            float loss = 0.0f;
            int rc = tn_fwd_one(&m, tx[idx], acts, cids, cw, cgsel, cusel,
                                cysel, cgs, cus, logits, probs, &loss,
                                ty[idx]);
            if (rc != JT_OK) {
                fprintf(stderr, "train_names: fwd failed at step=%ld\n",
                        step);
                goto cleanup;
            }
            sum += (double)loss;
            rc = tn_bwd_one(&m, tx[idx], acts, cids, cw, cgsel, cusel,
                            cysel, cgs, cus, probs, dlog, dh, dX, tW,
                            ty[idx], scale);
            if (rc != JT_OK) {
                fprintf(stderr, "train_names: bwd failed at step=%ld\n",
                        step);
                goto cleanup;
            }
        }
        last_train_ce = (float)(sum / (double)batch);
        /* 更新前G全走査 (fail-closed)。normクリップは行わない:
           合計勾配を縮めるとint8状態の量子化分解能が落ち、vの0丸め→
           mh/eps発散を招く。更新幅の上限はoptim8のeps=1e-4が担う。 */
        for (p = 0; p < m.lt.n_total; p++) {
            double g = (double)m.G[p];
            if (!isfinite(g)) {
                fprintf(stderr, "train_names: non-finite grad at step=%ld\n",
                        step);
                errno = EINVAL;
                goto cleanup;
            }
        }
        if (jt_optim8_step(&m.opt, m.P, m.G, m.lt.n_total) != JT_OK) {
            fprintf(stderr, "train_names: optim failed at step=%ld\n", step);
            goto cleanup;
        }
        if ((step + 1) % val_every == 0 || step + 1 == max_steps) {
            double sv = 0.0;
            long q = 0;
            for (q = 0; q < n_val_pairs; q++) {
                float loss = 0.0f;
                int rc =
                    tn_fwd_one(&m, vx[q], acts, cids, cw, cgsel, cusel,
                               cysel, cgs, cus, logits, probs, &loss, vy[q]);
                if (rc != JT_OK) {
                    fprintf(stderr, "train_names: val failed at step=%ld\n",
                            step);
                    goto cleanup;
                }
                sv += (double)loss;
            }
            last_val_ce = (float)(sv / (double)n_val_pairs);
            printf("train_names: step=%ld train_ce=%.4f val_ce=%.4f "
                   "elapsed=%.1fs\n",
                   step + 1, last_train_ce, last_val_ce,
                   tn_now_sec() - t0);
            fflush(stdout);
        }
    }

    printf("train_names: done steps=%ld train_ce: %.4f -> %.4f "
           "val_ce: %.4f -> %.4f (ratio %.3f)\n",
           step, init_train_ce, last_train_ce, init_val_ce, last_val_ce,
           (init_val_ce > 0.0f) ? last_val_ce / init_val_ce : 0.0f);
    fflush(stdout);

    /* ---- デモ用サンプリング: 最終重みでN個生成 (forwardのみ・更新なし) ----
       BOS開始、EOS/BOSまたはmax_lenで打切り。temp<=0はgreedy(argmax)、
       temp>0はlogits/Tのsoftmaxから温度サンプリング。乱数はtn_u01による
       決定論的ハッシュ (seed + sample_idx + pos由来) のみを使用。 */
    if (n_sample > 0) {
        long si = 0;
        printf("train_names: sampling n=%ld temp=%.3f seed=%llu maxlen=%d\n",
               n_sample, temp, (unsigned long long)seed, max_len);
        fflush(stdout);
        for (si = 0; si < n_sample; si++) {
            int cur = TN_BOS;
            int pos = 0;
            char out[65];
            int olen = 0;
            for (pos = 0; pos < max_len; pos++) {
                float dummy_loss = 0.0f;
                int nxt = 0;
                int v = 0;
                int rc = tn_fwd_one(&m, cur, acts, cids, cw, cgsel, cusel,
                                    cysel, cgs, cus, logits, probs,
                                    &dummy_loss, 0);
                if (rc != JT_OK) {
                    fprintf(stderr,
                            "train_names: sample fwd failed at %ld/%d\n", si,
                            pos);
                    goto cleanup;
                }
                if (!(temp > 0.0)) {
                    /* greedy: argmax (同値は先勝ち・決定論的) */
                    nxt = 0;
                    for (v = 1; v < TN_VOCAB; v++) {
                        if (logits[v] > logits[nxt]) {
                            nxt = v;
                        }
                    }
                } else {
                    /* 温度サンプリング: exp((logit-mx)/T) → 累積分布 */
                    double mx = (double)logits[0];
                    double sum = 0.0;
                    double acc = 0.0;
                    double thr = 0.0;
                    double u = 0.0;
                    uint64_t key = seed +
                                   (uint64_t)si * 0x9e3779b97f4a7c15ULL +
                                   (uint64_t)pos * 0xbf58476d1ce4e5b9ULL +
                                   0x123456789abcdefULL;
                    for (v = 1; v < TN_VOCAB; v++) {
                        if ((double)logits[v] > mx) {
                            mx = (double)logits[v];
                        }
                    }
                    for (v = 0; v < TN_VOCAB; v++) {
                        sum += exp(((double)logits[v] - mx) / temp);
                    }
                    if (!(sum > 0.0) || !isfinite(sum)) {
                        fprintf(stderr,
                                "train_names: sample temper failed at "
                                "%ld/%d\n",
                                si, pos);
                        errno = EINVAL;
                        goto cleanup;
                    }
                    u = tn_u01(key);
                    thr = u * sum;
                    nxt = TN_VOCAB - 1;
                    for (v = 0; v < TN_VOCAB; v++) {
                        acc += exp(((double)logits[v] - mx) / temp);
                        if (thr < acc) {
                            nxt = v;
                            break;
                        }
                    }
                }
                if (nxt == TN_EOS || nxt == TN_BOS) {
                    break;
                }
                if (nxt < TN_A_OFF || nxt >= TN_VOCAB || olen >= max_len) {
                    fprintf(stderr,
                            "train_names: sample bad id at %ld/%d\n", si,
                            pos);
                    errno = EINVAL;
                    goto cleanup;
                }
                out[olen++] = (char)('a' + (nxt - TN_A_OFF));
                cur = nxt;
            }
            out[olen] = '\0';
            printf("sample[%04ld]: %s\n", si, out);
        }
        fflush(stdout);
    }
    rc_all = 0;

cleanup:
    if (seqs != NULL) {
        tn_free_seqs(seqs, lens, n_names);
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
