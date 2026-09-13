// train_proxy100m: active約100M/総約1B級の実学習ループproxy (実行バイナリ、
//   ctest非登録)。Phase F準備用。train_proxy100m.cの長期漬け5M-proxyに対し、
//   expert形状を1B級と同一にした28M級スライスで実経路を回す。
// ---- 解析目標 (analytic, offload/esmoe前提。物理常駐ではない) ----
//   layers=24, d_model=1024, experts/layer=96, expert_hidden=128,
//   shared=2, top-k=4, vocab=48588, linear_every=6。
//   jt_model1b_counts検算: total=1,014,779,904 (約1.015B) /
//   active_body=96,804,864 (約96.8M)。総expert数24*96=2304はARCH §2の
//   2000〜4000帯。model1b既定(E=64・総0.712B)からEのみ96へ拡張したもの。
//   K固定のためactiveはほぼ不変 (Eにほぼ不変。model1b.h注記と同方針)。
// ---- 物理実行 (runnable, 本バイナリが実際に回すループ) ----
//   layers=4, d_model=1024, experts/layer=16, expert_hidden=128,
//   top-k=4, shared=2, seq=32, batch=2, fp32, 決定論的合成回帰。
//   物理総数=28,378,113 (約28M)。物理active/token≈9.4M (MoE発火分)。
// ---- proxyである旨と本物との差 (必読) ----
//   (1) expert形状 d×h=1024×128 が解析目標と同一のため、算術強度AI・
//       カーネル経路は一致する。速度・AIのproxyとして有効。
//   (2) 層4/24・expert数16/96のスライスのため、総数・activeは約1/10〜1/36。
//       収束・品質・ルーティング統計のproxyではない。
//   (3) 系列は32×2=64 toks/step (解析の学習想定B=8/S=2048とは別)。1000-step
//       所要の外挿はtoks/s基準で行い、層・系列差は親側で補正すること。
//   (4) 総1Bの全重み常駐は不可のため解析側はoffload前提
//       (resident=total-routing)。物理側は全常駐fp32で2GB gate内に収める。
// ループ: MoE層 fwd→bwd→optim8更新。5ステップ毎にckpt recompute照合、
//   5ステップ毎にesmoe往復。毎ステップsticky loss (実gate分布由来) を計測し
//   sticky列に記録する (損失合算・逆伝播はrouting.h準拠でTODO。 scaffold段
//   階のため測定のみ)。loss/steps-sec/tokens-sec/RSSをstderrへ1行ログ。
// held-out検証: train_proxy100mと同一方式 (salt 7・VAL_OFFSETで重なりなし)。
//   5ステップ毎＋開始前＋最終に計測し val_loss 列に記録する。
//   F4汎化課題 (--gen-task stream, 既定): trainはステップ毎に新規サンプルを
//   決定論的ストリーム生成 (salt 17・索引=step*TOKS+t) し、同一の真の関数
//   (T=dot(X,w_star)/4) のまま memorization を無効化する。固定64点の丸暗記
//   では将来ステップ・valは下がらない。valは固定held-out (salt 7) で真の
//   関数への汎化を評価する。--gen-task fixedで旧来の固定64点に戻せる。
// 起動時に解析目標の会計 (total/active/dense INT4/SSD/学習peak) と当該物理
//   構成のFLOPs/byte概算 (analysis/roofline.md §D4式) を表示する。
// 時間上限 (既定6000秒) とステップ上限のどちらかで正常終了 (exit 0)。
// loss非有限で即時exit 1。不正引数・2GB超・内部失敗もexit 1。
// 規約: C11, restrict積極使用, errnoベース + goto cleanup (AGENTS.MD 7.1)。
// クロスプラットフォーム (POSIX getrusage/RU_MAXRSS、Windowsは概算0)。
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#if defined(_POSIX_THREADS)
#include <pthread.h>
#define PX_HAVE_PTHREAD 1
#else
#define PX_HAVE_PTHREAD 0
#endif
#else
#define PX_HAVE_PTHREAD 0
#endif

#include "jimotono/checkpoint.h"
#include "jimotono/common.h"
#include "jimotono/esmoe.h"
#include "jimotono/model1b.h"
#include "jimotono/moe_layer.h"
#include "jimotono/optim8.h"

#define PX_LAYERS 4
#define PX_N 1024
#define PX_H 128
#define PX_E 16
#define PX_K 4
#define PX_S 2
#define PX_SEQ 32
#define PX_BATCH 2
#define PX_TOKS (PX_SEQ * PX_BATCH)
#define PX_RESID 1.0f
#define PX_MAX_GB 2.0
#define PX_CKPT_EVERY 5
#define PX_ESMOE_EVERY 5
#define PX_CKPT_TOL 1e-4f
// held-out val: trainと同一生成式・同一w_star。トークン索引をVAL_OFFSETだけ
// ずらすことで重なりなし (train t∈[0,TOKS), val t+OFFSET)。決定論的。
#define PX_VAL_TOKS PX_TOKS
#define PX_VAL_EVERY 5
#define PX_VAL_OFFSET 100000

// ---- 解析目標 (analytic 1B級。model1b既定のE=64→96拡張。K固定) ----
#define PX_A_LAYERS 24
#define PX_A_D 1024
#define PX_A_E 96
#define PX_A_H 128
#define PX_A_S 2
#define PX_A_K 4
#define PX_A_VOCAB 48588
#define PX_A_EVERY 6

// ---- sticky loss実経路用 (papers.md §3適用注意: λ=0.05〜0.1開始・W=2〜4) ----
#define PX_STICKY_LAMBDA 0.1f
#define PX_STICKY_ALPHA 0.05f
#define PX_STICKY_W 4

// 1層あたり: Wgate[E*N] + Wg/Wu/Wd[E*H*N]x3 + Wgs/Wus/Wds[S*H*N]x3
#define PX_P_WGATE ((size_t)PX_E * (size_t)PX_N)
#define PX_P_ROUTED ((size_t)PX_E * (size_t)PX_H * (size_t)PX_N)
#define PX_P_SHARED ((size_t)PX_S * (size_t)PX_H * (size_t)PX_N)
#define PX_P_LAYER (PX_P_WGATE + (size_t)3 * PX_P_ROUTED + (size_t)3 * PX_P_SHARED)
#define PX_P_HEAD ((size_t)PX_N + (size_t)1)

// 決定論的ハッシュ乱数 [-1,1] (整数演算のみで平台依存なし)。
// mod-11等の短周期パターンは次元1024と干渉し偽相関→活性発火を起こすため
// splitmix64流儀の64bitハッシュを使う。
static uint64_t px_hash64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static float px_pat(int a, int b, int salt) {
    uint64_t key = (uint64_t)(uint32_t)a * 0x9e3779b1ULL +
                   (uint64_t)(uint32_t)b * 85ULL +
                   (uint64_t)(uint32_t)salt * 0xc2b2ae35ULL + 0x27d4eb2fULL;
    uint64_t h = px_hash64(key);
    double u = (double)(h >> 11) * (1.0 / 9007199254740992.0);
    return (float)(u * 2.0 - 1.0);
}

static double px_now_sec(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

// ---- F2 breakdown probes (Phase F2 measurement only) ----
// 計測専用。計算内容・順序・並列化に一切触れない。wallはclock_gettime単調時計
// (px_now_sec)、cyclesはx86 rdtsc (非x86では0)。10フェーズの合計がstep実測に
// 一致するよう other = step_wall - sum(9相) で残差吸収する。
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || \
    defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(__rdtsc)
static inline uint64_t px_rdtsc(void) {
    return (uint64_t)__rdtsc();
}
#elif defined(__GNUC__) || defined(__clang__)
static inline uint64_t px_rdtsc(void) {
    uint32_t lo = 0;
    uint32_t hi = 0;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}
#else
static inline uint64_t px_rdtsc(void) {
    return 0ULL;
}
#endif
#define PX_HAVE_RDTSC 1
#else
static inline uint64_t px_rdtsc(void) {
    return 0ULL;
}
#define PX_HAVE_RDTSC 0
#endif

enum {
    PX_PH_DATA = 0,   // acts<=X memcpy
    PX_PH_FWD,        // forward計算 (worker内fwdのmax)
    PX_PH_BWD,        // backward計算 (worker内bwdのmax。G zero含む)
    PX_PH_OPTIM,      // grad norm/clip + optim8_step
    PX_PH_CKPT,       // ckpt recompute照合 (5step毎)
    PX_PH_ESMOE,      // esmoe往復 (register/prefetch/load/memcmp。5step毎)
    PX_PH_SYNC,       // fork/join + 順序付きreduction (単一Tでは0)
    PX_PH_ALLOC,      // esmoe往復内 malloc/free のみ
    PX_PH_LOGGING,    // loss MSE + sticky + val + stderrログ
    PX_PH_OTHER,      // weights_finite + 残差 (step_wall - sum)
    PX_PH_N
};
static const char *px_ph_name[PX_PH_N] = {
    "data", "fwd",        "bwd",      "optim",   "ckpt_recompute",
    "esmoe_io", "sync",   "alloc",    "logging", "other"};
static double px_ph_acc[PX_PH_N];
static long px_ph_cswv[PX_PH_N];
static long px_ph_csiv[PX_PH_N];
static long px_ph_inb[PX_PH_N];
static long px_ph_oub[PX_PH_N];
static uint64_t px_tsc_acc = 0ULL;
static double px_step_wall_acc = 0.0; // F2 probe: Σstep_wall (一致確認用)
// F2 probe: logging相の内訳 (合計はLOGGINGと一致。計測のみ)。
static double px_log_loss = 0.0;   // MSE loss
static double px_log_sticky = 0.0; // sticky loss実経路
static double px_log_val = 0.0;    // held-out val forward
static double px_log_misc = 0.0;   // stderrログ・getrusage等
static uint64_t px_esmoe_write_bytes = 0ULL; // register(pwrite)側の計測値

typedef struct px_snap {
    double t;
    long v;
    long iv;
    long inb;
    long oub;
} px_snap_t;

// getrusage拡張スナップ (ru_nvcsw/nivcsw + ru_inblock/oublock)。
// macOSではinblock/oublockは常に0の見込み (測定で確認する)。
static px_snap_t px_snap_now(void) {
    px_snap_t s;
    s.t = px_now_sec();
    s.v = 0L;
    s.iv = 0L;
    s.inb = 0L;
    s.oub = 0L;
#if defined(__unix__) || defined(__APPLE__)
    {
        struct rusage ru;
        memset(&ru, 0, sizeof(ru));
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            s.v = (long)ru.ru_nvcsw;
            s.iv = (long)ru.ru_nivcsw;
            s.inb = (long)ru.ru_inblock;
            s.oub = (long)ru.ru_oublock;
        }
    }
#endif
    return s;
}

static void px_prof_acc(int ph, px_snap_t a, px_snap_t b) {
    if (ph < 0 || ph >= PX_PH_N) {
        return;
    }
    px_ph_acc[ph] += b.t - a.t;
    px_ph_cswv[ph] += (b.v - a.v);
    px_ph_csiv[ph] += (b.iv - a.iv);
    px_ph_inb[ph] += (b.inb - a.inb);
    px_ph_oub[ph] += (b.oub - a.oub);
}

// RSSをMiBで返す (取得不可時は0)。
// D2: 自発的/非自発的コンテキストスイッチ差分 (ステップあたり) 用。
// getrusage(RUSAGE_SELF)のru_nvcsw/ru_nivcswを返す。取得不可時は0。
static void px_get_csw(long *v, long *iv) {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage ru;
    memset(&ru, 0, sizeof(ru));
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        if (v != NULL) {
            *v = (long)ru.ru_nvcsw;
        }
        if (iv != NULL) {
            *iv = (long)ru.ru_nivcsw;
        }
        return;
    }
#endif
    if (v != NULL) {
        *v = 0L;
    }
    if (iv != NULL) {
        *iv = 0L;
    }
}

// RSSをMiBで返す (取得不可時は0)。
static double px_rss_mib(void) {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage ru;
    memset(&ru, 0, sizeof(ru));
    if (getrusage(RUSAGE_SELF, &ru) != 0) {
        return 0.0;
    }
#if defined(__APPLE__)
    // macOS: ru_maxrssはbytes。
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#else
    // Linux: ru_maxrssはKB。
    return (double)ru.ru_maxrss / 1024.0;
#endif
#else
    return 0.0;
#endif
}

// ---- proxy100mスケール会計＋AI概算 (起動時表示) ----
// 解析目標の会計は jt_model1b_*純粋関数で算出する (ハードコード禁止)。
// AI概算は analysis/roofline.md §D4式 (fp32バイト、積和=2FLOP)。
// 失敗時はerrno維持で0以外を返す (呼び出し側はgoto cleanup)。
static int px_print_scale(void) {
    jt_model1b_config_t acfg;
    jt_model1b_counts_t c;
    jt_model1b_bytes_t b;
    jt_model1b_train_cfg_t t;
    jt_model1b_train_mem_t m;
    memset(&acfg, 0, sizeof(acfg));
    memset(&c, 0, sizeof(c));
    memset(&b, 0, sizeof(b));
    memset(&m, 0, sizeof(m));
    acfg.n_layers = PX_A_LAYERS;
    acfg.d_model = PX_A_D;
    acfg.n_experts = PX_A_E;
    acfg.expert_hidden = PX_A_H;
    acfg.n_shared = PX_A_S;
    acfg.top_k = PX_A_K;
    acfg.vocab = PX_A_VOCAB;
    acfg.linear_every = PX_A_EVERY;
    if (jt_model1b_counts(&acfg, &c) != JT_OK) {
        fprintf(stderr, "train_proxy100m: analytic counts failed\n");
        return 1;
    }
    if (jt_model1b_bytes(&acfg, &b) != JT_OK) {
        fprintf(stderr, "train_proxy100m: analytic bytes failed\n");
        return 1;
    }
    jt_model1b_train_default(&t);
    if (jt_model1b_train_mem(&acfg, &t, &m) != JT_OK) {
        fprintf(stderr, "train_proxy100m: analytic trainmem failed\n");
        return 1;
    }
    printf("proxy100m analytic: L=%d d=%d E=%d h=%d S=%d K=%d vocab=%d\n",
           acfg.n_layers, acfg.d_model, acfg.n_experts, acfg.expert_hidden,
           acfg.n_shared, acfg.top_k, acfg.vocab);
    printf("proxy100m analytic: total=%llu (%.3fB) active_body=%llu (%.2fM) "
           "active_moe=%llu\n",
           (unsigned long long)c.total, (double)c.total / 1e9,
           (unsigned long long)c.active_body, (double)c.active_body / 1e6,
           (unsigned long long)c.active_moe);
    printf("proxy100m analytic: dense_int8=%.1fMiB dense_int4=%.1fMiB "
           "ssd=%.1fMiB\n",
           (double)b.dense_total_int8 / 1048576.0,
           (double)b.dense_total_int4 / 1048576.0,
           (double)b.ssd_total / 1048576.0);
    printf("proxy100m analytic: train(B=%d,S=%d,offload=%d) peak=%.3fGiB "
           "(w=%.2f g=%.2f o=%.2f act=%.2f) seg=%d <=16GiB %s\n",
           t.batch, t.seq, t.offload_routing,
           (double)m.peak_total / 1073741824.0,
           (double)m.weights_ram / 1073741824.0,
           (double)m.grads_ram / 1073741824.0,
           (double)m.optim_ram / 1073741824.0,
           (double)m.act_peak / 1073741824.0, m.n_seg_used,
           (m.peak_total <= 16ULL * 1024ULL * 1024ULL * 1024ULL) ? "OK"
                                                                : "OVER");
    fflush(stdout);
    {
        // AI概算 (roofline.md §D4式。当該物理構成 n=PX_N, h=PX_H)。
        // expert形状が解析目標と同一 (1024×128) のため解析目標のAIと一致。
        double n = (double)PX_N;
        double h = (double)PX_H;
        double dot_f = 2.0 * h * n;
        double dot_b = 4.0 * (h * n + n + h);
        double fwd_f = 6.0 * h * n + 20.0 * h;
        double fwd_b = 4.0 * (3.0 * h * n + 2.0 * n + 2.0 * h);
        double bwd_f = 9.0 * h * n + 30.0 * h;
        double bwd_b = 4.0 * (6.0 * h * n + 2.0 * n + 2.0 * h);
        printf("proxy100m AI(n=%.0f,h=%.0f): dot=%.2f swiglu_fwd=%.2f "
               "swiglu_bwd=%.2f FLOPs/byte (fp32, ridge=21 -> all "
               "memory-bound)\n",
               n, h, dot_f / dot_b, fwd_f / fwd_b, bwd_f / bwd_b);
        fflush(stdout);
    }
    return 0;
}

// ステップ冒頭の重み一括検証 (unchecked区間の前提)。
// P[n_total] (全層重み＋head) 全体を1回だけ走査し、全有限なら1を返す。
// ステップ内ではP不変 (更新はステップ末のoptim stepのみ) のため、この
// 1回をもって当該ステップ内のトークン×層のper-call重み有限スキャンを
// 省略できる (unchecked使用条件 (1)(2) の充足)。
// 活性化由来の非有限は従来通り残る3点で検出する:
//   (a) fwd残差合算点のisfinite (px_fwd_layer_range内)、
//   (b) unchecked内の計算途中ガード (gate acc・yacc・dw_dp等)、
//   (c) loss合算点のisfinite＋更新前G全走査ガード。
// 重み自体の非有限はここでステップ粒度fail-closed (即時abort)。
static int px_weights_finite(const float *restrict P, size_t n) {
    if (P == NULL) {
        return 0;
    }
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)P[i])) {
            return 0;
        }
    }
    return 1;
}

typedef struct px_ckpt_ctx {
    const float *restrict P;
    const size_t *restrict layer_off;
    float *restrict cur;      // [N] 再計算カーソル
    const float *restrict bnd; // [n_seg+1][N] token0の境界スナップショット
    int n_seg;
    const int *restrict bounds;
} px_ckpt_ctx_t;

// ckpt再計算コールバック: cur = cur + MoE(cur)。
// 検証用途 (5ステップ毎・token0のみ) のため検証ありAPIのまま残す。
// コストは無視できる (全fwdコールの1/3000以下)。
static int px_ckpt_layer_fn(int seg_idx, int layer, void *vctx) {
    px_ckpt_ctx_t *c = (px_ckpt_ctx_t *)vctx;
    if (c == NULL || c->P == NULL || c->cur == NULL ||
        c->layer_off == NULL) {
        return JT_ERR_INVAL;
    }
    if (layer < 0 || layer >= PX_LAYERS) {
        return JT_ERR_INVAL;
    }
    if (seg_idx < 0 || (c->n_seg > 0 && seg_idx >= c->n_seg)) {
        return JT_ERR_INVAL;
    }
    {
        const float *base = c->P + c->layer_off[(size_t)layer];
        const float *Wgate = base;
        const float *Wg = Wgate + PX_P_WGATE;
        const float *Wu = Wg + PX_P_ROUTED;
        const float *Wd = Wu + PX_P_ROUTED;
        const float *Wgs = Wd + PX_P_ROUTED;
        const float *Wus = Wgs + PX_P_SHARED;
        const float *Wds = Wus + PX_P_SHARED;
        float M[PX_N];
        size_t ids[PX_K];
        float weights[PX_K];
        float Gsel[(size_t)PX_K * (size_t)PX_H];
        float Usel[(size_t)PX_K * (size_t)PX_H];
        float Ysel[(size_t)PX_K * (size_t)PX_N];
        float Gs[(size_t)PX_S * (size_t)PX_H];
        float Us[(size_t)PX_S * (size_t)PX_H];
        int rc = jt_moe_fwd(c->cur, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds, M,
                            PX_N, PX_H, PX_E, PX_K, PX_S, ids, weights,
                            NULL, Gsel, Usel, Ysel, Gs, Us);
        if (rc != JT_OK) {
            return rc;
        }
        for (int j = 0; j < PX_N; j++) {
            double v = (double)c->cur[j] + (double)PX_RESID * (double)M[j];
            c->cur[j] = (float)v;
        }
    }
    return JT_OK;
}

// F3-5: lrスケジュール (warmup 100 + cosine decay)。数値等価ではなく
// 学習動態の改善用。既定base lrは2e-4。
#define PX_LR_WARMUP 100L
#define PX_LR_DEFAULT 2e-4f
#ifndef PX_LR_PI
#define PX_LR_PI 3.14159265358979323846
#endif
// スケジュールlrを返す (stepは0-indexed、T=max_steps、W=warmup、B=base)。
// step<W: 線形warmup B*(step+1)/W。以降: cosine B*0.5*(1+cos(pi*(step+1-W)/(T-W)))。
// T<=W時はwarmupのみ。常に有限・非負を返す。
static float px_sched_lr(long step, long total, float base) {
    double B = (double)base;
    double W = 100.0;
    double T = (total > 0) ? (double)total : 1.0;
    double s = (double)(step + 1);
    double lr = B;
    if (!(B > 0.0) || !isfinite(B)) {
        return base;
    }
    if (s <= W) {
        lr = B * (s / W);
    } else if (T <= W) {
        lr = B;
    } else {
        double p = (s - W) / (T - W);
        if (p < 0.0) {
            p = 0.0;
        }
        if (p > 1.0) {
            p = 1.0;
        }
        lr = B * 0.5 * (1.0 + cos(p * PX_LR_PI));
    }
    if (!isfinite(lr) || lr < 0.0) {
        lr = 0.0;
    }
    return (float)lr;
}
static void px_usage(const char *prog) {
    fprintf(stderr,
            "usage: %s [--steps N] [--time SECS] [--lr LR] [--threads T] "
            "[--profile] [--no-offload] [--no-ckpt] [--no-sched] "
            "[--gen-task stream|fixed]\n"
            "  defaults: steps=1073741824 time=6000 lr=2e-4 threads=6 "
            "gen-task=stream\n"
            "  --gen-task stream: F4汎化課題 (既定。ステップ毎に新規train点)\n"
            "  --gen-task fixed: 旧来の固定64点 (暗記可能。比較用)\n"
            "  --profile: F2 breakdown (10-phase wall/csw/io, measurement "
            "only)\n"
            "  --no-offload: disable ESMoE roundtrip (28M resident, skip "
            "register/prefetch/load check)\n"
            "  --no-ckpt: disable ckpt recompute check (skip recompute "
            "verification)\n"
            "  --no-sched: disable lr schedule (fixed lr; default is warmup "
            "100 + cosine decay)\n",
            prog);
}

// 合成回帰データ生成 (T = dot(X, w_star)/4。決定論的)。
// F4汎化課題のtrain/val分割:
//   stream(既定): trainはステップs毎に salt=17・索引=s*TOKS+t の新規64点。
//     saltがval(salt 7)と異なるため長期漬けでもvalと重ならない。同一真関数。
//   fixed(旧来): trainは salt=7・t∈[0,TOKS) の固定64点 (暗記可能。step無視)。
// valは常に salt=7・VAL_OFFSETずらしの固定64点 (更新に一切使わない)。
static void px_fill_train(float *restrict X, float *restrict T,
                          const float *restrict w_star, long step,
                          int stream) {
    int salt = stream ? 17 : 7;
    long base = stream ? step * (long)PX_TOKS : 0L;
    for (int t = 0; t < PX_TOKS; t++) {
        long idx = base + (long)t;
        // px_patはint引数のため31bitへ折り畳み (決定論性は維持)。
        int tt = (int)(idx % 2000000000L);
        double acc = 0.0;
        for (int j = 0; j < PX_N; j++) {
            float v = px_pat(tt, j, salt);
            X[(size_t)t * (size_t)PX_N + (size_t)j] = v;
            acc += (double)v * (double)w_star[j];
        }
        T[t] = (float)(acc / 4.0);
    }
}

// ---- 並列ワーカー共有コンテキスト (読み取り専用。Gpart/dtmpは线程毎) ----
// fwdが書き込む中間値キャッシュ (Cids/Cw/CGsel/CUsel/CYsel/CGs/CUs) は
// bwdの再計算除去用。fwdがトークン非重複に書き込み、bwdは読み取りのみの
// ためトークン並列で安全。勾配加算は選択expertのみのスパース加算
// (非選択のtWはbwd契約で0のため加算不要。要素毎の加算順序は不変)。
typedef struct px_ctx {
    const float *restrict P;
    float *restrict acts;  // [(LAYERS+1)][TOKS][N]
    const float *restrict Xdata;
    const float *restrict Tdata;
    const size_t *restrict layer_off;
    size_t head_off;
    size_t *restrict Cids;   // [L][TOKS][K] fwdの選択ids
    float *restrict Cw;      // [L][TOKS][K] fwdの選択weights
    float *restrict CGsel;   // [L][TOKS][K*H]
    float *restrict CUsel;   // [L][TOKS][K*H]
    float *restrict CYsel;   // [L][TOKS][K*N]
    float *restrict CGs;     // [L][TOKS][S*H] (S==0時はNULL)
    float *restrict CUs;     // [L][TOKS][S*H] (S==0時はNULL)
} px_ctx_t;

// forward 1層分 [a,b) トークン。rc_outにJT_OK/ERR。
static void px_fwd_layer_range(const px_ctx_t *restrict ctx, int layer,
                               int a, int b, int *restrict rc_out) {
    int rc = JT_OK;
    const float *base = ctx->P + ctx->layer_off[(size_t)layer];
    const float *Wgate = base;
    const float *Wg = Wgate + PX_P_WGATE;
    const float *Wu = Wg + PX_P_ROUTED;
    const float *Wd = Wu + PX_P_ROUTED;
    const float *Wgs = Wd + PX_P_ROUTED;
    const float *Wus = Wgs + PX_P_SHARED;
    const float *Wds = Wus + PX_P_SHARED;
    for (int t = a; t < b && rc == JT_OK; t++) {
        const float *xin =
            ctx->acts +
            ((size_t)layer * (size_t)PX_TOKS + (size_t)t) * (size_t)PX_N;
        float *xout =
            ctx->acts +
            ((size_t)(layer + 1) * (size_t)PX_TOKS + (size_t)t) *
                (size_t)PX_N;
        // bwd使い回し用の中間値を保存 (再計算除去。同一入力のためビット不変)。
        size_t cbase = ((size_t)layer * (size_t)PX_TOKS + (size_t)t);
        size_t *ids = ctx->Cids + cbase * (size_t)PX_K;
        float *weights = ctx->Cw + cbase * (size_t)PX_K;
        float *Gsel =
            ctx->CGsel + cbase * (size_t)PX_K * (size_t)PX_H;
        float *Usel =
            ctx->CUsel + cbase * (size_t)PX_K * (size_t)PX_H;
        float *Ysel =
            ctx->CYsel + cbase * (size_t)PX_K * (size_t)PX_N;
        float *Gs = (PX_S > 0 && ctx->CGs != NULL)
                        ? (ctx->CGs + cbase * (size_t)PX_S * (size_t)PX_H)
                        : NULL;
        float *Us = (PX_S > 0 && ctx->CUs != NULL)
                        ? (ctx->CUs + cbase * (size_t)PX_S * (size_t)PX_H)
                        : NULL;
        float M[PX_N];
        // unchecked: 重みはステップ冒頭で一括検証済み＋ステップ内不変のため
        // per-call重みスキャンを省略 (同一計算核のためbit一致)。
        // 活性化NaNは内側途中ガード＋残差合算点isfiniteで検出する。
        int frc = jt_moe_fwd_unchecked(xin, Wgate, Wg, Wu, Wd, Wgs, Wus,
                                       Wds, M, PX_N, PX_H, PX_E, PX_K,
                                       PX_S, ids, weights, NULL, Gsel,
                                       Usel, Ysel, Gs, Us);
        if (frc != JT_OK) {
            rc = frc;
            break;
        }
        for (int j = 0; j < PX_N; j++) {
            double v = (double)xin[j] + (double)PX_RESID * (double)M[j];
            if (!isfinite(v)) {
                rc = JT_ERR_INVAL;
                break;
            }
            xout[j] = (float)v;
        }
    }
    if (rc_out != NULL) {
        *rc_out = rc;
    }
}

// D2粗粒化: 全層分 [a,b) トークンを1ワーカーで連続処理。
// 従来は層ごとにfork/join (6回/step) だったが、トークン並列は層間で
// 依存がない (acts[l+1][t]は同ワーカーのacts[l][t]からのみ決まる) ため
// 単一fork/joinに融合できる。各トークンの計算内容・順序は不変で
// ビット一致。ワーカーあたり作業量は約6倍 (12Tで約200KB→約1.2MB)。
static void px_fwd_all_range(const px_ctx_t *restrict ctx, int a, int b,
                             int *restrict rc_out) {
    int rc = JT_OK;
    for (int l = 0; l < PX_LAYERS && rc == JT_OK; l++) {
        px_fwd_layer_range(ctx, l, a, b, &rc);
    }
    if (rc_out != NULL) {
        *rc_out = rc;
    }
}

// backward [a,b) トークン。勾配はGout[n_total]へ加算。dtmpは層最大3.3MB私用域。
static void px_bwd_range(const px_ctx_t *restrict ctx, size_t n_total,
                         int a, int b, float *restrict Gout,
                         float *restrict dtmp, int *restrict rc_out) {
    int rc = JT_OK;
    (void)n_total;
    const float *Wo = ctx->P + ctx->head_off;
    float *GWo = Gout + ctx->head_off;
    float *Gbo = Gout + ctx->head_off + (size_t)PX_N;
    const float *alast =
        ctx->acts + (size_t)PX_LAYERS * (size_t)PX_TOKS * (size_t)PX_N;
    for (int t = a; t < b && rc == JT_OK; t++) {
        const float *al = alast + (size_t)t * (size_t)PX_N;
        double pred = (double)ctx->P[ctx->head_off + (size_t)PX_N];
        for (int j = 0; j < PX_N; j++) {
            pred += (double)al[j] * (double)Wo[j];
        }
        {
            float d =
                (float)(2.0 * (pred - (double)ctx->Tdata[t]) / (double)PX_TOKS);
            float dcur[PX_N];
            for (int j = 0; j < PX_N; j++) {
                GWo[j] += d * al[j];
                dcur[j] = d * Wo[j];
            }
            *Gbo += d;
            for (int l = PX_LAYERS - 1; l >= 0 && rc == JT_OK; l--) {
                const float *base = ctx->P + ctx->layer_off[(size_t)l];
                const float *Wgate = base;
                const float *Wg = Wgate + PX_P_WGATE;
                const float *Wu = Wg + PX_P_ROUTED;
                const float *Wd = Wu + PX_P_ROUTED;
                const float *Wgs = Wd + PX_P_ROUTED;
                const float *Wus = Wgs + PX_P_SHARED;
                const float *Wds = Wus + PX_P_SHARED;
                const float *xin =
                    ctx->acts +
                    ((size_t)l * (size_t)PX_TOKS + (size_t)t) * (size_t)PX_N;
                // fwd保存中間値の使い回し (再計算除去。同一値のためビット不変)。
                size_t cbase = ((size_t)l * (size_t)PX_TOKS + (size_t)t);
                const size_t *ids = ctx->Cids + cbase * (size_t)PX_K;
                const float *weights = ctx->Cw + cbase * (size_t)PX_K;
                const float *Gsel =
                    ctx->CGsel + cbase * (size_t)PX_K * (size_t)PX_H;
                const float *Usel =
                    ctx->CUsel + cbase * (size_t)PX_K * (size_t)PX_H;
                const float *Ysel =
                    ctx->CYsel + cbase * (size_t)PX_K * (size_t)PX_N;
                const float *Gs =
                    (PX_S > 0 && ctx->CGs != NULL)
                        ? (ctx->CGs +
                           cbase * (size_t)PX_S * (size_t)PX_H)
                        : NULL;
                const float *Us =
                    (PX_S > 0 && ctx->CUs != NULL)
                        ? (ctx->CUs +
                           cbase * (size_t)PX_S * (size_t)PX_H)
                        : NULL;
                float dXmoe[PX_N];
                {
                    float *tWgate = dtmp;
                    float *tWg = tWgate + PX_P_WGATE;
                    float *tWu = tWg + PX_P_ROUTED;
                    float *tWd = tWu + PX_P_ROUTED;
                    float *tWgs = tWd + PX_P_ROUTED;
                    float *tWus = tWgs + PX_P_SHARED;
                    float *tWds = tWus + PX_P_SHARED;
                    // unchecked: 重みはステップ冒頭で一括検証済み＋ステップ内不変、
                    // ids/weightsは同一ステップ内fwd産のため範囲検査を省略
                    // (同一計算核のためbit一致)。活性化NaNは内側途中ガード＋
                    // 更新前G全走査ガードで検出する。
                    int brc = jt_moe_bwd_unchecked(
                        dcur, xin, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds, ids,
                        weights, Gsel, Usel, Ysel, Gs, Us, dXmoe, tWgate,
                        tWg, tWu, tWd, tWgs, tWus, tWds, NULL, PX_N, PX_H,
                        PX_E, PX_K, PX_S);
                    if (brc != JT_OK) {
                        rc = brc;
                        break;
                    }
                    {
                        float *gbase =
                            Gout + ctx->layer_off[(size_t)l];
                        float *gWgate = gbase;
                        float *gWg = gWgate + PX_P_WGATE;
                        float *gWu = gWg + PX_P_ROUTED;
                        float *gWd = gWu + PX_P_ROUTED;
                        float *gWgs = gWd + PX_P_ROUTED;
                        float *gWus = gWgs + PX_P_SHARED;
                        float *gWds = gWus + PX_P_SHARED;
                        // スパース加算: 選択expert行のみ (非選択のtWはbwd契約
                        // で0のため加算不要。共有は常時発火で全行)。
                        // 要素毎のトークン方向の加算順序は密版と同一。
                        for (int p = 0; p < PX_K; p++) {
                            size_t e = ids[p];
                            size_t erow = e * (size_t)PX_H * (size_t)PX_N;
                            size_t egate = e * (size_t)PX_N;
                            size_t hhn = (size_t)PX_H * (size_t)PX_N;
                            for (size_t i = 0; i < hhn; i++) {
                                gWg[erow + i] += tWg[erow + i];
                                gWu[erow + i] += tWu[erow + i];
                                gWd[erow + i] += tWd[erow + i];
                            }
                            for (int j = 0; j < PX_N; j++) {
                                gWgate[egate + (size_t)j] +=
                                    tWgate[egate + (size_t)j];
                            }
                        }
                        for (size_t i = 0; i < PX_P_SHARED; i++) {
                            gWgs[i] += tWgs[i];
                            gWus[i] += tWus[i];
                            gWds[i] += tWds[i];
                        }
                    }
                }
                for (int j = 0; j < PX_N; j++) {
                    double v = (double)dcur[j] + (double)dXmoe[j];
                    dcur[j] = (float)v;
                }
            }
        }
    }
    if (rc_out != NULL) {
        *rc_out = rc;
    }
}

// held-out val forwardのみ (更新なし)。Vactsは呼び出し側確保の作業域
// [(LAYERS+1)][VAL_TOKS][N]。検証ありjt_moe_fwdを使い、残差合算点で
// isfiniteを確認する。成功時はMSEを返し、失敗時はNANを返す。
// val由来の勾配計算・重み更新は一切行わない。
static float px_val_loss(const float *restrict P,
                         const size_t *restrict layer_off, size_t head_off,
                         const float *restrict Xv, const float *restrict Tv,
                         float *restrict Vacts) {
    if (P == NULL || layer_off == NULL || Xv == NULL || Tv == NULL ||
        Vacts == NULL) {
        return (float)NAN;
    }
    memcpy(Vacts, Xv,
           (size_t)PX_VAL_TOKS * (size_t)PX_N * sizeof(float));
    for (int l = 0; l < PX_LAYERS; l++) {
        const float *base = P + layer_off[(size_t)l];
        const float *Wgate = base;
        const float *Wg = Wgate + PX_P_WGATE;
        const float *Wu = Wg + PX_P_ROUTED;
        const float *Wd = Wu + PX_P_ROUTED;
        const float *Wgs = Wd + PX_P_ROUTED;
        const float *Wus = Wgs + PX_P_SHARED;
        const float *Wds = Wus + PX_P_SHARED;
        for (int t = 0; t < PX_VAL_TOKS; t++) {
            const float *xin =
                Vacts +
                ((size_t)l * (size_t)PX_VAL_TOKS + (size_t)t) * (size_t)PX_N;
            float *xout =
                Vacts +
                ((size_t)(l + 1) * (size_t)PX_VAL_TOKS + (size_t)t) *
                    (size_t)PX_N;
            float M[PX_N];
            size_t ids[PX_K];
            float weights[PX_K];
            float Gsel[(size_t)PX_K * (size_t)PX_H];
            float Usel[(size_t)PX_K * (size_t)PX_H];
            float Ysel[(size_t)PX_K * (size_t)PX_N];
            float Gs[(size_t)PX_S * (size_t)PX_H];
            float Us[(size_t)PX_S * (size_t)PX_H];
            int frc = jt_moe_fwd(xin, Wgate, Wg, Wu, Wd, Wgs, Wus, Wds, M,
                                 PX_N, PX_H, PX_E, PX_K, PX_S, ids,
                                 weights, NULL, Gsel, Usel, Ysel, Gs, Us);
            if (frc != JT_OK) {
                return (float)NAN;
            }
            for (int j = 0; j < PX_N; j++) {
                double v =
                    (double)xin[j] + (double)PX_RESID * (double)M[j];
                if (!isfinite(v)) {
                    return (float)NAN;
                }
                xout[j] = (float)v;
            }
        }
    }
    {
        const float *Wo = P + head_off;
        float bo = P[head_off + (size_t)PX_N];
        const float *alast =
            Vacts + (size_t)PX_LAYERS * (size_t)PX_VAL_TOKS * (size_t)PX_N;
        double sum = 0.0;
        for (int t = 0; t < PX_VAL_TOKS; t++) {
            const float *a = alast + (size_t)t * (size_t)PX_N;
            double pred = (double)bo;
            for (int j = 0; j < PX_N; j++) {
                pred += (double)a[j] * (double)Wo[j];
            }
            double diff = pred - (double)Tv[t];
            sum += diff * diff;
        }
        return (float)(sum / (double)PX_VAL_TOKS);
    }
}

#if PX_HAVE_PTHREAD
typedef struct px_thr_arg {
    const px_ctx_t *ctx;
    size_t n_total;
    int layer;  // fwd用 (>=0)。bwd時は-1。
    int a;
    int b;
    float *Gout;  // bwd用 (fwd時はNULL)。
    float *dtmp;  // bwd用 (fwd時はNULL)。
    int rc;
    double fwd_sec; // F2 probe: 当該ワーカーのfwd wall (計測のみ)
    double bwd_sec; // F2 probe: 当該ワーカーのbwd wall (計測のみ)
} px_thr_arg_t;

// D2粗粒化用: 1ステップ分の fused forward全層 + backward を同一ワーカーで
// 処理 (単一fork/join/step)。Gpartゼロ埋めもワーカー側で並列化し、
// 主スレッドの直列memsetを除去する。トークン分割は従来と同一のため等価。
static void *px_thr_step(void *v) {
    px_thr_arg_t *arg = (px_thr_arg_t *)v;
    int rc = JT_OK;
    double t_f0 = 0.0;
    double t_f1 = 0.0;
    double t_b0 = 0.0;
    double t_b1 = 0.0;
    if (arg->Gout != NULL && arg->n_total > 0) {
        memset(arg->Gout, 0, arg->n_total * sizeof(float));
    }
    t_f0 = px_now_sec();
    px_fwd_all_range(arg->ctx, arg->a, arg->b, &rc);
    t_f1 = px_now_sec();
    arg->fwd_sec = t_f1 - t_f0;
    if (rc == JT_OK) {
        t_b0 = px_now_sec();
        px_bwd_range(arg->ctx, arg->n_total, arg->a, arg->b, arg->Gout,
                     arg->dtmp, &rc);
        t_b1 = px_now_sec();
        arg->bwd_sec = t_b1 - t_b0;
    } else {
        arg->bwd_sec = 0.0;
    }
    arg->rc = rc;
    return NULL;
}
#endif

// F2 probe: 10フェーズ内訳の表示 (計測のみ。計算内容・順序に触れない)。
// other残差により Σphase = Σstep_wall が成立する。cswのfwd/bwd/sync按分は
// wall比の近似 (getrusageはプロセス粒度のため)。macOSのinblock/oublockは
// 常に0の見込みで、esmoe tmpfile量はstats+register計数で代替する。
static void px_print_profile(long nthr, long steps_done, double run_wall,
                             const jt_esmoe_stats_t *es0,
                             const jt_esmoe_stats_t *es1, px_snap_t run0) {
    px_snap_t run1 = px_snap_now();
    double sum = 0.0;
    double step_avg = 0.0;
    double sum_ms = 0.0;
    double step_ms = 0.0;
    if (steps_done <= 0) {
        steps_done = 1;
    }
    for (int q = 0; q < PX_PH_N; q++) {
        sum += px_ph_acc[q];
    }
    step_avg =
        (steps_done > 0) ? px_step_wall_acc / (double)steps_done : 0.0;
    sum_ms = (steps_done > 0) ? sum / (double)steps_done * 1000.0 : 0.0;
    step_ms = step_avg * 1000.0;
    fprintf(stderr,
            "train_proxy100m [F2-profile]: threads=%ld steps=%ld "
            "run_wall=%.3fs\n",
            nthr, steps_done, run_wall);
    fprintf(stderr,
            "train_proxy100m [F2-profile]: %-14s %10s %7s %10s %10s\n",
            "phase", "ms/step", "pct", "csw_v/step", "csw_iv/step");
    for (int q = 0; q < PX_PH_N; q++) {
        double ms = px_ph_acc[q] / (double)steps_done * 1000.0;
        double pct = (sum > 0.0) ? px_ph_acc[q] / sum * 100.0 : 0.0;
        double cv = (double)px_ph_cswv[q] / (double)steps_done;
        double ci = (double)px_ph_csiv[q] / (double)steps_done;
        fprintf(stderr,
                "train_proxy100m [F2-profile]: %-14s %10.3f %6.2f%% "
                "%10.1f %10.1f\n",
                px_ph_name[q], ms, pct, cv, ci);
    }
    fprintf(stderr,
            "train_proxy100m [F2-profile]: check sum_phases_ms/step=%.3f "
            "step_wall_avg_ms/step=%.3f diff_us=%.2f\n",
            sum_ms, step_ms, (sum_ms - step_ms) * 1000.0);
    fprintf(stderr,
            "train_proxy100m [F2-profile]: logging_detail loss_ms=%.3f "
            "sticky_ms=%.3f val_ms=%.3f misc_ms=%.3f (per step avg)\n",
            px_log_loss / (double)steps_done * 1000.0,
            px_log_sticky / (double)steps_done * 1000.0,
            px_log_val / (double)steps_done * 1000.0,
            px_log_misc / (double)steps_done * 1000.0);
    {
        uint64_t rd1 = (es1 != NULL) ? es1->bytes_read : 0ULL;
        uint64_t rd0 = (es0 != NULL) ? es0->bytes_read : 0ULL;
        uint64_t ld1 = (es1 != NULL) ? es1->loads : 0ULL;
        uint64_t ld0 = (es0 != NULL) ? es0->loads : 0ULL;
        long ckpt_ev = steps_done / (long)PX_CKPT_EVERY;
        long esmoe_ev = steps_done / (long)PX_ESMOE_EVERY;
        double cps =
            (steps_done > 0)
                ? (double)px_tsc_acc / (double)steps_done
                : 0.0;
        double ghz =
            (step_avg > 0.0 && PX_HAVE_RDTSC) ? cps / step_avg / 1e9 : 0.0;
        fprintf(stderr,
                "train_proxy100m [F2-profile]: io run_inblock=%ld "
                "run_oublock=%ld esmoe_read_bytes=%llu esmoe_loads=%llu "
                "esmoe_write_bytes=%llu ckpt_events=%ld esmoe_events=%ld\n",
                run1.inb - run0.inb, run1.oub - run0.oub,
                (unsigned long long)(rd1 - rd0),
                (unsigned long long)(ld1 - ld0),
                (unsigned long long)px_esmoe_write_bytes, ckpt_ev,
                esmoe_ev);
        fprintf(stderr,
                "train_proxy100m [F2-profile]: tsc cycles/step=%.0f "
                "tsc_GHz_est=%.3f rdtsc=%d (参考値。fenceなし)\n",
                cps, ghz, PX_HAVE_RDTSC);
    }
}

int main(int argc, char **argv) {
    int rc_all = 1;
    long max_steps = 1073741824L;
    double time_limit = 6000.0;
    float lr = PX_LR_DEFAULT;
    float *P = NULL;
    float *G = NULL;
    float *X = NULL;
    float *T = NULL;
    float *acts = NULL;
    float *dtmp = NULL;
    float *Xv = NULL;    // held-out val入力 [(VAL_TOKS)][N] (更新に不使用)
    float *Tv = NULL;    // held-out val目標 [VAL_TOKS] (更新に不使用)
    float *Vacts = NULL; // val forward作業域 [(LAYERS+1)][VAL_TOKS][N]
    float *Gpart = NULL;   // [nthr][n_total] スレッド別勾配部分和
    float *dtmps = NULL;   // [nthr][P_LAYER] スレッド別bwd私用域
    // fwd中間値キャッシュ (bwd再計算除去用。[L][TOKS] 単位)。
    size_t *Cids = NULL;   // [L][TOKS][K]
    float *Cw = NULL;      // [L][TOKS][K]
    float *CGsel = NULL;   // [L][TOKS][K*H]
    float *CUsel = NULL;   // [L][TOKS][K*H]
    float *CYsel = NULL;   // [L][TOKS][K*N]
    float *CGs = NULL;     // [L][TOKS][S*H] (S==0時はNULLのまま)
    float *CUs = NULL;     // [L][TOKS][S*H] (S==0時はNULLのまま)
    float *Gates = NULL;   // [L][TOKS][E] dense gate分布 (sticky実経路用)
    long nthr_req = 0;     // 0=自動
    long nthr = 1;
    size_t layer_off[(size_t)PX_LAYERS + 1];
    size_t n_total = 0;
    size_t head_off = 0;
    jt_optim8_t opt;
    jt_esmoe_t es;
    int opt_inited = 0;
    int es_inited = 0;
    int n_seg = 0;
    int bounds[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    double t0 = 0.0;
    long step = 0;
    float w_star[PX_N];
    float val_loss = (float)NAN;  // 直近のval計測値 (毎ステップログ列へ)
    float val_init = (float)NAN;  // 開始前val (単調減少確認用)
    float sticky_loss = 0.0f;     // 直近のsticky loss層平均 (毎ステップ計測)
    long csw_prev_v = 0L;  // D2: 前ステップのru_nvcsw (自発的)
    long csw_prev_iv = 0L; // D2: 前ステップのru_nivcsw (非自発的)
    int profile = 0;       // F2 probe: --profileで10フェーズ内訳を出す
    int no_offload = 0; // F3-1: --no-offloadでESMoE往復を無効化
    int no_ckpt = 0;    // F3-2: --no-ckptでckpt recompute照合を無効化
    int no_sched = 0;   // F3-5: --no-schedでlrスケジュール無効化 (固定lr)
    int gen_stream = 1; // F4: 1=stream既定 (汎化課題), 0=fixed (旧来暗記可能)
    px_snap_t prof_run0;   // F2 probe: run全体のio差分開始点
    jt_esmoe_stats_t prof_es0; // F2 probe: esmoe stats開始点

    memset(&opt, 0, sizeof(opt));
    memset(&es, 0, sizeof(es));
    memset(layer_off, 0, sizeof(layer_off));

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--steps") == 0 ||
             strcmp(argv[i], "--max-steps") == 0) &&
            i + 1 < argc) {
            max_steps = atol(argv[++i]);
        } else if ((strcmp(argv[i], "--time") == 0 ||
                    strcmp(argv[i], "--time-limit") == 0) &&
                   i + 1 < argc) {
            time_limit = atof(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            lr = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            nthr_req = atol(argv[++i]);
        } else if (strcmp(argv[i], "--profile") == 0) {
            profile = 1;
        } else if (strcmp(argv[i], "--no-offload") == 0) {
            no_offload = 1;
        } else if (strcmp(argv[i], "--no-ckpt") == 0) {
            no_ckpt = 1;
        } else if (strcmp(argv[i], "--no-sched") == 0) {
            no_sched = 1;
        } else if (strcmp(argv[i], "--gen-task") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "stream") == 0) {
                gen_stream = 1;
            } else if (strcmp(argv[i], "fixed") == 0) {
                gen_stream = 0;
            } else {
                fprintf(stderr, "train_proxy100m: unknown --gen-task '%s' "
                                "(want stream|fixed)\n",
                        argv[i]);
                px_usage(argv[0]);
                errno = EINVAL;
                goto cleanup;
            }
        } else if (strcmp(argv[i], "-h") == 0 ||
                   strcmp(argv[i], "--help") == 0) {
            px_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "train_proxy100m: unknown arg '%s'\n", argv[i]);
            px_usage(argv[0]);
            errno = EINVAL;
            goto cleanup;
        }
    }
    if (max_steps <= 0 || !(time_limit > 0.0) || !(lr > 0.0f) ||
        !isfinite(lr) || nthr_req < 0 || nthr_req > 16) {
        fprintf(stderr, "train_proxy100m: invalid limits/args\n");
        errno = EINVAL;
        goto cleanup;
    }

    // ---- スレッド数 (F3-3: 既定=6。F2で4〜6Tが最速、online(12T)はsync増で
    // 後退のため過剰)。同一nthrでは静的分割+順序付きreductionのため決定的。
    // 明示--threads指定は1〜16で上書き可能。
    nthr = nthr_req;
    if (nthr == 0) {
        nthr = 6L;
    }
#if !PX_HAVE_PTHREAD
    nthr = 1L;
#endif
    if (nthr < 1L) {
        nthr = 1L;
    }
    if (nthr > 16L) {
        nthr = 16L;
    }

    // ---- パラメータ量と常駐見積り ----
    for (int l = 0; l < PX_LAYERS; l++) {
        layer_off[(size_t)l] = (size_t)l * PX_P_LAYER;
    }
    layer_off[(size_t)PX_LAYERS] = (size_t)PX_LAYERS * PX_P_LAYER;
    head_off = layer_off[(size_t)PX_LAYERS];
    n_total = head_off + PX_P_HEAD;
    {
        double w_bytes = (double)n_total * 4.0;
        double g_bytes = (double)n_total * 4.0;
        size_t nblocks = jt_optim8_nblocks(n_total);
        double o_bytes =
            (double)n_total * 2.0 + (double)nblocks * 8.0;
        double a_bytes = (double)(PX_LAYERS + 1) * (double)PX_TOKS *
                         (double)PX_N * 4.0;
        double t_bytes = (double)nthr * (double)n_total * 4.0 +
                         (double)nthr * (double)PX_P_LAYER * 4.0;
        // fwd中間値キャッシュ: [L][TOKS]×(ids/weights + K*H×2 + K*N + S*H×2)。
        double c_bytes =
            (double)PX_LAYERS * (double)PX_TOKS *
            ((double)(PX_K * sizeof(size_t)) +
             (double)((size_t)PX_K + (size_t)2 * (size_t)PX_K * (size_t)PX_H +
                      (size_t)PX_K * (size_t)PX_N +
                      (size_t)2 * (size_t)PX_S * (size_t)PX_H) *
                 4.0);
        // held-out val: Xv/Tv + forward作業域Vacts。
        double v_bytes = (double)PX_VAL_TOKS * (double)PX_N * 4.0 +
                         (double)PX_VAL_TOKS * 4.0 +
                         (double)(PX_LAYERS + 1) * (double)PX_VAL_TOKS *
                             (double)PX_N * 4.0;
        double est = w_bytes + g_bytes + o_bytes + a_bytes + t_bytes +
                     c_bytes + v_bytes;
        printf("train_proxy100m: layers=%d d=%d E=%d h=%d k=%d S=%d "
               "seq=%d batch=%d toks/step=%d fp32 threads=%ld\n",
               PX_LAYERS, PX_N, PX_E, PX_H, PX_K, PX_S, PX_SEQ, PX_BATCH,
               PX_TOKS, nthr);
        printf("train_proxy100m: params=%zu weight=%.2fMiB grad=%.2fMiB "
               "optim=%.2fMiB acts=%.2fMiB thr=%.2fMiB cache=%.2fMiB "
               "val=%.2fMiB est_resident=%.2fMiB\n",
               n_total, w_bytes / 1048576.0, g_bytes / 1048576.0,
               o_bytes / 1048576.0, a_bytes / 1048576.0,
               t_bytes / 1048576.0, c_bytes / 1048576.0,
                v_bytes / 1048576.0, est / 1048576.0);
        fflush(stdout);
        if (est > PX_MAX_GB * 1024.0 * 1024.0 * 1024.0) {
            fprintf(stderr, "train_proxy100m: est %.2fMiB exceeds %.0fGB\n",
                    est / 1048576.0, PX_MAX_GB);
            errno = ENOMEM;
            goto cleanup;
        }
    }

    // ---- 解析目標の会計＋AI概算 (起動時表示。失敗時はexit 1) ----
    if (px_print_scale() != 0) {
        goto cleanup;
    }

    P = (float *)malloc(n_total * sizeof(float));
    G = (float *)malloc(n_total * sizeof(float));
    X = (float *)malloc((size_t)PX_TOKS * (size_t)PX_N * sizeof(float));
    T = (float *)malloc((size_t)PX_TOKS * sizeof(float));
    acts = (float *)malloc((size_t)(PX_LAYERS + 1) * (size_t)PX_TOKS *
                           (size_t)PX_N * sizeof(float));
    dtmp = (float *)malloc(PX_P_LAYER * sizeof(float));
    Xv = (float *)malloc((size_t)PX_VAL_TOKS * (size_t)PX_N * sizeof(float));
    Tv = (float *)malloc((size_t)PX_VAL_TOKS * sizeof(float));
    Vacts = (float *)malloc((size_t)(PX_LAYERS + 1) * (size_t)PX_VAL_TOKS *
                            (size_t)PX_N * sizeof(float));
    {
        size_t nct = (size_t)PX_LAYERS * (size_t)PX_TOKS;
        Cids = (size_t *)malloc(nct * (size_t)PX_K * sizeof(size_t));
        Cw = (float *)malloc(nct * (size_t)PX_K * sizeof(float));
        CGsel = (float *)malloc(nct * (size_t)PX_K * (size_t)PX_H *
                               sizeof(float));
        CUsel = (float *)malloc(nct * (size_t)PX_K * (size_t)PX_H *
                               sizeof(float));
        CYsel = (float *)malloc(nct * (size_t)PX_K * (size_t)PX_N *
                               sizeof(float));
        if (PX_S > 0) {
            CGs = (float *)malloc(nct * (size_t)PX_S * (size_t)PX_H *
                                  sizeof(float));
            CUs = (float *)malloc(nct * (size_t)PX_S * (size_t)PX_H *
                                  sizeof(float));
        }
        Gates = (float *)malloc(nct * (size_t)PX_E * sizeof(float));
    }
    if (nthr > 1) {
        Gpart =
            (float *)malloc((size_t)nthr * n_total * sizeof(float));
        dtmps =
            (float *)malloc((size_t)nthr * PX_P_LAYER * sizeof(float));
    }
    if (P == NULL || G == NULL || X == NULL || T == NULL || acts == NULL ||
        dtmp == NULL || Xv == NULL || Tv == NULL || Vacts == NULL ||
        Cids == NULL || Cw == NULL || CGsel == NULL || CUsel == NULL ||
        CYsel == NULL || (PX_S > 0 && (CGs == NULL || CUs == NULL)) ||
        Gates == NULL ||
        (nthr > 1 && (Gpart == NULL || dtmps == NULL))) {
        fprintf(stderr, "train_proxy100m: OOM\n");
        errno = ENOMEM;
        goto cleanup;
    }

    // ---- 決定論的初期化 ----
    // F4汎化課題の真関数 w_star: sparse (64/1024次元のみ非ゼロ。j%16==0)。
    // dense旧来 (全1024次元×0.5) と目標分散を一致 (kC^2/144: 64*4/144≈1.78)
    // させ val_init を同等に保ちつつ、有効次元を絞って200ステップ以内の
    // 汎化学習を可能にする。非ゼロ値は決定論的ハッシュ。stream/fixed共通。
    for (int j = 0; j < PX_N; j++) {
        w_star[j] = (j % 16 == 0) ? 2.0f * px_pat(j, 0, 101) : 0.0f;
    }
    for (int l = 0; l < PX_LAYERS; l++) {
        float *base = P + layer_off[(size_t)l];
        float *Wgate = base;
        float *Wg = Wgate + PX_P_WGATE;
        float *Wu = Wg + PX_P_ROUTED;
        float *Wd = Wu + PX_P_ROUTED;
        float *Wgs = Wd + PX_P_ROUTED;
        float *Wus = Wgs + PX_P_SHARED;
        float *Wds = Wus + PX_P_SHARED;
        for (size_t i = 0; i < PX_P_WGATE; i++) {
            int e = (int)(i / (size_t)PX_N);
            int j = (int)(i % (size_t)PX_N);
            Wgate[i] = 0.05f * px_pat(e, j, 300 + l);
        }
        for (size_t i = 0; i < PX_P_ROUTED; i++) {
            Wg[i] = 0.05f * px_pat((int)i, 0, 310 + l);
            Wu[i] = 0.05f * px_pat((int)i, 1, 320 + l);
            Wd[i] = 0.05f * px_pat((int)i, 2, 330 + l);
        }
        for (size_t i = 0; i < PX_P_SHARED; i++) {
            Wgs[i] = 0.05f * px_pat((int)i, 0, 340 + l);
            Wus[i] = 0.05f * px_pat((int)i, 1, 350 + l);
            Wds[i] = 0.05f * px_pat((int)i, 2, 360 + l);
        }
    }
    {
        float *Wo = P + head_off;
        for (int j = 0; j < PX_N; j++) {
            Wo[j] = 0.0f;
        }
        P[head_off + (size_t)PX_N] = 0.0f;
    }
    // 合成回帰データ: T = dot(X, w_star)/4。train/val分割はpx_fill_train。
    // fixed時はここで1回だけ固定64点を生成。stream時は毎ステップ先頭で
    // 新規64点を生成する (ループ内)。valは常に固定held-out (更新に不使用)。
    px_fill_train(X, T, w_star, 0L, gen_stream);
    for (int t = 0; t < PX_VAL_TOKS; t++) {
        double acc = 0.0;
        for (int j = 0; j < PX_N; j++) {
            float v = px_pat(t + PX_VAL_OFFSET, j, 7);
            Xv[(size_t)t * (size_t)PX_N + (size_t)j] = v;
            acc += (double)v * (double)w_star[j];
        }
        Tv[t] = (float)(acc / 4.0);
    }

    // ---- optim8 ----
    {
        jt_optim8_cfg_t cfg;
        jt_optim8_cfg_default(&cfg);
        cfg.lr = lr;
        if (jt_optim8_init(&opt, n_total, &cfg) != JT_OK) {
            fprintf(stderr, "train_proxy100m: optim init failed\n");
            goto cleanup;
        }
        opt_inited = 1;
    }

    // ---- checkpoint計画 ----
    if (jt_ckpt_num_segments(PX_LAYERS, &n_seg) != JT_OK) {
        fprintf(stderr, "train_proxy100m: ckpt segments failed\n");
        goto cleanup;
    }
    if (n_seg + 1 > 16 ||
        jt_ckpt_boundaries(PX_LAYERS, n_seg, bounds, (size_t)(n_seg + 1)) !=
            JT_OK) {
        fprintf(stderr, "train_proxy100m: ckpt boundaries failed\n");
        goto cleanup;
    }

    // ---- esmoe (layer0 Wd形状: 16 experts × H*N floats) ----
    // F3-1: --no-offload時は全常駐 (物理28M) のためoffload自体が不要。
    // 初期化・往復照合を共にスキップし、その旨をログに明示する。
    if (no_offload) {
        fprintf(stderr,
                "train_proxy100m: esmoe offload disabled (--no-offload): "
                "28M resident, skipping register/prefetch/load roundtrip "
                "check\n");
    } else {
        jt_esmoe_cfg_t ecfg;
        ecfg.n_experts = (uint32_t)PX_E;
        ecfg.expert_bytes = (size_t)PX_H * (size_t)PX_N * sizeof(float);
        ecfg.row_bytes = (size_t)PX_N * sizeof(float);
        if (jt_esmoe_init(&es, &ecfg) != JT_OK) {
            fprintf(stderr, "train_proxy100m: esmoe init failed\n");
            goto cleanup;
        }
        es_inited = 1;
    }

    fprintf(stderr,
            "train_proxy100m: start max_steps=%ld time_limit=%.0fs lr=%g%s%s%s "
            "gen-task=%s\n",
            max_steps, time_limit, (double)lr,
            no_offload ? " --no-offload" : "",
            no_ckpt ? " --no-ckpt" : "", no_sched ? " --no-sched" : "",
            gen_stream ? "stream" : "fixed");
    if (no_ckpt) {
        fprintf(stderr,
                "train_proxy100m: ckpt recompute check disabled (--no-ckpt): "
                "skipping recompute verification\n");
    }
    if (no_sched) {
        fprintf(stderr,
                "train_proxy100m: lr schedule disabled (--no-sched): fixed "
                "lr=%g\n",
                (double)lr);
    } else {
        fprintf(stderr,
                "train_proxy100m: lr schedule warmup=%ld + cosine decay "
                "(base lr=%g, T=%ld)\n",
                PX_LR_WARMUP, (double)lr, max_steps);
    }
    // 開始前val (初期値記録。forwardのみ・更新なし)。
    val_init = px_val_loss(P, layer_off, head_off, Xv, Tv, Vacts);
    if (!isfinite((double)val_init)) {
        fprintf(stderr, "train_proxy100m: initial val loss non-finite\n");
        goto cleanup;
    }
    val_loss = val_init;
    fprintf(stderr, "train_proxy100m: val_init=%.6f (held-out %d toks)\n",
            (double)val_init, PX_VAL_TOKS);
    t0 = px_now_sec();
    px_get_csw(&csw_prev_v, &csw_prev_iv);
    // F2 probe: run全体のio差分開始点 (計測のみ)。
    prof_run0 = px_snap_now();
    memset(&prof_es0, 0, sizeof(prof_es0));
    if (es_inited) {
        if (jt_esmoe_stats(&es, &prof_es0) != JT_OK) {
            memset(&prof_es0, 0, sizeof(prof_es0));
        }
    }

    for (step = 0; step < max_steps; step++) {
        double loss_sum = 0.0;
        // F2 probe: ステップ境界 (計測のみ)。
        px_snap_t s_step0 = px_snap_now();
        uint64_t tsc0 = px_rdtsc();
        double ph0[PX_PH_N];
        long cv0[PX_PH_N];
        long ci0[PX_PH_N];
        long ib0[PX_PH_N];
        long ob0[PX_PH_N];
        for (int pq = 0; pq < PX_PH_N; pq++) {
            ph0[pq] = px_ph_acc[pq];
            cv0[pq] = px_ph_cswv[pq];
            ci0[pq] = px_ph_csiv[pq];
            ib0[pq] = px_ph_inb[pq];
            ob0[pq] = px_ph_oub[pq];
        }
        float loss = 0.0f;
        double gnorm = 0.0;
        double el = 0.0;
        // F4 stream: ステップ毎に新規train64点を決定論生成 (暗記無効化)。
        // fixed時は初期生成のまま (旧来動作)。valには一切触れない。
        if (gen_stream && step > 0) {
            px_fill_train(X, T, w_star, step, 1);
        }
        // ---- ステップ冒頭: 重み全体を1回だけ検証 (unchecked区間の前提)。
        // 以降のトークン×層 (3072コール/ステップ) のper-call重みスキャンを
        // 省略する。ステップ内P不変のため等価。非有限時はfail-closed abort。
        // F2 probe: 当該検証はother相へ計上 (計測のみ)。
        {
            px_snap_t s_w = px_snap_now();
            int wf_ok = px_weights_finite(P, n_total);
            px_prof_acc(PX_PH_OTHER, s_w, px_snap_now());
            if (!wf_ok) {
                fprintf(stderr,
                        "train_proxy100m: non-finite weights step=%ld\n",
                        step);
                goto cleanup;
            }
        }
        // ---- forward: acts[0]=X, acts[l+1]=acts[l]+MoE(acts[l]) ----
        // F2 probe: data相 (計測のみ)。
        {
            px_snap_t s_d = px_snap_now();
            memcpy(acts, X,
                   (size_t)PX_TOKS * (size_t)PX_N * sizeof(float));
            px_prof_acc(PX_PH_DATA, s_d, px_snap_now());
        }
        {
            px_ctx_t wctx;
            wctx.P = P;
            wctx.acts = acts;
            wctx.Xdata = X;
            wctx.Tdata = T;
            wctx.layer_off = layer_off;
            wctx.head_off = head_off;
            wctx.Cids = Cids;
            wctx.Cw = Cw;
            wctx.CGsel = CGsel;
            wctx.CUsel = CUsel;
            wctx.CYsel = CYsel;
            wctx.CGs = CGs;
            wctx.CUs = CUs;
            // D2粗粒化: forward全層+bwdを単一fork/join/stepに融合 (従来7回→1回)。
            // 各ワーカーがトークン範囲[a,b)をfwd全層→bwd連続処理し、
            // acts/キャッシュがL2に残る間の再利用を狙う。分割は従来と同一で等価。
#if PX_HAVE_PTHREAD
            if (nthr > 1) {
                pthread_t thrs[16];
                px_thr_arg_t args[16];
                // F2 probe: 融合区間のwall/csw (計測のみ)。
                px_snap_t s_fb0 = px_snap_now();
                double fwd_max = 0.0;
                double bwd_max = 0.0;
                size_t chunk = ((size_t)PX_TOKS + (size_t)nthr - 1) /
                               (size_t)nthr;
                int bad = 0;
                for (long w = 0; w < nthr; w++) {
                    size_t a = (size_t)w * chunk;
                    size_t b = a + chunk;
                    if (b > (size_t)PX_TOKS) {
                        b = (size_t)PX_TOKS;
                    }
                    args[w].ctx = &wctx;
                    args[w].n_total = n_total;
                    args[w].layer = -2;
                    args[w].a = (int)a;
                    args[w].b = (int)b;
                    args[w].Gout = Gpart + (size_t)w * n_total;
                    args[w].dtmp = dtmps + (size_t)w * PX_P_LAYER;
                    args[w].rc = JT_OK;
                    args[w].fwd_sec = 0.0;
                    args[w].bwd_sec = 0.0;
                    if (pthread_create(&thrs[w], NULL, px_thr_step,
                                       &args[w]) != 0) {
                        fprintf(stderr,
                                "train_proxy100m: pthread_create failed\n");
                        errno = ENOMEM;
                        goto cleanup;
                    }
                }
                for (long w = 0; w < nthr; w++) {
                    pthread_join(thrs[w], NULL);
                    if (args[w].rc != JT_OK) {
                        bad = args[w].rc;
                    }
                }
                if (bad != JT_OK) {
                    fprintf(stderr,
                            "train_proxy100m: fwd/bwd failed step=%ld rc=%d\n",
                            step, bad);
                    goto cleanup;
                }
                // 順序付きreduction (決定的)。
                memset(G, 0, n_total * sizeof(float));
                for (long w = 0; w < nthr; w++) {
                    const float *gp = Gpart + (size_t)w * n_total;
                    for (size_t i = 0; i < n_total; i++) {
                        G[i] += gp[i];
                    }
                }
                // F2 probe: 融合区間を fwd=max(worker fwd)/bwd=max(worker bwd)/
                // sync=残差(fork/join+reduction+create overhead)に分解。
                // cswはwall按分 (近似。脚注参照)。計算内容・順序は不変。
                {
                    px_snap_t s_fb1 = px_snap_now();
                    double fused_w = s_fb1.t - s_fb0.t;
                    long dv = s_fb1.v - s_fb0.v;
                    long div = s_fb1.iv - s_fb0.iv;
                    long din = s_fb1.inb - s_fb0.inb;
                    long dout = s_fb1.oub - s_fb0.oub;
                    double fw = 0.0;
                    double bw = 0.0;
                    double sy = 0.0;
                    for (long w = 0; w < nthr; w++) {
                        if (args[w].fwd_sec > fwd_max) {
                            fwd_max = args[w].fwd_sec;
                        }
                        if (args[w].bwd_sec > bwd_max) {
                            bwd_max = args[w].bwd_sec;
                        }
                    }
                    fw = fwd_max;
                    bw = bwd_max;
                    if (fw < 0.0) {
                        fw = 0.0;
                    }
                    if (bw < 0.0) {
                        bw = 0.0;
                    }
                    if (fused_w <= 0.0) {
                        fw = 0.0;
                        bw = 0.0;
                        sy = 0.0;
                    } else if (fw + bw > fused_w) {
                        double sc = fused_w / (fw + bw);
                        fw *= sc;
                        bw *= sc;
                        sy = 0.0;
                    } else {
                        sy = fused_w - fw - bw;
                    }
                    px_ph_acc[PX_PH_FWD] += fw;
                    px_ph_acc[PX_PH_BWD] += bw;
                    px_ph_acc[PX_PH_SYNC] += sy;
                    if (fused_w > 0.0) {
                        px_ph_cswv[PX_PH_FWD] += (long)(dv * (fw / fused_w));
                        px_ph_cswv[PX_PH_BWD] += (long)(dv * (bw / fused_w));
                        px_ph_cswv[PX_PH_SYNC] +=
                            dv - (long)(dv * (fw / fused_w)) -
                            (long)(dv * (bw / fused_w));
                        px_ph_csiv[PX_PH_FWD] += (long)(div * (fw / fused_w));
                        px_ph_csiv[PX_PH_BWD] += (long)(div * (bw / fused_w));
                        px_ph_csiv[PX_PH_SYNC] +=
                            div - (long)(div * (fw / fused_w)) -
                            (long)(div * (bw / fused_w));
                        px_ph_inb[PX_PH_SYNC] += din;
                        px_ph_oub[PX_PH_SYNC] += dout;
                    } else {
                        px_ph_cswv[PX_PH_SYNC] += dv;
                        px_ph_csiv[PX_PH_SYNC] += div;
                        px_ph_inb[PX_PH_SYNC] += din;
                        px_ph_oub[PX_PH_SYNC] += dout;
                    }
                }
            } else
#endif
            {
                int frc = JT_OK;
                // F2 probe: fwd相 (計測のみ)。
                {
                    px_snap_t s_f = px_snap_now();
                    px_fwd_all_range(&wctx, 0, PX_TOKS, &frc);
                    px_prof_acc(PX_PH_FWD, s_f, px_snap_now());
                }
                if (frc != JT_OK) {
                    fprintf(stderr,
                            "train_proxy100m: fwd failed step=%ld rc=%d\n",
                            step, frc);
                    goto cleanup;
                }
                {
                    int brc = JT_OK;
                    // F2 probe: bwd相 (G zero含む。計測のみ)。
                    px_snap_t s_b = px_snap_now();
                    memset(G, 0, n_total * sizeof(float));
                    px_bwd_range(&wctx, n_total, 0, PX_TOKS, G, dtmp, &brc);
                    px_prof_acc(PX_PH_BWD, s_b, px_snap_now());
                    if (brc != JT_OK) {
                        fprintf(stderr,
                                "train_proxy100m: bwd failed step=%ld rc=%d\n",
                                step, brc);
                        goto cleanup;
                    }
                }
            }
        }
        // ---- loss (MSE) ----
        // F2 probe: logging相へ計上 (計測のみ)。
        {
            px_snap_t s_l0 = px_snap_now();
            const float *Wo = P + head_off;
            float bo = P[head_off + (size_t)PX_N];
            const float *alast =
                acts + (size_t)PX_LAYERS * (size_t)PX_TOKS * (size_t)PX_N;
            for (int t = 0; t < PX_TOKS; t++) {
                const float *a = alast + (size_t)t * (size_t)PX_N;
                double pred = (double)bo;
                for (int j = 0; j < PX_N; j++) {
                    pred += (double)a[j] * (double)Wo[j];
                }
                double diff = pred - (double)T[t];
                loss_sum += diff * diff;
            }
            loss = (float)(loss_sum / (double)PX_TOKS);
            {
                px_snap_t s_l1 = px_snap_now();
                px_prof_acc(PX_PH_LOGGING, s_l0, s_l1);
                px_log_loss += s_l1.t - s_l0.t;
            }
        }
        if (!isfinite(loss)) {
            fprintf(stderr, "train_proxy100m: non-finite loss step=%ld\n",
                    step);
            goto cleanup;
        }

        // ---- sticky loss実経路 (fwd産の実ids/weightsをdense gate化) ----
        // ダミーgate禁止: Cids/Cw (実top-k結果) 由来のみを使う。共有expertは
        // routing.h方針で対象外 (routedのみ)。損失合算・逆伝播はTODOのため
        // 測定・記録のみ (scaffold段階の実経路確認)。
        // F2 probe: logging相へ計上 (計測のみ)。
        {
            px_snap_t s_s0 = px_snap_now();
            double ssum = 0.0;
            for (int l = 0; l < PX_LAYERS; l++) {
                for (int t = 0; t < PX_TOKS; t++) {
                    float *g =
                        Gates +
                        ((size_t)l * (size_t)PX_TOKS + (size_t)t) *
                            (size_t)PX_E;
                    size_t cbase =
                        ((size_t)l * (size_t)PX_TOKS + (size_t)t);
                    const size_t *ids = Cids + cbase * (size_t)PX_K;
                    const float *ww = Cw + cbase * (size_t)PX_K;
                    for (int e = 0; e < PX_E; e++) {
                        g[e] = 0.0f;
                    }
                    for (int p = 0; p < PX_K; p++) {
                        g[ids[p]] = ww[p];
                    }
                }
            }
            for (int l = 0; l < PX_LAYERS; l++) {
                float sl = 0.0f;
                int src = jt_moe_sticky_seq_loss(
                    Gates +
                        (size_t)l * (size_t)PX_TOKS * (size_t)PX_E,
                    (size_t)PX_TOKS, (size_t)PX_E, PX_STICKY_LAMBDA,
                    PX_STICKY_ALPHA, (size_t)PX_STICKY_W, &sl);
                if (src != JT_OK || !isfinite((double)sl)) {
                    fprintf(stderr,
                            "train_proxy100m: sticky loss failed step=%ld "
                            "layer=%d rc=%d\n",
                            step, l, src);
                    goto cleanup;
                }
                ssum += (double)sl;
            }
            sticky_loss = (float)(ssum / (double)PX_LAYERS);
            {
                px_snap_t s_s1 = px_snap_now();
                px_prof_acc(PX_PH_LOGGING, s_s0, s_s1);
                px_log_sticky += s_s1.t - s_s0.t;
            }
        }

        // ---- ckpt recompute照合 (5ステップ毎、token0) ----
        // F3-2: --no-ckpt時はスキップ (数値には触れないためloss軌道不変)。
        // 照合スキップをログ明示 (起動時に1行)。
        if (no_ckpt) {
            /* skip: ckpt recompute verification disabled */
        } else if ((step + 1) % PX_CKPT_EVERY == 0) {
            // F2 probe: ckpt_recompute相 (計測のみ)。
            px_snap_t s_c0 = px_snap_now();
            jt_ckpt_plan_t plan;
            px_ckpt_ctx_t ctx;
            float bnd[16 * PX_N];
            float cur[PX_N];
            plan.n_layers = PX_LAYERS;
            plan.n_seg = n_seg;
            plan.bounds = bounds;
            if (jt_ckpt_plan_init(&plan) != JT_OK) {
                fprintf(stderr, "train_proxy100m: ckpt plan invalid\n");
                goto cleanup;
            }
            for (int s = 0; s <= n_seg; s++) {
                int bl = bounds[s];
                const float *a = acts +
                                 ((size_t)bl * (size_t)PX_TOKS) * (size_t)PX_N;
                memcpy(bnd + (size_t)s * (size_t)PX_N, a,
                       (size_t)PX_N * sizeof(float));
            }
            memset(&ctx, 0, sizeof(ctx));
            ctx.P = P;
            ctx.layer_off = layer_off;
            ctx.bnd = bnd;
            ctx.n_seg = n_seg;
            ctx.bounds = bounds;
            ctx.cur = cur;
            for (int s = 0; s < n_seg; s++) {
                memcpy(cur, bnd + (size_t)s * (size_t)PX_N,
                       (size_t)PX_N * sizeof(float));
                int rrc = jt_ckpt_recompute_range(&plan, s, px_ckpt_layer_fn,
                                                  &ctx);
                if (rrc != JT_OK) {
                    fprintf(stderr,
                            "train_proxy100m: ckpt recompute rc=%d step=%ld "
                            "seg=%d\n",
                            rrc, step, s);
                    goto cleanup;
                }
                {
                    const float *want = bnd + (size_t)(s + 1) * (size_t)PX_N;
                    float worst = 0.0f;
                    for (int j = 0; j < PX_N; j++) {
                        float e = fabsf(cur[j] - want[j]);
                        worst = (e > worst) ? e : worst;
                    }
                    if (!(worst <= PX_CKPT_TOL)) {
                        fprintf(stderr,
                                "train_proxy100m: ckpt mismatch step=%ld seg=%d "
                                "worst=%g\n",
                                step, s, (double)worst);
                        goto cleanup;
                    }
                }
            }
            px_prof_acc(PX_PH_CKPT, s_c0, px_snap_now());
        }

        // ---- esmoe往復 (5ステップ毎、layer0 Wd実重み) ----
        // F3-1: --no-offload時はスキップ (数値には触れないためloss軌道不変)。
        if (no_offload) {
            /* skip: 28M residentのため往復照合なし */
        } else if ((step + 1) % PX_ESMOE_EVERY == 0) {
            // F2 probe: esmoe_io相 (allocは分離。計測のみ)。
            px_snap_t s_e0 = px_snap_now();
            double alloc0 = px_ph_acc[PX_PH_ALLOC];
            const float *base0 = P + layer_off[0];
            const float *Wd0 = base0 + PX_P_WGATE + (size_t)2 * PX_P_ROUTED;
            const unsigned char *wdb = (const unsigned char *)Wd0;
            size_t eb = (size_t)PX_H * (size_t)PX_N * sizeof(float);
            for (uint32_t e = 0; e < (uint32_t)PX_E; e++) {
                if (jt_esmoe_register(&es, e, wdb + (size_t)e * eb, eb) !=
                    JT_OK) {
                    fprintf(stderr,
                            "train_proxy100m: esmoe register failed step=%ld\n",
                            step);
                    goto cleanup;
                }
                // F2 probe: backingへのpwrite量 (statsに無いため計測)。
                px_esmoe_write_bytes += (uint64_t)eb;
            }
            if (jt_esmoe_evict(&es, 1) != JT_OK) {
                fprintf(stderr, "train_proxy100m: esmoe evict failed\n");
                goto cleanup;
            }
            {
                uint32_t ids[2] = {1, 3};
                if (jt_esmoe_prefetch(&es, ids, 2) != JT_OK) {
                    fprintf(stderr,
                            "train_proxy100m: esmoe prefetch failed step=%ld\n",
                            step);
                    goto cleanup;
                }
            }
            for (uint32_t e = 0; e < (uint32_t)PX_E; e++) {
                unsigned char *out = NULL;
                // F2 probe: alloc相 (mallocのみ。計測のみ)。
                {
                    double t_a0 = px_now_sec();
                    out = (unsigned char *)malloc(eb ? eb : 1);
                    px_ph_acc[PX_PH_ALLOC] += px_now_sec() - t_a0;
                }
                if (out == NULL) {
                    fprintf(stderr, "train_proxy100m: esmoe OOM\n");
                    errno = ENOMEM;
                    goto cleanup;
                }
                int lrc = jt_esmoe_load(&es, e, out, eb);
                int cmp = (lrc == JT_OK)
                              ? memcmp(out, wdb + (size_t)e * eb, eb)
                              : 1;
                // F2 probe: alloc相 (freeのみ。計測のみ)。
                {
                    double t_f0 = px_now_sec();
                    free(out);
                    px_ph_acc[PX_PH_ALLOC] += px_now_sec() - t_f0;
                }
                if (lrc != JT_OK || cmp != 0) {
                    fprintf(stderr,
                            "train_proxy100m: esmoe roundtrip mismatch step=%ld "
                            "e=%u\n",
                            step, e);
                    goto cleanup;
                }
            }
            // F2 probe: esmoe_io = ブロックwall - alloc分。csw/io差分は
            // esmoe_ioへ全計上 (allocのcswは無視できる。計測のみ)。
            {
                px_snap_t s_e1 = px_snap_now();
                double alloc_d = px_ph_acc[PX_PH_ALLOC] - alloc0;
                px_ph_acc[PX_PH_ESMOE] += (s_e1.t - s_e0.t) - alloc_d;
                px_ph_cswv[PX_PH_ESMOE] += (s_e1.v - s_e0.v);
                px_ph_csiv[PX_PH_ESMOE] += (s_e1.iv - s_e0.iv);
                px_ph_inb[PX_PH_ESMOE] += (s_e1.inb - s_e0.inb);
                px_ph_oub[PX_PH_ESMOE] += (s_e1.oub - s_e0.oub);
            }
        }

        // ---- optim更新 (非有限ガード + global norm clip 1.0付き) ----
        // F3-5: lrスケジュール適用 (warmup 100 + cosine)。--no-sched時は固定。
        // opt.lrをステップ毎に更新するのみで、更新式自体は不変。
        // F2 probe: optim相 (計測のみ)。
        px_snap_t s_o0 = px_snap_now();
        float cur_lr = lr;
        if (!no_sched) {
            cur_lr = px_sched_lr(step, max_steps, lr);
        }
        opt.lr = cur_lr;
        {
            double acc = 0.0;
            for (size_t i = 0; i < n_total; i++) {
                if (!isfinite((double)G[i])) {
                    fprintf(stderr,
                            "train_proxy100m: non-finite grad step=%ld i=%zu\n",
                            step, i);
                    goto cleanup;
                }
                acc += (double)G[i] * (double)G[i];
            }
            gnorm = sqrt(acc);
            if (gnorm > 1.0) {
                float s = (float)(1.0 / gnorm);
                for (size_t i = 0; i < n_total; i++) {
                    G[i] *= s;
                }
                gnorm = 1.0;
            }
        }
        if (jt_optim8_step(&opt, P, G, n_total) != JT_OK) {
            fprintf(stderr, "train_proxy100m: optim step failed step=%ld\n",
                    step);
            goto cleanup;
        }
        px_prof_acc(PX_PH_OPTIM, s_o0, px_snap_now());

        // ---- held-out val計測 (forwardのみ。valで更新は絶対にしない)。
        // 5ステップ毎＋最終ステップで再計測し、それ以外は直近値を維持。
        // F2 probe: logging相へ計上 (計測のみ)。
        if ((step + 1) % PX_VAL_EVERY == 0 || step + 1 == max_steps) {
            px_snap_t s_v0 = px_snap_now();
            float vv =
                px_val_loss(P, layer_off, head_off, Xv, Tv, Vacts);
            if (!isfinite((double)vv)) {
                fprintf(stderr,
                        "train_proxy100m: non-finite val loss step=%ld\n",
                        step + 1);
                goto cleanup;
            }
            val_loss = vv;
            {
                px_snap_t s_v1 = px_snap_now();
                px_prof_acc(PX_PH_LOGGING, s_v0, s_v1);
                px_log_val += s_v1.t - s_v0.t;
            }
        }

        // ---- ログ ----
        // F2 probe: logging相 (計測のみ)。
        px_snap_t s_g0 = px_snap_now();
        el = px_now_sec() - t0;
        {
            double done = (double)(step + 1);
            double sps = (el > 0.0) ? done / el : 0.0;
            double tps = sps * (double)PX_TOKS;
            long csw_cur_v = 0L;
            long csw_cur_iv = 0L;
            long csw_dv = 0L;
            long csw_div = 0L;
            px_get_csw(&csw_cur_v, &csw_cur_iv);
            csw_dv = csw_cur_v - csw_prev_v;
            csw_div = csw_cur_iv - csw_prev_iv;
            if (csw_dv < 0L) {
                csw_dv = 0L;
            }
            if (csw_div < 0L) {
                csw_div = 0L;
            }
            csw_prev_v = csw_cur_v;
            csw_prev_iv = csw_cur_iv;
            fprintf(stderr,
                    "step=%ld loss=%.6f val_loss=%.6f sticky=%.6f lr=%.6g "
                    "steps_sec=%.3f "
                    "toks_sec=%.1f rss_mib=%.1f gnorm=%.4f elapsed=%.1fs "
                    "csw_v=%ld csw_iv=%ld\n",
                    step + 1, (double)loss, (double)val_loss,
                    (double)sticky_loss, (double)cur_lr, sps, tps,
                    px_rss_mib(), gnorm, el, csw_dv, csw_div);
        }
        px_snap_t s_g1 = px_snap_now();
        px_prof_acc(PX_PH_LOGGING, s_g0, s_g1);
        px_log_misc += s_g1.t - s_g0.t;
        // F2 probe: other残差 = step_wall - sum(他9相)。合計一致用 (計測のみ)。
        {
            px_snap_t s_step1 = px_snap_now();
            uint64_t tsc1 = px_rdtsc();
            double accounted = 0.0;
            long cv_acc = 0L;
            long ci_acc = 0L;
            long ib_acc = 0L;
            long ob_acc = 0L;
            px_tsc_acc += (tsc1 - tsc0);
            // 全相 (OTHER直計上分=weights_finite含む) を差し引くことで
            // Σphase = Σstep_wall を厳密に成立させる。
            for (int pq = 0; pq < PX_PH_N; pq++) {
                accounted += px_ph_acc[pq] - ph0[pq];
                cv_acc += px_ph_cswv[pq] - cv0[pq];
                ci_acc += px_ph_csiv[pq] - ci0[pq];
                ib_acc += px_ph_inb[pq] - ib0[pq];
                ob_acc += px_ph_oub[pq] - ob0[pq];
            }
            px_ph_acc[PX_PH_OTHER] += (s_step1.t - s_step0.t) - accounted;
            px_ph_cswv[PX_PH_OTHER] += (s_step1.v - s_step0.v) - cv_acc;
            px_ph_csiv[PX_PH_OTHER] += (s_step1.iv - s_step0.iv) - ci_acc;
            px_ph_inb[PX_PH_OTHER] += (s_step1.inb - s_step0.inb) - ib_acc;
            px_ph_oub[PX_PH_OTHER] += (s_step1.oub - s_step0.oub) - ob_acc;
            px_step_wall_acc += (s_step1.t - s_step0.t);
        }
        if (el >= time_limit) {
            fprintf(stderr, "train_proxy100m: time limit %.0fs reached\n",
                    time_limit);
            break;
        }
    }

    {
        double el = px_now_sec() - t0;
        double done = (double)(step < max_steps ? step + 1 : max_steps);
        if (done < 0.0) {
            done = 0.0;
        }
        // 最終valを再計測 (forwardのみ) して単調減少の目安を報告する。
        // 緩いスモーク目安: final < init*0.5。失敗でもexitは変えない
        // (過剰な厳しさ禁止。判定はログの前半/後半平均と併せて行う)。
        {
            float vf =
                px_val_loss(P, layer_off, head_off, Xv, Tv, Vacts);
            if (isfinite((double)vf)) {
                val_loss = vf;
            }
        }
        fprintf(stderr,
                "train_proxy100m: done steps=%ld elapsed=%.1fs steps_sec=%.3f "
                "rss_mib=%.1f val_init=%.6f val_last=%.6f sticky_last=%.6f\n",
                step + 1 <= max_steps ? step + 1 : max_steps, el,
                (el > 0.0 && done > 0.0) ? done / el : 0.0, px_rss_mib(),
                (double)val_init, (double)val_loss, (double)sticky_loss);
    }
    // F2 probe: --profile時のみ内訳表示 (計測のみ)。
    if (profile) {
        jt_esmoe_stats_t prof_es1;
        memset(&prof_es1, 0, sizeof(prof_es1));
        if (es_inited) {
            if (jt_esmoe_stats(&es, &prof_es1) != JT_OK) {
                memset(&prof_es1, 0, sizeof(prof_es1));
            }
        }
        px_print_profile(nthr, (step < max_steps ? step + 1 : max_steps),
                         px_now_sec() - t0, &prof_es0, &prof_es1,
                         prof_run0);
    }
    rc_all = 0;

cleanup:
    if (es_inited) {
        jt_esmoe_fini(&es);
    }
    if (opt_inited) {
        jt_optim8_fini(&opt);
    }
    free(P);
    free(G);
    free(X);
    free(T);
    free(acts);
    free(dtmp);
    free(Xv);
    free(Tv);
    free(Vacts);
    free(Gpart);
    free(dtmps);
    free(Cids);
    free(Cw);
    free(CGsel);
    free(CUsel);
    free(CYsel);
    free(CGs);
    free(CUs);
    free(Gates);
    return rc_all;
}
