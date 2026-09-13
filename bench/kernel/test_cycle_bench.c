// test_cycle_bench: DESIGN.MD §6 目標のサイクル換算見積もり (P2 GAP-5対応)。
//
// 背景: DESIGN.MD §6 の目標 (1誤差伝播=学習1ステップ 20Mサイクル以下、
// 順伝播 6M以下、逆伝播 12M以下、更新 2M) が未計測だったため、
// bench_common (median/MAD・monotonic clock) で ns/op を測り、
// 壁時間→サイクル換算で対比表を出す。macOS では perf が使えないため
// cycles = median_ns * freq_hz / 1e9 で見積もる。freq は sysctl
// (hw.cpufrequency) で取得・記録し、取得不可時は既定値+推定扱いとする
// (前提を stdout に明記)。freq_locked=false (固定なし)。
//
// 測定対象 (小規模: dk=32/dv=64、n=64/h=64):
//   - gdn2 decode順伝播 (jt_gdn2_decode_step, ライブラリ核)
//   - gdn2 backward (jt_gdn2_decode_bwd, ライブラリ核)
//   - swiglu fwd (bench内参照実装。下記 NOTE 参照)
//   - swiglu bwd (jt_swiglu_bwd, ライブラリ核)
//   - rmsnorm fwd (bench内参照実装。下記 NOTE 参照)
//   - rmsnorm bwd (jt_rmsnorm_bwd, ライブラリ核)
// NOTE: swiglu/rmsnorm の forward は現時点でライブラリ核が存在しないため、
//   train_bwd.h の forward 定義式に整合する bench 内参照実装 (スカラーfp32)
//   を測定する。bwd との対照用であり、将来ライブラリ核が追加されたら
//   そちらへ置き換えること (TODO)。
//
// A100M級想定のFLOP外挿コメント:
//   本計測の1ケースは単一カーネル・小次元で約 0.2k〜80k FLOP 程度
//   (gdn2-fwd dk32*dv64 で約16k FLOP、gdn2-bwd で約60〜80k FLOP、
//   swiglu-fwd n64/h64 で約25k FLOP+64 exp、rms は数百 FLOP)。
//   これに対し A100M (active 100M) の1トークン順伝播は約 200MFLOP
//   (100M params × 2 FLOP)、逆伝播はその約2倍が目安。すなわち本 microbench
//   の総和 (~150〜200kFLOP) とは約1000倍の開きがある。層数 (数十層)・
//   MoE発火率・メモリ帯域律速 (DESIGN.MD §1: 帯域律速でFLOP線形外挿は不成立)
//   を無視した線形時間外挿は参考値に留める。TODO: 層数掛け+帯域モデル
//   (GB/token) での再見積を別タスクで実施すること。
//
// 合否: 「測定が完走すること」のみを assert する。DESIGN目標超過で fail
// にしない。超過時は対比表に OVER と表示し、TODO追跡とする。
// 規約: C11, restrict積極使用, errnoベース, クロスプラットフォーム
// (Linux/macOS/Windowsで同一ソースがビルド可能)。
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_common.h"
#include "jimotono/common.h"
#include "jimotono/gdn2.h"
#include "jimotono/train_bwd.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

// macOS (perf不可) の壁時間→サイクル換算に加え、当環境の単調クロック分解能が
// 1us (clock_getres=1e-6) のため、1 op が分解能以下のカーネル (rms等) は
// 単発計測が 0ns に量子化される。これを避ける標準手法として各 trial 内で
// repeat 回反復し、ns/op = 合計/repeat とする (反復回数は notes/表に記録)。
// repeat はカーネル毎に総時間が ~100〜250us/trial になるよう選ぶ。
#define CB_REP_GDN2_FWD 32
#define CB_REP_GDN2_BWD 16
#define CB_REP_SW_FWD 8
#define CB_REP_SW_BWD 8
#define CB_REP_RMS_FWD 512
#define CB_REP_RMS_BWD 512
// ---- 構成 (小規模: dk/dv=32/64、n/h=64) ----
#define CB_DK 32
#define CB_DV 64
#define CB_N 64
#define CB_H 64
#define CB_TRIALS 11
#define CB_WARMUP 3
#define CB_EPS 1e-6f

// DESIGN.MD §6 目標 (cycles)。
#define CB_TGT_FWD 6000000.0
#define CB_TGT_BWD 12000000.0
#define CB_TGT_STEP 20000000.0
#define CB_BUDGET_UPDATE 2000000.0

#ifdef JIMOTONO_BUILD_COMMIT
#define CB_COMMIT_STR JIMOTONO_BUILD_COMMIT
#else
#define CB_COMMIT_STR ""
#endif

static int g_fail = 0;
static volatile double g_sink = 0.0;
static int g_kernel_rc = JT_OK;

#define CB_CHECK(cond, ...)                                               \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);           \
            fprintf(stderr, __VA_ARGS__);                                 \
            fprintf(stderr, "\n");                                        \
            g_fail = 1;                                                   \
        }                                                                 \
    } while (0)

// 決定論的パターン [-1,1]。
static float cb_pat(int a, int b, int s) {
    int v = (a * 31 + b * 17 + s * 13) % 11;  // 0..10 (a,b,s>=0 のみ使用)
    return (float)(v - 5) / 5.0f;
}

// ---- bench内参照 forward (swiglu/rmsnorm。train_bwd.h の定義式に整合) ----
static float cb_silu_f(float z) {
    return z / (1.0f + expf(-z));
}

// forward定義: G=XWg, U=XWu, s=silu(G)*U, Y=sWd。Wg/Wu/Wd は [h][n] row-major。
static void cb_swiglu_fwd(const float *restrict X, const float *restrict Wg,
                           const float *restrict Wu, const float *restrict Wd,
                           float *restrict G, float *restrict U,
                           float *restrict Y, int n, int h) {
    for (int i = 0; i < h; i++) {
        double g = 0.0;
        double u = 0.0;
        for (int j = 0; j < n; j++) {
            g += (double)X[j] * (double)Wg[(size_t)i * (size_t)n + (size_t)j];
            u += (double)X[j] * (double)Wu[(size_t)i * (size_t)n + (size_t)j];
        }
        G[i] = (float)g;
        U[i] = (float)u;
    }
    for (int j = 0; j < n; j++) {
        double acc = 0.0;
        for (int i = 0; i < h; i++) {
            double s = (double)cb_silu_f(G[i]) * (double)U[i];
            acc += s * (double)Wd[(size_t)i * (size_t)n + (size_t)j];
        }
        Y[j] = (float)acc;
    }
}

// forward定義: r=1/sqrt(mean(X^2)+eps), Y=W*X*r。
static void cb_rmsnorm_fwd(const float *restrict X, const float *restrict W,
                            float *restrict Y, int n, float eps) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        mean += (double)X[i] * (double)X[i];
    }
    mean /= (double)n;
    {
        float r = (float)(1.0 / sqrt(mean + (double)eps));
        for (int i = 0; i < n; i++) {
            Y[i] = W[i] * X[i] * r;
        }
    }
}

// ---- 計測コンテキスト ----
typedef struct cb_gdn2_fwd_ctx {
    float *restrict S;
    float *restrict o;
    const float *restrict q;
    const float *restrict k;
    const float *restrict v;
    const float *restrict b;
    const float *restrict w;
    const float *restrict alpha;
    float *restrict scratch;
    size_t scratch_n;
    int dk;
    int dv;
    int repeat;
} cb_gdn2_fwd_ctx_t;

typedef struct cb_gdn2_bwd_ctx {
    const float *restrict Sp;
    const float *restrict q;
    const float *restrict k;
    const float *restrict v;
    const float *restrict b;
    const float *restrict w;
    const float *restrict alpha;
    const float *restrict dO;
    const float *restrict dSn;
    float *restrict dQ;
    float *restrict dK;
    float *restrict dV;
    float *restrict dB;
    float *restrict dW;
    float *restrict dA;
    float *restrict dSp;
    float *restrict scratch;
    size_t scratch_n;
    int dk;
    int dv;
    int repeat;
} cb_gdn2_bwd_ctx_t;

typedef struct cb_sw_fwd_ctx {
    const float *restrict X;
    const float *restrict Wg;
    const float *restrict Wu;
    const float *restrict Wd;
    float *restrict G;
    float *restrict U;
    float *restrict Y;
    int n;
    int h;
    int repeat;
} cb_sw_fwd_ctx_t;

typedef struct cb_sw_bwd_ctx {
    const float *restrict dY;
    const float *restrict X;
    const float *restrict G;
    const float *restrict U;
    const float *restrict Wd;
    const float *restrict Wg;
    const float *restrict Wu;
    float *restrict dX;
    float *restrict dWg;
    float *restrict dWu;
    float *restrict dWd;
    int n;
    int h;
    int repeat;
} cb_sw_bwd_ctx_t;

typedef struct cb_rms_fwd_ctx {
    const float *restrict X;
    const float *restrict W;
    float *restrict Y;
    int n;
    float eps;
    int repeat;
} cb_rms_fwd_ctx_t;

typedef struct cb_rms_bwd_ctx {
    const float *restrict dY;
    const float *restrict X;
    const float *restrict W;
    float *restrict dX;
    float *restrict dW;
    int n;
    float eps;
    int repeat;
} cb_rms_bwd_ctx_t;

static void cb_gdn2_fwd_case(void *ctx) {
    cb_gdn2_fwd_ctx_t *c = (cb_gdn2_fwd_ctx_t *)ctx;
    for (int r = 0; r < c->repeat; r++) {
        int rc = jt_gdn2_decode_step(c->S, c->o, c->q, c->k, c->v, c->b,
                                     c->w, c->alpha, c->dk, c->dv,
                                     c->scratch, c->scratch_n);
        if (rc != JT_OK) {
            g_kernel_rc = rc;
            return;
        }
        g_sink += (double)c->o[0];  // 反復ごとに消費 (最適化消去防止)
    }
}

static void cb_gdn2_bwd_case(void *ctx) {
    cb_gdn2_bwd_ctx_t *c = (cb_gdn2_bwd_ctx_t *)ctx;
    for (int r = 0; r < c->repeat; r++) {
        int rc = jt_gdn2_decode_bwd(c->Sp, c->q, c->k, c->v, c->b, c->w,
                                    c->alpha, c->dO, c->dSn, c->dQ, c->dK,
                                    c->dV, c->dB, c->dW, c->dA, c->dSp,
                                    c->dk, c->dv, c->scratch, c->scratch_n);
        if (rc != JT_OK) {
            g_kernel_rc = rc;
            return;
        }
        g_sink += (double)c->dSp[0] + (double)c->dQ[0];
    }
}

static void cb_sw_fwd_case(void *ctx) {
    cb_sw_fwd_ctx_t *c = (cb_sw_fwd_ctx_t *)ctx;
    for (int r = 0; r < c->repeat; r++) {
        cb_swiglu_fwd(c->X, c->Wg, c->Wu, c->Wd, c->G, c->U, c->Y, c->n,
                      c->h);
        g_sink += (double)c->Y[0];
    }
}

static void cb_sw_bwd_case(void *ctx) {
    cb_sw_bwd_ctx_t *c = (cb_sw_bwd_ctx_t *)ctx;
    for (int r = 0; r < c->repeat; r++) {
        int rc = jt_swiglu_bwd(c->dY, c->X, c->G, c->U, c->Wd, c->Wg,
                               c->Wu, c->dX, c->dWg, c->dWu, c->dWd, c->n,
                               c->h);
        if (rc != JT_OK) {
            g_kernel_rc = rc;
            return;
        }
        g_sink += (double)c->dX[0];
    }
}

static void cb_rms_fwd_case(void *ctx) {
    cb_rms_fwd_ctx_t *c = (cb_rms_fwd_ctx_t *)ctx;
    for (int r = 0; r < c->repeat; r++) {
        cb_rmsnorm_fwd(c->X, c->W, c->Y, c->n, c->eps);
        g_sink += (double)c->Y[0];
    }
}

static void cb_rms_bwd_case(void *ctx) {
    cb_rms_bwd_ctx_t *c = (cb_rms_bwd_ctx_t *)ctx;
    for (int r = 0; r < c->repeat; r++) {
        int rc =
            jt_rmsnorm_bwd(c->dY, c->X, c->W, c->dX, c->dW, c->n, c->eps);
        if (rc != JT_OK) {
            g_kernel_rc = rc;
            return;
        }
        g_sink += (double)c->dX[0];
    }
}

// ---- 環境取得 (クロスプラットフォーム・同一ソースでビルド可能) ----
static void cb_machine_name(char *buf, size_t cap) {
    if (cap == 0) {
        return;
    }
    // machineラベルは bench_common の共有ヘルパーで解決する
    // (JIMOTONO_MACHINE 上書き、未設定時は machine.yaml 既定)。
    if (jt_bench_machine_label(buf, cap) != JT_OK) {
        snprintf(buf, cap, "%s", "macmini-i7-8700B");
    }
}

static void cb_cpu_brand(char *buf, size_t cap) {
    const char *env = NULL;
    if (cap == 0) {
        return;
    }
    buf[0] = '\0';
    env = getenv("JIMOTONO_CPU");
    if (env != NULL && env[0] != '\0') {
        snprintf(buf, cap, "%s", env);
        return;
    }
#if defined(__APPLE__)
    {
        char tmp[256];
        size_t len = sizeof tmp;
        if (sysctlbyname("machdep.cpu.brand_string", tmp, &len, NULL, 0)
            == 0) {
            snprintf(buf, cap, "%s", tmp);
            return;
        }
    }
#endif
    snprintf(buf, cap, "%s", "unknown (set JIMOTONO_CPU to record)");
}

// 戻り値: 周波数Hz。*src_out に取得元 ("sysctl"/"env"/"default(estimated)")。
// macOS以外では sysctl が無いため env→default の順で解決する。
static double cb_freq_hz(char *src_out, size_t src_cap) {
    const char *env = getenv("JIMOTONO_FREQ_HZ");
    if (src_cap > 0) {
        src_out[0] = '\0';
    }
    if (env != NULL && env[0] != '\0') {
        double f = strtod(env, NULL);
        if (isfinite(f) && f > 0.0) {
            if (src_cap > 0) {
                snprintf(src_out, src_cap, "%s", "env:JIMOTONO_FREQ_HZ");
            }
            return f;
        }
    }
#if defined(__APPLE__)
    {
        uint64_t f = 0;
        size_t len = sizeof f;
        if (sysctlbyname("hw.cpufrequency", &f, &len, NULL, 0) == 0
            && f > 0) {
            if (src_cap > 0) {
                snprintf(src_out, src_cap, "%s", "sysctl:hw.cpufrequency");
            }
            return (double)f;
        }
    }
#endif
    if (src_cap > 0) {
        snprintf(src_out, src_cap, "%s", "default(estimated):3.2GHz");
    }
    return 3200000000.0;
}

typedef struct cb_result {
    const char *name;
    size_t trials;
    int repeat;
    double median_ns;
    double mad_ns;
    double cycles;
    double mad_cycles;
} cb_result_t;

// warmup後にN回 trials→median/MADし、repeatで割って ns/op に直す。
// 失敗時は g_fail を立てる (完走assert用)。
static void cb_run_case(const char *name, jt_bench_case_fn fn, void *ctx,
                        int repeat, cb_result_t *out) {
    double *samples = NULL;
    double med = 0.0;
    double mad = 0.0;
    int rc = JT_OK;
    out->name = name;
    out->trials = CB_TRIALS;
    out->repeat = repeat;
    out->median_ns = 0.0;
    out->mad_ns = 0.0;
    out->cycles = 0.0;
    out->mad_cycles = 0.0;
    CB_CHECK(repeat > 0, "%s: repeat=%d", name, repeat);
    if (repeat <= 0) {
        return;
    }
    samples = (double *)malloc((size_t)CB_TRIALS * sizeof(double));
    CB_CHECK(samples != NULL, "%s: samples malloc failed", name);
    if (samples == NULL) {
        return;
    }
    for (int i = 0; i < CB_WARMUP; i++) {
        fn(ctx);
    }
    g_kernel_rc = JT_OK;
    rc = jt_bench_trials(fn, ctx, (size_t)CB_TRIALS, samples);
    CB_CHECK(rc == JT_OK, "%s: trials rc=%d", name, rc);
    CB_CHECK(g_kernel_rc == JT_OK, "%s: kernel rc=%d", name, g_kernel_rc);
    if (rc != JT_OK || g_kernel_rc != JT_OK) {
        free(samples);
        return;
    }
    rc = jt_bench_median_mad(samples, (size_t)CB_TRIALS, &med, &mad);
    CB_CHECK(rc == JT_OK, "%s: median_mad rc=%d", name, rc);
    CB_CHECK(isfinite(med) && med >= 0.0, "%s: median range %f", name, med);
    CB_CHECK(isfinite(mad) && mad >= 0.0, "%s: mad range %f", name, mad);
    if (rc == JT_OK && isfinite(med) && isfinite(mad)) {
        // trial合計を repeat で割り ns/op (MADも同様にスケール)。
        out->median_ns = med / (double)repeat;
        out->mad_ns = mad / (double)repeat;
    }
    free(samples);
}

static const char *cb_verdict(double cycles, double target) {
    return (cycles <= target) ? "OK" : "OVER";
}

int main(void) {
    char machine[128];
    char cpu[256];
    char freq_src[64];
    double freq = 0.0;
    const char *commit = CB_COMMIT_STR;
    const char *env_commit = NULL;
    cb_result_t r_fwd = {0};
    cb_result_t r_bwd = {0};
    cb_result_t r_swf = {0};
    cb_result_t r_swb = {0};
    cb_result_t r_rmf = {0};
    cb_result_t r_rmb = {0};
    double fwd_cyc = 0.0;
    double bwd_cyc = 0.0;
    double step_cyc = 0.0;
    // ---- gdn2 fwd用 ----
    float *S = NULL;
    float *o = NULL;
    float *q = NULL;
    float *k = NULL;
    float *v = NULL;
    float *b = NULL;
    float *w = NULL;
    float *alpha = NULL;
    float *gsc = NULL;
    size_t gneed = 0;
    // ---- gdn2 bwd用 ----
    float *bSp = NULL;
    float *bdO = NULL;
    float *bdSn = NULL;
    float *bdQ = NULL;
    float *bdK = NULL;
    float *bdV = NULL;
    float *bdB = NULL;
    float *bdW = NULL;
    float *bdA = NULL;
    float *bdSp = NULL;
    float *bsc = NULL;
    size_t bneed = 0;
    // ---- swiglu用 ----
    float *sX = NULL;
    float *sWg = NULL;
    float *sWu = NULL;
    float *sWd = NULL;
    float *sG = NULL;
    float *sU = NULL;
    float *sY = NULL;
    float *sdY = NULL;
    float *sdX = NULL;
    float *sdWg = NULL;
    float *sdWu = NULL;
    float *sdWd = NULL;
    // ---- rms用 ----
    float *rX = NULL;
    float *rW = NULL;
    float *rY = NULL;
    float *rdY = NULL;
    float *rdX = NULL;
    float *rdW = NULL;
    int rc = JT_OK;

    cb_machine_name(machine, sizeof machine);
    cb_cpu_brand(cpu, sizeof cpu);
    freq = cb_freq_hz(freq_src, sizeof freq_src);
    env_commit = getenv("JIMOTONO_COMMIT");
    if ((commit == NULL || commit[0] == '\0') && env_commit != NULL) {
        commit = env_commit;
    }
    if (commit == NULL) {
        commit = "";
    }

    printf("machine: %s\n", machine);
    printf("cpu: %s\n", cpu);
    printf("freq_hz: %.0f\n", freq);
    printf("freq_src: %s\n", freq_src);
    printf("freq_locked: false\n");
    printf("trials: %d\n", CB_TRIALS);
    printf("commit: %s\n", commit);
    printf("premise: wall-time ns/op (monotonic clock) * freq_hz / 1e9 = cycles(estimate); "
           "macOS perf unavailable; freq from %s; dims dk=%d dv=%d n=%d h=%d\n",
           freq_src, CB_DK, CB_DV, CB_N, CB_H);

    // ---- 確保 ----
    rc = jt_gdn2_state_alloc(&S, CB_DK, CB_DV);
    CB_CHECK(rc == JT_OK && S != NULL, "gdn2 fwd state alloc rc=%d", rc);
    rc = jt_gdn2_state_alloc(&bSp, CB_DK, CB_DV);
    CB_CHECK(rc == JT_OK && bSp != NULL, "gdn2 bwd state alloc rc=%d", rc);
    rc = jt_gdn2_scratch_floats(CB_DK, CB_DV, &gneed);
    CB_CHECK(rc == JT_OK, "gdn2 fwd scratch rc=%d", rc);
    rc = jt_gdn2_decode_bwd_scratch_floats(CB_DK, CB_DV, &bneed);
    CB_CHECK(rc == JT_OK, "gdn2 bwd scratch rc=%d", rc);
    o = (float *)malloc((size_t)CB_DV * sizeof(float));
    q = (float *)malloc((size_t)CB_DK * sizeof(float));
    k = (float *)malloc((size_t)CB_DK * sizeof(float));
    b = (float *)malloc((size_t)CB_DK * sizeof(float));
    w = (float *)malloc((size_t)CB_DV * sizeof(float));
    v = (float *)malloc((size_t)CB_DV * sizeof(float));
    alpha = (float *)malloc((size_t)CB_DK * sizeof(float));
    gsc = (float *)malloc(gneed * sizeof(float));
    bdO = (float *)malloc((size_t)CB_DV * sizeof(float));
    bdSn = (float *)malloc((size_t)CB_DK * (size_t)CB_DV * sizeof(float));
    bdQ = (float *)malloc((size_t)CB_DK * sizeof(float));
    bdK = (float *)malloc((size_t)CB_DK * sizeof(float));
    bdV = (float *)malloc((size_t)CB_DV * sizeof(float));
    bdB = (float *)malloc((size_t)CB_DK * sizeof(float));
    bdW = (float *)malloc((size_t)CB_DV * sizeof(float));
    bdA = (float *)malloc((size_t)CB_DK * sizeof(float));
    bdSp = (float *)malloc((size_t)CB_DK * (size_t)CB_DV * sizeof(float));
    bsc = (float *)malloc(bneed * sizeof(float));
    sX = (float *)malloc((size_t)CB_N * sizeof(float));
    sWg = (float *)malloc((size_t)CB_H * (size_t)CB_N * sizeof(float));
    sWu = (float *)malloc((size_t)CB_H * (size_t)CB_N * sizeof(float));
    sWd = (float *)malloc((size_t)CB_H * (size_t)CB_N * sizeof(float));
    sG = (float *)malloc((size_t)CB_H * sizeof(float));
    sU = (float *)malloc((size_t)CB_H * sizeof(float));
    sY = (float *)malloc((size_t)CB_N * sizeof(float));
    sdY = (float *)malloc((size_t)CB_N * sizeof(float));
    sdX = (float *)malloc((size_t)CB_N * sizeof(float));
    sdWg = (float *)malloc((size_t)CB_H * (size_t)CB_N * sizeof(float));
    sdWu = (float *)malloc((size_t)CB_H * (size_t)CB_N * sizeof(float));
    sdWd = (float *)malloc((size_t)CB_H * (size_t)CB_N * sizeof(float));
    rX = (float *)malloc((size_t)CB_N * sizeof(float));
    rW = (float *)malloc((size_t)CB_N * sizeof(float));
    rY = (float *)malloc((size_t)CB_N * sizeof(float));
    rdY = (float *)malloc((size_t)CB_N * sizeof(float));
    rdX = (float *)malloc((size_t)CB_N * sizeof(float));
    rdW = (float *)malloc((size_t)CB_N * sizeof(float));
    CB_CHECK(o && q && k && b && w && v && alpha && gsc && bdO && bdSn
                 && bdQ && bdK && bdV && bdB && bdW && bdA && bdSp && bsc
                 && sX && sWg && sWu && sWd && sG && sU && sY && sdY && sdX
                 && sdWg && sdWu && sdWd && rX && rW && rY && rdY && rdX
                 && rdW,
             "malloc failed");
    if (g_fail != 0 || S == NULL || bSp == NULL || o == NULL || q == NULL
        || k == NULL || b == NULL || w == NULL || v == NULL
        || alpha == NULL || gsc == NULL || bdO == NULL || bdSn == NULL
        || bdQ == NULL || bdK == NULL || bdV == NULL || bdB == NULL
        || bdW == NULL || bdA == NULL || bdSp == NULL || bsc == NULL
        || sX == NULL || sWg == NULL || sWu == NULL || sWd == NULL
        || sG == NULL || sU == NULL || sY == NULL || sdY == NULL
        || sdX == NULL || sdWg == NULL || sdWu == NULL || sdWd == NULL
        || rX == NULL || rW == NULL || rY == NULL || rdY == NULL
        || rdX == NULL || rdW == NULL) {
        goto cleanup;
    }

    // ---- 初期化 (決定論的) ----
    for (int i = 0; i < CB_DK; i++) {
        q[i] = cb_pat(i, 0, 11) + 0.37f;
        k[i] = cb_pat(i, 1, 12) - 0.23f;
        b[i] = 0.5f + 0.1f * cb_pat(i, 2, 13);
        alpha[i] = 0.9f + 0.05f * cb_pat(i, 3, 14);
    }
    for (int j = 0; j < CB_DV; j++) {
        v[j] = cb_pat(0, j, 15) + 0.11f;
        w[j] = 0.6f + 0.1f * cb_pat(1, j, 16);
        o[j] = 0.0f;
        bdO[j] = 0.3f * cb_pat(2, j, 17) + 0.1f;
    }
    for (int i = 0; i < CB_DK; i++) {
        for (int j = 0; j < CB_DV; j++) {
            S[i * CB_DV + j] = 0.25f * cb_pat(i, j, 18);
            bSp[i * CB_DV + j] = 0.25f * cb_pat(i, j, 18);
            bdSn[i * CB_DV + j] = 0.2f * cb_pat(i, j, 19) + 0.05f;
        }
    }
    for (int j = 0; j < CB_N; j++) {
        sX[j] = 0.4f * cb_pat(0, j, 21) + 0.2f;
        sdY[j] = 0.5f * cb_pat(1, j, 22) + 0.1f;
        rX[j] = 0.5f * cb_pat(j, 0, 31) + 0.3f;
        rW[j] = 0.8f + 0.2f * cb_pat(j, 1, 32);
        rdY[j] = 0.4f * cb_pat(j, 2, 33) - 0.1f;
    }
    for (int i = 0; i < CB_H; i++) {
        for (int j = 0; j < CB_N; j++) {
            sWg[(size_t)i * (size_t)CB_N + (size_t)j] =
                0.3f * cb_pat(i, j, 23);
            sWu[(size_t)i * (size_t)CB_N + (size_t)j] =
                0.3f * cb_pat(i, j, 24);
            sWd[(size_t)i * (size_t)CB_N + (size_t)j] =
                0.3f * cb_pat(i, j, 25);
        }
    }
    // swiglu-bwd 用 G/U は forward 定義どおり事前計算 (cached中間値)。
    {
        cb_sw_fwd_ctx_t pre = {sX, sWg, sWu, sWd, sG,
                               sU, sY, CB_N, CB_H, 1};
        cb_swiglu_fwd(pre.X, pre.Wg, pre.Wu, pre.Wd, pre.G, pre.U, pre.Y,
                      pre.n, pre.h);
    }

    // ---- 計測 (完走のみ assert。目標超過は OVER 表示で fail にしない) ----
    {
        cb_gdn2_fwd_ctx_t c = {S, o, q, k, v, b, w, alpha, gsc, gneed,
                               CB_DK, CB_DV, CB_REP_GDN2_FWD};
        cb_run_case("gdn2_decode/dk32-dv64", cb_gdn2_fwd_case, &c,
                    CB_REP_GDN2_FWD, &r_fwd);
    }
    {
        cb_gdn2_bwd_ctx_t c = {bSp, q, k, v, b, w,          alpha, bdO,
                               bdSn,  bdQ, bdK, bdV, bdB,   bdW,   bdA,
                               bdSp,  bsc, bneed, CB_DK,    CB_DV,
                               CB_REP_GDN2_BWD};
        cb_run_case("gdn2_bwd/dk32-dv64", cb_gdn2_bwd_case, &c,
                    CB_REP_GDN2_BWD, &r_bwd);
    }
    {
        cb_sw_fwd_ctx_t c = {sX, sWg, sWu, sWd, sG,
                             sU, sY, CB_N, CB_H, CB_REP_SW_FWD};
        cb_run_case("swiglu_fwd/n64-h64", cb_sw_fwd_case, &c,
                    CB_REP_SW_FWD, &r_swf);
    }
    {
        cb_sw_bwd_ctx_t c = {sdY, sX, sG, sU, sWd, sWg, sWu,
                             sdX, sdWg, sdWu, sdWd, CB_N, CB_H,
                             CB_REP_SW_BWD};
        cb_run_case("swiglu_bwd/n64-h64", cb_sw_bwd_case, &c,
                    CB_REP_SW_BWD, &r_swb);
    }
    {
        cb_rms_fwd_ctx_t c = {rX, rW, rY, CB_N, CB_EPS, CB_REP_RMS_FWD};
        cb_run_case("rmsnorm_fwd/n64", cb_rms_fwd_case, &c,
                    CB_REP_RMS_FWD, &r_rmf);
    }
    {
        cb_rms_bwd_ctx_t c = {rdY, rX, rW, rdX, rdW, CB_N, CB_EPS,
                              CB_REP_RMS_BWD};
        cb_run_case("rmsnorm_bwd/n64", cb_rms_bwd_case, &c,
                    CB_REP_RMS_BWD, &r_rmb);
    }
    if (g_fail != 0) {
        goto cleanup;
    }

    // ---- サイクル換算 ----
    r_fwd.cycles = r_fwd.median_ns * freq / 1e9;
    r_fwd.mad_cycles = r_fwd.mad_ns * freq / 1e9;
    r_bwd.cycles = r_bwd.median_ns * freq / 1e9;
    r_bwd.mad_cycles = r_bwd.mad_ns * freq / 1e9;
    r_swf.cycles = r_swf.median_ns * freq / 1e9;
    r_swf.mad_cycles = r_swf.mad_ns * freq / 1e9;
    r_swb.cycles = r_swb.median_ns * freq / 1e9;
    r_swb.mad_cycles = r_swb.mad_ns * freq / 1e9;
    r_rmf.cycles = r_rmf.median_ns * freq / 1e9;
    r_rmf.mad_cycles = r_rmf.mad_ns * freq / 1e9;
    r_rmb.cycles = r_rmb.median_ns * freq / 1e9;
    r_rmb.mad_cycles = r_rmb.mad_ns * freq / 1e9;
    fwd_cyc = r_fwd.cycles + r_swf.cycles + r_rmf.cycles;
    bwd_cyc = r_bwd.cycles + r_swb.cycles + r_rmb.cycles;
    step_cyc = fwd_cyc + bwd_cyc + CB_BUDGET_UPDATE;

    // ---- スキーマ準拠 JSON Lines (docs/bench_schema.md) ----
    // median_ns/mad_ns は ns/op (trial合計/repeat)。反復回数は notes に記録。
    {
        char notes[384];
        const cb_result_t *rs[6] = {&r_fwd, &r_bwd, &r_swf,
                                    &r_swb, &r_rmf, &r_rmb};
        snprintf(notes, sizeof notes,
                 "cpu=%s freq_hz=%.0f(%s) freq_locked=false small-dims "
                 "dk=%d dv=%d n=%d h=%d wall-to-cycles estimate",
                 cpu, freq, freq_src, CB_DK, CB_DV, CB_N, CB_H);
        for (int i = 0; i < 6; i++) {
            char inotes[480];
            snprintf(inotes, sizeof inotes, "%s repeat=%d", notes,
                     rs[i]->repeat);
            int prc = jt_bench_print_json(stdout, rs[i]->name, machine,
                                          commit, rs[i]->trials,
                                          rs[i]->median_ns, rs[i]->mad_ns,
                                          inotes);
            CB_CHECK(prc == JT_OK, "json print %s rc=%d", rs[i]->name,
                     prc);
        }
    }

    // ---- DESIGN目標との対比表 ----
    printf("\n# cycle estimate vs DESIGN.MD §6 (estimate; OVER does not fail)\n");
    printf("%-22s %6s %14s %14s %14s %14s %8s\n", "case", "repeat",
           "median_ns", "mad_ns", "cycles", "mad_cyc", "verdict");
    printf("%-22s %6d %14.1f %14.1f %14.1f %14.1f %8s\n", r_fwd.name,
           r_fwd.repeat, r_fwd.median_ns, r_fwd.mad_ns, r_fwd.cycles,
           r_fwd.mad_cycles, "-");
    printf("%-22s %6d %14.1f %14.1f %14.1f %14.1f %8s\n", r_swf.name,
           r_swf.repeat, r_swf.median_ns, r_swf.mad_ns, r_swf.cycles,
           r_swf.mad_cycles, "-");
    printf("%-22s %6d %14.1f %14.1f %14.1f %14.1f %8s\n", r_rmf.name,
           r_rmf.repeat, r_rmf.median_ns, r_rmf.mad_ns, r_rmf.cycles,
           r_rmf.mad_cycles, "-");
    printf("%-22s %6s %14.1f %14s %14.1f %14s %8s\n", "fwd_sum(3 kernels)",
           "-", fwd_cyc / freq * 1e9, "-", fwd_cyc, "-",
           cb_verdict(fwd_cyc, CB_TGT_FWD));
    printf("%-22s %6s %14s %14s %14.0f %14s %8s\n", "fwd_target(6M)", "-",
           "-", "-", CB_TGT_FWD, "-", "-");
    printf("%-22s %6d %14.1f %14.1f %14.1f %14.1f %8s\n", r_bwd.name,
           r_bwd.repeat, r_bwd.median_ns, r_bwd.mad_ns, r_bwd.cycles,
           r_bwd.mad_cycles, "-");
    printf("%-22s %6d %14.1f %14.1f %14.1f %14.1f %8s\n", r_swb.name,
           r_swb.repeat, r_swb.median_ns, r_swb.mad_ns, r_swb.cycles,
           r_swb.mad_cycles, "-");
    printf("%-22s %6d %14.1f %14.1f %14.1f %14.1f %8s\n", r_rmb.name,
           r_rmb.repeat, r_rmb.median_ns, r_rmb.mad_ns, r_rmb.cycles,
           r_rmb.mad_cycles, "-");
    printf("%-22s %6s %14.1f %14s %14.1f %14s %8s\n", "bwd_sum(3 kernels)",
           "-", bwd_cyc / freq * 1e9, "-", bwd_cyc, "-",
           cb_verdict(bwd_cyc, CB_TGT_BWD));
    printf("%-22s %6s %14s %14s %14.0f %14s %8s\n", "bwd_target(12M)", "-",
           "-", "-", CB_TGT_BWD, "-", "-");
    printf("%-22s %6s %14s %14s %14.1f %14s %8s\n",
           "step_est(fwd+bwd+2M)", "-", "-", "-", step_cyc, "-",
           cb_verdict(step_cyc, CB_TGT_STEP));
    printf("%-22s %14s %14s %14.0f %14s %8s\n", "step_target(20M)", "-", "-",
           CB_TGT_STEP, "-", "-");
    printf("# note: update kernel not measured here (optim8 is separate "
           "workstream); step_est adds fixed update budget 2M cycles.\n");
    printf("# note: clock resolution here is 1us; MAD=0 means trial spread "
           "< 1us (quantization floor), not zero variance.\n");
    printf("# note: A100M extrapolation: measured kernels total ~150-200kFLOP "
           "vs ~200MFLOP/token fwd at active100M; linear time extrapolation "
           "is reference-only (layers/MoE firing/bandwidth ignored). "
           "TODO: re-estimate with layer count x bandwidth model.\n");
    printf("sink: %.6f\n", g_sink);

cleanup:
    jt_gdn2_state_free(S);
    jt_gdn2_state_free(bSp);
    free(o);
    free(q);
    free(k);
    free(b);
    free(w);
    free(v);
    free(alpha);
    free(gsc);
    free(bdO);
    free(bdSn);
    free(bdQ);
    free(bdK);
    free(bdV);
    free(bdB);
    free(bdW);
    free(bdA);
    free(bdSp);
    free(bsc);
    free(sX);
    free(sWg);
    free(sWu);
    free(sWd);
    free(sG);
    free(sU);
    free(sY);
    free(sdY);
    free(sdX);
    free(sdWg);
    free(sdWu);
    free(sdWd);
    free(rX);
    free(rW);
    free(rY);
    free(rdY);
    free(rdX);
    free(rdW);
    if (g_fail != 0) {
        fprintf(stderr, "cycle_bench: FAIL (measurement did not complete)\n");
        return 1;
    }
    printf("cycle_bench: OK (measurement completed; OVER is tracked as TODO, "
           "not failure)\n");
    return 0;
}
