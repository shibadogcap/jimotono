// bw_stream: STREAM Triad相当の実効帯域マイクロベンチ (Stage 0ブロッカー #4)。
//
// 背景: F2で「fwdは実用peak 21GB/sの7〜8割、bwdは約1/5」と記録されたが、
// STREAM未導入のため公称比 η=実効/公称 が未確定だった。本ベンチで
// 単一スレッドTriadの実効帯域を測り、ηを記録する。
// bench/commonは測定基盤専用のため本ファイルはbench/kernelに置く
// (ハーネス jt_bench_* は再利用する)。
//
// 仕様 (小規模・短時間):
//   - a[i] = b[i] + s*c[i] (double、N=4M。3配列=96MBでL3 12MBを超過しDRAMを測る)
//   - copy (a[i]=b[i]) を参照用に併測 (read/write上限の目安)
//   - trials=7・中央値。1 pass ≈ 96MB (triadは24B/elem、copyは16B/elem)
//   - 正しさのみassert (spot-check)。速度での合否判定はしない
//     (ctestは完走のみ見る。既存bench_tmac/bench_gdn2と同一流儀)
// 規約: C11, restrict積極使用, errnoベース, クロスプラットフォーム
// (Linux/macOS/Windowsで同一ソースがビルド可能)。AVX-512不使用
// (コンパイラの自動ベクトル化に委ね、明示intrinsicsは使わない)。
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_common.h"
#include "jimotono/common.h"

// 4M doubles ×3 = 96MB (L3 12MB超過、デュアルch DRAM常駐を想定)。
#define BW_N ((size_t)4 * (size_t)1024 * (size_t)1024)
#define BW_TRIALS 7
#define BW_WARMUP 2
#define BW_SCALAR 3.0

static int g_fail = 0;
static volatile double g_sink = 0.0;

#define BW_CHECK(cond, ...)                                                \
    do {                                                                   \
        if (!(cond)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);            \
            fprintf(stderr, __VA_ARGS__);                                  \
            fprintf(stderr, "\n");                                         \
            g_fail = 1;                                                    \
        }                                                                  \
    } while (0)

typedef struct bw_ctx {
    double *restrict a;
    double *restrict b;
    double *restrict c;
    double s;
    size_t n;
} bw_ctx_t;

static void bw_triad_case(void *ctx) {
    bw_ctx_t *x = (bw_ctx_t *)ctx;
    double *restrict a = x->a;
    const double *restrict b = x->b;
    const double *restrict c = x->c;
    double s = x->s;
    size_t n = x->n;
    for (size_t i = 0; i < n; i++) {
        a[i] = b[i] + s * c[i];
    }
    g_sink += a[n / 2] + a[0];
}

static void bw_copy_case(void *ctx) {
    bw_ctx_t *x = (bw_ctx_t *)ctx;
    double *restrict a = x->a;
    const double *restrict b = x->b;
    size_t n = x->n;
    for (size_t i = 0; i < n; i++) {
        a[i] = b[i];
    }
    g_sink += a[n / 2] + a[0];
}

int main(void) {
    char machine[128];
    double *a = NULL;
    double *b = NULL;
    double *c = NULL;
    double samples[BW_TRIALS];
    double med = 0.0;
    double mad = 0.0;
    size_t n = BW_N;
    int rc = JT_OK;

    if (jt_bench_machine_label(machine, sizeof machine) != JT_OK) {
        snprintf(machine, sizeof machine, "%s", "macmini-i7-8700B");
    }
    a = (double *)malloc(n * sizeof(double));
    b = (double *)malloc(n * sizeof(double));
    c = (double *)malloc(n * sizeof(double));
    BW_CHECK(a != NULL && b != NULL && c != NULL, "malloc failed");
    if (g_fail != 0) {
        goto cleanup;
    }
    for (size_t i = 0; i < n; i++) {
        b[i] = (double)(i % 1024) * 0.001;
        c[i] = (double)((i * 7) % 1024) * 0.002;
        a[i] = 0.0;
    }

    // ---- triad ----
    {
        bw_ctx_t x = {a, b, c, BW_SCALAR, n};
        for (int i = 0; i < BW_WARMUP; i++) {
            bw_triad_case(&x);
        }
        // 正しさspot-check (全件走査は帯域測定を汚すため代表点のみ)。
        BW_CHECK(a[0] == b[0] + BW_SCALAR * c[0], "triad spot-check [0]");
        BW_CHECK(a[n - 1] == b[n - 1] + BW_SCALAR * c[n - 1],
                 "triad spot-check [n-1]");
        BW_CHECK(a[n / 2] == b[n / 2] + BW_SCALAR * c[n / 2],
                 "triad spot-check [n/2]");
        if (g_fail != 0) {
            goto cleanup;
        }
        // 再初期化 (check後のaを汚さないよう測定前に戻す必要はない。
        // triadはaを上書きするためそのまま測定する)。
        rc = jt_bench_trials(bw_triad_case, &x, (size_t)BW_TRIALS,
                             samples);
        BW_CHECK(rc == JT_OK, "triad trials rc=%d", rc);
        if (rc != JT_OK) {
            goto cleanup;
        }
        rc = jt_bench_median_mad(samples, (size_t)BW_TRIALS, &med, &mad);
        BW_CHECK(rc == JT_OK, "triad median rc=%d", rc);
        if (rc != JT_OK) {
            goto cleanup;
        }
        // triad: read 16B + write 8B = 24B/elem。
        {
            double bytes = (double)n * 24.0;
            double gbps = bytes / med;
            printf("bw_stream: triad n=%zu trials=%d median_ns=%.1f "
                   "mad_ns=%.1f bytes=%.0f GB/s=%.2f\n",
                   n, BW_TRIALS, med, mad, bytes, gbps);
            printf("{\"name\":\"bw_stream_triad\",\"machine\":\"%s\","
                   "\"trials\":%d,\"median_ns\":%.1f,\"mad_ns\":%.1f,"
                   "\"bytes\":%.0f,\"gb_per_s\":%.3f,"
                   "\"notes\":\"double a=b+s*c N=4M single-thread\"}\n",
                   machine, BW_TRIALS, med, mad, bytes, gbps);
        }
    }

    // ---- copy (参照用) ----
    {
        bw_ctx_t x = {a, b, c, BW_SCALAR, n};
        for (int i = 0; i < BW_WARMUP; i++) {
            bw_copy_case(&x);
        }
        BW_CHECK(a[0] == b[0] && a[n - 1] == b[n - 1], "copy spot-check");
        if (g_fail != 0) {
            goto cleanup;
        }
        rc = jt_bench_trials(bw_copy_case, &x, (size_t)BW_TRIALS, samples);
        BW_CHECK(rc == JT_OK, "copy trials rc=%d", rc);
        if (rc != JT_OK) {
            goto cleanup;
        }
        rc = jt_bench_median_mad(samples, (size_t)BW_TRIALS, &med, &mad);
        BW_CHECK(rc == JT_OK, "copy median rc=%d", rc);
        if (rc != JT_OK) {
            goto cleanup;
        }
        // copy: read 8B + write 8B = 16B/elem。
        {
            double bytes = (double)n * 16.0;
            double gbps = bytes / med;
            printf("bw_stream: copy  n=%zu trials=%d median_ns=%.1f "
                   "mad_ns=%.1f bytes=%.0f GB/s=%.2f\n",
                   n, BW_TRIALS, med, mad, bytes, gbps);
            printf("{\"name\":\"bw_stream_copy\",\"machine\":\"%s\","
                   "\"trials\":%d,\"median_ns\":%.1f,\"mad_ns\":%.1f,"
                   "\"bytes\":%.0f,\"gb_per_s\":%.3f,"
                   "\"notes\":\"double a=b N=4M single-thread\"}\n",
                   machine, BW_TRIALS, med, mad, bytes, gbps);
        }
    }

    printf("bw_stream: OK (correctness only; bandwidth is reference)\n");

cleanup:
    free(a);
    free(b);
    free(c);
    if (g_fail != 0) {
        fprintf(stderr, "bw_stream: FAIL\n");
        return 1;
    }
    return 0;
}
