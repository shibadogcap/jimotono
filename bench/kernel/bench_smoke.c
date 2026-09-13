// bench_smoke: bench_commonハーネス自体のスモークテスト (ROADMAP 5.1/5.2)。
// 既存bench (gdn2/tmac/routing/io) には手を出さない。ハーネスのみ検証する。
// - 単調クロックの非減少性
// - 既知ベクトルの中央値/MAD (奇数/偶数/単一)
// - ダミー計測のN回 trials -> 中央値/MAD が期待範囲 (有限, >=0, 上限内)
// - TSV/JSON行出力の形式 (tmpfileで往復確認)
// - 不正入力の JT_ERR_INVAL 応答
// 成功時 exit 0 + "bench_smoke: OK"、失敗時 exit 1 + stderr。
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_common.h"

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

static int d_eq(double a, double b, double tol) {
    return fabs(a - b) <= tol;
}

// ---- ダミー計測対象: 最適化で消えないようvolatile sinkに蓄積 ----
static volatile double g_sink = 0.0;

static void dummy_case(void *ctx) {
    (void)ctx;
    double s = 0.0;
    for (int i = 0; i < 1000; i++) {
        s += (double)i * 0.5;
    }
    g_sink += s;
}

static void counting_case(void *ctx) {
    int *p = (int *)ctx;
    (*p)++;
}

static void test_clock_monotonic(void) {
    uint64_t t0 = 0;
    uint64_t t1 = 0;
    CHECK(jt_bench_now_ns(&t0) == JT_OK, "now_ns rc");
    CHECK(jt_bench_now_ns(&t1) == JT_OK, "now_ns rc2");
    CHECK(t1 >= t0, "monotonic t1=%llu t0=%llu",
          (unsigned long long)t1, (unsigned long long)t0);
    CHECK(jt_bench_now_ns(NULL) == JT_ERR_INVAL, "now_ns NULL");
}

static void test_median_odd(void) {
    // {10,30,20,50,40} -> median 30, MAD 10
    double s[5] = {10.0, 30.0, 20.0, 50.0, 40.0};
    double med = -1.0;
    double mad = -1.0;
    CHECK(jt_bench_median_mad(s, 5, &med, &mad) == JT_OK, "odd rc");
    CHECK(d_eq(med, 30.0, 1e-9), "odd median=%f want 30", med);
    CHECK(d_eq(mad, 10.0, 1e-9), "odd mad=%f want 10", mad);
}

static void test_median_even(void) {
    // {1,2,3,4} -> median 2.5, MAD 1.0
    double s[4] = {1.0, 2.0, 3.0, 4.0};
    double med = -1.0;
    double mad = -1.0;
    CHECK(jt_bench_median_mad(s, 4, &med, &mad) == JT_OK, "even rc");
    CHECK(d_eq(med, 2.5, 1e-9), "even median=%f want 2.5", med);
    CHECK(d_eq(mad, 1.0, 1e-9), "even mad=%f want 1.0", mad);
}

static void test_median_single(void) {
    double s[1] = {42.0};
    double med = -1.0;
    double mad = -1.0;
    CHECK(jt_bench_median_mad(s, 1, &med, &mad) == JT_OK, "single rc");
    CHECK(d_eq(med, 42.0, 1e-9), "single median=%f want 42", med);
    CHECK(d_eq(mad, 0.0, 1e-9), "single mad=%f want 0", mad);
}

static void test_median_invalid(void) {
    double s[2] = {1.0, 2.0};
    double med = 0.0;
    double mad = 0.0;
    double nan_s[2] = {1.0, NAN};
    double inf_s[2] = {1.0, INFINITY};
    CHECK(jt_bench_median_mad(NULL, 2, &med, &mad) == JT_ERR_INVAL, "NULL samples");
    CHECK(jt_bench_median_mad(s, 0, &med, &mad) == JT_ERR_INVAL, "n==0");
    CHECK(jt_bench_median_mad(s, 2, NULL, &mad) == JT_ERR_INVAL, "NULL median_out");
    CHECK(jt_bench_median_mad(s, 2, &med, NULL) == JT_ERR_INVAL, "NULL mad_out");
    CHECK(jt_bench_median_mad(nan_s, 2, &med, &mad) == JT_ERR_INVAL, "NaN reject");
    CHECK(jt_bench_median_mad(inf_s, 2, &med, &mad) == JT_ERR_INVAL, "Inf reject");
}

static void test_measure_trials(void) {
    double ns = -1.0;
    int n = 0;
    CHECK(jt_bench_measure(NULL, NULL, &ns) == JT_ERR_INVAL, "measure NULL fn");
    CHECK(jt_bench_measure(counting_case, &n, NULL) == JT_ERR_INVAL, "measure NULL out");
    CHECK(jt_bench_measure(counting_case, &n, &ns) == JT_OK, "measure rc");
    CHECK(n == 1, "measure calls fn once n=%d", n);
    CHECK(isfinite(ns) && ns >= 0.0, "measure ns finite >=0 ns=%f", ns);

    {
        double buf[11] = {0.0};
        double med = -1.0;
        double mad = -1.0;
        int c2 = 0;
        CHECK(jt_bench_trials(counting_case, &c2, 11, buf) == JT_OK, "trials rc");
        CHECK(c2 == 11, "trials calls 11 times c=%d", c2);
        CHECK(jt_bench_median_mad(buf, 11, &med, &mad) == JT_OK, "trials median rc");
        CHECK(isfinite(med) && med >= 0.0, "trials median range %f", med);
        CHECK(isfinite(mad) && mad >= 0.0, "trials mad range %f", mad);
        CHECK(jt_bench_trials(NULL, &c2, 11, buf) == JT_ERR_INVAL, "trials NULL fn");
        CHECK(jt_bench_trials(counting_case, &c2, 0, buf) == JT_ERR_INVAL, "trials 0");
        CHECK(jt_bench_trials(counting_case, &c2, 11, NULL) == JT_ERR_INVAL, "trials NULL buf");
    }
}

static void test_dummy_range(void) {
    // ハーネス自体のスモーク: ダミー計測11回の中央値/MADが期待範囲。
    // 期待: 有限, median >= 0, MAD >= 0, median < 1e9 ns (1秒)。
    enum { N = 11 };
    double buf[N] = {0.0};
    double med = -1.0;
    double mad = -1.0;
    CHECK(jt_bench_trials(dummy_case, NULL, N, buf) == JT_OK, "dummy trials rc");
    CHECK(jt_bench_median_mad(buf, N, &med, &mad) == JT_OK, "dummy median rc");
    CHECK(isfinite(med) && med >= 0.0 && med < 1e9, "dummy median range %f", med);
    CHECK(isfinite(mad) && mad >= 0.0 && mad < 1e9, "dummy mad range %f", mad);
    printf("smoke dummy: trials=%d median_ns=%.3f mad_ns=%.3f\n", N, med, mad);
    // TSV/JSONの見本出力 (stdoutで目視確認用。合否には使わない)。
    CHECK(jt_bench_print_tsv(stdout, "smoke/dummy", N, med, mad) == JT_OK, "tsv stdout");
    CHECK(jt_bench_print_json(stdout, "smoke/dummy", "smoke-machine", "smoke-commit", N,
                              med, mad, "bench_smoke self-check") == JT_OK,
          "json stdout");
}

static void test_print_roundtrip(void) {
    FILE *fp = tmpfile();
    char buf[1024];
    size_t len = 0;
    CHECK(fp != NULL, "tmpfile");
    if (fp == NULL) {
        return;
    }
    CHECK(jt_bench_print_tsv(fp, "smoke/dummy", 11, 1234.5, 67.25) == JT_OK, "tsv rc");
    CHECK(jt_bench_print_json(fp, "smoke/dummy", "m1", "abc123", 11, 1234.5, 67.25,
                              "note \"q\"") == JT_OK,
          "json rc");
    CHECK(jt_bench_print_tsv(NULL, "x", 1, 1.0, 0.0) == JT_ERR_INVAL, "tsv NULL fp");
    CHECK(jt_bench_print_tsv(fp, NULL, 1, 1.0, 0.0) == JT_ERR_INVAL, "tsv NULL name");
    CHECK(jt_bench_print_tsv(fp, "x", 0, 1.0, 0.0) == JT_ERR_INVAL, "tsv trials 0");
    CHECK(jt_bench_print_json(NULL, "x", "m", "c", 1, 1.0, 0.0, "") == JT_ERR_INVAL,
          "json NULL fp");
    CHECK(jt_bench_print_json(fp, NULL, "m", "c", 1, 1.0, 0.0, "") == JT_ERR_INVAL,
          "json NULL name");
    rewind(fp);
    len = fread(buf, 1, sizeof(buf) - 1u, fp);
    buf[len] = '\0';
    CHECK(strstr(buf, "smoke/dummy\t11\t1234.500000\t67.250000") != NULL, "tsv content");
    CHECK(strstr(buf, "\"name\":\"smoke/dummy\"") != NULL, "json name");
    CHECK(strstr(buf, "\"machine\":\"m1\"") != NULL, "json machine");
    CHECK(strstr(buf, "\"commit\":\"abc123\"") != NULL, "json commit");
    CHECK(strstr(buf, "\"trials\":11") != NULL, "json trials");
    CHECK(strstr(buf, "\"median_ns\":1234.500000") != NULL, "json median");
    CHECK(strstr(buf, "\"mad_ns\":67.250000") != NULL, "json mad");
    // JSONエスケープ: note "q" -> note \"q\"
    CHECK(strstr(buf, "note \\\"q\\\"") != NULL, "json escape");
    fclose(fp);
}

static void test_machine_label(void) {
    // JIMOTONO_MACHINE による machine ラベル上書き (既存 CHECK 流儀)。
    // cycle bench (test_cycle_bench.c) と共有の jt_bench_machine_label を検証し、
    // N150 相当の値が JSON 出力に載ることも往復確認する。
    char saved[256];
    const char *old = getenv("JIMOTONO_MACHINE");
    int had_old = (old != NULL);
    char buf[128];
    if (had_old) {
        snprintf(saved, sizeof saved, "%s", old);
    }
#if defined(_WIN32)
    CHECK(_putenv("JIMOTONO_MACHINE=n150") == 0, "setenv n150");
    CHECK(jt_bench_machine_label(buf, sizeof buf) == JT_OK, "machine n150 rc");
    CHECK(strcmp(buf, "n150") == 0, "machine n150 got=%s", buf);
    CHECK(_putenv("JIMOTONO_MACHINE=") == 0, "setenv empty");
    CHECK(jt_bench_machine_label(buf, sizeof buf) == JT_OK, "machine empty rc");
    CHECK(strcmp(buf, "macmini-i7-8700B") == 0, "machine empty fallback got=%s", buf);
#else
    CHECK(setenv("JIMOTONO_MACHINE", "n150", 1) == 0, "setenv n150");
    CHECK(jt_bench_machine_label(buf, sizeof buf) == JT_OK, "machine n150 rc");
    CHECK(strcmp(buf, "n150") == 0, "machine n150 got=%s", buf);
    CHECK(setenv("JIMOTONO_MACHINE", "", 1) == 0, "setenv empty");
    CHECK(jt_bench_machine_label(buf, sizeof buf) == JT_OK, "machine empty rc");
    CHECK(strcmp(buf, "macmini-i7-8700B") == 0, "machine empty fallback got=%s", buf);
    CHECK(unsetenv("JIMOTONO_MACHINE") == 0, "unsetenv");
    CHECK(jt_bench_machine_label(buf, sizeof buf) == JT_OK, "machine unset rc");
    CHECK(strcmp(buf, "macmini-i7-8700B") == 0, "machine unset fallback got=%s", buf);
#endif
    CHECK(jt_bench_machine_label(NULL, sizeof buf) == JT_ERR_INVAL, "machine NULL buf");
    CHECK(jt_bench_machine_label(buf, 0) == JT_ERR_INVAL, "machine cap 0");
    // N150 ラベル付き JSON 行の往復確認 (cycle bench の JSON 出力と同形式)。
    {
        FILE *fp = tmpfile();
        char out[1024];
        size_t len = 0;
        CHECK(fp != NULL, "tmpfile machine");
        if (fp != NULL) {
#if defined(_WIN32)
            CHECK(_putenv("JIMOTONO_MACHINE=n150") == 0, "setenv n150 json");
#else
            CHECK(setenv("JIMOTONO_MACHINE", "n150", 1) == 0, "setenv n150 json");
#endif
            CHECK(jt_bench_machine_label(buf, sizeof buf) == JT_OK, "machine relabel rc");
            CHECK(jt_bench_print_json(fp, "cycle/fwd_sum", buf, "abc123", 11, 100.0, 1.0,
                                      "n150 check") == JT_OK,
                  "json n150 rc");
            rewind(fp);
            len = fread(out, 1, sizeof(out) - 1u, fp);
            out[len] = '\0';
            CHECK(strstr(out, "\"machine\":\"n150\"") != NULL, "json machine n150");
            fclose(fp);
        }
    }
    // 環境復元 (他テストへの漏れ防止)。
    if (had_old) {
#if defined(_WIN32)
        char restore[320];
        snprintf(restore, sizeof restore, "JIMOTONO_MACHINE=%s", saved);
        _putenv(restore);
#else
        setenv("JIMOTONO_MACHINE", saved, 1);
#endif
    } else {
#if defined(_WIN32)
        _putenv("JIMOTONO_MACHINE=");
#else
        unsetenv("JIMOTONO_MACHINE");
#endif
    }
}

int main(void) {
    test_clock_monotonic();
    test_median_odd();
    test_median_even();
    test_median_single();
    test_median_invalid();
    test_measure_trials();
    test_dummy_range();
    test_print_roundtrip();
    test_machine_label();
    if (g_fail != 0) {
        printf("bench_smoke: FAIL\n");
        return 1;
    }
    printf("bench_smoke: OK\n");
    return 0;
}
