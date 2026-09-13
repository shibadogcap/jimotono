// JIMOTONO bench common harness implementation (ROADMAP 5.1/5.2).
// C11, restrict積極使用, errnoベース, goto cleanup, クロスプラットフォーム。
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "bench_common.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#include <time.h>
#else
#include <time.h>
#endif

int jt_bench_now_ns(uint64_t *restrict out) {
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
#if defined(_WIN32)
    static LARGE_INTEGER freq = {{0}};
    LARGE_INTEGER ctr = {{0}};
    if (freq.QuadPart == 0) {
        if (!QueryPerformanceFrequency(&freq) || freq.QuadPart <= 0) {
            errno = EIO;
            return JT_ERR_IO;
        }
    }
    if (!QueryPerformanceCounter(&ctr)) {
        errno = EIO;
        return JT_ERR_IO;
    }
    // オーバーフロー回避: 秒部と剰余部に分けてns化。
    {
        long long whole = ctr.QuadPart / freq.QuadPart;
        long long rem = ctr.QuadPart % freq.QuadPart;
        *out = (uint64_t)whole * 1000000000ULL
               + (uint64_t)((rem * 1000000000LL) / freq.QuadPart);
    }
    return JT_OK;
#elif defined(__APPLE__)
    // macOS fallback: mach_absolute_time (単調)。clock_gettimeが使える
    // 環境ではそちらを優先し、失敗時のみmachにフォールバックする。
#if defined(CLOCK_MONOTONIC)
    {
        struct timespec ts;
        memset(&ts, 0, sizeof ts);
        if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
            *out = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
            return JT_OK;
        }
        // clock_gettime失敗時はerrnoを保持したままmachへフォールバックする
        // (ENOSYS等の古いSDK対策)。mach成功時はerrnoを戻す。
    }
#endif
    {
        static mach_timebase_info_data_t tb = {0, 0};
        uint64_t t = mach_absolute_time();
        int saved_errno = errno;
        if (tb.denom == 0) {
            if (mach_timebase_info(&tb) != 0 || tb.denom == 0) {
                errno = EIO;
                return JT_ERR_IO;
            }
        }
        // t * numer / denom (ns)。128bitを使わず分割してオーバーフロー回避。
        {
            uint64_t q = t / tb.denom;
            uint64_t r = t % tb.denom;
            *out = q * (uint64_t)tb.numer + (r * (uint64_t)tb.numer) / (uint64_t)tb.denom;
        }
        errno = saved_errno;
        return JT_OK;
    }
#else
    {
        struct timespec ts;
        memset(&ts, 0, sizeof ts);
        if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
            // errnoはclock_gettimeが設定済み
            return JT_ERR_IO;
        }
        *out = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
        return JT_OK;
    }
#endif
}

int jt_bench_measure(jt_bench_case_fn fn, void *ctx, double *restrict ns_out) {
    uint64_t t0 = 0;
    uint64_t t1 = 0;
    int rc = JT_OK;

    if (fn == NULL || ns_out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    rc = jt_bench_now_ns(&t0);
    if (rc != JT_OK) {
        return rc;
    }
    fn(ctx);
    rc = jt_bench_now_ns(&t1);
    if (rc != JT_OK) {
        return rc;
    }
    *ns_out = (double)(t1 - t0);
    return JT_OK;
}

int jt_bench_trials(jt_bench_case_fn fn, void *ctx, size_t trials,
                    double *restrict samples_out) {
    if (fn == NULL || samples_out == NULL || trials == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (size_t i = 0; i < trials; i++) {
        double ns = 0.0;
        int rc = jt_bench_measure(fn, ctx, &ns);
        if (rc != JT_OK) {
            return rc;
        }
        samples_out[i] = ns;
    }
    return JT_OK;
}

static int jt_bench_cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) {
        return -1;
    }
    if (da > db) {
        return 1;
    }
    return 0;
}

static double jt_bench_median_sorted(const double *restrict sorted, size_t n) {
    if ((n % 2u) == 1u) {
        return sorted[n / 2u];
    }
    return (sorted[n / 2u - 1u] + sorted[n / 2u]) * 0.5;
}

int jt_bench_median_mad(const double *restrict samples, size_t n,
                        double *restrict median_out, double *restrict mad_out) {
    double *restrict tmp = NULL;
    double median = 0.0;
    double mad = 0.0;

    if (samples == NULL || median_out == NULL || mad_out == NULL || n == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(samples[i])) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    tmp = (double *)malloc(n * sizeof(double));
    if (tmp == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    memcpy(tmp, samples, n * sizeof(double));
    qsort(tmp, n, sizeof(double), jt_bench_cmp_double);
    median = jt_bench_median_sorted(tmp, n);
    // tmpを偏差バッファとして再利用: |x - median| の中央値 = MAD。
    for (size_t i = 0; i < n; i++) {
        tmp[i] = fabs(samples[i] - median);
    }
    qsort(tmp, n, sizeof(double), jt_bench_cmp_double);
    mad = jt_bench_median_sorted(tmp, n);
    free(tmp);
    *median_out = median;
    *mad_out = mad;
    return JT_OK;
}

int jt_bench_print_tsv(FILE *restrict fp, const char *restrict name, size_t trials,
                       double median_ns, double mad_ns) {
    if (fp == NULL || name == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (trials == 0 || !isfinite(median_ns) || !isfinite(mad_ns) || median_ns < 0.0
        || mad_ns < 0.0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (fprintf(fp, "%s\t%zu\t%.6f\t%.6f\n", name, trials, median_ns, mad_ns) < 0) {
        errno = EIO;
        return JT_ERR_IO;
    }
    if (fflush(fp) != 0) {
        return JT_ERR_IO;
    }
    return JT_OK;
}

// JSON文字列エスケープ (" と \ と制御文字)。NULLは""扱い(呼び出し側で正規化)。
static int jt_bench_json_write_str(FILE *restrict fp, const char *restrict s) {
    if (fputc('"', fp) == EOF) {
        return -1;
    }
    for (const unsigned char *p = (const unsigned char *)s; *p != '\0'; p++) {
        unsigned char c = *p;
        switch (c) {
        case '"':
            if (fputs("\\\"", fp) == EOF) {
                return -1;
            }
            break;
        case '\\':
            if (fputs("\\\\", fp) == EOF) {
                return -1;
            }
            break;
        case '\b':
            if (fputs("\\b", fp) == EOF) {
                return -1;
            }
            break;
        case '\f':
            if (fputs("\\f", fp) == EOF) {
                return -1;
            }
            break;
        case '\n':
            if (fputs("\\n", fp) == EOF) {
                return -1;
            }
            break;
        case '\r':
            if (fputs("\\r", fp) == EOF) {
                return -1;
            }
            break;
        case '\t':
            if (fputs("\\t", fp) == EOF) {
                return -1;
            }
            break;
        default:
            if (c < 0x20u) {
                if (fprintf(fp, "\\u%04x", (unsigned)c) < 0) {
                    return -1;
                }
            } else {
                if (fputc((int)c, fp) == EOF) {
                    return -1;
                }
            }
            break;
        }
    }
    if (fputc('"', fp) == EOF) {
        return -1;
    }
    return 0;
}

int jt_bench_print_json(FILE *restrict fp, const char *restrict name,
                        const char *restrict machine, const char *restrict commit,
                        size_t trials, double median_ns, double mad_ns,
                        const char *restrict notes) {
    const char *m = NULL;
    const char *c = NULL;
    const char *nt = NULL;

    if (fp == NULL || name == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (trials == 0 || !isfinite(median_ns) || !isfinite(mad_ns) || median_ns < 0.0
        || mad_ns < 0.0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    m = (machine != NULL) ? machine : "";
    c = (commit != NULL) ? commit : "";
    nt = (notes != NULL) ? notes : "";
    if (fputs("{\"name\":", fp) == EOF) {
        goto io_fail;
    }
    if (jt_bench_json_write_str(fp, name) != 0) {
        goto io_fail;
    }
    if (fputs(",\"machine\":", fp) == EOF) {
        goto io_fail;
    }
    if (jt_bench_json_write_str(fp, m) != 0) {
        goto io_fail;
    }
    if (fputs(",\"commit\":", fp) == EOF) {
        goto io_fail;
    }
    if (jt_bench_json_write_str(fp, c) != 0) {
        goto io_fail;
    }
    if (fprintf(fp, ",\"trials\":%zu,\"median_ns\":%.6f,\"mad_ns\":%.6f,\"notes\":",
                trials, median_ns, mad_ns)
        < 0) {
        goto io_fail;
    }
    if (jt_bench_json_write_str(fp, nt) != 0) {
        goto io_fail;
    }
    if (fputs("}\n", fp) == EOF) {
        goto io_fail;
    }
    if (fflush(fp) != 0) {
        goto io_fail;
    }
    return JT_OK;

io_fail:
    errno = EIO;
    return JT_ERR_IO;
}
