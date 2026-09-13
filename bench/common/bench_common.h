#pragma once
#ifndef JIMOTONO_BENCH_COMMON_H
#define JIMOTONO_BENCH_COMMON_H
// JIMOTONO bench common harness (ROADMAP 5.1/5.2).
// - 単調クロック (ns): Linux clock_gettime/CLOCK_MONOTONIC,
//   macOS clock_gettime (10.12+) / mach_absolute_time fallback,
//   Windows QueryPerformanceCounter fallback.
// - N回計測 -> 中央値 (median) + MAD (中央絶対偏差)。
// - TSV / JSON行出力 (results/bench/<date>-<machine>.json 用)。
//
// 規約: C11, restrict積極使用, errnoベース, クロスプラットフォーム
// (Linux/macOS/Windowsで同一ソースがビルド可能)。
// 単位: 時間はすべてナノ秒 (ns, double)。trialsは奇数推奨 (例: 11)。

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "jimotono/common.h"

// ベンチ対象の1回分の処理。計測オーバーヘッド外の準備は呼び出し側で行う。
typedef void (*jt_bench_case_fn)(void *ctx);

// 単調クロックの現在時刻 (ns)。
// 戻り値: JT_OK / JT_ERR_INVAL (out==NULL, errno=EINVAL)。
//          取得失敗時は JT_ERR_IO (errnoはOS呼び出しの値を保持)。
int jt_bench_now_ns(uint64_t *restrict out);

// fn(ctx)を1回実行し、経過nsを返す。
// 戻り値: JT_OK / JT_ERR_INVAL (fn==NULL || ns_out==NULL) / JT_ERR_IO (時計失敗)。
int jt_bench_measure(jt_bench_case_fn fn, void *ctx, double *restrict ns_out);

// fn(ctx)をtrials回実行し、経過nsをsamples_out[trials]に格納する。
// 戻り値: JT_OK / JT_ERR_INVAL (fn==NULL || samples_out==NULL || trials==0)
//          / JT_ERR_IO (時計失敗)。
int jt_bench_trials(jt_bench_case_fn fn, void *ctx, size_t trials,
                    double *restrict samples_out);

// samples[n]の中央値とMAD (median(|x - median|)) を求める。入力は変更しない。
// 偶数nの中央値は中央2値の平均 (MADも同様)。
// 非有限値 (NaN/Inf) 混入時は JT_ERR_INVAL。
// 戻り値: JT_OK / JT_ERR_INVAL (NULL, n==0, 非有限値混入, out==NULL)
//          / JT_ERR_NOMEM (内部コピー確保失敗)。
int jt_bench_median_mad(const double *restrict samples, size_t n,
                        double *restrict median_out, double *restrict mad_out);

// TSV行出力: "<name>\t<trials>\t<median_ns>\t<mad_ns>\n" (%.6f)。
// 戻り値: JT_OK / JT_ERR_INVAL (fp==NULL || name==NULL) / JT_ERR_IO (出力失敗)。
int jt_bench_print_tsv(FILE *restrict fp, const char *restrict name, size_t trials,
                       double median_ns, double mad_ns);

// JSON行出力 (1行1オブジェクト, docs/bench_schema.md準拠):
// {"name":"...","machine":"...","commit":"...","trials":N,
//  "median_ns":X,"mad_ns":Y,"notes":"..."}
// machine/commit/notesがNULLの場合は""として出力する。文字列はJSONエスケープする。
// 戻り値: JT_OK / JT_ERR_INVAL (fp==NULL || name==NULL) / JT_ERR_IO (出力失敗)。
int jt_bench_print_json(FILE *restrict fp, const char *restrict name,
                        const char *restrict machine, const char *restrict commit,
                        size_t trials, double median_ns, double mad_ns,
                        const char *restrict notes);

#endif  // JIMOTONO_BENCH_COMMON_H
