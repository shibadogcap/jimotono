// jt_gemm_mat_f32 / jt_transpose_f32: Phase G Step 4a マイクロカーネル。
// C11・restrict・errno＋goto cleanup・クロスプラット（AGENTS.MD 7.1）。
// AVX-512不使用。AVX2は#ifdef __AVX2__でガードし、非AVX2は同一k順スカラー。
// FMA不使用（mul/add分離でbit同一）。

#include "jimotono/moe_gemm.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

int jt_transpose_f32(const float *restrict src, float *restrict dst, int rows,
                     int cols) {
    int rc = JT_ERR_INVAL;
    if (src == NULL || dst == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows <= 0 || cols <= 0) {
        errno = EINVAL;
        goto cleanup;
    }
    if ((size_t)rows > SIZE_MAX / (size_t)cols) {
        errno = EINVAL;
        goto cleanup;
    }
    // exact転置。順序依存なし（要素独立）。
    for (int r = 0; r < rows; r++) {
        const float *srow = src + (size_t)r * (size_t)cols;
        for (int c = 0; c < cols; c++) {
            dst[(size_t)c * (size_t)rows + (size_t)r] = srow[c];
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}

// スカラー参照核（f32蓄積・k逐次）。AVX2核とbit一致する。
// 非AVX2ビルドでのみ使用（AVX2時は#else側が有効化されるため未定義化）。
#ifndef __AVX2__
static void jt_mat_scalar_block(const float *restrict A, const float *restrict B,
                                float *restrict C, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        const float *arow = A + (size_t)m * (size_t)K;
        float *crow = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                // f32 mul→f32 addの順序をAVX2核と同一にする（FMAなし）。
                acc += arow[k] * B[(size_t)k * (size_t)N + (size_t)n];
            }
            crow[n] = acc;
        }
    }
}
#endif  // __AVX2__ fallback reference kernel only

#ifdef __AVX2__
// 1 Nベクトル（8 f32）× mb行（mb≤12）のマイクロ。
// acc 12本（ymm0–ymm11）がmb行に対応。B用1本＋A broadcast用はレーン内で再利用し
// 同時live ≤15本（D6準拠）。k順序は逐次でスカラー参照と同一。
static void jt_mat_micro_8xmb(const float *restrict A, const float *restrict B,
                               float *restrict C, int mb, int K, int N) {
    // acc[12]（mb行分のみ使用）。未使用レーンは触れない。
    __m256 acc[12];
    for (int m = 0; m < mb; m++) {
        acc[m] = _mm256_setzero_ps();
    }
    for (int k = 0; k < K; k++) {
        // B[k][n0..n0+7]（連続8）。
        __m256 bv = _mm256_loadu_ps(B + (size_t)k * (size_t)N);
        for (int m = 0; m < mb; m++) {
            // A[m][k]のbroadcast（strided loadのスカラー→broadcast）。
            // 順序はk逐次で固定（bit同一）。
            __m256 av = _mm256_set1_ps(A[(size_t)m * (size_t)K + (size_t)k]);
            __m256 pv = _mm256_mul_ps(av, bv);
            acc[m] = _mm256_add_ps(acc[m], pv);
        }
    }
    for (int m = 0; m < mb; m++) {
        _mm256_storeu_ps(C + (size_t)m * (size_t)N, acc[m]);
    }
}
#endif

int jt_gemm_mat_f32(const float *restrict A, const float *restrict B,
                    float *restrict C, int M, int N, int K) {
    int rc = JT_ERR_INVAL;
    if (A == NULL || B == NULL || C == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    // M==0は起動スキップ（C不変でJT_OK。§4.2）。
    if (M == 0) {
        if (N <= 0 || K <= 0) {
            errno = EINVAL;
            goto cleanup;
        }
        rc = JT_OK;
        goto cleanup;
    }
    if (M < 0 || N <= 0 || K <= 0) {
        errno = EINVAL;
        goto cleanup;
    }
    if ((size_t)M > SIZE_MAX / (size_t)K ||
        (size_t)M > SIZE_MAX / (size_t)N ||
        (size_t)K > SIZE_MAX / (size_t)N) {
        errno = EINVAL;
        goto cleanup;
    }
    // M方向はMR=12でブロック（M_e mod 12テール・M_e<12はテール経路のみ）。
    // N方向は論理NR=4。AVX2では8-wide（2ブロック融合）で処理し、
    // 剰余（N%8・N%4）はスカラーフォールバックで吸収する。
    for (int m0 = 0; m0 < M; m0 += JT_GEMM_MR) {
        int mb = M - m0;
        if (mb > JT_GEMM_MR) {
            mb = JT_GEMM_MR;
        }
        const float *Ab = A + (size_t)m0 * (size_t)K;
        float *Cb = C + (size_t)m0 * (size_t)N;
#ifdef __AVX2__
        {
            int n0 = 0;
            int nvec = N & ~7;
            for (; n0 < nvec; n0 += 8) {
                jt_mat_micro_8xmb(Ab, B + n0, Cb + n0, mb, K, N);
            }
            // テール（N%8。N%4を含む）：同一k順スカラーでbit一致。
            if (n0 < N) {
                for (int m = 0; m < mb; m++) {
                    const float *arow = Ab + (size_t)m * (size_t)K;
                    float *crow = Cb + (size_t)m * (size_t)N;
                    for (int n = n0; n < N; n++) {
                        float acc = 0.0f;
                        for (int k = 0; k < K; k++) {
                            acc += arow[k] *
                                   B[(size_t)k * (size_t)N + (size_t)n];
                        }
                        crow[n] = acc;
                    }
                }
            }
        }
#else
        // 非AVX2は全て同一k順スカラー（AVX2核とbit一致）。
        jt_mat_scalar_block(Ab, B, Cb, mb, N, K);
#endif
    }
    // 有限確認（公開fail-closedのため。スクラッチ部分更新の検出用。
    // 呼出し側が公開出力を更新しないことでfail-closedを担保する）。
    for (int m = 0; m < M; m++) {
        const float *crow = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; n++) {
            if (!isfinite((double)crow[n])) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}
