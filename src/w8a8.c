// jt_w8a8: Stage 3 W8A8 INT8量子化・逆量子化・int8 GEMM。
// P3b §3準拠 (W8A8本線。T-MAC/INT4/INT2は非スコープ)。
// C11・restrict・errno＋goto cleanup・クロスプラット (AGENTS.MD 7.1)。
// Stage 3b-1: int8 GEMMの同一関数内SIMD最適化 (G3)。
//   バックエンド切替 (コンパイル時 dispatch。公開API・検証・tol不変):
//   - AVX2 emul (既定の高速路。i7-8700B/N100で実行): 連続dotは
//     _mm256_maddubs_epi16＋_mm256_madd_epi16 のexact構成 (even/odd分離)。
//     q∈[-127,127] (-128不使用) のため中間pair和は最大32258<32767で
//     飽和せず、スカラーとbit同一。#ifdef __AVX2__ ガード。
//     GEMMは形状dispatch: N==1はdot直結 (B列連続)、小Mは
//     アウタープロダクト型 (cvt/mullo exact。madd系のpair加算は
//     k方向reduction専用のためN方向には使えない) でpack不要、
//     M≥16はpanel転置pack＋連続dot (pack税が希釈される域のみ)。
//   - AVX512-VNNI (Ryzen AI 360用): _mm512_dpbssd_epi32 (s8*s8 exact) の
//     連続dotのみ。手元verify不可のため一般GEMMはAVX2路を共用し、
//     未検証コードを最小化。コンパイル確認＋単体テストの等価性で担保。
//     注: clangに __AVX512__ マクロは存在しないため
//     defined(__AVX512F__) && defined(__AVX512VNNI__) でガードする。
//   - AVX-VNNI (N100。__AVXVNNIINT8__): 将来パス予約のみ (本TUはAVX2路に
//     フォールスルー。_mm256_dpbssd_epi32 は -mavxvnniint8 で利用可を
//     確認済み。plain -mavxvnniでは不可)。NEON-I8MMは将来予約。いずれも非対応CPUでは同一k順
//     スカラーフォールバック (bit同一)。
//   - FMA不使用。整数accは加算順序不変で完全一致するため、float化の
//     順序を既存と同一に保てば出力はbit同一 (G3: bit一致 or 相対差1e-6)。

#include "jimotono/w8a8.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "jimotono/common.h"
#include "jimotono/train_bwd.h"

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVXVNNI__)
#include <immintrin.h>
#endif

// 既定OFF (fp32)。プロセス内グローバル。
static int g_jt_w8a8_enabled = 0;

int jt_w8a8_set_enabled(int enabled) {
    if (enabled != 0 && enabled != 1) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    g_jt_w8a8_enabled = enabled;
    return JT_OK;
}

int jt_w8a8_get_enabled(int *restrict out_enabled) {
    if (out_enabled == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_enabled = g_jt_w8a8_enabled;
    return JT_OK;
}

int jt_w8a8_is_enabled(void) { return g_jt_w8a8_enabled; }

// 前方宣言 (SIMD路のフォールバックで使用)。
static int jt_w8a8_gemm_scalar_body(const int8_t *restrict Aq,
                                    const int8_t *restrict Bq, double sA_d,
                                    const float *restrict sB_col,
                                    float sB_single, float *restrict C,
                                    int M, int N, int K);

// ---- Stage 3b-1 SIMDバックエンド (同一関数内。公開API不変) ----
// 整数accは数学的に順序不変 (K≤JT_BWD_MAX_WIDEで|acc|≤6.6e7<2^31) のため、
// どのバックエンドも同一accを与え、後段のfloat化 (既存順序) と合わせれば
// 出力はbit同一になる (G3)。検証・q値域検査は呼出し側の既存路が先に行う。

#if defined(__AVX2__) && \
    !(defined(__AVX512F__) && defined(__AVX512VNNI__))
// 連続int8 dot (exact)。maddubsはu8*s8のpair加算 (int16飽和) のため、
// even/odd分離で単一積相当を取り出す: pat=[1,0..]で偶レーン、
// [0,1..]で奇レーン。q∈[-127,127]より抽出値は±127以内で飽和なし。
// mullo積は最大16129、even+oddは最大32258<32767でexact。
// maddはint16→int32のpair加算 (exact)。合計はint32でexact。
static int32_t jt_w8a8_dot_avx2(const int8_t *restrict a,
                                const int8_t *restrict b, int k) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i pat10 = _mm256_set_epi8(
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1);
    const __m256i pat01 = _mm256_set_epi8(
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0);
    const __m256i ones = _mm256_set1_epi16(1);
    int i = 0;
    int n = k & ~31;
    for (; i < n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i eA = _mm256_maddubs_epi16(pat10, va);
        __m256i oA = _mm256_maddubs_epi16(pat01, va);
        __m256i eB = _mm256_maddubs_epi16(pat10, vb);
        __m256i oB = _mm256_maddubs_epi16(pat01, vb);
        __m256i eP = _mm256_mullo_epi16(eA, eB);
        __m256i oP = _mm256_mullo_epi16(oA, oB);
        __m256i s = _mm256_add_epi16(eP, oP);
        __m256i r = _mm256_madd_epi16(s, ones);
        acc = _mm256_add_epi32(acc, r);
    }
    {
        __m128i lo = _mm256_castsi256_si128(acc);
        __m128i hi = _mm256_extracti128_si256(acc, 1);
        __m128i s4 = _mm_add_epi32(lo, hi);
        __m128i s2 = _mm_add_epi32(s4, _mm_srli_si128(s4, 8));
        __m128i s1 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
        int32_t total = _mm_cvtsi128_si32(s1);
        for (; i < k; i++) {
            total += (int32_t)a[i] * (int32_t)b[i];
        }
        return total;
    }
}
#endif  // AVX2 dot (VNNIビルドではvnni dotを使用するため除外)

#ifdef __AVX2__
// アウタープロダクト型ブロック (exact)。固定kの寄与 a*Brow[n] (n方向) は
// レーン毎に独立のためpair加算系 (maddubs/madd) は使えない。
// int8→int32符号拡張＋mullo＋addでexactに累積する。k順逐次。
// 32/16/8幅ブロック＋スカラーテール。accはレジスタ保持 (k間でメモリ不経由)。
static void jt_w8a8_block32_avx2(const int8_t *restrict arow,
                                 const int8_t *restrict Bq, int N, int K,
                                 int n0, int32_t *restrict out) {
    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    __m256i c2 = _mm256_setzero_si256();
    __m256i c3 = _mm256_setzero_si256();
    for (int k = 0; k < K; k++) {
        const int8_t *brow = Bq + (size_t)k * (size_t)N + (size_t)n0;
        __m256i bv = _mm256_loadu_si256((const __m256i *)brow);
        __m128i lo128 = _mm256_castsi256_si128(bv);
        __m128i hi128 = _mm256_extracti128_si256(bv, 1);
        __m256i lo16 = _mm256_cvtepi8_epi16(lo128);
        __m256i hi16 = _mm256_cvtepi8_epi16(hi128);
        __m256i b0 =
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo16));
        __m256i b1 =
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo16, 1));
        __m256i b2 =
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(hi16));
        __m256i b3 =
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256(hi16, 1));
        __m256i a0 = _mm256_set1_epi32((int)arow[k]);
        c0 = _mm256_add_epi32(c0, _mm256_mullo_epi32(a0, b0));
        c1 = _mm256_add_epi32(c1, _mm256_mullo_epi32(a0, b1));
        c2 = _mm256_add_epi32(c2, _mm256_mullo_epi32(a0, b2));
        c3 = _mm256_add_epi32(c3, _mm256_mullo_epi32(a0, b3));
    }
    _mm256_storeu_si256((__m256i *)(out + 0), c0);
    _mm256_storeu_si256((__m256i *)(out + 8), c1);
    _mm256_storeu_si256((__m256i *)(out + 16), c2);
    _mm256_storeu_si256((__m256i *)(out + 24), c3);
}

static void jt_w8a8_block16_avx2(const int8_t *restrict arow,
                                 const int8_t *restrict Bq, int N, int K,
                                 int n0, int32_t *restrict out) {
    __m256i c0 = _mm256_setzero_si256();
    __m256i c1 = _mm256_setzero_si256();
    for (int k = 0; k < K; k++) {
        const int8_t *brow = Bq + (size_t)k * (size_t)N + (size_t)n0;
        __m128i bv = _mm_loadu_si128((const __m128i *)brow);
        __m256i w16 = _mm256_cvtepi8_epi16(bv);
        __m256i b0 =
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w16));
        __m256i b1 =
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w16, 1));
        __m256i a0 = _mm256_set1_epi32((int)arow[k]);
        c0 = _mm256_add_epi32(c0, _mm256_mullo_epi32(a0, b0));
        c1 = _mm256_add_epi32(c1, _mm256_mullo_epi32(a0, b1));
    }
    _mm256_storeu_si256((__m256i *)(out + 0), c0);
    _mm256_storeu_si256((__m256i *)(out + 8), c1);
}

static void jt_w8a8_block8_avx2(const int8_t *restrict arow,
                                const int8_t *restrict Bq, int N, int K,
                                int n0, int32_t *restrict out) {
    __m256i c0 = _mm256_setzero_si256();
    for (int k = 0; k < K; k++) {
        const int8_t *brow = Bq + (size_t)k * (size_t)N + (size_t)n0;
        __m128i bv = _mm_loadl_epi64((const __m128i *)brow);
        __m128i w16 = _mm_cvtepi8_epi16(bv);
        __m256i b0 = _mm256_cvtepi16_epi32(w16);
        __m256i a0 = _mm256_set1_epi32((int)arow[k]);
        c0 = _mm256_add_epi32(c0, _mm256_mullo_epi32(a0, b0));
    }
    _mm256_storeu_si256((__m256i *)out, c0);
}
#endif  // __AVX2__

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
// AVX512-VNNI連続dot (Ryzen用。手元実行不可のため最小限)。
// _mm512_dpbssd_epi32 (s8*s8) は手元のclangに未提供のため、
// _mm512_dpbusd_epi32 (u8*s8)＋0x80バイアス補正でexactに求める。
// dpbusdはint32直接蓄積 (int16飽和なし) のため補正式
//   raw=Σ(a+128)*b=Σab＋128*Σb  →  Σab=raw−128*Σb
// は全域でexact (|Σb|≤127*K≤5.2e5、128*Σb≤6.7e7<2^31)。
// 一般GEMMはAVX2路を共用し、本関数はN==1路でのみ使用する。
static int32_t jt_w8a8_dot_vnni(const int8_t *restrict a,
                                const int8_t *restrict b, int k) {
    __m512i acc = _mm512_setzero_si512();
    const __m512i bias = _mm512_set1_epi8((char)0x80);
    int i = 0;
    int n = k & ~63;
    int32_t raw = 0;
    int32_t sumB = 0;
    for (; i < n; i += 64) {
        __m512i va = _mm512_loadu_si512((const void *)(a + i));
        __m512i vb = _mm512_loadu_si512((const void *)(b + i));
        va = _mm512_xor_si512(va, bias);
        acc = _mm512_dpbusd_epi32(acc, va, vb);
    }
    {
        __m256i lo = _mm512_castsi512_si256(acc);
        __m256i hi = _mm512_extracti64x4_epi64(acc, 1);
        __m256i s = _mm256_add_epi32(lo, hi);
        __m128i s128 = _mm_add_epi32(_mm256_castsi256_si128(s),
                                     _mm256_extracti128_si256(s, 1));
        __m128i s2 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
        __m128i s1 = _mm_add_epi32(s2, _mm_srli_si128(s2, 4));
        raw = _mm_cvtsi128_si32(s1);
    }
    for (; i < k; i++) {
        raw += (int32_t)((uint8_t)(a[i] ^ (char)0x80)) * (int32_t)b[i];
    }
    for (int j = 0; j < k; j++) {
        sumB += (int32_t)b[j];
    }
    return raw - 128 * sumB;
}
#endif  // AVX512-VNNI

// AVX-VNNI (N100): 将来パス予約。本TUはAVX2路にフォールスルーする
// (実装任意のため)。調査結果 (Apple Clang 17):
//   - plain -mavxvnni (__AVXVNNI__) ではint8ドット intrinsicは使えない
//     (_mm256_dpbssd_epi32は 'avxvnniint8' featureを要求しエラー)。
//   - -mavxvnniint8 (__AVXVNNIINT8__) で _mm256_dpbssd_epi32 (s8*s8,
//     int32蓄積・補正不要・exact) が利用可 (コンパイル確認済み)。
// 将来N100向けビルドでは -mavxvnniint8 をTU毎に付け、
//   #ifdef __AVXVNNIINT8__ → _mm256_dpbssd_epi32連続dot
// の分岐をjt_w8a8_dot_*と同形で追加する (本変更のVNNI dotを参照)。
// NEON-I8MM (ARM64) は将来予約。

#if defined(__AVX2__) || (defined(__AVX512F__) && defined(__AVX512VNNI__))
// 最良dot選択 (N==1路・panel路で共用)。いずれも整数exact。
static int32_t jt_w8a8_dot_best(const int8_t *restrict a,
                                const int8_t *restrict b, int k) {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    return jt_w8a8_dot_vnni(a, b, k);
#elif defined(__AVX2__)
    return jt_w8a8_dot_avx2(a, b, k);
#else
    {
        int32_t total = 0;
        for (int i = 0; i < k; i++) {
            total += (int32_t)a[i] * (int32_t)b[i];
        }
        return total;
    }
#endif
}

// panel-pack＋dot路 (M>1用)。BのNブロック (NB列) をP[j][k]に転置packし、
// 各行を連続dotで処理する。packはNブロック毎にM行で共用するため、
// Mが大きいほどpack税が希釈される。しきい値未満の小Mは呼出し側で
// アウタープロダクト型を使う (pack税が逆転するため)。
// malloc失敗時はC未更新のままスカラー本体に委譲 (fail-closed)。
// 整数exactのため出力はbit同一。
#define JT_W8A8_PANEL_NB 32
// panel路を使う最小M (実測で設定。小Mはpack転置税が支配的)。
#define JT_W8A8_PANEL_MIN_M 16
static int jt_w8a8_gemm_panel(const int8_t *restrict Aq,
                              const int8_t *restrict Bq, double sA_d,
                              const float *restrict sB_col, float sB_single,
                              float *restrict C, int M, int N, int K) {
    int8_t *P = NULL;
    if (K <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if ((size_t)K > SIZE_MAX / (size_t)JT_W8A8_PANEL_NB) {
        errno = ERANGE;
        return JT_ERR_INVAL;
    }
    P = (int8_t *)malloc((size_t)JT_W8A8_PANEL_NB * (size_t)K);
    if (P == NULL) {
        return jt_w8a8_gemm_scalar_body(Aq, Bq, sA_d, sB_col, sB_single, C,
                                        M, N, K);
    }
    for (int n0 = 0; n0 < N; n0 += JT_W8A8_PANEL_NB) {
        int nb = N - n0;
        if (nb > JT_W8A8_PANEL_NB) {
            nb = JT_W8A8_PANEL_NB;
        }
        for (int j = 0; j < nb; j++) {
            int8_t *prow = P + (size_t)j * (size_t)K;
            int nc = n0 + j;
            for (int k = 0; k < K; k++) {
                prow[k] = Bq[(size_t)k * (size_t)N + (size_t)nc];
            }
        }
        for (int m = 0; m < M; m++) {
            const int8_t *arow = Aq + (size_t)m * (size_t)K;
            float *crow = C + (size_t)m * (size_t)N + n0;
            for (int j = 0; j < nb; j++) {
                int32_t acc =
                    jt_w8a8_dot_best(arow, P + (size_t)j * (size_t)K, K);
                double ss = sB_col ? sA_d * (double)sB_col[n0 + j]
                                   : sA_d * (double)sB_single;
                double v = 0.0;
                float f = 0.0f;
                if (!isfinite(ss) || ss <= 0.0) {
                    free(P);
                    errno = ERANGE;
                    return JT_ERR_INVAL;
                }
                v = (double)acc * ss;
                f = (float)v;
                if (!isfinite((double)f)) {
                    free(P);
                    errno = ERANGE;
                    return JT_ERR_INVAL;
                }
                crow[j] = f;
            }
        }
    }
    free(P);
    return JT_OK;
}

// 高速GEMM本体 (同一関数内dispatch先)。呼出し前に検証・q値域検査・
// scale有限検査が済んでいること。sB_col!=NULLでper-channel列scale、
// NULLでper-tensor単一scale (sB_single)。
// float化は既存路と同一式・同一順序のため出力はbit同一。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=ERANGE。既存路と同一)。
static int jt_w8a8_gemm_fast(const int8_t *restrict Aq,
                             const int8_t *restrict Bq, double sA_d,
                             const float *restrict sB_col, float sB_single,
                             float *restrict C, int M, int N, int K) {
    if (N == 1) {
        // B列は連続 (stride 1) のためdot直結。
        double ss =
            sB_col ? sA_d * (double)sB_col[0] : sA_d * (double)sB_single;
        if (!isfinite(ss) || ss <= 0.0) {
            errno = ERANGE;
            return JT_ERR_INVAL;
        }
        for (int m = 0; m < M; m++) {
            const int8_t *arow = Aq + (size_t)m * (size_t)K;
            int32_t acc = jt_w8a8_dot_best(arow, Bq, K);
            double v = 0.0;
            float f = 0.0f;
            v = (double)acc * ss;
            f = (float)v;
            if (!isfinite((double)f)) {
                errno = ERANGE;
                return JT_ERR_INVAL;
            }
            C[(size_t)m * (size_t)N] = f;
        }
        return JT_OK;
    }
#ifdef __AVX2__
    if (M < JT_W8A8_PANEL_MIN_M) {
        // decode GEMV相当〜小M: pack税なしのアウタープロダクト型が最速。
        for (int m = 0; m < M; m++) {
        const int8_t *arow = Aq + (size_t)m * (size_t)K;
        float *crow = C + (size_t)m * (size_t)N;
        int n0 = 0;
        int n32 = N & ~31;
        for (; n0 < n32; n0 += 32) {
            int32_t buf[32];
            jt_w8a8_block32_avx2(arow, Bq, N, K, n0, buf);
            for (int j = 0; j < 32; j++) {
                double ss = sB_col ? sA_d * (double)sB_col[n0 + j]
                                   : sA_d * (double)sB_single;
                double v = 0.0;
                float f = 0.0f;
                if (!isfinite(ss) || ss <= 0.0) {
                    errno = ERANGE;
                    return JT_ERR_INVAL;
                }
                v = (double)buf[j] * ss;
                f = (float)v;
                if (!isfinite((double)f)) {
                    errno = ERANGE;
                    return JT_ERR_INVAL;
                }
                crow[n0 + j] = f;
            }
        }
        if (N - n0 >= 16) {
            int32_t buf[16];
            jt_w8a8_block16_avx2(arow, Bq, N, K, n0, buf);
            for (int j = 0; j < 16; j++) {
                double ss = sB_col ? sA_d * (double)sB_col[n0 + j]
                                   : sA_d * (double)sB_single;
                double v = 0.0;
                float f = 0.0f;
                if (!isfinite(ss) || ss <= 0.0) {
                    errno = ERANGE;
                    return JT_ERR_INVAL;
                }
                v = (double)buf[j] * ss;
                f = (float)v;
                if (!isfinite((double)f)) {
                    errno = ERANGE;
                    return JT_ERR_INVAL;
                }
                crow[n0 + j] = f;
            }
            n0 += 16;
        }
        if (N - n0 >= 8) {
            int32_t buf[8];
            jt_w8a8_block8_avx2(arow, Bq, N, K, n0, buf);
            for (int j = 0; j < 8; j++) {
                double ss = sB_col ? sA_d * (double)sB_col[n0 + j]
                                   : sA_d * (double)sB_single;
                double v = 0.0;
                float f = 0.0f;
                if (!isfinite(ss) || ss <= 0.0) {
                    errno = ERANGE;
                    return JT_ERR_INVAL;
                }
                v = (double)buf[j] * ss;
                f = (float)v;
                if (!isfinite((double)f)) {
                    errno = ERANGE;
                    return JT_ERR_INVAL;
                }
                crow[n0 + j] = f;
            }
            n0 += 8;
        }
        for (; n0 < N; n0++) {
            int32_t acc = 0;
            double ss = sB_col ? sA_d * (double)sB_col[n0]
                               : sA_d * (double)sB_single;
            double v = 0.0;
            float f = 0.0f;
            if (!isfinite(ss) || ss <= 0.0) {
                errno = ERANGE;
                return JT_ERR_INVAL;
            }
            for (int k = 0; k < K; k++) {
                acc += (int32_t)arow[k] *
                       (int32_t)Bq[(size_t)k * (size_t)N + (size_t)n0];
            }
            v = (double)acc * ss;
            f = (float)v;
            if (!isfinite((double)f)) {
                errno = ERANGE;
                return JT_ERR_INVAL;
            }
            crow[n0] = f;
        }
        }  // for m
        return JT_OK;
    }  // if (M < PANEL_MIN_M)
    // M>=MIN_M: pack税が希釈されるpanel＋dot路へ。
    return jt_w8a8_gemm_panel(Aq, Bq, sA_d, sB_col, sB_single, C, M, N, K);
#else
    // スカラー一般路 (AVX2なしVNNI単体等の理論経路。整数exactでbit同一)。
    return jt_w8a8_gemm_scalar_body(Aq, Bq, sA_d, sB_col, sB_single, C, M,
                                    N, K);
#endif
}
#endif  // SIMD fast GEMM body

static int jt_w8a8_valid_dims_q(size_t n) { return n > 0; }

// スカラーGEMM本体 (全バックエンドの基準・フォールバック)。
// sB_col!=NULLでper-channel列scale、NULLでper-tensor単一scale。
// 戻り値: JT_OK / JT_ERR_INVAL (errno=ERANGE。既存路と同一)。
static int jt_w8a8_gemm_scalar_body(const int8_t *restrict Aq,
                                    const int8_t *restrict Bq, double sA_d,
                                    const float *restrict sB_col,
                                    float sB_single, float *restrict C,
                                    int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        const int8_t *arow = Aq + (size_t)m * (size_t)K;
        float *crow = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            double ss = sB_col ? sA_d * (double)sB_col[n]
                               : sA_d * (double)sB_single;
            double v = 0.0;
            float f = 0.0f;
            if (!isfinite(ss) || ss <= 0.0) {
                errno = ERANGE;
                return JT_ERR_INVAL;
            }
            for (int k = 0; k < K; k++) {
                acc += (int32_t)arow[k] *
                       (int32_t)Bq[(size_t)k * (size_t)N + (size_t)n];
            }
            v = (double)acc * ss;
            f = (float)v;
            if (!isfinite((double)f)) {
                errno = ERANGE;
                return JT_ERR_INVAL;
            }
            crow[n] = f;
        }
    }
    return JT_OK;
}

int jt_w8a8_quant_per_tensor(const float *restrict src, size_t n,
                             int8_t *restrict dst,
                             float *restrict out_scale) {
    int rc = JT_ERR_INVAL;
    double maxabs = 0.0;
    double scale_d = 0.0;
    float scale_f = 1.0f;
    if (src == NULL || dst == NULL || out_scale == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_w8a8_valid_dims_q(n)) {
        errno = EINVAL;
        goto cleanup;
    }
    // fail-closed: 先に全要素の有限性を検査し、NaN/Inf混入時は書き込まない。
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)src[i])) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < n; i++) {
        double a = fabs((double)src[i]);
        if (a > maxabs) {
            maxabs = a;
        }
    }
    if (maxabs == 0.0) {
        *out_scale = 1.0f;
        for (size_t i = 0; i < n; i++) {
            dst[i] = 0;
        }
        rc = JT_OK;
        goto cleanup;
    }
    scale_d = maxabs / (double)JT_W8A8_QMAX;
    if (!isfinite(scale_d) || scale_d <= 0.0) {
        errno = ERANGE;
        goto cleanup;
    }
    scale_f = (float)scale_d;
    if (!isfinite((double)scale_f) || scale_f <= 0.0f) {
        errno = ERANGE;
        goto cleanup;
    }
    *out_scale = scale_f;
    for (size_t i = 0; i < n; i++) {
        double qd = round((double)src[i] / scale_d);
        if (qd > (double)JT_W8A8_QMAX) {
            qd = (double)JT_W8A8_QMAX;
        } else if (qd < (double)JT_W8A8_QMIN) {
            qd = (double)JT_W8A8_QMIN;
        }
        dst[i] = (int8_t)((int)qd);
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_dequant_per_tensor(const int8_t *restrict src, float scale,
                               size_t n, float *restrict dst) {
    int rc = JT_ERR_INVAL;
    double sc = 0.0;
    if (src == NULL || dst == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_w8a8_valid_dims_q(n)) {
        errno = EINVAL;
        goto cleanup;
    }
    sc = (double)scale;
    if (!isfinite(sc) || sc <= 0.0) {
        errno = EINVAL;
        goto cleanup;
    }
    for (size_t i = 0; i < n; i++) {
        int qv = (int)src[i];
        if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < n; i++) {
        double v = (double)((int)src[i]) * sc;
        float f = (float)v;
        if (!isfinite((double)f)) {
            errno = ERANGE;
            goto cleanup;
        }
        dst[i] = f;
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_quant_per_channel(const float *restrict src, size_t rows,
                              size_t cols, int8_t *restrict dst,
                              float *restrict scales) {
    int rc = JT_ERR_INVAL;
    if (src == NULL || dst == NULL || scales == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows == 0 || cols == 0) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows > SIZE_MAX / cols) {
        errno = EINVAL;
        goto cleanup;
    }
    {
        size_t n = rows * cols;
        for (size_t i = 0; i < n; i++) {
            if (!isfinite((double)src[i])) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        const float *srow = src + r * cols;
        int8_t *drow = dst + r * cols;
        double maxabs = 0.0;
        double scale_d = 0.0;
        float scale_f = 1.0f;
        for (size_t c = 0; c < cols; c++) {
            double a = fabs((double)srow[c]);
            if (a > maxabs) {
                maxabs = a;
            }
        }
        if (maxabs == 0.0) {
            scales[r] = 1.0f;
            for (size_t c = 0; c < cols; c++) {
                drow[c] = 0;
            }
            continue;
        }
        scale_d = maxabs / (double)JT_W8A8_QMAX;
        if (!isfinite(scale_d) || scale_d <= 0.0) {
            errno = ERANGE;
            goto cleanup;
        }
        scale_f = (float)scale_d;
        if (!isfinite((double)scale_f) || scale_f <= 0.0f) {
            errno = ERANGE;
            goto cleanup;
        }
        scales[r] = scale_f;
        for (size_t c = 0; c < cols; c++) {
            double qd = round((double)srow[c] / scale_d);
            if (qd > (double)JT_W8A8_QMAX) {
                qd = (double)JT_W8A8_QMAX;
            } else if (qd < (double)JT_W8A8_QMIN) {
                qd = (double)JT_W8A8_QMIN;
            }
            drow[c] = (int8_t)((int)qd);
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_dequant_per_channel(const int8_t *restrict src,
                                const float *restrict scales, size_t rows,
                                size_t cols, float *restrict dst) {
    int rc = JT_ERR_INVAL;
    if (src == NULL || scales == NULL || dst == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows == 0 || cols == 0) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows > SIZE_MAX / cols) {
        errno = EINVAL;
        goto cleanup;
    }
    for (size_t r = 0; r < rows; r++) {
        double sc = (double)scales[r];
        if (!isfinite(sc) || sc <= 0.0) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    {
        size_t n = rows * cols;
        for (size_t i = 0; i < n; i++) {
            int qv = (int)src[i];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        double sc = (double)scales[r];
        const int8_t *srow = src + r * cols;
        float *drow = dst + r * cols;
        for (size_t c = 0; c < cols; c++) {
            double v = (double)((int)srow[c]) * sc;
            float f = (float)v;
            if (!isfinite((double)f)) {
                errno = ERANGE;
                goto cleanup;
            }
            drow[c] = f;
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}

// int8 GEMM共通検証。M==0は呼出し側で起動スキップ扱い。
static int jt_w8a8_gemm_validate(const int8_t *restrict Aq,
                                 const int8_t *restrict Bq, float sA,
                                 const float *restrict sB_col,
                                 const float *restrict C, int M, int N,
                                 int K, int need_col) {
    if (Aq == NULL || Bq == NULL || C == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (M < 0 || N <= 0 || K <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (M == 0) {
        return JT_OK;  // 起動スキップ (呼出し側でC不変)
    }
    if (M > JT_BWD_MAX_WIDE || N > JT_BWD_MAX_WIDE || K > JT_BWD_MAX_WIDE) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if ((size_t)M > SIZE_MAX / (size_t)K ||
        (size_t)M > SIZE_MAX / (size_t)N ||
        (size_t)K > SIZE_MAX / (size_t)N) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (!isfinite((double)sA) || sA <= 0.0f) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (need_col) {
        if (sB_col == NULL) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        for (int n = 0; n < N; n++) {
            if (!isfinite((double)sB_col[n]) || sB_col[n] <= 0.0f) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
        }
    }
    return JT_OK;
}

int jt_w8a8_gemm_per_tensor(const int8_t *restrict Aq,
                            const int8_t *restrict Bq, float sA, float sB,
                            float *restrict C, int M, int N, int K) {
    int rc = JT_ERR_INVAL;
    if (M == 0) {
        // jt_gemm_mat_f32と同一契約: M==0はC不変でJT_OK。ただしN/Kは正。
        if (Aq == NULL || Bq == NULL || C == NULL) {
            errno = EINVAL;
            goto cleanup;
        }
        if (N <= 0 || K <= 0) {
            errno = EINVAL;
            goto cleanup;
        }
        rc = JT_OK;
        goto cleanup;
    }
    if (jt_w8a8_gemm_validate(Aq, Bq, sA, NULL, C, M, N, K, 0) != JT_OK) {
        goto cleanup;  // errnoは下位で設定済み
    }
    if (!isfinite((double)sB) || sB <= 0.0f) {
        errno = EINVAL;
        goto cleanup;
    }
    // q値域検査 (破損qの黙殺防止。O(MK+KN))。
    for (int m = 0; m < M; m++) {
        const int8_t *arow = Aq + (size_t)m * (size_t)K;
        for (int k = 0; k < K; k++) {
            int qv = (int)arow[k];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (int k = 0; k < K; k++) {
        const int8_t *brow = Bq + (size_t)k * (size_t)N;
        for (int n = 0; n < N; n++) {
            int qv = (int)brow[n];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    {
        double ss = (double)sA * (double)sB;
        if (!isfinite(ss) || ss <= 0.0) {
            errno = ERANGE;
            goto cleanup;
        }
#if defined(__AVX2__) || \
    (defined(__AVX512F__) && defined(__AVX512VNNI__))
        // G3同一関数内最適化: 検証済みのまま高速路へ (bit同一)。
        rc = jt_w8a8_gemm_fast(Aq, Bq, (double)sA, NULL, sB, C, M, N, K);
#else
        rc = jt_w8a8_gemm_scalar_body(Aq, Bq, (double)sA, NULL, sB, C, M,
                                      N, K);
#endif
        goto cleanup;
    }
cleanup:
    return rc;
}

int jt_w8a8_gemm_per_channel(const int8_t *restrict Aq,
                             const int8_t *restrict Bq, float sA,
                             const float *restrict sB_col,
                             float *restrict C, int M, int N, int K) {
    int rc = JT_ERR_INVAL;
    if (M == 0) {
        if (Aq == NULL || Bq == NULL || C == NULL) {
            errno = EINVAL;
            goto cleanup;
        }
        if (N <= 0 || K <= 0) {
            errno = EINVAL;
            goto cleanup;
        }
        rc = JT_OK;
        goto cleanup;
    }
    if (jt_w8a8_gemm_validate(Aq, Bq, sA, sB_col, C, M, N, K, 1) != JT_OK) {
        goto cleanup;
    }
    for (int m = 0; m < M; m++) {
        const int8_t *arow = Aq + (size_t)m * (size_t)K;
        for (int k = 0; k < K; k++) {
            int qv = (int)arow[k];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (int k = 0; k < K; k++) {
        const int8_t *brow = Bq + (size_t)k * (size_t)N;
        for (int n = 0; n < N; n++) {
            int qv = (int)brow[n];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
#if defined(__AVX2__) || \
    (defined(__AVX512F__) && defined(__AVX512VNNI__))
    // G3同一関数内最適化: 検証済みのまま高速路へ (bit同一)。
    rc = jt_w8a8_gemm_fast(Aq, Bq, (double)sA, sB_col, 1.0f, C, M, N, K);
#else
    rc = jt_w8a8_gemm_scalar_body(Aq, Bq, (double)sA, sB_col, 1.0f, C, M,
                                  N, K);
#endif
cleanup:
    return rc;
}

int jt_w8a8_fakequant_per_tensor(const float *restrict src, size_t n,
                                 float *restrict dst) {
    int rc = JT_ERR_INVAL;
    double maxabs = 0.0;
    double scale_d = 0.0;
    if (src == NULL || dst == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_w8a8_valid_dims_q(n)) {
        errno = EINVAL;
        goto cleanup;
    }
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)src[i])) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < n; i++) {
        double a = fabs((double)src[i]);
        if (a > maxabs) {
            maxabs = a;
        }
    }
    if (maxabs == 0.0) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = 0.0f;
        }
        rc = JT_OK;
        goto cleanup;
    }
    scale_d = maxabs / (double)JT_W8A8_QMAX;
    if (!isfinite(scale_d) || scale_d <= 0.0) {
        errno = ERANGE;
        goto cleanup;
    }
    for (size_t i = 0; i < n; i++) {
        double qd = round((double)src[i] / scale_d);
        double v = 0.0;
        float f = 0.0f;
        if (qd > (double)JT_W8A8_QMAX) {
            qd = (double)JT_W8A8_QMAX;
        } else if (qd < (double)JT_W8A8_QMIN) {
            qd = (double)JT_W8A8_QMIN;
        }
        v = qd * scale_d;
        f = (float)v;
        if (!isfinite((double)f)) {
            errno = ERANGE;
            goto cleanup;
        }
        dst[i] = f;
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_fakequant_per_channel(const float *restrict src, size_t rows,
                                  size_t cols, float *restrict dst) {
    int rc = JT_ERR_INVAL;
    if (src == NULL || dst == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows == 0 || cols == 0) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows > SIZE_MAX / cols) {
        errno = EINVAL;
        goto cleanup;
    }
    {
        size_t n = rows * cols;
        for (size_t i = 0; i < n; i++) {
            if (!isfinite((double)src[i])) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        const float *srow = src + r * cols;
        float *drow = dst + r * cols;
        double maxabs = 0.0;
        double scale_d = 0.0;
        for (size_t c = 0; c < cols; c++) {
            double a = fabs((double)srow[c]);
            if (a > maxabs) {
                maxabs = a;
            }
        }
        if (maxabs == 0.0) {
            for (size_t c = 0; c < cols; c++) {
                drow[c] = 0.0f;
            }
            continue;
        }
        scale_d = maxabs / (double)JT_W8A8_QMAX;
        if (!isfinite(scale_d) || scale_d <= 0.0) {
            errno = ERANGE;
            goto cleanup;
        }
        for (size_t c = 0; c < cols; c++) {
            double qd = round((double)srow[c] / scale_d);
            double v = 0.0;
            float f = 0.0f;
            if (qd > (double)JT_W8A8_QMAX) {
                qd = (double)JT_W8A8_QMAX;
            } else if (qd < (double)JT_W8A8_QMIN) {
                qd = (double)JT_W8A8_QMIN;
            }
            v = qd * scale_d;
            f = (float)v;
            if (!isfinite((double)f)) {
                errno = ERANGE;
                goto cleanup;
            }
            drow[c] = f;
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}
