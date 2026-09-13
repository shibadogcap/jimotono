#pragma once
#ifndef JIMOTONO_LUT_AWARE_H
#define JIMOTONO_LUT_AWARE_H
// JIMOTONO LUT-aware学習足場 (HGQ-LUT方式、Phase 2)。
// 準拠: AGENTS.MD §3.3 (学習時はLUT-Dense層を通常のテンソル演算で学習し、
//   推論時にLUTへコンパイル)、§2.2混合精度 (ビット幅はmixed_prec.hのpolicy)。
//
// 用語注記 (MINOR-1対応。knowledge/papers.md §1警告との整合):
// 本プロジェクトの「LUT-aware / HGQ-LUT方式」は、学習時にLUT-Dense層を通常の
// fp32テンソル演算で学習し推論時にLUTへコンパイルする学習足場を指す
// (AGENTS.MD §3.3)。T-MAC論文 (2407.00088) のCPU動的LUTとも、FPGA用HGQ
// (BN+Dense+Actを真理値表に静的展開) とも別物であり、T-MAC文脈ではHGQと
// 呼ばない (papers.md §1「学習→推論の流れ (注意: HGQ-LUTとは別物)」参照)。
// コード側 jt_lut_* のリネームはAPI影響が大きいため行わない。
//
// 設計:
// - forward (jt_lut_dense_fwd): fp32テンソル演算でLUT-Dense等価値を計算する
//   (Y = XW [+b])。同時に重みをper-block量子化→逆量子化して量子化誤差統計
//   (mse/max/mean)を記録するが、順伝播値自体はfp32 (HGQ-LUTの学習時定義)。
// - backward (jt_lut_ste_pass): straight-through estimator。量子化ノードを
//   素通しする (=上流勾配をそのまま下流へコピー)。実勾配式は将来の融合核用に
//   TODOとして残すが、足場としてコピー素通し+有限ガードを提供する。
// - export (jt_lut_export_desc): 推論時LUTコンパイル用のレイアウト記述子を
//   出す。テーブル+スケールを同ページ配置 (DESIGN 5.1-5「LUTとスケールを重みと
//   同じページに」) し、64B整列・4KiBページ内収納フラグを持つ。
//   実バイナリ出力 (jt_lut_export_binary) はTODOで JT_ERR_NOSUP。
//
// 規約: C11, restrict, errnoベース + goto cleanup (実装側)。
// fail-closed: 非有限入力は JT_ERR_INVAL。行列はrow-major。
// リトルエンディアン前提。SIMDなし。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// 量子化誤差統計 (W vs dequant(quant(W))、fp32域)。
typedef struct jt_lut_err {
    float mse;          // 平均二乗誤差 (>=0)
    float max_abs_err;  // 最大絶対誤差 (>=0)
    float mean_abs_err; // 平均絶対誤差 (>=0)
} jt_lut_err_t;

// LUT-Dense順伝播: Y[m][n] = X[m][k] W[k][n] (+ b[n]、b==NULLで省略)。
// bits ∈ {2,4,8}、block は重み列方向のper-block量子化粒度 (>0)。
// err != NULL時は量子化誤差統計を格納 (順伝播値には影響しない)。
// 全入力の有限性を先に検査し、NG時はY/errを更新しない (fail-closed)。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM。
int jt_lut_dense_fwd(const float *restrict X, const float *restrict W,
                     const float *restrict b, float *restrict Y, size_t m,
                     size_t n, size_t k, int bits, size_t block,
                     jt_lut_err_t *restrict err);

// STE素通し: downstream[i] = upstream[i] (勾配の量子化ノード素通し)。
// 有限ガード付きコピー。別名 (重なり) 禁止。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_lut_ste_pass(const float *restrict upstream, float *restrict downstream,
                    size_t count);

// 推論LUTレイアウト記述子 (同ページ配置)。
#define JT_LUT_EXPORT_MAGIC 0x4A545554u // "JTUT" (LE)
#define JT_LUT_PAGE_SIZE 4096u
#define JT_LUT_ALIGN 64u

typedef struct jt_lut_export_desc {
    uint32_t magic;        // JT_LUT_EXPORT_MAGIC
    int32_t bits;          // {2,4,8}
    uint32_t rows;         // 重み行数 (k側)
    uint32_t cols;         // 重み列数 (n側)
    uint32_t block_len;    // per-block粒度
    uint64_t table_bytes;  // パック重み ceil(rows*cols*bits/8)
    uint64_t scale_bytes;  // rows方向? cols方向block毎 fp32: rows*ceil(cols/block)*4
    uint64_t table_offset; // 0
    uint64_t scale_offset; // align_up(table_bytes, 64)
    uint64_t page_bytes;   // align_up(scale_offset+scale_bytes, 64)
    int32_t same_page;     // page_bytes <= 4096 ? 1 : 0 (同ページ収納可否)
} jt_lut_export_desc_t;

// 記述子生成 (純粋関数)。実バイナリは出さない。
// rows/cols==0、bits∉{2,4,8}、block==0はINVAL。積オーバーフローはNOMEM。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM。
int jt_lut_export_desc(uint32_t rows, uint32_t cols, int bits,
                       uint32_t block, jt_lut_export_desc_t *restrict desc);

// 実バイナリ出力 (TODO): 現状は JT_ERR_NOSUP (errno=ENOSYS)。
// シグネチャ予約 (テーブル+スケールを同ページ配置で書き出す将来位置)。
int jt_lut_export_binary(const float *restrict W, uint32_t rows,
                         uint32_t cols, int bits, uint32_t block,
                         void *restrict out, size_t out_cap,
                         size_t *restrict out_written);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_LUT_AWARE_H
