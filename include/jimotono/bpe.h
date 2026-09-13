#pragma once
#ifndef JIMOTONO_BPE_H
#define JIMOTONO_BPE_H
// jt_bpe: llm-jp-tokenizer v2.2 (48,588語彙) 対応の最小トークナイザ (Phase F5)。
//
// 背景: v2.2 は SentencePiece Unigram (byte-fallback) であり、BPEのマージ規則を
// 持たない。よって本実装は「マージ規則のトライ木」ではなく「スコア付きピースの
// バイトトライ木 + Viterbi 最良経路」で等価な符号化を行う (API名の bpe は
// Phase指定の呼称を維持)。参照: scripts/bpe_make_vocab.py の版記録。
//
// 最小 subset (llm-jp仕様との差異 D1〜D3):
//   D1. 正規化は identity のみ。入力U+0020をU+2581(▁)へ置換し、非空入力には
//       無条件でダミー接頭辞▁を付与する (add_dummy_prefix=true と同値。実測確認)。
//       NFKC 等は行わない (v2.2 normalizer自体が identity のため差異なし)。
//   D2. split_by_unicode_script / split_by_number / split_digits のハード分割を
//       行わず、全域Viterbiで解く。v2.2語彙には境界横断ピース
//       (数字+文字混在・ASCII+CJK混在・複数数字) が0件のため、加算スコアの
//       Viterbiでは分割の有無が最適経路に影響せず、参照実装と一致する (検証済)。
//       将来語彙が横断ピースを含む場合は要再訪。
//   D3. Viterbiは対数確率の最大和。タイは先勝ち (strict > のみ更新)。
//       参照実装との等価性は bench_bpe の golden で検証する。
//
// 語彙ファイル (.jtvocab v1, LE。mmap読み):
//   header 32B: magic 'J','T','V','B', version u32=1, n u32,
//               max_id u32, reserved[5] u32=0
//   entries × n: u32 id, float score, u32 flags (bit0=byteピース),
//                u32 len, u8 bytes[len] (パディングなし)
//   生成は scripts/bpe_make_vocab.py (通常+バイトの全ピースを格納)。
//
// 規約: C11、restrict積極使用、errnoベース + goto cleanup (AGENTS.MD 7.1)。
// fail-closed: 不正引数・不正UTF-8・語彙外ID・不足byteピースは符号化せず
// エラー返却。不明文字の黙殺置換 (代替文字挿入など) は行わない。
// スレッド安全性: jt_bpe_t の open/close と encode/decode の同時呼び出しは
// 非対応 (1C1T前提)。encode/decode 自体は都度確保のため再入可能。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define JT_BPE_MAGIC0 'J'
#define JT_BPE_MAGIC1 'T'
#define JT_BPE_MAGIC2 'V'
#define JT_BPE_MAGIC3 'B'
#define JT_BPE_VERSION 1u
#define JT_BPE_HEADER_LEN 32u
#define JT_BPE_MAX_ID 48588u  // jt_data_pack の JT_DP_VOCAB_SIZE と一致 (上界検査用)
#define JT_BPE_FLAG_BYTE 1u   // byteフォールバックピース (<0xNN>相当)

// 不透明ハンドル (mmap/確保領域 + トライ + 復号表を保持)。
typedef struct jt_bpe {
    void *opaque;
} jt_bpe_t;

// vocab_path (.jtvocab v1) を開き、トライ木と復号表を構築する。
// 戻り値: JT_OK / JT_ERR_INVAL (引数・形式不正、errno=EINVAL)
//          JT_ERR_IO (open/read失敗、errnoはOS由来)
//          JT_ERR_NOMEM (確保失敗、errno=ENOMEM)
int jt_bpe_open(jt_bpe_t *restrict b, const char *restrict vocab_path);

// 資源解放 (未open/NULLは無害)。
void jt_bpe_close(jt_bpe_t *restrict b);

// 語彙サイズ (格納ピース数) と最大ID+1。未open時は0。
uint32_t jt_bpe_vocab_size(const jt_bpe_t *restrict b);
uint32_t jt_bpe_vocab_max(const jt_bpe_t *restrict b);

// テキスト (UTF-8, lenバイト) → トークンID列。
//   空入力は成功・0トークン (*out_n=0)。
//   不正UTF-8は JT_ERR_INVAL (errno=EINVAL)。
//   cap不足は JT_ERR_NOMEM (*out_nに必要数、errno=ENOMEM)。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM
int jt_bpe_encode(const jt_bpe_t *restrict b, const char *restrict text,
                  size_t len, uint32_t *restrict out, size_t cap,
                  size_t *restrict out_n);

// トークンID列 → バイト列 (UTF-8とは限らない。byteフォールバック由来の
// 任意バイトを含みうる。呼び出し側で必要に応じ検証すること)。
// 復号は ▁→U+0020 置換 + 先頭1空白除去 (ダミー接頭辞の逆操作)。
//   語彙外IDは JT_ERR_INVAL (errno=EINVAL)。
//   cap不足は JT_ERR_NOMEM (*out_nに必要数、errno=ENOMEM)。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM
int jt_bpe_decode(const jt_bpe_t *restrict b, const uint32_t *restrict ids,
                  size_t n, char *restrict out, size_t cap,
                  size_t *restrict out_n);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_BPE_H
