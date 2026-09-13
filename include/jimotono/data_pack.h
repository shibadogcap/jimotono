#pragma once
#ifndef JIMOTONO_DATA_PACK_H
#define JIMOTONO_DATA_PACK_H
// jt_data_pack: トークンID列の素朴バイナリ形式 writer/reader (Phase 2 data-pipe)。
//
// 背景: HDF5は本環境に存在しない (pkg-config/brew/h5ccいずれも不在を確認)。
// 依存追加はせず、独自の素朴バイナリ + mmap読み取りで実装する。
// JSONL直接読みは禁止 (16ワーカーで破綻するため)。学習時は本形式をmmapし、
// ランダムアクセスでバッチ供給する。
//
// ---- ファイル配置 (v1, 全整数リトルエンディアン。AGENTS.MDどおりLE前提) ----
//   [header 64B]
//     magic[4]   = 'J','T','D','P'
//     version u32    = 1
//     header_len u32 = 64
//     flags u32      = 0 (bit予約。将来のuint16パックはTODO、下記参照)
//     nseq u64       = シーケンス数
//     total u64      = 全トークン数 (u32単位)
//     off_table_off u64 = オフセット表のバイト位置 (= 64 + total*4)
//     checksum u32   = FNV-1a(32bit)。payloadバイト + オフセット表バイトの順に更新
//     reserved[12]   = 0
//   [payload: total * u32 トークンID]
//   [offset table: (nseq+1) * u64 累積トークン数。off[0]=0, off[nseq]=total]
//   ファイル長は off_table_off + (nseq+1)*8 に完全一致すること (末尾ゴミ拒否)。
//
// ---- 設計判断 ----
//   * uint32格納 (語彙48,588のため値域はu32)。将来のuint16パックはTODO:
//     flags bit0 = "payloadがuint16" を予約。reader v1はflags!=0を拒否する。
//   * オフセット表付き可変長 (固定長レコードではない)。空シーケンス可
//     (off[i+1]==off[i])。
//   * readerはmmap対応 (Linux/macOS)。mmap失敗時はmalloc+readにfallback。
//     WindowsはTODOスタブ (jt_dp_openがJT_ERR_NOSUP)。
//   * fail-closed: マジック・版数・header_len・境界・単調性・チェックサム・
//     語彙値域 (< 48588) のいずれか不正でopen失敗。部分書き込みファイルは
//     checksum不一致で必ず拒否される。
//
// 規約: C11、restrict積極使用、errnoベース + goto cleanup (AGENTS.MD 7.1)。
// 依存: libcのみ (HDF5不要)。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define JT_DP_MAGIC0 'J'
#define JT_DP_MAGIC1 'T'
#define JT_DP_MAGIC2 'D'
#define JT_DP_MAGIC3 'P'
#define JT_DP_VERSION 1u
#define JT_DP_HEADER_LEN 64u
#define JT_DP_VOCAB_SIZE 48588u  // llm-jp-tokenizer v2.2 (AGENTS.MD 2.1)
// TODO(uint16-pack): 将来payloadをuint16で詰める場合はflags bit0を立て、
// reader/writer双方を拡張すること。v1 readerはflags!=0をJT_ERR_INVALで拒否する。

// ---- 一括書き込み ----
//   pathに新規作成 (既存ファイルは切詰め)。seqs[i]はlens[i]長のu32列。
//   nseq==0可 (seqs/lensはNULL可)。空シーケンス (lens[i]==0) 可。
//   全IDは JT_DP_VOCAB_SIZE 未満であること (以上はJT_ERR_INVAL)。
//   戻り値: JT_OK / JT_ERR_INVAL (引数不正・値域外、errno=EINVAL)
//            JT_ERR_IO (fopen/fwrite/fseek失敗、errnoはstdio由来)
//            JT_ERR_NOMEM (確保失敗、errno=ENOMEM)
int jt_dp_write_file(const char *restrict path,
                     const uint32_t *const *restrict seqs,
                     const size_t *restrict lens, size_t nseq);

// ---- ストリーミング書き込み (pack_jsonl等の変換ツール用) ----
//   open → add* → closeの順に呼ぶ。open前にゼロ初期化不要 (openが全設定)。
//   closeはオフセット表追記 + ヘッダ backpatch + fclose を行う。
//   エラー時は部分ファイルが残りうるが、readerはchecksumで必ず拒否する
//   (呼び出し側でremoveしてよい)。
typedef struct jt_dp_writer {
    void *opaque;  // 内部状態 (呼び出し側は触らない)
} jt_dp_writer_t;

int jt_dp_writer_open(jt_dp_writer_t *restrict w, const char *restrict path);
int jt_dp_writer_add(jt_dp_writer_t *restrict w,
                     const uint32_t *restrict ids, size_t len);
int jt_dp_writer_close(jt_dp_writer_t *restrict w);

// ---- mmap reader (Linux/macOS。WindowsはTODOスタブでJT_ERR_NOSUP) ----
typedef struct jt_dp_reader {
    void *opaque;  // 内部状態 (呼び出し側は触らない)
} jt_dp_reader_t;

//   pathを開き、全検証 (マジック・版数・境界・単調性・checksum・語彙値域) を行う。
//   戻り値: JT_OK / JT_ERR_INVAL (検証失敗、errno=EINVAL)
//            JT_ERR_IO (open/stat/read失敗・空ファイル等、errnoはOS由来)
//            JT_ERR_NOMEM (確保失敗)
//            JT_ERR_NOSUP (Windows未対応、errno=ENOSYS)
int jt_dp_open(const char *restrict path, jt_dp_reader_t *restrict out);
// NULL可 (未openへのcloseは無害)。
void jt_dp_close(jt_dp_reader_t *restrict r);

// シーケンス数・長さ・ゼロコピー参照・コピー取得。
// idx範囲外はJT_ERR_INVAL。seq_copyはcap不足でJT_ERR_NOMEM (*out_lenに必要数)。
int jt_dp_nseq(const jt_dp_reader_t *restrict r, uint64_t *restrict out_n);
int jt_dp_seq_len(const jt_dp_reader_t *restrict r, uint64_t idx,
                  uint64_t *restrict out_len);
int jt_dp_seq_ptr(const jt_dp_reader_t *restrict r, uint64_t idx,
                  const uint32_t **restrict out_ptr,
                  uint64_t *restrict out_len);
int jt_dp_seq_copy(const jt_dp_reader_t *restrict r, uint64_t idx,
                   uint32_t *restrict dst, uint64_t cap,
                   uint64_t *restrict out_len);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_DATA_PACK_H
