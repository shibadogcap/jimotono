#pragma once
#ifndef JIMOTONO_ESMOE_H
#define JIMOTONO_ESMOE_H
// ES-MoE SSDオフロード足場 (Phase 2)。
// AGENTS.MD 4.1: エキスパートパラメータをCPU/SSDにオフロード。
// 準拠: https://ina.kaist.ac.kr/projects/esmoe/ (概念のみ。自前実装)。
//
// 方針 (DESIGN.MD §3/§5):
// - エキスパート重みはSSD上の単一backingファイルにオフロードし、
//   jt_io_pread_batch 経由で必要エキスパート (将来は行単位) のみ読み込む。
// - 現状は buffered pread で機能正しさを優先。O_DIRECT/io_uring本格対応は
//   TODO (下記)。HOT/COLD使い分けは io_batch.h方針に従う。
// - 共起クラスタリング・ホット/コールド分離 (DESIGN.MD §5.1-1,4) は
//   コメント設計のみ (本足場では実装しない)。
//
// [DESIGN §5.1 将来配置 (コメント設計)]
//   1. 共起クラスタリング: 同時発火するexpertをディスク上で隣接配置し、
//      runs圧縮 (jt_io_runs_compress) で順次run化して読む。配置メタは
//      エクスポート時に焼く (本足場はexpert_id順の線形配置のみ)。
//   2. ホット/コールド分離: 頻出expertはファイル先頭 + DRAM pin +
//      buffered I/O (O_DIRECT禁止)。コールドはO_DIRECT streaming。
//      本足場は全expertを一様backing + buffered preadとして扱う。
//   3. プリトランスポーズ/プリブロック・gate/up/downインターリーブ・
//      LUT/スケール同ページ化はエクスポート形式の仕事。本APIは
//      expert_bytes/row_bytesの不透明バイト列として運ぶ。
//
// 規約: C11, restrict, errnoベース + goto cleanup (実装側)。
//       malloc/free。OOM時は JT_ERR_NOMEM。
// 禁止: llama.cpp等リンクなし (AGENTS.MD 7.2)。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct jt_esmoe_cfg {
    uint32_t n_experts;  // >0
    size_t expert_bytes;  // >0, 1 expertあたりバイト数
    size_t row_bytes;  // 0=expert単位 (既定)。>0で行単位読みを許可
} jt_esmoe_cfg_t;

typedef struct jt_esmoe_stats {
    uint64_t hits;        // キャッシュヒット (load/prefetch内)
    uint64_t misses;      // キャッシュミス (SSD読み発生)
    uint64_t loads;       // SSD読み回数 (expert単位換算: prefetch/load/read_rows合算)
    uint64_t evicts;      // evict実行回数 (実解放のみ計上)
    uint64_t prefetches;  // prefetch呼び出し回数
    uint64_t bytes_read;  // SSDから読んだ総バイト数
} jt_esmoe_stats_t;

typedef struct jt_esmoe {
    void *file;  // backing FILE* (不透明。POSIX/Windows差を.c内に隔離)
    int fd;  // pread_batch用fd (fileno/_fileno)
    uint32_t n_experts;
    size_t expert_bytes;
    size_t row_bytes;  // 0ならexpert_bytesと同値扱い
    unsigned char **slots;  // [n_experts] DRAMキャッシュ (NULL=非保持)
    unsigned char *present;  // [n_experts] 0/1
    unsigned char *registered;  // [n_experts] 0/1 (register済みのみ読み可)
    jt_esmoe_stats_t stats;
} jt_esmoe_t;

// 初期化 (backingはtmpfile)。cfg==NULLはINVAL。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM / JT_ERR_IO (tmpfile失敗)
int jt_esmoe_init(jt_esmoe_t *restrict es, const jt_esmoe_cfg_t *restrict cfg);

// 解放 (二重fini安全。NULL安全)。
void jt_esmoe_fini(jt_esmoe_t *restrict es);

// 登録: expert_idの重みlenバイトをbackingに書き込む (len==expert_bytes必須)。
// 上書き可。書き込み後はキャッシュを更新せず (次回prefetch/loadで読む)。
// TODO(Phase 2): SmartUpdate対応でSSD上直接更新パスを追加する。
// 戻り値: JT_OK / JT_ERR_INVAL (未初期化・範囲外・len不一致・NULL)
int jt_esmoe_register(jt_esmoe_t *restrict es, uint32_t expert_id,
                      const void *restrict data, size_t len);

// evict: DRAMキャッシュを破棄 (backingは保持)。未保持なら何もしない。
// 戻り値: JT_OK / JT_ERR_INVAL (未初期化・範囲外)
int jt_esmoe_evict(jt_esmoe_t *restrict es, uint32_t expert_id);

// prefetch: ids[0..n)の未保持expertを jt_io_pread_batch 経由で一括読みし
// DRAMに保持する。保持済みはhits計上のみ。n==0はJT_OK (何もしない)。
// TODO(Phase 2, O_DIRECT/io_uring): コールド行は jt_io_pread_batch_uring +
//   O_DIRECT streamingに切替える。4096整列バッファ・offset/length倍数化が
//   前提 (io_batch.h Phase 2注意)。HOT expertはbuffered + pinのまま残す。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM / JT_ERR_IO (read失敗)
int jt_esmoe_prefetch(jt_esmoe_t *restrict es, const uint32_t *restrict ids,
                      size_t n);

// load: expert1件をdst[len)に読み出す (len==expert_bytes必須)。
// キャッシュ保持済みならmemcpy (hits)。未保持ならSSDから読みキャッシュ化
// (misses)。未登録expertはINVAL。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_NOMEM / JT_ERR_IO
int jt_esmoe_load(jt_esmoe_t *restrict es, uint32_t expert_id,
                  void *restrict dst, size_t len);

// read_rows: expert内の行範囲をSSDから直接読む (キャッシュバイパス)。
// offset = expert_id*expert_bytes + row_start*row_bytes,
// len = row_count*row_bytes。row_bytes==0設定時はINVAL (expert単位のみ)。
// 将来のDelta Prefetching (活性15〜18%行のみ読み) の読出し口。
// TODO(Phase 2): ids→runs圧縮 (jt_io_runs_compress) と結合し複数行runを
//   まとめて読む。NeuroPrefetcher式の投機的先読み和集合 (S∪L∪M) は呼出側。
// 戻り値: JT_OK / JT_ERR_INVAL / JT_ERR_IO
int jt_esmoe_read_rows(jt_esmoe_t *restrict es, uint32_t expert_id,
                       size_t row_start, size_t row_count,
                       void *restrict dst);

// stats取得 (out==NULLはINVAL)。
int jt_esmoe_stats(const jt_esmoe_t *restrict es,
                   jt_esmoe_stats_t *restrict out);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_ESMOE_H
