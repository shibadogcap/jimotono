#pragma once
#ifndef JIMOTONO_IO_BATCH_H
#define JIMOTONO_IO_BATCH_H
// jt_io_batch: SSD streaming batch-read abstraction (papers.md §4).
//
// NeuroPrefetcher Delta path: neuron-major単一ファイル上の活性行集合を
// find_runs_gap1相当で (offset,length) に結合し、ランダムI/Oを削減する。
// 本ヘッダはC11層のspec配列I/Fに統一し、OS差を吸収する。
//
// ---- ホット / コールド使い分け方針 ----
//   HOT  (共有expert / attention射影 / 頻出routed expert):
//        LRU + pin + buffered I/O (ページキャッシュを使う)。O_DIRECT禁止。
//        理由: 毎トークン発火し再利用率が高く、キャッシュヒットが支配的。
//   COLD (delta incoming行 / 低頻度routed expert):
//        O_DIRECT streaming (ページキャッシュをバイパス)。Phase 2で有効化。
//        理由: 使い捨てに近くキャッシュ汚染だけが残る。82〜85%持続の残り
//        15〜18%のみを読むため、短い順次runの束になる。
//   Phase 1 (本実装): どちらも buffered pread ループで処理する。
//        O_DIRECT / io_uring は使わない。性能較正用の起動時 iobench は Phase 2。
//
// ---- OS fallback注意 ----
//   Linux:   Phase 2で io_uring + O_DIRECT (下記TODO)。Phase 1は pread ループ。
//   macOS:   O_DIRECT がない。代替は fcntl(fd, F_NOCACHE, 1)。
//            preadv は利用可だが per-spec offset がないため、本I/Fのspec配列を
//            そのまま渡せるpreadループを正とする。F_NOCACHE適用は呼び出し側判断。
//   Windows: FILE_FLAG_NO_BUFFERING が代替だが、4096整列バッファ + セクタ倍数
//            長が必須で制約が強い。Phase 1は buffered fallback (_lseeki64+_read)。
//            OVERLAPPED 非同期化は Phase 2 TODO。
//
// ---- アライメント ----
//   Phase 1 (buffered): アライメント要求なし。任意のdst/lengthでよい。
//   TODO(Phase 2, O_DIRECT時): dstは4096整列 (posix_memalign/_aligned_malloc)、
//   offset/lengthは4096倍数に切上げパディングすること。未対応のままO_DIRECTを
//   有効化しないこと。
//
// ---- io_uring TODO (Linuxのみ, Phase 2, 別関数に分離) ----
//   TODO(uring): int jt_io_pread_batch_uring(int fd, specs, n) を新設する。
//   liburing sliding-window: batch=256、ring sizeは2の冪、
//   prep_read → submit → wait_cqe + peekでdrain → resubmit の窓進行。
//   dense prefix計算とI/Oをオーバーラップさせる。GIL相当の排他なし (C層)。
//   本ファイルの jt_io_pread_batch は同期preadラッパーのまま残す (fallback用)。
//
// 規約: C11、restrict積極使用、errnoベース + goto cleanup (呼び出し側含む)。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

// preadv互換バッチ読みの1要求。offsetはファイル先頭からのバイト位置。
// NOTE: struct iovec {void*iov_base; size_t iov_len;} に対し offset を付加した
// 形であり、macOS preadv 経路へは (iov_base=dst, iov_len=length) + offset に
// 分解して渡せる。Phase 1では分解せず pread ループで処理する。
typedef struct jt_io_spec {
    uint64_t offset;  // byte offset from file start
    size_t length;    // bytes to read (0はno-opとしてskip)
    void *dst;        // destination (length>0なら非NULL)
} jt_io_spec_t;

// runs圧縮の1 run。offset/lengthともバイト単位。
typedef struct jt_io_run {
    uint64_t offset;  // byte offset from file start (= base + start_id*record_bytes)
    size_t length;    // bytes (= run_len*record_bytes)
} jt_io_run_t;

// fd上のspecs[0..n)を順にpreadする (Phase 1: 同期ループ、short-readリトライ)。
//   n==0: 何もせず JT_OK (fd/specsの検証なし、specsはNULL可)。
//   戻り値: JT_OK / JT_ERR_INVAL (引数不正, errno=EINVAL)
//            JT_ERR_IO (read失敗・EOF前打ち切り。errnoはpread由来、EOF時はEIO)
//   NOTE: ファイルオフセットを変更しない (POSIX pread)。
//   Windows Phase 1実装は _lseeki64+_read のためオフセットが動く (単一スレッド
//   1C1T前提)。OVERLAPPED化は Phase 2 TODO。
int jt_io_pread_batch(int fd, const jt_io_spec_t *restrict specs, size_t n);

// ソート済みID列を連続runに結合する (find_runs_gap1相当)。
//   ids[0..n): 昇順ソート済み行/expert ID。重複は無視 (同一runに畳む)。
//   record_bytes: 1行あたりバイト数 (>0)。1ならID単位の (offset,length)。
//   base_offset: ファイル先頭からの基準バイト位置 (neuron-major先頭等)。
//   out[0..out_cap): 結果書き込み先。out_n: run数を返す。
//   例: ids={0,1,2,5,6}, record_bytes=1, base=0 → out={(0,3),(5,2)}.
//   戻り値: JT_OK (成功。n==0なら*out_n=0。ids/outはNULL可)
//            JT_ERR_INVAL (未ソート・record_bytes==0・NULL引数・オーバーフロー)
//            JT_ERR_NOMEM (out_cap不足。*out_nに必要数を設定、errno=ENOMEM)
//   NOTE: 内部でソートしない。未ソート入力はエラーにして呼び出し側で直す。
int jt_io_runs_compress(const uint32_t *restrict ids, size_t n,
                        uint64_t record_bytes, uint64_t base_offset,
                        jt_io_run_t *restrict out, size_t out_cap,
                        size_t *restrict out_n);

#endif  // JIMOTONO_IO_BATCH_H
