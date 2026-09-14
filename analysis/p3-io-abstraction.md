# P3 I/O抽象化設計（文書のみ・実装禁止）

- worktree: `/Users/shibadogcap/jimotono-wt-ioabs` / branch: `docs/p3/io-abstraction`
- 性質：設計文書のみ。実装（`.c`/`.h`/CMake変更）禁止、ビルド・実行禁止（Stage 3と並行のため読解のみ）。
- 前提：`include/jimotono/io_batch.h`、`include/jimotono/io_direct.h`、`src/io_batch.c`、`src/io_batch_uring.c`、`src/io_direct.c`、`include/jimotono/common.h`、`include/jimotono/data_pack.h`、`DESIGN.MD §3`、`RULE.MD`。
- 規約：C11・`restrict`・errnoベース＋`goto cleanup`・同一ソース3OSビルド可能（AGENTS.md §7.1）。`JT_ERR_*`（`common.h`：OK/NOMEM/INVAL/IO/ALIGN/NOSUP）は番号変更禁止。

## 1. 共通IF：`jt_io_submit` / `poll` / `wait` 設計

### 1.1 目的

既存3経路（同期batch／uring batch／direct sync）を束ね、呼び出し側（将来のNeuroPrefetcher経路・decode先読み）からOS差を見えなくする。**新規スレッドを作らない**（1C1T前提。RULE.md 並列規則とおり重負荷は親が逐次実行）。

### 1.2 既存との関係（層構造）

```text
decode先読み (S∪L∪M union → runs_compress → submit)
  └─ jt_io (new: submit/poll/wait, backend選択のみ)   ← 本設計の追加範囲（将来）
       ├─ HOT: jt_io_pread_batch (buffered同期, fallback正経路) — 変更なし
       ├─ COLD-sync: jt_io_direct_{open,alloc,pread,close} (Linux実経路/他OSスタブ) — 変更なし
       └─ COLD-async: jt_io_pread_batch_uring (Linux+liburing実経路/他NOSUP) — 変更なし
```

- 新層は**薄いディスパッチ＋状態保持**に限定する。I/O意味論（short-readリトライ、EOF→EIO、`len==0` no-op、`n==0` no-op、未ソートINVAL等）は既存のまま継承し、再定義しない。
- `jt_io_runs_compress`（find_runs_gap1相当）は新層の**前段**のまま。ソートは呼び出し側責任（内部ソート禁止）は維持。
- `io_batch.h`のHOT/COLD使い分けを新層の選択規則として昇格させる（§1.4）。

### 1.3 提案IFスケッチ（将来実装用・本wtでは実装しない）

```c
typedef enum jt_io_kind { JT_IO_HOT = 0, JT_IO_COLD = 1 } jt_io_kind_t;
typedef struct jt_io_req {
    uint64_t offset;   /* jt_io_spec_tと同義 */
    size_t   length;   /* 0はno-opとしてskip */
    void    *dst;
} jt_io_req_t;
typedef struct jt_io_token { uint64_t id; } jt_io_token_t; /* poll/wait用 */

/* 非同期投入。戻り値は投入受付のみ表す（完了はpoll/waitで知る）。 */
int jt_io_submit(int fd, jt_io_kind_t kind,
                 const jt_io_req_t *restrict reqs, size_t n,
                 jt_io_token_t *restrict out_tok);
/* 非ブロッキング回収：doneに完了件数。未完了はJT_OK+*done==0。 */
int jt_io_poll(jt_io_token_t tok, size_t *restrict done);
/* ブロッキング待機：全件完了まで待つ（EINTRは内部リトライ）。 */
int jt_io_wait(jt_io_token_t tok);
```

セマンティクス決定事項：

| 項目 | 決定 |
|---|---|
| `fd`種別 | HOTはbuffered fd、COLDは`jt_io_direct_open`のfd（buffered fallback fdも可）をそのまま渡す。新層はopenしない |
| バッファ要件 | HOT：整列要求なし。COLD：`jt_io_direct_is_aligned(dst,off,len,512)`を**投入前に検査**し違反は`JT_ERR_INVAL+EINVAL`（fail-closed、I/O発行なし） |
| エラー | `JT_OK/INVAL/IO/NOMEM/NOSUP`を既存どおり。`errno`はOS由来を上書きしない。投入後エラーは`poll/wait`で`JT_ERR_IO`として返す（どのreqかは呼び出し側ログ用に別途out indexを返す設計余地あり・P3本体判断） |
| キャンセル | なし（エラー時再投入しない。uring実装の現行「エラー時取り消しなし」と一致） |
| スレッド安全性 | 同一fdへの並行`submit`は呼び出し側で直列化（現行`io_batch_uring.c`注記を継承） |
| 窓幅 | `batch=256`・ring depth 2の冪（現行`JT_URING_BATCH/DEPTH=256`）をLinux既定として温存。他OS backendのキュー深さは各OS既定（§2） |

### 1.4 Backend選択規則（HOT/COLD × OS能力）

- HOT（共有expert／attention射影／頻出routed expert）：常にbuffered同期（`jt_io_pread_batch`）。**O_DIRECT禁止**（`io_batch.h`方針維持）。新層は`kind=HOT`を同期実行＋即時完了tokenとして返す。
- COLD（delta incoming行／低頻度routed expert）：OS能力で選択。
  - Linux＋liburing有効：`jt_io_pread_batch_uring`（buffered fd）または将来のdirect＋uring組合せ（P3本体判断。本設計ではdirect＋uringの結合を**許可せず分離維持**――現行「O_DIRECT有効化禁止」注意と矛盾させない）。
  - Linux＋liburing無効：`jt_io_pread_batch`同期fallback（`JT_ERR_NOSUP+ENOSYS`時は呼び出し側がfallback）。
  - macOS／Windows：当面すべて同期fallback。新層は`JT_ERR_NOSUP`を返さず、**同期実行して完了扱い**にするか、`submit`自体を同期パスに直結させる（P3本体で二択。推奨は後者＝呼び出し側分岐を増やさない）。

## 2. OS別backend設計

### 2.1 Linux：`O_DIRECT` ＋ `io_uring`（唯一の実経路）

- open：`open(path, O_RDONLY|O_DIRECT)`、失敗`e ∈ {EINVAL, EOPNOTSUPP, ENOTSUP, ENOSYS}`時のみbuffered `open(path, O_RDONLY)`へfallback（現行`io_direct.c`どおり）。`ENOENT`等はfallbackせず`JT_ERR_IO`。
- alloc：`posix_memalign(align ∈ {512以上の2の冪})`。推奨4096（将来のWindows互換のため）。
- pread：512整列（dst/off/len）を事前検査→fail-closed。short-readリトライ、EOF前打ち切りは`EIO`。
- uring：buffered fd前提、sliding-window batch=256、`prep_read→submit→wait_cqe＋peek drain→resubmit`。`HAVE_LIBURING`なしは`JT_ERR_NOSUP+ENOSYS`スタブ（現行どおり）。`JIMOTONO_USE_URING=OFF`既定を維持。
- 制約・fallback表：

| 条件 | 動作 |
|---|---|
| tmpfs等O_DIRECT非対応FS | open時fallback（buffered fd）。`pread`側で再fallbackしない（同一fdでmode変更不可） |
| 整列違反 | `JT_ERR_INVAL+EINVAL`、I/O未発行 |
| 巨大spec（`UINT_MAX`超） | chunk分割（uring側実装済み）／preadループ（同期側） |
| offset `> 0x7FFFFFFFFFFFFFFF` | `JT_ERR_INVAL`（現行どおり） |
| ring init失敗 | `JT_ERR_IO`（errnoは`-qret`） |

### 2.2 macOS：`F_NOCACHE` ＋ `kqueue`（非同期はreadinessのみ）

- open/read本体：`pread`ループ（現行`io_batch.c` POSIX経路が正）。`O_DIRECT`は存在しないため使わない。
- `F_NOCACHE`：`fcntl(fd, F_NOCACHE, 1)`を**呼び出し側判断**で適用（`io_batch`内部では触らない＝現行注記維持）。fd単位の助言であり、COLD fdにのみ適用しHOT fdには適用しない。
- `kqueue`：`EVFILT_READ`は**readiness通知**であり完了通知ではない。`io_uring`のようなsubmission/completionキューにはならない。したがって非同期化の選択肢は (a) 同期`pread`のまま窓進行を呼び出し側で行う、(b) `aio_read`（POSIX AIO）＋kqueue合流、(c) 専用I/Oスレッド（1C1T原則に反するため非推奨）。P3本体判断とするが、既定は(a)同期＋先読み重ね（§5）でSSD待ちを隠す。
- `preadv`は利用可だがper-spec offsetがないため本I/Fのspec配列は分解が必要。現行方針どおり`pread`ループを正とし、`preadv`最適化はP3本体判断（実装しない）。
- 制約・fallback表：

| 条件 | 動作 |
|---|---|
| 整列 | 要求なし（`F_NOCACHE`は整列不要）。4096整列バッファはそのまま使えるが必須化しない |
| 非対応FS（nfs/smb/tmp的構成） | `F_NOCACHE`が無視されてもエラーにしない（bufferedとして動作）。`fcntl`失敗は`JT_ERR_IO`にせず無視orログに留める（P3本体で決定。本設計の推奨は無視＋HOT扱い継続） |
| kqueue未使用時 | 機能縮退なし（同期正経路が残る） |

### 2.3 Windows：`FILE_FLAG_NO_BUFFERING` ＋ `IOCP`（制約最強・当面スタブ）

- Phase 1正経路：`_lseeki64＋_read` buffered fallback（現行`io_batch.c`どおり）。単一スレッド1C1T前提（ファイルオフセットを動かすためFD共有不可）。
- `FILE_FLAG_NO_BUFFERING`実経路（将来・本wt実装なし）の制約：
  - バッファは**セクタ（通常4096、取得は`GetDiskFreeSpace`）整列**（`_aligned_malloc`）必須。
  - `length`・`offset`ともセクタ倍数必須。末端の非倍数tailは**切上げパディング読み＋呼び出し側切詰め**が必要（パディング設計はP3本体）。
  - ボリュームによりセクタサイズが異なる（512e／4Kネイティブ）。起動時に取得し、`jt_io_direct_is_aligned`相当の検査alignを可変化する。
- `IOCP`実経路（将来）：`OVERLAPPED`＋`GetQueuedCompletionStatus`。`io_uring`のCQE対応物。`_read`ループとは別関数に分離すること（現行TODOどおり）。
- 制約・fallback表：

| 条件 | 動作 |
|---|---|
| 整列/倍数違反 | fail-closed（`JT_ERR_INVAL+EINVAL`、I/O未発行）。WindowsではLinuxより違反コストが高い（ERROR_INVALID_PARAMETER）ため事前検査必須 |
| 非対応FS（FAT32上のNO_BUFFERING等） | buffered fallback（`CreateFile` without flag再open）。Linuxのopen-fallbackと同型 |
| `_read`の`UINT_MAX`幅 | 現行どおりchunk分割 |
| OVERLAPPED未実装時 | 同期fallback（現行）。新層は完了扱いで返す |

## 3. macOS `F_NOCACHE`の挙動差異（RULE記録用）

**要旨（RULE転記用）：** macOSの`fcntl(fd, F_NOCACHE, 1)`はLinuxの`O_DIRECT`ほど強力でない。`O_DIRECT`がページキャッシュ迂回をI/O単位で強制するのに対し、`F_NOCACHE`はfd単位の「キャッシュしない助言」に留まり、(i)他fd経由の同一ファイルキャッシュ残存を防げない、(ii)メタデータ／先読みの影響が残る、(iii)整列要件がなくfail-closed検査の対象にならない。したがってmacOSでは「COLD＝キャッシュ汚染ゼロ」を保証できず、HOT/COLD分離の効果はLinuxより弱い。ベンチ比較は同一条件とみなさず、OS別に記録すること。

詳細：

| 観点 | Linux `O_DIRECT` | macOS `F_NOCACHE` |
|---|---|---|
| 強度 | I/O単位の迂回強制。整列違反はエラー（fail-closedで検出可） | fd単位の助言。効果はbest-effortで、無視されてもエラーにならない |
| ページキャッシュ残存 | 当該I/Oは迂回。ただし他buffered fd経由の残存は別問題（fd分離で管理） | 当該fd以外（他fd／mmap／Unified Buffer Cache経由）で同一ファイルがキャッシュされ得る。COLD保証が弱い |
| メタデータ・先読み | データ迂回。先読みは迂回対象 | `F_NOCACHE`適用後もFS先読み・メタデータキャッシュの影響が残り得る。短run束のランダム性と相まって実測ばらつきがLinuxより大きい想定 |
| 整列 | 512必須・4096推奨。違反はINVAL | 不要。検査対象にしない（4096整列バッファの流用は可だが必須化しない） |
| 非対応時の検出 | open errno（EINVAL系）で検出→buffered fallbackが明確 | 失敗しても静かにbuffered動作になり得る。fallback検出ができない分、テストは「一致性」でなく「汚染影響の上限」で評価する（§4） |
| 設計上の帰結 | COLD＝汚染ゼロを主張できる（単一fd管理下） | COLD＝汚染低減（保証なし）。HOT pin＋LRU側の効果に寄せる（§5）。macOS数値はLinux数値と並べない |

## 4. S1 1B実実行前の完了必須項目＋3OS同一pack読み込みテスト計画

### 4.1 完了必須項目（S1実実行ゲート。いずれも文書・テスト定義まで。本wtで実装しない）

1. 新層IF凍結：`jt_io_submit/poll/wait`のシグネチャ・エラー表・`len==0`/`n==0` no-op・スレッド共有禁止をP3本体が承認（本書§1.3が叩き台）。
2. HOT/COLD fd分離規則の明文化：COLD fdに`F_NOCACHE`（macOS）／`O_DIRECT`（Linux）を適用し、HOT fdに適用しないことのレビュー合意。
3. 整列ヘルパ凍結：Linux 512検査（現行）＋Windows用セクタサイズ可変検査（将来）の仕様合意。macOSは検査対象外の合意。
4. fallbackマトリクス凍結：§2の3表（Linux／macOS／Windows）をP3本体が承認。特にmacOS `fcntl`失敗時無視・Windows tailパディング方針の二択解消。
5. 同一packテスト定義（§4.2）の承認。100MB超DLなし（RULE・P3c §3の制約維持）。fixtureは小規模合成packのみ。
6. NVMe測定 harness 定義：`103 MiB/token`（AGENTS.md §8）の計測点（uring統計 or submitカウンタ）を定義。S1での合否判定はP3本体評価に委譲し、本テストでは**一致性のみ**をゲートにする（速度は参考値）。
7. Stage 3との干渉排除確認：本wtでビルド・実行を行わないことの継続（本書作成時点も遵守）。

### 4.2 3OS同一pack読み込みテスト計画（一致性テスト。性能合否なし）

- 対象：`.jtdp`小fixture（例：nseq=8・total≤1Kトークン・全ID `< 48588`・checksum正）＋neuron-major合成coldファイル（例：64行×256B、各行パターン`row_id ^ 0xA5`）。
- 配布：fixture生成スクリプトの**出力バイト列のSHA-256を記録**し、3OSで同一ハッシュのfixtureを用いる（生成器の再実行差異を排除）。
- 手順（3OS共通）：
  1. `jt_dp_open`→全検証通過→`jt_dp_seq_copy`で全seq読み戻し一致（checksum・語彙値域・末尾ゴミ拒否のfail-closedも確認：壊しfixture各1件でINVALを期待）。
  2. coldファイルをbuffered open→`jt_io_pread_batch`でspec配列（runs_compress出力＋no-op混在＋EOF超過1件）読み→先頭2件一致・EOF超過は`JT_ERR_IO`を期待。
  3. Linuxのみ追加：`jt_io_direct_open→alloc(4096)→pread`往復一致＋整列違反3件（dst/off/len）で`INVAL+EINVAL`。非対応FS（tmpfs）ではfallback成功を期待。`JIMOTONO_USE_URING=ON`時のみ`jt_io_pread_batch_uring`一致（OFF時は`NOSUP+ENOSYS`＋同期fallback一致）。
  4. macOSのみ追加：`F_NOCACHE`適用fdと非適用fdで(2)の一致結果が同一であること（性能差は記録のみ・合否に使わない）。
  5. Windowsのみ追加：`_lseeki64＋_read`経路で(2)が一致すること（オフセット共有注意：単一スレッド実行）。
- 合否：全OSで一致性項目がgreen。速度・MiB/tokenは**参考記録**（5%未満差はノイズ、RULE.md §6）。
- 禁止：重負荷ベンチの同時実行（RULE.md §6相互排除）。本テストは短時間correctnessのみ。

## 5. SSD/RAM待ちを発生させない設計原則（非同期＋先読み＋pin）

1. **decodeクリティカルパスにブロッキングI/Oを置かない。** COLD読みは必ず先読み窓（§1.3 `submit`→計算→`wait`）に入れ、dense prefix計算とオーバーラップさせる（現行uring注記「dense prefix計算とのオーバーラップは呼び出し側」どおり）。
2. **HOTはpin＋LRU＋buffered。** 共有expert／attention射影／頻出routed expertは常駐・pinし、`O_DIRECT`/`F_NOCACHE`を適用しない。ページキャッシュを味方にする（DESIGN.MD §3.1 Colibri式）。
3. **COLDはstreaming＋使い捨て。** delta 15〜18%（残り82〜85%持続）のみを`runs_compress`でrun化し、短い順次runの束として読む。ページキャッシュに残さない（Linux O_DIRECT／macOS F_NOCACHE助言）。
4. **先読みは3予測器の和集合（S∪L∪M）に bounded overfetch で。** 単一予測器（71.6%級）に賭けず、和集合＋capでカバーし、`batch=256`窓で回す。過剰fetch分は`103 MiB/token`予算内で抑える（P3b §2・P3統合C3の解決どおり単一はフォールバック）。
5. **router-lookaheadで1層先行。** 次層routingを現層post-attentionから予測し、現層MLP計算中に次層分を`submit`する。
6. **inflight上限を固定（256）。** fill→submit→drain→resubmitの窓進行を崩さない。retryは窓内（容量256）に畳む（現行uring実装どおり）。
7. **RAM待ちは二重化で隠す。** 計算バッファを2面にし、前トークンのCOLD到着待ちと現トークン計算を重ねる。`mmap` reader（`.jtdp`）のmajor faultは起動時タッチで前倒しする。
8. **測定なき主張をしない。** `poll`/`wait`の待ち時間・pin-hit率・MiB/tokenをsubmitカウンタで記録し、η実測・sync逓増・O_DIRECT/SSD待ち混入（P3統合§3申送り）と切り分ける。5%未満差はノイズ扱い。

## 参照

- `include/jimotono/io_batch.h`（HOT/COLD方針・OS fallback注意・uring TODO）
- `include/jimotono/io_direct.h`（512/4096整列・fail-closed・uringとの分離）
- `src/io_batch.c`（Phase 1 preadループ・Windows `_lseeki64＋_read`・macOS F_NOCACHE注記）
- `src/io_batch_uring.c`（batch=256・depth 256・retry窓・NOSUP流儀）
- `src/io_direct.c`（Linux実経路＋非対応FS fallback・非Linuxスタブ）
- `include/jimotono/common.h`（`JT_ERR_*`・LE限定）
- `include/jimotono/data_pack.h`（`.jtdp` v1・mmap reader・Windowsスタブ）
- `DESIGN.MD §3`、`RULE.MD §6`、`analysis/p3-integration.md §1–§4`
