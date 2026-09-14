# P3e データセットストリーミング設計（文書のみ・実装禁止）

- worktree: `/Users/shibadogcap/jimotono-wt-p3e` / branch: `docs/p3/dataset-stream`
- 性質：設計文書のみ。実装（`.c`/`.h`/CMake変更）禁止、ビルド・実行禁止、重負荷実行禁止。他wt接触なし、`git checkout/switch`なし。コミットしない。
- 位置づけ：**S2前必須**。S1はTinyStoriesでネット不要（ローカル完結）。本書はS2（10B拡大）〜S3（Corpus本格）への入力。
- 前提資産：`include/jimotono/data_pack.h`（`.jtdp` v1）／`src/data_pack.c`／`include/jimotono/bpe.h`（llm-jp v2.2 Viterbi）／`src/bpe.c`／`bench/data/pack_jsonl.c`（F6 pack経路・暫定仕様）／`analysis/p3-training-scale.md` §3（S1→S2→S3段階）／`analysis/p3-integration.md`（S3順序凍結）／`analysis/p3-io-abstraction.md`（SSD I/O層）／`analysis/tokenizer-gap.md`（3.6 MB/s実測）／AGENTS.md §5・§7（データセット・規約・禁止事項）。
- 規約：C11・`restrict`・errnoベース＋`goto cleanup`・同一ソース3OSビルド可能（AGENTS.md §7.1）。LE前提。llama.cpp / vLLM / SGLang / PyTorch 直接リンク禁止、HF形式への固執禁止、GPU前提禁止（AGENTS.md §7.2）。

## 0. 要約（選択のみ）

| 項目 | 選択 | 理由1行 |
|---|---|---|
| 転送 (§1) | **HTTP Rangeを正経路**。HF streaming不採用、GitLab APIは限定利用 | libcのみ＋再開 native＋3OS同一ソース。HF依存（python/hf libs）は禁止事項に抵触 |
| 形式 (§2) | **`.jtdp`拡張を推奨**（shard＋manifest＋sidecar）。WebDataset/MDS不採用 | 既存mmap reader・FNV・fail-closedを継承。依存追加ゼロ |
| 再開・検証 (§3) | shard-manifest＋per-shard checksum＋ライセンスsidecar | 部分取得・再開・監査をshard粒度で閉じる |
| トークナイズ (§4) | **オフライン既定**（bpe C＋pack後継）。オンラインは将来 | 学習時は`.jtdp` mmap再利用。3.6 MB/sは一回切りで臨界外 |
| 優先度 (§5) | ja_wiki subset → SwallowCode-v2 → JFCD | `p3-training-scale.md` S3・`p3-integration.md`順序凍結と同一 |
| 層分離 (§6) | `jt_io_*`拡張禁止。**新規HTTP層**（例 `jt_ds_net_*`）に分離 | SSD I/O（O_DIRECT/uring/F_NOCACHE）と失敗意味論が異質 |
| 計画 (§7) | D0設計凍結 → D1実装 → D2検証。S2 exitの前段ゲート | S2実データ投入前に一致性・ライセンス・再開をgreen化 |

## 既存資産との対応表（必読）

| 既存資産 | 本設計での扱い | 変更 |
|---|---|---|
| `data_pack.h/v1`（magic `JTDP`・header 64B・u32 payload・offset表・FNV-1a・末尾ゴミ拒否・語彙値域<48588・mmap reader・Windows readerスタブ） | shard payload形式として**そのまま再利用**。v1 readerは`flags!=0`拒否を維持 | 変更なし（拡張は別物 §2） |
| `src/data_pack.c`（writer open→add→close・close時backpatch・部分ファイルはchecksumで拒否） | shard writerとして再利用。manifest側で未完shardを除外 | 変更なし |
| `bpe.h/bpe.c`（Unigram Viterbi・`.jtvocab` v1・▁接頭辞・fail-closed・3.6 MB/s実測・golden 38/38） | オフライン変換のコアに据える（§4） | 変更なし（高速化は別タスク） |
| `bench/data/pack_jsonl.c`（空白区切り整数列の暫定仕様・JSONL本格パーサ禁止の明記・ctest外） | **暫定廃止予定**。後継 `pack_text`（text→bpe→`.jtdp`）に置換（§4） | 将来実装時に置換（本wtでは文書のみ） |
| F6 pack経路（JSONL直接読み禁止・16ワーカー破綻の教訓→バイナリ経由正規） | 継承。ストリーミング取得物も**必ず`.jtdp`化してから学習** | 方針継承 |
| `jt_io_*`（`io.h`・HOT/COLD・O_DIRECT/uring・F_NOCACHE・Windowsスタブ） | **触らない**。ネットワーク層と混ぜない（§6） | 変更禁止 |
| S1 TinyStories（`results/bench/20260914-eval.md`・F6 val比較・1.92GB→約9min試算） | S1はネット不要のまま。本設計の検証も合成fixture＋TinyStories headで閉じる | 変更なし |

---

## 1. 転送：HTTP Range / HF streaming / GitLab API の比較と選択

### 1.1 比較表

| 観点 | (A) HTTP Range（推奨・正経路） | (B) HF streaming（不採用） | (C) GitLab API（限定利用） |
|---|---|---|---|
| 依存 | libc＋OSソケット/TLSのみ。自前C実装可 | `huggingface_hub`・`datasets`（python）or libcurl＋HF resolve規約。AGENTS §7.2の依存回避方針と衝突 | HTTP Range上の薄いwrapper＋認証ヘッダ。限定範囲なら依存増なし |
| 再開 | Range (`bytes=start-end`) でnative再開。shard粒度と相性◎ | 内部でRange相当を行うがAPI越しで制御弱。オフライン再現が取りにくい | Range併用可。ただしレート制限・ページング規約が別途必要 |
| 3OS同一ソース | 可（ソケット＋TLS抽象で閉じる） | python依存を持ち込むと3OSビルド均一が崩れる | 可（(A)の派生として実装） |
| 対象データ | HF resolve URLもGitLab生ファイルも一様に扱える（URLが取れればよい） | SwallowCode-v2・JFCD（HF）に強いがllm-jp系の出自混在に弱い | llm-jp系がGitLab／独自配布の場合の取得に限定有効 |
| 再現性・監査 | URL＋Range＋SHA-256/FNVをmanifestに記録しやすい | バージョン解決がサーバ側改訂に引きずられる（`revision`固定でも完全同一の保証が弱い） | project/fileパス＋commit SHA固定で再現可（記録は必須） |
| ライセンス記録 | 取得元URLをsidecarに直結できる | ライセンス表示はあるが取得物の対応付けが間接的 | API応答のlicense欄をsidecarに転記可 |
| 失敗意味論 | タイムアウト・切断・416等を自前で定義（SSD I/Oと別物 §6） | SDKの例外体系を持ち込む（C層と不整合） | レート制限429・認証401等の別体系 |

### 1.2 選択

- **正経路：(A) HTTP Range**。全データセット共通の取得プリミティブは `GET + Range + リトライ + 再開` のみとする。
- **(B) HF streamingは不採用**。取得したい実体がHFにある場合も、HF SDK経由ではなく **HF resolve URLに対する素のHTTP Range取得** に落とす（SDK非リンク・python非依存）。`revision`（commit SHA）はmanifestに記録し、URL解決結果（リダイレクト先の実URL＋ETag/Last-Modifiedがあれば）をsidecarに残す。
- **(C) GitLab APIは限定利用**。llm-jp Corpus v4系の配布がGitLab／独自形式の場合に、**ファイル列挙・commit SHA固定・生ファイルURL発行** の用途に限定。実バイト転送自体は(A)に委譲する。レート制限・認証は取得ツール側の設定とし、Cランタイムに持ち込まない。

### 1.3 未解決（U1-x）＋相対コスト＋ゲート

- U1-1：llm-jp Corpus v4の実配布URL形態の確定（HTTP直リンク可か／GitLab経由か／要認証か）。現状AGENTS §5の参照はリポジトリ級であり、**実ファイルURL未確定**。→ コスト：S（調査半日級）。ゲート：D0凍結条件。未確定のままD1に入らない。
- U1-2：TLS実装の選定（OS native：Linux mbedTLS/OpenSSL・macOS SecureTransport/Network.framework・Windows WinHTTP/Schannel のいずれを薄抽象にするか）。→ コスト：M（設計中級。実装はD1）。ゲート：D0で選択肢を2案まで絞ること。
- U1-3：HF resolveのリダイレクト・認証付きURL（parquet等の実体）のRange可否の実測。→ コスト：S（HEAD/Range probeのみ。重負荷なし）。ゲート：D2検証のprobe項目。
- U1-4：GitLabのレート制限・ページング上限の実値。→ コスト：S。ゲート：限定利用のままならD2で上限記録のみ。

---

## 2. 形式：WebDataset / MDS / 独自`.jtdp`拡張の比較（`.jtdp`拡張を推奨）

### 2.1 比較表

| 観点 | WebDataset（tar＋shard） | MDS（MosaicML Streaming） | **`.jtdp`拡張（推奨）** |
|---|---|---|---|
| 依存 | tar展開＋pythonエコ贔屓。C単体実装は可能だが仕様追従コスト大 | python＋独自index。C移植は実質rewrite | **依存ゼロ**（現行`data_pack`継承） |
| ランダムアクセス | tar内offset走査。mmap直結しにくい | chunk indexあり。だが形式が外部規定 | **offset表つき可変長**でmmap直結済み |
| 検証 | 外付け（別途hash） | 内蔵hashあり | **FNV-1a内蔵＋fail-closed一式**（magic/版数/境界/単調性/語彙値域/末尾ゴミ拒否） |
| 語彙結合 | 形式と無関係（別途） | 同左 | **<48588値域検査がreaderに組込み済み** |
| 学習読み | python DataLoader前提 | python DataLoader前提 | **mmap readerでC学習直結**（F6実績） |
| 3OS | tar差異・python差異を持ち込む | python差異を持ち込む | **同一ソース3OS**（Windows readerは現行TODOスタブのまま申送り） |
| 移行コスト | 変換器＋検証の全面新規 | 同左 | **writer再利用＋manifest追加のみ** |

### 2.2 推奨：`.jtdp`拡張の構成（v1本体は不変）

v1 shard（`.jtdp`）のバイナリは**一切変えない**。拡張は「shard分割＋manifest＋sidecar」の外側に置く：

```text
dataset/
  manifest.jtdm          # shard一覧・順序・checksum・tokenizer版・license参照（テキストJSON。Cパーサは最小subset）
  shard-00000.jtdp       # 既存v1そのまま（u32 payload＋offset表＋FNV）
  shard-00001.jtdp
  ...
  licenses/
    shard-00000.license.json  # 取得元URL・revision・license名・取得日時・hash
    ...
```

- `manifest.jtdm`（新規・テキスト）：`{ version, tokenizer:{vocab, jtvocab_sha256, normalizer_note}, shards:[{file, nseq, total, fnv, sha256, source_url, revision, license_file}], order_seed }`。C側は将来の最小JSON subset読みのみ（本wt実装なし）。
- shard粒度：nseqまたはtotalトークンで均等分割（例：1 shard ≒ 数十MB級。D0で確定）。空シーケンス保持（行↔seq対応のF6規約を継承）。
- 将来のuint16パック（`flags bit0`予約）は本設計の範囲外。v1 readerの`flags!=0`拒否は維持し、拡張で緩めない。
- ライセンスsidecarはshard単位（取得元が混在しうるため）。manifestは参照集約のみ。

### 2.3 未解決（U2-x）＋相対コスト＋ゲート

- U2-1：shardサイズ既定（nseq vs total基準・何MB級か）。→ コスト：S（D0決定）。ゲート：D0凍結項目。
- U2-2：`manifest.jtdm`のJSON subset仕様（許容型・fail-closed条件）。→ コスト：S。ゲート：D0凍結項目。C実装はD1。
- U2-3：Windows mmap readerスタブ（現行TODO）の扱い。ストリーミングとは独立課題として申送り、本設計では**Linux/macOS先行・Windowsはbuffered fallback読込**の方針記録に留める。→ コスト：M（別タスク）。ゲート：D2ではLinux＋macOS一致性のみを合否にする。
- U2-4：uint16パック将来対応との両立（拡張manifestに`pack`欄を予約するか）。→ コスト：XS（予約1欄）。ゲート：D0で欄予約の有無のみ決定。

---

## 3. 再開：チェックポイント・部分取得、検証：チェックサム・ライセンス記録

### 3.1 再開設計（shard粒度で閉じる）

- 取得単位はshard。shard内はさらにRange-chunk（例：1–8 MiB）で分割取得し、**到着済みchunk bitmapを`.part` sidecarに保持**する。プロセス再起動後はbitmap＋実サイズから未取得Rangeのみ再発行する。
- 完了条件：shard全chunk到着 → `.jtdp`全検証（§3.2）通過 → `.part`削除 → manifestに`done`記録。検証不通過のshardは**除外**し、学習入力に混ぜない（fail-closed継承）。
- 学習側チェックポイント（step・data cursor）は取得側と**別物**。対応は `manifest order + shard内seq index` のカーソルで行い、取得未完shardはカーソル範囲外に置く（未完データで学習を進めない）。

### 3.2 検証設計（二層checksum＋ライセンス）

| 層 | 内容 | 既存対応 |
|---|---|---|
| L1（shard内） | 現行FNV-1a（payload＋offset表）＋magic/版数/境界/単調性/語彙値域/末尾一致 | `jt_dp_open`全検証をそのまま使う |
| L2（配布） | manifest記載のSHA-256（shardファイル全体）。取得完了時に照合 | 新規（D1実装。検証ツール側） |
| L3（出自） | sidecar：source_url・revision/commit・ETag・取得日時・license名・license本文hash | 新規（監査用） |

- ライセンス記録は**shard単位JSON**＋manifest集約。最低欄：`source_url, revision, license_name, license_url_or_path, retrieved_at, sha256, note`。Corpus混植（ja_wiki＋code混在shard等）はshardを分け、混在させない（出自追跡のため）。
- 検証失敗時の扱い：L1/L2不一致→当該shard破棄＋再取得1回→再不一致なら除外記録（manifestに`excluded`）して先に進む。学習は除外済みmanifestでのみ走る。

### 3.3 未解決（U3-x）＋相対コスト＋ゲート

- U3-1：Range-chunk幅・並列数・リトライ回数・backoff既定。→ コスト：S（D0決定）。ゲート：D0凍結。
- U3-2：SHA-256実装の調達（libcのみ方針との整合。自前かOS APIか）。→ コスト：S（選定のみD0、実装D1）。ゲート：D0で調達先決定。
- U3-3：学習カーソル形式（order_seed＋shard/seq index）の仕様。→ コスト：S。ゲート：D1実装前凍結。
- U3-4：除外shard発生時の学習継続可否の閾値（例：除外率>X%で停止）。→ コスト：XS。ゲート：D2検証定義に含める。

---

## 4. トークナイズ：オンライン vs オフライン（bpe C実装＋pack経路の活用）

### 4.1 比較と選択：オフライン既定

| 観点 | オフライン（推奨・既定） | オンライン |
|---|---|---|
| 経路 | 取得text → `jt_bpe_encode` → `jt_dp_writer_add` → `.jtdp` shard → 学習はmmap | 学習ループ内で都度`encode` |
| 既存対応 | `bpe.c`＋`data_pack.c` writerの直結。F6「バイナリ経由正規」を継承 | 現行C APIは都度確保型で学習内反復に向かない |
| 性能 | 3.6 MB/s実測（`tokenizer-gap.md`）は一回切り。16MB→約5s、1.92GB→約9min試算。S2前の前処理としては許容域 | 毎step encodeはViterbi O(n·plen)が学習律速に混入。27×高速化（100MB/s級）はrewrite級でS2前には載せない |
| 再現性 | `.jtvocab` SHA＋`manifest.jtdm`のtokenizer欄で固定。val比較は`val_ce_pb`正規化（tokenizer-gap比較規則） | 語彙・版ずれがstep間に混入しうる |
| メモリ | 変換時ピークのみ。学習時は`.jtdp` mmapで定常化 | 学習peakにDP bufferが上乗せ |

- 選択：**オフライン既定**。オンラインはS3以降の検討課題とし、S2前は実装しない。
- F6 pack経路の活用：`pack_jsonl.c`の暫定仕様（空白区切り整数列・JSONL本格パーサ禁止）は**トークナイザ確定（v2.2・bpe.c golden 38/38）をもって廃止予定**とし、後継 `pack_text`（text抽出→`jt_bpe_encode`→`jt_dp_writer_add`の一貫パイプライン）に置換する。JSONL直接読み禁止・バイナリ経由正規の原則は維持。後継のJSON text抽出は最小subset（`pack_jsonl.c`のfail-closed流儀を継承し、非対応入力は即エラー）に留める。

### 4.2 未解決（U4-x）＋相対コスト＋ゲート

- U4-1：後継packの入力text抽出subset（JSONLのどのkeyを本文とするか。データセット毎に異なる）。→ コスト：S（データセット毎の対応表D0確定）。ゲート：D1実装前凍結。
- U4-2：bpe高速化（100MB/s級・OmniToken方式）の要否。現状low priority・トリガ3条件（GB級／反復loop入り／wall-clock 5%超）を維持し、S2前は**着手しない**。→ コスト：L（rewrite級。S2対象外）。ゲート：トリガ発火時のみ再訪。S2ゲートに含めない。
- U4-3：byte-fallback由来の非UTF-8復号物のpack時扱い（`bpe_decode`注意の裏返し。encode側fail-closed：不正UTF-8は符号化せずエラー）。→ コスト：XS。ゲート：D0でfail-closed維持を再確認。

---

## 5. 優先度：ja_wiki → SwallowCode → JFCD

- 順序（固定。`p3-training-scale.md` S3・`p3-integration.md` §段階と同一）：**① ja_wiki subset → ② SwallowCode-v2 → ③ JFCD（Japanese Function Calling Dataset）**。
- 理由：日本語基盤（LLM-jp Corpus v4内ja_wiki）で言語土台→コード（SwallowCode-v2）でコーディング→ツールコール（JFCD）で機能呼出し、の依存順。S3 exitの評価（MT-Bench／SWE-bench Pro／ツールコール成功率はP3本体評価）に対応する調達順でもある。
- S1との関係：S1はTinyStoriesのみでネット不要。本順序はS2データ投入開始の順序であり、S1 exitをブロックしない。
- shard分離：①②③を**同一shardに混ぜない**（ライセンス出自・ドメイン混合率の追跡のため）。混合学習比率（例：ja何%等）はP3本体（学習スケール設計）の決定に委ね、本書では混合を読まない（取得順のみ固定）。

### 未解決（U5-x）＋相対コスト＋ゲート

- U5-1：ja_wiki subsetの範囲定義（記事数・ダンプ版・除外条件）。→ コスト：S。ゲート：D0で版固定（ダンプ日＋版SHA）。
- U5-2：SwallowCode-v2の取得実体（HF revision・ファイル列挙）。→ コスト：S。ゲート：D0でrevision固定。
- U5-3：JFCDの取得実体・ライセンス確認。→ コスト：S。ゲート：D0でlicense記録可否を確認（不可なら除外記録）。
- U5-4：ドメイン混合比率・カリキュラム（順次か混合か）。→ コスト：M（学習設計）。ゲート：**S2前決定不要**。P3本体に申送り（本書のゲートに含めない）。

---

## 6. ネットワークI/OはSSD I/Oと別レイヤー（`jt_io_*`拡張禁止、HTTP層新規の方針）

### 6.1 方針

- `jt_io_*`（`io.h`：HOT/COLD・submit/poll/wait・O_DIRECT/uring・F_NOCACHE・Windowsスタブ）は**拡張禁止**。ネットワーク取得は別層の新規HTTP層（仮称 `jt_ds_net_*`。D0で正式名凍結）に分離する。
- 層図：

```text
S2/S3 data pipeline (acquire → pack → train)
  ├─ acquire (NEW, network): jt_ds_net_{fetch_range,resume,verify_manifest}  ← 本設計
  │     失敗: timeout / reset / 416 / 429 / 401 / hash mismatch（リトライ＋除外）
  ├─ pack (reuse): jt_bpe_encode + jt_dp_writer_* → shard-*.jtdp + manifest.jtdm
  └─ train read (EXISTING, SSD): jt_dp_open/mmap + jt_io_submit/poll/wait   ← 変更なし
        失敗: INVAL(fail-closed) / IO(short-read/EOF→EIO) / NOSUP（既存意味論のまま）
```

- 理由：(i)失敗意味論が異質（SSDは整列・EOF・O_DIRECT可否、netはtimeout・再開・レート制限）。(ii)HOT/COLD・pin・O_DIRECTのSSD関心事をnetに漏らさない。(iii)`JT_ERR_*`番号変更禁止（io-abstraction前提）のもと既存呼び出し側を壊さない。(iv)テスト分離（netはfixtureサーバ、SSDは合成packで各々完結）。
- 禁止の再確認：`jt_io_submit`にURL・Range・リトライを持ち込まない。`io_direct`/`io_batch`にソケット分岐を足さない。取得済みバイトのSSD書き込みは通常buffered file I/Oで行い、`jt_io_direct_*`（O_DIRECT足場）を流用しない（取得パスはpage-cache汚染の関心外であり、COLD保証の対象にしない）。

### 未解決（U6-x）＋相対コスト＋ゲート

- U6-1：新規層の正式名・ファイル配置（`jt_ds_net_*`仮称の承認）。→ コスト：XS。ゲート：D0凍結。
- U6-2：エラー表（net独自コードを`JT_ERR_*`既存に写像する表。番号追加なし）。→ コスト：S。ゲート：D0凍結。
- U6-3：TLS抽象の境界（net層内か更に下層か）。→ コスト：S（U1-2と合同）。ゲート：D0で2案化。

---

## 7. S2前の設計＋実装＋検証の段階計画

### 7.1 フェーズ（S1と並行可。S1をブロックしない）

| フェーズ | 内容 | 入力 | 出力 | ゲート |
|---|---|---|---|---|
| D0 設計凍結（本文書の承継） | U1-1・U2-1/2/4・U3-1/2/3・U4-1/3・U5-1/2/3・U6-1/2/3の解消。shard/Manifest/sidecar/error表の凍結 | 本書 | 凍結仕様（改訂は設計改訂扱い） | P3本体承認。未凍結のままD1に入らない |
| D1 実装 | net層＋pack後継＋manifest/sidecar生成＋verifyツールの実装（別wt・別branch。本wtでは行わない） | D0凍結仕様 | 取得→pack→verifyの一貫ツール | ビルド3OS通過・単体テストgreen |
| D2 検証（S2前必須） | 下表の検証項目を全green化。重負荷なし・100MB超DLなし | D1成果物 | 検証記録 | **全項目greenがS2データ投入の前提**。除外発生時は除外記録＋P3本体判断 |

### 7.2 D2検証項目（一致性・再開・ライセンス。速度合否なし）

1. 合成fixture往復：text fixture → pack後継 → shard `.jtdp` → `jt_dp_open`全検証→全seq一致。壊しfixture（magic破損・checksum破損・語彙値域外・末尾ゴミ）でINVAL拒否を確認（`data_pack` fail-closedの継承確認）。
2. TinyStories head往復：16MB head級の小fixtureでtext→bpe→pack→mmap読み一致。`val_ce_pb`正規化ログの雛形確認（tokenizer-gap規則）。
3. 再開probe：ローカルfixture HTTPサーバ（重負荷なし）に対するRange分割取得→中断→再開→SHA-256一致。chunk bitmapの再開正当性を確認。
4. 実URL probe（軽量）：各データセットの先頭小Range（例：1MiB以内）のHEAD/Range可否・ETag・revision解決のみ。**100MB超DL禁止**（io-abstraction §4.2の制約継承）。
5. ライセンス記録：①②③各1 shard分のsidecar生成→必須欄充足を確認。不可データは除外記録。
6. 層分離確認：`jt_io_*`無変更（diff空）の確認＋net層単独テスト。SSD読み（合成packの`jt_dp_open`＋`jt_io`経路）とnet取得のテストが相互独立であること。
7. Windows申送り：Windowsではreaderスタブ前提のbuffered確認のみ（一致性合否はLinux＋macOS）。

### 7.3 相対コスト総括（S2前範囲）

| 項目 | 相対コスト |
|---|---|
| D0設計凍結（調査＋仕様） | S（本書承継＋U解消。半〜1日級想定だが断定しない） |
| D1実装（net層＋pack後継＋verify） | M（C新規・TLS含む最大項） |
| D2検証（fixture＋probe＋記録） | S（重負荷なし） |
| bpe高速化（対象外・将来） | L（S2前に着手しない） |

### 7.4 S2 entryゲート（本書がP3本体に申送る条件）

- G-S2-DS1：D0凍結仕様のP3本体承認（shard・manifest・error表・版固定）。
- G-S2-DS2：D2検証1〜7の全green記録（速度は参考値のみ）。
- G-S2-DS3：①②③の版・revision・license記録の充足（不可分は除外記録＋本体判断）。
- G-S2-DS4：`jt_io_*`無変更の確認（SSD層とnet層の分離維持）。
- いずれか不通過の場合はS2データ投入に入らず、S1反復（TinyStories）継続＋設計改訂。

## 参照

- `include/jimotono/data_pack.h`（`.jtdp` v1仕様・fail-closed・Windowsスタブ）
- `src/data_pack.c`（writer/reader実体・FNV-1a）
- `include/jimotono/bpe.h`／`src/bpe.c`（v2.2 Viterbi・`.jtvocab` v1）
- `bench/data/pack_jsonl.c`（F6暫定pack・廃止予定）
- `include/jimotono/io.h`（`jt_io_*`・拡張禁止対象）
- `analysis/p3-io-abstraction.md`（SSD層設計・3OSテスト流儀・100MB超DL禁止）
- `analysis/p3-training-scale.md` §3（S1→S2→S3・S2/S3 exit条件）
- `analysis/p3-integration.md`（S3順序凍結・段階改訂則）
- `analysis/tokenizer-gap.md`（3.6 MB/s・100MB/s目標・比較規則）
- AGENTS.md §5・§7・§8（データセット・規約・禁止・成功基準）
