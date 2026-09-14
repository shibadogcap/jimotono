# P3b注意機構カーネル設計（FlashAttention＋PagedAttention・注意機構のみ）

- branch: `docs/p3/attn-kernels`（本wtのみ。コミットしない）
- 性質：文書のみ。実装禁止、重負荷実行なし（Web調査＋既存計測値の再利用のみ。新規bench・DL・学習実行なし）。
- 前提：`analysis/p3-architecture.md`（P3a）が最優先であり、他すべての前提。本書はP3aと矛盾させない。矛盾発見時は本書を直さず設計会議へ申送る、のではなく本書内で完結させる（P3本体への申送りは行わない。本書はP3bの一部として完結する）。
- 位置づけ：注意機構のみ。P3bの一部。P3本体・P3c実装・Phase G再拡大には着手しない。4文書を混ぜない（MoE粒度・語彙・学習スケール・デコード投機の再定義はしない。必要分は依存として輸入する）。
- コスト表記：相対コストは Phase G Step 4 micro（`jt_gemm_mat_f32`、Mr=12×Nr=4）を「中」とした相対（小＜中＜大）。絶対人月・絶対期間は記さない（P3a §冒頭と同一）。
- 数値の使い分け：帯域上限は公称値（N100 8B／Ryzen 2.6B）を設計目標に使わず、保守値（N100 約1.3B・Ryzen 約0.4B）を上限目安に使う。toks/sのcross-vocab直接比較は行わない。
- 整合対象（注意機構側のみ。再定義しない）：`analysis/p3-architecture.md` §0–§2、`analysis/kv-cache-design.md` §2–§4・P3-1〜P3-4、`analysis/decoding-architecture.md` §1、`analysis/p3-design-inputs.md` §(c)C2（prefillはGEMM体制・計算律速）、`analysis/roofline.md` D4・D6・D7 G1、`analysis/active-upper-bound.md` §1–§4、AGENTS.md §1・§2.1・§7.1・§8。
- 非スコープ：GDN-2逐次再帰本体・チャンクWY学習形・MoE dispatch・語彙head・W8A8/T-MAC使い分け・GRPO順序自体の変更はいずれも本書で再定義しない。GRPOについては§4の反映文案（前提一文＋理由）の提示に留め、`analysis/p3-training-scale.md`自体には触らない。

## 0. 境界条件（再掲ではなく境界）

- 対象はフル注意層（全層の約1/5。4:1時20%、5:1時約17%）の2機構のみ：(i) prefill用FlashAttentionタイル化、(ii) decode・生成N回用PagedAttentionページ管理。
- 線形層（GDN-2）は両機構の適用対象外ではないが扱いが異なる：FlashAttentionの対象外（§1で除外理由を明示）、PagedAttentionの対象外（§3で除外理由を明示）。除外は「未対応」ではなく設計上の除外である。
- KV圧縮（CSA2＋FP4）は捨てない。本書の両機構は圧縮の上（併用前提）に載る設計とし、非圧縮KVへの先祖返りは設計改訂とする（§5）。
- `kv_prefix.h` は本wtに存在しないことを確認済み（`include/jimotono/` に該当なし。glob一致なし）。よって§2の統合方針は「前提としての期待IF」を記述し、ヘッダの新規作成・実装には着手しない。

## 1. FlashAttention：フル注意層prefill用タイル化

- **方針**：
  - 目的は長文脈prefillのRAM有界化である。N×N注意行列を物質化せず、Qタイル外ループ×KVタイル内ループ＋オンラインsoftmax（m＝rowmax、l＝rowsumの逐次再スケール）でO(N^2)中間をO(N・d)に抑える。CPU版の対応付け：GPU SRAM→L2/L1常駐タイル、HBM→DRAM＋SSD永続（圧縮済みFP4グローバルKV）。
  - 適用先はフル層prefillのみ。GDN-2線形層・decode逐次（1トークンGEMV体制）・学習チャンクWYには適用しない（混入は設計改訂）。prefillがGEMM体制・計算律速であること（C2）と整合し、タイル化は帯域律速のdecodeを速くするためのものではなくprefillのピークRAMを有界にするためのものと位置づける。
  - タイル方針：Qブロック（例 Br＝128〜256行）× KVブロック（例 Bc＝128〜256列）の2重ループ。1タイルがL2に常駐すること（FP16換算で Qタイル＋Kタイル＋Vタイル＋O蓄積＋m/lベクタがL2の1/2以下に入ること）。AVX2（Mr=12×Nr=4確立済み）／AVX-512／NEONは `#ifdef` 分岐で同一数値契約とし、`restrict`＋`goto cleanup`＋`errno`ベース（AGENTS.md §7.1）を守る。`Br/Bc` 既定値の固定はしない（L2サイズ・head_dim・INT8/FP16走査幅で変わるためノブとして予約）。
  - KV再読み込み回避：素朴なQ外×KV内ではKVをQタイル数分だけ再走査する。これを「回避」とは再読みゼロではなく再読みの有界化＋再利用最大化と定義する：(i) K/VタイルのL2常駐再利用（内ループ内でK・Vを1回ずつ読む順序に固定）、(ii) Qタイル側のO/m/l蓄積をレジスタ＋L1に留めDRAM往復させない、(iii) CSA2層間共有（Reindex/Reuse）と組む場合は前方Full層のmain KV再利用をタイル外で解決し、タイル内では再スコア（候補プール内のみ）に限定する。KV再読み量の見積り式は `O(N^2・d / Bc)` のオーダーではなくタイル設計の受入条件（下記ゲート）に含める。
  - 精度・量子化との接点：注意射影はINT8（P3a §6維持）。タイル内演算は既存INT8/FP16内積パスを再利用し、FP4格納KVはオンフライ逆量子化（16エントリLUT×E4M3スケール乗算、§5と同一）でタイル内に展開する。事前逆量子化（タイル外で全KVを展開）は帯域利得を消すため不可。RoPE適用後量子化のため格納順＝走査順とし、タイル化が転置を発生させないこと。
  - 長文脈RAM有界化の定義：prefillピークRAMが `O(N・d + Br・d + Bc・d)` に収まること（N×N項を含まないこと）。N=1M級でも中間行列で落ちないことが目標であり、toks/s絶対値の断定はしない。
- **未解決**：
  - Br/Bc の既定値未定（L2サイズ別・AVX2/AVX-512/NEON別の実測で確定する。本書ではノブ予約に留める）。
  - INT8注意パス＋FP4オンフライ展開の融合形のCコスト未確定（タイル内LUT展開＋FMAの実効帯域は未実測）。
  - CSA2候補プール（8トークン=1ブロック・top-2048ブロック→16,384候補）のブロック粒度とFlashタイル Bc の整合（倍数関係にするか否か）未確定。
  - SWA相当の局所KV（FP8維持・SSD非永続）に対するタイル化の要否未確定（n_win=128の小窓はタイル化せず素朴注意で足りる可能性があり、本書では適用任意とする）。
- **相対コスト**：中（タイル制御＋オンラインsoftmax＋SIMD分岐＋オンフライ逆量子化の融合が主。Br/Bc仕様化のみなら小。合計して中とし、大にはしない。Phase G Step 4 micro＝中との混同禁止）。
- **ゲート条件**：
  - prefillピークRAMがN×N項なし（`O(N・d)` 有界）であることを式＋実装時実測で示すこと。N×N物質化の混入は設計改訂。
  - 適用対象がフル層prefillのみであること（GDN-2逐次・decode GEMV・学習チャンクWYへの混入禁止）。
  - 事前逆量子化の混入禁止（§5と同一）。スケール隣接配置（同一キャッシュライン）を崩さないこと。
  - KV再読み量をタイル設計の受入条件として記録すること（再読みゼロの主張禁止。有界化の実測で判定）。

## 2. PagedAttention：KVページ管理

- **方針**：
  - 目的はdecode・生成N回におけるKVの非連続管理＋prefix共有である。連続KV（N×d物質化）をやめ、固定長ブロック（例 16トークン/block。CSA2候補プールの8トークン単位の倍数に取ることを推奨。既定値固定はしない）×ブロックテーブル（seq→block_id列）で管理する。割当てはアリーナ／プールアロケータ（AGENTS.md §7.1）＋`errno`＋`goto cleanup`とし、OSページキャッシュ依存・`malloc`逐次確保は不可。
  - 非連続ブロック：QO（query-output）計算時にブロックテーブル経由で gather 走査する。転置・コピーバックによる連続化は不可（コピーが帯域律速のdecodeを悪化させるため）。FP4格納（nibble pack＋per-16 E4M3スケール隣接）はブロック内に閉じること（スケールの別ページ参照禁止。DESIGN §5思想と同一）。
  - prefix sharing：同一prefixを持つ系列間でブロックを参照共有（refcount＋CoW）する。共有単位はブロック単位とし、トークン単位の部分共有はしない（管理複雑化のため）。分岐点以降の追記はCoWで分離する。SWA相当の局所KV（FP8・DRAM一時・TTL付き）は共有対象外とし、グローバル圧縮KV（FP4永続）のみ共有する（kv-cache §2.4の永続分離と同一）。
  - GRPOのNサンプル共有：同一プロンプトからのN生成はprefixブロック（プロンプト部）を全サンプルで共有し、サンプル固有の継続部のみ別ブロックに追記する。これにより生成N回分のKVが `1×prefix＋N×suffix` になり、素朴なN倍化を避ける（詳細の前提文案は§4）。
  - `kv_prefix.h` 統合方針（ファイル不在のため前提として記述。ヘッダ作成・実装はしない）：
    - 前提IF（期待形。名称は仮）：`kv_prefix_lookup(hash)→block_id列`、`kv_prefix_insert(hash, block_id列)`、`kv_prefix_refcount_inc/dec`、`kv_prefix_invalidate(ttl)`。ハッシュ単位はブロック整列済みprefix（ブロック境界に丸めたもの）のみとし、非整列末尾はハッシュ対象外（Bounded Replayのn_win=128再計算部と同一視）。
    - 統合点：PagedAttentionのブロック割当て器が `kv_prefix_lookup` ヒット時は新規確保せず既存block_idをrefcount＋1で共有し、ミス時のみ新規確保＋`kv_prefix_insert` する。退避（SSD永続）は圧縮済みFP4グローバルKVのみ（SWA非永続は§1・kv-cache P3-1と同一）。TTL・LRU（最低72h相当の方針）はprefix側の責務とし、ページ管理側で再定義しない。
    - 将来 `kv_prefix.h` が現れた場合の照合条件：ハッシュ粒度がブロック境界丸めであること、共有対象がFP4グローバルKVのみであること、refcount/CoW境界がブロック単位であること。上記いずれかを欠く場合は本書の統合方針側ではなく先方IFの設計改訂として扱う（本書は直さず記録する。P3本体への申送りではない）。
- **未解決**：
  - ブロック長既定値未定（16推奨の根拠はCSA2の8トークン単位との倍数整合のみ。head_dim・FP4 nibble整列・64Bライン整列との連立でP3b内実測により確定する）。
  - refcount/CoWの並行安全性（io_uring＋compute重ね合わせ時のordering）未設計。
  - DRAM一時プール（SWA FP8・128×フル層×同時セッション数）とページ済みグローバルKVの容量配分未算定。
  - 大バッチ・長文脈時のブロックテーブル走査オーバーヘッド（gather離散化の帯域罰則）未実測。
- **相対コスト**：小〜中（テーブル＋refcount＋CoW＋アリーナ割当ての本体は小。io_uring重ね合わせ時のordering＋batch-union併用＋prefix IF整合を含めると中。大にはしない）。
- **ゲート条件**：
  - 共有対象がFP4グローバルKVのみであること（SWA FP8局所の共有・永続化は禁止）。
  - ブロック単位共有・ブロック単位CoWを崩さないこと（トークン単位部分共有の混入は設計改訂）。
  - 連続化コピーバック（全KVの連続再配置）をdecode本線に混入させないこと。
  - prefixヒット率・N共有時の実メモリ（`1×prefix＋N×suffix` への削減）をP3b内実測で示すこと（論文値転載で代替不可）。

## 3. GDN-2固定ステートはPagedAttention対象外

- **方針**：GDN-2線形層の再帰状態 `S_t` はPagedAttentionの管理対象外とする。除外は恒久的な設計除外であり、一時的未対応ではない。
- **理由**：
  - GDN-2（および縮退先KDA）の記憶は系列長Nに依存しない固定サイズ再帰状態（例 S=64KB/層。D7 G1の前提と同一）であり、「KVキャッシュ＋持越し状態」のうち増大するのはKV側のみである（`analysis/decoding-architecture.md` §1.2と同一）。可変長KVに対するページング（N増大への対処）は固定長Sには適用対象が存在しない。
  - 増分デコードの等価形が異なる：フル層は「KV追記＋全KV走査」、GDN-2は「S持越し＋1ステップ更新（`S_t = ...`）」である。前者はブロックテーブル＋gather走査で効くが、後者は単一Sの持越し＋G1融合（7→2パス）で効く。ページングをSに適用しても走査削減にならず、テーブル参照の固定費だけが残る。
  - prefix共有の粒度が合わない：GDN-2のprefix共有はSのコピー（64KB/層×20線形層≒1.3MB級のmemcpy）で足り、refcount＋CoW＋ブロックテーブルの機構要否がない。GRPOのN共有においても線形層側は「SのN分複製（有界・N倍してもN×1.3MB級）」で済み、KV側のような `1×prefix＋N×suffix` のページ共有を要しない。S複製のN倍化はKV微小（§5）の範囲内で吸収可能であり、ページ機構の導入根拠にならない。
  - チャンク形との混同防止：学習側チャンクWY（UT変換逆行列等）は学習エンジン側の責務であり、デコードCエンジンに混入させない（P3a §1ゲートと同一）。PagedAttentionをSに適用する議論はチャンク切出しの混入に繋がりやすいため、本書で明示的に禁止する。
- **未解決**：なし（除外自体は確定）。残るのはG1融合（S 5パス分＝320KB/層削減）の実測のみであり、それは `analysis/p3-inference-engine.md` §4の受入条件に属し本書では再定義しない。
- **相対コスト**：小（除外であり実装を生まない。コストはG1融合側に計上済みであり本書で重複計上しない）。
- **ゲート条件**：GDN-2のSをブロックテーブル管理下に置かないこと（混入は設計改訂）。線形層のprefix再利用はSコピー（memcpy有界）に限定し、ページ共有として計上しないこと。チャンク形をデコードCエンジンに混入させないこと（P3a §1と同一）。

## 4. P3c §4のGRPO前提への反映文案（本書内提示。`analysis/p3-training-scale.md`自体は触らない）

- 本節は文案の提示に留まる。`analysis/p3-training-scale.md` の編集・追記・改訂は行わない（厳守）。
- **反映文案（一文）**：
  - 「GRPOの生成N回はPagedAttentionによるprefix共有（プロンプト部のブロック共有＋継続部のCoW分離）を前提とし、共有なしのN倍KVを予算に計上しない。」
- **理由（文案に付随する根拠。本書の注意機構範囲内）**：
  - 共有なしでは生成N回分のKVがN倍化し、KV微小（CSA2＋FP4＋top-1024・window128・heavy128:1）の前提と `dense≦2GB／全体≦8GB` および `NVMe≦103MiB/token` の受入条件を同時に破る。N共有により `1×prefix＋N×suffix` に抑えることが、P3c §4の前提条件(ii)（生成N回分の計算・メモリ・KV予算でKV微小を維持）の成立条件になる。
  - 共有対象はFP4グローバルKVのみ（SWA FP8局所は共有・永続化しない。§2と同一）であり、GDN-2のSは対象外（§3と同一。SはN分複製で有界のため別途ページ機構を要しない）。
  - 本一文は順序（SFT→ORPO→SPO→GRPO＋RLVR）・順伝播回数（生成N回＋更新1回）・RLVRルールベース報酬のみのいずれも変更しない。前提の明確化のみであり、P3cの段階・ゲート定義の変更ではない。

## 5. KV圧縮（CSA2＋FP4）を捨てないこと（併用前提）

- **方針**：`analysis/kv-cache-design.md` P3-1〜P3-4を維持し、本書の両機構はその上に載る。内訳：(i)層間共有（Reindex/Reuse相当の静的割当。キャッシュ共有とindex再利用は分離可能。線形層を挟んだ共有可否は静的割当に含める）、(ii)FP4格納（E2M1・16ch毎E4M3スケール・RoPE適用後量子化・注意時オンフライ逆量子化。事前逆量子化不可。スケール同一キャッシュライン隣接。nibble pack）、(iii)SWA相当の局所KVはFP8維持＋SSD非永続＋Bounded Replay（末尾 n_win=128のみ再計算。永続は圧縮済みFP4グローバルKVのみ）。CED本体・HCA復活・indexer別圧縮経路（V4式）は非採用。indexer Kはmain KVからの投影（CSA2式）に統一。
  - FlashAttentionとの接点：タイル内に展開するのはオンフライ逆量子化後の値のみとし、格納・永続・ページの形式はFP4のままとする。タイル化が「非圧縮で計算しやすくするための展開」を恒常化しないこと。
  - PagedAttentionとの接点：ページ・共有・永続の単位は圧縮済みブロック（FP4＋隣接スケール）とする。非圧縮ブロックのページ化・共有は不可。indexerのTop-K選択（先頭Full層のtop-512＋8トークン単位の候補プール16,384、以降Reindexはプール内再スコア）はページ管理と直交し、ページ側で再定義しない（層間共有の静的割当に従う）。
  - フォールバック：QATなしFP4の精度リスクに対し層単位FP8フォールバックを必須付帯（P3-2）として残す。フォールバック層のページ形式はFP8ブロックとし、FP4ブロックと混在可能（テーブル側で形式ビットを保持）とする。V4.1形式（per-16 E4M3・global scaleなし）とNVFP4（per-64 UE8M0）混同の防止を維持する。
- **未解決**：QATなしFP4精度未検証（層単位FP8フォールバックの感度評価は学習設計側と連携。本書で合否を付けない）。線形層を挟んだ層間共有の静的割当はP3b内で確定する（kv-cache §3(a)の申送り先は本書であり、P3本体への申送りではない）。FP4/FP8混在時のタイル内2経路のCコスト未確定。
- **相対コスト**：中（FP4オンフライ逆量子化＋隣接配置＋RoPE後量子化の走査順維持が主。層間共有の静的割当自体は小。QATなし精度検証を含めると中。大にはしない。§1のタイル側コストとの二重計上はしない）。
- **ゲート条件**：
  - 非圧縮KVへの先祖返りは設計改訂（FP4格納＋オンフライ展開の既定を崩さないこと）。
  - 事前逆量子化の混入禁止、スケール別ページのメタデータ読み禁止、CED本体の混入禁止（kv-cache P3-1〜P3-4と同一）。
  - FP4導入は層単位FP8フォールバック付きであること（フォールバックなしの全面FP4化は禁止）。

## 参照URL一覧（本書の注意機構範囲に限定）

- https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention
- https://arxiv.org/html/2605.22791
- https://github.com/NVlabs/GatedDeltaNet-2
- https://github.com/FareedKhan-dev/kimi-k3-in-c
- https://www.alphaxiv.org/abs/2609.deepseek-v4-1-flash.pdf
- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- https://deepwiki.com/legend/deepseek-v4-vllm-patches/3-nvfp4-kv-cache-quantization
- https://github.com/danielwoz/vllm-dspark-nvfp4
- https://deepseekv4.space/blog/deepseek-v4-technical-report
- https://docs.sglang.io/docs/hardware-platforms/cpu_server
- https://github.com/sgl-project/sglang/blob/4a50cd78/sgl-kernel/csrc/cpu/mamba/fla.cpp

## 禁止事項の遵守記録

- P3本体への申送り：なし（本書はP3bの一部として完結。P3本体設計・P3c実装への申送り事項は置かない。§4は文案提示に留め `analysis/p3-training-scale.md` 自体には触らない）。
- 4文書混交：なし（MoE・語彙・学習スケール・デコード投機の再定義をせず依存輸入に留めた。注意機構のみ）。
- 実装：なし（ヘッダ新規作成・カーネル実装・ `kv_prefix.h` 作成のいずれもなし。前提IFの記述に留めた）。
- 重負荷実行：なし（既存計測値の再利用のみ。新規bench・DL・学習実行なし。`include/jimotono/` の目录確認およびglob確認のみ）。
- コミット：しない（作業ツリー内文書のみ）。
- 他wt接触・checkout/switch：なし（本wt・本branchのみ）。
