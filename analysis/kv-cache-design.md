# KVキャッシュ設計 — DeepSeek V4.1 Flash (CSA2 + FP4) 調査

- branch: `docs/p3/survey-kv-decode`（本wtのみ。コミットしない）
- 位置づけ: P3設計入力。実装着手は禁止（調査＋文書のみ）。
- 前提: JIMOTONOは線形注意（Gated DeltaNet-2）:フル = 4:1〜5:1、フル層のみKV要（AGENTS.md §2.1）。

## 結論（先に）

1. **CSA2の層間共有＋FP4 KV＋SWA Bounded Replayは、我々の比率と互換。フル層（全層の約1/5）にのみ適用すればよい。** 線形層は固定サイズ状態でKV不要のため、そもそも競合しない。
2. **CED（Causal Encoder-Decoder）本体はP3で採用しない。** アーキテクチャ全体の共同設計＋再学習が前提であり、借りるべきはキャッシュ技術（CSA2/FP4/Replay）のみ。
3. **FP4 KVのC実装は「格納FP4・注意時にオンフライ逆量子化・16ch毎E4M3スケール隣接配置」**を推奨。事前逆量子化は帯域利得を消すため不可。QATなしでの精度は未検証のため、**FP8フォールバックを層単位で残す**（ARCHITECTURE.md §3の「FP4の精度不足」懸念と整合）。
4. **SWA Bounded Replayは採用推奨。** 中断再開時に直近128トークンのみ再計算し、SSD永続化は圧縮済みグローバルKVのみ。永続容量は約1/8。

## 1. 調査対象の特定

- モデル: DeepSeek-V4.1-Flash（552B backbone MoE、40層CED = encoder 20 + decoder 20、prefill 8B / decode 16B active、最大1Mトークン、45Tトークン事前学習）
  https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- 論文: "DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression"
  https://www.alphaxiv.org/abs/2609.deepseek-v4-1-flash.pdf
- vLLM側のFP4実装参照（V4世代・Blackwell SM120向けNVFP4）:
  https://deepwiki.com/legend/deepseek-v4-vllm-patches/3-nvfp4-kv-cache-quantization
  https://github.com/danielwoz/vllm-dspark-nvfp4
- 解説（数値の裏取り用）:
  https://dowithsudo.com/blog/deepseek-v4-1-flash-architecture/
  https://dev.to/prabhakar_chaudhary_7afe4/deepseek-v41-flash-how-a-causal-encoder-decoder-architecture-cuts-agent-memory-costs-by-75-ecp

## 2. CSA2の圧縮メカニズム

CSA2（Compressed Sparse Attention 2）は、KV削減の3次元（エントリサイズ・系列長・層深さ）を同時に攻める。V4のCSA+HCAハイブリッドと異なり**pure CSA2**。

### 2.1 層モード3種（静的割当）

| モード | main KV | indexer K | Top-K indices |
|---|---|---|---|
| Full | 自層で新規計算 | main KVから投影（新規） | 自層indexerで新規選択 |
| Reindex | 前方Full層のものを再利用 | 再利用 | 自層クエリで再スコア→新規選択 |
| Reuse | 再利用 | 再利用 | 前方層の選択をそのまま再利用 |

- キャッシュ共有（main KV＋indexer K）とindex再利用（Top-K）は**分離可能**。Reindexは共有しつつ選択だけ変える中間点。
- V4のCSAからの簡素化: 圧縮のオーバーラップ除去、絶対位置埋め込み除去、indexer Kは隠れ状態からの別圧縮経路ではなく**main KVからの投影**。実装容易＋学習効率向上。
- Reuse層のデコードは融合11 CUDAカーネルのみ（indexer計算ゼロ）。

### 2.2 Hierarchical Sparse Indexer（decoder側）

- 先頭Full層がTop-512選択と同時にブロック単位候補選択（8トークン=1ブロック、 top-2048ブロック → **16,384候補プール**）を構築。
- 以降のReindex層は全コンテキスト走査せず**候補プール内のみ**再スコア。層深方向のindexer計算量はコンテキスト長に依存せず定数 bounded。
- 890 bytes/token（HBM常駐グローバルKV）= V4-Flash（3,514）の約1/4、V1比437×。

### 2.3 FP4 KVキャッシュ

- OCP標準MXFP4、**E2M1＋16チャネル毎に1つのE4M3スケール**。第二レベルglobal scaleは省略（NVFP4のper-64 UE8M0方式とは異なる）。
- 省略の根拠（論文§2.4.4）: 表現上限 448×6=2688 に対し、KV latentの上限はRMSNorm後 √512≈22.6、学習時観測最大≈10。測定可能な精度低下なし。
- **RoPE適用後に量子化**（RoPE前量子化の精度利得は微小、デコード時オーバーヘッド増のため）。non-RoPE/RoPE成分は同一フォーマット。
- **SWA KVはFP8のまま**（量子化感度が高い）。V4ですでにindexer Q/KにQAT適用済み、V4.1でmain KVに拡張（post-trainingでQAT導入）。
- FP8 main KV比で格納量ほぼ半減（HBM＋SSD両方）。

### 2.4 SWA Bounded Replay

- SWA KVは**SSDに永続化しない**。10%ホストDRAMの分散プールに分単位TTLで一時的保持。
- SSD永続は圧縮済みFP4グローバルKVのみ（LRU、最低72h保持）。永続容量はV4-Flash比**約1/8**。
- Globalヒット＋SWAミス時はキャッシュ済みprefixの**末尾n_win=128トークンのみリプレイ**して局所SWA状態を再構築→デコード開始。SWAの実効受容野は層深方向に複利で効かないことが根拠。

## 3. 評価

### (a) 線形注意比率4:1〜5:1との互換性

- **互換。フル層のみ適用で妥当。** 線形層（GDN-2）は固定サイズ再帰状態でKVを持たないため、CSA2/FP4/Replayの適用対象は全層の約1/5（4:1時20%、5:1時約17%）に限定される。削減技術は残り1/5に**乗算的**に効く。
- 層間共有（Reindex/Reuse）の適用先もフル層間に限定でき、線形層を挟んだ共有可否はP3設計時に静的割当として決めればよい（CSA2自体が静的モード割当なので相性が良い）。
- **CEDは非採用。** encoder最終状態からdecoderグローバルKVを投影する設計は、prefill活性8B/decode活性16Bという非対称活性と一体であり、我々の均一層スタック＋SSDストリーミング設計とは前提が異なる。借用対象はキャッシュ技術のみとする。
- 参考: Kimi K3（69 KDA : 24 MLA ≈ 2.9:1）、LFM2-24B-A2B（30 conv : 10 attn = 3:1）はいずれも我々よりフル寄り。我々の4:1〜5:1は線形寄りの攻めた比率であり、フル層側の圧縮（本調査）が保険になる関係。

### (b) FP4 KVのC実装方針

- **オンザフライ逆量子化を推奨（事前逆量子化は不可）。** 事前逆量子化はメモリ/flow帯域をFP8同等に戻し、FP4格納の意味を消す。注意計算時にレジスタ上で E2M1→FP16/FP8 展開し、既存の内積パスを再利用する（vLLM NVFP4パッチが E2M1→FP8 nibble展開後にFP8 MMAパス再利用で値をlossless再現する手法と同思想）。
- **レイアウト:** 16chグループ毎にE4M3スケール1個を**同一キャッシュラインに隣接配置**（スケール別ページのメタデータ読みを避ける。DESIGN.md §5の方針と整合）。E2M1はnibble pack（2要素/byte）。
- **逆量子化:** 16エントリLUT×スケール乗算。AVX2/NEONではgatherまたはスカラLUT＋FMA。RoPE適用後量子化のため、格納順＝注意時走査順にし、転置を発生させない。
- **QATの扱い:** V4.1はpost-training QATが精度保持の前提。我々はP3でQATなしのため、**層感度ゲート（層単位FP8フォールバック）**を設計に残す。SWA相当の局所KVはFP8維持（論文と同一判断）。
- **量子化粒度の差異に注意:** V4.1形式（per-16 E4M3、global scaleなし）とvLLM NVFP4（per-64 UE8M0＋フッタ360B/トークン内訳: NoPE 224B＋RoPE BF16 128B＋scale 8B）は別物。C移植の参照にするのは**カーネル構造（fused quant/store、展開後演算再利用）**であり、フォーマット定数はV4.1側に従う。

### (c) SWA Bounded Replayコスト

- リプレイ量: 中断再開1回あたり **末尾128トークン × フル層のみ** の再計算。フル層が全層の1/5のため、再開コストは128トークンprefillの約1/5相当。長セッションに償却すれば無視可能。
- 必要機構: 直近128トークンのリング保持（トークンID列のみで足り、再計算でSWA状態復元）。DRAM一時プールのサイジングはP3で算定（同時セッション数×128×フル層SWA状態量）。
- 永続SSD量の目安式（トークンあたり）: `フル層数 × 共有後KVエントリ × 0.5B（FP4）`。層間共有で「フル層数」は実効的にさらに減少（Full層分のみ）。1M級コンテキストでもHBM常駐分はGiB桁に収まる設計（V4.1実績: 1Mトークン≈0.87GiB）。

## 4. P3推奨（優先度順）

1. **P3-1（推奨）:** フル層KVに「層間共有（Reindex/Reuse相当の静的割当）＋FP4格納（E2M1/per-16 E4M3、RoPE後量子化）＋オンフライ逆量子化」を採用。SWA相当の局所KVはFP8維持＋SSD非永続（Bounded Replay、n_win=128）。
2. **P3-2（必須付帯）:** FP4は層単位FP8フォールバック付きで導入（QATなしの精度リスク対策）。感度評価はP3の学習設計側と連携。
3. **P3-3（非採用）:** CED本体、HCA復活、indexer別圧縮経路（V4式）は採用しない。indexer Kはmain KVからの投影（CSA2式）に統一。
4. **P3-4（設計制約）:** スケールは重みと同一ページ/同一ライン配置（DESIGN.md §5）。`restrict`＋`goto cleanup`＋SIMD `#ifdef` ガードはAGENTS.md §7.1に従う。

## 5. 参照URL

- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
- https://www.alphaxiv.org/abs/2609.deepseek-v4-1-flash.pdf
- https://deepwiki.com/legend/deepseek-v4-vllm-patches/3-nvfp4-kv-cache-quantization
- https://deepwiki.com/legend/deepseek-v4-vllm-patches/3.2-kv-cache-write-path:-quantization-and-storage
- https://github.com/danielwoz/vllm-dspark-nvfp4
- https://dowithsudo.com/blog/deepseek-v4-1-flash-architecture/
- https://dev.to/prabhakar_chaudhary_7afe4/deepseek-v41-flash-how-a-causal-encoder-decoder-architecture-cuts-agent-memory-costs-by-75-ecp
- https://deepseekv4.space/blog/deepseek-v4-technical-report （V4 CSAの前提）
