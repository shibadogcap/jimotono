# ARCHITECTURE.MD

## 1. 線形注意の選択

### 推奨：Gated DeltaNet-2

**理由**：
- Gated DeltaNet-2は、Gated DeltaNetとKDAの両方を一般化し、消去ゲート（erase gate）と書き込みゲート（write gate）をチャネル単位で分離
  https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention
- 1.3Bパラメータ、100B FineWeb-Eduトークンで、Mamba-2、Gated DeltaNet、KDA、Mamba-3を上回る総合結果
- 長コンテキストRULER needle-in-a-haystackで特に優位
- 固定サイズのランニングステートを保持し、KVキャッシュ不要
- SGLangがGDN、KDA、Mamba2-style SSMのCPUバックエンドをサポート

**性能劣化要因**：
- 線形注意の比率が低すぎるとKVキャッシュが増大、高すぎるとリコール性能が低下
- AVX-512のダウンクロック：ZMMレジスタ使用時に最大30%の性能低下
- チャンクワイズ並列アルゴリズムの実装ミスは致命的

## 2. MoE構造

### 推奨：細粒度MoE + 共有エキスパート

**パラメータ配分**：
- 総パラメータ：10〜20B
- dense部分（常時RAM）：3〜5B（埋め込み、共有エキスパート2つ、アテンション）
- ルーティングエキスパート：1エキスパート2〜4M × 2000〜4000個
- アクティブ：100M以下（top-8〜16）

**参照実装**：
- TinyMoE-100m：総100M / アクティブ22.5M（10層、8エキスパート中2活性化）
  https://huggingface.co/FlameF0X/TinyMoE-100m-2x8
- DeepSeek V4：1共有 + 384ルーティング、top-6活性化
  https://deepseekv4.space/blog/deepseek-v4-technical-report

**性能劣化要因**：
- エキスパート粒度が粗すぎると表現力が低下
- 共有エキスパートがないとルーティング予測ミス時に生成が停止
- StickyMoE損失なしではエキスパートスイッチ率が高く、SSD読み込みが頻発

## 3. KVキャッシュ設計

### 推奨：CSA（Compressed Sparse Attention）

**設計**：
- Sliding window：各トークンは直近128トークンにアテンド
- CSA（4:1）：学習済みゲート付きプーリングで4トークン→1に圧縮。オーバーラップウィンドウで境界を平滑化
- HCA（128:1）：グローバルコンテキスト用の重圧縮。128トークン→1
- Indexer：学習済みスコアリングでtop-1024の圧縮位置を選択
- Attention sink：学習可能なper-headバイアス
- 結果：V3.2比でシングルトークン推論FLOPs 27%、KVキャッシュ 10%
  https://deepseekv4.space/blog/deepseek-v4-technical-report

**性能劣化要因**：
- 圧縮率が高すぎると長文脈のリコール性能が低下
- Indexerのtop-k選択が外れると重要なKVを落とす
- FP4の精度不足で長文脈の性能が劣化

## 4. ルーティング設計

### 推奨：StickyMoE Soft-Hard Variant

**設計**：
- Soft consistency loss：隣接トークン間のルーティング変更にペナルティ。ステップサイズを制御
- Hard anchor constraint：ウィンドウ内の最初のトークンのゲート分布をアンカーとし、以降のトークンがアンカーから離れるほどペナルティを増加。半径を制御
- 結果：fine-grained smoothness と coarse-grained commitment の両立
  https://arxiv.org/html/2607.08780v1

**性能劣化要因**：
- Softのみでは長距離ドリフトが発生（小さなステップの蓄積でアンカーから遠ざかる）
- Hardのみではウィンドウ内のギザギザしたスイッチを防げない
- アンカーウィンドウサイズWの選定が重要