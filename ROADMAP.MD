# ROADMAP

## 0. 基本原則

- 工期は書かない。完了条件と依存関係のみで進捗を管理する
- 各Phaseは「完了条件」「並列実行可能」「ドキュメント更新」「敵対レビュー」で構成
- クロスプラットフォーム（x86_64 + ARM64、Linux/macOS/Windows）を常に維持
- 成果物はartifacts/に置き、gitにはマニフェストのみコミット

## 1. ブランチ戦略

### 1.1 長期ブランチ

| ブランチ | 役割 | 寿命 |
|---|---|---|
| main | 常にビルド可能。ベンチマーク合格済みのコミットのみ | 永続 |
| develop | 統合ブランチ。各Phaseの成果をマージ | 永続 |
| arch/ | アーキテクチャ確定前の探索ブランチ | Phase 3まで |
| engine/ | 推論・学習エンジンの開発 | 永続 |
| model/ | モデル定義・重み形式・トークナイザー | 永続 |
| data/ | データパイプライン・前処理 | Phase 4まで |
| bench/ | ベンチマーク・評価ハーネス | 永続 |
| docs/ | ドキュメント（AGENTS/ARCHITECTURE/DESIGN/RULE/ROADMAP） | 永続 |

### 1.2 短期ブランチ命名

```
例:
  feat/p1/gdn2-kernel
  perf/p2/io-uring-batch
  fix/p1/routing-predict-rollback
  bench/p3/active-params-sweep
  docs/p2/update-kernel-count-target
```

### 1.3 保護ルール

- main：直push禁止。PR + 敵対レビュー1名以上の承認必須
- develop：直push禁止。CI green + セルフレビュー必須
- それ以外：自由。ただし毎日developにリベース

## 2. ワークツリー配置

### 2.1 推奨レイアウト

```
~/moe-cpu/
├── main/              # mainのワークツリー（読み取り専用運用）
├── dev/               # developのワークツリー（統合作業）
├── wt-kernel/         # engine/配下のカーネル開発
├── wt-training/       # 学習エンジン開発
├── wt-model/          # モデル定義・量子化形式
├── wt-data/           # データパイプライン
├── wt-bench/          # ベンチマーク実行専用（クリーン状態維持）
├── wt-review/         # 敵対レビュー専用（対象PRをチェックアウト）
└── artifacts/         # git管理外。チェックポイント・量子化済み重み・LUT
```

### 2.2 ワークツリーの使い分け原則

- wt-bench は常にクリーン：ベンチマーク実行時はgit clean -fdxしてから計測。汚染されたワークツリーでの計測値は無効
- wt-review は読み取り専用：ビルドと実行は許可、コミットは禁止
- artifacts はgit外：チェックポイント、量子化済み重み、LUTバイナリはartifacts/に置き、gitにはマニフェストのみコミット
- 同一ブランチを複数ワークツリーでチェックアウトしない：gitの制約。並列開発は必ず別ブランチ

### 2.3 ワークツリー作成・破棄の規約

```bash
# 作成
git worktree add ../wt-kernel -b feat/p1/gdn2-kernel develop

# 破棄（マージ後）
git worktree remove ../wt-kernel
git branch -d feat/p1/gdn2-kernel
```

- 1ヶ月以上更新のないワークツリーは破棄候補としてレビュー
- 破棄前に未コミット変更がないかgit statusで確認

## 3. コミット規約

### 3.1 メッセージ形式

```
<type>(<scope>): <subject>

<body: 何をなぜ変えたか>

<footer: ベンチマーク、参照、Closes>
```

type：
- feat：新機能
- perf：性能改善（必ずベンチマーク数値を本文に含める）
- fix：バグ修正
- refactor：挙動を変えない整理
- bench：ベンチマーク追加・変更
- docs：ドキュメント
- chore：ビルド・CI
- arch：アーキテクチャ変更（再学習が必要な変更は必ずBREAKING CHANGE:を付す）

scope：kernel, routing, moe, attn, ssd, lut, train, data, bench, api

### 3.2 性能コミットの必須項目

```
perf(kernel): fuse swiglu gate/up/down into single kernel

- before: 3 kernels/token/层, 12.4 us
- after:  1 kernel/token/层, 4.1 us
- hw: N100, 1C1T, DDR5-4800
- commit: <hash>
- config: configs/bench/kernel_fusion.yaml
```

ベンチマーク数値のないperfコミットはreject。

### 3.3 アーキテクチャ変更コミット

```
arch(attn): switch linear attention from Mamba-2 to Gated DeltaNet-2

BREAKING CHANGE: requires full retraining from scratch.
- old: Mamba-2, 4:1 ratio
- new: GDN-2, 5:1 ratio
- rationale: RULER NIAH at 1.3B, 100B tokens
- ablation: results/ablation/gdn2_vs_mamba2.md
- ref: https://research.nvidia.com/...
```

BREAKING CHANGE:付きコミットはmainに直接マージ不可。arch/ブランチ経由。

### 3.4 コミット粒度

- 1コミット1目的。カーネル修正とベンチマーク追加は別コミット
- ビルド可能な単位でコミット。壊れた中間状態をコミットしない
- WIPコミットはdevelopにpushしない。ローカルでsquashしてから

## 4. モデル成果物の扱い

### 4.1 gitに入れるもの

- モデル定義（Cのヘッダ、構造体、設定ファイル）
- 量子化形式の仕様書
- トークナイザー定義（語彙ファイルはLFSまたは別リポジトリ）
- 学習設定（YAML、ハイパーパラメータ）
- ベンチマーク設定と期待値

### 4.2 gitに入れないもの（artifacts/へ）

| 成果物 | 保存先 | 命名 |
|---|---|---|
| チェックポイント | artifacts/ckpt/<run-id>/step-<n>.bin | run-idはYYYYMMDD-<hash> |
| 量子化済み重み | artifacts/quant/<model>/<scheme>/ | schemeはgdn2-4bit-mix等 |
| LUTバイナリ | artifacts/lut/<model>/<scheme>/ | |
| KVキャッシュダンプ | artifacts/kv/<run-id>/ | デバッグ用のみ |
| ルーティングトレース | artifacts/route/<run-id>/ | |

### 4.3 マニフェストコミット

成果物を生成したら、マニフェスト（JSON）のみgitにコミット：

```json
{
  "artifact": "ckpt/20250913-a1b2c3/step-50000.bin",
  "sha256": "...",
  "config_hash": "...",
  "commit": "a1b2c3d",
  "bench": {"n100_tok_s": 6.2, "ram_mb": 3800},
  "notes": "GDN-2 5:1, A100M, INT4 mixed"
}
```

これにより、どのコミットがどの成果物を生んだかが追跡可能になる。

## 5. ベンチマーク再現性

### 5.1 計測の前提

- クリーンワークツリー必須：wt-benchでgit clean -fdx後にビルド
- CPU固定：tasksetでコア固定、cpupowerで周波数固定（可能なら）
- N回計測：中央値とMAD（中央絶対偏差）を記録。外れ値は破棄
- 外乱記録：ブラウザ・エディタの有無、他プロセス、温度

### 5.2 ベンチマークコミットの形式

```
bench(p1): add GDN-2 kernel microbenchmark

- harness: bench/kernel/gdn2.c
- cases: seq_len {512, 2048, 8192}, batch {1, 8}
- metric: us/call, cycles/call
- baseline commit: <hash>
- results: results/bench/20250913-gdn2.md
```

### 5.3 回帰検知

- developへのマージ前に、前回のdevelopとの差分を必ず計測
- 5%以上の性能劣化はマージブロック。原因特定まで保留
- 劣化が許容される場合はperf-regression:タグを付けて記録

## 6. 敵対レビュー手順

### 6.1 トリガー

- mainへのPR
- BREAKING CHANGE:付きコミットを含むPR
- Phase完了時
- 性能回帰が5%以上のPR

### 6.2 手順

```bash
# wt-reviewでクリーンチェックアウト
cd ~/moe-cpu/wt-review
git fetch origin
git checkout <pr-branch>
git clean -fdx

# ビルド（クロスプラットフォーム）
make clean && make x86_64
make clean && make arm64

# ベンチマーク
make bench

# 主張の検証
# - コミットメッセージの数値と実測値の一致
# - 参照元の主張と実装の一致
# - エッジケース（空入力、最大長、量子化境界）
```

### 6.3 レビュー記録

レビュー結果はreviews/<pr-id>.mdに記録：

```markdown
# Review: PR #123 feat/p1/gdn2-kernel

## 検証項目
- [x] ビルド（x86_64, arm64）
- [x] ベンチマーク再現（主張: 4.1us, 実測: 4.3us, 差5%以内）
- [x] 参照元との整合（GDN-2論文の式と一致）
- [x] エッジケース（seq_len=1, seq_len=32768）

## 指摘
- チャンクサイズが32固定。論文は64推奨。理由をコメントに追記のこと
- AVX-512パスに_mm512_...のマスク処理漏れ

## 判定
条件付き承認（指摘2件の修正後）
```

### 6.4 レビューの独立性

- 実装者とレビュアーは別ワークツリーを使用
- レビュアーは実装者のブランチをマージせずチェックアウトして検証
- レビュアーはwt-review以外で作業しない（汚染防止）

## 7. マージ方針

### 7.1 マージ方法

| ブランチ | 方法 | 理由 |
|---|---|---|
| feature → develop | Squash merge | 履歴を線形に保つ |
| develop → main | Merge commit | Phase単位の履歴を保持 |
| hotfix → main | Squash merge | 最小差分 |

### 7.2 マージ前チェックリスト

- [ ] CI green（x86_64 + arm64）
- [ ] ベンチマーク回帰なし（5%以内）
- [ ] 敵対レビュー完了（必要な場合）
- [ ] ドキュメント更新（該当する場合）
- [ ] マニフェスト更新（成果物がある場合）
- [ ] BREAKING CHANGE:の有無を確認

### 7.3 マージ後の処理

- ブランチ削除（リモート・ローカル）
- ワークツリー削除（使用していた場合）
- developのmainへの反映タイミングはPhase完了時

## 8. Phase別のワークツリー運用

| Phase | アクティブワークツリー | 凍結するもの |
|---|---|---|
| 0 | wt-kernel, wt-bench | なし |
| 1 | wt-kernel, wt-bench, wt-model | wt-data |
| 2 | wt-training, wt-kernel, wt-bench | wt-data（Phase 4まで） |
| 3 | wt-model, wt-training, wt-bench, wt-review | なし |
| 4 | wt-data, wt-training, wt-bench, wt-review | wt-kernel（安定版） |
| 5 | wt-bench, wt-review, wt-kernel（最適化のみ） | wt-data |

凍結＝新規コミット禁止。読み取りとビルドのみ許可。

## 9. ナレッジ更新のタイミング

| トリガー | 更新対象 |
|---|---|
| 週次 | 文献調査メモ（knowledge/papers.md） |
| Phase完了 | ARCHITECTURE.MD, DESIGN.MD, RULE.MD |
| 性能回帰検知 | DESIGN.MDの目標数値、ROADMAPのリスク |
| 敵対レビュー | RULE.MDの未解決課題 |
| アーキテクチャ変更 | AGENTS.MD, ARCHITECTURE.MD, ROADMAP |
| 成果物生成 | マニフェスト、artifacts/README.md |

ナレッジ更新はdocs/ブランチで行い、developにPR。

## 10. Phase 0: 基礎学習と環境構築

### 完了条件
- llm.cをビルドし、GPT-2のforward/backwardを理解
- T-MACのLUTカーネルを読解
- Gated DeltaNet-2の論文と実装を読解
- クロスプラットフォームビルド（x86_64 + ARM64）の検証

### 並列実行可能
- 文献調査（T-MAC, Gated DeltaNet-2, StickyMoE, NeuroPrefetcher）
- ベンチマーク環境のセットアップ（N100, i5-8th, Ryzen AI 360）

### ドキュメント更新
- knowledge/papers.md の初期版

### 敵対レビュー
- Phase 0終了時に「アーキテクチャ選択の妥当性」をレビュー

## 11. Phase 1: 推論エンジン

### 完了条件
- CでGated DeltaNet-2カーネルを実装
- MoEルーティング（top-k + 共有エキスパート）を実装
- T-MAC方式のLUTカーネルを実装
- SSDストリーミング（io_uring + O_DIRECT）を実装
- N100でINT4量子化した小型MoEを動作確認

### 並列実行可能
- LUTカーネルとSSDストリーミングは独立に開発可能
- ルーティング予測ヘッドはMoE本体と並行開発

### ドキュメント更新
- Phase 1完了時にARCHITECTURE.MDを更新
- ベンチマーク結果をDESIGN.MDに反映

### 敵対レビュー
- 「カーネル起動回数50回/token以下」の達成を検証
- 「1トークンあたり103 MiB以下」のNVMe読み込みを検証

## 12. Phase 2: 学習エンジン

### 完了条件
- Cでforward/backwardを実装
- 勾配チェックポインティング + 8bitオプティマイザを実装
- ES-MoEのSSDオフロード機構を実装
- StickyMoE損失（soft + hard）を実装
- 総1BクラスのMoEで学習パイプラインを検証

### 並列実行可能
- カーネル融合とSSDオフロードは独立に開発可能
- 混合精度学習とLUT-aware学習は並行開発

### ドキュメント更新
- Phase 2完了時にDESIGN.MDの目標数値を実測値で更新

### 敵対レビュー
- 「16GB RAMで学習可能か」の検証
- 「1誤差伝播あたり20Mサイクル以下」の達成を検証
- 「勾配チェックポインティングのメモリ削減効果」を測定

## 13. Phase 3: モデルアーキテクチャ

### 完了条件
- 総10〜20B / アクティブ100MのMoEを構築
- Gated DeltaNet-2の線形：フル比率をアブレーションで決定
- CSA（4:1圧縮 + top-1024選択）を実装
- エキスパート内混合精度（down=4bit, gate/up=2bit）を実装
- 小規模データ（10Bトークン）で学習

### 並列実行可能
- アーキテクチャ探索とデータパイプラインは並行
- 混合精度戦略とルーティング設計は独立

### ドキュメント更新
- ARCHITECTURE.MDを確定版に更新
- RULE.MDの未解決課題を更新

### 敵対レビュー
- 「Qwen3.8-27B相当の性能」が達成可能かレビュー
- スケーリング則の外挿が妥当か検証

## 14. Phase 4: 大規模学習とRL

### 完了条件
- LLM-jp Corpus v4でCPT
- Japanese Function Calling DatasetでSFT
- ORPOまたはSPOで選好学習
- GRPO + RLVRで推論強化
- 日本語MT-Bench、SWE-bench Pro、ツールコール成功率を評価

### 並列実行可能
- CPTとSFTは順次だが、RLはSFT完了後に並行開発可能
- 評価パイプラインは学習と並行

### ドキュメント更新
- 全ドキュメントを最終版に更新
- ベンチマーク結果をROADMAPにフィードバック

### 敵対レビュー
- 「日本語MT-Bench GPT-4o以上」の達成を検証
- 「ツールコール成功率90%以上」の達成を検証

## 15. Phase 5: 統合と最適化

### 完了条件
- OpenAI互換APIサーバーを実装
- エンドツーエンドでN100/Ryzen AI 360で動作確認
- ブラウザエディタとの共存テスト
- ドキュメントの最終化

### 並列実行可能
- APIサーバーと最適化は並行
- クロスプラットフォームテストは独立

### ドキュメント更新
- 全ドキュメントの整合性確認
- ナレッジベースの最終更新

### 敵対レビュー
- 全成功基準の最終検証
- クロスプラットフォーム（x86_64 + ARM64、Linux/macOS/Windows）での動作確認
