# RULE.MD

## 1. 決め打ち項目と動的項目の分類

| 分類 | 定義 | 例 |
|---|---|---|
| アーキテクチャ固定 | 学習開始後に変更すると再学習が必要 | 線形注意のタイプ、エキスパートの粒度、語彙サイズ |
| スケーリング則で決定 | 小規模実験から外挿可能 | 総パラメータ、アクティブパラメータ、学習率 |
| 実行時動的 | 推論時に負荷に応じて変化 | top-k、KVキャッシュの圧縮率、プリフェッチ深さ |
| 学習中動的 | 学習の進行に応じて変化 | ビット幅のカリキュラム、ルーティング温度、損失重み |

## 2. 変更手続き

### 2.1 アーキテクチャ固定項目の変更
- 再学習が必要。変更は学習の早い段階でのみ許容
- 変更する場合は、小規模アブレーション（1B以下）で検証してから本学習に反映

### 2.2 スケーリング則項目の変更
- 小規模実験（100M〜1B）から外挿。外挿が不確実な場合は、中間規模（3B〜7B）で検証
- 変更後は、学習率とバッチサイズを再調整

### 2.3 実行時動的項目の変更
- 負荷と帯域に応じて動的に調整。変更は即座に反映可能

### 2.4 学習中動的項目の変更
- 学習の進行に応じてスケジュール。変更のログを記録し、再現性を確保

## 3. ベンチマークと評価

### 3.1 必須ベンチマーク

| 指標 | 測定方法 |
|---|---|
| 推論速度 | N100、i5-8th、Ryzen AI 360でのtok/s |
| RAM使用量 | dense部分と全体のピーク値 |
| カーネル起動回数 | 1トークンあたりの回数 |
| 1トークンあたりNVMe読み込み | io_uring統計 |
| ルーティング予測精度 | top-1一致率 |
| スイッチ率 | 隣接トークン間のエキスパート変更頻度 |
| 日本語MT-Bench | GPT-4oとの比較 |
| SWE-bench Pro | コーディング性能 |
| ツールコール成功率 | 日本語Function Calling |

### 3.2 性能劣化の検知

- 帯域競合：ブラウザエディタ起動時のtok/s低下を測定
- NUMAペナルティ：マルチソケット環境でのクロスNUMAアクセス
- AVX-512ダウンクロック：ZMMレジスタ使用時の周波数低下
- SSDレイテンシ：コールドエキスパート読み込み時のスパイク
- 分岐ミス：perf statでbranch-missesを計測

## 4. 未解決課題

| 課題 | 必要な情報 |
|---|---|
| Gated DeltaNet-2の15Bスケール | 1.3Bまでの検証のみ |
| 16GB RAMでの総15B MoE学習 | 直接の実証なし |
| 日本語ツールコールRL | 日本語データでのGRPO検証は未報告 |
| シングルコアでの帯域飽和点 | 1コアでデュアルチャネル帯域をどこまで使えるか |
| CSAの日本語タスク性能 | DeepSeek V4での日本語評価は未報告 |
| 1誤差伝播あたり20Mサイクルの達成可能性 | 理論値のみ。プロトタイプで検証が必要 |

## 6. 並列開発の運用規則（worktree分離・2026-09-13制定）

背景：単一ディレクトリで4ブランチをcheckoutし合った結果、17:41〜17:55に約20回のcheckout往復が発生し、サブエージェント2つが同一シェルを取り合う競合を起こした。この再発を防ぐため以下を厳守する。

- 1ブランチ1worktree原則：並列作業は必ず別worktree。同一ディレクトリでの`git checkout`/`git switch`によるブランチ往復は禁止
- 対応表（固定）：`develop`→`/Users/shibadogcap/jimotono`、`feat/p1/lut-kernel`→`~/jimotono-wt-lut`、`feat/p1/gdn2-kernel`→`~/jimotono-wt-gdn2`、`feat/p1/moe-routing`→`~/jimotono-wt-routing`、`feat/p1/ssd-stream`→`~/jimotono-wt-ssd`
- サブエージェントへの指示はworktreeパス・ブランチ・ビルドdirを明示し、`checkout/switch`禁止・他wtへの接触禁止・ビルドdir分離（各wt内の`build-*`を使用し共有`build-dbg`を使わない）を毎回含める
- main wt（`/Users/shibadogcap/jimotono`）は`develop`専用の統合場所とし、P1個別実装は各wtでのみ行う
- worktree作成・破棄はROADMAP 2.3手順に従う（`git worktree add ~/jimotono-wt-<name> -b <branch> develop`、`git worktree list`で確認）

## 5. 参照すべき研究・実装一覧

| 領域 | 参照元 | URL |
|---|---|---|
| 線形注意スケーリング則 | Scaling Laws for Linear Complexity | https://aclanthology.org/2024.emnlp-main.916.pdf |
| Gated DeltaNet-2 | NVIDIA Research | https://research.nvidia.com/publication/2026-05_gated-deltanet-2-decoupling-erase-and-write-linear-attention |
| TinyMoE-100m | Hugging Face | https://huggingface.co/FlameF0X/TinyMoE-100m-2x8 |
| FRI-MxMoE | ACL 2026 | https://aclanthology.org/2026.acl-long.982/ |
| ZAYA1-8B混合精度 | Hugging Face | https://huggingface.co/OsaurusAI/ZAYA1-8B-JANGTQ_K |
| NeuroPrefetcher | ACM ICPP 2026 | https://github.com/nobeldhar/NeuroPrefetcher |
| StickyMoE | arXiv | https://arxiv.org/html/2607.08780v1 |
| DeepSeek V4 CSA | DeepSeek V4 Space | https://deepseekv4.space/blog/deepseek-v4-technical-report |
| T-MAC | arXiv | https://ar5iv.labs.arxiv.org/html/2407.00088 |
| nanoRL | GitHub | https://github.com/alex000kim/nanoRL |
| Colibri/Curie | GitHub | https://github.com/Scottcjn/colibri |
| llm-jp-tokenizer | GitHub | https://github.com/llm-jp/llm-jp-tokenizer |
| Japanese Function Calling Dataset | Hugging Face | https://huggingface.co/datasets/Tonari-no-usagi/Japanese_Function_Calling_Dataset |
| ES-MoE | KAIST | https://ina.kaist.ac.kr/projects/esmoe/ |