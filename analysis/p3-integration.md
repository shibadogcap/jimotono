# P3統合（P3d・設計＋ロードマップのみ）

- branch: `docs/p3/integration`（本wtのみ。コミットしない）
- 性質：文書のみ。実装禁止、重負荷実行なし。本書はP3a/b/cの統合記録であり、新規設計の着手ではない。
- 前提：`analysis/p3-architecture.md`（P3a）を最優先前提とし、P3b（`analysis/p3-inference-engine.md`）・P3c（`analysis/p3-training-scale.md`）と矛盾させない。矛盾発見時は本書を直さず設計会議へ申送る（Phase Gの再拡大は禁止）。
- コスト表記：相対コストは Phase G Step 4 micro（`jt_gemm_mat_f32`、Mr=12×Nr=4）を「中」とした相対（小＜中＜大）。絶対人月・絶対期間は記さない（P3a/b/cと同一）。
- 数値の使い分け：帯域上限は公称値（N100 8B／Ryzen 2.6B）を設計目標に使わず、保守値（N100 約1.3B・Ryzen 約0.4B）を上限目安に使う（P3a §4と同一）。

## 0. 事前確認結果

1. `analysis/p3-architecture.md` は存在し、主要決定は完備（中断不要）：GDN-2（§1）・比率4:1〜5:1（§2）・MoE細粒度2〜4M×数千・top-8〜16・共有2（§3）・active A100M基準（§4）・語彙V=48,588＋factorized head・untie既定（§5）・総10〜20B（§6）。
2. P3bの3予測器先読みの設計根拠は本書§2および `analysis/p3-inference-engine.md` 追補Aに明確化した（1予測器案との比較・帯域コスト・複雑性・フォールバック関係）。
3. P3cのRL順序の意図は `analysis/p3-training-scale.md` 追補Aに明確化した（ORPO＋SPO両用は誤記ではなく役割分離、SimPOはORPO代替、実験計画とパイプライン設計を分離）。

## 1. 横断論点

### 1.1 アーキテクチャ ↔ 推論（P3a §1・§2・§3 ↔ P3b §1・§2・§4）

- GDN-2のdecode逐次再帰／学習チャンク分離はP3a §1・P3b §2・`analysis/decoding-architecture.md` §4で一致（チャンク形のデコード混入禁止、bit同一契約 `-ffp-contract=off` 相当、3段階ゲート検証踏襲）。
- 比率4:1〜5:1（P3a §2）はフル層KV（CSA2＋FP4）の適用対象を全層の約1/5に限定する（P3b §1）。線形層は固定サイズ再帰状態でKV不要のため競合しない。
- MoE top-k 8〜16・共有2・cap=1.5・renormalize有・counting sort決定論的順序（P3a §3）はdecode先読み（P3b §2）のbatch-union・pin前提と直結する。top-k・容量・粒度の変更は設計改訂。
- G1融合（S 7→2、320KB/層・20線形層で6.4MB/token級）はGEMM化後の帯域残渣としてdecode実装の受入条件（P3b §4、P3a §1）。GEMM化（トークンバッチ、AI 0.5→23）の効果に上乗せ計上しない。

### 1.2 学習規模 ↔ サイズ（P3c §1・§3 ↔ P3a §6・§4）

- 段階は S1＝1B確立（0.712B構成・active 96M・dense 69.6MiB・SSD 264MiB）→ S2＝10B拡大（下限10B、P3a範囲内で単点固定はP3本体設計）→ S3＝LLM-jp Corpus v4本格（ja_wiki subset→SwallowCode-v2→JFCD順）に固定（P3c §3）。
- 28M常駐域ではoffload発動禁止（ckpt/esmoe微小 0.3%以下実測）。総10〜20Bの常駐不能域でのみ粒度ポリシー（安価recompute／重量offload＋fraction予約）が発動する（P3c §1、P3a §6）。
- S2学習の単機（16GB）成立は粒度ポリシー＋8bit optimizer（1/3.8）＋MEFT疎更新の併用が条件。N100でのS2学習実実行は不可（step時間未実測のため予測不可と明記）。
- S1推論fetch約16MiB/tokenは103MiB基準の内側に入り得る水準（未計測要素多し）。S2推論可否はP3a §4の条件付き選択肢則（η≧0.3・KV微小・未縮小head除去済み・NVMe内収載の実測）に従う。

### 1.3 語彙 ↔ active（P3a §5・§4 ↔ P3b §3・§5）

- 語彙V=48,588維持、d_model作業仮定1024、headはfactorized（1024→256→48,588、k=256固定）・emb full＋head factorizedのuntie既定・INT8（P3a §5）。head単体49.75M→12.70M（約3.92×減、−37.1MB/token）、語彙全体約62.45M。
- A100M基準の内訳目安：100M INT4≒50MB＋縮小head INT8約12.7MB＋KV微小≒63〜100MB/token（P3a §4、P3b §5）。AGENTS.md §8のN100 5tok/s（0.5GB/s）・Ryzen 30tok/s（3.0GB/s）はいずれも保守値に対してマージンがあり帯域は律速でない。律速候補は起動固定費（≦50/token）・NVMe IOPS/遅延（≦103MiB/token）・スレッドsync逓増。
- head疎化の実装はP4に記録のみ（着手禁止）。T-MAC LUTはGEMM体制（prefill・学習）の副線に限定しdecode GEMVの本線にしない（P3b §3、P3c §1）。

### 1.4 RL ↔ ツールコール（P3c §4 ↔ AGENTS §4.3・§5・§8）

- 順序：(1)SFT → (2)ORPO（SFTと同時・1回）→ (3)SPO（過最適化抑制・1回）→ (4)GRPO（KL削除・生成N回＋更新1回）＋RLVR（ルールベース報酬のみ）。SimPOはORPO代替、三重同時適用禁止（P3c §4＋追補A）。
- 前提：ORPO/SPOはSFTデータ＋選好ペア（JFCD等のツールコール選好を含む）が前提。GRPO/RLVRは (i)コード実行サンドボックス＋成否判定器、(ii)生成N回分の計算・メモリ・KV予算（CSA2＋FP4微小維持）、(iii)W8A8本線確定が前提。GRPO先行開始禁止。
- 成功基準（MT-Bench・SWE-bench Pro 50・ツールコール成功率90%）の合否はP3本体評価に委譲し、本書・P3cで付けない。

## 2. P3a/b/c間の矛盾・不整合の列挙（本書を直さず申送り）

| # | 箇所 | 内容 | 扱い |
|---|---|---|---|
| C1 | 比率 | AGENTS §2.1は4:1〜7:1、P3a §2は4:1〜5:1に狭窄 | P3aを前提に維持。AGENTS表記更新はP3本体設計へ申送り（本wtでは変更しない） |
| C2 | tied/untie | AGENTS §2.2はtied embeddings、P3a §5はuntie既定（factorized headがtyingを破るため） | P3aを前提に維持。AGENTS表記更新はP3本体設計へ申送り（本wtでは変更しない） |
| C3 | 予測器数 | AGENTS §3.2は単一予測器、§4.2は3予測器和集合（S∪L∪M） | P3b §2＋追補Aで解決：既定は和集合、単一はフォールバック／対照。不整合としては解消済み、AGENTS両記述の整理はP3本体設計へ申送り |
| C4 | RL表記 | ROADMAP Phase 4は「ORPOまたはSPO」、P3c §4はORPO→SPO順次（SimPO代替） | P3c追補Aで意図明確化（両用は誤記でない）。ROADMAP改訂（本タスク）で順次表記に統一する |
| C5 | データ順序 | ROADMAP Phase 4はCPT→SFT→RL、P3c §3はja_wiki subset→SwallowCode-v2→JFCD順の段階投入 | ROADMAP改訂でP3c順序に統一する（CPT一括688Bは単機不可のため） |
| C6 | active選択肢 | AGENTS §1はactive 100M固定、P3a §4はA300-500M条件付き選択肢を併記 | P3a則（既定100M、選択肢は(i)–(iv)＋η実測の設計改訂でのみ）を維持。不整合ではなく条件付き記述として申送り |
| C7 | LUT位置づけ | AGENTS §3.3はLUT一般記述、P3b §3・P3c §1はGEMM体制副線に限定（decode GEMV本線化禁止） | P3b/c則を維持。AGENTS記述の限定注記はP3本体設計へ申送り |

## 3. 未解決課題の集約（P3本体設計への入力。優先度順ではない）

- GDN-2学習カーネル（チャンクWY非対称erase因子＋gate-aware backward）のC学習エンジン側コスト未確定（P3a §1・P3b §2・P3c §1の共有申送り）。
- G1融合の削減トラフィック実測未実施（decode受入条件。P3b §4）。
- QATなしFP4 KVの精度未検証（層単位FP8フォールバック必須付帯。P3a §1・P3b §1）。
- INT2 gate/up・INT4 downの精度実測未実施、HGQ-LUT学習→推論コンパイル検証未実施（P3a §6・P3b §3）。
- ルーティング予測99%以上・スイッチ率59%減・ミス3.92倍減の自前実測未達。Delta 82〜85%・lookahead 71.6%は他系値で自系再取得要（P3b §2）。pin-hit率の自前計測要（P3c §2）。
- η実測未実施（STREAM未計測±30%）、単ch・小GEMV・競合時下振れ、長文脈・大バッチKV加算、O_DIRECT/SSD待ち混入、スレッドsync逓増（P3a §4・P3b §5・P3c §5の共有）。
- bwd効率がfwdの約1/5の原因未特定、1B学習1ステップ時間未実測（regime差のため予測不可）、lrスケジュール未実装（P3c §5・§3）。
- 10Bの層数・expert数割付け詳細、offload発動点・fraction既定値、sidecar記録フォーマット詳細、O_DIRECTドライブ別実測、`.coli_kv`相当の学習時要否（P3c §1–§3）。
- 5:1端の長文脈検索劣化定量、CSA層間共有の静的割当確定、DRAM一時プールサイジング、V4.1/NVFP4混同防止（P3a §2・P3b §1）。
- 日本語選好ペア調達量、報酬ハック対策、SPO先行効果、生成N回時のoffload・pin競合、RL開始段階（S1後半〜S2提案）の確定（P3c §4）。
- NEON実機・AVX-512移植未実施（AVX2 Mr=12×Nr=4のみ確定）、variance-aware capはsteady-state>5%時の再評価条件として残置（P3a §3）。
- Corpus取得・ライセンス・前処理（`.jtdp`再利用、C encoder書換えは訓練クリティカルパス外）未着手。100MB超DL未実行を維持（P3c §3）。

## 4. P3実装順序（ROADMAP改訂の根拠。本書は方針のみ）

- P3a（前提凍結）→ P3b（推論設計）→ P3c（学習スケール設計）→ 本統合（P3d）の順で設計を固定し、実装は S1＝1B確立を最初に実実行する。S1 exit通過後にS2＝10B拡大、S2 exit通過後にS3＝Corpus本格へ進む（段階改訂は設計改訂扱い）。
- ゲートは Phase GのG1–G4（`analysis/phase-g-summary.md` §1）を流用する：G1 bit一致（perm/off/dropマスク・loss相対差≦1e-6）、G2関数等価性（最終val差≦1%・勾配±10%・drop率は§1.2トリガ分離）、G3同一関数内最適化（§5.2維持・G2基準流用禁止）、G4 routing結合発散枠（4項目：構造等価・val≦1%・飽和<0.3%・性能目標、train loss不使用）。Step 4 microを新ベースラインに固定し回帰判定に使う。
- P4/P5は再定義する：P4＝10B級拡大＋データ本格＋RL順次（ORPO→SPO→GRPO＋RLVR）＋評価、P5＝OpenAI互換API＋エンドツーエンド動作＋最終文書化。詳細はROADMAP改訂を参照。

## 参照

- `analysis/p3-architecture.md` §0–§6
- `analysis/p3-inference-engine.md` §0–§5＋追補A
- `analysis/p3-training-scale.md` §1–§5＋追補A
- `analysis/phase-g-summary.md` §1–§6（G1–G4・Step 4 micro新ベースライン）
- `analysis/p3-design-inputs.md` §(a)–§(c)、`analysis/decoding-architecture.md` §1–§4、`analysis/kv-cache-design.md`、`analysis/active-upper-bound.md`、`analysis/vocab-budget.md`、`analysis/roofline.md` D4–D7、`analysis/f2-breakdown.md`、`analysis/training-speedup.md` P0/P1、`analysis/tokenizer-gap.md`
- AGENTS.md §1・§2.1・§2.2・§3.2・§4.2・§4.3・§5・§8
