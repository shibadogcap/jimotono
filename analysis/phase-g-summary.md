# Phase G 総括 — MoE GEMM化のみ（P3a申送り用）

- branch: `docs/p3/architecture`（本wtのみ。コミットしない）
- 性質：文書のみ。実装禁止、重負荷実行なし。本書は既存記録の総括であり新規測定なし。
- スコープ：**Phase G = MoEトークンソート＋エキスパート単位バッチングのみ**。
  KV近似・デコード変更・学習高速化全般・factorized head・G1融合（GDN-2融合）・量子化（INT4/INT2/INT8・T-MAC LUT・HGQ-LUT含む）は非スコープ（`analysis/gemm-design.md` 冒頭宣言を維持）。
- 本書の結論をP3本体設計に混ぜない。P3の前提は `analysis/p3-architecture.md`（P3a最優先文書）に一本化する。
- 数値の一次ソースは `analysis/gemm-design.md` §7・§10〜§13、`analysis/load-balancing-check.md`、`analysis/roofline.md` D4/D6、`analysis/f2-breakdown.md`。本書は要約＋承認記録であり、再測定・再解釈は行わない。

## 1. ゲート4分類（G1–G4）

`analysis/gemm-design.md` §5.4–§5.6 の分類を総括として再掲する。定義変更なし。

| ゲート | 対象 | 基準 | 適用Step | 結果 |
|---|---|---|---|---|
| G1 bit一致 | 関数変更なし・順序のみ | perm/off/dropマスクbit一致、loss相対差≦1e-6（§5.2） | Step 1（ソートのみ・GEMV維持） | 合格（§8.3：相対差2e-9） |
| G2 関数変更 | drop・renormalize・加算順序の変化を伴う | 関数等価性3項目：最終val差≦1%・収束単調性（事実記録）・勾配ノルム平均/分散±10%以内。drop率は基準から除外し§1.2トリガで評価 | Step 2（バッチfwd素朴GEMM）＋Step 3（バッチbwd） | 通過（§10.6・§11：200 steps val差0.29%→500 steps val差0.059%、勾配平均0.08%/分散3.11%） |
| G3 最適化（同一関数内） | 同一関数内での最適化 | §5.2維持（bit一致または相対差1e-6）。G2基準（1%）は流用しない | Step 4の単位・短期・drop=0域（テールG1 bit一致・batch fwd 1e-6・20 steps再走行stderr bit一致） | 通過（単位・短期・drop=0域。系統レベルdrop>0の200 steps最終値では不通過と記録しG4へ申送り。§12.3.1） |
| G4 最適化（数値等価だがrouting結合で発散） | 同一perm/drop/renormalize仕様の下でGEMM内丸め順序のみが変わり、重み→routing帰還で長時間軌道が発散し得る最適化 | 4項目：(1)構造的等価性（perm/off/drop bit一致・gate初期一致・テールG1・決定論性）、(2)最終val差≦1%、(3)増幅有界性（飽和<0.3%=完全通過・線形~0.8%=条件付き・指数数%超=不通過）、(4)性能目標（§6主判定2項目維持）。train lossは一切使わない | Step 4（f32混合マイクロカーネル `jt_gemm_mat_f32`）の長時間系統軌道 | **完全通過**（§13.3） |

- G3 strict（stepwise 1e-6×200 steps・drop>0）の不通過記録は維持する（§12.3.1）。超過値の許容判断ではなく、G4の枠組みで再評価することが承認済みである（§5.6末尾の承認記録）。
- Step 3とStep 4の直接比較は不可であることを明記する（Step 3素朴GEMMコードは現developに存在せず、`git checkout/switch`・他wt接触が禁止のため復元不可。§13冒頭）。代替として (a) Step 4 micro単独軌道安定性、(b) 同一ビルドsingleとのval差軌道、(c) 既知Step 2/3記録値との比較で評価した。

## 2. G4完全通過の承認記録

`analysis/gemm-design.md` §13（1000 steps、TinyStories micro vs same-build single＋既記録比較）の受領値のみで構成する。

- 条件：`train_tinystories --data data/tinystories_head16M.txt --steps 1000 --batch 4 --seq 128（T=512） --val-every 20 --patience 0 --max-val-pairs 2000 --aux-weight 0.01` を `--moe-batch`（micro）と単体路（single）で逐次・単一プロセス各1回。同一シード・同一データ順。dims d=64・layers=2・E=8・K=2・S=1・H=32・V=258（§10・§11と同一）。所要 micro 32.7s（30.58 steps/s）／single 149.8s（6.68 steps/s、比4.58xは参考値）。
- val差軌道（val_ce相対差のみ。trainは不使用）：100 step 0.31% → **200 step 0.38%（峰）** → 300 step 0.10% → 400 step 0.00% → 500 step 0.04% → 600 step 0.03% → 700 step 0.08% → 800 step 0.08% → **900 step 0.13%（飽和帯最大）** → 1000 step 0.08%（§13.1）。
- 形状判定：step200の0.38%を峰としてstep300以降は0.00–0.13%に収束し単調増加しない。線形漸増・指数発散の兆候なし。**増幅0.38%→0.13%飽和**として記録する。
- 4項目評価（§13.3）：(1)構造的等価性 pass（初step drop 159/2048がLB-check記録7.76%・§12.3.1と一致。テールG1・20 steps再走行bit一致は§12.1維持）、(2)統計的同等性 pass（最終val差0.08%≦1%）、(3)増幅有界性 pass（完全通過。飽和帯300–1000 steps最大0.13%<0.3%。峰0.38%は初盤高drop過渡1–100窓8.24%に対応する一過性）、(4)性能目標 pass（§12.3.2の記録を維持）。
- よって **G4増幅検証は完全通過**とする。train lossは判定に一切使っていない。
- 再現性の傍証（合否に使わない）：single step200 val 2.5039＝§10.1一致、single step500 val 2.3561＝§11一致、micro step200 val 2.5135＝§12.3.1一致、micro step500 val 2.3552＝§12.3.2一致。別ビルド・別日再走行でval完全再現（決定論性）。

## 3. Step 4新ベースライン固定

§13.4の宣言を受領し、以降の比較基準として固定する。

- **Step 4 micro（`jt_gemm_mat_f32`、Mr=12×Nr=4・f32蓄積＋f64加算）を以降の比較の新ベースラインに固定する。** Step 5以降の等価性・性能比較はStep 3素朴GEMM（コード現存せず）ではなく本Step 4 microを基準とする。
- 追跡基準値：最終val（1000 steps）micro 2.3187／single 2.3169（差0.08%）、飽和帯上限0.13%、所要 micro 32.7s（30.58 steps/s）／single 149.8s（6.68 steps/s）、drop合計2.83%・最終100 step平均1.86%。
- 回帰判定（micro比のsteps/s・最終val差・増幅形状）は本節値を追跡基準とする。Step 3比への差し戻しは行わない。

## 4. 主要数値（受領値の一覧。合否の再判定はしない）

| 項目 | 値 | ソース |
|---|---|---|
| steps/s倍率 TinyStories | micro 30.49 vs same-build single 6.71 = **4.54x**（Step 2比＝単体ループ比。同一ビルド・同一条件） | §12.3.2 |
| steps/s倍率 longrun | micro 4.748 vs same-build single 0.827 = **5.74x** | §12.3.2 |
| Step 3比回帰なし | TinyStories 30.49 vs naive batch 14.49＝2.10x、longrun 4.748 vs naive batch 1.14＝4.16x（いずれも回帰なし） | §12.3.2 |
| 理論天井比 | GEMMカーネル単体でFLOPs加重平均29.92 GFLOPS、天井（単一コア・mul/add分離ピーク51.2 GFLOPS）比 **58.4%**（>50%でpass）。形状別では(64,256,512)が45.7%と50%未達（Mr/Nr再調整はsoak改訂扱い） | §12.3.2 |
| AI fwd | routed平均（M_e=64、n=256・h=64）**14.22**。routed上限（cap=80）15.61≒15.6、密（M=512）23.27≒23、加重全体約18.7（ridge 21の約89%。帯域律速のまま）。**AIは診断指標であり主判定には使わない**（§6主判定は天井比＋3xのみ） | §6.1 |
| drop steady-state | 500 steps batch **最後100 step平均（401–500）3.34%**、合計3.86%（39512/1024000）。100 step窓：8.16%→3.44%→2.41%→1.94%→3.34%。§1.2トリガ（恒常的>5%）未達→Step 4可。予測2.5%以下は未達として記録（許容判断なし） | §11.3–§11.4 |
| G4 drop参考 | micro 1000 steps合計2.83%、最終100 step平均1.86%（§11予測2.5%以下に収まる）。合否に使わない | §13.2 |
| Step 1基準 | loss相対差2e-9（≦1e-6以内）。Step 2以降の追跡基準 | §8.3 |
| Step 3 baseline | 単体6.24 steps/s／バッチ14.49 steps/s＝2.32x。Step 4bの回帰なし判定の基準 | §7.1・§11 |

## 5. 設計決定（仕様として固定。ハイパーパラメータではない）

- **renormalize する**（旧「renormalizeしない」を撤回。G2改訂）。トークン毎kept重み和Sで `w' = w/S`（S>0時。S==0時寄与0）。`out_weights`はkeptについてrenormalize後値にin-place更新、droppedは元のまま残すが未使用（dropマスクでskip）。bwdはrenormalize後重みでsoftmaxヤコビアンを計算し、分母S経由2次項はstraight-throughとして無視。変更時は設計改訂とする（§1.2・§3.3・§5.5）。
- **cap=1.5を既定化**（旧1.25から改訂。調整可能範囲1.0–1.5の上限）。改訂理由：長時間走行で1.25は終盤6.64%とトリガ超過（§9.4）。longrun T=512・k=2・E=16でcap_e=96、TinyStories T=512・k=2・E=8でcap_e=192。メモリ増は+40KB級（longrun）・+1KB級（TinyStories）で予算内（§1.2）。
- **aux-weight μ=0.01を既定維持**（推奨範囲0.01–0.1、seq==1 legacy路はログのみ・grad適用はT路のみ）。LB-check（30 steps μ=0/0.01/0.1振り）で勾配伝播を確認済み（μ=0.1でdrop 7.3%→1.6%、amax約25%減、entropy上昇）。μ引上げ・cap変更等の設計改訂は本タスクでは行わない（§10.5・§11.4・`analysis/load-balancing-check.md` §4）。
- drop率・renormalize発火率はG2合否に使わず§1.2トリガ（steady-state判定。最後100 step平均）で評価する。トリガ（恒常的>5%）超過時のみcapacity見直しを検討し、自動調整はしない（§1.2・§5.4）。

## 6. 残課題（P3aへの申送り。Phase Gの再拡大はしない）

- **variance-aware cap未実施**：判定ルール（steady-state>5%かつμ適用確認済みなら(b) variance-aware cap提案）の条件未達（3.34%<5%）のため提案自体を行わない。実装なし・記録のみの条件にも該当しない（§11.4）。将来steady-stateが>5%に悪化した場合の再評価条件として残す。
- **NEON実機なし**：AVX2（Mr=12×Nr=4・acc12本・f32蓄積＋f64加算）のみ確定。AVX-512／ARM NEONへの移植は未実施。AVX-512は新規コードにintrinsicsなし・既存ガードは非活性のまま（§12・§12.4）。他ISAはレジスタ本数比例換算の方針のみ残す（§4.3）。
- **STREAM未計測**：eff_BW=29.4GB/sは推定（±30%）。絶対値外挿には±30%を見込む。端to端TinyStories天井比〜1%は小問題オーバーヘッド支配の診断記録であり、主判定はGEMMカーネル単体天井で評価したことを併記し隠蔽しない（§7.1.3・§12.3.2）。
- **FMA未使用**：bit同一要請のためmul/add分離を維持（FMAなし。ピークは半分目安51.2 GFLOPS単一コアで天井再計算）。FMA導入時は「FMA vs mul/add分離」の差分評価を別途要し、G3基準の流用禁止（§4.1・§5.3・§6.2）。
- **形状別未達の残置**：(64,256,512) 45.7%と50%未達（K=512のL3ストリーミング域）。Mr/Nr再調整はsoak改訂扱いとし場当たり調整しない（§12.3.2・§7 Step 5）。
- **G1融合・量子化・KV・decodeはPhase Gの残課題ではない**：G1（GDN-2融合）・量子化・KV・decodeは非スコープとしてP3本体設計へ申送り済み。本書での追加実行・再拡大は禁止する。

## 参照

- `analysis/gemm-design.md` §1.2（cap/renormalize）、§3（bwd）、§4（マイクロカーネル）、§5.4–§5.6（G1–G4）、§6（目標・AI 14.22・天井比・3x）、§7.1（Step 3 baseline 2.32x・AI診断）、§8.3（Step 1 2e-9）、§10–§11（G2fix・500 steps・steady-state 3.34%・Step 4可）、§12–§13（Step 4・G4完全通過・新ベースライン）
- `analysis/load-balancing-check.md`（μ感度・実装バグなし・LB修正なし・drop=0区間bit一致）
- `analysis/roofline.md` D4（AI 0.38–0.50・ridge 21・全カーネル帯域律速）、D6（GEMM化 AI≈23・expert固定トークン走査）、D5（LLCスラッシング棄却・帯域天井）
- `analysis/f2-breakdown.md`（bwd 46%支配・val償却・sync逓増。学習目標の基準値としてP3へ申送り）
- P3前提は `analysis/p3-architecture.md` に一本化（本書はPhase G総括でありP3設計は行わない）

## 7. 再測定ベースライン（-mavx2修正後。Stage 1の比較基準。既存値の書き換えなし）

- branch: `perf/p3/g-remeasure`（本wtのみ。コミットしない）。実装変更なし（測定＋文書追記のみ）。
- 背景：Stage0の `-mavx2` 欠落修正（swiglu_bwd 3.8x・moe_bwd 2.18x・bit同一）により§3–§4の数値が陳腐化。
  本節の値をStage 1以降の比較基準（新ベースライン）とする。§1–§6の既存値は書き換えない。
- 条件：本wt内 `build-rem/` のみ。`--no-sched` 付与（Stage0のlrスケジュール既定ONによる動力学変更を除外し
  旧条件と同一動力学で比較。既定ONではsingle 200 steps valが2.5039→2.9440に変わることを確認済み。
  所要はON/OFFで同一）。逐次・単一プロセス・実行前 `uptime` 確認（up 13 days、load約1.7–2.8）。
  詳細は `analysis/gemm-design.md` §14、`analysis/f2-breakdown.md` §7、
  `analysis/roofline.md` D4追記を一次ソースとする。本書は要約＋承認記録であり再解釈は行わない。

| 項目 | 新ベースライン（旧値） | ソース |
|---|---|---|
| steps/s倍率 TinyStories 500 steps | micro **31.25**（旧30.49） vs same-build single **6.76**（旧6.71）＝**4.62x**（旧4.54x） | §14.1 |
| steps/s倍率 200 steps | micro 31.75 vs single 6.78＝**4.68x** | §14.1 |
| Step 3比回帰なし | micro 31.25 vs naive batch 14.49＝**2.16x**（旧2.10x。回帰なし維持） | §14.1 |
| 理論天井比 | **58.4%維持**（核不変を確認。7形状再測−4〜+8%で一致。(64,256,512)の50%未達も維持） | §14.2 |
| 最終val（500 steps） | micro **2.3552**／single **2.3561**（いずれも旧値と完全一致） | §14.1 |
| 最終val（200 steps） | micro **2.5135**／single **2.5039**（いずれも旧値と完全一致） | §14.1 |
| F2内訳（代表6T） | step 1751ms（旧1886）・fwd 68ms・bwd 865ms・optim 225ms・sync 127ms・val償却444ms。最速点は2Tへ移動 | F2 §7 |
| 帯域参照 | STREAM Triad実測 **21.24GB/s**（旧推定29.4の−27.7%。±30%留保内）。bwd効率は実測比約22%（約1/4.5）として確定 | roofline D4追記 |

- ゲート判定は有効のまま（bit同一のため）。最終valの完全一致（micro 2.3552/2.5135・single 2.3561/2.5039）により
  G1–G4の合否（§1–§2）は揺らがない。再判定は行わない。
- Stage 1の比較は本節新ベースライン（`--no-sched`・固定lr系） against Stage 1改変で行うこと。
  既定スケジュールON系の動力学比較が必要な場合は200 steps参考値（single val 2.9440・micro val 2.9529・差0.30%）を起点とし、本節固定lr系と混ぜないこと。
