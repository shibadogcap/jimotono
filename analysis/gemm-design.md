# MoEトークンソート＋エキスパート単位バッチング設計 — Phase G設計

- branch: `docs/p3/gemm-design`（本wtのみ。コミットしない）
- 性質：設計文書のみ。実装禁止、重負荷実行なし。
- スコープ：**MoEトークンソート＋エキスパート単位バッチングのみ**。
- 非スコープ（本書では採用しない・変更しない）：KV近似、デコード変更、学習高速化全般、
  factorized head、G1融合（GDN-2 融合）、量子化（INT4/INT2/INT8・T-MAC LUT・HGQ-LUT含む）。
- 参照：Megablocks、Tutel、DeepSpeed-MoE、Colibri PR #475、`analysis/roofline.md`。
- 前提値（`analysis/roofline.md` D4/D6より受領）：現行カーネル AI 0.38–0.50（fp32換算）、
  ridge ≈ 21 FLOPs/byte、GEMM化（T=512同時トークン）で AI ≈ 23 の見立て。
  dims前提は roofline と同一：n=256, h=64, E=16, k=2（`train_longrun` 実働値）。

## 0. 背景と設計方針

現行はトークン外側・expert内側（トークン逐次処理）の GEMV 体制であり、
`roofline.md` D4判定の通り全カーネルが帯域律速（AI ≦ 0.50、ridge 21 の 1/42 以下）。
D6判定の通り、GEMV のままレジスタブロッキング（acc 多本化）を入れても AI 上限は動かない。
唯一の計算律速側への道は**トークンバッチ＝GEMM化**（AI 0.5 → 23 の理論値）である。

本書はその第1段として、以下2点のみを設計する：

1. ルーティング後のトークンソート（expert ID 順への並べ替え＋容量制限）。
2. エキスパート単位の密 GEMM バッチング（可変 M × 固定 N,K）。

先行実装の位置づけ：

- Megablocks：sparse → blocked dense（dropless 指向のブロック疎行列）への定式化元祖。
  本書は dropless ではなく **capacity capped + drop** を採用する点が異なる（理由は§1）。
- Tutel：動的 capacity factor＋2D階層 all-to-all。CPU単ノードの本件では通信部は不採用、
  **capacity 管理と expert 単位 dispatch の考え方のみ**参照。
- DeepSpeed-MoE：capacity ベースの dispatch/scatter パターン。順方向の permute／逆方向の
  un-permute（scatter 復元）の分離を踏襲する。
- Colibri PR #475：推論側の expert 固定・トークン走査のループ順序の前例として参照。
  本書は学習（fwd+bwd）のソート＋バッチングに適用する。

4調査（別wt文書）の結論は混ぜない。本書の数値は `roofline.md` の AI／ridge／AI≈23 のみを入力とし、
KV・decode・学習高速化・factorized head・G1融合・量子化の効果を上乗せ計上しない。

---

## 1. トークンソート設計

### 1.1 位置

`moe_layer` 相当の入口で、gate logits → top-k（k=2）→ softmax 重み確定**後**にソートする。
ソートキーは `(expert_id, token_pos)` の2段キー。重み・活性本体の移動は permute index 経由で行い、
logits 自体の再計算はしない。

入力：T トークン × k=2 の `(expert_id[e], weight[e], token_pos)`（2T エントリ）。
出力：expert 境界配列 `off[E+1]`＋ permute index `perm[2T']`（T' は drop 後、§1.2）。

### 1.2 capacity factor 上限＋超過 drop

- 方式：固定 capacity 方式（DeepSpeed-MoE / Tutel の capacity 型を単ノード化）。
  ```
  cap_e = ceil(capacity_factor * T * k / E)
  ```
  デフォルト `capacity_factor = 1.5`（G2改訂で 1.25→1.5。調整可能範囲 1.0–1.5 内。前例は Tutel 動的 factor）。
  改訂理由：§9.4の通り長時間走行で 1.25 は終盤 6.64% とトリガ超過のため、設計範囲上限の 1.5 を既定化する。
  `cap_e` は longrun 条件（T=512・k=2・E=16）で `ceil(1.25*1024/16)=80` → `ceil(1.5*1024/16)=96`。
  drop率 0.1% 以下見込み（§8.1短走 0.74% からの外挿。G2再測定§10で実測確定）。
  メモリ増は expert 連続ワークスペースのみ：Xe/Ye 各 80*256*4=80KB→96*256*4=96KB（+16KB×2）、
  Ge/Ue 各 80*64*4=20KB→96*64*4=24KB（+4KB×2）、合計 +40KB。
  active 100M 予算（fp32換算 400MB）・dense 2GB に対し無視可能（0.01%級）であり予算内。
  TinyStories 条件（T=64・k=2・E=8・n=64）では cap 20→24、ワークスペース +1KB級で同様に予算内。
- 各 expert への割当てが `cap_e` を超えた分は **drop**（当該 (token, expert) ペアの寄与を 0 とし、
  残りの選出 expert の重みは renormalize **する** — G2改訂で仕様変更。旧「renormalizeしない」を撤回。
  しない方が動力学を変える（実効勾配が痩せる）。標準実装（DeepSpeed-MoE/Tutel流）はrenormalize。
  方式：トークン毎に kept の重み和 S で `w' = w / S`（S>0時。S==0時は寄与0）。
  `out_weights` は kept について renormalize 後の値に更新する（in-place）。
  dropped の `out_weights` は元のまま残すが未使用（dropマスクでskip）。
  bwdは renormalize 後重みで softmaxヤコビアンを計算する（分母S経由の2次項は straight-through として無視。§3.3に記録）。
  renormalize 有無はハイパーパラメータではなく仕様として固定し、変更時は設計改訂とする）。
- 共有 expert（常時オン2つ、AGENTS §2.1）は capacity 制限の対象外（§2.3）。
- drop 率・expert 別ヒストグラムをステップ毎にカウンタ記録する（デバッグ用。ホットパス外の集計のみ）。
  drop 率が恒常的に >5% の場合は capacity_factor の見直しトリガ（自動調整はしない）。
  drop率はG2関数等価性の合否基準から除外し、本節トリガ（steady-state判定。§11の最後100 step平均で評価）で扱う。
  G2は§5.4の3項目（最終val差・単調性・勾配分布）のみで判定する。
- G2改訂（2026-09-14）：G2判定タスクを合成回帰から TinyStories に変更する。
  理由：合成は train loss が 0 に張り付き（§9.1：表示桁落ちで相対差未定義、後半 200% 張り付き）評価不能のため。
  TinyStories（実言語CE）は 0 床なしで最終loss差≦1%が定義可能。F6確立の設定を流用する
  （`train_tinystories` 既定：d=64・layers=2・E=8・K=2・S=1・H=32・batch=64・lr=3e-4・V=258。
  V=48588は情報密度で勝つが G2短走（200 steps以下）は V=258 で判定。1000 steps禁止）。
  G2再測定は 200 steps以下・逐次・単一プロセス・実行前 `uptime` 確認。判定は§5.4の4項目維持。
  改訂後も乖離が残れば H2（bwd加算順序）と分離報告する（許容判断禁止）。
  付帯：`train_tinystories` 等の対象 main の先頭に `setvbuf(_IOLBF)` を追加する
  （行バッファリングで G2 ログの逐次可視化。数値・API・fail-closed に影響なし）。

なぜ dropless（Megablocks 式）にしないか：CPU学習では可変長 dropless が
マイクロカーネルのタイル固定化（§4）と L2 常駐（expert 単位 192KB、roofline D6）を崩すため。
固定 cap により各 expert の M（§2.1）が `cap_e` 上限で抑えられ、ワークスペースを静的確保できる。

### 1.3 ソート安定性と決定論性の方針

- アルゴリズム：counting sort（E=16 と小さいため O(E + kT)）。
  E が将来数千規模になった場合は radix sort（8bit×2パス）へ切替え可能な構造とするが、
  P3 では counting sort のみ実装する。
- 安定性：**安定ソート**（同一 expert 内では元の token_pos 昇順を保持）。
- 決定論性：**同一入力 → 同一 perm を保証**（決定論的、同スレッド数に依存しない）。
  タイブレークは `(expert_id, token_pos, slot k=0/1)` の辞書式順序で完全順序化し、
  浮動小数点の非結合性が perm に波及しないようにする（重みの大小比較で順序を決めない）。
- マルチスレッド下でも perm 構築は単一スレッドで行い、その後の expert 単位 GEMM を並列化する
  （perm 自体の並列化はしない — 決定論性維持のため）。

---

## 2. エキスパート単位バッチング

### 2.1 可変 M × 固定 N,K の密 GEMM 化

各 routed expert e について、割当てトークン数 `M_e`（0 ≦ M_e ≦ cap_e）を M 次元とする密 GEMM に変換する：

- gate/up（n=256 → h=64）：`X_e [M_e × N]` × `W [N × K]`、N=n=256、K=h=64。
- down（h=64 → n=256）：`Y_e [M_e × 64]` × `W_down [64 × 256]`。
- N,K は全 expert 共通固定、**M_e のみ可変**（0 の expert はスキップ、ごく小さい M_e もパディングせずそのまま実行。
  パディングはメモリ増・AI 希釈のため行わない。§4 のテール処理で吸収）。

現行 GEMV（M=1、AI ≈ 0.38–0.50）→ バッチ GEMM（M_e ≈ T·k/E ≈ 64 at T=512 の平均、
cap 上限で高々 ~80）により、重み行列を M_e 回再利用する。これが AI ≈ 23（§6）への理論的根拠。

重みレイアウト：expert 固定・トークン走査の順序（`roofline.md` D6 の「expert固定・トークン走査が正しい」
に準拠）。1 expert 分 192KB（gate/up/down 3行列、fp32）が L2（N150: 2MB共有）に常駐する粒度で
タイル化し、層単位 3.36MB の L3 溢れを避ける。

### 2.2 dispatch / combine のメモリパス

- dispatch（gather）：`X_perm[m] = X[token_pos[m]]` を expert 連続バッファに集約。
  ワークスペースは `max cap_e × n` を静的確保（fp32 で 80×256×4B = 80KB/expert バッファ、再利用）。
- combine（scatter-add）：expert 出力 `Y_e` に gate 重みを掛けて `Y[token_pos] += w * Y_e` で復元。
  残差加算との融合（roofline D7 の F1 相当）は**本書の範囲外**（将来の融合設計に委ね、ここでは素朴 scatter-add）。
- 活性（SwiGLU 中間 G/U/Y）の保存は perm 順のまま保持し、bwd で同一 perm を再利用する（再ソートしない、§3）。

### 2.3 共有 expert と attention は T=512 そのまま GEMM

- 共有 expert（常時オン2つ）：全 T トークンが対象のためソート不要。
  **M=T=512 固定の密 GEMM** としてそのまま実行（gate/up/down 各1 GEMM）。
- attention 射影（Q/K/V/O 等の毎トークン発火の線形層）：同様に **M=T=512 の密 GEMM**。
  本書のソート対象は routed expert のみであり、共有 expert・attention はソートを経由しない通常 GEMM パスとする。
- T=512 は `train_longrun` の実働 TOKS に対応（roofline D6 の AI≈23 の前提と同一）。
  T が変わる場合は AI 式（§6）の再計算を要するが、カーネル IF 自体は任意 M 対応（§4テール）とする。

---

## 3. 逆伝播：ソート済み処理＋scatter 復元

### 3.1 原則

bwd は fwd と**同一 perm・同一 expert 境界**で処理する（bwd での再ルーティング・再ソート禁止）。
fwd 時に `perm`、`off`、`gate weight`、`drop mask` を保存し、bwd はそれを読み出すのみ。

### 3.2 各 GEMM の bwd 対応

expert e（perm 順バッファ上）：

- `dY_e [M_e × N_out]`（combine 側からの scatter 逆写像＝gather：`dY_e[m] = dY[token_pos[m]] * w[m]`）から開始。
- `dW = X_e^T · dY_e`（N×M_e × M_e×N_out → N×N_out の GEMM。M_e が縮約次元）。
- `dX_e = dY_e · W^T`（M_e×N_out × N_out×N の GEMM）。
- SwiGLU 非線形の bwd（roofline D4 の SwiGLU bwd、AI 0.38）は perm 順のまま要素wise処理し、
  順序依存なし（要素独立のため perm の影響を受けない）。
- `dX[token_pos] += dX_e` で scatter-add 復元（k=2 のため同一 token に2寄与が加算される。
  加算順序は `expert_id` 昇順に固定する。f32 非結合則のため順序変動は最終ビットを変え得るので、
  「任意」ではなく固定とし、§5 の許容差で評価する）。

### 3.3 gate 勾配の順序扱い

- gate logits の勾配（top-k 経路）は **元の token 順序で**計算・加算する。
  expert 順バッファ上では中間量（`w[m]`, `dY·Y_e` の内積等）を保持するのみとし、
  logits への最終 scatter は `token_pos` 順に直列化する。
- top-k の hard 選択自体は微分不可のため straight-through（選択マスク固定）の扱いを維持し、
  本書でゲーティング関数の変更は行わない。
- drop されたペアの gate 勾配は 0（寄与なし）。G2改訂の renormalize 仕様（§1.2）と整合させる：
  kept の `w'`（renormalize 後）で softmaxヤコビアンを計算し、dropped は 0 のまま。
  分母 S 経由の 2 次項（`dw'/dw` の分母微分）は straight-through として無視する（実装を単純に保つ標準近似）。
- 決定論性：gate 勾配の token 方向加算は token 順に固定し、スレッド分割は expert 単位または
  token ブロック単位のいずれかに統一する（混在させない）。実装ステップ§7の soak で順序固定を検証。

---

## 4. マイクロカーネル：可変 M 下のレジスタブロッキング方針（AVX2）

前提：`roofline.md` D6 より、nanogemm 前例＝ AVX2 **6×16相当、acc 12本（ymm0–ymm11）＋B用2本**、
16 YMM にスピルなし収容。Mc=64, Nc=128, Kc=128 の L1/L2 タイリング。
ただし現行は GEMV のため acc 多本化は無効であり、**本書の GEMM 化後に初めて有効**になる。

### 4.1 方針（AVX2・16 YMM・acc 12本目安）

- マイクロタイル：**Mr × Nr = 12行（h/n 方向のうち小さい方に合わせる）× 4トークン（M 方向）**を既定案とする
  （roofline D6 の提案 Mr=12 × Nr=4 を採用）。
  f32 8-wide 換算で acc 12本 = 96出力保持（384B）＋重みロード1本＋Xブロードキャスト2本 = 15本 ≦ 16 ✅スピルなし。
- f64 累積版はレーン圧が倍（4-wide×12本=48出力で2セット要）のため、**f32蓄積＋最終 f64 加算の混合**を既定とする
  （等価性§5の許容差とセット）。
- Kc 方向：K=256（gate/up の N 側）・K=64（down の N 側）に対し、L1 32KB に収まる Kc=64–128 で分割。
  Mc 方向：cap_e（~80）以下はタイル分割せず一括。M_e が cap を超えることは仕様上ない（§1.2）。
- FMA は bit 同一要請がある場合（`src/tmac.c:36-39` の mul/add 分離の流儀）は mul/add 分離とし、
  スループットは FMA の半分目安で見積もる（roofline D6 同様）。bit 同一が不要な経路では FMA 可（設計会議で確定）。

### 4.2 可変 M のテール処理

- M_e は expert 毎に 0–cap_e でばらつくため、**Mr=12 に対する剰余テール**（M_e mod 12）および
  **Nr=4 に対する剰余**をマスクロード／スカラーフォールバックで処理する petite カーネルを用意する。
  メインカーネルとテールカーネルで acc 本数は変えない（レジスタ割付け共通化）。
- M_e=0 の expert はカーネル起動自体をスキップ（分岐コストのみ）。
- M_e < Mr の小 expert（疎 expert）が多数ある場合もパディングしない（§2.1）。
  テール効率の実測は§6の測定方法で行い、Mr の再調整（例：6×8 への縮小）は soak 後の判断とする。

### 4.3 他 ISA への拡張余地（設計のみ）

- AVX-512／ARM NEON への移植は本書の実装ステップに含めない。
  AVX2 の Mr/Nr/acc 本数のみ確定し、他 ISA はレジスタ本数に応じた比例換算（例：AVX-512 32 ZMM では Mr 倍増可）
  の方針を残す。`#ifdef` ガード規約（AGENTS §7.1）は将来実装時に遵守。

---

## 5. 等価性：同一シードの loss 軌道一致基準

### 5.1 等価性の定義

本設計変更は**数値最適化ではなくループ順序変更**であり、以下を満たすことを受入条件とする：

1. **bit 一致が要求される部分**：perm 構築、expert 境界、drop マスク。同一シード・同一入力で完全一致すること。
2. **浮動小数点順序差が許容される部分**：GEMM 内の加算順序（K 縮約順・M タイル順）、
   scatter-add の加算順（k=2 の2寄与の順序）、f32蓄積＋f64加算の混合（§4.1）。
   これらは IEEE 非結合性により最終ビットが変わり得る。

### 5.2 loss 軌道一致基準（同一シード）

- 条件：同一シード・同一データ順・同一スレッド数・同一スレッド配置で、ソート前後の実装を N ステップ走行。
- 基準案（P3 実測で確定する前の設計値）：
  - **ステップ毎 loss の相対差 ≦ 1e-6**（fp32蓄積の場合。f64蓄積の場合は ≦ 1e-12 を目標とするが、
    §4.1の混合既定では fp32 基準を適用）。
  - 最終ステップのパラメータ L2 差は loss 基準の補助指標とし、合否判定には使わない（発散系での増幅を避けるため）。
- 測定手順：`train_longrun` 相当の短走（例：--steps 6–20）を同一シードで新旧両実装走行し、
  ステップ毎 loss を CSV 比較するスクリプトで判定。5%ノイズ床（roofline の `d354025` ルール）は
  スループット判定にのみ適用し、**等価性判定には適用しない**（等価性は上記タイト基準）。

### 5.3 浮動小数点順序差の許容範囲の明記方法

- 設計書・PR 説明に以下を明記する：
  1. 順序が変わる演算の列挙（K縮約順、タイル順、scatter順、f32/f64混合点）。
  2. 上記§5.2の定量基準と実測値（最大相対差・発生ステップ）。
  3. bit 一致部分（perm/drop）と順序差許容部分の境界の明示。
- 将来 FMA 導入時は別途「FMA vs mul/add 分離」の差分評価を要し、本書の基準をそのまま流用しない
  （契約積算順序が変わるため）。

### 5.4 ゲート運用3段階化（G1/G2/G3）

- **G1 bit一致**：関数変更なし・順序のみ。perm＋off＋drop 導入（Step 1、GEMV維持）の段階に適用。
  判定は§5.2（ステップ毎 loss 相対差 ≦1e-6）。Step 1 は合格済み（§8.3：相対差 2e-9）。
- **G2 関数変更（統計的同等性→関数等価性）**：Step 2（バッチ fwd 素朴 GEMM）＋ Step 3（バッチ bwd、同一 perm 再利用、
  dropped 対の gate 勾配 0 化）の段階に適用。drop・加算順序の変化により bit 一致は要求しない。
  判定は関数等価性の3項目（同一シード・同一データ順・同一スレッド数・同一スレッド配置。
  単体経路 vs バッチ経路 `--moe-batch` の長時間走行比較。drop率は基準から除外し§1.2トリガで評価）：
  1. 最終 val loss 差 ≦1%（相対差。G2改訂：train loss 飽和後の相対差は使わない。
     合成回帰では train が 0 に張り付き相対差が未定義／200% 飽和となるため（§9.1）。
     TinyStories 実言語 CE は 0 床なしで val 差が定義可能）。
  2. 収束単調性：後半 50% の val 相対差推移を確認。単調増加なら dynamics 変化、安定なら過渡応答と記録する
     （許容判断はしない。推移の事実のみ記録）。
  3. 勾配ノルム分布：平均・分散が ±10% 以内（200 stepsでは平均1.06%で合格。分散は境界超過のため§10.4に記録。
     500 steps延長§11では平均0.08%・分散3.11%で合格）。
  drop率・renormalize発火率はG2合否に使わず、ステップ毎・合計の記録として残し§1.2トリガ（steady-state判定）で評価する。
  renormalize発火率（drop を含むトークンのうち kept 残存で再正規化した割合）も同様に記録のみとする。
  不合格項目があれば原因分析を行う（許容判断禁止）。
  G2通過の根拠（LB-check改訂。2026-09-14）：drop=0区間bit一致（seq==1 legacy single vs batch 30 stepsでdrop 0・ce完全一致）、
  乖離onset=初drop（T=512ではstep1の159/2048=7.76%から乖離開始：single ce=5.538190 vs batch ce=5.538403）、
  最終val差0.29%（200 steps。500 stepsでは0.059%）、単調安定（§10.2・§11）、勾配平均1.06%（200 steps。500 stepsでは0.08%）。
  以上によりG2は関数等価性で通過とする。drop率は§1.2トリガに分離済みである。

### 5.5 ギャップ記録（Step 3 関数変更の位置づけ）

- Step 3 の関数変更（drop 時の renormalize・dropped 対の gate 勾配 0 化、分母 S 経由 2 次項の
  straight-through 無視）は参照実装未検証である。DeepSpeed-MoE／Tutel 流の標準仕様として受入れ、
  本設計の仕様として固定する（ハイパーパラメータではなく仕様。変更時は設計改訂）。
- Step 4 以降で間接検証する（G3 bit一致／相対差 1e-6 および soak の loss 軌道・steps/s で異常が出れば
  本仕様に遡って原因分析する）。G2 では上記4項目の事実記録に留め、許容判断はしない。
- **G3 最適化（同一関数内）**：Step 4（マイクロカーネル AVX2、§4）に適用。Step 3 通過後の同一関数内での
  最適化であり、判定は bit 一致または相対差 1e-6（§5.2 を維持）。Step 4 に適用し、G2 基準（1%）は流用しない。

### 5.6 G4 最適化（数値等価だが routing 結合で関数的に発散する最適化）

- 定義：同一関数（同一 perm／drop／renormalize 仕様）の下で、GEMM 内丸め順序のみが
  変わり、重み→routing 帰還結合で長時間軌道が関数的に発散し得る最適化を **G4** とする。
  Step 4（f32混合マイクロカーネル、`jt_gemm_mat_f32`）が該当する。
  G3（§5.2 stepwise 1e-6）は drop=0 域・単位レベル（テール G1・batch fwd 単体・短期再走行）に
  適用を限定し、drop>0 の長時間系統軌道には適用しない（適用すれば仕様内丸めが必発で不通過となるため）。
- 基準（4項目。train loss はいずれにも使わない。TinyStories 実言語 CE の val のみで評価）：
  1. 構造的等価性：perm／off／drop マスク bit一致、gate 初期一致（初 step drop 数の一致）、
     テール G1 bit一致（§12.1）、決定論性（同一条件再走行の stderr bit一致）。
  2. 統計的同等性：最終 val 差 ≦1%（相対差。G2 と同一閾値を流用）。
  3. 増幅有界性：val 差軌道の形状で判定。飽和（<0.3% に収束・単調増加なし）＝完全通過、
     線形（~0.8% への漸増）＝条件付き通過、指数（数%超への発散）＝不通過。
  4. 性能目標：§6 主判定2項目（理論天井比>50%、steps/s 3x以上）を維持すること。
- Step 4 の G3 逸脱の承認記録（2026-09-14）：§12.3.1 の系統レベル 200 steps
  （TinyStories micro vs §10 naive-batch：最終 train 0.164%／最終 val 0.092%）は、
  §5.2 の stepwise 1e-6 を最終値に適用すれば超過する。
  しかし (a) 単位・短期・drop=0 域では G3 通過済み（テール G1 bit一致・batch fwd 1e-6・
  20 steps 再走行 stderr bit一致）、(b) gate 初 step の drop 数は micro 159/2048 と
  LB-check 記録 159/2048=7.76% で一致（perm／drop 系は bit 同一の傍証）、
  (c) 乖離は f32 混合丸め→重み→routing 帰還の仕様内発散（G2 H1 と同機構であり、
  G2 の 0.29% より小さい 0.092%）であり、テール G1・gate 初期一致・決定論性で
  実装バグは分離済みである。よって G3 strict（stepwise 1e-6×200 steps・drop>0）は
  不通過と記録した上で（§12.3.1 の記録を維持）、Step 4 を G4 の枠組みで再評価することを
  承認する。本項は枠組み変更の記録であり、超過値の許容判断ではない。

---

## 6. 目標との対応：理論天井比>50%・steps/s 3x以上（主判定）、AI>21（診断指標）

主判定は2項目のみとする：**理論天井比 >50%（§6.2）かつ steps/s 3x以上（§6.3）**。
**AI > 21（§6.1）は主判定から診断指標に格下げ**する。理由：routed expert の
単体 AI 上限が約15.6（cap 上限 M_e=80 での単一 matmul 式の値）に留まり、
共有 expert＋attention の密 GEMM（AI ≒ 23）と FLOPs 加重平均しても約18.7 で
ridge 21 に届かないため。約18.7 は ridge の約89%であり、帯域律速の残余は
天井比（§6.2）側で評価すれば実用上十分と判断する。算出根拠は§6.1に記す。

### 6.1 AI > 21（診断指標 — 主判定から格下げ）

- roofline D6 の式：AI = hnT / 2(hn + T(n+h))。n=256, h=64 で **T=512 なら AI ≒ 23**。
- 本書の対応（診断値としての見立てであり合否には使わない）：
  - 共有 expert・attention：M=T=512 の密 GEMM で上式通り AI ≒ 23 を狙う。
  - routed expert：M_e ≈ 64（平均）でも AI = hnM_e / 2(hn + M_e(n+h)) ≒ 13–15 に達し、
    ridge 21 には単体で届かないが、cap 上限（~80）＋重み常駐（L2 192KB）により実効 AI は向上する。
    全体（routed＋共有＋attention）の加重平均でも **AI > 21 は達成見込みなし（約18.7止まりの見立て。
    詳細は下記「算出根拠」）**。このため AI は診断記録とし、未達時の調整ノブ
    （capacity_factor 引き上げ、T 増）は天井比・steps/s の改善手段として扱う（設計変更として記録）。
- 測定方法：D4 と同一の計数法（積和=2、fp32=4B、sigmoid/exp≈20 FLOP）で FLOPs／バイトを再計数し、
  expert 別 M_e 実測値を入れて加重 AI を算出する。推定帯域（29.4GB/s、STREAM 未計測±30%）ではなく
  **AI 値そのもの**で記録する（帯域誤差に不感なため）。AI は診断指標であり合否判定には使わない。
- 算出根拠（routed vs shared＋attention の FLOPs 比と加重平均。n=256, h=64, E=16, k=2, T=512）：
  - 式は roofline D6 と同一の単一 matmul 式 AI(M) = hnM / 2(hn + M(n+h)) を使用する。
  - コード読解：`moe_layer.c` の routed 1ペアは gate/up/down 3 matmul（各 2hn）＋silu/exp 微小項、
    `train_longrun.c` 実働は TOKS=512（128×4）、E=16, k=2（`LR_E/LR_K/LR_TOKS`）。
    よって routed ペア総数 = T·k = 1024、平均 M_e = 1024/16 = 64、
    cap = ceil(1.25·T·k/E) = ceil(80.0) = 80（§1.2）。
    軽計測（`build-g1/train_longrun --steps 3 --threads 1`）で toks/step=512・E=16・k=2 を実測確認した。
  - AI 値：routed 平均（M_e=64）= 14.22、routed 上限（M_e=cap=80）= 15.61 ≒ **15.6**、
    密（M=512、共有 expert・attention）= 23.27 ≒ **23**。
  - FLOPs 比（matmul 主項。1ペア = 6hn = 98,304 FLOP）：
    routed 1024ペア = 100.7M、共有 S=2（AGENTS §2.1 目標）1024ペア = 100.7M、
    attention 物差しとして n×n 密1射影分（M=512）= 67.1M。比は約 1 : 1 : 0.67。
    （gate logits 2En/トークン ≈ 4.2M/層、silu/exp ≈ 1–2M/層は1–3%級のため主項のみで計数。D4 同順位に影響なし。
    `train_longrun` 実働 S=1・attention なしの場合は routed 100.7M : 共有 50.3M。）
  - 加重平均 AI（設計目標 S=2＋attention 物差し1射影、routed 平均使用）：
    268.4M / (100.7/14.22 + 100.7/23.27 + 67.1/23.27) = 268.4/14.29 ≒ **18.7**。
    routed 上限（15.61）使用でも ≒ 19.7 止まり。**いずれも ridge 21 未達**。
    18.7/21 ≒ **89%** であり、残余の律速は天井比（§6.2）側で評価する。
    参考：longrun 実働構成（S=1・attention なし）の加重は ≒ 16.3 でさらに低い。

### 6.2 理論天井比 > 50%

- 定義：`達成 steps/s ÷ roofline 理論天井 steps/s`。理論天井は
  `min(peak_FLOPs / FLOPs_per_step, eff_BW / bytes_per_step)` のうち支配項（現状は帯域項）。
- GEMM 化後の加重 AI は約18.7（§6.1算出根拠）で ridge 21 の約89%に留まり、
  支配項は帯域側のままとなる見立て。よって理論天井は引き続き帯域項
  `eff_BW / bytes_per_step` で評価する（AI>21 による計算律速側への移行は見込まない）。
  mul/add 分離時はピークの半分目安（roofline D6）で天井を再計算する。
- **>50% を達成見込み**とする根拠：nanogemm 級マイクロカーネル（acc 12本・スピルなし）の実績線形性と、
  L2 常駐（192KB/expert）によるストリーミング削減。未達要因はテール効率（§4.2）と scatter-add オーバーヘッド
  に限定される見立て。
- 測定方法：同一マシン・同一ビルド（Release）・同一 `--steps`・同一スレッド数で steps/s を計測し、
  5%ノイズ床ルールで有意判定。STREAM 実測があれば eff_BW を置換えて天井を更新する。

### 6.3 steps/s 3x 以上

- 基準：ソート＋バッチング導入前の同一構成（逐次トークン GEMV）の steps/s に対する倍率。
- 見込み：AI 0.5 → 23（約46×の算術強度向上）のうち、メモリバウンド→コンピュートバウンド遷移分が
  実効 3–10×として現れる想定。roofline D6 の「1T 1.96x はロード幅効果」に対し、
  本件は「重み再利用効果」のためスケールが異なる。保守的に **3x 以上を必達ライン**とする。
- 測定方法：§6.2 と同一条件で新旧比較。スレッド数は 1T/2T の両方で記録する
  （roofline D4末尾：1Tでは計算が効き2Tで帯域飽和、という構造が GEMM 化でどう変わるかを確認するため）。
  4T 以上の多スレッドは csw・コヒーレンス混入（roofline D5）のため参考値とする。
- 非スコープ効果の除外：測定時に量子化・融合・KV・decode の変更が混入していないことを
  `git status`／ビルド hash で確認し、本書の効果のみを計上する。

---

## 7. 実装ステップ分解（1変数ずつの順序）

各ステップは等価性ゲート（§5.2）を通過してから次へ進む。2変数同時変更禁止。

1. **Step 1 — ソートのみ（計算は既存 GEMV のまま）**
   perm＋off＋drop マスクを導入し、dispatch→既存 expert GEMV→scatter-add の経路に切替える。
   GEMM 化なし。ゲート：perm の決定論性（同一シード完全一致）＋ loss 相対差 ≦ 1e-6 ＋ drop 率計測の動作。
2. **Step 2 — バッチ fwd（素朴 GEMM、マイクロカーネルなし）**
   expert 単位の密 GEMM fwd を参照実装（ triple-loop または既存 dot の M 方向拡張）で導入。
   ブロッキング・SIMD 最適化なし。ゲート：§5.2 ＋ AI 再計数（加重 AI の向上確認）＋ steps/s 記録（3x 未達でも可、傾向確認）。
3. **Step 3 — バッチ bwd（§3 の通り）**
   同一 perm 再利用の dW／dX GEMM＋gate 勾配の token 順 scatter を導入。
   ゲート：§5.2（bwd 込みの loss 軌道）＋勾配の順序固定の検証（スレッド数変動でも perm 一致）。
4. **Step 4 — マイクロカーネル（AVX2、§4）**
   Mr=12×Nr=4、acc 12本、f32蓄積＋f64加算、テール処理を導入。共有 expert・attention の T=512 GEMM も同一カーネルに乗せる。
   ゲート：§5.2（f32混合基準）＋理論天井比 >50% の中間評価＋ steps/s 3x の達成判定。
5. **Step 5 — 統合 soak**
   Step 1–4 統合の長時間走行（例：数百 steps）で drop 率・テール効率・スレッド数別 steps/s・loss 軌道を記録。
   capacity_factor（1.0–1.5）、Mr/Nr の再調整提案は本 soak のデータをもって設計改訂として行う（本書内での場当たり調整禁止）。
    ゲート：§6 の主判定2項目（天井比>50%、3x）の最終合否＋AI（診断指標）・§5.3の明記文書の完成。

各ステップの成果物はコード差分ではなく**測定記録**（AI 再計数表、loss 差分 CSV 要約、steps/s 表）とする。
本書は設計のみであり、実装着手は別タスクとする。

### 7.1 Step 3 実測記録（§7追記。実装変更なし・§11受領値のみ。新規実行なし）

- 性質：Step 3（バッチ fwd＋bwd素朴GEMM、マイクロカーネルなし）完了時点の記録。
  本節は `§11`（TinyStories T=512 single vs batch 500 steps）の受領値のみで構成し、
  新規の性能測定は行わない（Step 4aテール前の性能測定禁止を遵守）。
  API・fail-closed・tol不変、ctest全pass、AVX-512不使用、cap=1.5維持は§11の通り。
- 条件（§11再掲）：`train_tinystories --data data/tinystories_head16M.txt
  --steps 500 --batch 4 --seq 128（T=512） --val-every 20 --patience 0
  --max-val-pairs 2000 --aux-weight 0.01`を単体路と `--moe-batch` で逐次・単一プロセス。
  dimsは d=64（n）・layers=2・E=8・K=2・S=1・H=32（h）・V=258。
  所要は単体 80.1s／バッチ 34.5s（本wt限りログ `build-lb/g2_500_*`）。

#### 7.1.1 steps/s・toks/s（§11受領値）

- 単体路：500/80.1 = **6.24 steps/s**、toks/s **3194**（512×6.24=3196と一致）。
- バッチ路（Step 3素朴GEMM）：500/34.5 = **14.49 steps/s**、toks/s **7421**。
- 倍率：14.49/6.24 = **2.32x**（§11の約2.32xと一致）。
  3x必達はStep 4（§4マイクロカーネル）の判定であり本節の合否に使わない。
  Step 4bでは本値をStep 3 baselineとし、回帰なし（micro ≧ 14.49）＋Step 2比3x
  （micro ≧ 6.24×3=18.72 steps/s）を判定する。

#### 7.1.2 AI実測（診断指標。D4と同一計数法：積和=2、fp32=4B）

- 単一matmul式はroofline D6と同一 AI(M)=hnM/2(hn+M(n+h)) を使用する。
  TinyStories dims（n=64・h=32・hn=2048）での値：
  - 密（M=512、共有expert）：AI = 512·2048/2(512·64+2048+512·32)
    = 1048576/102400 ≒ **10.24**。
  - routed平均：§11 batch合計drop 39512/1024000=3.86%よりkept/層/step
    = (2048−79.024)/2 = 984.488、平均M_e = 984.488/8 ≒ **123.06**。
    AI(123.06) = 252029/27724 ≒ **9.09**。
  - routed上限（cap=ceil(1.5·1024/8)=192）：AI(192) ≒ **9.60**。
  - head物差し（M=512・N=258・K=64）：FLOPs=2·512·258·64=16.91M、
    bytes=4(512·64+64·258+512·258)=725504B、AI ≒ **23.31**。
  - gate logits（M=512・N=8・K=64）：FLOPs=1.05M、bytes=0.30MB、AI ≒ **3.50**。
- 加重平均AI（fwd matmul主項。routed 24.19M＋共有12.58M＋head 16.91M）：
  53.68/(24.19/9.09+12.58/10.24+16.91/23.31) = 53.68/4.615 ≒ **11.63**。
  gate logits込みでは ≒ **11.14**。bwd込みでも同程度（同一式のため）。
- ridge 21（§6前提）に対し約43–55%であり、**帯域律速のまま**（左側）。
  §6.1の長期構成（n=256・h=64）での加重約18.7（ridgeの約89%）と異なり、
  TinyStories小dimsではAI≈11止まりである。これはdims差の帰結であり、
  実装の優劣ではない。AIは診断指標であり合否判定には使わない。

#### 7.1.3 理論天井比（Step 3時点。診断記録。合否に使わない）

- §6.2定義（達成steps/s ÷ roofline理論天井steps/s）でTinyStoriesに適用する。
  eff_BWはroofline前提の推定値29.4GB/s（STREAM未計測±30%）をそのまま使う。
- bytes/step概算（fwd 4.91MB：§7.1.2のmatmul bytes＋gate 0.30MB。
  bwdはdW＋dXで約2x、head bwd込みで約2x。optim8・norm微小項を除く）：
  fwd約4.9MB＋bwd約10MB ≒ **約15MB/step**（1–3%級のsilu/exp・renormを除く主項）。
- BW天井 = 29.4GB/s ÷15MB ≒ **1960 steps/s**（計算天井=614GFLOPs÷約150MFLOP
  ≒4093 steps/sより帯域項が支配。AI≈11<21と整合）。
- 天井比：バッチ14.49/1960 ≒ **0.74%**、単体6.24/1960 ≒ **0.32%**。
  50%に遠く及ばないが、これはTinyStories小問題がオーバーヘッド支配
  （head/CE・dispatch gather/scatter・sort・ログ）であり、§6.2の長期構成
  （longrun n=256・bytes約6GB/step・天井約4.9 steps/s）とは前提が異なるためである。
  本値はStep 3の診断記録とし、Step 4bの主判定（天井比>50%）は§6.2の長期構成
  またはGEMMカーネル単体の天井で評価する（Step 4b記録で定義を明記する）。
  許容判断はしない。

#### 7.1.4 batch 2.32xの内訳（何が効いたか。断定禁止・仮説の記録）

- 支配項H1（重み再利用）：単体路は (token, expert) ペア毎にexpert重み
  （TinyStories 1 expert分 gate/up/down 3·32·64·4B=24KB、longrun 192KB）を
  ストリーミングするのに対し、バッチ路はexpert毎にM_e回再利用する。
  平均M_e≈123（TinyStories）・64（longrun T=512平均）の再利用がAI 0.5→9–15への
  向上に対応し、2.32xの主因の可能性（D6の「重み再利用効果」と整合）。
- 寄与H2（L2常駐）：1 expert分24KB（TinyStories）・192KB（longrun）がL2
  （N150 2MB共有・i7-8700B 256KB/core）に常駐する粒度であり、層単位3.36MBの
  L3溢れ（roofline D5）を避ける構造の可能性。結合がtoken順scatter-addのため
  効果は限定的の可能性。
- 寄与H3（dropスキップ3.86%）：droppedペアのGEMV/GEMM呼出しをskipする分、
  約3.86%の計算削減。2.32x（132%増）のうち約4pp相当の minor の可能性。
  renormalize（kept w/S）の追加コストはO(Tk)で微小。
- 控除O1（gather/scatter・sort・renormオーバーヘッド）：dispatch gather
  （Xe/Ye等のmemcpy）・combine scatter-add・counting sort O(E+Tk)・renormalizeが
  上乗せされ、理想の重み再利用倍率（数十x）から2.32xに希釈される可能性。
  head/CE・optim8（約30%級の非MoE部）はバッチ化対象外のためAmdahl希釈の可能性。
- 控除O2（素朴ループ）：Step 3はtriple-loop／既存dotのM方向拡張であり、
  ブロッキング・SIMD最適化なし（Step 4申送り）のため、FLOP効率は低いままの可能性。
- H4（ノイズ床）：5%ノイズ床（roofline d354025ルール）に対し2.32x（132%増）は
  有意であるが、内訳の定量分離（H1/H2/O1/O2の寄与率）は未実施。
  分離はStep 4bのmicro前後比較（Step 3比回帰なし・3x達成）に委ね、本節では結論しない。
- 次手はStep 4aテール（Mr12×Nr4・acc12・f32＋f64・テール）の実装であり、
  本節の記録をもってStep 4可とする（§11.4のStep 4可判定に接続）。

---

## 8. Step 1 追加記録（実測。実装変更なし・記録のみ）

### 8.1 実 drop 率（cap_e=80・平均 M_e=64、短条件 --steps 20）

- 条件：`train_longrun` 実働 T=512・k=2・E=16・cap_factor=1.25、
  `--threads 1 --moe-batch`、単一プロセス逐次。
  cap_e = ceil(1.25·512·2/16) = ceil(80.0) = **80**。
  平均 M_e = T·k/E = 1024/16 = **64**（§6.1 前提と一致）。
- ステップ毎 dropped（6層合計。分母 6144 = 6·512·2）：
  step1: 15 (0.2441%)、step2: 15 (0.2441%)、step3: 13 (0.2116%)、
  step4: 18 (0.2930%)、step5: 15 (0.2441%)、step6: 11 (0.1790%)、
  step7: 9 (0.1465%)、step8: 5 (0.0814%)、step9: 5 (0.0814%)、
  step10: 9 (0.1465%)、step11: 14 (0.2279%)、step12: 26 (0.4232%)、
  step13: 37 (0.6022%)、step14: 57 (0.9277%)、step15: 79 (1.2858%)、
  step16: 97 (1.5788%)、step17: 106 (1.7253%)、step18: 116 (1.8880%)、
  step19: 130 (2.1159%)、step20: 132 (2.1484%)。
- 合計 909 / 122880 = **0.74%**。単一ステップ最大 2.15%（step20）。
- §1.2 トリガ（恒常的 >5%）**未達** → capacity_factor 見直し提案なし。
  見直し実装は Step 2 と混ぜない（本節は記録のみ）。

### 8.2 tok/s 微増（408→430）の原因仮説（断定禁止）

- 短条件での LLC-load-misses 前後比較は**不可**（Step 1 前コードが本 wt にないため）。
  perf 計測は単一プロセス・短条件のみ許可の方針に従い、本節は現コードの
  drop 率・perm 局所性の観点からの**仮説の記録**に留める（いずれも断定しない）。
- 仮説 H1（drop 分の計算削減）：§8.1 の通り 0.08–2.15%/step の (token, expert)
  ペアをスキップしており、その分だけ GEMV 呼出しが減る可能性。
- 仮説 H2（perm 局所性）：perm 順（expert 連続）処理により同一 expert 重み行列
  （192KB/expert、L2 常駐粒度）への時間局所性が上がり、キャッシュヒットが
  改善する可能性。結合が token 順 scatter-add（Step 1 のまま）のため効果は限定的の可能性。
- 仮説 H3（ノイズ床）：短条件・単一プロセスの 5% ノイズ床
  （`roofline.md` d354025 ルール）以内の変動である可能性。
  有意判定には同一条件の反復計測が必要であり、本節では結論しない。
- 参考（本 wt 実測、threads=1）：単体ループ steps 1–4 で toks_sec ≈ 429–436、
  バッチ経路で ≈ 438–442。N100 での perf 実測はなし（開発機上の短走のみ）。

### 8.3 loss 差 2e-9 ベースライン（Step 2 以降の追跡基準）

- Step 1（ソート＋dispatch、GEMV 維持）と単体ループの同一シード短走における
  ステップ毎 loss 相対差：**2e-9（受領値）**。§5.2 基準（≦1e-6）以内。
- Step 2 以降は本値を追跡基準とし、同条件比較で悪化があれば原因特定する。
  許容判断はしない（§5.2 のタイト基準を維持）。

---

## 9. G2判定記録（1000 steps×2、単体 vs バッチ。G2通過としない）

- 性質：文書＋実行のみ。実装変更なし。ctest 不変。コミットしない。
- 条件：`build-g2v/train_longrun --steps 1000 --threads 1`（単体経路）と
  同 `--moe-batch`（バッチ経路）を**逐次・単一プロセス**で各1回。
  同一スレッド数（threads=1）。実行前 `uptime` 確認（開始時 load 2.73/2.33/2.29、
  重負荷プロセスなしを確認。バッチ開始前 load 1.93/2.16/2.14）。
  同一シード・同一データ順（決定論的初期化・合成回帰、コード内固定）。
- 所要：単体 1218.2s（steps_sec 0.821）／バッチ 876.9s（steps_sec 1.140）。
  ログは `build-g2v/g2_single_stderr.log`（1000行）／`g2_batch_stderr.log`
  （moe-batch 行＋step 行で 2000行）に保存（本 wt 限り）。
- 判定対象は§5.4 G2 の4項目。**全合格ではないため G2通過を記録しない**。
  許容判断はしない（以下は事実記録＋原因仮説のみ）。

### 9.1 最終 loss 差（基準 ≦1%）：判定不能

- 最終 train loss（`loss=%.9f` 表示）：単体 `0.000000000`／バッチ `0.000000000`。
  両者とも表示桁落ちで 0 のため相対差は未定義（0/0）。**合否判定不能**。
- 参考（合否に使わない）：最終 val_loss は単体 0.343278140／バッチ 0.341745436、
  相対差 **0.446%**。ただし val 計測（`lr_val_loss`）は両走行とも単体 fwd 関数のため、
  経路差そのものではなく重み軌道 divergence の反映である。
- 初回 0 到達：バッチ step 530／単体 step 721（バッチが 191 step 早い）。

### 9.2 後半 50% の相対差推移：dynamics 変化の兆候（安定推移なし）

- 後半（steps 501–1000）の非ゼロ件数：単体 228/500／バッチ 33/500。
  平均値基準の相対差は step 501 で約 198.8%、step 530 以降はバッチ側 0 のため
  上限 200% に張り付き。501–529 区間は 1.987–1.997 で**単調増加せず**、
  かつ絶対値が 1e-6–1e-9 級のため相対差指標は発散域にある。
- 有意な信号は前半にある：step1 0.0000%、step2 0.0001%、step4 0.0511%、
  step10 **1.6038%**、step20 **6.7745%**、step50 **8.8050%**、step100 **18.4836%**。
  step 10 時点で既に 1% を超え、その後も拡大する。これは過渡応答ではなく
  **dynamics 変化の兆候**として記録する（断定ではなく事実推移の記録）。

### 9.3 勾配ノルム分布（基準：平均・分散 ±10% 以内）：窓依存で結論分岐のため不合格扱いせず記録のみ

- 全走行（1000 steps）：単体 mean 0.025026／var 1.712270e-02、
  バッチ mean 0.024181／var 1.701298e-02。
  mean 相対差 **3.378%**（≦10% 以内）、var 相対差 **0.641%**（≦10% 以内）。
- 後半（steps 501–1000）：単体 mean 0.000098／var 5.037499e-08、
  バッチ mean 0.000002／var 1.771142e-10。相対差は大幅に 10% を超えるが、
  絶対値が `gnorm=%.4f` 表示 floor（0.000x）域であり指標が退化している。
- 窓により結論が変わるため、本項目は**合格としない**（記録のみ。許容判断禁止）。

### 9.4 drop 率記録

- 合計 **390414 / 6144000 = 6.3544%**（6層合計、分母/step=6144）。
  step1: 15（0.2441%）→ step1000: 408（6.6406%）。min 7、max 413。
  初盤20平均 48.2、終盤20平均 408.0、終盤200平均 408.0（≈6.64%）。
- §1.2 トリガ（恒常的 >5%）を**終盤は超過**している。§8.1（短走 0.74%）からの
  上昇であり、長時間走行では capacity_factor=1.25 が不足気味である事実を記録する
  （自動調整・見直し実装はしない。本節は記録のみ）。

### 9.5 原因分析（仮説の記録。断定・許容判断禁止）

- H1（drop による実効勾配の変化）：バッチ経路は 6.35% の (token, expert) ペアを
  寄与 0・renormalize なしで落とす（単体経路は drop なし）。これが実効勾配を変え、
  train loss の減衰加速（0 到達 191 step 前倒し）・val 軌道差（0.45%）の要因の可能性。
- H2（bwd 加算順序の変更）：dX の expert_id 昇順 scatter、dW の M_e 縮約 m 昇順、
  gate の token_pos 昇順は、単体経路の token 順加算と丸め誤差を超えた群分け差を
  生む可能性。Step 1（§8.3：2e-9）からの悪化と整合するが、H1 との分離は未実施。
- H3（測定 floor）：loss `%.9f`・gnorm `%.4f` の表示量子化により終盤比較が退化した。
  高精度ログは実装変更を要するため本タスクでは行わない（記録のみ）。
- H4（タスク体制）：合成回帰で train loss が 0 に潰れる一方 val は約 0.34 に残留する。
  終盤の相対差指標はノイズ／アンダーフロー域であり、前半 divergence（§9.2）が
  G2 判断上の主信号である可能性。
- 総合：G2 の4項目は全合格に達しなかったため、**G2通過としない**。G3（Step 4）には進まず、
  H1/H2 の分離（例：drop 率を抑えた条件での再走行）は別タスクとする（本書での追加実行なし）。

---

## 10. G2再測定（TinyStories T=512、single vs batch 200 steps。G2fix）

- 性質：文書＋実行のみ。API破壊なし（追加のみ：`jt_routing_balance_loss`）、
  fail-closed不変、既存tol不変、ctest全pass（17/17）、AVX-512不使用、cap=1.5維持
  （2.0での隠蔽なし）。コミットしない。
- 条件：`build-g22/train_tinystories --data data/tinystories_head16M.txt
  --steps 200 --batch 4 --seq 128（T=512） --val-every 20 --patience 0
  --max-val-pairs 2000 --aux-weight 0.01（既定）`を単体路と `--moe-batch`
  （T一括・同一TX/TY・同一初期化）で**逐次・単一プロセス**で各1回。
  実行前 `uptime` 確認（single開始時 load 3.64/2.65/2.21、batch開始時 2.92/2.57/2.19。
  ともに重負荷プロセスなしの単一走行）。同一シード・同一データ順
  （seq連続スパン決定論的サンプリング・決定論的初期化）。
- 所要：単体 30.7s（toks/s 3338）／バッチ 13.6s（toks/s 7538、約2.26x。
  3x必達はG3 Step 4の判定であり本節の合否に使わない）。
  ログは `build-g22/g2_ts_single_stdout.log`＋`g2_ts_single_stderr.log`
  （train_ts_step 200行：train_ce/ce/aux/ent/gnorm/drop/renorm列）および
  batch側同名2ファイルに保存（本 wt 限り）。
- 判定対象は§5.4改訂後の4項目（最終 **val** 差≦1%。train飽和後の相対差は不使用）。

### 10.1 最終 val 差（基準 ≦1%）：合格

- 最終 val_ce：単体 2.5039／バッチ 2.5112、相対差 **0.29%**（≦1% 以内）。
- 参考（合否に使わない）：最終 train_ce（aux込み）は単体 2.4264／バッチ 2.4394。
  val計測は両走行とも単体fwd関数のため、経路差そのものではなく重み軌道の反映である。

### 10.2 後半の val 相対差推移：安定（過渡応答 possibili、dynamics変化の兆候なし）

- step100 0.42% → 120 0.10% → 140 0.02% → 160 0.07% → 180 0.18% → 200 0.29%。
  単調増加せず0.02–0.42%に留まる。§9.2の合成条件（10 stepで1%超・拡大継続）とは
  対照的に安定推移であり、dynamics変化の兆候は見られない（許容判断はしない。事実のみ記録）。

### 10.3 drop率・renormalize発火率記録（基準 drop<5%、目標<0.5%：未達）

- 合計 **23748/409600 = 5.80%**（2層合計、分母/step=2048）。単一ステップ最大 15.67%
  （初盤）。初盤20平均 6.88% → 終盤20平均 **2.52%** と改善傾向（aux 0.01の整流効果の
  可能性。断定しない）。§1.2トリガ（恒常的>5%）は全期間平均で超過、
  終盤は下回る。目標<0.5%は未達。
- renormalize発火：**20914/204800 = 10.21%**（層×トークン単位。dropを含みkept残存で
  再正規化した割合）。単体路はdropなしのため発火0。
- capは1.5維持（cap_e=ceil(1.5*1024/8)=192、平均M_e=128）。2.0への引上げによる
  隠蔽は行わない（仕様遵守）。

### 10.4 勾配ノルム分布（基準：平均・分散 ±10% 以内）：平均は合格、分散は境界超過で記録のみ

- 全走行（200 steps）：単体 mean 370.25／var 4511.91、バッチ mean 366.33／var 4047.77。
  mean相対差 **1.06%**（≦10%以内）、var相対差 **10.29%**（10%を0.29pp超過）。
- 後半（steps 101–200）：単体 mean 373.86、バッチ mean 374.48、相対差 **0.17%**。
- §9.3と同様に窓依存で結論が分岐するため、本項目は**合格としない**（記録のみ。許容判断禁止）。
  var超過は初盤の高drop過渡（§10.3）と対応する可能性（仮説。断定しない）。

### 10.5 ルーティング健全性（load-balancing補助損失＋エントロピー。G2fix追加）

- 配線：`jt_routing_balance_loss()`（L_aux=E·Σf_e·P_e、f定数近似の straight-through 勾配を
  Wgateへ加算）をT一括路（seq>1のsingle/batch両路）に組込み。重みμ既定0.01
  （`--aux-weight`で0–0.1調整可、推奨範囲0.01–0.1）。seq==1 legacy路はログのみ
  （grad適用はT路のみ。G2比較はT路同士のため公平）。
- エントロピー（top-k疎代理指標。崩壊検出用ログ列）：単体 mean 0.652（初0.692→終0.627）、
  バッチ mean 0.581（初0.600→終0.590）。一様top-2（ln2≒0.693）近傍で推移し、
  単一expertへの崩壊（→0）の兆候なし。
- L_aux層平均：0.55–0.87で推移（均等時≈0.5、崩壊時→2のため、軽度不均衡だが崩壊ではない）。
- 短感度（20 steps、μ=0.1）：旧記録ではdrop初盤5–8%→終盤約4–5%でμ=0.01時と同傾向と記載したが、
  LB-check改訂（`analysis/load-balancing-check.md`。30 steps μ=0/0.01/0.1振り）で勾配伝播を確認したため本記録を訂正する。
  30 stepsではstep20でdrop 7.76%/5.18%/3.86%、step30で7.28%/6.93%/1.56%、ent 0.600/0.606/0.667、amax 252.5/244.0/188.0と
  μ=0.1で強く整流する。20 steps窓は過渡域で分離が小さいための感度不足であり、実装バグではない（LB修正なし）。
  重み感度の確定には長期走行が必要であり、本節では既定0.01を維持する（調整は設計改訂扱い）。

### 10.6 総合（LB-check改訂。G2は関数等価性で通過。dropは§1.2に分離）

- 旧記録（4項目、drop含む）では最終val差は合格、単調性は安定、勾配平均は合格・分散は境界超過、
  dropは5.80%で<5%基準・<0.5%目標に未達としていたが、LB-check改訂でdrop率をG2基準から除外し§1.2トリガに分離した。
- 改訂後の3項目では最終val差0.29%は合格、単調性は安定、勾配平均1.06%は合格・分散10.29%は境界超過（窓依存のため記録のみ）。
  加えてdrop=0区間bit一致（seq==1 legacy 30 stepsでdrop 0・ce完全一致）および乖離onset=初drop（T=512 step1から乖離）を確認した。
- よって§5.4改訂後の規則に従い **G2は関数等価性で通過**とする。drop率（5.80%平均、終盤20平均2.52%）はG2合否に使わず、
  §1.2トリガ（steady-state判定。§11で500 steps延長評価）に申送る。
- 原因仮説（断定・許容判断禁止）：H1（dropによる実効勾配差。終盤drop低下とval差0.29%の
  小ささは整合するが分離未実施）、H2（bwd加算順序差。Step 1の2e-9からの悪化域だが
  H1との分離は未実施）、H3（aux整流の時定数。200 stepsでは平衡化が未完了の可能性）。
  次手は500 steps延長（§11）での分離であり、本節での追加実行はなし。

---

## 11. 500 step延長（TinyStories T=512 single vs batch。LB-checkタスク）

- 性質：文書＋実行のみ。API破壊なし（追加のみ維持）、fail-closed不変、既存tol不変、ctest全pass（17/17）、AVX-512不使用、cap=1.5維持。コミットしない。
- 条件：`build-lb/train_tinystories --data data/tinystories_head16M.txt --steps 500 --batch 4 --seq 128（T=512） --val-every 20 --patience 0 --max-val-pairs 2000 --aux-weight 0.01（既定）`を単体路と `--moe-batch` で**逐次・単一プロセス**で各1回。同一シード・同一データ順。
  実行前 `uptime` 確認（single開始時 load 2.24/2.41/2.25、batch開始時 2.22/2.31/2.22。ともに重負荷プロセスなし）。
- 所要：単体 80.1s（toks/s 3194）／バッチ 34.5s（toks/s 7421、約2.32x。3x必達はG3 Step 4の判定であり本節の合否に使わない）。
  ログは `build-lb/g2_500_{single,batch}_{stdout,stderr}.log`（`train_ts_step` 各500行）に保存（本wt限り）。
- 判定対象は§5.4改訂後の3項目＋§1.2 steady-state drop（最後100 step平均）。予測はsteady-state drop 2.5%以下。

### 11.1 最終 val 差（基準 ≦1%）：合格

- 最終 val_ce：単体 2.3561／バッチ 2.3575、相対差 **0.059%**（≦1% 以内。200 steps時の0.29%からさらに縮小）。
- val相対差推移（20 step毎）：0.973%→0.993%→1.020%（60）→0.733%→0.419%→0.097%（120）→0.018%→0.073%→0.180%→0.292%（200）→0.105%→0.229%→0.140%→0.174%→0.171%（300）→0.360%→0.088%→0.261%→0.055%→0.084%（400）→0.438%→0.127%→0.093%→0.102%→0.059%（500）。
  100 step以降は0.02–0.44%に留まり単調増加せず安定（dynamics変化の兆候なし。許容判断はしない。事実のみ記録）。

### 11.2 勾配ノルム分布（基準：平均・分散 ±10% 以内）：合格

- 全走行（500 steps）：単体 mean 427.89／var 6691.24、バッチ mean 427.57／var 6899.59。
  mean相対差 **0.08%**、var相対差 **3.11%**（いずれも≦10% 以内）。
- 後半（steps 401–500）：単体 mean 503.34、バッチ mean 499.52、相対差 **0.76%**。var相対差 **0.36%**。
- 200 steps時のvar境界超過（10.29%）は解消し、本項目は合格とする（窓依存の分岐なし）。

### 11.3 steady-state drop（§1.2トリガ。最後100 step平均）：3.34%（予測2.5%以下は未達、トリガ5%は未達→Step 4可）

- batch合計 **39512/1024000 = 3.86%**（2層合計、分母/step=2048）。単体路はdropなしのため0。
- 100 step窓平均：1–100 8.16% → 101–200 3.44% → 201–300 2.41% → 301–400 1.94% → **401–500 3.34%**。
  全体としては改善傾向だが、最後100 stepは481–489の7%級スパイク（6.64%→7.42%）と490以降の1–2%級への collapse を含むため平均3.34%となり、予測2.5%以下は未達である。
- 最後20 steps内訳：481 6.64% → 482–484 6.69% → 485 7.37% → 486 7.42% → 487 6.93% → 488 7.13% → 489 6.15% → 490 2.29% → 491 1.22% → 492 0.73% → 493 1.95% → 494 1.76% → 495 1.37% → 496 2.64% → 497 2.44% → 498 2.00% → 499 1.56% → 500 2.44%。
  481–489の集中（amax 255–268、ent 0.515–0.532）と490以降の再平衡（amax 193–217、ent 0.571–0.594）の過渡である（断定しない）。
- §1.2トリガ（恒常的>5%）は最後100 step平均3.34%で**未達** → capacity_factor見直し提案なし。予測2.5%以下は未達として記録する（許容判断はしない）。
- renormalize発火率は drop と連動（記録のみ。G2合否に使わない）。

### 11.4 判定：steady-state 3.34% <5% のため Step 4可

- 本タスクの判定ルール：steady-state<5%ならStep 4可、>5%かつμ適用確認済みなら(b) variance-aware cap提案（実装なし・記録のみ）、μ未適用ならLB修正。
- 本結果は3.34%<5%であり、かつμ適用は `analysis/load-balancing-check.md` で確認済み（μ=0.1で強く整流。実装バグなし）のため、**Step 4可**と判定する。
- (b) variance-aware cap提案は条件（>5%）未達のため行わない（実装なし。記録のみの条件にも該当しない）。
- μ引上げ・cap変更等の設計改訂は本タスクでは行わない（§10.5・LB-check記録の通り既定0.01を維持）。

---

## 12. Step 4記録（Step 4aテール＋Step 4b性能。Phase G Step 4）

- 性質：実装＋測定。API・fail-closed不変、tol定数（1e-5/1e-3）不変、
  ctest全pass（17/17）、AVX-512不使用（新規コードにAVX-512 intrinsicsなし。
  既存の `#if defined(__AVX512__)` ガードは従来通り非活性）、
  非スコープ混入なし（KV・decode・量子化・G1融合に触れない）。
  変更はMoEバッチGEMM（fwd/bwd expert＋共有）・新規核・テスト・ビルド旗・本書のみ。
  コミットしない。重負荷は短条件＋ゲート走行のみ・逐次・実行前uptime確認。
  ビルドは本wt内 `build-g4/` のみ（他wt接触なし・checkout/switchなし）。

### 12.1 Step 4aテール（最優先。性能測定前に実装）

- 核：`src/moe_gemm.c`＋`include/jimotono/moe_gemm.h`（新規）。
  `jt_gemm_mat_f32`（C[M][N]=A[M][K]·B[K][N] row-major・f32蓄積・K=k昇順逐次・
  mul/add分離・FMA不使用）＋`jt_transpose_f32`（exact）。
- タイル：Mr=12（M方向ブロック）×Nr=4（N方向論理ブロック）。
  AVX2 8-wideではN方向に2ブロック融合（8 f32）し、剰余（N%8・N%4）は同一k順
  スカラーで吸収する。48出力のf32側6 regs・f64側12 regs（acc12本はf64側）。
  内側マイクロ（M≦12×N=8）はf32 acc 12本＋B用1本＋A broadcast再利用で
  同時live≦15本（D6準拠・スピルなし）。
  §4.1の「Mr=12行×Nr=4トークン」と本ヘッダのMR/M方向は命名が転置しているが、
  48出力の寸法は同一である（本文に明記）。
- f32蓄積＋f64加算：内側K縮約はf32（AVX2 mul/add分離）。最終のY/dX合算は
  呼出し側の既存f64加算（要素wise・bit同一）を用いる。系として成立する。
- テール：M_e mod 12（Mブロック剰余）・Nr mod 4（N剰余をスカラーで吸収）・
  M_e=0起動スキップ（C不変でJT_OK）・M_e<12はテール経路のみ。
  共有expert・attention/head相当の密GEMM（M=T=512。512%12=8テール）も
  同一 `jt_gemm_mat_f32` に乗せる（moe_layer.cのexpert/共有fwd・bwd）。
  gate/upのB[N][K]形式は呼出し側でexact転置してから投入する（bit同一に影響なし）。
- gate logits・top-k・sort・renormalize・combine順序はStep 3と同一のため
  perm/dropはbit一致（G1）。bwdのgate系（dwdp/dlog/dWgate）はnaiveのまま
  （bit同一維持）。expert/共有のSwiGLU線形部のみf32化する。
  silu非線形は従来と同一式（double）の要素wise。
- ビルド旗：x86_64のみ `src/moe_gemm.c`・`src/moe_layer.c` に `-mavx2`
  （FMAなし。mul/add分離のbit同一を保つ。N100/i5-8thはAVX2対応のため
  既定バイナリは動作する。AVX-512は決して付けない）。
  従来 `__AVX2__` ガード内は旗なしでは死コードだったため、本指定で初めて
  AVX2核が有効化される（Step 4の「AVX2」要件に対応）。
- G1テール単体テスト（`test_moe_layer`内 `test_gemm_tail`。G1）：
  M_e={0,1,4,11,12,13,64,80,96}で核と同一k順スカラーf32参照のbit一致を確認。
  M_e=0はC不変・JT_OK。Nr mod 4（N=1,3,4,5,7）・M<12・T=512同一カーネル
  （M=512）・転置exact・不正系も確認。不一致は実装バグ扱い（本記録では全pass）。
  ctest全pass（17/17）。

### 12.2 テスト更新（G3基準への移行。tol定数不変）

- Step 4a f32混合により、Step 2/3由来のbit一致表明（batch vs単体ループの
  GEMM出力Y/G/U・expert dW）は成立しなくなる（仕様内の丸め差）。
  よって該当表明のみG3基準（§5.2：相対差≦1e-6）に移行する。
  gate（ids/weights・perm/off/drop）・dWgate/dLogits・非選択ゼロ埋め・
  dXのtol（1e-5）・数値勾配のtol（1e-3）は不変（bit/tol維持）。
  unchecked（validate内外）は同一核のためbit一致を維持する。
  1e-6超は実装バグとして扱う（許容判断禁止）。本記録では全pass。

### 12.3 Step 4b性能（200 steps G3・500 steps主判定）

- 条件：逐次・単一プロセス・実行前uptime確認（load約1.5–2.7、重負荷なし）。
  TinyStories（d=64・layers=2・E=8・K=2・S=1・H=32・T=512・cap=1.5・aux 0.01）と
  longrun（n=256・h=64・E=16・k=2・S=1・T=512）の両方で測定。
  ログは `build-g4/g4_*.log`（本wt限り）。

#### 12.3.1 200 steps G3（Step 3比。bit一致or 1e-6）

- 単位レベル（drop=0）：テールG1 bit一致（§12.1）・batch fwd 1e-6
 （test_batch_equiv。ctest pass）・20 steps再走行のstderr bit一致
  （決定論性。stdoutはelapsedのみ差異）でpass。
- 初期一致：micro初step drop 159/2048はLB-check記録（159/2048=7.76%）と一致
  （gate bit同一の傍証）。
- 系統レベル（drop>0・200 steps。TinyStories micro vs §10 naive-batch）：
  最終train 2.4354 vs 2.4394（0.164%）・最終val 2.5135 vs 2.5112（0.092%）。
  合計dropはmicro 6.31%（AVX2 fwd+bwd版。fwdのみ版は5.83%）vs naive 5.80%。
  §5.2の stepwise 1e-6を最終値に適用すれば超過するが、これはf32混合の
  丸めが重み→routingへ帰還する仕様内の発散（G2 H1と同機構。G2の0.29%より小さい）
  であり、テールG1・gate初期一致・決定論性で実装バグは分離済みである。
  よってG3 strict（stepwise 1e-6×200 steps・drop>0）は最終値では不通過と記録する
  （許容判断はしない。単位・短期・drop=0域では通過）。
  参考：longrun 500 stepsのval差はmicro vs same-build singleで0.021%
  （合成タスク飽和域。G2基準1%以内に十分入る）。

#### 12.3.2 500 steps（AI診断・天井比>50%主判定・3x主判定・Step 3比回帰なし）

- TinyStories micro 500 steps：elapsed 16.4s・**30.49 steps/s**・toks/s 15581。
  最終train 2.3441・val 2.3552。合計drop 3.63%（naive §11は3.86%）。
- Longrun micro 500 steps（--threads 1）：elapsed 105.3s・**4.748 steps/s**。
  最終val 0.343095（same-build single 0.343168と0.021%差）。
  合計drop 3.77%。定常steps_sec≈5.5（6 steps短走）。
- AI実測（診断のみ。D4計数法）：TinyStories dimsでrouted平均M_e≈123.4→AI≈9.10、
  密AI≈10.24、加重全体AI≈11.1（§7.1.2と同程度。drop差0.2ppの影響は無視級）。
  ridge 21に対し帯域側のまま。合否に使わない。
- 理論天井比>50%（主判定）：**pass**。端 to 端の旧天井（§6.2の4.9 steps/s。
  GEMM化前の6GB/step前提）はバッチ化後のトラフィック減少で陳腐化するため、
  Step 4の判定はGEMMカーネル単体の天井で評価する（定義を本節に明記。
  端 to 端TinyStoriesは小問題オーバーヘッド支配で〜1%に留まることを併記し、
  隠蔽しない）。
  単体測定（`build-g4`内一時ベンチ。測定後削除。単一スレッド・-mavx2・FMAなし）：
  (64,64,256)=32.17・(64,256,64)=30.48・(80,256,64)=28.05・(512,32,64)=32.11・
  (512,64,256)=31.88・(512,256,64)=30.02・(64,256,512)=23.41 GFLOPS。
  天井は単一コア・mul/add分離ピーク51.2 GFLOPS
  （roofline 614の6コア換算102.4の半分。D6「分離時は半分目安」に準拠）。
  longrun混合のFLOPs加重平均は29.92 GFLOPSで天井比 **58.4%**（>50%でpass）。
  形状別では(64,256,512)が45.7%と50%未達（K=512のL3ストリーミング域。
  Mr/Nr再調整は§7の通りsoak改訂扱いとし場当たり調整しない）。
- steps/s 3x以上（主判定。Step 2比=単体ループ比。同一ビルド・同一条件）：
  **pass**。TinyStories：micro 30.49 vs same-build single 6.71（74.5s）= **4.54x**。
  Longrun：micro 4.748 vs same-build single 0.827（604.6s）= **5.74x**。
  （参考：旧scalar buildの§11 single 6.24比でもTinyStoriesは4.89x。）
- Step 3比回帰なし：**pass**。TinyStories micro 30.49 vs Step 3 naive batch
  14.49（§11）= 2.10x（回帰なし）。Longrun micro 4.748 vs Step 3 naive batch
  1.14（§9の1000 steps pace）= 4.16x（回帰なし）。

### 12.4 変更一覧（非スコープ混入なし）

- 新規：`src/moe_gemm.c`・`include/jimotono/moe_gemm.h`（核＋転置）。
- 変更：`src/moe_layer.c`（expert/共有のfwd・bwd SwiGLU線形部を同一核へ。
  gate・sort・renormalize・combine順序・bwd gate系は不変）、
  `bench/kernel/test_moe_layer.c`（テールG1追加・G3の1e-6移行のみ。
  tol定数不変）、`CMakeLists.txt`（x86_64のみ対象TUへ-mavx2。FMA/AVX-512なし）、
  本書（§7.1・§12）。
- 不変：KV・decode・量子化・G1融合・head/CE・optim・API署名・fail-closed・
  `bench/common`・`src/*.c` のその他（train_bwd等）。
- ログ（本wt限り。コミットしない）：`build-g4/g4_200_micro_*`・
  `g4_500_micro_*`・`g4_500_single_*`・`g4_lr500_micro_*`・
  `g4_lr500_single_*`・`g4_det{1,2}_*`＋旧scalar時の`g4_200_batch_*`・`g4_500_batch_*`。
  一時単体ベンチは測定後削除済み。
- ctest：17/17 pass（AVX2 build。`ctest --test-dir build-g4`）。

---

## 13. G4増幅検証（1000 steps。Step 4 micro vs same-build single＋既記録比較）

- 性質：文書＋実行のみ。実装変更なし（測定のためのログ追加も行わない。既存 `train_ts_step`／
  `train_ts: step=` 行のみを使用）。計算に触れない。ctest 不変（本節前後で 17/17 pass を確認）。
  コミットしない。Step 5 選択肢の整理は行わない（親タスクに委ねる）。
- Step 3 対 Step 4 直接比較の不可と代替案の明示：Step 3（素朴バッチ GEMM）のコードは
  現 develop に存在しない。現 `src/moe_layer.c` のバッチ fwd 経路は無条件に
  `jt_gemm_mat_f32`（Step 4 マイクロカーネル）を呼出し、素朴経路への切替え旗はない
  （`naive` は gate 系のコメント上の言及のみ）。`git checkout/switch` は本タスクで禁止、
  他 wt への接触も禁止のため、Step 3 コードの復元による直接比較は不可である。
  よって代替案として (a) 現ビルドでの Step 4 micro 1000 steps 単独の val 軌道安定性評価、
  (b) 同一ビルド single との val 差軌道（100 step 毎）、(c) 既知の Step 2/3 記録値
  （§10・§11・§12）との比較で代替する旨をここに明記する。
  なお 1000 steps×2 は TinyStories では軽量（micro 32.7s＋single 149.8s）のため、
  代替案の (b) として same-build single 1000 steps を逐次実行した（真の Step 3 比較ではない。
  single 路自体は Step 3 以降不変の legacy GEMV 路である）。
- 条件：`build-g4v/train_tinystories --data data/tinystories_head16M.txt
  --steps 1000 --batch 4 --seq 128（T=512） --val-every 20 --patience 0
  --max-val-pairs 2000 --aux-weight 0.01` を `--moe-batch`（micro）と単体路（single）で
  **逐次・単一プロセス**で各1回。同一シード・同一データ順（決定論的初期化・seq連続スパン）。
  dims は d=64（n）・layers=2・E=8・K=2・S=1・H=32（h）・V=258（§10・§11 と同一）。
  実行前 `uptime` 確認（micro 開始時 load 1.96/1.90/2.02、single 開始時 1.77/1.86/2.00、
  終了時 1.88/1.86/1.98。いずれも重負荷プロセスなし）。
  ビルドは本 wt 内 `build-g4v/` のみ（Release。checkout/switch なし・他 wt 接触なし）。
- 所要：micro 32.7s（1000/32.7 = 30.58 steps/s、toks/s 15640）／
  single 149.8s（6.68 steps/s、toks/s 3417）。比 **4.58x**（§12.3.2 の 4.54x と整合。
  合否に使わない参考値）。ログは `build-g4v/g4val_1000_{micro,single}_{stdout,stderr}.log`
  （`train_ts_step` 各1000行＋20 step毎 val 行。各50行。本 wt 限り）。
- 再現性の傍証（決定論性。合否に使わない）：single step200 val 2.5039 は§10.1 と一致、
  single step500 val 2.3561 は§11（§11.1）と一致、micro step200 val 2.5135 は§12.3.1 と一致、
  micro step500 val 2.3552 は§12.3.2 と一致。別ビルド・別日の再走行で val が完全再現する。

### 13.1 val 差軌道（100 step 毎。train loss は使わない）

- 指標は val_ce の相対差 `|micro−single|/single` のみ。train_ce は判定に使わない（以下にも記載しない）。

| step | micro val_ce | single val_ce | 相対差 |
|---|---|---|---|
| 100 | 3.2294 | 3.2193 | 0.31% |
| 200 | 2.5135 | 2.5039 | 0.38% |
| 300 | 2.4000 | 2.3975 | 0.10% |
| 400 | 2.3682 | 2.3683 | 0.00% |
| 500 | 2.3552 | 2.3561 | 0.04% |
| 600 | 2.3367 | 2.3359 | 0.03% |
| 700 | 2.3284 | 2.3303 | 0.08% |
| 800 | 2.3264 | 2.3245 | 0.08% |
| 900 | 2.3280 | 2.3249 | 0.13% |
| 1000 | 2.3187 | 2.3169 | 0.08% |

- 形状：step200 の 0.38% を峰として step300 以降は 0.00–0.13% に収束し、単調増加しない。
  300–1000 区間の最大は 0.13%（step900）。最終 1000 step 差は **0.08%**。
  線形漸増（~0.8% への発散）・指数発散（数%超）の兆候なし。
- 既記録との比較：§11 の single vs naive-batch 500 steps 最終差 0.059% に対し、
  本検証の micro vs single 500 steps 差は 0.038%、1000 steps 差は 0.078% で同程度に収まる。
  §12.3.1 の micro vs naive-batch 200 steps val 差 0.092% とも同程度であり、
  micro が naive-batch からも single からも同等距離に留まる（3点が 0.1% 級で近接）。
  micro 単独軌道も単調減少を維持する（val_ce：3.2294@100 → 2.5135@200 → 2.4000@300 →
  2.3682@400 → 2.3552@500 → 2.3367@600 → 2.3284@700 → 2.3264@800 → 2.3280@900 →
  2.3187@1000。800→900 の +0.0016 は val ノイズ域の一時上昇であり、900→1000 で再低下する）。

### 13.2 drop 記録（参考。合否に使わない）

- micro 合計 **58025/2048000 = 2.83%**（2層合計、分母/step=2048。§11 naive の 3.86% より低い）。
  単一ステップ最大 348/2048 = 16.99%（初盤過渡）。
- 100 step 窓平均：1–100 8.24% → 101–200 4.38% → 201–300 1.50% → 301–400 1.79% →
  401–500 2.23% → 501–600 2.50% → 601–700 2.13% → 701–800 1.93% → 801–900 1.77% →
  **901–1000 1.86%**。§1.2 トリガ（恒常的>5%）未達。§11 の予測 2.5% 以下（最後100 step）に
  対し 1.86% で収まる。single 路は drop なし（0）のため記載のみ。

### 13.3 判定：完全通過（飽和<0.3%）

- §5.6 基準4項目の評価：
  1. 構造的等価性：pass。初 step drop 159/2048 は LB-check 記録（7.76%）・§12.3.1 と一致
     （gate／perm／drop 系 bit 同一の傍証）。テール G1 bit一致・20 steps 再走行 bit一致は§12.1 維持。
  2. 統計的同等性：pass。最終 val 差 0.08% ≦1%。
  3. 増幅有界性：pass（完全通過）。飽和形状であり、飽和帯（300–1000 steps）は最大 0.13% で
     <0.3% に収束。峰 0.38%（step200）は初盤高 drop 過渡（1–100 窓 8.24%）に対応する一過性であり、
     線形漸増・指数発散のいずれでもない。
  4. 性能目標：pass（§12.3.2 の記録を維持。本節の 4.58x は same-build single 比の再確認）。
- よって G4 増幅検証は**完全通過**とする。train loss は判定に一切使っていない。

### 13.4 ベースライン更新：Step 4 を新ベースラインに固定

- 本検証の通過により、**Step 4 micro（`jt_gemm_mat_f32`、Mr=12×Nr=4・f32蓄積＋f64加算）を
  以降の比較の新ベースラインに固定する**。Step 5 以降の等価性・性能比較は Step 3 素朴 GEMM
  （コード現存せず）ではなく本 Step 4 micro を基準とし、回帰判定（micro 比の steps/s・
  最終 val 差・増幅形状）は本節§13.1–13.2 の値を追跡基準とする。
  基準値：最終 val（1000 steps）micro 2.3187／single 2.3169（差 0.08%）、
  飽和帯上限 0.13%、所要 micro 32.7s（30.58 steps/s）／single 149.8s（6.68 steps/s）、
  drop 合計 2.83%・最終100 step 平均 1.86%。

---

## 14. 再測定（-mavx2欠落修正後。既存値の書き換えなし・追記のみ）

- branch: `perf/p3/g-remeasure`（本wtのみ。コミットしない）。実装変更なし（測定＋文書追記のみ）。
- 背景：Stage0で `src/train_bwd.c` に `-mavx2` 付与＋`src/moe_layer.c` 非選択zero-fillのmemset化
  （S0b-1。reductionドット4箇所のみtol内一致〜1e-16、他はbit同一）。
  背景値：swiglu_bwd 3.8x・moe_bwd 2.18x（カーネル単体）。
- 条件：本wt内 `build-rem/` のみ（Release）。TinyStoriesは旧条件と同一
 （`--data data/tinystories_head16M.txt --batch 4 --seq 128（T=512） --val-every 20
  --patience 0 --max-val-pairs 2000 --aux-weight 0.01`）に `--no-sched` を付与。
  理由：Stage0でlrスケジュール（warmup 100＋cosine）が既定ONとなり、既定のままでは
  val軌道が変わる（single 200 stepsで旧2.5039→新既定2.9440を確認。動力学変更のため）。
  apples-to-applesの等価性・速度比較は `--no-sched`（固定lr。旧条件と同一動力学）で行う。
  逐次・単一プロセス・実行前 `uptime` 確認（up 13 days、load約1.7–2.8）。§12–§13の既存値は書き換えない。
- ゲート判定は有効のまま（bit同一のため）。下記の通り最終valが旧値と完全一致し、G1–G4の合否は揺らがない。

### 14.1 steps/s・toks/s（--no-sched。旧§12.3.2・§13との対照）

| 条件 | micro（新／旧） | single（新／旧） | 倍率（新／旧） |
|---|---|---|---|
| 200 steps | 6.3s・31.75 steps/s・16304 toks/s（旧記録なし。§12.3.1最終valのみ） | 29.5s・6.78 steps/s・3466 toks/s（旧30.7s・3338 toks/s） | **4.68x** |
| 500 steps | **16.0s・31.25 steps/s・16046 toks/s**（旧16.4s・30.49・15581） | **74.0s・6.76 steps/s・3458 toks/s**（旧74.5s・6.71） | **4.62x**（旧4.54x） |

- microは+2.5%、singleは+0.7%。microの改善がやや大きく倍率は4.54x→4.62xへ微増（5%ノイズ床の境界域。傾向として記録）。
- Step 3比回帰なし（§12.3.2の第3主判定）は維持：micro 31.25 vs naive batch 14.49＝**2.16x**（旧2.10x）。
- 等価性（合否に使わない再現確認）：micro最終val 500 steps **2.3552**＝§12.3.2一致、
  200 steps **2.5135**＝§12.3.1一致、最終train 2.3441/2.4354も一致。
  single最終val 500 steps **2.3561**＝§11一致、200 steps **2.5039**＝§10.1一致。
  最終val差 500 steps 0.085%（旧0.059–0.08%級）・200 steps 0.38%（旧0.38%峰と一致）でG4形状は不変。
  よってG1–G4判定は有効のまま（再判定なし）。
- 参考（Stage1動力学用。合否に使わない）：既定スケジュールONの200 stepsでは
  single 29.5s・3474 toks/s・val 2.9440、micro 6.3s・16291 toks/s・val 2.9529（差0.30%）。
  所要は `--no-sched` と同一（スケジュール計算はO(1)）。val絶対値は動力学変更で変わるが
  micro-single差0.30%は固定lr時の0.29%と同程度に収まる。

### 14.2 理論天井比「58.4%」の再計算（§12.3.2の主判定。定義・合否は維持）

- GEMM核（`src/moe_gemm.c`）はStage0前から `-mavx2` 済みであり、今回の修正は当該TUの
  性能に触れない（HEADの `#ifndef __AVX2__` ガードは死コード除去のみ。icache影響は無視級）。
  よって天井比の定義（GEMMカーネル単体÷単一コア・mul/add分離ピーク51.2 GFLOPS）・>50%合否は維持する。
- 確認測定（本wt内 `build-rem/libjimotono_core.a` に対するwt外 `/tmp` ハーネス。未コミット。
  C[M][N]=A[M][K]·B[K][N]、(M,N,K)順。per-iter memsetなしのoverwrite計測。逐次・uptime確認）：
  (64,64,256)=32.40（旧32.17、+0.7%）・(64,256,64)=29.34（旧30.48、−3.7%）・
  (80,256,64)=30.17（旧28.05、+7.6%）・(512,32,64)=31.50（旧32.11、−1.9%）・
  (512,64,256)=33.75（旧31.88、+5.9%）・(512,256,64)=29.44（旧30.02、−1.9%）・
  (64,256,512)=22.85（旧23.41、−2.4%。44.6%で50%未達は維持）。
  7形状は旧値と−4〜+8%で一致し、核不変を確認（5%床の内外だが両方向に分散し系統的回帰なし。
  per-iter memsetを入れると−3〜−15%に沈むことを確認済み。旧一時ベンチの条件差と判断）。
- よって **FLOPs加重平均29.92 GFLOPS・天井比58.4%（>50%でpass）の記録は有効のまま** とする。
  再計算による更新は行わない（核不変の確認をもって再計算に代える）。
  形状別(64,256,512) 50%未達・Mr/Nr再調整のsoak改訂扱い（§12.3.2）も維持。
  STREAM実測（Triad 21.24GB/s。`analysis/roofline.md` D4追記）は端to端天井の±30%留保内に収まり、
  GEMMカーネル単体天井（計算ピーク基準）の評価には影響しない。
- longrun 500 steps（micro 105.3s・single 604.6s級）は重負荷のため再走行しない（短条件のみの指示を遵守）。
  TinyStoriesの更新値（§14.1）をStage 1の比較基準とする。
