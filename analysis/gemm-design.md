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
  デフォルト `capacity_factor = 1.25`（調整可能。1.0–1.5 の範囲で P3 実測確定、前例は Tutel 動的 factor）。
- 各 expert への割当てが `cap_e` を超えた分は **drop**（当該 (token, expert) ペアの寄与を 0 とし、
  残りの選出 expert の重みは renormalize **しない** — 学習ダイナミクスを変えないため。
  renormalize 有無はハイパーパラメータではなく仕様として固定し、変更時は設計改訂とする）。
- 共有 expert（常時オン2つ、AGENTS §2.1）は capacity 制限の対象外（§2.3）。
- drop 率・expert 別ヒストグラムをステップ毎にカウンタ記録する（デバッグ用。ホットパス外の集計のみ）。
  drop 率が恒常的に >5% の場合は capacity_factor の見直しトリガ（自動調整はしない）。

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
- `dX[token_pos] += dX_e` で scatter-add 復元（k=2 のため同一 token に2寄与が加算される。順序は加算の可換性により
  決定論的合計が保証される範囲で任意。§5 の許容差で規定）。

### 3.3 gate 勾配の順序扱い

- gate logits の勾配（top-k 経路）は **元の token 順序で**計算・加算する。
  expert 順バッファ上では中間量（`w[m]`, `dY·Y_e` の内積等）を保持するのみとし、
  logits への最終 scatter は `token_pos` 順に直列化する。
- top-k の hard 選択自体は微分不可のため straight-through（選択マスク固定）の扱いを維持し、
  本書でゲーティング関数の変更は行わない。
- drop されたペアの gate 勾配は 0（寄与なし）。renormalize しない仕様（§1.2）と整合させる。
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

---

## 6. 目標との対応：AI>21、理論天井比>50%、steps/s 3x以上

### 6.1 AI > 21（ridge 越え）

- roofline D6 の式：AI = hnT / 2(hn + T(n+h))。n=256, h=64 で **T=512 なら AI ≒ 23**。
- 本書の対応：
  - 共有 expert・attention：M=T=512 の密 GEMM で上式通り AI ≒ 23 を狙う。
  - routed expert：M_e ≈ 64（平均）でも AI = hnM_e / 2(hn + M_e(n+h)) ≒ 13–15 に達し、
    ridge 21 には単体で届かないが、cap 上限（~80）＋重み常駐（L2 192KB）により実効 AI は向上する。
    全体（routed＋共有＋attention）の加重平均で **AI > 21 を達成見込み**。
    未達の場合は capacity_factor 引き上げ（drop 減＋M_e 増）または T 増が調整ノブ（設計変更として記録）。
- 測定方法：D4 と同一の計数法（積和=2、fp32=4B、sigmoid/exp≈20 FLOP）で FLOPs／バイトを再計数し、
  expert 別 M_e 実測値を入れて加重 AI を算出する。推定帯域（29.4GB/s、STREAM 未計測±30%）ではなく
  **AI 値そのもの**で合否判定する（帯域誤差に不感なため）。

### 6.2 理論天井比 > 50%

- 定義：`達成 steps/s ÷ roofline 理論天井 steps/s`。理論天井は
  `min(peak_FLOPs / FLOPs_per_step, eff_BW / bytes_per_step)` のうち支配項（現状は帯域項）。
- GEMM 化後は AI ≈ 23 > ridge 21 により計算律速側へ移行し、天井は FLOPs 項で決まる。
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
   ゲート：§6 の3目標（AI>21、天井比>50%、3x）の最終合否＋§5.3の明記文書の完成。

各ステップの成果物はコード差分ではなく**測定記録**（AI 再計数表、loss 差分 CSV 要約、steps/s 表）とする。
本書は設計のみであり、実装着手は別タスクとする。
