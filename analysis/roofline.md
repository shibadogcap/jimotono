# Roofline 分析メモ — Phase D4–D7（性能分析担当）

- branch: `perf/p2/roofline`（本wtのみ。コミットしない）
- 前提: AVX2化済み・2Tピーク **1270 toks/s**（`results/bench/20260914-phased-val.md`）
- 任務: D2/D3の「LLC thrash＋帯域」は現象記述であり原因特定ではない。切り分けのみ行う。**量子化に着手しない。実装変更しない。**

## リッジポイント（前提・共通）

- ピーク 614 GFLOP/s、実効帯域 29.4 GB/s（= 42×0.7、STREAM未計測）→ **ridge ≈ 20.9 ≒ 21 FLOPs/byte**
- 注意: 614 GFLOP/sは前提として受領（i7-8700Bの素朴なFMA換算とは一致しないが、本分析の結論は不感 —
  全カーネルがridgeの1/20以下であり、ピークが半分でも判定は変わらない。§D4末尾参照）。
- 5%未満の差はノイズとして扱う（`d354025 docs(rule): 5pct noise floor`）。

---

## D4 — 各カーネルの算術強度（コード読解、fp32バイト）

対象 dims: `train_longrun` 実働値 **n=256, h=64, E=16, k=2, S=1**
（`bench/kernel/train_longrun.c:41-59`）、GDN-2は論文既定 **dk=dv=128**（`include/jimotono/gdn2.h:32`）。
FLOPsは積和=2で計数。バイトはfp32=4B。sigmoid/expは約20 FLOP/要素の概算（順位に影響しない範囲）。

### 判定表

| カーネル | FLOPs | メモリトラフィック (fp32) | 算術強度 | 左右 | 備考 |
|---|---|---|---|---|---|
| GEMM相当ドット積 (h×n, swiglu G/U・gate・headの単位) | 2hn = 32,768 | 4(hn+n+h) = 66,816 B | **0.49** | **左（帯域）** | `src/train_bwd.c:970-981`、`src/moe_layer.c:190-210`。X再利用を数えても同値 |
| SwiGLU fwd (1 expert) | 6hn+20h ≒ 99,584 | W 3hn+X+G/U/Y ≒ 199,168 B | **0.50** | **左（帯域）** | `src/train_bwd.c:926-1042`。G/U/Yの3行列走査 |
| SwiGLU bwd (1 expert) | 9hn+30h ≒ 149,376 | R+W 両方 ≒ 396,800 B | **0.38** | **左（帯域）** | `src/train_bwd.c:693-839`。dX(4hn)+dW(3hn)+acc(2hn) |
| GDN-2 decode 1 step | 9·dk·dv ≒ 147,456 | S 7パス+ベクトル ≒ 461,824 B | **0.32** | **左（帯域）** | `src/gdn2.c:252-434`。S=64KBを7回なめる構造 |
| RMSNorm fwd | 4n+20 ≒ 1,044 | 2·X+W+Y = 4,096 B | **0.25** | **左（帯域）** | `src/train_bwd.c:1062-1123`。X 2パス構造 |
| RMSNorm bwd | 9n = 2,304 | 5n = 5,120 B | **0.45** | **左（帯域）** | `src/train_bwd.c:864-923` |

### D4判定

- **全カーネルが左（帯域律速）**。ridge 21に対し最大でも0.50（**1/42**）、最小0.25（**1/84**）。
- 順位（>5%差のみ有意）: SwiGLU fwd ≒ GEMMドット（0.50/0.49、差2%→**ノイズ、同位**）>
  RMS bwd（0.45）> SwiGLU bwd（0.38）> GDN-2（0.32）> RMS fwd（0.25）。
- 頑健性: ピーク614→307 GFLOP/sに半減してもridgeは10.4で、依然全カーネルは1/20以下。STREAM未計測の帯域誤差（±30%）でも判定不変。
- Phase D実測との整合: 1TでAVX2 1.96xだが2Tで飽和（1270 peak）＝ **1コアでは計算が効くが、2コアで帯域の天井に到達**。全カーネル左という本分析と一致。
- T-MAC LUT参考値（現行longrunはdense実行のため参考）: bits=4・K=256で重み側1024B→idx 256B（4×減）だが、
  GEMV体制ではQLUT構築（act走査+8×16列挙、`src/tmac.c:230-385`）が出力1点あたりに amortize されず支配的。
  LUTが効くのはM（同時出力）が大きいGEMM体制のみ。→ D6へ。

---

## D5 — N150（ssh `ubuntu`）perf計測：LLC-load-misses / LLC-loads

### 手順（実施済み）

1. `~/jimotono-bench` が旧版（`train_longrun.c`欠落・hash不一致）だったため、指示通り
   `git -C /Users/shibadogcap/jimotono archive develop | ssh ubuntu tar -x` で同期（md5一致確認）。
   ビルドは `cmake -S . -B build` + `--target train_longrun`（bench dir内のみ）。
2. `perf` は paranoid=4 で非特権不可 → パスワードレスsudo確認後に `sudo perf stat` で計測。
   `perf stat -e LLC-loads,LLC-load-misses,cache-misses,context-switches,cpu-migrations -- ./build/train_longrun --steps 6 --threads {1,2,4}` を**逐次・単一プロセス**で実行。
3. N100側の書き込みは `~/jimotono-bench` 内（+ビルド物）のみ。一時ログは削除済み。

### 実機値（`lscpu`）

- CPU: **Intel N150**（N100ではない）、4C/4T、L1d 32KB×4、**L2 2MB（共有）、L3 6MB（共有）**。
- ⚠️ 前提の「L3=12MB」は誤り。**実値6MB**で以下の溢れ点計算を行う。

### LLCミス率表（--steps 6）

| threads | LLC-loads | LLC-load-misses | **ミス率** | cache-misses（総LLCミス） | context-switches | cpu-migrations |
|---|---|---|---|---|---|---|
| 1 | 92,961,328 | 10,614,024 | **11.4%** | 525,196,535 | 614 | 0 |
| 2 | 131,229,244 | 13,930,397 | **10.6%** | 1,371,174,180 | 663 | 3 |
| 4 | 200,832,276 | 7,538,816 | **3.8%** | 1,444,080,500 | 4,871 | 9 |

### 作業集合 vs L3=6MB（`train_longrun`実働値から計算）

- 重み総量: n_total=5,038,337 → **19.2MB**（L3の3.2×）。重み+勾配=**38.4MB**。
- fwd中間キャッシュ [L][TOKS]: **10.6MB**。活性 (L+1)×512×256: **3.5MB**。optim8 state: ~10MB。
- 1ステップ常駐の総和 ≒ **65MB ≫ 6MB**。フル常駐は不可能。
- 溢れ点の内訳:
  - 1 expert分（gate/up/down 3行列）= **192KB** → L2（2MB共有）に十分収まる。✅
  - 1層分 W+G = 3.36MB×2 = **6.7MB** → **L3=6MBをわずかに超過**。層粒度が溢れ点。
  - 層ループ外側・トークン内側（`lr_fwd_all_range`、`train_longrun.c:314-323`）のため、
    層重み3.36MBは512トークンで再利用されL3ヒットし、bwdの勾配分で溢れる構造。→ ミス率~11%の説明になる。

### D5判定

- **ミス率30%超の条件を満たさず（最大11.4%、4Tで3.8%）。LLCスラッシング主因説は棄却。**
- D2/D3の「LLC thrash」は現象の言い換えであり原因ではない — 本計測がそれを裏付けた。
  真の律速は下記の**ストリーミング帯域**。
- 裏付け（独立した2経路の一致）:
  - `cache-misses` 総量 ÷6 steps ≒ **87.5M lines/step ×64B ≒ 5.6GB/step**。
  - コード読解からの推定: fwd 1.73GB + bwd 3.47GB + reduction/optim/norm ≒ **~6GB/step**。→ **一致（±10%）**。
  - 需要ロードミス（LLC-load-misses）は113MB/stepに過ぎず、残りはHWプリフェッチがカバーしたストリーミング。
    すなわち **latencyではなくbandwidthの天井**。
- 副次所見: 4Tでdemandミス率は下がる（層重みのスレッド間共有が効く）が `cache-misses` 総量とcsw（4,871）が増加。
  D2の「4T~608・6T 498」逓減はこのコヒーレンストラフィック+同期コストの側面が濃い（LLCスラッシュではない）。

---

## D6 — GEMMマイクロカーネルのレジスタ/キャッシュブロッキング（分析のみ）

参照: nanogemm（Web確認）— AVX2 **6×16マイクロカーネル、acc 12本（ymm0–ymm11）+ B用2本**、16 YMMにスピルなしで収容。
Mc=64, Nc=128, Kc=128のL1/L2タイリング。FMA使用だが本件はbit同一要請でmul/add分離（`src/tmac.c:36-39`）のため、
スループットはその半分目安。

### 現実の確認：対象はドット積（GEMV）である

- 現行 `jt_bwd_avx2_dot`（`src/train_bwd.c:150-169`）はf64×4レーンのreductionでacc実質1本。
  重みは1出力あたり1回しか使われない → **レジスタブロッキングの効き所（重み再利用）が存在しない**。
  AVX2化で1T 1.96x出たのは「メモリ→レジスタの8倍幅ロード」の効果であり、acc本数の効果ではない。
- 結論: **GEMVのままではacc 12本化してもAI上限は0.50のまま**。マイクロカーネル投資の前にGEMM化（トークンバッチ）が必須。

### キャッシュブロッキング理論値（i7-8700B: L1 32KB / L2 256KB、N150実値: L1d 32KB/core、L2 2MB共有）

- L2タイル（256KB目安）: **expert単位 192KBが自然タイル**（1 expertの3行列がL2に収まる）。
  層単位3.36MBは不可。→「expert固定・トークン走査」の順序が正しい（現行 `moe_layer` 呼び出しはトークン外側のため逆。将来の並べ替え余地）。
- L1マイクロタイル（32KB目安）: X 1KB + 出力タイル。提案 **Mr=12行（h方向）× Nr=4トークン**:
  f32 8-wideで acc 12本 = 96出力保持（384B）+ 重みロード1本 + Xブロードキャスト2本 = 15本 ≤ 16 ✅スピルなし。
  f64累積版では4-wide×12本=48出力で2セット要（レーン圧が倍）→ f32蓄積+最終f64加算の混合が現実的。
- GEMM化時のAI（T同時トークン）: AI = hnT / 2(hn + T(n+h))。n=256,h=64で **T=512なら AI ≒ 23 → ridge 21の右側に到達可能**。
  すなわちトークンバッチは唯一の「計算律速側へ渡る道」だが、現行逐次トークン処理では到達不能。
- L3 6MB（N150）でのタイル上限: 層重み3.36MB + 勾配タイル + 活性で6.7MB超 → **層融合+勾配遅延書き戻し**が溢れ点回避の条件（分析のみ）。

### D6判定

- acc 12本目安はnanogemm実績と整合し、16 YMMに収まることは確認。ただし現GEMV体制では無効。
- 優先すべきはマイクロカーネルではなく **ループ順序（expert固定・トークンバッチ）** であり、それがD7の融合議論に接続する。
- **実装変更はしない**（本メモまで）。

---

## D7 — 未融合ペア列挙・削減見積・優先度（実装しない）

スコープ注意: `train_longrun` ホットループは **MoE fwd/bwd + head + optim8** のみ。
RMSNorm・GDN-2はカーネルとして存在するがlongrunでは未使用 → 将来（1B decode）向けに別枠評価する。

### longrun内ペア

| # | 未融合ペア | 削減トラフィック/step（推定） | AI向上 | 優先度 |
|---|---|---|---|---|
| F1 | SwiGLU Y出力→MoE重み付き和→残差加算の3パス融合（`moe_layer.c:223-294` + `train_longrun.c:295-302`） | YpのW+R/エキスパート ≒ 3exp×512×6×256×4×2 ≈ **18–24MB** | 0.50→0.52（重み支配で希釈） | **MEDIUM** |
| F2 | SwiGLU dX累積とdW外積のX/dY再読込融合（`train_bwd.c:765-833`） | X+dY再読 ≒ 512×6×3exp×2KB ≈ **18MB** | 0.38→0.40 | **MEDIUM** |
| F3 | SwiGLU G/U 2ドットのX単一パス化（`train_bwd.c:967-981`） | X再読 1KB×3exp×512×6 ≈ **9MB** | 0.50→0.51 | **LOW–MEDIUM** |
| F4 | gate logits→topk→softmax小融合 | ~数十KB | 微小 | **LOW** |
| F5 | gnorm→clip→optim8更新の全域3パス融合 | ~40MBだがstep1回・非ホット | — | **LOW** |

分母は ~6GB/step のため、**F1–F3合計でも削減は約1%**。帯域律速下では活性融合は効かない。

### 将来（1B decode / GDN-2・RMSNorm使用時）ペア

| # | 未融合ペア | 削減トラフィック/トークン（推定） | 優先度 |
|---|---|---|---|
| G1 | GDN-2 decode内 S の7パス→2パス融合（scale+erase+write+readout、`gdn2.c:314-428`） | S 5パス分 = 5×64KB = **320KB/層**。20線形層で**6.4MB/トークン** | **HIGH（decode時）** |
| G2 | RMSNorm+GEMM融合（r確定後のスケールを次GEMMのXロードに畳み込み） | X 1読分/使用箇所 | **MEDIUM** |
| G3 | GEMM+Epilogue（silu/bias加算の出力レジスタ内処理） | Y 1往復分/エキスパート | **MEDIUM** |

### D7判定（優先度付け）

1. **現行longrunの活性融合（F1–F3）はどれも全体の1%級** — やる価値は将来の重み削減後。
2. **G1（GDN-2融合）はdecode時の支配項になり得る唯一の融合** — 1B decode設計時に最優先。
3. 帯域律速を動かすレバーは融合ではなく **D6のトークンバッチ（GEMM化、AI 0.5→23の理論値）** と、
   重み側トラフィック削減（量子化 — **本任務では着手禁止**、Phase E以降の判断材料として残す）。

---

## 外挿への影響

- 1270 toks/s（2T peak）は **DRAMストリーミング帯域の飽和点** と確定（D4全左 + D5帯域一致 + 1T→2Tで頭打ち）。
  「LLCスラッシングが原因で外挿が崩れる」懸念は棄却 — むしろ帯域律速は線形に外挿しやすい性質。
- 1B外挿（÷140 → 9 toks/s楽観）の前提は変わらないが、その内訳が明確化:
  - 5M-proxyの6GB/stepは重み19MB×再利用構造の帰結。1B（0.712B、active 96M）ではトークンあたりfetch 16MiB水準
    （`20260914-1b-scale.md`）がそのまま帯域で割った値がdecode速度になる。
  - GDN-2未融合のまま1B decodeに入るとG1項（6.4MB/トークン級）が上乗せされる — 融合設計をdecode実装の受入条件にすべき。
- STREAM未計測（29.4 GB/sは推定）のため、絶対値の外挿には±30%を見込むこと。相対比較（スレッド数・融合有無）は5%ノイズ床で判定可能。
- 次の打ち手候補（Phase E向け・本任務では実施しない）: expert固定トークンバッチの試作、G1融合のdecode設計、N150でのSTREAM実測。

## 再現情報

- 計測元: `develop` == `d354025` を `~/jimotono-bench` にarchive同期（hash一致確認済み）、`build/train_longrun` Releaseビルド。
- ホスト: ssh `ubuntu` = Intel N150 / 4C / 11GiB / Ubuntu 24.04 / kernel 7.0.0-31-generic / gcc 13.3。
- 生 perf 出力の要点は§D5表の通り（elapsed 14–15s、user時間はスレッド数比例: 15.3s/21.4s/33.1s）。
- ローカルビルドは未実施（解析任務のため最小限 = ゼロ）。`build-roof/` 不使用。

---

## D4追記 —「bwd効率1/5」の再解釈（-mavx2修正後。既存値の書き換えなし）

- branch: `perf/p3/g-remeasure`（本wtのみ。コミットしない）。実装変更なし（測定＋文書追記のみ）。
- 背景：Stage0で `src/train_bwd.c` に `-mavx2` 付与＋`src/moe_layer.c` 非選択zero-fillのmemset化
  （S0b-1。reductionドット4箇所のみtol内一致〜1e-16、他はbit同一）。
  背景値：swiglu_bwd 3.8x・moe_bwd 2.18x（カーネル単体）。
- 本追記の一次ソース：`analysis/f2-breakdown.md` §7（`build-rem/`、`--steps 12` 逐次・uptime確認済み）、
  `bench/kernel/bw_stream.c`（S0b-4）実測値。本節はD4判定表のAI値・ridgeを書き換えない。

### 実測帯域（推定→実測）

- `bw_stream`（本wt内 `build-rem/bw_stream`、単一スレッド・正しさassert済み。速度合否なし）：
  Triad **21.24 GB/s**、Copy **18.73 GB/s**（開発機 Mac mini i7-8700B、up 13 days、load約1.7–2.5）。
- 旧前提 eff_BW=29.4GB/s（42×0.7、STREAM未計測±30%）に対し実測21.24は **−27.7%** であり、
  §D4末尾・外挿節で宣言した±30%以内に収まる。絶対値外挿の±30%留保は実測で裏付けられた形となり、
  相対比較（5%ノイズ床）の有効性は不変。ridge再計算値（614÷21.24≒28.9）は参考記録とし、
  D4判定（全カーネル左）は AI最大0.50≪28.9 で不変のため判定書き換えなし。

### 「bwd効率1/5」の再計算（F2 §7値＋STREAM実測）

- 旧記述（F2 §5-1・§6）：6Tで fwd 2.3GB/113ms≒21GB/s（推定29.4の7〜8割）に対し
  bwd 4GB/872ms≒4.6GB/sで約1/5の効率。
- 新値（F2 §7。トラフィック推定は同一コードのため不変としwallのみ更新）：
  - 1T（スレッド間共有なしの clean な比較点）：fwd 2.3GB/178ms≒**12.9GB/s**（STREAM比約61%）、
    bwd 4GB/850ms≒**4.7GB/s**（STREAM比約22%）。bwd/fwd比≒**36%（約1/2.8）**。
    旧1T（fwd 4.5GB/s・bwd 2.6GB/s・比約57%）から fwd 2.86x・bwd 1.81x ともに改善したが、
    fwdの改善が大きく相対比は拡大した。
  - 6T（旧「1/5」の条件）：fwd 2.3GB/68ms≒33.8GB/s（STREAM超過＝推定の破綻。層重みのスレッド間共有で
    実効トラフィックが減るため単純除算は不可。MT比は参考値に留める）、
    bwd 4GB/865ms≒**4.6GB/s**（STREAM比約22%で旧値と同一）。bwd絶対値はMTで完全に張り付き。
- 結論：**「bwd効率1/5」は実測帯域比約22%（約1/4.5）として確定**する。旧表現との差は丸め級であり、
  順位・対策優先度（bwd効率調査 > val > optim > sync。F2 §6）は不変。
  修正でbwd絶対値が1T 2.6→4.7GB/sへ改善したものの、MTでは4.6GB/sの壁が残り、
  原因仮説（H=128小行列のベクトル化・依存連鎖・Gpart zero/add/reduction 1.7GB・スパース加算パターン。
  F2 §6(a)）は分離されず残る。D4の「全カーネル左（帯域律速）」判定は不変。
  AI値そのもの（0.38–0.50）はコード読解の計数であり修正の影響を受けない（AVX2化は同一FLOPs/同一バイト）。
