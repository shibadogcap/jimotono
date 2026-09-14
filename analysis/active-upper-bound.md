# アクティブ上限の再計算 — N100 8B の計算過程と保守目標

- branch: `docs/p3/pre-g`（本wtのみ。コミットしない）
- 位置づけ: Phase G 事前確認用の文書のみ。実装禁止、重負荷実行なし。
- 目的: `analysis/p3-design-inputs.md` §(a) の「N100 INT4換算 約8B上限」の計算過程を再現可能に明示し、**公称 8B ではなく保守値を目標に使え**と明記する。
- スコープ注意: **本書の結論を Phase G に混ぜない。G のスコープは MoE GEMM 化のみ。** 本書は P3 設計入力である。
- 方法論：toks/s の cross-vocab 直接比較は行わない。LFM2/Gemma 値は帯域→byte/token→実効効率 η の較正材料としてのみ使う。

## 1. 式と前提（すべて明示）

デコード帯域式（`p3-design-inputs.md` A1 と同一）：

```
tok/s ≒ eff_BW / bytes_per_token
bytes_per_token ≒ active_bytes + kv_bytes_per_token（短文脈では微小）
eff_BW = peak_BW × η（η＝実効効率）
active_bytes = active_params × byte_per_param
```

| 記号 | 意味 | 本書の値 |
|---|---|---|
| peak_BW（N100） | メモリピーク帯域 | **約40GB/s**（受領前提） |
| peak_BW（Ryzen AI 360） | 同上 | **約80GB/s**（保守前提。実機・構成で変動） |
| η（公称） | 実効効率 | **0.5**（LFM2較正 0.57 より保守的） |
| η（悲観） | 実効効率の下振れ点 | **0.1**（メモリ1ch・小GEMV・競合時を想定） |
| byte_per_param | 精度換算 | **INT4: 0.5B/p、INT8: 1.0B/p、INT2: 0.25B/p**（スケール・メタデータ別。スケール込み Q4_K_M 相当なら約0.56B/p を使う。§3参照） |
| 目標 tok/s | AGENTS.md §8 | N100 **5**、Ryzen AI 360 **30** |

η の較正根拠（存在証明としての利用のみ。優劣比較には使わない）：

- LFM2-24B-A2B：Q4_K_M（スケール込み約0.56B/p と仮定→約1.3GB/token）で 112tok/s。112×1.3GB ≒ **145GB/s実効**。Ryzen AI Max+ 395（peak 約256GB/s）比で **η≒0.57**。
- Gemma-4 26B MoE（active 3.8B・124tok/s）はタスク前提の受領値（一次ソース未確認）。INT4換算（0.5B/p→約1.9GB/token）で約236GB/s実効の計算になるが、本書では上限表の較正には使わず、2〜4B級 active が CPU 級帯域で 100tok/s 超という存在証明に限定する。
- JIMOTONO の公称 η=0.5 は較正 0.57 を下回る保守側の丸めである。

## 2. 公称上限の再計算（η=0.5、INT4 routed 想定 0.5B/p）

```
eff_BW = peak × 0.5
byte/token予算 = eff_BW / 目標tok/s
active上限 = byte/token予算 / 0.5B/p
```

| 機体 | peak | eff（η=0.5） | 目標 | byte/token予算 | active上限（INT4） |
|---|---|---|---|---|---|
| N100 | 40GB/s | **20GB/s** | 5tok/s | 20/5 = **4.0GB** | 4.0/0.5 = **約8B** |
| Ryzen AI 360 | 80GB/s | **40GB/s** | 30tok/s | 40/30 ≒ **1.33GB** | 1.33/0.5 = **約2.6B** |

- これが「N100 8B上限」の全計算過程である。**8B は η=0.5・KV微小・オーバーヘッド無視の理論上限**であり、設計目標ではない。
- 精度を変えた場合の読替え（同式、η=0.5）：INT8（1.0B/p）なら N100 約4B・Ryzen 約1.3B。混合精度（INT8/INT4/INT2混成、AGENTS.md §2.2）の実効上限は加重平均 byte_per_param で割り直すこと。
- P3 の現行想定（active 100M INT4 ≒ 50MB ＋ 縮小head INT8 約12.7MB ＋ KV微小 ≒ 63〜100MB/token）との対比：N100 で 5×100MB = 0.5GB/s（eff の 2.5%）、Ryzen で 30×100MB = 3.0GB/s（eff の 7.5%）。帯域は公称上は潤沢。

## 3. 保守見積もり（KV・ルーティング・実効帯域を加味）

公称 8B / 2.6B をそのまま目標に使ってはならない。以下を加味した保守値を目標に使うこと。

### 3.1 差し引くもの

1. **KV バイト：** CSA2＋FP4（E2M1、16ch毎 E4M3 スケール）＋ top-1024・window128・heavy128:1 により短文脈では微小（`kv-cache-design.md` §2〜3）。ただし長文脈・大バッチでは `フル層数 × 共有後KVエントリ × 0.5B` が byte/token に加算される。式上は `active_bytes` に上乗せし、上限を目減りさせる。
2. **ルーティング・固定費：** router 評価・top-k 選択・gather、SwiGLU/RMSNorm 等のメモリパス、カーネル起動≦50/token（AGENTS.md §8）の固定費、スレッド sync（`f2-breakdown.md` で 6T 6.2%→12T 11.3% に逓増）。バイト換算できないが実効 η を押し下げる方向に効く。
3. **実効帯域の下振れ：** 単ch・小GEMV・他プロセス競合・O_DIRECT/SSD待ちの混入で η=0.5 は出ない場合がある。下振れ点として **η=0.1** を置く。
4. **精度換算の上振れ：** ブロック量子化のスケール・メタデータ込みでは 0.5B/p を超える（Q4_K_M 相当で約0.56B/p）。INT8 混成部（共有expert・注意射影・head、AGENTS.md §2.2）が増えるほど実効 byte_per_param は 0.5 より大きくなる。

### 3.2 保守上限表

(a) η下振れのみ（KV・固定費を無視した素朴な悲観）：

| 機体 | eff（η=0.1） | byte/token予算 | active上限（INT4、0.5B/p） |
|---|---|---|---|
| N100 | 4GB/s | 0.80GB | **約1.6B** |
| Ryzen AI 360 | 8GB/s | 約0.27GB | **約0.53B** |

(b) さらに KV＋スケール＋固定費で 15〜20% 目減りを見込む場合（乗率 0.8）：

| 機体 | (a)の上限 | ×0.8（保守） | 保守上限の目安 |
|---|---|---|---|
| N100 | 約1.6B | ×0.8 | **約1.3B** |
| Ryzen AI 360 | 約0.53B | ×0.8 | **約0.4B** |

- η=0.3 の中間点も参考に残す：N100 約4.8B（×0.8で約3.8B）、Ryzen 約1.6B（×0.8で約1.3B）。
- **指示：設計目標には公称 8B / 2.6B を使わず、保守値 N100 約1.3B・Ryzen 約0.4B 側を上限の目安として使うこと。** P3 の active 100M 目標は保守値に対しても N100 で約13×、Ryzen で約4×のマージンがあり、帯域は律速ではないという結論（`p3-design-inputs.md` §(a)）は保守側でも維持される。
- したがって律速候補は帯域ではなく、未縮小 head（`vocab-budget.md` で除去）、カーネル起動固定費、NVMe 読込≦103MiB/token の IOPS・遅延、スレッド sync である。NVMe 予算（想定約100MB/token）は境界上のため、pin-hit 率と Delta 的持続率で 103MiB 内に収めることを P3 受入条件とする（`p3-design-inputs.md` C1 と同一）。

## 4. 使ってよい数値・使ってはならない数値

- 使ってよい：本書 §2 の式・前提・較正値、§3 の保守上限（N100 約1.3B・Ryzen 約0.4B 目安）、精度換算表。
- 使ってはならない：公称 8B / 2.6B を設計目標・受入条件として引用すること。LFM2 の 112tok/s・Gemma の 124tok/s を同語彙の速度比較に使うこと（禁止事項）。
- Phase G への混入禁止：本書の帯域・上限値は MoE GEMM 化の受入条件ではない。G の判定は GEMM パスの帯域効率・整数SIMD/AMX 分岐等のカーネル側で行う。

## 参照

- `analysis/p3-design-inputs.md` §(a)（A1 帯域式・A2 上限表。本書はその計算過程の明示＋保守化）
- `analysis/kv-cache-design.md` §2〜3（KV 微小化の根拠と FP8 フォールバック）
- `analysis/decoding-architecture.md` §2（LFM2 値の扱い＝比率の傍証のみ）
- `analysis/training-speedup.md` §5（W8A8・85%帯域効率は GEMM パス目標であり本書の η とは別物）
- `analysis/roofline.md` D4（全カーネル帯域律速＝帯域式適用の正当化）、`analysis/f2-breakdown.md`（sync・ckpt・esmoe_io 実測）
- AGENTS.md §8（5tok/s・30tok/s・103MiB/token・起動≦50/token）
