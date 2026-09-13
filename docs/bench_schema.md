# bench結果スキーマ (ROADMAP 5.1/5.2)

`configs/bench/machine.yaml` の項目と対応する `results/bench/<date>-<machine>.json`
の記録形式。1ファイル複数行 (JSON Lines): 1行=1ケースの計測結果。
`bench/common/bench_common.h` の `jt_bench_print_json()` がこの形式を出力する。

## machine.yaml との対応

| JSONキー | machine.yaml項目 | 必須 | 説明 |
|---|---|---|---|
| `name` | (ケース名) | 必須 | ベンチケース名。例: `tmac/K512-bits4`, `gdn2/d64x128` |
| `machine` | `machine` | 必須 | マシン識別子。例: `macmini-i7-8700B`, `N100` |
| `commit` | `build_commit` | 必須 | ビルドに使ったgit hash (40桁)。`git rev-parse HEAD` |
| `trials` | `trials` | 必須 | 計測回数N (奇数推奨。例: 11)。中央値+MAD用 |
| `median_ns` | (計測値) | 必須 | N回計測の中央値 (ns, `%.6f`) |
| `mad_ns` | (計測値) | 必須 | 中央絶対偏差 `median(\|x - median\|)` (ns, `%.6f`) |
| `notes` | `notes` | 任意 | 外乱記録 (ブラウザ有無・温度・負荷等)。空文字可 |

`machine.yaml` の `os` / `cpu` / `cores_pinned` / `freq_locked` / `build_type`
はファイル名・`notes`・別途 `scripts/bench_env.sh` の出力
(`results/bench/<date>-env.txt`) で補完する。JSON行には含めない
(1行1ケースの集計に集中し、環境詳細はenv.txt側に寄せる)。

## 例

```json
{"name":"tmac/K512-bits4","machine":"macmini-i7-8700B","commit":"dc948b0abc1234567890abcdef1234567890abcd","trials":11,"median_ns":1234.500000,"mad_ns":67.250000,"notes":"browser closed, load 1.2"}
{"name":"gdn2/d64-dv128","machine":"macmini-i7-8700B","commit":"dc948b0abc1234567890abcdef1234567890abcd","trials":11,"median_ns":987.000000,"mad_ns":12.000000,"notes":"browser closed, load 1.2"}
```

## TSV (補助・人間可読)

`jt_bench_print_tsv()` の出力。列: `name<TAB>trials<TAB>median_ns<TAB>mad_ns`。

```text
tmac/K512-bits4	11	1234.500000	67.250000
gdn2/d64-dv128	11	987.000000	12.000000
```

## 運用 (ROADMAP 5.1)

1. `wt-bench` で `git clean -fdx` 後にビルド (クリーンツリー必須)。
2. CPU固定 (`taskset` / affinity)・周波数固定 (可能なら)。
3. `sh scripts/bench_env.sh | tee results/bench/<date>-env.txt` で外乱記録。
4. ベンチ実行 → `results/bench/<date>-<machine>.json` に追記。
5. 中央値+MADで記録し、外れ値は破棄 (MADが大きい場合は再計測)。
