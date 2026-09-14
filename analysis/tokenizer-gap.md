# Tokenizer throughput gap (llm-jp v2.2 C encoder)

date: 2026-09-14
scope: `src/bpe.c` (Unigram Viterbi, single-thread) on consumer hardware
method: `bench_bpe` sequential 1-pass + warmup (RULE.MD: performance numbers are
reference-only when other agents run in parallel)

## current

- corpus: `data/tinystories_head16M.txt` (16,776,860 B)
- vocab: `data/llmjp-v22.jtvocab` (48,578 pieces, max_id 48,585)
- result: 16,776,860 B -> 4,497,181 tokens in 4.5992 s
  = **3.648 MB/s** (977,827 toks/s), golden 38/38 match (encode + roundtrip)
- 16 MB head tokenizes in ~5 s. one-time `--write-pack` then `.jtdp` reuse,
  so it is NOT on the training critical path today.

## why slow (expected, not a bug)

- per-position byte-trie walk (up to max_plen) + float32 Viterbi DP: O(n * plen)
- single-thread, malloc/free per `encode` call (dp/prev/pid + normalized buffer)
- correctness-first minimal implementation (Phase F5, fail-closed, libc only)

## target

- **100 MB/s+ (OmniToken方式)** — parent-designated fast path.
- sketch (not started): chunked parallel encode, DP buffer reuse / arena,
  trie flattening, short-piece fast path. 27x speedup needed, so this is a
  rewrite-class task, not tuning.

## priority: low

- training reads `.jtdp` (mmap), never re-tokenizes per step.
- 16 MB wall-clock share (~5 s once) is negligible vs training runs.
- raising priority only via the trigger below.

## trigger (revisit when ANY holds)

- corpus scales to GB class (e.g. full TinyStories 1.92 GB -> ~9 min at 3.6 MB/s),
  making tokenization a visible pipeline stage.
- tokenizer enters an iterative loop (online encode during training/eval).
- tokenizer wall-clock share exceeds ~5% of an end-to-end experiment budget.

## comparison rule (RULE.MD)

- never compare tokenizers by toks/s. axes: same wall-clock, same byte count.
- val loss normalized: `val_ce_pb = val_ce_per_token * tokens/bytes`
  (tokens/bytes = measured value of that V). logged by `train_tinystories`.
