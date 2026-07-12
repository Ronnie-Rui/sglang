# Quest E2E Roofline Ablations

These benchmark-only patches split runtime Quest cost without adding experiment
flags to production SGLang. They are active only in server processes launched
through `launch.py` (or with this directory prepended to `PYTHONPATH`).

## Modes

`fixed-sparse` keeps the runtime-sparse graph topology and real FA3 sparse
attention, but replaces query-dependent selection with a deterministic prefix
plus recent-page plan of the same width. Quest representation allocation,
construction, update, score, and top-k are disabled. Page identity and locality
are intentionally different, so this is a compute roofline rather than an
accuracy or exact-memory-traffic control.

`retrieval-dense` executes the complete Quest representation and retrieval
lifecycle while leaving FA3 metadata dense.

`runtime-dense` disables Quest representation and retrieval while also leaving
FA3 metadata dense. It is the matched control for `retrieval-dense`: both keep
runtime-sparse graph breaks and FA3 scheduler behavior.

Use a normal Quest server and a dense server without `--enable-hisparse` as the
other two controls. Convert output throughput to milliseconds per generated
token before subtracting runs:

- `retrieval-dense - runtime-dense` estimates representation plus retrieval tax.
- `fixed-sparse - runtime-dense` estimates fixed sparse attention and metadata tax.
- `runtime-dense - dense` estimates runtime-sparse graph and scheduler tax.
- `normal Quest - fixed-sparse` is the remaining integrated Quest gap; it also
  includes page-identity and locality differences, so it is not a pure software
  overhead measurement.

Throughput percentages are not additive. None of these modes is an accuracy
benchmark; use ratio-1 parity on the normal Quest server for that.

## Launch

Run from the repository root. The launcher rejects configurations that do not
enable Quest with FA3. Keep all other server and client arguments identical.
The artifact directory must exist before `bench_serving` opens its output file.

```bash
ROOT=/root/workspace/quest-roofline
mkdir -p "$ROOT/logs" "$ROOT/results"

python benchmark/quest_roofline/launch.py --mode fixed-sparse -- \
  --model-path /root/public-storage/model/Qwen/Qwen3-1.7B \
  --host 127.0.0.1 --port 30000 \
  --attention-backend fa3 --disable-radix-cache \
  --mem-fraction-static 0.8 --max-running-requests 8 --random-seed 42 \
  --enable-hisparse \
  --hisparse-config '{"algorithm":"quest","backend":"fa3","page_size":16,"sparsity_ratio":0.244094488,"num_recent_pages":4,"min_sparse_prompt_len":2048}' \
  2>&1 | tee "$ROOT/logs/fixed_sparse_8k_c8_server.log"
```

Change only `--mode fixed-sparse` to `retrieval-dense` or `runtime-dense` for
the other ablations. A valid log contains `mode=`, every expected `applied=`,
and an `active=` marker proving the patched live Quest path was called.

Use the same client for normal Quest, all three ablations, and dense control:

```bash
python -m sglang.bench_serving \
  --backend sglang --base-url http://127.0.0.1:30000 \
  --dataset-name random-ids \
  --model /root/public-storage/model/Qwen/Qwen3-1.7B \
  --num-prompts 24 --random-input-len 8192 --random-output-len 128 \
  --random-range-ratio 1.0 --max-concurrency 8 --request-rate inf \
  --warmup-requests 2 --seed 42 --temperature 0 --disable-tqdm \
  --output-details --output-file "$ROOT/results/fixed_sparse_8k_c8.jsonl"
```

For the established 32K eager cases, change the Quest ratio to `0.062124249`,
add both `--cuda-graph-backend-decode disabled` and
`--cuda-graph-backend-prefill disabled`, and use:

```bash
# c1: use --num-prompts 8 --warmup-requests 1 --max-concurrency 1
# c4: use --num-prompts 8 --warmup-requests 1 --max-concurrency 4
# both: --random-input-len 32000 --random-output-len 128
```

Stop each server by its recorded PID and verify port 30000 is free before
starting the next variant. Run each case at least twice when deciding whether a
small delta is real.
