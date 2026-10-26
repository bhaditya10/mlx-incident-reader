# Train a tiny AI locally with MLX

**From random weights to a tested model and a local oMLX endpoint—on your Mac.**

Incident Reader is the hands-on companion to Aditya Bharti's **MLX & oMLX Under the Hood** workshop. It turns short synthetic incident reports into JSON so you can inspect tokenization, attention, gradients, training, evaluation, caching and serving in one small project.

This is **from-scratch training**, not LoRA or fine-tuning. We use an established Qwen2-compatible architecture and Apple's MLX libraries; we do not claim to have invented a new architecture.

> **A useful learning project, not a reliable incident tool.** The selected model produces valid JSON on 100% of the locked test, but gets the entire record correct on only **44.13%**. A rules parser gets 100% on that restricted test. The gap is part of the lesson, not something to hide.

[Quick start](#quick-start-run-the-trained-model) · [Train it yourself](#train-it-yourself) · [Serve locally](#serve-through-omlx) · [Results](#what-the-model-actually-learned) · [Troubleshooting](#troubleshooting)

## What you will build

Input:
```text
Checkout is recovering. Severity is SEV2. Impact lasted 18 minutes.
```

The supplied checkpoint generates:
```json
{"service":"checkout","severity":"SEV2","status":"recovering","duration_minutes":18}
```

Change the wording and the answer can become wrong. Missing duration means `null`; an explicitly reported duration of zero means `0`. The checker evaluates those distinctions independently of JSON formatting.

## Prerequisites

| Requirement | Details |
|---|---|
| Hardware | Apple-silicon Mac **M1 or newer**, with native arm64 Python. Intel Mac, Windows and Linux are not validated by this package. |
| macOS | 15 or later for the pinned oMLX wheel; core installation may have different upstream requirements. |
| Python | **3.12**; recorded with 3.12.14. Use `python3.12`, not a different interpreter accidentally supplied by another environment. |
| Memory | **16 GB RAM recommended; minimum RAM has not been verified.** This is a 3.2M-parameter model. Keep several GB of headroom for macOS, Python and other applications. |
| Disk | Reserve several GB for Python dependencies, data and resumable checkpoints. The inference checkpoint itself is about 12.9 MB. |
| Network | Required for cloning and initial dependency installation. Generation/training uses local files; no hosted inference API or API key is required. |
| Optional | Node.js 24 for the TypeScript streaming example; Git/Command Line Tools for cloning and source dependencies. |

Check before installing:
```sh
uname -m                 # should report arm64
sw_vers -productVersion
python3.12 --version
```

If Python is absent, install Python 3.12 from a trusted distribution. Install Apple's Command Line Tools with `xcode-select --install` if Git is absent. Review any licence prompts yourself.

## Quick start: run the trained model

```sh
git clone https://github.com/bhaditya10/mlx-incident-reader.git
cd mlx-incident-reader
python3.12 unpack.py
cd companion
python3.12 -m venv .venv
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MLX_LM=0.31.4.dev132+g94cdcae13 \
  .venv/bin/python -m pip install -r repro/core-lock.txt
./run generate --checkpoint checkpoints/incident-reader \
  --text 'Checkout is recovering. Severity is SEV2. Impact lasted 18 minutes.'
```

The version variable restores the recorded build version because the pinned source archive lacks Git metadata. Dependencies are pinned to the recorded workshop environment, not automatically upgraded to today's versions.

The trained, untrained and 4-bit checkpoints are included in checksum-verified asset archives. `unpack.py` extracts them and the frozen data/evidence without overwriting changed files. No separate model-host account is needed. The trained FP32 weights should have SHA256:
```text
486c4224de9ea5579118d0d085e0055b976a5c54094cc5504de81f7b78de45be
```

Verify the implementation and 16 frozen predictions:
```sh
.venv/bin/python repro/verify.py
```

This checks Unicode tokenization, causal masking, reference-forward agreement and cached/uncached agreement. Floating-point agreement uses tolerances; it is not a promise of bit-identical training on every Mac.

## Train it yourself

The repository includes the frozen synthetic splits: 30,000 training, 3,000 validation, 3,000 test and 200 challenge records. Read `data/manifest.json` and `MODEL-CARD.md` before changing them. Wording families and full attribute tuples are separated across splits.

Start a short learning pilot first:
```sh
./run train --run my-pilot --steps 20 --validation-limit 8 --checkpoint-every 20
./run export --run my-pilot --export-name my-pilot
./run generate --checkpoint checkpoints/my-pilot
```

A 20-update model is immature; do not expect a correct answer. For the workshop schedule:
```sh
./run train --run my-training --steps 3000 --seconds 10800
./run export --run my-training --export-name my-trained-model
```

Training uses AdamW, learning rate 0.0003, a 100-update warm-up, cosine decay, microbatches of 8 accumulated four times, and gradient clipping at 1.0. Loss is computed on the output and end token, excluding prompt/padding positions. The run stops at the step or elapsed-time limit, whichever comes first. Choose a new run/export name each time: existing outputs are preserved.

Checkpoints contain weights, optimizer state and random-state information. To continue a stopped run, preserve the original total schedule and validation limit:
```sh
./run train --run my-interrupted --steps 3000 --stop-after 100
./run train --run my-resumed --steps 3000 \
  --resume runs/my-interrupted/step-100
```

To regenerate data, work in a separate clone and explicitly move the existing `data` directory aside first. Run `./run prepare-data`, then `.venv/bin/python challenge.py`. Never change a frozen dataset midway through a reported experiment.

## Evaluate before trusting the output

For a quick check:
```sh
./run evaluate --checkpoint checkpoints/incident-reader \
  --run my-check --split validation --limit 16
```

For a complete aggregate evaluation with the recorded exact-length batching:
```sh
.venv/bin/python evaluate.py --run main-02 \
  --checkpoint checkpoints/incident-reader --split test --output-name my-test
```

Use a fresh output name. Reports are written to the repository's `evidence/` directory. Select checkpoints using validation only; the locked test is for reporting, not choosing the best model.

## What the model actually learned

| Metric | Locked test (3,000) | Challenge (200) |
|---|---:|---:|
| Valid JSON | 100% | 85.5% |
| Valid schema | 99.47% | 81.5% |
| Entire record correct | **44.13%** | **11%** |
| Rules baseline, entire record | **100%** | **90%** |

Checkpoint 700 from `main-02` was selected using validation loss. Training continued to 3,000 updates; later updates were not automatically better. See raw predictions and summaries under `evidence/`, and the [model card](companion/MODEL-CARD.md).

This is a small synthetic task with limited wording diversity. Scores do not estimate production incident accuracy. Unknown services, contradictions, negation and unfamiliar phrasing are deliberate failure probes. Teacher-forced next-token accuracy (99.17%) measures a different thing from complete generated-record accuracy.

## Follow the code

| File | What to inspect |
|---|---|
| `companion/tokenizer.py` | Byte IDs, control tokens and prompt format |
| `companion/model.py` | Embeddings, grouped-query attention, causal mask, RoPE, residuals and output logits |
| `companion/cli.py` | Training, loss masking, gradients, AdamW, checkpoint selection and generation |
| `companion/data.py`, `challenge.py` | Synthetic facts, split construction and harder cases |
| `companion/checker.py` | Independent schema/field checks and rules baseline |
| `companion/evaluate.py` | Frozen greedy evaluation and aggregate metrics |
| `companion/benchmark.py` | Cached/uncached checks and explicitly named memory metrics |
| `companion/client.ts` | HTTP streaming, cancellation, buffering and validation |

Architecture: 4 blocks, hidden width 256, feed-forward width 768, 4 query heads, 2 KV heads, 512-token context, 260-token byte vocabulary, tied embeddings, **3,216,640 parameters**. Training starts from random weights. The included untrained checkpoint makes the initialization inspectable.

## Serve through oMLX

Serving uses the real oMLX 0.7.1 package, not a custom server masquerading as oMLX. See [the serving guide](docs/SERVING.md) for the separate installation, exact version constraints and verification status.

Once the serving environment is installed, run from `companion/`:
```sh
.venv/bin/python -c 'from omlx.cli import main; main()' serve --model-dir checkpoints \
  --base-path ../local-server --host 127.0.0.1 --port 18036 \
  --max-concurrent-requests 1 --headless
```

Keep memory guards enabled and bind only to loopback for this workshop. Model settings must cap context at 512 and output at 120, disable KV quantization for equivalence checks, and use greedy sampling.

In another terminal:
```sh
cd mlx-incident-reader/companion
npm ci
npm run check
node client.ts
```

The deliberately different client wording may yield an incorrect value even when schema validation passes. This is intentional teaching material. Streaming chunks are transport chunks, not guaranteed individual tokens.

## Troubleshooting

- **`No module named mlx`**: confirm you installed into `companion/.venv`, or set `INCIDENT_PYTHON` to the intended interpreter.
- **Wrong architecture / Metal unavailable**: use a native Apple-silicon Python outside Rosetta. Verify `uname -m` and Python's `platform.machine()`.
- **Install fails on MLX-LM version metadata**: include the documented `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MLX_LM` variable.
- **Out of memory**: stop unrelated model workloads, close unused applications and inspect memory pressure. Do not disable safety guards. Training stores gradients and optimizer state in addition to weights.
- **Run/export/evaluation already exists**: use a new name; the refusal protects earlier results.
- **Long input rejected or incomplete JSON**: the model context is only 512 byte-level tokens and generation is capped. This is not a long-document assistant.
- **Valid JSON, wrong answer**: compare fields to the input and checker; JSON validity is not factual correctness.
- **First request slower**: separate model loading, prefill, decode and warm-cache effects. Do not compare first-load latency to a warm run.
- **Different result on another version**: attach the version manifest and exact input to your issue; do not silently upgrade the recorded environment while reproducing numbers.

## Contributing

Start with a reproducible case: exact input, expected record, actual output, checkpoint SHA256, package versions and the smallest relevant command. State whether the issue belongs to data, model, evaluation, server or client. Add a regression test where possible. Do not include private incident reports or credentials.

This repository does not imply that an upstream contribution was submitted or accepted. Improvements after inspecting the locked test must use a new experiment and fresh test protocol.

## Sources and licensing

Original workshop code, synthetic datasets and checkpoints: MIT licence; see [LICENSE](LICENSE). Third-party packages retain their own licences. The architecture follows the [MLX-LM Qwen2 implementation](https://github.com/ml-explore/mlx-lm/blob/94cdcae13b266c337bcaca09b97b9c5a9c0e2cde/mlx_lm/models/qwen2.py); attribution and upstream licence are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- [MLX](https://github.com/ml-explore/mlx)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [oMLX](https://github.com/jundot/omlx)
- [Original short introduction](https://www.youtube.com/watch?v=680jAB6MW88)

The full workshop link will be added after publication. No paid API, hosted inference service or private operational data is needed for the core exercises.
