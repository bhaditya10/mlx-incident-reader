# Serve the checkpoint with oMLX 0.7.1

Validated on 26 October 2026 in a fresh Python 3.12 virtual environment on the same Apple-silicon Mac used for production. This is a fresh-environment check, not certification across every Mac/macOS version.

## Install

From `companion/`, use the full serving lock **instead of** the smaller core lock if you want both training and serving in one environment:

```sh
python3.12 -m venv .venv
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MLX_LM=0.31.4.dev132+g94cdcae13 \
  .venv/bin/python -m pip install -r repro/server-lock.txt
.venv/bin/python -m pip check
```

Python 3.12 and macOS 15+ are required by this pinned oMLX wheel. Git must work because several upstream dependencies are pinned Git commits. The full lock includes oMLX's document/audio/vision dependencies even though this workshop uses only text. It is therefore larger than the core installation. Some transitive versions differ from the original core lock because oMLX requires them; the model and its 16 frozen predictions were rechecked after installation.

## Configure a bounded local server

```sh
.venv/bin/python configure_server.py
.venv/bin/python -c 'from omlx.cli import main; main()' serve \
  --model-dir checkpoints --base-path ../local-server \
  --host 127.0.0.1 --port 18036 --max-concurrent-requests 1 --headless
```

The configuration limits this model to a 512-token context and 120 output tokens, disables KV quantization, and sets temperature 0, top-p 1, top-k 0 and repetition penalty 1. Memory guards remain enabled. Do not expose this educational setup to the public internet.

In another terminal, run `node client.ts` from `companion/` (Node 24). Or run the independent one-shot equivalence check when no server occupies port 18036:

```sh
.venv/bin/python trained_server_gate.py
```

The check launches its own server, compares its text with direct MLX-LM generation and shuts it down. It refuses to overwrite a prior `evidence/trained-server-gate` directory; rename that evidence directory before a new run.

## Verified result

The canonical 18-minute checkout report returned identical text through direct MLX-LM and oMLX, with the correct duration of 18. The installed dependency check reported no broken requirements. This proves this smoke test, not exhaustive equivalence or production reliability.
