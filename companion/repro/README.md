# Reproducible environments

Run installation from `companion/`. Core training/inference uses `core-lock.txt`; full oMLX serving uses `server-lock.txt`. Both were installed in a fresh Python 3.12 environment on the validation Mac.

```sh
python3.12 -m venv .venv
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_MLX_LM=0.31.4.dev132+g94cdcae13 .venv/bin/python -m pip install -r repro/core-lock.txt
.venv/bin/python repro/verify.py
```

See [serving](../../docs/SERVING.md) for the full server path and [validation](../../docs/VALIDATION.md) for scope. A fresh virtual environment is not a clean-machine certification.
