# Release validation — 26 October 2026

A new isolated Python 3.12 environment was created without system site packages on the production Mac. The pinned core dependencies installed successfully. The full official oMLX 0.7.1 dependency closure was then installed and frozen separately in `companion/repro/server-lock.txt`. `pip check` reported no broken requirements.

Checks performed:
- tokenizer Unicode round trips;
- educational/reference forward agreement within 1e-3;
- causal-mask agreement within 1e-5;
- cached/uncached agreement within 1e-3;
- all 16 frozen previously evaluated predictions identical;
- trained checkpoint SHA256 unchanged;
- direct MLX-LM and real oMLX output identical on the canonical report;
- two-update fresh training/export/evaluation smoke test.

Original large-set evaluation is preserved; this release check does not rerun a new training study or change the selected checkpoint. A fresh virtual environment is not a clean physical machine. The companion includes actual failures and does not claim production readiness.
