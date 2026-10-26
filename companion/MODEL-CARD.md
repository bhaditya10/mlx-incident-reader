# Incident Reader — educational checkpoint

Randomly initialized Qwen2-compatible decoder trained locally with MLX. No pretrained checkpoint, teacher-model answers, LoRA or fine-tuning. Architecture is established work; this project does not claim to invent transformers.

4 blocks, hidden width 256, intermediate width 768, 4 query heads, 2 KV heads, tied embeddings, 260-token byte vocabulary, context 512. Actual parameter count: **3,216,640**. FP32 training. Training examples: 30,000 synthetic reports. Validation: 3,000, with different wording families and disjoint complete attribute tuples. Test: 3,000 with another held-out set of wording families and tuples. Challenge: 200 deterministic cases from 10 separately authored families; these are not 200 independent real incidents.

Main run `main-02` completed 3,000 updates. Checkpoint 700 was selected solely by lowest validation loss (0.0236266013), before test evaluation. Later updates reduced training loss without improving validation loss. No adaptation after seeing test or challenge results.

| Metric | Locked test, n=3,000 | Challenge, n=200 |
|---|---:|---:|
| Valid JSON | 100% | 85.5% |
| Valid schema | 99.47% | 81.5% |
| Entire record correct | 44.13% | 11% |
| Rules baseline, entire record | 100% | 90% |

Test field accuracy: service 60.7%, severity 94.53%, status 79.1%, duration 98.87%. Scores are exact comparisons to synthetic gold labels. These measurements do not estimate performance on natural operational incidents. Wording-family diversity is small, and related phrasing is possible despite exact/normalized duplicate checks. The model is not fit for operational incident triage or business decisions.

Unknown services and contradictory facts use null in the challenge gold policy. This behavior was not taught comprehensively by the restricted training distribution. Generation uses greedy decoding with an end token and 120-token cap. JSON/schema validity never establishes factual accuracy.

Checkpoint SHA256: `486c4224de9ea5579118d0d085e0055b976a5c54094cc5504de81f7b78de45be`.

Numerical forward-pass, causal-mask and cached/uncached checks passed at documented tolerances. Direct MLX-LM and oMLX returned identical text for the declared serving smoke test; this is not exhaustive server equivalence. See the evidence directory for raw outputs, package versions and split hashes.

## Refinement evaluation (2026-10-25)
The frozen checkpoint scores59.8% complete records on validation and44.13% on the locked test. Teacher-forced next-token accuracy on test is99.17%, a different metric with correct preceding output supplied. The separate4-bit export scores44.83%; this small aggregate difference is not evidence of a general quality improvement. Detailed paired counts, memory metrics, timing ranges and limitations are in ../docs/RESULTS.md. No test-guided checkpoint changes were made.
