# Incident Reader: measured results

The selected model and locked test have not changed. This pass measures and explains the existing model rather than tuning away its failures.

## What the accuracy gap actually shows

| Measurement | Validation | Locked test |
|---|---:|---:|
| Complete records correct, generated independently | 59.80% | 44.13% |
| Next-token accuracy, correct preceding tokens supplied | 99.43% | 99.17% |
| Structure-token accuracy under that same supplied-prefix condition | 99.99% | 100.00% |
| Value-token accuracy under that condition | 98.04% | 97.07% |

These are different evaluation procedures. A nearly correct sequence of individual predictions is not necessarily a completely correct record. During teacher forcing, later predictions receive the correct preceding answer even if an earlier prediction was wrong. During generation, the model receives its own preceding output. Repeated JSON keys and punctuation also occupy many target positions.

These results demonstrate the distinction. They do not establish a single cause of every failed record. Exact accuracy differs substantially between wording families, but those groups also have different attribute mixtures. A controlled matched-paraphrase experiment would be needed to isolate wording as a causal factor.

The rules parser scores 100% on this restricted test. The neural model is a learning vehicle here; a simpler rules implementation is more reliable on this restricted test.

Evidence: `evidence/teacher-forcing-audit.json`, `evidence/error-analysis.json`, and the preserved prediction files.

## A measurable training illustration

A separate one-example run starts from the original random checkpoint. It uses a training example, not test data. With the correct answer prefix supplied, the probability of the next service character, `s`, changes from 0.136% to 2.426% after one AdamW update. Answer loss changes from 5.982 to 4.415.

One displayed parameter changes from 0.036185 to 0.037134. This coordinate illustrates a change; it is not presented as the sole cause of the output change. Gradient clipping and AdamW state are involved, so the update is not simply learning-rate times raw gradient.

The main checkpoint remains untouched. This is a separate teaching experiment.

## Checkpoint resumption

The full CLI path now supports an interruption without changing the total learning-rate schedule. It restores optimizer state and data-sampling state, retains earlier validation-selection history, and labels the resumed step correctly.

In a three-step control versus a run interrupted after step two, the largest parameter difference was 1.49e-8, optimizer-state difference 9.31e-10 and validation-loss difference 4.77e-7. The initial strict-equality assertion failed. The measured differences pass explicit floating-point tolerances; this is numerical agreement, not bitwise identity or proof of long-run stability.

## Weight quantization

| Measurement | FP32 | 4-bit export |
|---|---:|---:|
| Safetensors file size | 12,871,834 bytes | 2,036,268 bytes |
| Locked-test complete-record accuracy | 44.13% | 44.83% |
| Cached generation median, five measured repetitions | 90.70 ms | 82.37 ms |
| Observed time range | 87.26–127.59 ms | 75.09–88.82 ms |
| Median peak MLX allocator bytes | 21,103,968 | 21,909,760 |

Both timing conditions generated 84 output tokens for the same prompt. Each had one warm-up; model loading was separate. The ranges overlap. This tiny model is not representative of large-model throughput. Allocator memory is not process RSS or system memory.

Quantization changed 472 answer strings: 111 previously wrong records became correct, while 90 previously correct records became wrong. The small aggregate increase does not establish that quantization improves quality. Conversion reported approximately 5.037 effective bits per parameter, reflecting mixed storage and overhead rather than a literal four bits for every stored value.

## Server behavior

The isolated server remained on loopback with memory guards enabled. Tests included repeated prompts, longer prompts, two concurrent clients, an over-limit prompt and client disconnection.

- A 522-token prompt exceeded the configured 512-token limit and received HTTP 400.
- Repeated 69- and 133-token prompts reported zero cached tokens. Repeated 261- and 389-token prompts reported 256 cached tokens. The installed scheduler's default paged-cache block size is 256 tokens.
- The server log recorded prefill interrupted at 401/402 tokens with one request aborted after the client disconnected. A subsequent request returned successfully. This does not establish instantaneous GPU interruption or leak freedom.
- Client concurrency is not itself proof of a particular batching execution path. Server-managed cache conditions were not reset between every request; these are not isolated cold-cache benchmarks.
