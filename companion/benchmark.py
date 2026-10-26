"""Direct MLX inference: one warm-up, five measured repeats, explicit metrics."""
import argparse, json, time, statistics, hashlib
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
from tokenizer import prompt
from checker import check

P = Path(__file__).resolve().parent
def benchmark(checkpoint, destination):
    destination = Path(destination)
    if destination.exists():
        raise RuntimeError('Preserve existing measurements; use a new destination')
    start = time.perf_counter()
    model, tok = load(str(checkpoint))
    mx.eval(model.parameters())
    loading = time.perf_counter() - start
    text = 'Checkout is recovering. Severity is SEV2. Impact lasted 18 minutes.'
    expected = dict(service='checkout', severity='SEV2', status='recovering', duration_minutes=18)
    ids = tok.encode(prompt(text))
    records = []
    for cached in [True, False]:
        for repeat in range(6):
            mx.clear_cache(); mx.reset_peak_memory()
            cache = [KVCache() for _ in model.layers] if cached else None
            sequence = ids[:]; output = []; start = time.perf_counter(); first = None
            for i in range(120):
                x = mx.array([sequence if not cached or i == 0 else [output[-1]]])
                y = int(mx.argmax(model(x, cache)[:, -1, :], axis=-1).item())
                now = time.perf_counter()
                if first is None: first = now
                if y == 258: break
                output.append(y); sequence.append(y)
            elapsed = time.perf_counter() - start
            answer = tok.decode(output)
            records.append(dict(cached=cached, warmup=repeat == 0, repeat=repeat,
                prompt_tokens=len(ids), output_tokens=len(output), ended=y == 258,
                seconds=elapsed, first_token_seconds=first-start,
                mlx_peak_allocator_bytes=mx.get_peak_memory(),
                mlx_active_allocator_bytes=mx.get_active_memory(),
                answer=answer, correctness=check(answer, expected)))
    summary = {}
    for cached in [True,False]:
        times=[r['seconds'] for r in records if r['cached']==cached and not r['warmup']]
        summary[str(cached)]={'median_seconds':statistics.median(times),'range_seconds':[min(times),max(times)]}
    result=dict(model=str(checkpoint), model_sha256=hashlib.sha256((Path(checkpoint)/'model.safetensors').read_bytes()).hexdigest(),
        loading_seconds=loading, prompt=text, settings='greedy, max120, EOS258; fresh KV each repeat; allocator cache cleared, resident model',
        note='MLX allocator memory is not process RSS or total system RAM. Timed token reads synchronize computation. No p95 estimate.',
        records=records, summary=summary)
    destination.write_text(json.dumps(result,indent=2)); print(json.dumps(summary,indent=2))
if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    benchmark(Path(a.checkpoint),a.output)
