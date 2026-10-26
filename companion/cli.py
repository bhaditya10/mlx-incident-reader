import argparse, json, time, random, math, shutil, hashlib
from pathlib import Path
P = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare-data', 'train', 'generate', 'evaluate', 'export', 'benchmark'])
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--seconds', type=int, default=10800)
    ap.add_argument('--run', default='main')
    ap.add_argument('--export-name', default='incident-reader', help='New export folder; never overwrites')
    ap.add_argument('--checkpoint', default='checkpoints/untrained')
    ap.add_argument('--text', default='Checkout is recovering. Severity is SEV2. Impact lasted 18 minutes.')
    ap.add_argument('--split', choices=['validation', 'test', 'challenge'], default='validation')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--validation-limit', type=int, default=0)
    ap.add_argument('--stop-after', type=int, default=0, help='Checkpoint and stop without changing total schedule steps')
    ap.add_argument('--checkpoint-every', type=int, default=100)
    ap.add_argument('--resume', help='Checkpoint with optimizer and training-state; new output run required')
    args = ap.parse_args()
    if args.command == 'export':
        from export_selected import export_selected
        print(json.dumps(export_selected(args.run, args.export_name), indent=2))
        return
    if args.command == 'benchmark':
        from benchmark import benchmark
        benchmark(P / args.checkpoint, P.parent / 'evidence' / f'benchmark-{args.run}.json')
        return
    if args.command == 'prepare-data':
        from data import prepare
        from checker import selftest
        selftest()
        prepare(P / 'data')
        return
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from model import Model, CONFIG
    from tokenizer import prompt
    from transformers import AutoTokenizer
    from mlx_lm.models.cache import KVCache
    tok = AutoTokenizer.from_pretrained(P / 'checkpoints/untrained')
    mx.random.seed(20261025)
    model = Model()
    model.load_weights(str(P / (args.resume or args.checkpoint) / 'model.safetensors'))
    mx.eval(model.parameters())

    def generate(text, limit=120):
        ids = tok.encode(prompt(text))
        cache = [KVCache() for _ in range(4)]
        x = mx.array([ids])
        result = []
        for _ in range(limit):
            y = int(mx.argmax(model(x, cache)[:, -1, :], axis=-1).item())
            if y == 258:
                break
            result.append(y)
            x = mx.array([[y]])
        return tok.decode(result, skip_special_tokens=False)
    if args.command == 'generate':
        print(generate(args.text))
        return
    if args.command == 'train':
        run = P / 'runs' / args.run
        if run.exists():
            raise RuntimeError('Run already exists; preserve it and choose a new run name.')
        run.mkdir(parents=True)
        (run / 'config.json').write_text(json.dumps(vars(args), indent=2))
        rng = random.Random(20261025)

        def rows(split):
            return [json.loads(s) for s in (P / 'data' / f'{split}.jsonl').read_text().splitlines()]

        def encode(row):
            pre = tok.encode(prompt(row['input']))
            target = tok.encode(json.dumps(row['expected'], separators=(',', ':'))) + [258]
            seq = pre + target
            assert len(seq) <= 512
            return (seq, len(pre))
        train = [encode(r) for r in rows('train')]
        valrows = rows('validation')
        valrows = valrows[:args.validation_limit] if args.validation_limit else valrows
        val = [encode(r) for r in valrows]

        def batch(items):
            length = max((len(i[0]) for i in items))
            ids = []
            masks = []
            for seq, n in items:
                ids.append(seq + [256] * (length - len(seq)))
                masks.append([float(n <= j < len(seq)) for j in range(1, length)])
            return (mx.array(ids), mx.array(masks))

        def loss(m, ids, mask):
            logits = m(ids[:, :-1])
            targets = ids[:, 1:]
            per_token = nn.losses.cross_entropy(
                logits, targets, reduction='none'
            )
            return (per_token * mask).sum() / mask.sum()
        vg = nn.value_and_grad(model, loss)
        optimizer = optim.AdamW(learning_rate=0.0003)
        start_step = 0
        elapsed_before = 0
        if args.resume:
            resume = P / args.resume
            saved = json.loads((resume / 'training-state.json').read_text())
            def tuples(x):
                return tuple(tuples(v) for v in x) if isinstance(x, list) else x
            rng.setstate(tuples(saved['random_state']))
            optimizer.state = tree_unflatten(list(mx.load(str(resume / 'optimizer.safetensors')).items()))
            start_step = saved['step']
            elapsed_before = saved['elapsed_seconds']
            previous_config = json.loads((resume.parent / 'config.json').read_text())
            if args.steps != previous_config['steps'] or args.validation_limit != previous_config['validation_limit']:
                raise ValueError('Resume requires unchanged total steps and validation set')
            if args.steps <= start_step:
                raise ValueError('--steps must exceed saved step; preserve original total steps for the same schedule')
        started = time.monotonic()
        best = float('inf')
        if args.resume:
            past = [json.loads(x) for x in (resume.parent / 'metrics.jsonl').read_text().splitlines()]
            candidates = [r for r in past if 0 < r['step'] <= start_step and 'validation_loss' in r]
            if candidates:
                prior = min(candidates, key=lambda r:r['validation_loss'])
                best = prior['validation_loss']
                (run / 'selected.json').write_text(json.dumps({'checkpoint':str(resume.parent / f"step-{prior['step']}"),'criterion':'lowest validation loss including pre-resume history','validation_loss':best}))

        def validation():
            return sum((float(loss(model, *batch(val[i:i + 8]))) * len(val[i:i + 8]) for i in range(0, len(val), 8))) / len(val)
        initial = validation()
        print(json.dumps({'step': start_step, 'validation_loss': initial}), flush=True)
        with (run / 'metrics.jsonl').open('w') as log:
            log.write(json.dumps({'step': start_step, 'validation_loss': initial}) + '\n')
            log.flush()
            for step in range(start_step + 1, args.steps + 1):
                lr = 0.0003 * (step / 100 if step <= 100 else 0.5 * (1 + math.cos(math.pi * (step - 100) / max(1, args.steps - 100))))
                optimizer.learning_rate = lr
                grads = None
                total = 0
                for _ in range(4):
                    ids, mask = batch(rng.sample(train, 8))
                    l, g = vg(model, ids, mask)
                    mx.eval(l, g)
                    total += float(l) / 4
                    grads = tree_map(lambda x: x / 4, g) if grads is None else tree_map(lambda a, b: a + b / 4, grads, g)
                grads, norm = optim.clip_grad_norm(grads, 1.0)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                if not math.isfinite(total) or not math.isfinite(float(norm)):
                    raise RuntimeError('Nonfinite training')
                record = {'step': step, 'loss': total, 'gradient_norm': float(norm), 'learning_rate': lr, 'elapsed_seconds': elapsed_before + time.monotonic() - started, 'mlx_peak_bytes': mx.get_peak_memory()}
                if step % args.checkpoint_every == 0 or step == args.steps or step == args.stop_after or record['elapsed_seconds'] >= args.seconds:
                    record['validation_loss'] = validation()
                    checkpoint = run / f'step-{step}'
                    checkpoint.mkdir()
                    for f in (P / 'checkpoints/untrained').iterdir():
                        if f.suffix != '.safetensors':
                            shutil.copy2(f, checkpoint / f.name)
                    mx.save_safetensors(str(checkpoint / 'model.safetensors'), dict(tree_flatten(model.parameters())))
                    mx.save_safetensors(str(checkpoint / 'optimizer.safetensors'), dict(tree_flatten(optimizer.state)))
                    (checkpoint / 'training-state.json').write_text(json.dumps({'step': step, 'random_state': rng.getstate(), 'elapsed_seconds': record['elapsed_seconds']}))
                    if record['validation_loss'] < best:
                        best = record['validation_loss']
                        (run / 'selected.json').write_text(json.dumps({'checkpoint': str(checkpoint), 'criterion': 'lowest validation loss; validation sample count recorded in run config', 'validation_loss': best}))
                log.write(json.dumps(record) + '\n')
                log.flush()
                if step % 10 == 0 or 'validation_loss' in record:
                    print(json.dumps(record), flush=True)
                if record['elapsed_seconds'] >= args.seconds or step == args.stop_after:
                    break
        return
    if args.command == 'evaluate':
        from checker import check, rules
        rows = [json.loads(s) for s in (P / 'data' / f'{args.split}.jsonl').read_text().splitlines()]
        rows = rows[:args.limit] if args.limit else rows
        dest = P.parent / 'evidence' / f'eval-{args.run}-{args.split}.jsonl'
        if dest.exists():
            raise RuntimeError('Do not overwrite evaluation')
        with dest.open('w') as f:
            for r in rows:
                answer = generate(r['input'])
                record = {'id': r['id'], 'input': r['input'], 'expected': r['expected'], 'answer': answer, 'check': check(answer, r['expected']), 'rules': check(json.dumps(rules(r['input'])), r['expected'])}
                f.write(json.dumps(record) + '\n')
                f.flush()
                print(json.dumps(record), flush=True)
        return
    raise ValueError('Unknown command')
if __name__ == '__main__':
    main()
