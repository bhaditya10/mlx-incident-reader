"""Frozen greedy evaluation, grouped by exact prompt length (no padding effects)."""
import argparse,json,time,hashlib
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
from tokenizer import prompt
from checker import check,rules,selftest
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(); ap.add_argument('--run',required=True); ap.add_argument('--split',choices=['validation','test','challenge'],required=True); ap.add_argument('--checkpoint'); ap.add_argument('--output-name'); a=ap.parse_args()
selection=json.loads((P/'runs'/a.run/'selected.json').read_text()); checkpoint=Path(a.checkpoint) if a.checkpoint else P/'runs'/a.run/Path(selection['checkpoint']).name; dest=P.parent/'evidence'/(a.output_name or f'{a.run}-{a.split}')
if dest.exists(): raise RuntimeError('Evaluation already exists; preserve frozen output')
dest.mkdir(); selftest(); model,tok=load(str(checkpoint)); rows=[json.loads(s) for s in (P/'data'/f'{a.split}.jsonl').read_text().splitlines()]; groups={}
for row in rows:
 ids=tok.encode(prompt(row['input'])); groups.setdefault(len(ids),[]).append((row,ids))
results=[]; start=time.monotonic()
with (dest/'predictions.jsonl').open('w') as f:
 for length,group in sorted(groups.items()):
  for i in range(0,len(group),16):
   batch=group[i:i+16]; x=mx.array([ids for row,ids in batch]); cache=[KVCache() for _ in model.layers]; outputs=[[] for _ in batch]; done=[False]*len(batch)
   for step in range(120):
    ys=mx.argmax(model(x,cache)[:,-1,:],axis=-1).tolist()
    for j,y in enumerate(ys):
     if not done[j]:
      if y==258: done[j]=True
      else: outputs[j].append(y)
    if all(done): break
    x=mx.array([[y] for y in ys])
   for (row,ids),output,ended in zip(batch,outputs,done):
    answer=tok.decode(output,skip_special_tokens=False); result={**row,'answer':answer,'ended':ended,'output_tokens':len(output),'check':check(answer,row['expected']),'rules':check(json.dumps(rules(row['input'])),row['expected'])}; results.append(result); f.write(json.dumps(result)+'\n')
   f.flush()
  print(json.dumps({'completed':len(results),'total':len(rows),'elapsed':time.monotonic()-start}),flush=True)
summary={'count':len(results),'checkpoint':str(checkpoint),'selection':selection,'split':a.split,'elapsed_seconds':time.monotonic()-start,'generation':'greedy, max120, stop258, exact-length prompt batches16','checkpoint_sha256':hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()}
for name in ['check','rules']:
 summary[name]={key:sum(r[name][key] for r in results)/len(results) for key in ['json','schema','exact']}; summary[name]['fields']={key:sum(r[name]['fields'][key] for r in results)/len(results) for key in ['service','severity','status','duration_minutes']}
summary['categories']={c:{'count':sum(r.get('category')==c for r in results),'exact':sum(r['check']['exact'] for r in results if r.get('category')==c)/sum(r.get('category')==c for r in results)} for c in set(r.get('category') for r in results) if c}
(dest/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
