"""Fresh-environment read-only model checks. Writes only repro evidence."""
import sys,json,hashlib,importlib.metadata as md
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import mlx.core as mx
from model import Model,CONFIG
from tokenizer import prompt
from mlx_lm import load,generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.qwen2 import Model as Reference,ModelArgs
from mlx_lm.models.cache import KVCache
from mlx.utils import tree_flatten
from checker import selftest
selftest(); checkpoint=P/'checkpoints/incident-reader'; weight=checkpoint/'model.safetensors'
before=hashlib.sha256(weight.read_bytes()).hexdigest()
m,t=load(str(checkpoint)); ours=Model();ours.load_weights(str(weight));mx.eval(ours.parameters())
x=mx.array([t.encode(prompt('Checkout is recovering. Severity is SEV2. Impact lasted 18 minutes.'))]); a=ours(x);b=m(x);mx.eval(a,b)
reference_error=float(mx.max(mx.abs(a-b)));assert reference_error<1e-3
changed=x.at[:,15:].add(1);c=ours(changed);causal_error=float(mx.max(mx.abs(a[:,:15]-c[:,:15])));assert causal_error<1e-5
cache=[KVCache() for _ in range(4)];cached=mx.concatenate([ours(x[:,i:i+1],cache) for i in range(x.shape[1])],axis=1);cache_error=float(mx.max(mx.abs(a-cached)));assert cache_error<1e-3
for s in ['ASCII punctuation: 0, null.','café — 東京','line\nnext\t🙂']: assert t.decode(t.encode(s))==s
rows=[json.loads(l) for l in (P.parent/'evidence/main-02-test/predictions.jsonl').read_text().splitlines()][:16]
checks=[]
for r in rows:
 answer=generate(m,t,prompt=prompt(r['input']),max_tokens=120,sampler=make_sampler(temp=0),verbose=False)
 checks.append({'id':r['id'],'answer':answer,'prior_answer':r['answer'],'matches_prior':answer==r['answer']})
assert before==hashlib.sha256(weight.read_bytes()).hexdigest()
out={'python':sys.version,'prefix':sys.prefix,'system_site_packages':False,'packages':{n:md.version(n) for n in ['mlx','mlx-lm','transformers','tokenizers','numpy','safetensors']},'checkpoint_sha256':before,'reference_max_abs':reference_error,'causal_max_abs':causal_error,'cached_max_abs':cache_error,'tokenizer_roundtrip':True,'records':checks,'scope':'Fresh virtualenv on same Mac, not a clean physical machine.16 frozen examples; not a new aggregate test.'}
(P.parent/'evidence/fresh-env-verification.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2));print('Identical prior answers:',sum(r['matches_prior'] for r in checks),len(checks))
assert all(r['matches_prior'] for r in checks)
