"""Small inspectable experiments; run separately from training/benchmarking."""
import json,time,statistics
from pathlib import Path
import mlx.core as mx

def lazy():
 mx.random.seed(7); x=mx.random.normal((1024,1024)); mx.eval(x); mx.synchronize()
 rows=[]
 for i in range(6):
  start=time.perf_counter(); y=mx.sin(x)@x; built=time.perf_counter(); mx.eval(y); mx.synchronize(); ended=time.perf_counter()
  rows.append({'warmup':i==0,'construction_seconds':built-start,'construction_and_execution_seconds':ended-start,'checksum':float(y[0,0])})
 return rows

def gradient():
 def loss(w): return (w*2-6)**2
 w=mx.array(0.0); history=[]
 for step in range(5):
  l,g=mx.value_and_grad(loss)(w); history.append({'step':step,'weight':float(w),'loss':float(l),'gradient':float(g)}); w=w-.1*g
 return history

def attention():
 q=mx.array([[1.,0.],[0.,1.],[1.,1.]])
 k=q; v=mx.array([[10.,0.],[0.,20.],[30.,30.]])
 scores=q@k.T/(2**.5); mask=mx.arange(3)[None,:]<=mx.arange(3)[:,None]
 weights=mx.softmax(mx.where(mask,scores,-float('inf')),axis=-1)
 return {'q':q.tolist(),'k':k.tolist(),'v':v.tolist(),'scores':scores.tolist(),'allowed':mask.tolist(),'weights':weights.tolist(),'output':(weights@v).tolist(),'scope':'Illustrative 3-position/2-dimensional attention; not the trained model attention.'}
if __name__=='__main__':
 result={'lazy':lazy(),'gradient':gradient(),'attention':attention()}
 path=Path(__file__).resolve().parent.parent/'evidence'/'mechanism-labs.json'; path.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
