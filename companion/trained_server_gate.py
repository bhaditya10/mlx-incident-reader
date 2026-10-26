"""Isolated loopback oMLX test against direct MLX-LM greedy output."""
import json,subprocess,sys,time,urllib.request
from pathlib import Path
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from tokenizer import prompt
P=Path(__file__).resolve().parent
out=P.parent/'evidence'/'trained-server-gate'
if out.exists(): raise RuntimeError('Preserve prior server test')
out.mkdir()
text='Checkout is recovering. Severity is SEV2. Impact lasted 18 minutes.'
model,tok=load(str(P/'checkpoints/incident-reader'))
formatted=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
assert formatted==prompt(text)
direct=generate(model,tok,prompt=formatted,max_tokens=120,sampler=make_sampler(temp=0),verbose=False)
del model; mx.clear_cache()
(out/'model_settings.json').write_text(json.dumps({'version':1,'models':{'incident-reader':{'turboquant_kv_enabled':False,'max_context_window':512,'max_tokens':120,'temperature':0,'top_p':1,'top_k':0,'repetition_penalty':1}}}))
log=(out/'server.log').open('w')
server=subprocess.Popen([sys.executable,'-c','from omlx.cli import main; main()','serve','--model-dir',str(P/'checkpoints'),'--base-path',str(out),'--host','127.0.0.1','--port','18036','--max-concurrent-requests','1','--headless'],stdout=log,stderr=subprocess.STDOUT)
try:
 for _ in range(60):
  if server.poll() is not None: raise RuntimeError('Server exited')
  try:
   json.load(urllib.request.urlopen('http://127.0.0.1:18036/v1/models',timeout=1));break
  except Exception: time.sleep(1)
 payload={'model':'incident-reader','messages':[{'role':'user','content':text}],'temperature':0,'max_tokens':120,'repetition_penalty':1,'top_k':0,'top_p':1,'stream':False}
 req=urllib.request.Request('http://127.0.0.1:18036/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=120) as resp: result=json.load(resp)
 answer=result['choices'][0]['message']['content']
 record={'request':payload,'formatted_prompt':formatted,'prompt_token_ids':tok.encode(formatted),'direct':direct,'server':answer,'exact_text_match':direct==answer,'response':result,'elapsed_seconds':time.monotonic()-start}
 (out/'response.json').write_text(json.dumps(record,indent=2)); print(json.dumps(record,indent=2))
 assert direct==answer
finally:
 server.terminate()
 try: server.wait(timeout=20)
 except subprocess.TimeoutExpired: server.kill();server.wait()
 log.close()
