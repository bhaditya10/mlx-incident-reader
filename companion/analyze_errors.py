"""Describe frozen errors; no model selection or parameter changes."""
import json,collections
from pathlib import Path
P=Path(__file__).resolve().parent;E=P.parent/'evidence'
report={}
for split in ['validation','test','challenge']:
 rows=[json.loads(x) for x in (E/f'main-02-{split}/predictions.jsonl').read_text().splitlines()]
 groups={}
 for key in sorted(set(r.get('family',r.get('category','unknown')) for r in rows)):
  rr=[r for r in rows if r.get('family',r.get('category','unknown'))==key]
  groups[key]={'count':len(rr),'exact':sum(r['check']['exact'] for r in rr)/len(rr),'fields':{k:sum(r['check']['fields'][k] for r in rr)/len(rr) for k in rows[0]['expected']}}
 confusion={}
 for field in rows[0]['expected']:
  c=collections.Counter()
  for r in rows:
   try: answer=json.loads(r['answer'])
   except Exception:continue
   if not isinstance(answer,dict):continue
   if answer.get(field)!=r['expected'][field]:c[(str(r['expected'][field]),str(answer.get(field)))]+=1
  confusion[field]=[{'expected':a,'observed':b,'count':n} for (a,b),n in c.most_common(12)]
 report[split]={'groups':groups,'confusions':confusion,'examples':[r for r in rows if not r['check']['exact']][:5]}
(E/'error-analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps({s:r['groups'] for s,r in report.items()},indent=2))
