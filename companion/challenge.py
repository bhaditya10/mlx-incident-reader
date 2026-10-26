"""Separately authored challenge families. Deliberately out of training distribution.
Ambiguous fields -> null. Unknown service -> null (closed vocabulary).
Gold labels are explicitly specified here, not extracted by the generator/parser.
"""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent/'data'; rows=[]
for n in range(20):
 d=n*3
 cases=[
 ('unseen',f'An interruption to checkout has ended after {d} minutes of impact. The record is SEV2 and resolved.', ['checkout','SEV2','resolved',d]),
 ('order',f'SEV3. {d} minutes of impact. Identified: the incident in billing.', ['billing','SEV3','identified',d]),
 ('irrelevant-number',f'Search is recovering at SEV1. Ticket {800+n} mentions {d} minutes of impact; 400 requests were retried.', ['search','SEV1','recovering',d]),
 ('missing',f'Update {n}: inventory is investigating a SEV4 incident. We do not yet know how long customers were affected.', ['inventory','SEV4','investigating',None]),
 ('zero',f'Report {n}: auth is resolved. Severity SEV2. Customer impact was explicitly 0 minutes.', ['auth','SEV2','resolved',0]),
 ('negation',f'Case {n}: gateway is not resolved; it is recovering. SEV2. Impact lasted {d} minutes.', ['gateway','SEV2','recovering',d]),
 ('contradiction',f'Case {n}: catalog status is identified. The same report states its current status is resolved. SEV3. Duration {d} minutes.', ['catalog','SEV3',None,d]),
 ('unknown',f'Report {n}: analytics is investigating. Severity SEV1. Duration {d} minutes.', [None,'SEV1','investigating',d]),
 ('truncated',f'Case {n}: notifications is recovering. Severity SEV4. The impact lasted', ['notifications','SEV4','recovering',None]),
 ('unicode',f'Note {n} — café 東京: checkout is recovering; severity SEV2; impact {d} minutes. ✓', ['checkout','SEV2','recovering',d])]
 for category,text,gold in cases:
  rows.append({'id':f'challenge-{len(rows):03d}','category':category,'input':text,'expected':dict(zip(['service','severity','status','duration_minutes'],gold))})
path=P/'challenge.jsonl'; path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(P/'challenge-manifest.json').write_text(json.dumps({'count':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'independence':'Separate authoring module and wording families; 20 parameterized variants per category, not 200 independent natural incidents.','policy':'Closed service vocabulary; ambiguity maps affected field to null. Challenge includes behaviors absent from training.'},indent=2))
print('Wrote 200 challenge cases in 10 families.')
