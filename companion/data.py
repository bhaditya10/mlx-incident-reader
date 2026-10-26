"""Synthetic task, predetermined family + attribute-combination partitions."""
import hashlib,json,random,re
from pathlib import Path
SERVICES=['checkout','billing','search','inventory','auth','notifications','gateway','catalog']
SEVERITIES=['SEV1','SEV2','SEV3','SEV4']; STATUSES=['investigating','identified','recovering','resolved']
FAMILIES={
 'train':[
 '{s} is {st}. Severity is {sev}. {duration}',
 'Service: {s}. Status: {st}. Severity: {sev}. {duration}',
 'Incident for {s}: {sev}, {st}. {duration}',
 'The {s} service has severity {sev} and status {st}. {duration}',
 '{sev} incident. We are {st} on {s}. {duration}',
 '{duration} Current status for {s}: {st}. Classification: {sev}.',
 ],
 'validation':[
 'Update regarding {s}: classified {sev}; marked {st}. {duration}',
 '{duration} For service {s}, the incident is {st} at severity {sev}.',
 ],
 'test':[
 'Operations bulletin: {s} / {sev} / {st}. {duration}',
 'Regarding the {sev} issue affecting {s}, status is now {st}. {duration}',
 ]}
DURATIONS=['Impact lasted {d} minutes.','Duration was {d} minutes.','Customer impact: {d} minutes.']
def bucket(s,sev,st,d):
 n=int(hashlib.sha256(json.dumps([s,sev,st,d]).encode()).hexdigest()[:8],16)%10
 return 'train' if n<8 else 'validation' if n==8 else 'test'
def prepare(folder):
 folder=Path(folder)
 if any((folder / f'{split}.jsonl').exists() for split in ['train','validation','test']): raise RuntimeError('Data already exists; preserve it and use a separate clone for regeneration.')
 folder.mkdir(exist_ok=True); manifest={}; all_text=set(); normalized=set(); rng=random.Random(20261025)
 for split,n in [('train',30000),('validation',3000),('test',3000)]:
  rows=[]; seen=set()
  while len(rows)<n:
   s=rng.choice(SERVICES); sev=rng.choice(SEVERITIES); st=rng.choice(STATUSES); d=rng.choice([None]*20+list(range(121)))
   if bucket(s,sev,st,d)!=split: continue
   f=rng.randrange(len(FAMILIES[split])); duration=rng.choice(DURATIONS).format(d=d) if d is not None else rng.choice(['Duration not reported.','No duration was provided.',''])
   text=FAMILIES[split][f].format(s=s,sev=sev,st=st,duration=duration).strip()
   if text in seen: continue
   seen.add(text); assert text not in all_text
   norm=re.sub(r'\W+',' ',text.lower()).strip(); assert norm not in normalized or norm in {re.sub(r'\W+',' ',r['input'].lower()).strip() for r in rows}
   rows.append({'id':f'{split}-{len(rows):05d}','family':f'{split}-{f}','input':text,'expected':dict(service=s,severity=sev,status=st,duration_minutes=d)})
  path=folder/f'{split}.jsonl'; path.write_text(''.join(json.dumps(r)+'\n' for r in rows)); all_text.update(seen); normalized.update(re.sub(r'\W+',' ',r['input'].lower()).strip() for r in rows)
  manifest[split]={'count':n,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'families':len(FAMILIES[split])}
 manifest['split_policy']='Disjoint wording families AND disjoint full attribute tuples by SHA256 bucket; same individual values can recur.'
 manifest['near_duplicate_scope']='Normalized exact text and template-family separation checked; semantic overlap is intentional. Paraphrases are not guaranteed independent natural language.'
 (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))
