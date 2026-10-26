"""Export a validation-selected checkpoint without overwriting evidence."""
import argparse,json,shutil,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent

def export_selected(run_name='main-02',export_name='incident-reader'):
 run=P/'runs'/run_name
 selected=json.loads((run/'selected.json').read_text())
 recorded=Path(selected['checkpoint'])
 # Resolve within the named run first so a copied project is portable.
 src=run/recorded.name
 if not src.is_dir():raise FileNotFoundError(src)
 dest=P/'checkpoints'/export_name
 if dest.exists():raise RuntimeError('Preserve existing export; choose a new export name')
 dest.mkdir(parents=False)
 for f in src.iterdir():
  if f.is_file() and f.name not in ['optimizer.safetensors','training-state.json']:
   shutil.copy2(f,dest/f.name)
 manifest={**selected,'source_resolved':str(src),'export':str(dest),'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir() if f.is_file()}}
 target=P.parent/'evidence'/f'export-{export_name}.json'
 if target.exists():raise RuntimeError('Export evidence exists')
 target.write_text(json.dumps(manifest,indent=2));return manifest

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',default='main-02');ap.add_argument('--name',default='incident-reader');a=ap.parse_args()
 print(json.dumps(export_selected(a.run,a.name),indent=2))
