"""Create bounded, loopback workshop settings without overwriting prior settings."""
import json
from pathlib import Path
folder = Path(__file__).resolve().parent.parent / 'local-server'
folder.mkdir(exist_ok=True)
path = folder / 'model_settings.json'
if path.exists():
    raise SystemExit('Settings already exist; inspect them before changing them.')
path.write_text(json.dumps({'version': 1, 'models': {'incident-reader': {
    'turboquant_kv_enabled': False, 'max_context_window': 512,
    'max_tokens': 120, 'temperature': 0, 'top_p': 1, 'top_k': 0,
    'repetition_penalty': 1}}}, indent=2))
print(path)
