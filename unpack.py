"""Verify and unpack the included workshop assets. Standard library only."""
from pathlib import Path
import hashlib, json, zipfile, shutil
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'ASSETS.json').read_text())
for name, info in manifest.items():
    archive = root / name
    if hashlib.sha256(archive.read_bytes()).hexdigest() != info['sha256']:
        raise SystemExit(f'Checksum mismatch: {name}')
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                raise SystemExit('Unsafe archive path')
            if member.is_dir():
                continue
            content = z.read(member)
            if target.exists():
                if target.read_bytes() != content:
                    raise SystemExit(f'Preserving changed file: {target.relative_to(root)}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print(f'Verified and unpacked {name}')
source = root / 'companion/checkpoints/incident-reader/model.safetensors'
target = root / 'companion/runs/main-02/step-700/model.safetensors'
if not target.exists():
    shutil.copy2(source, target)
for name in ['run','demo','walkthrough','update-demo']:
    path = root / 'companion' / name
    if path.exists(): path.chmod(path.stat().st_mode | 0o111)
print('Ready. Follow README.md to create companion/.venv.')
