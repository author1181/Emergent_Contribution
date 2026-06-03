#!/usr/bin/env python3
"""Generate MANIFEST.sha256 over all committed files (run before tagging a release)."""
import hashlib, os
SKIP = {'.git', '__pycache__', '.ipynb_checkpoints'}
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
lines = []
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in SKIP]
    for fn in sorted(files):
        if fn in ('MANIFEST.sha256',): continue
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, '.')
        lines.append(f"{sha256(p)}  {rel}")
open('MANIFEST.sha256', 'w').write("\n".join(sorted(lines)) + "\n")
print(f"wrote MANIFEST.sha256 ({len(lines)} files)")
