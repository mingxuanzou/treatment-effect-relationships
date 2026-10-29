#!/usr/bin/env python3
"""Restore the exact bundled NVSS input from small text-encoded pieces."""
from pathlib import Path
import base64,gzip,hashlib,json
ROOT=Path(__file__).resolve().parent

def prepare():
    spec=json.loads((ROOT/'data_parts/manifest.json').read_text())
    target=ROOT/spec['output']
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==spec['sha256']:
        return
    packed=base64.b64decode(''.join((ROOT/p).read_text() for p in spec['parts']),validate=True)
    raw=gzip.decompress(packed)
    assert len(raw)==spec['size'] and hashlib.sha256(raw).hexdigest()==spec['sha256']
    target.parent.mkdir(parents=True,exist_ok=True)
    tmp=target.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(target)
    print('Restored verified NVSS analytic input.')
if __name__=='__main__':prepare()
