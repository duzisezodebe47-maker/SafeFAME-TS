"""Copy a signed Bundle directory to removable/shared storage without ZIP."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(folder):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(folder).as_posix(): digest(p) for p in sorted(
        p for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json')}
    if actual != manifest['files']:
        raise ValueError(f'Hash/inventory verification failed: {folder}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('destination_parent', type=Path)
    args = parser.parse_args()
    source = args.bundle.resolve()
    before = verify(source)
    target = args.destination_parent.resolve() / source.name
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    after = verify(target)
    if before['bundle_signature'] != after['bundle_signature']:
        raise ValueError('Transferred Bundle signature changed')
    print(json.dumps({'status': 'PASS', 'destination': str(target),
                      'bundle_signature': after['bundle_signature'],
                      'files': len(after['files'])}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
