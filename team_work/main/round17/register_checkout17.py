"""仅登记已核对为换行转换的Git对象字节，避免干净克隆被误报损坏。"""
from common17 import *


if __name__ == '__main__':
    dest = HERE / 'checkout_bytes.json'
    if dest.exists():
        raise ValueError('兼容清单已存在，拒绝覆盖')
    reg = read(HERE / 'registry.json')
    mapping = {}
    for rel in reg['checkout_normalized_files']:
        current = (ROOT / rel).read_bytes()
        blob = subprocess.check_output(['git', 'show', reg['baseline_commit'] + ':' + rel], cwd=ROOT)
        if current.replace(b'\r\n', b'\n') != blob.replace(b'\r\n', b'\n'):
            raise ValueError('超出换行差异: ' + rel)
        if hashlib.sha256(current).hexdigest() != reg['tracked_files'][rel]:
            raise ValueError('登记后文件已变动: ' + rel)
        mapping[rel] = hashlib.sha256(blob).hexdigest()
    dump(dest, {'registry_sha256': sha(HERE / 'registry.json'), 'git_blob_sha256': mapping,
                'role': '仅接受登记时本地字节或基准Git对象字节；不改旧文件或旧哈希'})
    print(len(mapping))
