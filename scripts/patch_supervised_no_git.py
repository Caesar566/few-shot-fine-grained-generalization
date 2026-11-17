"""Run with python3 in the extracted project root; changes provenance logging only."""
from pathlib import Path
import ast
import hashlib

path = Path('src/pattern_recognition_coursework/engine/supervised.py')
source = path.read_text(encoding='utf-8')
OLD = "write_json(run/'code_version.json',dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'src').rglob('*.py')}))"
NEW = "write_json(run/'code_version.json',code_version(ROOT))"
HELPER = 'def code_version(root):\n    """Record actual source hashes even when distributed without Git metadata."""\n    root = Path(root)\n    result = dict(head=None, files={str(f.relative_to(root)): sha(f)\n                                   for f in sorted((root/\'src\').rglob(\'*.py\'))})\n    try:\n        result[\'head\'] = subprocess.check_output(\n            [\'git\', \'rev-parse\', \'HEAD\'], cwd=root, text=True,\n            stderr=subprocess.PIPE).strip()\n        result[\'git_available\'] = True\n    except (OSError, subprocess.CalledProcessError) as error:\n        result[\'git_available\'] = False\n        result[\'git_error\'] = str(error)\n        metadata = root/\'SERVER_TRANSFER_METADATA.json\'\n        if metadata.is_file():\n            try:\n                transfer = json.loads(metadata.read_text(encoding=\'utf-8\'))\n                result[\'transfer_source_commit\'] = transfer.get(\'git_commit\')\n                result[\'transfer_source_branch\'] = transfer.get(\'git_branch\')\n            except (OSError, ValueError, AttributeError) as metadata_error:\n                result[\'transfer_metadata_error\'] = str(metadata_error)\n    return result\n\n\n'
if NEW in source and 'def code_version(root):' in source:
    print('Already patched:', path)
else:
    if source.count(OLD) != 1 or 'def code_version(root):' in source:
        raise SystemExit('Unexpected source version; no files changed')
    updated = source.replace(OLD, NEW).replace('def write_json(path, value):', HELPER + 'def write_json(path, value):', 1)
    ast.parse(updated)
    backup = path.with_name(path.name + '.before_no_git_fix')
    if backup.exists():
        raise SystemExit('Backup already exists; no files changed')
    backup.write_bytes(path.read_bytes())
    path.write_text(updated, encoding='utf-8')
    print('Patched:', path, 'Backup:', backup)
print('SHA256:', hashlib.sha256(path.read_bytes()).hexdigest())
