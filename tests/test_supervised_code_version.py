"""Source provenance remains available in portable archives without .git."""
import hashlib
import json
import subprocess
from unittest.mock import patch

import pytest

from pattern_recognition_coursework.engine.supervised import code_version


@pytest.mark.parametrize('error', [FileNotFoundError('git'),
    subprocess.CalledProcessError(128, ['git', 'rev-parse', 'HEAD'])])
def test_archive_without_git(tmp_path, error):
    (tmp_path/'src').mkdir()
    (tmp_path/'src/example.py').write_bytes(b'x = 1\n')
    (tmp_path/'SERVER_TRANSFER_METADATA.json').write_text(json.dumps(
        {'git_commit': 'original-commit', 'git_branch': 'original-branch'}))
    with patch('subprocess.check_output', side_effect=error):
        result = code_version(tmp_path)
    assert result['head'] is None and result['git_available'] is False
    assert result['transfer_source_commit'] == 'original-commit'
    assert result['files'] == {'src/example.py': hashlib.sha256(b'x = 1\n').hexdigest()}


def test_repository_records_actual_head(tmp_path):
    with patch('subprocess.check_output', return_value='actual-head\n'):
        result = code_version(tmp_path)
    assert result['head'] == 'actual-head' and result['git_available'] is True


def test_archive_without_transfer_metadata(tmp_path):
    with patch('subprocess.check_output', side_effect=FileNotFoundError('git')):
        result = code_version(tmp_path)
    assert result['head'] is None and result['files'] == {}
    assert 'transfer_source_commit' not in result
