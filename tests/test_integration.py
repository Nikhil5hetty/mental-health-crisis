import subprocess
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not (ROOT/'data/raw').is_dir(),reason='Published raw datasets absent')
def test_build_processed_references():
    result=subprocess.run([sys.executable,'-I',str(ROOT/'scripts/build_processed.py')],
                          cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert 'Crisis by k humans: [36, 33, 33, 26, 78]' in result.stdout
    assert "Medical advice: {'no': 1729, 'yes': 220, 'not_sure': 51}" in result.stdout
    assert 'RQ2 complete-case items: 190' in result.stdout
    assert 'PASS: all reference and alignment checks' in result.stdout
