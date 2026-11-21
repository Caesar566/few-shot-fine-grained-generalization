"""Execute Stage06 tests, then smoke, then full only if the frozen smoke passes."""
from pathlib import Path
import json
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'outputs/stage06_nabirds'


def run(command,log):
    with Path(log).open('w') as stream:
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    (STAGE/'tests').mkdir(parents=True,exist_ok=True)
    run([sys.executable,'-m','pytest','tests/test_nabirds_automatic.py','tests/test_episode_plan_and_nabirds.py',
         'tests/test_representation_comparison.py','tests/test_advanced_models.py','tests/test_few_shot_baselines.py',
         '-q',f'--junitxml={STAGE}/tests/cuda_tests.xml'],STAGE/'tests/cuda_tests.log')
    common=[sys.executable,'-m','pattern_recognition_coursework.engine.nabirds_transfer',
            '--allow-automatic-screening','--device','cuda','--batch-size','8','--query-chunk','4']
    smoke=STAGE/'smoke_imagenet_cuda';full=STAGE/'full_seed42_600_cuda'
    run([*common,'--mode','smoke','--output-dir',str(smoke),'--workers','0'],STAGE/'smoke_console.log')
    gate=json.loads((smoke/'checks.json').read_text())
    assert gate['status']=='PASS' and gate['mode']=='smoke'
    if gate['device']!='cuda':raise RuntimeError('CUDA requested for this run; smoke fell back to CPU, stop before full')
    print('Stage06 CUDA smoke passed. Starting all three frozen backbones on the same 600 paired plans.',flush=True)
    run([*common,'--mode','full','--validation-report',str(smoke/'checks.json'),
         '--output-dir',str(full),'--workers','2'],STAGE/'full_console.log')
    print('Stage06 full evaluation complete:',full/'report.md',flush=True)


if __name__=='__main__':main()
