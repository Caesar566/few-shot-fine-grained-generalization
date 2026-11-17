"""Run only stage-01/data tests and persist their unfiltered outcomes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
target = OUT/'evidence/pytest_results.json'
if target.exists():
    raise SystemExit('Refusing to overwrite the existing pytest evidence')
env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
         PYTHONPATH=os.pathsep.join([str(ROOT/'src'),str(OUT)]),
         OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
args=[sys.executable,'-B','-m','pytest',
      'outputs/stage01_data/test_data_checks.py','tests/test_data_pipeline.py',
      'tests/test_episode_plan_and_nabirds.py','-q','-p','no:cacheprovider','-o','addopts=',
      '--basetemp=outputs/stage01_data/evidence/pytest_tmp',
      '--junitxml=outputs/stage01_data/evidence/junit.xml']
proc=subprocess.run(args,cwd=ROOT,env=env,text=True,capture_output=True)
(OUT/'evidence/pytest_stdout.txt').write_text(proc.stdout,encoding='utf-8')
(OUT/'evidence/pytest_stderr.txt').write_text(proc.stderr,encoding='utf-8')
cases=[]
tree=ET.parse(OUT/'evidence/junit.xml')
for case in tree.iter('testcase'):
    status='passed';detail=''
    for tag in ['failure','error','skipped']:
        node=case.find(tag)
        if node is not None:
            status=tag;detail=(node.text or '')
            break
    cases.append({'class':case.attrib.get('classname'),'name':case.attrib['name'],
                  'status':status,'detail':detail})
counts={k:sum(c['status']==k for c in cases) for k in ['passed','failure','error','skipped']}
target.write_text(json.dumps({'command':args,'returncode':proc.returncode,
    'counts':counts,'test_cases':cases},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(proc.stdout)
print('Recorded original exit code:',proc.returncode,'counts:',counts)
refresh=subprocess.run([sys.executable,'-B',str(OUT/'check_data.py'),'--report-only'],cwd=ROOT,env=env)
if refresh.returncode:
    raise SystemExit(refresh.returncode)
raise SystemExit(proc.returncode)
