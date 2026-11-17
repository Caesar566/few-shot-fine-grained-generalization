"""Read-only repository checks; writes evidence only to docs/research_audit/evidence.

No torch emulation: extracted stdlib helpers are explicitly identified as such.
"""
import ast
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import runpy
import statistics
import subprocess
import sys
from collections import Counter, defaultdict

ROOT = Path('/home/caesar/Projects/few-shot-fine-grained-generalization')
OUT = ROOT / 'docs/research_audit/evidence'
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(ROOT)

def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

protected = [p for base in ['src','scripts','tests','reports','output','data/splits']
             for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
protected += [ROOT/p for p in ['.gitignore','README.md','PROJECT_STATUS.md','pyproject.toml','uv.lock']]
before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
dump('protected_files_before.json', before)

commands = {}
def command(label, args, env=None):
    r = subprocess.run(args, text=True, capture_output=True, env=env)
    commands[label] = {'args':args,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    print(label, 'exit', r.returncode, flush=True)
    return r

command('git', ['git','status','--short'])
command('head', ['git','log','-1','--format=%H %s'])
command('branch', ['git','branch','--show-current'])
command('venv_imports', [str(ROOT/'.venv/bin/python'),'-B','-c',
    "import sys,importlib.util; print(sys.version); "
    "[(print(m,importlib.util.find_spec(m))) for m in ['torch','torchvision','numpy','scipy','PIL','pytest']]" ])
command('venv_pytest', [str(ROOT/'.venv/bin/python'),'-B','-m','pytest','-q','-p','no:cacheprovider','-o','addopts=','--collect-only'])
env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT/'src'))
command('system_pytest_collection', [sys.executable,'-B','-m','pytest','--collect-only','-q','-p','no:cacheprovider','-o','addopts='],env)
cub=ROOT/'cub2002011/versions/7/CUB_200_2011'
command('cub_official_validation', [str(ROOT/'.venv/bin/python'),'-B','scripts/validate_cub.py','--data-root',str(cub),'--check-images'])

syntax=[]
for base in ['src','scripts','tests']:
    for p in sorted((ROOT/base).rglob('*.py')):
        compile(p.read_text(),str(p),'exec')
        syntax.append(str(p.relative_to(ROOT)))
import tomllib
project=tomllib.loads((ROOT/'pyproject.toml').read_text())
lock=tomllib.loads((ROOT/'uv.lock').read_text())
dump('syntax_and_config.json',{'syntax_pass':syntax,'count':len(syntax),'requires_python':project['project']['requires-python'],'lock_packages':len(lock.get('package',[]))})

def extracted(file, names, ns):
    """Execute unchanged selected AST definitions, excluding unavailable imports."""
    path=ROOT/file
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected}==set(names)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),ns)
    return ns

checks=[]
def check(name, fn):
    try:
        result=fn()
        checks.append({'name':name,'status':'passed','details':result})
        print(name,'passed',flush=True)
    except Exception as e:
        checks.append({'name':name,'status':'failed','error':repr(e)})
        print(name,'FAILED',repr(e),flush=True)

sampler_ns=extracted('src/pattern_recognition_coursework/data/cub_dataset.py',['FewShotBatchSampler'],
                     {'random':random,'Sampler':object,'CUBFewShotDataset':object})
Sampler=sampler_ns['FewShotBatchSampler']
plan_ns=extracted('src/pattern_recognition_coursework/data/episodes.py',
                 ['dataset_fingerprint','plan_hash','validate_plan'],{'Path':Path,'hashlib':hashlib,'json':json})
class ManifestDataset:
    def __init__(self,root,manifest):
        self.data_root=root
        rows=[s.split(maxsplit=2) for s in manifest.read_text().splitlines() if s.strip()]
        names=sorted({r[1] for r in rows}); mapping={n:i for i,n in enumerate(names)}
        self.samples=[(root/'images'/r[2],mapping[r[1]]) for r in rows]
        self.class_to_indices=defaultdict(list)
        for i,(_,label) in enumerate(self.samples):self.class_to_indices[label].append(i)
    def __len__(self):return len(self.samples)

def sampler_check():
    d=ManifestDataset(cub,ROOT/'data/splits/train_images.txt')
    for shot in [1,5]:
        for seed in [42,43,44]:
            sampler=Sampler(d,20,5,shot,5,seed)
            batches=list(sampler)
            assert batches==list(sampler)
            assert batches!=list(Sampler(d,20,5,shot,5,seed+1))
            plan={'schema':'few-shot-episodes-v1','dataset_fingerprint':plan_ns['dataset_fingerprint'](d),
                  'n_way':5,'k_shot':shot,'n_query':5,'episodes':20,'batches':batches}
            plan_ns['validate_plan'](plan,d,5,shot,5,20)
            bad=copy.deepcopy(plan);bad['batches'][0][-1]=bad['batches'][0][0]
            try:plan_ns['validate_plan'](bad,d,5,shot,5,20)
            except ValueError:pass
            else:raise AssertionError('duplicate accepted')
    return {'episodes_checked':120,'way':5,'shots':[1,5],'query_per_class':5,'seeds':[42,43,44],
            'scope':'unchanged extracted sampler/plan functions, not torch Dataset/DataLoader execution'}
check('sampler_and_plan',sampler_check)

nab_ns={'Path':Path,'PurePosixPath':__import__('pathlib').PurePosixPath,'PureWindowsPath':__import__('pathlib').PureWindowsPath,
        'hashlib':hashlib,'json':json,'re':__import__('re'),'unicodedata':__import__('unicodedata'),
        'Counter':Counter,'defaultdict':defaultdict,'PROTOCOL':'nabirds-manual-nonoverlap-v1',
        'METADATA_FILES':('classes.txt','images.txt','image_class_labels.txt','hierarchy.txt')}
extracted('src/pattern_recognition_coursework/data/nabirds.py',
          ['sha256_file','normalize_bird_name','_table','_safe_image_path','load_nabirds_metadata','_ancestors','create_review'],nab_ns)
def nab_check():
    meta=nab_ns['load_nabirds_metadata'](ROOT/'nabirds')
    missing=[r for r in meta['images'].values() if not (ROOT/'nabirds/images'/r).is_file()]
    assert not missing
    review=nab_ns['create_review'](ROOT/'nabirds',ROOT/'data/splits/train_classes.txt')
    assert all(r['decision']=='pending' for r in review['rows'])
    out={'metadata_classes':len(meta['classes']),'image_bearing_classes':len(set(meta['labels'].values())),
         'indexed_images':len(meta['images']),'missing_images':missing,'pending_review_rows':len(review['rows']),
         'candidate_overlap_rows':sum(bool(r['candidate_cub_overlap']) for r in review['rows']),
         'metadata_sha256':meta['hashes'],'scope':'unchanged extracted stdlib metadata helpers; no real Dataset or human approval'}
    dump('nabirds_metadata.json',out)
    return out
check('nabirds_metadata_and_paths',nab_check)

def decode_check():
    from PIL import Image
    paths=[]
    for root in [cub,ROOT/'nabirds']:
        rows=(root/'images.txt').read_text().splitlines()
        for row in random.Random(42).sample(rows,20):
            path=root/'images'/row.split(maxsplit=1)[1]
            with Image.open(path) as im:im.load();assert im.convert('RGB').size[0]>0
            paths.append(str(path.relative_to(ROOT)))
    return {'decoded':len(paths),'paths':paths,'scope':'Pillow only, not torchvision transforms; remaining files not decoded'}
check('sample_image_decode',decode_check)

def duplicate_check():
    groups=defaultdict(list)
    for split in ['train','val','test']:
        for line in (ROOT/f'data/splits/{split}_images.txt').read_text().splitlines():
            i,c,r=line.split(maxsplit=2)
            groups[digest(cub/'images'/r)].append({'split':split,'id':i,'class':c,'path':r})
    duplicates=[{'sha256':h,'items':rows} for h,rows in groups.items() if len(rows)>1]
    out={'indexed_images':sum(map(len,groups.values())),'unique_sha256':len(groups),'duplicate_groups':duplicates,
         'cross_split_duplicate_groups':[g for g in duplicates if len({r['split'] for r in g['items']})>1],
         'scope':'CUB exact bytes only; no NABirds/cross-dataset/pretraining near-duplicate audit'}
    dump('cub_exact_duplicates.json',out)
    assert not out['cross_split_duplicate_groups']
    return out
check('cub_exact_duplicates',duplicate_check)

sel=extracted('src/pattern_recognition_coursework/analysis/select.py',['select_fusion_weight'],{'defaultdict':defaultdict,'math':math})
def select_check():
    rows=[{'schema_version':2,'status':'validation_only','seed':s,'model_config':{'method':'global_local','metric':'cosine','local_weight':w},
           'training_config':{'seed':s,'epochs':3},'validation_plan_hash':f'plan-{s}','best_val_accuracy':.6+w/10}
          for s in [42,43,44] for w in [0,.5]]
    assert sel['select_fusion_weight'](rows)['selected_local_weight']==.5
    bad=copy.deepcopy(rows);bad[0]['cub_novel_test']={}
    try:sel['select_fusion_weight'](bad)
    except ValueError:pass
    else:raise AssertionError('test result accepted')
    return 'valid synthetic selection and rejection of explicit CUB test record; stdlib helper execution only'
check('fusion_selection_guards',select_check)

sum_ns={'math':math,
        'PROTOCOL_FIELDS':('source_dataset','target_dataset','target_manifest_hash','n_way','k_shot','n_query','episodes','image_size','preprocessing'),
        'TASK_FIELDS':('n_way','k_shot','n_query','episodes','image_size','preprocessing')}
extracted('src/pattern_recognition_coursework/analysis/summarize.py',['_records_by_seed','_paired'],sum_ns)
record_ns=extracted('tests/test_result_summary.py',['record'],{})
def checkpoint_gap_check():
    cub_rows=[record_ns['record'](s,.65) for s in [42,43,44]]
    nab_rows=[record_ns['record'](s,.40,target='nabirds') for s in [42,43,44]]
    for r in cub_rows:r['checkpoint_sha256']='a'*64
    for r in nab_rows:r['checkpoint_sha256']='b'*64
    result=sum_ns['_paired'](cub_rows,nab_rows,'cub','nabirds',same_model=True)
    return {'bug_reproduced':True,'different_checkpoint_hashes_accepted':True,'accepted_seeds':result[0],
            'scope':'unchanged _paired implementation; synthetic records; not a real domain-gap experiment'}
check('checkpoint_identity_gap_reproducer',checkpoint_gap_check)

def result_check():
    summary={}
    error_audit=json.loads((ROOT/'reports/records/cub_error_audit_v1.json').read_text())
    t2=math.sqrt(2*.95**2/(1-.95**2))
    for shot in [1,5]:
        p=ROOT/f'reports/records/cub_study_v1_{shot}shot_summary.json'
        r=json.loads(p.read_text())
        values={m:[100*error_audit['results'][f'{m}_{shot}shot_seed{s}']['accuracy'] for s in r['seeds']] for m in ['protonet','global_local']}
        values['gain']=[b-a for a,b in zip(values['protonet'],values['global_local'])]
        checks_here={}
        for name,key in [('protonet','baseline'),('global_local','candidate'),('gain','paired_improvement')]:
            v=values[name];mean=statistics.mean(v);std=statistics.stdev(v);half=t2*std/math.sqrt(3)
            expected=[mean-half,mean+half]
            assert math.isclose(mean,r[key]['mean_pp'],abs_tol=1e-8)
            assert all(math.isclose(a,b,abs_tol=1e-7) for a,b in zip(expected,r[key]['ci95_pp']))
            checks_here[key]={'mean_pp':mean,'ci95_pp':expected}
        summary[str(shot)]=checks_here
    digest_current=hashlib.sha256()
    for p in sorted((ROOT/'src/pattern_recognition_coursework').rglob('*.py')):
        digest_current.update(p.relative_to(ROOT/'src/pattern_recognition_coursework').as_posix().encode());digest_current.update(b'\0');digest_current.update(bytes.fromhex(digest(p)))
    summary['current_source_fingerprint']=digest_current.hexdigest()
    records={}
    for p in sorted((ROOT/'reports/records').glob('*.json')):
        r=json.loads(p.read_text());records[p.name]={'sha256':digest(p),'top_level_keys':list(r)}
        if 'baseline_training_config' in r:
            records[p.name]['historical_source_fingerprint']=r['baseline_training_config'].get('code_fingerprint_sha256')
            records[p.name]['split_hashes_match']={s:h==digest(ROOT/f'data/splits/{s}_images.txt') for s,h in r['baseline_training_config'].get('split_hashes',{}).items()}
    with (ROOT/'reports/records/cub_error_review_v1.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    summary['error_review']={'rows':len(rows),'annotated_rows':sum(any(r.get(k,'').strip() for k in ['pose','lighting','background','occlusion','label_issue']) for r in rows)}
    summary['records']=records
    dump('historical_result_consistency.json',summary)
    return {'summary_arithmetic_consistent':True,'scope':'JSON cross-consistency and n=3 t-interval arithmetic only; no checkpoint or predictions available'}
check('historical_summary_arithmetic',result_check)

inventory={}
for base in ['cub2002011','nabirds','data','reports','output','outputs']:
    p=ROOT/base;counts=Counter();artifacts=[];n=0
    if p.exists():
        for cur,dirs,files in os.walk(p):
            for f in files:
                n+=1;q=Path(cur)/f;counts[q.suffix]+=1
                if q.suffix in {'.pt','.pth','.ckpt','.json','.jsonl','.csv','.log'} or 'tfevents' in f:artifacts.append(str(q.relative_to(ROOT)))
    inventory[base]={'exists':p.exists(),'files':n,'suffixes':dict(counts),'artifacts':artifacts}
dump('inventory.json',inventory)
dump('commands.json',commands)
dump('stdlib_checks.json',checks)
after={p:digest(ROOT/p) for p in before}
dump('protected_files_after.json',after)
dump('preservation.json',{'changed_protected_files':[p for p in before if before[p]!=after[p]],'files_checked':len(before)})
print('Evidence directory:',OUT,'checks',len(checks),'failed',sum(r['status']!='passed' for r in checks),flush=True)
