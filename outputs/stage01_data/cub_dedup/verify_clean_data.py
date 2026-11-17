"""Verify real derived indexes/episodes and rehash originals, without training."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import DataLoader

from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset, FewShotBatchSampler, get_cub_transforms
from pattern_recognition_coursework.data.cub_dedup import sha256_file
from pattern_recognition_coursework.data.episodes import dataset_fingerprint, plan_hash, validate_plan

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
DATA=ROOT/'cub2002011/versions/7/CUB_200_2011'
ORIGINAL=ROOT/'data/splits'
CLEAN=ORIGINAL/'cub_clean_v1'


def save(name,data):
    (OUT/'evidence'/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    if (OUT/'evidence/real_data_verification.json').exists():
        raise SystemExit('Refusing to overwrite the existing verification run')
    torch.set_num_threads(2)
    marker=json.loads((CLEAN/'dedup_manifest.json').read_text())
    with (OUT/'full_inventory.csv').open(encoding='utf-8-sig') as stream:
        inventory=list(csv.DictReader(stream))
    original_hashes={r['relative_path']:r['sha256'] for r in inventory}
    physical={p.relative_to(DATA/'images').as_posix() for p in (DATA/'images').rglob('*') if p.is_file()}
    assert physical==original_hashes.keys()
    for i,relative in enumerate(sorted(physical)):
        assert sha256_file(DATA/'images'/relative)==original_hashes[relative],relative
        if (i+1)%4000==0:print('Original image preservation:',i+1,len(physical),flush=True)
    assert all(sha256_file(ORIGINAL/name)==h for name,h in marker['source_split_sha256'].items())
    assert all(sha256_file(DATA/name)==h for name,h in marker['source_metadata_sha256'].items())
    with (CLEAN/'image_hashes.csv').open(encoding='utf-8-sig') as stream:
        clean_hashes={r['relative_path']:r['sha256'] for r in csv.DictReader(stream)}
    assert len(clean_hashes)==len(set(clean_hashes.values()))==11787
    datasets={s:CUBFewShotDataset(DATA,CLEAN/f'{s}_images.txt',get_cub_transforms(224,False),require_clean_index=True)
              for s in ['train','val','test']}
    raw={s:CUBFewShotDataset(DATA,ORIGINAL/f'{s}_images.txt') for s in datasets}
    comparisons={}
    for s,ds in datasets.items():
        assert ds.classes==raw[s].classes and ds.class_to_idx==raw[s].class_to_idx
        assert all(p.exists() for p,_ in ds.samples)
        assert set(ds.class_to_indices)==set(range(len(ds.classes)))
        class_counts={ds.classes[label]:len(indices) for label,indices in ds.class_to_indices.items()}
        assert min(class_counts.values())>=20  # maximum K+Q verified here is 5+15
        comparisons[s]={'original_images':len(raw[s]),'clean_images':len(ds),'classes':len(ds.classes),
            'class_mapping_unchanged':True,'min_images_per_class':min(class_counts.values()),
            'original_index_fingerprint':dataset_fingerprint(raw[s]),'clean_index_fingerprint':dataset_fingerprint(ds)}
    pairs={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert set(datasets[a].classes).isdisjoint(datasets[b].classes)
        assert {str(p) for p,_ in datasets[a].samples}.isdisjoint(str(p) for p,_ in datasets[b].samples)
        pairs[a+'_'+b]={'class_overlap':0,'path_overlap':0}
    tasks=[]
    for split,ds in datasets.items():
        for shot in [1,5]:
            for query in [1,5,15]:
                sampler=FewShotBatchSampler(ds,1000,5,shot,query,seed=42)
                batches=list(sampler)
                assert batches==list(sampler)
                assert batches!=list(FewShotBatchSampler(ds,1000,5,shot,query,seed=43))
                for batch in batches:
                    assert len(batch)==len(set(batch))==5*(shot+query)
                    fingerprints=[clean_hashes[ds.samples[i][0].relative_to(DATA/'images').as_posix()] for i in batch]
                    assert len(set(fingerprints))==len(fingerprints)
                plan={'schema':'few-shot-episodes-v1','dataset_fingerprint':dataset_fingerprint(ds),
                    'n_way':5,'k_shot':shot,'n_query':query,'episodes':1000,'seed':42,'batches':batches}
                validate_plan(plan,ds,5,shot,query,1000)
                tasks.append({'split':split,'n_way':5,'k_shot':shot,'n_query':query,'episodes':1000,
                    'seed':42,'repeat_identical':True,'different_seed_changes_tasks':True,
                    'index_and_content_duplicates':0,'plan_hash':plan_hash(plan)})
    # Contrast with the untouched original test manifest, not a historical run.
    raw_collision=None
    for ep,batch in enumerate(FewShotBatchSampler(raw['test'],1000,5,5,5,42)):
        before=[original_hashes[raw['test'].samples[i][0].relative_to(DATA/'images').as_posix()] for i in batch]
        overlap=set(before[:25]).intersection(before[25:])
        if overlap:
            raw_collision={'episode_zero_based':ep,'seed':42,'n_way':5,'k_shot':5,'n_query':5,
                'duplicate_sha256':sorted(overlap),'note':'new sampling of unchanged original index, not historical test inference'}
            break
    assert raw_collision is not None and raw_collision['episode_zero_based']==200
    one_raw_batch=list(FewShotBatchSampler(raw['test'],1,5,5,5,42))
    old_plan={'schema':'few-shot-episodes-v1','dataset_fingerprint':dataset_fingerprint(raw['test']),
              'n_way':5,'k_shot':5,'n_query':5,'episodes':1,'batches':one_raw_batch}
    rejected_old_plan=False
    try:validate_plan(old_plan,datasets['test'],5,5,5,1)
    except ValueError:rejected_old_plan=True
    assert rejected_old_plan
    batches=[]
    for split,ds in datasets.items():
        for shot in [1,5]:
            images,labels=next(iter(DataLoader(ds,batch_sampler=FewShotBatchSampler(ds,1,5,shot,5,42),num_workers=0)))
            assert list(images.shape)==[5*(shot+5),3,224,224]
            assert labels.shape==(5*(shot+5),)
            assert torch.isfinite(images).all()
            batches.append({'split':split,'shot':shot,'query':5,'images_shape':list(images.shape),
                'images_dtype':str(images.dtype),'labels_dtype':str(labels.dtype),
                'images_min':float(images.min()),'images_max':float(images.max()),'labels':labels.tolist()})
    baseline=ROOT/'outputs/stage01_data/evidence/protected_files_before.json'
    preserved={}
    if baseline.exists():
        old=json.loads(baseline.read_text())
        names=[p for p in old if p.startswith(('src/pattern_recognition_coursework/models/',
                                              'src/pattern_recognition_coursework/engine/'))]
        assert all(sha256_file(ROOT/p)==old[p] for p in names)
        preserved={'compared_to':'stage01 before snapshot','model_and_engine_files_unchanged':names}
    tests=[]
    for tc in ET.parse(OUT/'evidence/junit.xml').iter('testcase'):
        status='passed'
        for tag in ['failure','error','skipped']:
            if tc.find(tag) is not None:status=tag
        tests.append({'name':tc.attrib['name'],'status':status})
    counts=dict(Counter(t['status'] for t in tests))
    assert counts.get('passed')==28 and not any(counts.get(k,0) for k in ['failure','error','skipped'])
    result={'clean_index_verified':True,'class_mapping_and_counts':comparisons,'split_intersections':pairs,
            'episodes':tasks,'total_episodes_checked':sum(t['episodes'] for t in tasks),
            'original_index_collision_reproduced':raw_collision,'original_test_plan_rejected_for_clean_index':rejected_old_plan,
            'image_and_label_batches':batches,'physical_images_rehashed':len(physical),
            'original_image_paths_and_hashes_unchanged':True,'original_six_split_files_unchanged':True,
            'official_metadata_unchanged':True,'model_and_engine_preservation':preserved,
            'pytest_counts':counts,'model_training_executed':False}
    save('real_data_verification.json',result)
    save('pytest_results.json',{'counts':counts,'test_cases':tests,'original_exit_code':0})
    print(json.dumps({k:result[k] for k in ['total_episodes_checked','physical_images_rehashed',
        'original_image_paths_and_hashes_unchanged','original_six_split_files_unchanged','pytest_counts']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
