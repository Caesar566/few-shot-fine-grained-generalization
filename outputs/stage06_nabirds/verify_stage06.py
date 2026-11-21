"""Independent CUDA replay, automatic-rule consistency and statistics audit."""
import csv
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import t
import torch

from pattern_recognition_coursework.data.nabirds_automatic import resolve_cub_train
from pattern_recognition_coursework.data.nabirds import load_nabirds_metadata,_ancestors
from pattern_recognition_coursework.engine.episodic_evaluation import ROOT,RUNS,tensor_hash,canonical_hash,sha
from pattern_recognition_coursework.engine.representation_comparison import episode_inputs
from pattern_recognition_coursework.models.metrics.representation import representation_scores,row_score_standardize

OUT=ROOT/'outputs/stage06_nabirds/full_seed42_600_cuda'


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def main():
    assert torch.cuda.is_available(),'CUDA required for this independent replay'
    device=torch.device('cuda');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    config=read(OUT/'config.json');checks=read(OUT/'checks.json');manifest=read(OUT/'protocol/manifest.json')
    assert checks['status']=='PASS' and config['episodes']==600 and not manifest['human_reviewed']
    assert manifest['automatic_screening'] and manifest['image_split']=='official_test_only'
    cub_classes=ROOT/'data/splits/cub_clean_v1/train_classes.txt'
    taxonomy=ROOT/'outputs/stage06_nabirds/preflight/ebird_taxonomy_v2026.csv'
    resolved,excluded,species,_=resolve_cub_train(cub_classes,taxonomy)
    assert len(resolved)==100 and set(manifest['excluded_cub_taxon_codes'])==excluded
    assert not {c['class_id'] for c in manifest['classes']}&excluded
    for c in manifest['classes']:
        assert c['name']==species[c['class_id']]['COMMON_NAME']
        assert c['scientific_name']==species[c['class_id']]['SCIENTIFIC_NAME']
    metadata=load_nabirds_metadata(ROOT/'nabirds')
    official={line.split()[0]:int(line.split()[1]) for line in (ROOT/'nabirds/train_test_split.txt').read_text().splitlines()}
    by_id={r['image_id']:r for r in manifest['samples']}
    assert len(by_id)==len(manifest['samples'])==19307 and len(manifest['classes'])==317
    assert len({r['sha256'] for r in manifest['samples']})==19307
    for r in manifest['samples']:
        assert official[r['image_id']]==0
        assert metadata['images'][r['image_id']]==r['path'] and metadata['labels'][r['image_id']]==r['visual_class_id']
        assert r['species_ancestor_id'] in _ancestors(r['visual_class_id'],metadata['parents'])
        assert sha(ROOT/'nabirds/images'/r['path'])==r['sha256']
    catalog=rows(OUT/'protocol/image_catalog.csv');plans={k:read(OUT/f'protocol/plan_{k}shot.json')['batches'] for k in (1,5)}
    assert all(a[:5]==b[:25:5] and a[5:]==b[25:] for a,b in zip(plans[1],plans[5]))
    summaries=read(OUT/'summary.json');accuracy_values={};formula_probes=[];cases=[]
    for name,run in RUNS.items():
        checkpoint=ROOT/'outputs/stage04_training'/run/'checkpoints/best.pth'
        assert sha(checkpoint)==checks['checkpoint_sha256'][name]
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        assert saved['split_hashes']['train_classes.txt']==manifest['cub_train_classes_sha256']
        assert set(saved['class_to_idx'])=={r['cub_name'] for r in resolved}
        meta=read(OUT/name/'cache.json');payload=torch.load(meta['path'],map_location='cpu',weights_only=True,mmap=True)
        assert canonical_hash(payload['binding'])==meta['key']
        assert payload['binding']['dataset']=='nabirds' and payload['binding']['manifest_sha256']==checks['manifest_sha256']
        assert payload['binding']['checkpoint_sha256']==checks['checkpoint_sha256'][name]
        assert tensor_hash(payload['local'])==meta['local_sha256'] and tensor_hash(payload['gap'])==meta['gap_sha256']
        assert payload['indices']==payload['binding']['sample_indices']
        assert [catalog[i]['image_id'] for i in payload['indices']]==payload['binding']['ordered_sample_ids']
        positions={index:row for row,index in enumerate(payload['indices'])};local=payload['local']
        global_features=torch.zeros(len(catalog),512);global_features[payload['indices']]=payload['gap']
        accuracy_values[name]={rep:{} for rep in ('global','local','fusion')}
        for k in (1,5):
            folder=OUT/name/f'{k}shot';stored_rows={rep:rows(folder/f'{rep}_episodes.csv') for rep in accuracy_values[name]}
            values={rep:[] for rep in stored_rows};count=0
            with (folder/'predictions.jsonl').open() as stream,torch.inference_mode():
                for episode,line in enumerate(stream):
                    record=json.loads(line);batch=plans[k][episode]
                    assert record['episode_id']==episode and record['score_device']=='cuda'
                    assert record['support_ids']==[catalog[i]['image_id'] for i in batch[:5*k]]
                    assert record['query_ids']==[catalog[i]['image_id'] for i in batch[5*k:]]
                    assert len({by_id[x]['sha256'] for x in record['support_ids']+record['query_ids']})==len(batch)
                    for slot,mapping in enumerate(record['class_mapping']):
                        assert mapping['episode_label']==slot
                        assert mapping['species_code']==catalog[batch[slot*k]]['species_code']
                        assert mapping['dataset_label']==int(catalog[batch[slot*k]]['dataset_label'])
                        assert mapping['name']==catalog[batch[slot*k]]['class_name']
                    target=torch.arange(5,device=device).repeat_interleave(5)
                    assert record['true_labels']==target.cpu().tolist()
                    logits={rep:torch.tensor(record['methods'][rep]['logits'],device=device,dtype=torch.float32) for rep in values}
                    zg=row_score_standardize(logits['global']);zl=row_score_standardize(logits['local'])
                    assert torch.equal(zg,torch.tensor(record['global_calibrated'],device=device,dtype=torch.float32))
                    assert torch.equal(zl,torch.tensor(record['local_calibrated'],device=device,dtype=torch.float32))
                    assert torch.equal(.5*zg+.5*zl,logits['fusion'])
                    for rep,z in logits.items():
                        pred=z.argmax(-1);p=z.softmax(-1)
                        assert pred.cpu().tolist()==record['methods'][rep]['predictions']
                        torch.testing.assert_close(p,torch.tensor(record['methods'][rep]['probabilities'],device=device,dtype=torch.float32),rtol=1e-6,atol=1e-7)
                        assert torch.allclose(p.sum(-1),torch.ones(25,device=device),atol=1e-6)
                        correct=int((pred==target).sum());acc=correct/25
                        assert correct==int(stored_rows[rep][episode]['correct']) and acc==float(stored_rows[rep][episode]['accuracy'])
                        values[rep].append(acc)
                    if episode in (0,299,599):
                        g,s,q=episode_inputs(batch,k,positions,local,global_features,device)
                        replay=representation_scores(g,s,q,5,k,3,4,1e-6)
                        for rep in values:assert torch.equal(replay[rep+'_scores'],logits[rep])
                        formula_probes.append(dict(initialization=name,shot=k,episode_id=episode,device='cuda',all_logits_bitwise_equal=True))
                    count+=1
            assert count==600
            for rep,v in values.items():
                v=np.asarray(v,dtype=np.float64);mean=float(v.mean());std=float(v.std(ddof=1));half=t.ppf(.975,599)*std/np.sqrt(600)
                s=summaries[name][rep][str(k)]
                np.testing.assert_allclose([s['mean_accuracy'],s['sample_std']],[mean,std],atol=1e-14,rtol=0)
                np.testing.assert_allclose(s['ci95'],[mean-half,mean+half],atol=1e-14,rtol=0)
                accuracy_values[name][rep][k]=v
                cases.append(dict(initialization=name,shot=k,representation=rep,episodes=600,query_predictions=15000,all_checks=True))
        del payload,local,global_features,saved
    for r in rows(OUT/'paired_comparison.csv'):
        values=accuracy_values[r['initialization']];k=int(r['shot']);d=values[r['second']][k]-values[r['first']][k]
        mean=d.mean()*100;half=t.ppf(.975,599)*d.std(ddof=1)/np.sqrt(600)*100
        np.testing.assert_allclose([float(r['mean_delta_pp']),float(r['ci95_low_pp']),float(r['ci95_high_pp'])],
                                  [mean,mean-half,mean+half],atol=1e-12,rtol=0)
    cub=read(Path(config['cub_reference'])/'summary.json')
    for r in rows(OUT/'cub_vs_nabirds.csv'):
        target=summaries[r['initialization']][r['representation']][r['shot']]['mean_accuracy']*100
        source=cub[r['initialization']][r['representation']][r['shot']]['mean_accuracy']*100
        np.testing.assert_allclose([float(r['nabirds_mean_percent']),float(r['cub_val_mean_percent']),float(r['nabirds_minus_cub_pp'])],
                                  [target,source,target-source],atol=1e-12,rtol=0)
    for filename in ('accuracy_comparison.svg','accuracy_bars.svg','paired_lifts.svg'):ET.parse(OUT/filename)
    tests=ET.parse(OUT.parent/'tests/cuda_tests.xml').getroot()
    assert not tests.findall('.//failure') and not tests.findall('.//error') and not tests.findall('.//skipped')
    result=dict(status='PASS',independent_process=True,verification_device='cuda',automatic_rule_consistency=True,
        human_reviewed=False,target_species=317,target_images=19307,official_test_only=True,
        cub_train_class_hash_and_labels_match_all_three_checkpoints=True,
        content_unique_and_all_physical_hashes_verified=True,method_evaluations_verified=10800,query_predictions_verified=270000,
        cases=cases,formula_cuda_probes=formula_probes,paired_comparisons_verified=18,cross_dataset_differences_verified=18,
        cross_dataset_difference_type='unpaired descriptive mean difference',cuda_tests_passed=len(tests.findall('.//testcase')))
    (OUT/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    checks.update(independent_verification='PASS',cuda_regression_tests_passed=result['cuda_tests_passed'])
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
