"""Independent CUDA replay of saved full representation results and statistics."""
import argparse
import csv
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import t
import torch

from pattern_recognition_coursework.engine.episodic_evaluation import cache_load,sha,tensor_hash,canonical_hash
from pattern_recognition_coursework.engine.representation_comparison import episode_inputs
from pattern_recognition_coursework.models.metrics.representation import representation_scores,row_score_standardize


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    args=parser.parse_args();out=args.run_dir.resolve();config=read(out/'config.json');checks=read(out/'checks.json')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable for independent replay')
    device=torch.device(args.device);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert checks['status']=='PASS' and config['mode']=='full' and config['episodes']==600
    source=Path(config['source_dir']);baseline=Path(config['global_baseline_dir'])
    for filename in ['plan_1shot.json','plan_5shot.json','paired_episodes.jsonl','image_catalog.csv']:
        assert (out/filename).read_bytes()==(source/filename).read_bytes()==(baseline/filename).read_bytes()
    plans={k:read(out/f'plan_{k}shot.json')['batches'] for k in (1,5)}
    assert all(a[:5]==b[:25:5] and a[5:]==b[25:] for a,b in zip(plans[1],plans[5]))
    catalog=rows(out/'image_catalog.csv');summaries=read(out/'summary.json');all_accuracies={};evidence=[];formula_replays=[]
    for name in ('imagenet','random','simclr'):
        meta=read(out/name/'cache.json');path=Path(meta['path'])
        payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        assert canonical_hash(payload['binding'])==meta['key']
        assert tensor_hash(payload['local'])==meta['local_sha256']
        assert payload['indices']==meta['binding']['sample_indices']
        assert [catalog[i]['image_id'] for i in payload['indices']]==meta['binding']['ordered_sample_ids']
        assert payload['binding']['checkpoint_sha256']==checks['checkpoint_sha256'][name]
        run=read(source/'checkpoints.json') if (source/'checkpoints.json').exists() else None
        oldmeta=read(source/name/'cache.json')
        assert oldmeta['binding']['checkpoint_sha256']==checks['checkpoint_sha256'][name]
        global_features=cache_load(oldmeta['path'],oldmeta['binding'],2947)
        local=payload['local'];positions={index:row for row,index in enumerate(payload['indices'])}
        all_accuracies[name]={rep:{} for rep in ('global','local','fusion')}
        for k in (1,5):
            directory=out/name/f'{k}shot'
            accuracy_rows={rep:rows(directory/f'{rep}_episodes.csv') for rep in ('global','local','fusion')}
            accuracies={rep:[] for rep in accuracy_rows};count=0
            with (directory/'predictions.jsonl').open() as stream,torch.inference_mode():
                for eid,line in enumerate(stream):
                    record=json.loads(line);batch=plans[k][eid]
                    assert record['episode_id']==eid and record['score_device']==checks['device']
                    assert record['support_ids']==[catalog[i]['image_id'] for i in batch[:5*k]]
                    assert record['query_ids']==[catalog[i]['image_id'] for i in batch[5*k:]]
                    assert len(set(record['support_ids']+record['query_ids']))==len(batch)
                    expected_names=[catalog[batch[i*k]]['class_name'] for i in range(5)]
                    assert record['class_mapping']==expected_names
                    target=torch.arange(5,device=device).repeat_interleave(5)
                    assert record['true_labels']==target.cpu().tolist()
                    tensors={rep:torch.tensor(record['methods'][rep]['logits'],dtype=torch.float32,device=device) for rep in accuracy_rows}
                    zg=row_score_standardize(tensors['global']);zl=row_score_standardize(tensors['local'])
                    assert torch.equal(zg,torch.tensor(record['global_calibrated'],dtype=torch.float32,device=device))
                    assert torch.equal(zl,torch.tensor(record['local_calibrated'],dtype=torch.float32,device=device))
                    assert torch.equal(.5*zg+.5*zl,tensors['fusion'])
                    for rep,logits in tensors.items():
                        predictions=logits.argmax(-1);probabilities=logits.softmax(-1)
                        assert predictions.cpu().tolist()==record['methods'][rep]['predictions']
                        torch.testing.assert_close(probabilities,torch.tensor(record['methods'][rep]['probabilities'],dtype=torch.float32,device=device),rtol=1e-6,atol=1e-7)
                        assert torch.allclose(probabilities.sum(-1),torch.ones(25,device=device),atol=1e-6)
                        correct=int((predictions==target).sum());accuracy=correct/25
                        assert correct==int(accuracy_rows[rep][eid]['correct'])
                        assert accuracy==float(accuracy_rows[rep][eid]['accuracy'])
                        accuracies[rep].append(accuracy)
                    if eid in (0,299,599):
                        g,s,q=episode_inputs(batch,k,positions,local,global_features,device)
                        replay=representation_scores(g,s,q,5,k,3,4,1e-6)
                        differences={}
                        for rep in tensors:
                            z=replay[f'{rep}_scores'];differences[rep]=float((z-tensors[rep]).abs().max())
                            assert torch.equal(z,tensors[rep])
                        formula_replays.append(dict(initialization=name,shot=k,episode_id=eid,device=str(device),max_abs_differences=differences))
                    count+=1
            assert count==600
            old=rows(baseline/name/'cosine'/f'{k}shot/episodes.csv')
            assert accuracies['global']==[float(row['accuracy']) for row in old]
            for rep in accuracy_rows:
                values=np.asarray(accuracies[rep],dtype=np.float64);mean=float(values.mean());std=float(values.std(ddof=1))
                half=float(t.ppf(.975,599)*std/np.sqrt(600));summary=summaries[name][rep][str(k)]
                np.testing.assert_allclose([summary['mean_accuracy'],summary['sample_std']],[mean,std],rtol=0,atol=1e-14)
                np.testing.assert_allclose(summary['ci95'],[mean-half,mean+half],rtol=0,atol=1e-14)
                all_accuracies[name][rep][k]=values
                evidence.append(dict(initialization=name,shot=k,representation=rep,episodes=count,predictions=15000,
                                     ids_mapping_predictions_probabilities_statistics_valid=True))
        del payload,local,global_features
    pair_rows=rows(out/'paired_comparison.csv');assert len(pair_rows)==18
    for row in pair_rows:
        values=all_accuracies[row['initialization']];k=int(row['shot'])
        d=values[row['second']][k]-values[row['first']][k]
        mean=float(d.mean())*100;half=float(t.ppf(.975,599)*d.std(ddof=1)/np.sqrt(600))*100
        np.testing.assert_allclose([float(row['mean_delta_pp']),float(row['ci95_low_pp']),float(row['ci95_high_pp'])],
                                   [mean,mean-half,mean+half],rtol=0,atol=1e-12)
    for filename in ['accuracy_comparison.svg','accuracy_bars.svg','paired_lifts.svg']:ET.parse(out/filename)
    result=dict(status='PASS',independent_process=True,verification_device=str(device),
                configurations=evidence,method_evaluations_verified=10800,query_predictions_verified=270000,
                all_fusion_calibration_replayed=True,paired_comparisons_verified=18,
                exact_global_baseline_accuracy_reused=True,raw_formula_cuda_probes=formula_replays)
    write=lambda path,value:Path(path).write_text(json.dumps(value,indent=2)+'\n')
    write(out/'independent_verification.json',result)
    checks.update(independent_prediction_and_statistics_verification='PASS',independent_cuda_formula_probes=18)
    write(out/'checks.json',checks)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
