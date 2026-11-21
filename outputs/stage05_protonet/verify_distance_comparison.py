"""Independent covariance/formula replay of the distance-only experiment."""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import t
import torch

from pattern_recognition_coursework.engine.distance_comparison import (
    FixedMetricScorer,folder_hashes,
)
from pattern_recognition_coursework.engine.episodic_evaluation import (
    cache_load,canonical_hash,sha,tensor_hash,
)


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:return list(csv.DictReader(stream))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();out=args.run_dir.resolve();config=read(out/'config.json')
    base=Path(config['baseline_dir']);bindings=read(out/'feature_bindings.json')
    torch.set_num_threads(config['environment']['threads'])
    plans={k:read(out/f'plan_{k}shot.json') for k in (1,5)}
    for name in ['plan_1shot.json','plan_5shot.json','paired_episodes.jsonl','image_catalog.csv']:
        assert (out/name).read_bytes()==(base/name).read_bytes()
    paired=[json.loads(line) for line in (out/'paired_episodes.jsonl').read_text().splitlines()]
    labels=np.array([int(row['dataset_label']) for row in rows(out/'train_image_catalog.csv')])
    summaries=read(out/'summary.json');records=[];all_accuracies={};covariance_checks=[]
    for name in ('imagenet','random','simclr'):
        binding=bindings[name];oldmeta=read(base/name/'cache.json')
        features=cache_load(binding['val_cache_path'],oldmeta['binding'],2947)
        assert tensor_hash(features)==binding['val_feature_sha256']
        assert sha(binding['checkpoint'])==binding['checkpoint_sha256']
        trainmeta=read(out/name/'train_cache.json')
        train=cache_load(trainmeta['path'],trainmeta['binding'],5890).double().numpy()
        covariance=read(out/name/'covariance.json')
        # Independent NumPy fit from raw TRAIN features and TRAIN labels only.
        residual_sum=np.zeros(512,dtype=np.float64)
        for label in np.unique(labels):
            members=train[labels==label]
            residual_sum+=np.square(members-members.mean(0)).sum(0)
        raw=residual_sum/(len(train)-len(np.unique(labels)))
        np.testing.assert_allclose(raw,covariance['raw_within_class_variance'],rtol=1e-12,atol=1e-12)
        jitter=max(config['absolute_jitter'],config['relative_jitter']*raw.mean())
        diagonal=(1-config['shrinkage'])*raw+config['shrinkage']*raw.mean()+jitter
        np.testing.assert_allclose(diagonal,covariance['diagonal'],rtol=1e-12,atol=1e-12)
        assert (diagonal>0).all() and not covariance['used_query_features'] and not covariance['used_query_labels']
        covariance_checks.append(dict(initialization=name,independent_numpy_within_class_fit=True,
            degrees_of_freedom=5790,positive_diagonal=True,
            maximum_diagonal_absolute_difference=float(np.abs(diagonal-np.array(covariance['diagonal'])).max())))
        for k in (1,5):
            prototype_hashes=[]
            for batch in plans[k]['batches']:
                prototype_hashes.append(tensor_hash(features[batch[:5*k]].reshape(5,k,512).mean(1)))
            assert canonical_hash(prototype_hashes)==binding['raw_prototype_hashes'][str(k)]
        all_accuracies[name]={metric:{} for metric in ('euclidean','cosine','mahalanobis')}
        for k in (1,5):
            all_accuracies[name]['euclidean'][k]=np.array([float(r['accuracy']) for r in rows(base/name/f'{k}shot/episodes.csv')])
        for metric in ('cosine','mahalanobis'):
            scorer=FixedMetricScorer(metric,torch.tensor(covariance['diagonal'],dtype=torch.float64))
            for k in (1,5):
                case=out/name/metric/f'{k}shot';episode_rows=rows(case/'episodes.csv')
                accuracies=[];error_max=0.;predictions_checked=0
                with (case/'predictions.jsonl').open() as stream,torch.inference_mode():
                    for episode,line in enumerate(stream):
                        saved=json.loads(line);batch=plans[k]['batches'][episode];z=features[batch]
                        logits=scorer.classify_embeddings(z,5,k,5)
                        stored=torch.tensor([q['logits'] for q in saved['queries']],dtype=torch.float32)
                        assert torch.equal(logits,stored)
                        probability=logits.softmax(-1)
                        assert torch.equal(probability,torch.tensor([q['probabilities'] for q in saved['queries']],dtype=torch.float32))
                        prototype=z[:5*k].reshape(5,k,512).mean(1).double().numpy()
                        query=z[5*k:].double().numpy()
                        if metric=='cosine':
                            reference=query@prototype.T
                            reference/=np.maximum(np.linalg.norm(query,axis=1),1e-12)[:,None]
                            reference/=np.maximum(np.linalg.norm(prototype,axis=1),1e-12)[None,:]
                        else:
                            reference=-np.sum(np.square(query[:,None,:]-prototype[None,:,:])/diagonal,axis=-1)/512
                        np.testing.assert_allclose(stored.numpy(),reference,rtol=1e-5,atol=1e-6)
                        error_max=max(error_max,float(np.abs(stored.numpy()-reference).max()))
                        correct=0
                        for i,q in enumerate(saved['queries']):
                            member=paired[episode]['query'][i];true=i//5;predicted=int(logits[i].argmax())
                            assert q['image_id']==member['image_id'] and q['dataset_index']==member['dataset_index']
                            assert q['episode_label']==true and q['predicted_label']==predicted
                            assert q['class_name']==paired[episode]['class_mapping'][true]['class_name']
                            assert q['predicted_class']==paired[episode]['class_mapping'][predicted]['class_name']
                            assert q['correct']==(true==predicted)
                            correct+=int(q['correct']);predictions_checked+=1
                        assert correct==int(episode_rows[episode]['correct'])
                        assert float(episode_rows[episode]['accuracy'])==correct/25
                        accuracies.append(correct/25)
                assert len(accuracies)==600
                values=np.array(accuracies);mean=values.mean();std=values.std(ddof=1)
                half=t.ppf(.975,599)*std/np.sqrt(600);summary=summaries[name][metric][str(k)]
                np.testing.assert_allclose([summary['mean_accuracy'],summary['sample_std']], [mean,std],rtol=0,atol=1e-14)
                np.testing.assert_allclose(summary['ci95'],[mean-half,mean+half],rtol=0,atol=1e-14)
                all_accuracies[name][metric][k]=values
                records.append(dict(initialization=name,metric=metric,shot=k,episodes=600,
                    query_predictions=predictions_checked,logits_and_probabilities_bitwise_reproduced=True,
                    raw_prototype_means_unchanged=True,independent_formula_max_absolute_difference=error_max,
                    all_ids_labels_predictions_and_statistics_match=True))
        del features,train
    for row in rows(out/'paired_comparison.csv'):
        name=row['initialization'];k=int(row['shot'])
        d=all_accuracies[name][row['second']][k]-all_accuracies[name][row['first']][k]
        mean=d.mean()*100;half=t.ppf(.975,599)*d.std(ddof=1)/np.sqrt(600)*100
        np.testing.assert_allclose([float(row['mean_delta_pp']),float(row['ci95_low_pp']),float(row['ci95_high_pp'])],
                                  [mean,mean-half,mean+half],rtol=0,atol=1e-12)
    assert folder_hashes(base)==read(out/'baseline_file_hashes.json')
    for name in ['accuracy_comparison.svg','paired_differences.svg']:ET.parse(out/name)
    tests=ET.parse(out.parent/'distance_comparison_tests/junit.xml').getroot()
    assert not tests.findall('.//failure') and not tests.findall('.//error')
    result=dict(status='PASS',independent_process=True,new_episodes_verified=7200,new_query_predictions_verified=180000,
                reused_baseline_episodes=3600,original_euclidean_files_unchanged=True,
                covariance_checks=covariance_checks,configurations=records,paired_comparisons_verified=18,
                tests_passed=len(tests.findall('.//testcase')))
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    checks=read(out/'checks.json');checks.update(independent_replay='PASS',tests_passed=result['tests_passed'])
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    report=out/'report.md';text=report.read_text();marker='\n## 独立进程核验\n'
    if marker in text:text=text.split(marker)[0]
    text+=marker+'\n独立进程已复算三个Train类内协方差、全部7200个新Episode的公式与180000条预测、18组配对统计；logits和概率逐位复现，原始均值原型哈希不变，旧欧氏目录全部文件哈希不变。'
    text+=f' {result["tests_passed"]}项测试通过。[核验证据](independent_verification.json) · [测试日志](../distance_comparison_tests/pytest.log)。\n'
    report.write_text(text)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
