"""Independent disk replay: verify every saved query and recompute all statistics."""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.stats import t
import torch

from pattern_recognition_coursework.engine.episodic_evaluation import cache_load, sha
from pattern_recognition_coursework.models.metrics.episode import prototype_scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args();out=args.run_dir.resolve()
    torch.set_num_threads(2)
    config=json.loads((out/'config.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    plans={k:json.loads((out/f'plan_{k}shot.json').read_text()) for k in (1,5)}
    paired=[json.loads(line) for line in (out/'paired_episodes.jsonl').read_text().splitlines()]
    assert len(paired)==config['episodes']
    evidence=[]
    for name in ('imagenet','random','simclr'):
        cache=json.loads((out/name/'cache.json').read_text())
        assert sha(json.loads((out/'checkpoints.json').read_text())[name]['path'])==cache['binding']['checkpoint_sha256']
        features=cache_load(cache['path'],cache['binding'],2947)
        for k in (1,5):
            with (out/name/f'{k}shot/episodes.csv').open() as stream:epoch_rows=list(csv.DictReader(stream))
            accuracies=[];queries=0
            with (out/name/f'{k}shot/predictions.jsonl').open() as stream:
                for eid,line in enumerate(stream):
                    record=json.loads(line);batch=plans[k]['batches'][eid]
                    assert record['episode_id']==eid and len(record['queries'])==25
                    membership=paired[eid]
                    support=membership[f'support_{k}shot'];query_members=membership['query']
                    assert not {r['sha256'] for r in support}&{r['sha256'] for r in query_members}
                    assert [r['dataset_index'] for r in support]==batch[:5*k]
                    assert [r['dataset_index'] for r in query_members]==batch[5*k:]
                    z=features[batch]
                    # Directly reuse the fixed metric, without instantiating another Backbone.
                    logits=prototype_scores(z[:5*k],z[5*k:],5,k,'euclidean')
                    stored=torch.tensor([q['logits'] for q in record['queries']],dtype=torch.float32)
                    assert torch.equal(logits,stored)
                    probability=logits.softmax(-1)
                    stored_probability=torch.tensor([q['probabilities'] for q in record['queries']],dtype=torch.float32)
                    assert torch.equal(probability,stored_probability)
                    assert torch.allclose(probability.sum(-1),torch.ones(25),atol=1e-6)
                    predictions=logits.argmax(-1).tolist();correct=0
                    for i,q in enumerate(record['queries']):
                        true=i//5
                        assert q['episode_label']==true and q['image_id']==query_members[i]['image_id']
                        assert q['dataset_index']==query_members[i]['dataset_index']
                        assert q['class_name']==membership['class_mapping'][true]['class_name']
                        assert q['predicted_label']==predictions[i]
                        assert q['predicted_class']==membership['class_mapping'][predictions[i]]['class_name']
                        assert q['correct']==(true==predictions[i])
                        correct+=int(q['correct']);queries+=1
                    assert correct==int(epoch_rows[eid]['correct'])
                    assert abs(correct/25-float(epoch_rows[eid]['accuracy']))<1e-14
                    accuracies.append(correct/25)
            assert len(accuracies)==config['episodes']
            values=np.asarray(accuracies);mean=float(values.mean());std=float(values.std(ddof=1))
            half=float(t.ppf(.975,len(values)-1)*std/np.sqrt(len(values)))
            s=summary[name][str(k)]
            assert abs(s['mean_accuracy']-mean)<1e-14 and abs(s['sample_std']-std)<1e-14
            np.testing.assert_allclose(s['ci95'],[mean-half,mean+half],rtol=0,atol=1e-14)
            evidence.append(dict(initialization=name,shot=k,episodes=len(accuracies),queries=queries,
                                 all_logits_bitwise_equal=True,all_probabilities_bitwise_equal=True,
                                 all_labels_ids_class_names_predictions_match=True,
                                 mean_std_and_student_t_ci_recomputed=True))
    ET.parse(out/'accuracy_distribution.svg')
    tests=ET.parse(out.parent/'batch_evaluation_tests/junit.xml').getroot()
    assert not tests.findall('.//failure') and not tests.findall('.//error')
    result=dict(status='PASS',independent_process=True,configurations=evidence,
                total_query_predictions=sum(r['queries'] for r in evidence),
                automated_tests=len(tests.findall('.//testcase')),svg_xml_valid=True)
    (out/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    checks=json.loads((out/'checks.json').read_text())
    checks.update(independent_saved_prediction_replay='PASS',automated_tests=result['automated_tests'])
    (out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    report=out/'report.md';text=report.read_text()
    marker='\n## 独立进程复核\n'
    if marker in text:text=text.split(marker)[0]
    total_episodes=sum(r['episodes'] for r in evidence)
    text+=marker+f'\n已在独立进程从磁盘读取缓存和完整预测，对全部{total_episodes}个Episode、{result["total_query_predictions"]}个Query逐项复算：logits与概率逐位相等，ID/标签/类别名称和预测一致，均值、ddof=1标准差与Student-t CI独立重算一致。'
    text+=f'[完整证据](independent_verification.json)。{result["automated_tests"]}项回归测试通过：[测试日志](../batch_evaluation_tests/pytest.log)。\n'
    report.write_text(text)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
