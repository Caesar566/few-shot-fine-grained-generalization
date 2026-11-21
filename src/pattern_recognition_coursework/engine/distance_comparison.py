"""Distance-only comparison on exact saved CUB Val episodes and raw features.

Euclidean results are referenced, never rescored or overwritten. Covariance is
estimated exclusively from CUB TRAIN within-class features, without backprop.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import itertools
import json
import logging
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from pattern_recognition_coursework.analysis.episodic_statistics import accuracy_summary, paired_accuracy_difference
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset,get_cub_transforms
from pattern_recognition_coursework.data.episodes import dataset_fingerprint,plan_hash
from pattern_recognition_coursework.data.paired_episodes import validate_paired_plans
from pattern_recognition_coursework.engine.episodic_evaluation import (
    ROOT,RUNS,audit_dataset,cache_load,cached_features,canonical_hash,
    evaluate_cached,sha,state_hash,tensor_hash,write_json,
)
from pattern_recognition_coursework.engine.experiment import set_seed
from pattern_recognition_coursework.models.few_shot import FewShotModel
from pattern_recognition_coursework.models.metrics.episode import prototype_scores
from pattern_recognition_coursework.models.metrics.shared_covariance import (
    pooled_diagonal_covariance,shared_diagonal_mahalanobis_scores,
)

METRICS=('euclidean','cosine','mahalanobis')
COLORS={'euclidean':'#2468b4','cosine':'#16856a','mahalanobis':'#c97312'}


def read(path):return json.loads(Path(path).read_text())


def write_rows(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def folder_hashes(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


class FixedMetricScorer:
    """Only the distance changes; no encoder, normalization or learnable state."""
    def __init__(self,metric,diagonal=None):
        if metric not in ('cosine','mahalanobis'):raise ValueError('Euclidean must be reused, not rescored')
        self.metric=metric;self.diagonal=diagonal

    def classify_embeddings(self,features,n_way,k_shot,n_query):
        if features.ndim!=2 or features.shape!=(n_way*(k_shot+n_query),512):
            raise ValueError('Expected raw 512-d features in existing episode order')
        support=features[:n_way*k_shot];query=features[n_way*k_shot:]
        if self.metric=='cosine':
            # Existing implementation averages RAW support before internal norm division.
            return prototype_scores(support,query,n_way,k_shot,'cosine')
        return shared_diagonal_mahalanobis_scores(support,query,n_way,k_shot,self.diagonal)


def load_baseline_rows(base,name,k,plan,catalog):
    path=base/name/f'{k}shot'
    with (path/'episodes.csv').open() as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=600:raise ValueError('Baseline must contain 600 episodes per setting')
    # Validate the saved predictions and identities without recomputing Euclidean scores.
    with (path/'predictions.jsonl').open() as stream:
        records=0
        for episode,line in enumerate(stream):
            record=json.loads(line);batch=plan['batches'][episode]
            if record['episode_id']!=episode or len(record['queries'])!=25:raise ValueError('Invalid baseline predictions')
            correct=0
            for i,q in enumerate(record['queries']):
                member=catalog[batch[5*k+i]];true=i//5
                if q['image_id']!=member['image_id'] or q['dataset_index']!=member['dataset_index'] or q['episode_label']!=true:
                    raise ValueError('Baseline sample identity or label mismatch')
                prediction=int(np.argmax(q['logits']))
                if q['predicted_label']!=prediction or q['correct']!=(prediction==true):
                    raise ValueError('Baseline saved prediction mismatch')
                correct+=int(q['correct'])
            if int(rows[episode]['episode_id'])!=episode or int(rows[episode]['correct'])!=correct:
                raise ValueError('Baseline accuracy rows mismatch')
            if abs(float(rows[episode]['accuracy'])-correct/25)>1e-14:raise ValueError('Baseline accuracy mismatch')
            records+=1
    if records!=600:raise ValueError('Incomplete baseline predictions')
    summary=read(path/'metrics.json');recomputed=accuracy_summary([float(r['accuracy']) for r in rows])
    for key in ('mean_accuracy','sample_std','ci95_half_width'):
        if abs(summary[key]-recomputed[key])>1e-12:raise ValueError('Baseline statistics mismatch')
    return rows,summary


def audit_train(dataset,splits):
    with (splits/'image_hashes.csv').open(encoding='utf-8-sig') as stream:expected={r['image_id']:r for r in csv.DictReader(stream)}
    rows=[line.split(maxsplit=2) for line in dataset.split_file.read_text().splitlines() if line.strip()]
    catalog=[]
    for index,((path,label),(image_id,name,relative)) in enumerate(zip(dataset.samples,rows)):
        digest=sha(path);ref=expected[image_id]
        if ref['split']!='train' or digest!=ref['sha256'] or ref['relative_path']!=relative:
            raise ValueError(f'Train content differs from clean manifest: {path}')
        catalog.append(dict(dataset_index=index,image_id=image_id,dataset_label=label,
                            class_name=name,relative_path=relative,sha256=digest))
    if len(dataset)!=5890 or len(dataset.classes)!=100 or len({r['sha256'] for r in catalog})!=5890:
        raise ValueError('Expected 5890 unique clean CUB Train images and 100 classes')
    return catalog,dict(images=len(dataset),classes=len(dataset.classes),
                        dataset_fingerprint=dataset_fingerprint(dataset),
                        ordered_content_sha256=canonical_hash(catalog),
                        manifest_sha256=sha(dataset.split_file),all_actual_train_bytes_verified=True)


def resolve_val_cache(metadata,directory):
    # Prefer the explicitly mapped cache directory for portable server transfers.
    candidates=[directory/f"{metadata['key']}.pt"]
    if metadata.get('path'):candidates.append(Path(metadata['path']))
    for path in candidates:
        if path.is_file():return path
    raise FileNotFoundError('Exact original Val feature cache missing. Transfer the three original .pt files '
                            'and pass --baseline-cache-dir; do not silently re-extract or normalize features.')


def build_comparison(out,results,plans,base):
    table=[];paired=[];deltas=[]
    for name in RUNS:
        for k in (1,5):
            for metric in METRICS:
                summary=results[name][metric][k]['summary']
                table.append(dict(initialization=name,distance=metric,shot=k,episodes=600,
                    mean_accuracy_percent=summary['mean_accuracy']*100,sample_std_pp=summary['sample_std']*100,
                    ci95_low_percent=summary['ci95'][0]*100,ci95_high_percent=summary['ci95'][1]*100,
                    ci95_half_width_pp=summary['ci95_half_width']*100,plan_sha256=plan_hash(plans[k]),
                    result_origin='reused_baseline' if metric=='euclidean' else 'new_evaluation'))
            for a,b in itertools.combinations(METRICS,2):
                first=[float(r['accuracy']) for r in results[name][a][k]['rows']]
                second=[float(r['accuracy']) for r in results[name][b][k]['rows']]
                summary=paired_accuracy_difference(first,second)
                paired.append(dict(initialization=name,shot=k,first=a,second=b,episodes=600,
                    mean_delta_pp=summary['mean_delta']*100,sample_std_delta_pp=summary['sample_std_delta']*100,
                    ci95_low_pp=summary['ci95'][0]*100,ci95_high_pp=summary['ci95'][1]*100))
                deltas.extend(dict(initialization=name,shot=k,first=a,second=b,episode_id=i,
                                   first_accuracy=x,second_accuracy=y,delta_pp=(y-x)*100) for i,(x,y) in enumerate(zip(first,second)))
    write_rows(out/'comparison.csv',table);write_rows(out/'paired_comparison.csv',paired)
    write_rows(out/'paired_episode_differences.csv',deltas)
    write_json(out/'summary.json',{name:{metric:{str(k):results[name][metric][k]['summary'] for k in (1,5)} for metric in METRICS} for name in RUNS})
    plot_comparison(out,table,paired)
    lines=['# Stage 05：仅改变距离的配对对比实验','',
           '**固定阶段4 Backbone、原始Val特征缓存、Support均值原型、600对Episode和数据划分；只改变距离。** '
           '不训练Backbone，不用NABirds。本次新增7200次Episode评分；既有3600次欧氏结果直接复用，总表覆盖10800次评分。','',
           '| 初始化 | 距离 | Shot | Mean Accuracy | 样本标准差（pp） | 95% CI |',
           '|---|---|---:|---:|---:|---|']
    for r in table:
        lines.append(f"| {r['initialization']} | {r['distance']} | {r['shot']} | {r['mean_accuracy_percent']:.3f}% | {r['sample_std_pp']:.3f} | [{r['ci95_low_percent']:.3f}%, {r['ci95_high_percent']:.3f}%] |")
    lines += ['', '## 严格固定的部分','',
              '- 直接读取并逐字节复制原`plan_1shot.json/plan_5shot.json/paired_episodes.jsonl/image_catalog.csv`，没有重新采样。两种Shot类别和Query相同，1-shot Support是5-shot Support子集。',
              '- 三组Val特征直接读取旧缓存，校验完整binding、权重SHA256、编码器state指纹、Transform、输入内容指纹、特征SHA256、shape及dtype，不重新提取或转换Val特征。',
              '- 图像预处理仍是RGB、Resize257、CenterCrop224、ToTensor、ImageNet像素Normalize。像素Normalize与特征L2归一化不同。',
              '- 原始512维特征不做额外L2归一化或标准化。每个原型始终为原始Support特征算术平均，不预先归一化Support，不归一化或重建原型。',
              '- 欧氏结果仅核验已有预测ID/Accuracy/统计并引用原文件；不重新计算其600个Episode，也不改旧目录任何文件。`baseline_references.json`保存原结果位置和SHA256。','',
              '## 三种距离的精确定义','',
              '`p_c = mean_k Support[c,k]`，所有距离使用同一个原始均值原型。','',
              '- **Euclidean**：沿用旧`prototype_scores`的平方欧氏分支，`logits=-sum((q-p)^2)/512`；结果直接复用。',
              '- **Cosine**：复用旧`prototype_scores(..., metric="cosine")`。该函数先求原始均值原型，然后在距离公式内部对q和p除以各自范数，eps=1e-12。`distance=1-cos(q,p)`，实际`logits=cos(q,p)`，与负余弦距离仅差所有类别共享常数1，Softmax与argmax相同。没有把归一化后的Support均值作为原型。',
              '- **Mahalanobis**：新增全局共享对角版本。使用原始q、p，`D=sum_d((q_d-p_cd)^2/v_d)`，实际`logits=-D/512`。未开平方；以逐元素除法计算，禁止矩阵求逆。原型均值保持float32，协方差估计与距离累加用float64提高稳定性，最终logits转回float32。',
              '现有Episode-Support收缩马氏分支保持原样，本次没有使用它，也没有将其结果混入共享Train协方差实验。所有温度均固定为1，没有调温或额外归一化消融。不同距离的NLL/概率尺度不同，本实验主要比较Accuracy。','',
              '## Train类内共享对角协方差','',
              '每个Backbone分别对5890张/100类CUB Train图片使用同一确定性eval预处理提取原始特征，eval模式、冻结参数、无优化器。只把Train标签传入协方差估计器，不传Val/Query特征或标签。','',
              '1. 计算每个Train类别的均值mu_c。',
              '2. 计算类内残差x_i-mu_yi，池化方差`S_d=sum_i residual_id^2/(N-C)`；实际自由度为5890-100=5790，不包含类间均值差异。',
              '3. `scale=mean_d(S_d)`；`v_d=(1-alpha)*S_d+alpha*scale+jitter`。',
              '4. 默认固定`alpha=0.1`、`jitter=max(1e-8,1e-4*scale)`，没有根据Val或NABirds调参。同一Backbone的v同时用于1-shot和5-shot所有Episode。',
              '完整原始方差、正则化对角线、输入Train特征哈希、样本清单、类计数、正则化数值及条件数见各模型`covariance.json`和`covariance_diagonal.csv`。全局协方差是本次马氏度量需要估计的统计量，不是Backbone训练或额外特征变换。','',
              '## 配对准确率差值','',
              '同一初始化、同一Shot、同一个episode_id进行距离配对。`delta_e=Accuracy_second,e-Accuracy_first,e`，报告均值差（百分点）和对600个delta的Student-t 95% CI；不是将两个独立CI相减。统计为描述性，未进行多重比较校正。','',
              '| 初始化 | Shot | second − first | 平均差（pp） | 配对95% CI（pp） |',
              '|---|---:|---|---:|---|']
    for r in paired:
        lines.append(f"| {r['initialization']} | {r['shot']} | {r['second']} − {r['first']} | {r['mean_delta_pp']:+.3f} | [{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}] |")
    lines += ['', '## 图表与输出','',
              '![距离均值与95% CI](accuracy_comparison.svg)','',
              '![配对准确率差与95% CI](paired_differences.svg)','',
              '[完整表](comparison.csv) · [配对表](paired_comparison.csv) · [逐Episode差值](paired_episode_differences.csv) · [配置](config.json) · [验收](checks.json) · [日志](run.log)','',
              '新增距离的每个模型/Shot均保存`episodes.csv`、`metrics.json`、`predictions.jsonl`，预测包含样本ID、原始与局部标签、真实/预测类别、全部5个logits和概率。Support信息通过episode_id连接本目录复制的`paired_episodes.jsonl`。欧氏完整预测保留在旧目录，通过`baseline_references.json`引用，不复制或覆盖其日志。','',
              '**本次仍为CUB Val验证集评估，三个Checkpoint曾用此Val选择最佳轮次，不应称为独立最终测试或NABirds跨数据集成绩。CI单位为Episode，不包含训练seed或模型选择不确定性。**','',
              '## 运行命令','', '```bash',
              'uv run python -m pattern_recognition_coursework.engine.distance_comparison \\',
              '  --baseline-dir outputs/stage05_protonet/episodic_eval_seed42_600 \\',
              '  --baseline-cache-dir data/cache/stage05_features \\',
              '  --output-dir outputs/stage05_protonet/distance_comparison_server \\',
              '  --device cuda --batch-size 64 --workers 4', '```','',
              '输出目录必须不存在。服务器需同步三份旧Val缓存.pt（约17.3 MiB），不能只同步Git中的报告；缺少旧缓存时明确报错，不悄悄重新提取或改变特征。`--cache-dir`控制新增Train特征缓存位置；本地CPU使用`--device cpu --workers 0`。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def plot_comparison(out,table,paired):
    def frame(height,title):
        return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" viewBox="0 0 1120 {height}">',
                f'<rect width="1120" height="{height}" fill="white"/><g font-family="sans-serif" fill="#222">',
                f'<text x="30" y="32" font-size="21">{title}</text>']
    svg=frame(790,'Distance comparison: mean episode accuracy with 95% Student-t CI')
    for panel,k in enumerate((1,5)):
        top=95+panel*345
        svg.append(f'<text x="30" y="{top}" font-size="18">5-way {k}-shot</text>')
        for tick in range(0,101,20):
            x=330+tick*7
            svg.extend([f'<line x1="{x}" y1="{top+20}" x2="{x}" y2="{top+260}" stroke="#ddd"/>',
                        f'<text x="{x-12}" y="{top+286}" font-size="12">{tick}%</text>'])
        for i,r in enumerate(row for row in table if row['shot']==k):
            y=top+35+i*27;x=330+r['mean_accuracy_percent']*7
            lo=330+r['ci95_low_percent']*7;hi=330+r['ci95_high_percent']*7;color=COLORS[r['distance']]
            svg.extend([f'<text x="35" y="{y+4}" font-size="13">{r["initialization"]} / {r["distance"]}</text>',
                        f'<line x1="{lo}" y1="{y}" x2="{hi}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                        f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>',
                        f'<text x="1045" y="{y+4}" font-size="12">{r["mean_accuracy_percent"]:.2f}%</text>'])
    svg.append('</g></svg>');(out/'accuracy_comparison.svg').write_text('\n'.join(svg))
    svg=frame(870,'Paired episode accuracy difference (second minus first), 95% CI')
    limit=max(1,max(abs(r[key]) for r in paired for key in ('ci95_low_pp','ci95_high_pp')))*1.15
    def xp(value):return 715+value/limit*320
    svg.append('<line x1="715" y1="65" x2="715" y2="790" stroke="#888" stroke-dasharray="5,4"/>')
    for tick in (-1,-.5,0,.5,1):
        x=xp(tick*limit)
        svg.append(f'<text x="{x-15}" y="825" font-size="12">{tick*limit:+.1f} pp</text>')
    for i,r in enumerate(paired):
        y=85+i*39
        label=f'{r["initialization"]} {r["shot"]}-shot: {r["second"]} - {r["first"]}'
        svg.extend([f'<text x="20" y="{y+4}" font-size="12">{label}</text>',
                    f'<line x1="{xp(r["ci95_low_pp"])}" y1="{y}" x2="{xp(r["ci95_high_pp"])}" y2="{y}" stroke="#2468b4" stroke-width="3"/>',
                    f'<circle cx="{xp(r["mean_delta_pp"])}" cy="{y}" r="4" fill="#2468b4"/>',
                    f'<text x="1050" y="{y+4}" font-size="12">{r["mean_delta_pp"]:+.2f}</text>'])
    svg.append('</g></svg>');(out/'paired_differences.svg').write_text('\n'.join(svg))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir',type=Path,default=ROOT/'outputs/stage05_protonet/episodic_eval_seed42_600')
    p.add_argument('--baseline-cache-dir',type=Path,default=ROOT/'data/cache/stage05_features')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT/'cub2002011/versions/7/CUB_200_2011')
    p.add_argument('--splits-dir',type=Path,default=ROOT/'data/splits/cub_clean_v1')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'data/cache/stage05_metric_train_features')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--shrinkage',type=float,default=.1)
    p.add_argument('--relative-jitter',type=float,default=1e-4);p.add_argument('--absolute-jitter',type=float,default=1e-8)
    for name,run in RUNS.items():p.add_argument(f'--{name}-checkpoint',type=Path,default=ROOT/'outputs/stage04_training'/run/'checkpoints/best.pth')
    args=p.parse_args(argv)
    if args.batch_size<1 or args.workers<0 or args.threads<1:p.error('Invalid count')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    # Validate covariance parameters before expensive feature extraction.
    pooled_diagonal_covariance(torch.tensor([[0.,0.],[1.,1.]]),torch.tensor([0,0]),
                               args.shrinkage,args.relative_jitter,args.absolute_jitter)
    base=args.baseline_dir.resolve();out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'outputs/stage05_protonet') or out.is_relative_to(base):p.error('Choose a new sibling output inside outputs/stage05_protonet')
    if out.exists():p.error('Output already exists; refusing overwrite')
    if args.cache_dir.resolve().is_relative_to(base):p.error('Train cache cannot be written into baseline directory')
    oldconfig=read(base/'config.json')
    if oldconfig['episodes']!=600 or oldconfig['n_way']!=5 or oldconfig['n_query']!=5 or oldconfig['metric']!='euclidean':p.error('Expected completed 600-episode Euclidean baseline')
    if oldconfig['temperature']!=1 or oldconfig['image_size']!=224:p.error('Expected original temperature=1 and image_size=224')
    if oldconfig.get('feature_normalization') or oldconfig.get('feature_standardization') or oldconfig.get('adapter'):p.error('Baseline feature processing differs')
    if read(base/'checks.json')['status']!='PASS':p.error('Baseline has not passed verification')
    baseline_before=folder_hashes(base)
    out.mkdir(parents=True)
    logger=logging.getLogger('distance_comparison');logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(out/'run.log'));logger.addHandler(logging.StreamHandler())
    started=time.perf_counter();set_seed(oldconfig['seed']);torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if args.device=='cuda' or args.device=='auto' and torch.cuda.is_available() else 'cpu')
    sources={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').rglob('*.py'))}
    val=CUBFewShotDataset(args.data_root,args.splits_dir/'val_images.txt',get_cub_transforms(224,False),require_clean_index=True)
    catalog,val_audit=audit_dataset(val,args.splits_dir,logger)
    if val_audit['ordered_content_sha256']!=read(base/'data_audit.json')['ordered_content_sha256']:raise ValueError('Val data changed since baseline')
    plans={k:read(base/f'plan_{k}shot.json') for k in (1,5)}
    validate_paired_plans(val,plans[1],plans[5])
    if any(plan_hash(plans[k])!=oldconfig['episode_plan_hashes'][str(k)] for k in (1,5)):
        raise ValueError('Original baseline plan hash mismatch; support/query may have changed')
    for filename in ['plan_1shot.json','plan_5shot.json','paired_episodes.jsonl','image_catalog.csv']:
        shutil.copyfile(base/filename,out/filename)
    train=CUBFewShotDataset(args.data_root,args.splits_dir/'train_images.txt',get_cub_transforms(224,False),require_clean_index=True)
    train_catalog,train_audit=audit_train(train,args.splits_dir)
    write_rows(out/'train_image_catalog.csv',train_catalog)
    write_json(out/'data_audit.json',dict(val=val_audit,train=train_audit))
    logger.info('Verified Train covariance source: %d images, %d classes; original Val caches will be reused exactly',len(train),len(train.classes))
    env=dict(python=sys.version,torch=str(torch.__version__),device=str(device),
             gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,threads=args.threads,
             versions={n:importlib.metadata.version(n) for n in ['torchvision','Pillow','numpy','scipy']},
             dtype='float32',mode='eval',deterministic=True,tf32=False)
    config=dict(baseline_dir=str(base),seed=oldconfig['seed'],episodes=600,shots=[1,5],n_way=5,n_query=5,
                distances=list(METRICS),raw_features=True,prototype='raw support mean',feature_l2_normalization=False,
                cosine='norm division only inside existing cosine formula, after raw prototype mean',
                mahalanobis='Train-only shared shrunk diagonal covariance',
                shrinkage=args.shrinkage,relative_jitter=args.relative_jitter,absolute_jitter=args.absolute_jitter,
                temperature=1,training=False,new_episode_evaluations=7200,reused_episode_evaluations=3600,
                total_episode_evaluations=10800,environment=env,batch_size=args.batch_size,workers=args.workers)
    write_json(out/'config.json',config);write_json(out/'command.json',dict(argv=sys.argv if argv is None else argv))
    write_json(out/'source_sha256.json',sources);write_json(out/'baseline_file_hashes.json',baseline_before)
    results={};bindings={};baseline_refs={}
    for name in RUNS:
        path=getattr(args,f'{name}_checkpoint');digest=sha(path)
        meta=read(base/name/'cache.json');binding=meta['binding']
        if digest!=binding['checkpoint_sha256'] or canonical_hash(binding)!=meta['key']:raise ValueError('Baseline checkpoint/cache identity differs')
        if binding['dataset_fingerprint']!=val_audit['dataset_fingerprint'] or binding['actual_content_sha256']!=val_audit['ordered_content_sha256']:
            raise ValueError('Baseline feature/data identity differs')
        for key in ('normalization','adapter'):
            if binding[key]:raise ValueError('Original features must not be normalized/adapted')
        if binding['mode']!='eval' or not binding['frozen'] or binding['preprocessing']!=repr(val.transform):raise ValueError('Baseline preprocessing/mode differs')
        for key,expected in binding['source_sha256'].items():
            if sources[key]!=expected:raise ValueError(f'Encoder/preprocessing source changed: {key}')
        val_cache_path=resolve_val_cache(meta,args.baseline_cache_dir.resolve())
        val_cache_before=sha(val_cache_path);features=cache_load(val_cache_path,binding,len(val))
        if tensor_hash(features)!=meta['feature_sha256']:raise ValueError('Val feature bytes changed')
        model=FewShotModel(method='protonet',metric='euclidean',backbone_weights=path,backbone_prefix='backbone.').to(device).eval().requires_grad_(False)
        before=state_hash(model)
        if before!=binding['encoder_state_sha256']:raise ValueError('Encoder state differs from baseline')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['split_hashes']['train_images.txt']!=train_audit['manifest_sha256']:raise ValueError('Train split differs from checkpoint')
        if checkpoint['split_hashes']['val_images.txt']!=val_audit['val_manifest_sha256']:raise ValueError('Val split differs from checkpoint')
        for key,value in model.backbone.state_dict().items():assert torch.equal(value.cpu(),checkpoint['model']['backbone.'+key])
        name_out=out/name;name_out.mkdir();results[name]={metric:{} for metric in METRICS};baseline_refs[name]={}
        for k in (1,5):
            rows,summary=load_baseline_rows(base,name,k,plans[k],catalog)
            results[name]['euclidean'][k]=dict(rows=rows,summary=summary)
            baseline_refs[name][str(k)]={filename:dict(path=str((base/name/f'{k}shot'/filename).resolve()),
                                                     sha256=sha(base/name/f'{k}shot'/filename))
                                       for filename in ('episodes.csv','metrics.json','predictions.jsonl')}
        train_binding=dict(schema='frozen-cub-train-features-v1',checkpoint_sha256=digest,encoder_state_sha256=before,
                           dataset_fingerprint=train_audit['dataset_fingerprint'],actual_content_sha256=train_audit['ordered_content_sha256'],
                           preprocessing=repr(train.transform),rgb=True,mode='eval',frozen=True,dtype='float32',feature_dim=512,
                           normalization=False,adapter=False,source_sha256=binding['source_sha256'],environment=env,encoding_batch_size=args.batch_size)
        logger.info('%s: loading exact Val features and extracting Train features for covariance',name)
        train_features,train_cache=cached_features(model,train,train_binding,args.cache_dir.resolve(),args.batch_size,args.workers,device,True,logger)
        model.cpu();labels=torch.tensor([label for _,label in train.samples],dtype=torch.int64)
        diagonal,raw_variance,covariance=pooled_diagonal_covariance(train_features,labels,args.shrinkage,args.relative_jitter,args.absolute_jitter)
        covariance.update(checkpoint_sha256=digest,train_feature_sha256=tensor_hash(train_features),
                          train_dataset_sha256=train_audit['ordered_content_sha256'],diagonal=diagonal.tolist(),
                          raw_within_class_variance=raw_variance.tolist(),shared_across_shots_and_episodes=True)
        write_json(name_out/'covariance.json',covariance);write_json(name_out/'train_cache.json',train_cache)
        write_rows(name_out/'covariance_diagonal.csv',[dict(dimension=i,raw_within_variance=float(raw_variance[i]),regularized_variance=float(diagonal[i])) for i in range(512)])
        prototype_hashes={}
        for k in (1,5):
            prototype_hashes[str(k)]=canonical_hash([tensor_hash(features[batch[:5*k]].reshape(5,k,512).mean(1)) for batch in plans[k]['batches']])
        for metric in ('cosine','mahalanobis'):
            scorer=FixedMetricScorer(metric,diagonal)
            for k in (1,5):
                logger.info('Scoring %s / %s / %d-shot: same saved raw features and plans',name,metric,k)
                shot_out=name_out/metric/f'{k}shot';shot_out.mkdir(parents=True)
                rows,summary=evaluate_cached(scorer,features,plans[k],catalog,shot_out/'predictions.jsonl',logger)
                write_rows(shot_out/'episodes.csv',rows);write_json(shot_out/'metrics.json',summary)
                results[name][metric][k]=dict(rows=rows,summary=summary)
                logger.info('%s %s %d-shot: %.4f%% +/- %.4f pp (95%% CI)',name,metric,k,
                            summary['mean_accuracy']*100,summary['ci95_half_width']*100)
        if before!=state_hash(model) or digest!=sha(path) or val_cache_before!=sha(val_cache_path) or tensor_hash(features)!=meta['feature_sha256']:
            raise ValueError('Model, checkpoint or original Val features changed during evaluation')
        bindings[name]=dict(checkpoint=str(path.resolve()),checkpoint_sha256=digest,checkpoint_epoch=checkpoint['epoch'],
                            val_cache_path=str(val_cache_path),val_feature_sha256=meta['feature_sha256'],
                            raw_prototype_hashes=prototype_hashes,encoder_unchanged=True,original_val_cache_unchanged=True)
        del model,checkpoint,features,train_features
    if baseline_before!=folder_hashes(base):raise ValueError('Old Euclidean files changed')
    if sources!={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').rglob('*.py'))}:raise ValueError('Source changed while running')
    write_json(out/'baseline_references.json',baseline_refs);write_json(out/'feature_bindings.json',bindings)
    checks=dict(status='PASS',original_baseline_files_unchanged=True,checkpoints_and_encoder_states_unchanged=True,
                exact_original_val_features_reused=True,original_plans_copied_byte_for_byte=True,
                mean_raw_support_prototypes_unchanged=True,no_extra_feature_normalization=True,
                covariance_fit_split='CUB Train only',covariance_uses_query_labels=False,covariance_uses_query_features=False,
                covariance_diagonal_positive=True,no_matrix_inverse=True,new_evaluations=7200,reused_evaluations=3600,
                total_evaluations=10800,new_query_predictions=180000,total_query_predictions=270000,
                training_performed=False,seconds=time.perf_counter()-started)
    write_json(out/'checks.json',checks);build_comparison(out,results,plans,base)
    logger.info('PASS. 7200 new episodes, 3600 reused; report: %s',out/'report.md')
    return checks


if __name__=='__main__':main()
