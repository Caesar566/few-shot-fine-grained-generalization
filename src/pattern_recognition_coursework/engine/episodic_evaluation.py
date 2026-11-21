"""Paired CUB-Val ProtoNet evaluation of three Stage04 supervised backbones.

No fitting or checkpoint selection. Reuses existing sampler, classifier and metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from pattern_recognition_coursework.analysis.episodic_statistics import accuracy_summary
from pattern_recognition_coursework.analysis.metrics import aggregate_episode_metrics, evaluation_metrics
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset, get_cub_transforms
from pattern_recognition_coursework.data.episodes import dataset_fingerprint, plan_hash
from pattern_recognition_coursework.data.paired_episodes import make_paired_plans
from pattern_recognition_coursework.engine.experiment import run_episodes, set_seed
from pattern_recognition_coursework.models.few_shot import FewShotModel

ROOT = Path(__file__).resolve().parents[3]
RUNS = {'imagenet':'supervised_imagenet_seed42_server_bs128_fresh',
        'random':'supervised_random_seed42_bs128', 'simclr':'supervised_simclr_seed42_bs128'}
COLORS = {'imagenet':'#2468b4', 'random':'#d77a13', 'simclr':'#16856a'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hash(model):
    return canonical_hash({key:tensor_hash(value) for key,value in model.state_dict().items()})


def validate_features(features, count):
    if not isinstance(features, torch.Tensor) or features.dtype != torch.float32:
        raise ValueError('Cached features must be float32 tensors')
    if features.device.type != 'cpu' or tuple(features.shape) != (count,512):
        raise ValueError('Cached feature device/shape mismatch')
    if not features.isfinite().all() or float(features.std()) == 0:
        raise ValueError('Nonfinite or constant feature cache')


def cache_load(path, expected, count):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('binding') != expected:
        raise ValueError('Feature cache binding mismatch; refusing stale cache')
    features = payload.get('features')
    validate_features(features, count)
    if payload.get('feature_sha256') != tensor_hash(features):
        raise ValueError('Feature cache tensor checksum mismatch')
    return features


def cached_features(model, dataset, binding, cache_dir, batch_size, workers, device,
                    use_disk_cache, logger):
    key = canonical_hash(binding)
    path = cache_dir/f'{key}.pt'
    reused = use_disk_cache and path.is_file()
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Feature caching requires eval mode and a frozen model')
    if reused:
        features = cache_load(path,binding,len(dataset))
        logger.info('Reused feature cache %s',path)
    else:
        pieces=[]
        loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=workers,
                          pin_memory=device.type=='cuda')
        with torch.inference_mode():
            for batch,(images,_) in enumerate(loader):
                pieces.append(model.encode(images.to(device,non_blocking=device.type=='cuda')).cpu())
                if (batch+1)%10==0 or (batch+1)==len(loader):
                    logger.info('Encoding frozen backbone: %d/%d images',min((batch+1)*batch_size,len(dataset)),len(dataset))
        features=torch.cat(pieces).contiguous()
        validate_features(features,len(dataset))
        if use_disk_cache:
            cache_dir.mkdir(parents=True,exist_ok=True)
            # Complete tensors are published atomically; mismatched files are never trusted.
            temporary=path.with_suffix(f'.{os.getpid()}.tmp')
            torch.save(dict(binding=binding,feature_sha256=tensor_hash(features),features=features),temporary)
            os.replace(temporary,path)
    return features,dict(key=key,path=str(path) if use_disk_cache else None,reused=reused,
                         disk_cache_enabled=use_disk_cache,binding=binding,feature_shape=list(features.shape),
                         feature_sha256=tensor_hash(features),tensor_bytes=features.numel()*features.element_size())


def audit_dataset(dataset, splits, logger):
    with (splits/'image_hashes.csv').open(encoding='utf-8-sig') as stream:
        expected={r['image_id']:r for r in csv.DictReader(stream)}
    manifest=[line.split(maxsplit=2) for line in dataset.split_file.read_text().splitlines() if line.strip()]
    train=[line.split(maxsplit=2) for line in (splits/'train_images.txt').read_text().splitlines() if line.strip()]
    for col in (0,1,2):
        if {r[col] for r in manifest}&{r[col] for r in train}:
            raise ValueError('CUB train/val overlap')
    catalog=[]
    for index,((path,label),(image_id,name,relative)) in enumerate(zip(dataset.samples,manifest)):
        digest=sha(path);reference=expected[image_id]
        if reference['split']!='val' or reference['relative_path']!=relative or digest!=reference['sha256']:
            raise ValueError(f'Val image differs from audited content: {path}')
        catalog.append(dict(dataset_index=index,image_id=image_id,dataset_label=label,class_name=name,
                            relative_path=relative,sha256=digest))
    val_hashes={r['sha256'] for r in catalog}
    if len(val_hashes)!=len(catalog):raise ValueError('Val contains byte-identical duplicate images')
    if val_hashes&{r['sha256'] for r in expected.values() if r['split']=='train'}:
        raise ValueError('Train/val content overlap')
    if len(dataset.classes)!=50 or len(dataset)!=2947:raise ValueError('Expected 50 classes and 2947 clean CUB Val images')
    logger.info('Validated all %d actual Val image hashes, %d unseen classes',len(dataset),len(dataset.classes))
    return catalog,dict(images=len(catalog),classes=len(dataset.classes),
                        dataset_fingerprint=dataset_fingerprint(dataset),
                        ordered_content_sha256=canonical_hash(catalog),
                        val_manifest_sha256=sha(dataset.split_file),
                        train_val_class_id_path_content_disjoint=True,
                        all_actual_val_bytes_match_clean_index=True)


def save_plans(out,plans,catalog):
    for k,plan in plans.items():write_json(out/f'plan_{k}shot.json',plan)
    with (out/'image_catalog.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(catalog[0]));w.writeheader();w.writerows(catalog)
    appearances={i:0 for i in sorted({row['dataset_label'] for row in catalog})}
    with (out/'paired_episodes.jsonl').open('w') as stream:
        for episode,(one,five) in enumerate(zip(plans[1]['batches'],plans[5]['batches'])):
            mapping=[]
            for slot in range(5):
                row=catalog[five[slot*5]];appearances[row['dataset_label']]+=1
                mapping.append(dict(episode_label=slot,dataset_label=row['dataset_label'],class_name=row['class_name']))
            def members(indices,support,k):
                return [dict(catalog[index],episode_label=(i//k if support else i//5)) for i,index in enumerate(indices)]
            record=dict(episode_id=episode,class_mapping=mapping,
                        support_1shot=members(one[:5],True,1),
                        support_5shot=members(five[:25],True,5),query=members(five[25:],False,5))
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    return dict(episodes=plans[1]['episodes'],class_appearances=appearances,
                same_queries_and_class_order=True,one_shot_is_five_shot_subset=True,
                both_plan_hashes={str(k):plan_hash(p) for k,p in plans.items()})


def evaluate_cached(model, features, plan, catalog, path, logger):
    k=plan['k_shot'];metrics=[];episode_rows=[]
    labels=torch.arange(5).repeat_interleave(5)
    with path.open('w') as predictions_file, torch.inference_mode():
        for episode,batch in enumerate(plan['batches']):
            logits=model.classify_embeddings(features[batch],5,k,5)
            probabilities=logits.softmax(-1);pred=logits.argmax(-1)
            metric=evaluation_metrics(logits,labels);metrics.append(metric)
            episode_rows.append(dict(episode_id=episode,correct=int((pred==labels).sum()),queries=25,
                                     accuracy=metric['accuracy'],cross_entropy=metric['nll']))
            classes=[catalog[batch[i*k]]['class_name'] for i in range(5)]
            # One JSON object per episode stores ALL 25 query predictions and all five scores/probabilities.
            queries=[]
            for i,index in enumerate(batch[5*k:]):
                predicted=int(pred[i]);true=int(labels[i])
                queries.append(dict(catalog[index],episode_label=true,predicted_label=predicted,
                                    predicted_class=classes[predicted],correct=predicted==true,
                                    logits=logits[i].tolist(),probabilities=probabilities[i].tolist()))
            predictions_file.write(json.dumps(dict(episode_id=episode,queries=queries),ensure_ascii=False)+'\n')
            if (episode+1)%100==0:logger.info('%d-shot scored %d/%d episodes',k,episode+1,len(plan['batches']))
    aggregate=aggregate_episode_metrics(metrics)
    summary=accuracy_summary(aggregate['episode_accuracies'])
    assert abs(summary['sample_std']-aggregate['episode_std']) < 1e-12
    return episode_rows,dict(**summary,additional_metrics=aggregate)


def verify_direct(model,dataset,features,plan,device,probes):
    """Use actual uncached image forward and the existing engine as a reference."""
    k=plan['k_shot'];batches=plan['batches'][:probes];records=[]
    loader=DataLoader(dataset,batch_sampler=batches,num_workers=0)
    direct_scores=[]
    model.to(device)
    with torch.inference_mode():
        for episode,(images,_) in enumerate(loader):
            direct=model(images.to(device),5,k,5).cpu()
            direct_scores.append(direct)
        engine,_=run_episodes(model,DataLoader(dataset,batch_sampler=batches),5,k,5,device)
    model.cpu()
    # Scores from cached features run on CPU; only the frozen image encoder uses device.
    for episode,direct in enumerate(direct_scores):
        cached=model.classify_embeddings(features[batches[episode]],5,k,5)
        torch.testing.assert_close(cached,direct,rtol=1e-4,atol=1e-5)
        assert torch.equal(cached.argmax(-1),direct.argmax(-1))
        records.append(dict(episode_id=episode,max_absolute_logit_difference=float((cached-direct).abs().max()),
                            predictions_equal=True))
    cached_metrics=[evaluation_metrics(model.classify_embeddings(features[b],5,k,5),
                                      torch.arange(5).repeat_interleave(5)) for b in batches]
    assert abs(engine['accuracy']-np.mean([r['accuracy'] for r in cached_metrics]))<1e-12
    assert abs(engine['loss']-np.mean([r['nll'] for r in cached_metrics]))<1e-5
    return dict(probes=records,existing_run_episodes_agrees=True,
                engine_accuracy=engine['accuracy'],engine_loss=engine['loss'])


def distribution_plot(out,results):
    # Vector histograms use exact correct-query counts 0..25, avoiding arbitrary bins.
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="970" viewBox="0 0 1120 970">',
         '<rect width="1120" height="970" fill="white"/>',
         '<g font-family="sans-serif" fill="#222">',
         '<text x="50" y="30" font-size="21">CUB Val episode accuracy distributions (25 queries / episode)</text>']
    hist_rows=[]
    for panel,k in enumerate((1,5)):
        top=105+panel*425;height=285;left=75;width=1000;group=width/26
        hist={name:np.bincount([r['correct'] for r in results[name][k]['rows']],minlength=26) for name in RUNS}
        maximum=max(int(x.max()) for x in hist.values()); ymax=max(1,maximum)*1.15
        svg.append(f'<text x="75" y="{top-35}" font-size="19">5-way {k}-shot, {len(results["imagenet"][k]["rows"])} paired episodes</text>')
        for step in range(6):
            value=ymax*step/5;y=top+height-height*step/5
            svg.extend([f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="#ddd"/>',
                        f'<text x="25" y="{y+4}" font-size="12">{value:.0f}</text>'])
        for j,(name,color) in enumerate(COLORS.items()):
            svg.extend([f'<rect x="{640+j*145}" y="{top-53}" width="13" height="13" fill="{color}"/>',
                        f'<text x="{658+j*145}" y="{top-42}" font-size="13">{name}</text>'])
            for correct,count in enumerate(hist[name]):
                x=left+correct*group+3+j*group*.28;barheight=height*int(count)/ymax
                svg.append(f'<rect x="{x:.2f}" y="{top+height-barheight:.2f}" width="{group*.27:.2f}" height="{barheight:.2f}" fill="{color}"/>')
                hist_rows.append(dict(initialization=name,shot=k,correct_queries=correct,accuracy=correct/25,episodes=int(count)))
        svg.extend([f'<path d="M{left},{top} V{top+height} H{left+width}" fill="none" stroke="#333"/>',
                    f'<text x="10" y="{top-8}" font-size="12">Episodes</text>'])
        for correct in (0,5,10,15,20,25):
            x=left+(correct+.5)*group
            svg.append(f'<text x="{x-12}" y="{top+height+22}" font-size="12">{correct*4}%</text>')
        svg.append(f'<text x="475" y="{top+height+47}" font-size="14">Episode accuracy</text>')
    svg.append('</g></svg>');(out/'accuracy_distribution.svg').write_text('\n'.join(svg))
    with (out/'histogram_counts.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(hist_rows[0]));writer.writeheader();writer.writerows(hist_rows)


def finish(out,results,checkpoints,config,checks):
    comparison=[]
    for name in RUNS:
        for k in (1,5):
            s=results[name][k]['summary']
            comparison.append(dict(initialization=name,shot=k,episodes=s['episodes'],
                mean_accuracy_percent=s['mean_accuracy']*100,sample_std_percentage_points=s['sample_std']*100,
                ci95_low_percent=s['ci95'][0]*100,ci95_high_percent=s['ci95'][1]*100,
                ci95_half_width_percentage_points=s['ci95_half_width']*100,
                checkpoint_sha256=checkpoints[name]['sha256'],plan_sha256=checks['both_plan_hashes'][str(k)]))
    with (out/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparison[0]));writer.writeheader();writer.writerows(comparison)
    write_json(out/'summary.json',{name:{str(k):results[name][k]['summary'] for k in (1,5)} for name in RUNS})
    distribution_plot(out,results)
    lines=['# Stage 05：成对批量 Episodic Evaluation','',
           '**CUB Val验证集评估，不是NABirds测试，也不是独立最终测试。** 三个Checkpoint在阶段4已使用CUB Val选最佳，本次只检验固定模型在更多任务上的表现，不再选模型或调整超参数。','',
           f'每种shot使用{config["episodes"]}个固定seed={config["seed"]}的5-way Episode，每类5张Query。三组模型共评分{config["episodes"]*6}个Episode、{config["episodes"]*150}个Query预测（同一图片可以在不同Episode重复出现）。','',
           '| 初始化 | Shot | Mean Accuracy | 样本标准差（百分点） | 95% CI |',
           '|---|---:|---:|---:|---|']
    for r in comparison:
        lines.append(f"| {r['initialization']} | {r['shot']} | {r['mean_accuracy_percent']:.3f}% | {r['sample_std_percentage_points']:.3f} | [{r['ci95_low_percent']:.3f}%, {r['ci95_high_percent']:.3f}%] |")
    lines += ['', '## 固定协议与复用','',
              '- 数据：原始划分不改，清洗后的CUB Val 50类/2947张。全部实际图片重新SHA256核验；与CUB训练集在类别、ID、路径及已审计内容哈希上互斥。',
              '- 从原`FewShotBatchSampler`生成5-shot计划：每类无放回采10张，前5张Support，后5张Query。1-shot只保留每类第1张Support，完整保留相同Query及类别顺序。用原`validate_plan`校验两个清单，再检查配对子集关系。',
              '- 同一shot的三个Backbone读取同一plan对象。`paired_episodes.jsonl`保存Support/Query官方ID、相对路径、内容哈希和局部标签；`image_catalog.csv`提供完整数据映射，两个`plan_*shot.json`保持现有清单格式。',
              '- 确定性预处理：RGB → Resize短边257 → CenterCrop224 → ToTensor → ImageNet Normalize；无测试随机增强。',
              '- 复用`FewShotModel.encode/classify_embeddings`与`prototype_scores`：eval模式、冻结参数、512维GAP特征；无L2归一化、标准化、Adapter，温度1。原型为类内Support均值，logits=`-平方欧氏距离/512`。没有增加或替换度量。',
              '- 复用`evaluation_metrics/aggregate_episode_metrics`计算逐Episode准确率等。原评估入口要求旧episodic Checkpoint，不能直接读取阶段4监督Checkpoint，因此新增独立入口，训练模块不改。','',
              '## 指标与置信区间','',
              '`a_e = 本Episode正确Query数 / 25`；`mean = Σa_e/n`；样本标准差`s = sqrt(Σ(a_e-mean)^2/(n-1))`；`SEM=s/sqrt(n)`；`95% CI = mean ± t(0.975,n-1)*SEM`。CI不裁剪边界。',
              '统计单位是Episode，不是将15000张Query当作独立样本。这里的CI仅衡量给定Checkpoint和固定CUB Val上随机任务采样的均值不确定性，不包括训练seed、模型选择、数据集抽样或跨数据集不确定性；三种初始化当前各只有一个训练seed。标准差不是标准误。','',
              '## 冻结特征缓存','',
              '- 每个模型只编码一次2947张验证图；对每个Episode仍重新从其Support求原型。缓存图像特征，不缓存跨Episode原型。',
              '- cache绑定Checkpoint完整SHA256、编码器state指纹、ordered数据指纹及实际图片内容哈希、Transform完整repr及版本、图像尺寸、RGB、eval模式、冻结状态、float32、编码器源码和运行环境（设备、批量大小等）。不复用训练模式或其他权重/预处理的缓存。',
              '- 缓存读取时同时校验binding、`[2947,512]`尺寸、dtype、有限性与特征内容哈希；不匹配即报错。缓存文件通过临时文件原子发布。',
              '- 每个模型/shot前3个Episode另用实际图片直接前向，并调用现有`run_episodes`交叉检查缓存结果；logits容差1e-5绝对值/1e-4相对值，预测必须完全相同。','',
              '## 图表与证据','',
              '![Episode Accuracy分布](accuracy_distribution.svg)','',
              '[比较表](comparison.csv) · [统计JSON](summary.json) · [配置](config.json) · [采样核验](plan_checks.json) · [数据核验](data_audit.json) · [Checkpoint绑定](checkpoints.json) · [验收](checks.json) · [实际日志](run.log)','']
    for name in RUNS:
        lines.append(f'- **{name}**：[缓存绑定]({name}/cache.json) · [直接前向核验]({name}/direct_verification.json)；'
                     f'[1-shot准确率]({name}/1shot/episodes.csv) · [1-shot全部预测]({name}/1shot/predictions.jsonl)；'
                     f'[5-shot准确率]({name}/5shot/episodes.csv) · [5-shot全部预测]({name}/5shot/predictions.jsonl)。')
    lines += ['', '每条预测记录提供全部5个logits、概率、真实标签与类别、预测类别、样本ID及路径；Support集合通过同一episode_id连接`paired_episodes.jsonl`。', '',
              '## 运行命令','', '```bash',
              'uv run python -m pattern_recognition_coursework.engine.episodic_evaluation \\',
              '  --output-dir outputs/stage05_protonet/episodic_eval_seed42_600_new \\',
              '  --episodes 600 --seed 42 --device auto --batch-size 64 --workers 0',
              '```','',
              '输出目录必须是新的，避免覆盖旧实验。默认读取本地三组阶段4完整`best.pth`。可用`--imagenet-checkpoint`、`--random-checkpoint`、`--simclr-checkpoint`显式指定。`--cache-dir`指定可复用的缓存位置；`--no-cache`关闭磁盘读写但仍一次提取全部冻结特征；`--batch-size`仅控制特征编码批量，不改变Episode数量。服务器可使用`--device cuda --workers 4`。', '',
              '若原始数据路径不同，指定`--data-root`、`--splits-dir`；只改变根目录，不改变索引内容。检查源码、输入权重在评估前后均保持不变，未创建优化器，也未读NABirds或CUB Test图片。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=ROOT/'cub2002011/versions/7/CUB_200_2011')
    p.add_argument('--splits-dir',type=Path,default=ROOT/'data/splits/cub_clean_v1')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'data/cache/stage05_features')
    p.add_argument('--episodes',type=int,default=600);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=64);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--cache',action=argparse.BooleanOptionalAction,default=True)
    for name,run in RUNS.items():
        p.add_argument(f'--{name}-checkpoint',type=Path,default=ROOT/'outputs/stage04_training'/run/'checkpoints/best.pth')
    args=p.parse_args(argv)
    if args.episodes<2 or args.seed<0 or args.batch_size<1 or args.workers<0 or args.threads<1:p.error('Invalid count/seed')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; no silent fallback')
    out=args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'outputs/stage05_protonet'):p.error('Output must be inside outputs/stage05_protonet')
    if out.exists():p.error('Output already exists; choose a new directory')
    for name in RUNS:
        if not getattr(args,f'{name}_checkpoint').is_file():p.error(f'Missing {name} checkpoint')
    out.mkdir(parents=True)
    logger=logging.getLogger('episodic_evaluation');logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(out/'run.log'));logger.addHandler(logging.StreamHandler())
    started=time.perf_counter();device=torch.device('cuda' if args.device=='cuda' or args.device=='auto' and torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True);set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sources={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').rglob('*.py'))}
    dataset=CUBFewShotDataset(args.data_root,args.splits_dir/'val_images.txt',get_cub_transforms(224,False),require_clean_index=True)
    catalog,audit=audit_dataset(dataset,args.splits_dir,logger)
    write_json(out/'data_audit.json',audit)
    plans=make_paired_plans(dataset,args.episodes,args.seed)
    # Independently regenerate plans to check seeded sampler reproducibility.
    assert plans==make_paired_plans(dataset,args.episodes,args.seed)
    plan_checks=save_plans(out,plans,catalog);write_json(out/'plan_checks.json',plan_checks)
    env=dict(python=sys.version,platform=platform.platform(),device=str(device),
             gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
             torch=str(torch.__version__),versions={name:importlib.metadata.version(name) for name in ['torchvision','Pillow','numpy','scipy']},
             threads=args.threads,deterministic=True,tf32=False)
    config=dict(episodes=args.episodes,seed=args.seed,n_way=5,shots=[1,5],n_query=5,
                metric='euclidean',logits='-sum_squared_distance / 512',temperature=1,
                feature_standardization=False,feature_normalization=False,adapter=False,
                split='CUB clean Val',image_size=224,mode='eval',frozen=True,training=False,
                data_root=str(args.data_root),splits_dir=str(args.splits_dir),
                cache=args.cache,batch_size=args.batch_size,workers=args.workers,
                transform=repr(dataset.transform),environment=env,
                episode_plan_hashes=plan_checks['both_plan_hashes'])
    write_json(out/'config.json',config);write_json(out/'environment.json',env)
    write_json(out/'code_version.json',dict(package_source_sha256=sources))
    write_json(out/'command.json',dict(argv=sys.argv if argv is None else argv))
    results={};checkpoints={};verification={}
    for name in RUNS:
        path=getattr(args,f'{name}_checkpoint');digest=sha(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint['split_hashes']['val_images.txt']!=audit['val_manifest_sha256']:
            raise ValueError('Checkpoint validation split differs')
        model=FewShotModel(method='protonet',metric='euclidean',backbone_weights=path,backbone_prefix='backbone.').to(device).eval().requires_grad_(False)
        for key,value in model.backbone.state_dict().items():
            assert torch.equal(value.cpu(),checkpoint['model']['backbone.'+key])
        before=state_hash(model)
        binding=dict(schema='frozen-cub-val-features-v1',checkpoint_sha256=digest,encoder_state_sha256=before,
                     dataset_fingerprint=audit['dataset_fingerprint'],actual_content_sha256=audit['ordered_content_sha256'],
                     preprocessing=repr(dataset.transform),rgb=True,image_size=224,mode='eval',frozen=True,
                     dtype='float32',feature_dim=512,normalization=False,adapter=False,
                     source_sha256={key:value for key,value in sources.items() if key.endswith(('models/few_shot.py','backbones/resnet.py','data/cub_dataset.py'))},
                     environment=env,encoding_batch_size=args.batch_size)
        logger.info('Evaluating %s: checkpoint epoch=%d SHA256=%s device=%s',name,checkpoint['epoch'],digest,device)
        model_out=out/name;model_out.mkdir()
        features,cache=cached_features(model,dataset,binding,args.cache_dir.resolve(),args.batch_size,args.workers,device,args.cache,logger)
        write_json(model_out/'cache.json',cache)
        model.cpu()
        results[name]={};verification[name]={}
        for k in (1,5):
            shot_out=model_out/f'{k}shot';shot_out.mkdir()
            # Existing ProtoNet scores use CPU tensors consistently; only encoding uses device.
            rows,summary=evaluate_cached(model,features,plans[k],catalog,shot_out/'predictions.jsonl',logger)
            with (shot_out/'episodes.csv').open('w',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            write_json(shot_out/'metrics.json',summary)
            results[name][k]=dict(rows=rows,summary=summary)
            direct=verify_direct(model,dataset,features,plans[k],device,min(3,args.episodes))
            verification[name][str(k)]=direct
            # Re-score every saved episode, not just a spot-check, to verify cached reproducibility.
            with torch.inference_mode():
                for batch,row in zip(plans[k]['batches'],rows):
                    predictions=model.classify_embeddings(features[batch],5,k,5).argmax(-1)
                    assert int((predictions==torch.arange(5).repeat_interleave(5)).sum())==row['correct']
            logger.info('%s %d-shot: %.4f%% +/- %.4f pp (95%% CI), sample std=%.4f pp',
                        name,k,summary['mean_accuracy']*100,summary['ci95_half_width']*100,summary['sample_std']*100)
        write_json(model_out/'direct_verification.json',verification[name])
        assert before==state_hash(model) and digest==sha(path)
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters())
        checkpoints[name]=dict(path=str(path.resolve()),sha256=digest,epoch=checkpoint['epoch'],
                               initialization=checkpoint['config']['weights'],loader=model.backbone_load_info,
                               encoder_state_unchanged=True,checkpoint_file_unchanged=True)
        del model,checkpoint,features
    assert sources=={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').rglob('*.py'))}
    checks=dict(status='PASS',**plan_checks,weight_files_and_states_unchanged=True,
                package_source_unchanged=True,cached_vs_direct_image_forward=True,
                existing_run_episodes_agrees=True,repeat_all_cached_accuracies_equal=True,
                evaluated_model_shot_combinations=6,total_scored_episodes=args.episodes*6,
                total_query_predictions=args.episodes*150,training_performed=False,
                seconds=time.perf_counter()-started)
    write_json(out/'checkpoints.json',checkpoints);write_json(out/'checks.json',checks)
    finish(out,results,checkpoints,config,checks)
    logger.info('PASS. Complete report: %s',out/'report.md')
    return checks


if __name__=='__main__':main()
