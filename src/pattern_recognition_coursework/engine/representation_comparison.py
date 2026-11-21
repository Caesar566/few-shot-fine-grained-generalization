"""CUDA-first frozen Global/Local/Fusion comparison on saved paired episodes."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import itertools
import json
import logging
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import shutil
import sys
import time
import traceback
import uuid

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset

from pattern_recognition_coursework.analysis.episodic_statistics import accuracy_summary,paired_accuracy_difference
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset,get_cub_transforms
from pattern_recognition_coursework.data.paired_episodes import validate_paired_plans
from pattern_recognition_coursework.data.episodes import plan_hash
from pattern_recognition_coursework.engine.episodic_evaluation import (
    ROOT,RUNS,audit_dataset,cache_load,canonical_hash,sha,state_hash,tensor_hash,write_json,
)
from pattern_recognition_coursework.engine.distance_comparison import folder_hashes,resolve_val_cache,write_rows
from pattern_recognition_coursework.engine.experiment import set_seed
from pattern_recognition_coursework.models.few_shot import FewShotModel
from pattern_recognition_coursework.models.metrics.representation import representation_scores

REPS=('global','local','fusion')


def read(path):return json.loads(Path(path).read_text())


def choose_device(requested,logger):
    if requested=='cpu':return torch.device('cpu')
    if torch.cuda.is_available():return torch.device('cuda')
    logger.warning('CUDA unavailable: requested=%s, explicitly falling back to CPU; large full runs will be slower',requested)
    return torch.device('cpu')


def describe(t):
    return dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device),finite=bool(t.isfinite().all()),
                minimum=float(t.min()),maximum=float(t.max()),mean=float(t.mean()),std=float(t.std(unbiased=False)))


def collect_local(model,dataset,indices,global_features,binding,cache_dir,batch_size,workers,device,logger):
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval model required')
    key=canonical_hash(binding);path=cache_dir/f'{key}.pt';hit=path.exists();started=time.perf_counter()
    reused_images=0;reused_sources=[]
    if hit:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['binding']!=binding or payload['indices']!=indices:raise ValueError('Local cache identity mismatch')
        local=payload['local'];gap=payload['gap']
        if tensor_hash(local)!=payload['local_sha256'] or tensor_hash(gap)!=payload['gap_sha256']:raise ValueError('Local cache checksum mismatch')
        reused_images=len(indices)
    else:
        local=torch.empty(len(indices),512,7,7,dtype=torch.float32)
        gap=torch.empty(len(indices),512,dtype=torch.float32)
        filled=torch.zeros(len(indices),dtype=torch.bool)
        destination={index:row for row,index in enumerate(indices)}
        compatible=lambda b:{k:v for k,v in b.items() if k not in ('sample_indices','ordered_sample_ids')}
        # A smoke subset and a full run have different cache keys, but identical
        # frozen extraction bindings. Reuse verified rows, never another model.
        for previous in sorted(cache_dir.glob('*.pt')):
            if previous==path:continue
            candidate=torch.load(previous,map_location='cpu',weights_only=True,mmap=True)
            if compatible(candidate.get('binding',{}))!=compatible(binding):continue
            old_indices=candidate.get('indices',[])
            if len(set(old_indices))!=len(old_indices):raise ValueError('Duplicate indices in partial cache')
            old_ids=candidate['binding'].get('ordered_sample_ids')
            new_ids=binding.get('ordered_sample_ids')
            matches=[]
            for old_row,index in enumerate(old_indices):
                if index in destination and not filled[destination[index]]:
                    new_row=destination[index]
                    if old_ids is not None and new_ids is not None and old_ids[old_row]!=new_ids[new_row]:
                        raise ValueError('Partial cache sample ID mismatch')
                    matches.append((old_row,new_row))
            if not matches:continue
            old_local=candidate['local'];old_gap=candidate['gap']
            if old_local.shape!=(len(old_indices),512,7,7) or old_gap.shape!=(len(old_indices),512):raise ValueError('Partial cache shape mismatch')
            if old_local.dtype!=torch.float32 or old_gap.dtype!=torch.float32:raise ValueError('Partial cache must be FP32')
            if tensor_hash(old_local)!=candidate['local_sha256'] or tensor_hash(old_gap)!=candidate['gap_sha256']:raise ValueError('Partial cache checksum mismatch')
            old_rows,new_rows=zip(*matches)
            local[list(new_rows)]=old_local[list(old_rows)];gap[list(new_rows)]=old_gap[list(old_rows)]
            filled[list(new_rows)]=True
            reused_sources.append(dict(path=str(previous),images=len(matches),local_sha256=candidate['local_sha256']))
            logger.info('Reusing %d verified Local rows from %s',len(matches),previous.name)
            del candidate,old_local,old_gap
            if filled.all():break
        reused_images=int(filled.sum())
        missing_rows=torch.nonzero(~filled).flatten().tolist()
        missing_indices=[indices[row] for row in missing_rows];offset=0
        with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
            loader=DataLoader(Subset(dataset,missing_indices),batch_size=batch_size,shuffle=False,num_workers=workers,
                              pin_memory=device.type=='cuda')
            for batch,(images,_) in enumerate(loader):
                images=images.to(device,non_blocking=device.type=='cuda')
                maps=model.encode(images)
                if maps.shape[1:]!=(512,7,7) or maps.dtype!=torch.float32:raise ValueError('Expected FP32 Layer4 [B,512,7,7]')
                pooled=model.backbone.avgpool(maps).flatten(1)
                target=missing_rows[offset:offset+len(images)]
                local[target]=maps.cpu();gap[target]=pooled.cpu()
                offset+=len(images)
                if batch==0 or (batch+1)%25==0 or offset==len(missing_indices):
                    logger.info('Layer4 extraction on %s: %d/%d new images, reused=%d, batch=%d',device,offset,len(missing_indices),reused_images,batch_size)
        cache_dir.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(f'.{os.getpid()}.tmp')
        torch.save(dict(binding=binding,indices=indices,local=local,gap=gap,
                        local_sha256=tensor_hash(local),gap_sha256=tensor_hash(gap)),temporary)
        os.replace(temporary,path)
    if local.shape!=(len(indices),512,7,7) or gap.shape!=(len(indices),512):raise ValueError('Local cache size mismatch')
    if local.dtype!=torch.float32 or gap.dtype!=torch.float32 or local.device.type!='cpu' or not local.isfinite().all():
        raise ValueError('Invalid CPU FP32 local feature cache')
    # New target datasets have no pre-existing Global cache. Their GAP vector
    # comes from this exact same frozen Layer4 forward; CUB callers retain the
    # original, stricter cache-to-baseline check.
    if global_features is not None:
        torch.testing.assert_close(gap,global_features[indices],rtol=1e-4,atol=5e-5)
    torch.testing.assert_close(gap,local.mean((-1,-2)),rtol=1e-4,atol=5e-5)
    return local,dict(key=key,path=str(path),hit=hit,binding=binding,local_sha256=tensor_hash(local),
        gap_sha256=tensor_hash(gap),shape=list(local.shape),cpu_bytes=local.numel()*4,
        reused_images=reused_images,extracted_images=len(indices)-reused_images,partial_sources=reused_sources,
        gap_vs_original_global_max_abs=float((gap-global_features[indices]).abs().max()) if global_features is not None else None,
        extraction_seconds=time.perf_counter()-started)


def episode_inputs(batch,k,row_for_index,local,global_features,device):
    support=local[[row_for_index[i] for i in batch[:5*k]]]
    query=local[[row_for_index[i] for i in batch[5*k:]]]
    return global_features[batch].to(device),support,query


def local_diagnostics(out,k,global_features,support,query,scores,device,local_k):
    # Diagnostic only: first Query and first class, on the same evaluation device.
    s=F.normalize(support[:k].to(device).flatten(2).transpose(1,2).reshape(k*49,512),dim=-1)
    q=F.normalize(query[:1].to(device).flatten(2).transpose(1,2)[0],dim=-1)
    similarity=q@s.T;top=similarity.topk(local_k,dim=-1)
    torch.testing.assert_close(top.values.mean(),scores['local_scores'][0,0],rtol=1e-5,atol=1e-6)
    with (out/'similarity_q0_class0.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['query_patch',*[f'support_descriptor_{i}' for i in range(k*49)]])
        w.writerows([i,*row] for i,row in enumerate(similarity.cpu().tolist()))
    write_json(out/'topk_q0_class0.json',dict(indices=top.indices.cpu().tolist(),values=top.values.cpu().tolist(),
        mean=float(top.values.mean()),support_image_slot='index // 49',support_patch='index % 49'))
    write_json(out/'diagnostics.json',dict(global_features=describe(global_features),
        local_support=describe(support.to(device)),local_query=describe(query.to(device)),
        similarity=describe(similarity),query_descriptors=[49,512],class_descriptors=[k*49,512],
        top_k=local_k,score_aggregation='mean over all 49 query patches and their top-k similarities',
        **{name:describe(value) for name,value in scores.items()}))


def cpu_reference(model,dataset,batch,k,global_cpu,cached_support,cached_query,gpu_scores,device,args):
    """Small, explicit diagnostic exception; bulk inference/scoring stays on CUDA."""
    model.cpu();pieces=[]
    try:
        with torch.inference_mode():
            for images,_ in DataLoader(Subset(dataset,batch),batch_size=args.batch_size):pieces.append(model.encode(images))
            maps=torch.cat(pieces);cached=torch.cat([cached_support,cached_query])
            # Deep FP32 CPU/CUDA convolutions accumulate rounding differences,
            # especially at near-zero ReLU activations. Scores/predictions below
            # remain checked separately; this does not change inference precision.
            torch.testing.assert_close(maps,cached,rtol=2e-4,atol=2e-4)
            reference=representation_scores(global_cpu[batch],maps[:5*k],maps[5*k:],5,k,args.local_k,args.query_chunk,args.calibration_eps)
            result=dict(feature_max_abs_diff=float((maps-cached).abs().max()),
                        feature_mean_abs_diff=float((maps-cached).abs().mean()),
                        feature_tolerance=dict(rtol=2e-4,atol=2e-4),score_max_abs_diff={})
            for name in ('global_scores','local_scores','fusion_scores'):
                actual=gpu_scores[name].cpu()
                torch.testing.assert_close(reference[name],actual,rtol=1e-4,atol=5e-5)
                if not torch.equal(reference[name].argmax(-1),actual.argmax(-1)):raise ValueError('CPU/CUDA predictions differ in functional probe')
                result['score_max_abs_diff'][name]=float((reference[name]-actual).abs().max())
            result['predictions_equal']=True
            return result
    finally:model.to(device).eval()


def write_report(out,results,checks,config):
    table=[];pairs=[];deltas=[]
    for name,methods in results.items():
        for k in (1,5):
            for rep in REPS:
                case=methods[rep][k];s=case['summary']
                table.append(dict(initialization=name,representation=rep,shot=k,episodes=s['episodes'],
                    mean_accuracy_percent=s['mean_accuracy']*100,sample_std_pp=s['sample_std']*100,
                    ci95_low_percent=s['ci95'][0]*100,ci95_high_percent=s['ci95'][1]*100))
            for a,b in itertools.combinations(REPS,2):
                d=paired_accuracy_difference(methods[a][k]['accuracies'],methods[b][k]['accuracies'])
                pairs.append(dict(initialization=name,shot=k,first=a,second=b,mean_delta_pp=d['mean_delta']*100,
                    sample_std_delta_pp=d['sample_std_delta']*100,ci95_low_pp=d['ci95'][0]*100,ci95_high_pp=d['ci95'][1]*100))
                deltas.extend(dict(initialization=name,shot=k,first=a,second=b,episode_id=i,
                    first_accuracy=x,second_accuracy=y,delta_pp=(y-x)*100)
                    for i,(x,y) in enumerate(zip(methods[a][k]['accuracies'],methods[b][k]['accuracies'])))
    write_rows(out/'comparison.csv',table);write_rows(out/'paired_comparison.csv',pairs)
    write_rows(out/'paired_episode_differences.csv',deltas)
    write_json(out/'summary.json',{name:{rep:{str(k):c['summary'] for k,c in shots.items()} for rep,shots in methods.items()} for name,methods in results.items()})
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="820"><rect width="1120" height="820" fill="white"/><g font-family="sans-serif" fill="#222">',
         '<text x="25" y="30" font-size="20">Global / Local / calibrated Fusion: mean accuracy and 95% CI</text>']
    colors={'global':'#2468b4','local':'#d77a13','fusion':'#16856a'}
    for panel,k in enumerate((1,5)):
        top=90+panel*350
        svg.append(f'<text x="25" y="{top}" font-size="18">5-way {k}-shot ({config["mode"]})</text>')
        for tick in range(0,101,20):
            x=330+tick*7
            svg.append(f'<line x1="{x}" y1="{top+20}" x2="{x}" y2="{top+270}" stroke="#ddd"/><text x="{x-12}" y="{top+290}" font-size="12">{tick}%</text>')
        for i,r in enumerate(x for x in table if x['shot']==k):
            y=top+40+i*27;color=colors[r['representation']]
            svg.append(f'<text x="25" y="{y+4}" font-size="13">{r["initialization"]} / {r["representation"]}</text><line x1="{330+r["ci95_low_percent"]*7}" y1="{y}" x2="{330+r["ci95_high_percent"]*7}" y2="{y}" stroke="{color}" stroke-width="3"/><circle cx="{330+r["mean_accuracy_percent"]*7}" cy="{y}" r="4" fill="{color}"/><text x="1040" y="{y+4}" font-size="12">{r["mean_accuracy_percent"]:.2f}%</text>')
    svg.append('</g></svg>');(out/'accuracy_comparison.svg').write_text('\n'.join(svg))
    write_full_figures(out,table,pairs)
    lines=['# Global / Local / Fusion 冻结表示对比','',
        f'模式：**{config["mode"]}**；实际设备：**{checks["device_name"]}**，dtype=float32。'
        f'每种Shot实际使用原计划前{config["episodes"]}个Episode；不重采样，不训练，不读取NABirds。', '',
        '**smoke结果仅供功能验证，不能代替600-Episode比较。**' if config['mode']=='smoke' else '完整批量评估：三种Backbone、两种Shot、每种600个Episode；Global原结果直接复用。', '',
        '| 初始化 | 表示 | Shot | Mean Accuracy | 样本标准差（pp） | 95% CI |','|---|---|---:|---:|---:|---|']
    for r in table:lines.append(f'| {r["initialization"]} | {r["representation"]} | {r["shot"]} | {r["mean_accuracy_percent"]:.3f}% | {r["sample_std_pp"]:.3f} | [{r["ci95_low_percent"]:.3f}%, {r["ci95_high_percent"]:.3f}%] |')
    lines += ['', '## 数学定义与设备','',
        '- Global：原始GAP 512维，原始Support均值Prototype，再计算Cosine。读取旧Global特征和旧Cosine结果；当前设备也用原函数计算Prototype/Cosine以核对旧logits和预测，并供Fusion使用，原结果不覆盖。',
        '- Local：同一权重、同一确定性预处理的Layer4 `[N,512,7,7]`。每图49个512维描述子；在余弦匹配内部对每个描述子除范数，不将局部描述子先平均为原型。',
        '- 每个类汇集K×49个Support描述子；每个Query的49个描述子分别与候选类全部描述子匹配，取Top-3；类分数为49×3个余弦相似度的均值。相似度矩阵为`[query_chunk,49,K*49]`，按Query分块、按类处理。',
        '- Fusion：先对每个Query的每一路五类分数做行内z-score：`z=(s-mean_classes(s))/max(population_std_classes(s),1e-6)`；再用`0.5*z_global+0.5*z_local`。参数预先固定，不使用Query真标签、不拟合温度/模块、不调权重。常数分数行输出零向量。',
        '- 所有前向、Prototype、Cosine、Top-k匹配、校准、Fusion、Softmax与预测在实际评估设备上使用PyTorch执行；CPU只保留完整特征缓存、保存结果和汇总标量统计。smoke额外执行少量CPU前向以核对GPU逻辑。',
        '- Global原始缓存与Local cache分别绑定权重SHA256、预处理及版本、完整Val内容指纹、有序样本ID、FP32/eval状态、计算设备和源码。每次只上传Episode Support和一个Query块，绝不把全部Val Local缓存搬进GPU。',
        '- 初始特征提取Batch Size=8、Query chunk=4，默认匹配2GB显存；不启用AMP/TF32。全程eval、inference_mode，无优化器与反向传播。', '',
        '## 核验证据与统计','',
        '- `feature_binding.json/cache.json`：缓存命中、SHA256、样本顺序、Layer4/GAP对照及提取耗时。',
        '- `*/diagnostics.json/similarity_q0_class0.csv/topk_q0_class0.json`：首个真实Episode维度、匹配矩阵、Top-k及分数范围。',
        '- `*/predictions.jsonl`：每个Episode的ID、类映射、Support/Query、三路完整分数、校准分数、概率和预测；每路Accuracy另存CSV。',
        '- `performance.json/checks.json`：实际设备、总耗时、CUDA峰值allocated/reserved显存、缓存命中和CPU/GPU核验；不使用估算显存伪造实测。',
        '- 使用Episode Accuracy的ddof=1样本标准差，Student-t 95%均值CI；配对差为second−first，在同一个episode_id上计算。smoke仅3个任务，CI很宽且可能超出[0,100]，不作性能结论。', '',
        '![表示对比](accuracy_comparison.svg)', '', '[完整比较表](comparison.csv) · [配对差值](paired_comparison.csv) · [配置](config.json) · [验收](checks.json) · [日志](run.log)', '',
        '![准确率柱状图](accuracy_bars.svg)', '', '![配对提升与95% CI](paired_lifts.svg)', '',
        '旧`global_local`未校准融合分支保持原样，本次复用底层Global/Local函数并在独立入口执行固定校准。CUB Val曾用于阶段4选择Checkpoint，因此本阶段不是独立最终测试或NABirds跨数据集评估。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def write_full_figures(out,table,pairs):
    colors={'global':'#2468b4','local':'#d77a13','fusion':'#16856a'}
    names=list(dict.fromkeys(r['initialization'] for r in table))
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="870"><rect width="1120" height="870" fill="white"/><g font-family="sans-serif" fill="#222">',
         '<text x="35" y="30" font-size="21">Mean episode accuracy (%) with 95% Student-t CI</text>']
    for j,(rep,color) in enumerate(colors.items()):svg.append(f'<rect x="{700+j*130}" y="14" width="13" height="13" fill="{color}"/><text x="{718+j*130}" y="26" font-size="13">{rep}</text>')
    for panel,k in enumerate((1,5)):
        top=90+panel*385;height=265
        svg.append(f'<text x="70" y="{top-22}" font-size="18">5-way {k}-shot</text>')
        for tick in range(0,101,20):
            y=top+height-tick*height/100
            svg.append(f'<line x1="85" y1="{y}" x2="1060" y2="{y}" stroke="#ddd"/><text x="35" y="{y+4}" font-size="12">{tick}%</text>')
        for i,name in enumerate(names):
            for j,rep in enumerate(REPS):
                r=next(r for r in table if r['initialization']==name and r['shot']==k and r['representation']==rep)
                x=150+i*300+j*76;y=top+height-r['mean_accuracy_percent']*height/100
                lo=top+height-r['ci95_low_percent']*height/100;hi=top+height-r['ci95_high_percent']*height/100
                svg.append(f'<rect x="{x}" y="{y}" width="58" height="{top+height-y}" fill="{colors[rep]}"/><path d="M{x+29},{lo} V{hi} M{x+22},{lo} H{x+36} M{x+22},{hi} H{x+36}" stroke="#222" fill="none"/><text x="{x+4}" y="{y-9}" font-size="11">{r["mean_accuracy_percent"]:.2f}</text>')
            svg.append(f'<text x="{170+i*300}" y="{top+height+27}" font-size="15">{name}</text>')
    svg.append('</g></svg>');(out/'accuracy_bars.svg').write_text('\n'.join(svg))
    limit=max(1,max(abs(r[key]) for r in pairs for key in ('ci95_low_pp','ci95_high_pp')))*1.15
    height=max(360,140+39*len(pairs));point=lambda x:710+310*x/limit
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}"><rect width="1120" height="{height}" fill="white"/><g font-family="sans-serif" fill="#222">',
         '<text x="25" y="30" font-size="20">Paired accuracy lift (second minus first), 95% CI</text>',
         f'<line x1="710" y1="60" x2="710" y2="{height-65}" stroke="#888" stroke-dasharray="5,4"/>']
    for i,r in enumerate(pairs):
        y=85+i*39
        label=f'{r["initialization"]} {r["shot"]}-shot: {r["second"]} - {r["first"]}'
        svg.append(f'<text x="25" y="{y+4}" font-size="13">{label}</text><line x1="{point(r["ci95_low_pp"])}" y1="{y}" x2="{point(r["ci95_high_pp"])}" y2="{y}" stroke="#2468b4" stroke-width="3"/><circle cx="{point(r["mean_delta_pp"])}" cy="{y}" r="4" fill="#2468b4"/><text x="1040" y="{y+4}" font-size="12">{r["mean_delta_pp"]:+.2f} pp</text>')
    for fraction in (-1,-.5,0,.5,1):svg.append(f'<text x="{point(fraction*limit)-15}" y="{height-35}" font-size="12">{fraction*limit:+.1f} pp</text>')
    svg.append('</g></svg>');(out/'paired_lifts.svg').write_text('\n'.join(svg))


def _run(argv=None,run_token=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--episode-source-dir',type=Path,default=ROOT/'outputs/stage05_protonet/episodic_eval_seed42_600')
    p.add_argument('--global-baseline-dir',type=Path,default=ROOT/'outputs/stage05_protonet/distance_comparison_seed42_600')
    p.add_argument('--baseline-cache-dir',type=Path,default=ROOT/'data/cache/stage05_features')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'data/cache/stage05_representation_features')
    p.add_argument('--data-root',type=Path,default=ROOT/'cub2002011/versions/7/CUB_200_2011')
    p.add_argument('--splits-dir',type=Path,default=ROOT/'data/splits/cub_clean_v1')
    p.add_argument('--validation-report',type=Path)
    p.add_argument('--smoke-episodes',type=int,default=3)
    p.add_argument('--backbones',nargs='+',choices=list(RUNS))
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--batch-size','--feature-batch-size',dest='batch_size',type=int,default=8)
    p.add_argument('--query-chunk',type=int,default=4);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--local-k',type=int,default=3);p.add_argument('--calibration-eps',type=float,default=1e-6)
    p.add_argument('--threads',type=int,default=2)
    for name,run in RUNS.items():p.add_argument(f'--{name}-checkpoint',type=Path,default=ROOT/'outputs/stage04_training'/run/'checkpoints/best.pth')
    args=p.parse_args(argv)
    if min(args.batch_size,args.query_chunk,args.threads,args.smoke_episodes)<1 or args.workers<0 or args.smoke_episodes<2:p.error('Invalid counts')
    if not 1<=args.local_k<=49 or not 0<args.calibration_eps<1:p.error('Invalid local-k/epsilon')
    names=args.backbones or (['imagenet'] if args.mode=='smoke' else list(RUNS))
    if args.mode=='full' and set(names)!=set(RUNS):p.error('Full experiment requires all three backbones')
    out=args.output_dir.resolve();source=args.episode_source_dir.resolve();baseline=args.global_baseline_dir.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'outputs/stage05_protonet') or out.is_relative_to(source) or out.is_relative_to(baseline):p.error('Choose a new sibling output under outputs/stage05_protonet')
    signature=dict(local_k=args.local_k,calibration='per-query population zscore across five class scores',
                   calibration_eps=args.calibration_eps,fusion_weight=.5,global_metric='cosine',raw_global_prototype_mean=True,fp32=True,
                   implementation_sha256={str(f.relative_to(ROOT)):sha(f) for f in
                       [ROOT/'src/pattern_recognition_coursework/models/metrics'/name for name in ['representation.py','local.py','episode.py']]})
    if args.mode=='full':
        if not args.validation_report:p.error('Run smoke first and pass its checks.json via --validation-report')
        gate=read(args.validation_report)
        if gate['status']!='PASS' or gate['mode']!='smoke' or gate['method_signature']!=signature:p.error('Smoke report failed or method differs')
    source_config=read(source/'config.json')
    if source_config['episodes']!=600 or source_config['n_way']!=5 or source_config['n_query']!=5:p.error('Expected original paired 600-episode source')
    count=600 if args.mode=='full' else args.smoke_episodes
    if count>600:p.error('Smoke cannot exceed original plan')
    out.mkdir(parents=True)
    write_json(out/'status.json',dict(state='running',run_token=run_token,phase='startup'))
    logger=logging.getLogger('representation_comparison');logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(out/'run.log'));logger.addHandler(logging.StreamHandler())
    device=choose_device(args.device,logger)
    device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'
    logger.info('Requested=%s; ACTUAL device=%s (%s); FP32, no AMP/TF32; feature batch=%d query chunk=%d',args.device,device,device_name,args.batch_size,args.query_chunk)
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True);set_seed(source_config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter()
    protected={str(path):folder_hashes(path) for path in [source,baseline]}
    sources={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').rglob('*.py'))}
    dataset=CUBFewShotDataset(args.data_root,args.splits_dir/'val_images.txt',get_cub_transforms(224,False),require_clean_index=True)
    catalog,audit=audit_dataset(dataset,args.splits_dir,logger)
    plans={k:read(source/f'plan_{k}shot.json') for k in (1,5)}
    validate_paired_plans(dataset,plans[1],plans[5])
    if any(plan_hash(plans[k])!=source_config['episode_plan_hashes'][str(k)] for k in (1,5)):raise ValueError('Episode plan changed')
    for filename in ['plan_1shot.json','plan_5shot.json','paired_episodes.jsonl','image_catalog.csv']:
        if (source/filename).read_bytes()!=(baseline/filename).read_bytes():raise ValueError('Global baseline uses different samples')
        shutil.copyfile(source/filename,out/filename)
    indices=sorted({i for k in (1,5) for batch in plans[k]['batches'][:count] for i in batch})
    row_for_index={index:row for row,index in enumerate(indices)}
    config=dict(mode=args.mode,episodes=count,backbones=names,seed=source_config['seed'],device=str(device),device_name=device_name,
        requested_device=args.device,method_signature=signature,feature_batch_size=args.batch_size,query_chunk=args.query_chunk,
        source_dir=str(source),global_baseline_dir=str(baseline),workers=args.workers,
        plan_hashes={str(k):plan_hash(v) for k,v in plans.items()},used_episode_ids=list(range(count)),
        validation_report=str(args.validation_report) if args.validation_report else None)
    write_json(out/'config.json',config);write_json(out/'data_audit.json',audit)
    write_json(out/'command.json',dict(argv=sys.argv if argv is None else argv))
    write_json(out/'source_sha256.json',sources)
    validation_seconds=time.perf_counter()-started
    results={};performance={};bindings={};reference_checks={}
    for name in names:
        model_started=time.perf_counter()
        write_json(out/'status.json',dict(state='running',run_token=run_token,phase='load_and_feature_extraction',backbone=name))
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        path=getattr(args,f'{name}_checkpoint');digest=sha(path);meta=read(source/name/'cache.json')
        if meta['binding']['checkpoint_sha256']!=digest or meta['binding']['actual_content_sha256']!=audit['ordered_content_sha256']:raise ValueError('Weight/data mismatch')
        if meta['binding']['normalization'] or meta['binding']['adapter'] or meta['binding']['mode']!='eval' or not meta['binding']['frozen']:
            raise ValueError('Expected original raw frozen Global features')
        if meta['binding']['preprocessing']!=repr(dataset.transform):raise ValueError('Preprocessing mismatch')
        for key,value in meta['binding']['source_sha256'].items():
            if sources[key]!=value:raise ValueError('Backbone/preprocessing code differs from baseline')
        original_global=cache_load(resolve_val_cache(meta,args.baseline_cache_dir),meta['binding'],len(dataset))
        if tensor_hash(original_global)!=meta['feature_sha256']:raise ValueError('Global cache content mismatch')
        if read(baseline/'feature_bindings.json')[name]['checkpoint_sha256']!=digest:raise ValueError('Global cosine weights differ')
        model=FewShotModel(method='global_local',metric='cosine',backbone_weights=path,backbone_prefix='backbone.').to(device).eval().requires_grad_(False)
        before=state_hash(model)
        if before!=meta['binding']['encoder_state_sha256']:raise ValueError('Encoder does not match original')
        if args.mode=='full' and name=='imagenet' and read(args.validation_report)['checkpoint_sha256']['imagenet']!=digest:raise ValueError('Smoke checkpoint differs')
        binding=dict(schema='raw-layer4-val-cache-v1',checkpoint_sha256=digest,encoder_state_sha256=before,
            dataset_content_sha256=audit['ordered_content_sha256'],sample_indices=indices,
            ordered_sample_ids=[catalog[i]['image_id'] for i in indices],preprocessing=repr(dataset.transform),
            original_global_sha256=meta['feature_sha256'],mode='eval',frozen=True,dtype='float32',shape=[512,7,7],
            feature_source_sha256=meta['binding']['source_sha256'],torch=str(torch.__version__),
            device=str(device),device_name=device_name,batch_size=args.batch_size,
            versions={n:importlib.metadata.version(n) for n in ['torchvision','Pillow','numpy']},tf32=False)
        name_out=out/name;name_out.mkdir()
        logger.info('%s checkpoint=%s; extracting/caching only %d needed images',name,digest,len(indices))
        local,cache=collect_local(model,dataset,indices,original_global,binding,args.cache_dir.resolve(),args.batch_size,args.workers,device,logger)
        probe_started=time.perf_counter()
        # Even a warm-cache smoke run must exercise actual backbone inference on
        # the requested device, and verify that Layer4 -> GAP matches full forward.
        with torch.inference_mode():
            probe_count=min(args.batch_size,len(indices))
            probe_images=torch.stack([dataset[i][0] for i in indices[:probe_count]]).to(device)
            probe_maps=model.encode(probe_images)
            probe_gap=model.backbone(probe_images)
            torch.testing.assert_close(probe_gap,model.backbone.avgpool(probe_maps).flatten(1),rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(probe_maps.cpu(),local[:probe_count],rtol=1e-4,atol=5e-5)
            cache['forward_probe']=dict(device=str(device),input_shape=list(probe_images.shape),
                layer4_shape=list(probe_maps.shape),gap_shape=list(probe_gap.shape),
                local_cache_max_abs_diff=float((probe_maps.cpu()-local[:probe_count]).abs().max()))
            del probe_images,probe_maps,probe_gap
        write_json(name_out/'cache.json',cache)
        if device.type=='cuda':torch.cuda.synchronize(device)
        probe_seconds=time.perf_counter()-probe_started
        results[name]={rep:{} for rep in REPS};reference_checks[name]={};classification_start=time.perf_counter()
        shot_seconds={}
        for k in (1,5):
            shot_started=time.perf_counter()
            write_json(out/'status.json',dict(state='running',run_token=run_token,phase='classification',backbone=name,shot=k))
            case_out=name_out/f'{k}shot';case_out.mkdir()
            with (baseline/name/'cosine'/f'{k}shot/predictions.jsonl').open() as stream:
                old_predictions=[json.loads(line) for line in itertools.islice(stream,count)]
            rows={rep:[] for rep in REPS};max_global_difference=0.
            with (case_out/'predictions.jsonl').open('w') as stream,torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
                labels=torch.arange(5,device=device).repeat_interleave(5)
                for episode,batch in enumerate(plans[k]['batches'][:count]):
                    g,s,q=episode_inputs(batch,k,row_for_index,local,original_global,device)
                    scores=representation_scores(g,s,q,5,k,args.local_k,args.query_chunk,args.calibration_eps)
                    old=old_predictions[episode]
                    for i,member in enumerate(old['queries']):
                        if member['dataset_index']!=batch[5*k+i] or member['image_id']!=catalog[batch[5*k+i]]['image_id'] or member['episode_label']!=i//5:raise ValueError('Global baseline query mapping mismatch')
                    old_logits=torch.tensor([r['logits'] for r in old['queries']],device=device,dtype=torch.float32)
                    torch.testing.assert_close(scores['global_scores'],old_logits,rtol=1e-5,atol=2e-6)
                    old_pred=torch.tensor([r['predicted_label'] for r in old['queries']],device=device)
                    if not torch.equal(scores['global_scores'].argmax(-1),old_pred):raise ValueError('Global CUDA prediction differs from reused baseline')
                    max_global_difference=max(max_global_difference,float((old_logits-scores['global_scores']).abs().max()))
                    record=dict(episode_id=episode,class_mapping=[catalog[batch[i*k]]['class_name'] for i in range(5)],
                        support_ids=[catalog[i]['image_id'] for i in batch[:5*k]],query_ids=[catalog[i]['image_id'] for i in batch[5*k:]],
                        true_labels=labels.cpu().tolist(),score_device=str(device),methods={})
                    for rep in REPS:
                        z=scores[f'{rep}_scores'];pred=z.argmax(-1);prob=z.softmax(-1)
                        correct=int((pred==labels).sum());loss=float(F.cross_entropy(z,labels))
                        rows[rep].append(dict(episode_id=episode,correct=correct,queries=25,accuracy=correct/25,cross_entropy=loss))
                        record['methods'][rep]=dict(logits=z.cpu().tolist(),probabilities=prob.cpu().tolist(),predictions=pred.cpu().tolist())
                    record['global_calibrated']=scores['global_calibrated'].cpu().tolist()
                    record['local_calibrated']=scores['local_calibrated'].cpu().tolist()
                    stream.write(json.dumps(record)+'\n')
                    if episode==0:
                        local_diagnostics(case_out,k,g,s,q,scores,device,args.local_k)
                        if args.mode=='smoke':reference_checks[name][str(k)]=cpu_reference(model,dataset,batch,k,original_global,s,q,scores,device,args)
                    if (episode+1)%100==0:logger.info('%s %d-shot %d/%d episodes scored on %s',name,k,episode+1,count,device)
            for rep in REPS:
                write_rows(case_out/f'{rep}_episodes.csv',rows[rep])
                accuracy=[r['accuracy'] for r in rows[rep]]
                summary=accuracy_summary(accuracy)
                if rep=='global' and args.mode=='full':
                    old_summary=read(baseline/name/'cosine'/f'{k}shot/metrics.json')
                    if abs(summary['mean_accuracy']-old_summary['mean_accuracy'])>1e-12:raise ValueError('Reused Global accuracy differs')
                    summary=old_summary
                results[name][rep][k]=dict(summary=summary,accuracies=accuracy)
            reference_checks[name][f'{k}shot_global_reference_max_abs']=max_global_difference
            if device.type=='cuda':torch.cuda.synchronize(device)
            shot_seconds[str(k)]=time.perf_counter()-shot_started
            logger.info('%s %d-shot: Global=%.4f Local=%.4f Fusion=%.4f',name,k,*[results[name][r][k]['summary']['mean_accuracy'] for r in REPS])
        if device.type=='cuda':torch.cuda.synchronize(device)
        performance[name]=dict(feature_extraction=cache,classification_seconds=time.perf_counter()-classification_start,
            forward_probe_seconds=probe_seconds,shot_seconds=shot_seconds,model_total_seconds=time.perf_counter()-model_started,
            cuda_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
            cuda_peak_reserved_mb=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None)
        write_json(out/'performance.json',performance)
        if before!=state_hash(model) or digest!=sha(path):raise ValueError('Encoder or weight file changed')
        bindings[name]=digest;del model,local,original_global
        if device.type=='cuda':torch.cuda.empty_cache()
    if sources!={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').rglob('*.py'))}:raise ValueError('Source changed while running')
    for path,expected in protected.items():
        if folder_hashes(Path(path))!=expected:raise ValueError('Previous experiment files changed')
    checks=dict(status='PASS',mode=args.mode,device=str(device),device_name=device_name,
        method_signature=signature,checkpoint_sha256=bindings,episodes_per_shot=count,
        old_results_unchanged=True,weights_unchanged=True,same_episode_ids_and_mapping=True,
        local_shape=[512,7,7],global_shape=[512],score_device=str(device),
        cpu_reference=reference_checks,training=False,seconds=time.perf_counter()-started)
    checks.update(total_method_evaluations=len(names)*count*2*3,
        reused_global_evaluations=len(names)*count*2,new_local_fusion_evaluations=len(names)*count*2*2,
        data_and_plan_validation_seconds=validation_seconds)
    write_json(out/'checks.json',checks);write_json(out/'performance.json',performance)
    write_report(out,results,checks,config)
    checks['seconds']=time.perf_counter()-started
    write_json(out/'checks.json',checks)
    write_json(out/'status.json',dict(state='completed',run_token=run_token,seconds=checks['seconds']))
    logger.info('PASS: %s validation on %s (%s). Report %s',args.mode,device,device_name,out/'report.md')
    return checks


def main(argv=None):
    token=uuid.uuid4().hex;arguments=list(sys.argv[1:] if argv is None else argv)
    output=None;started=time.perf_counter()
    for i,arg in enumerate(arguments):
        if arg=='--output-dir' and i+1<len(arguments):output=Path(arguments[i+1]).resolve()
        elif arg.startswith('--output-dir='):output=Path(arg.split('=',1)[1]).resolve()
    try:return _run(argv,token)
    except BaseException as error:
        # Never modify an existing/foreign run when CLI refuses its directory.
        status=output/'status.json' if output else None
        if status and status.is_file() and read(status).get('run_token')==token:
            previous=read(status)
            failure=dict(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
                phase=previous,elapsed_seconds=time.perf_counter()-started)
            write_json(output/'failure.json',failure)
            write_json(status,dict(state=failure['state'],run_token=token,phase=previous.get('phase')))
            logging.getLogger('representation_comparison').exception('Evaluation failed/interrupted')
            if (output/'checks.json').exists():
                checks=read(output/'checks.json');checks['status']='FAILED';write_json(output/'checks.json',checks)
        raise


if __name__=='__main__':main()
