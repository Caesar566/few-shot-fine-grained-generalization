"""Stage06: frozen CUDA-first NABirds evaluation under explicit automatic screening."""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
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

from pattern_recognition_coursework.analysis.episodic_statistics import accuracy_summary
from pattern_recognition_coursework.data.nabirds_automatic import AutomaticNABirdsDataset
from pattern_recognition_coursework.data.cub_dataset import get_cub_transforms
from pattern_recognition_coursework.data.paired_episodes import validate_paired_plans
from pattern_recognition_coursework.data.episodes import plan_hash
from pattern_recognition_coursework.engine.episodic_evaluation import ROOT,RUNS,sha,state_hash,tensor_hash,canonical_hash,write_json
from pattern_recognition_coursework.engine.distance_comparison import folder_hashes,write_rows
from pattern_recognition_coursework.engine.representation_comparison import (
    choose_device,collect_local,episode_inputs,local_diagnostics,write_report,
)
from pattern_recognition_coursework.engine.experiment import set_seed
from pattern_recognition_coursework.models.few_shot import FewShotModel
from pattern_recognition_coursework.models.metrics.representation import representation_scores

REPS=('global','local','fusion')


def read(path):return json.loads(Path(path).read_text())


def stage06_report(out,results,checks,config,manifest,cub_reference):
    # Reuse the already verified tables, paired Student-t statistics and figures.
    write_report(out,results,checks,config)
    cub=read(cub_reference/'summary.json');comparison=[]
    for name,methods in results.items():
        for rep in REPS:
            for k,case in methods[rep].items():
                target=case['summary'];source=cub[name][rep][str(k)]
                comparison.append(dict(initialization=name,representation=rep,shot=k,
                    cub_val_mean_percent=source['mean_accuracy']*100,
                    cub_val_sample_std_pp=source['sample_std']*100,cub_val_ci95_low_percent=source['ci95'][0]*100,cub_val_ci95_high_percent=source['ci95'][1]*100,
                    nabirds_mean_percent=target['mean_accuracy']*100,nabirds_sample_std_pp=target['sample_std']*100,
                    nabirds_ci95_low_percent=target['ci95'][0]*100,nabirds_ci95_high_percent=target['ci95'][1]*100,
                    nabirds_minus_cub_pp=(target['mean_accuracy']-source['mean_accuracy'])*100,
                    difference_type='descriptive unpaired cross-dataset mean difference'))
    write_rows(out/'cub_vs_nabirds.csv',comparison)
    rows=list(csv.DictReader((out/'comparison.csv').open()))
    lines=['# Stage 06：NABirds 冻结跨数据集 Few-shot 测试','',
        '**本结果采用用户明确选择的自动保守物种筛选协议，不声称人工审核通过。** '
        '不训练、不选模型、不搜索权重、不根据NABirds成绩修改任何超参数。','',
        f'模式：**{config["mode"]}**；每种Shot使用冻结计划前{config["episodes"]}个Episode；实际设备：**{checks["device_name"]}**。',
        'smoke只用于功能检查，不能当作600任务结果。' if config['mode']=='smoke' else
        '**三种初始化×三种表示×两种Shot×600，共10800次方法评分、270000条Query方法预测，均为本次NABirds新结果。**', '',
        '## 测试池与语义审核边界','',
        f'- 原始NABirds：{manifest["raw_statistics"]["images"]}图像、{manifest["raw_statistics"]["visual_categories"]}视觉类别、{manifest["raw_statistics"]["hierarchy_nodes"]}层级节点。',
        f'- 只使用官方test split(value=0)，合并性别/年龄/外观子类后自动筛选为**{len(manifest["classes"])}个物种、{len(manifest["samples"])}张图像**；物种最少{min(c["image_count"] for c in manifest["classes"])}张，满足5 Support+5 Query。',
        '- 使用固定Cornell eBird taxonomy v2026快照，以规范英文名、所有格拼写、科学名/属及层级作保守筛选。未知或冲突的目标物种整类排除，不能用“没有字符串匹配”冒充确认不重叠。',
        '- CUB Train 100类全部解析为可追溯排除集合；Sayornis、Geococcyx等属级标签排除全属，Great Grey Shrike等历史拆分类名取保守超集，Myrtle Warbler映射并排除Yellow-rumped物种整体。该规则在第一次NABirds前向之前冻结。',
        '- 自动检查不能替代专家物种鉴定；同义名或历史分类残余不确定性仍是限制。`manifest.json`明确human_reviewed=false，原人工审核加载器未被绕过或伪造审批。',
        '- 所有候选实际图片SHA256已核对；派生清单排除与任意CUB split的内容重叠和目标跨类重复。同类重复仅逻辑保留确定性一个文件，不移动/删除任何原图。',
        '- 本协议只要求相对于CUB Train的物种不重叠；不宣称与ImageNet外部预训练物种完全不重叠，也不宣称所有目标物种均未出现在CUB开发验证类别中。','',
        '## 完全冻结的模型与任务','',
        '- 直接加载Stage04三份best.pth，与Stage05已完成表示实验的Checkpoint SHA256逐项一致，eval+inference_mode、FP32，无优化器，无反向传播。',
        '- 三组Backbone、Global/Local/Fusion共用同一NABirds计划。每种600个5-way任务，每类5 Query；1-shot/5-shot类序、Query完全相同，1-shot Support是5-shot子集。',
        '- 原始GAP512维求均值Prototype再算Cosine；Layer4 49×512描述子使用原DN4函数，类内K×49候选、逐Query位置Top-3，再对49×3项求均值。',
        '- Fusion保留先前固定的每Query五类分数行内population z-score，eps=1e-6，再按0.5/0.5融合；不利用真实Query标签调参。',
        '- 复用同一评分函数及其源码哈希；RGB→Resize257→CenterCrop224→ToTensor→ImageNet像素Normalize，特征与Prototype不做额外预归一化。',
        '- CUDA优先，完整CPU特征缓存按目标manifest、图片ID、权重、模式、预处理与软件版本隔离；只上传Episode Support和Query块，默认Batch8/chunk4，关闭AMP/TF32。','',
        '## NABirds 完整比较','',
        '| 初始化 | 表示 | Shot | Mean Accuracy | 样本标准差（pp） | 95% CI |','|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f'| {r["initialization"]} | {r["representation"]} | {r["shot"]} | {float(r["mean_accuracy_percent"]):.3f}% | {float(r["sample_std_pp"]):.3f} | [{float(r["ci95_low_percent"]):.3f}%, {float(r["ci95_high_percent"]):.3f}%] |')
    lines += ['', '## 与CUB Val并列','',
        '| 初始化 | 表示 | Shot | CUB Val | NABirds | NABirds−CUB（pp） |','|---|---|---:|---:|---:|---:|']
    for r in comparison:
        lines.append(f'| {r["initialization"]} | {r["representation"]} | {r["shot"]} | {r["cub_val_mean_percent"]:.3f}% | {r["nabirds_mean_percent"]:.3f}% | {r["nabirds_minus_cub_pp"]:+.3f} |')
    lines += ['', f'跨数据集差值仅为描述性均值差。CUB Val是50类、NABirds是自动筛选的{len(manifest["classes"])}类，两个数据集的任务不同，不能按同一个episode_id把跨数据集结果当作配对样本，也不能将差值全归因为纯视觉域偏移。配对统计仅用于同一NABirds任务内三种表示的比较。', '',
        '## 配对统计、性能和文件','',
        '`paired_comparison.csv`报告Local−Global、Fusion−Global、Fusion−Local的均值差及基于600个配对差的Student-t 95% CI，单位百分点；统计描述性，未做多重比较校正。Mean与ddof=1样本标准差、Student-t均值CI沿用Stage05。', '',
        '![NABirds准确率](accuracy_bars.svg)','', '![NABirds配对差值](paired_lifts.svg)','',
        '[对比表](comparison.csv) · [CUB并列表](cub_vs_nabirds.csv) · [配对差值](paired_comparison.csv) · [逐Episode差值](paired_episode_differences.csv) · [配置](config.json) · [验收](checks.json) · [性能](performance.json) · [日志](run.log)', '',
        '每模型/Shot的`predictions.jsonl`保存样本ID、类别映射、Support/Query、三路logits/概率/预测及校准分数；对应每路逐Episode Accuracy另存CSV。`protocol/`保留本次manifest与600计划副本，Stage04/05旧文件不覆盖。', '',
        f'总评估耗时：{checks["seconds"]:.2f}秒。GPU峰值allocated/reserved、提取与分类耗时、完整/部分缓存命中见performance.json；PyTorch峰值不包括驱动上下文或其他进程。', '',
        '## 分类依据来源','',
        '[NABirds官方介绍](https://dl.allaboutbirds.org/nabirds)；[Cornell eBird taxonomy说明](https://support.ebird.org/en/support/solutions/articles/48000837816-the-ebird-taxonomy)；[固定v2026分类CSV](https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&locale=en&version=2026)。实际CSV、版本列表和SHA256保留在Stage06 preflight目录。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def _run(argv,token):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke');p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--protocol-dir',type=Path,default=ROOT/'outputs/stage06_nabirds/protocol')
    p.add_argument('--data-root',type=Path,default=ROOT/'nabirds')
    p.add_argument('--taxonomy',type=Path,default=ROOT/'outputs/stage06_nabirds/preflight/ebird_taxonomy_v2026.csv')
    p.add_argument('--cub-reference',type=Path,default=ROOT/'outputs/stage05_protonet/representation_full_seed42_600_cuda')
    p.add_argument('--cache-dir',type=Path,default=ROOT/'data/cache/stage06_nabirds_features')
    p.add_argument('--allow-automatic-screening',action='store_true');p.add_argument('--validation-report',type=Path)
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--query-chunk',type=int,default=4);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--smoke-episodes',type=int,default=3);p.add_argument('--threads',type=int,default=2)
    for name,run in RUNS.items():p.add_argument(f'--{name}-checkpoint',type=Path,default=ROOT/'outputs/stage04_training'/run/'checkpoints/best.pth')
    args=p.parse_args(argv)
    if not args.allow_automatic_screening:p.error('This explicitly automatic protocol requires --allow-automatic-screening')
    if min(args.batch_size,args.query_chunk,args.smoke_episodes,args.threads)<1 or args.smoke_episodes<2 or args.workers<0:p.error('Invalid counts')
    out=args.output_dir.resolve();protocol=args.protocol_dir.resolve();cub=args.cub_reference.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'outputs/stage06_nabirds') or out.is_relative_to(protocol):p.error('Use a new Stage06 output directory outside the frozen protocol')
    cub_config=read(cub/'config.json');cub_checks=read(cub/'checks.json')
    signature=cub_config['method_signature']
    if cub_config['mode']!='full' or cub_checks['status']!='PASS':p.error('Completed CUB representation reference required')
    for f,digest in signature['implementation_sha256'].items():
        if sha(ROOT/f)!=digest:p.error('Scoring implementation differs from frozen CUB config')
    if signature['fusion_weight']!=.5 or signature['local_k']!=3 or signature['calibration_eps']!=1e-6:p.error('Unexpected frozen method')
    manifest_path=protocol/'manifest.json';manifest_digest=sha(manifest_path)
    if args.mode=='full':
        if not args.validation_report:p.error('Run Stage06 smoke and pass --validation-report')
        gate=read(args.validation_report)
        if gate['status']!='PASS' or gate['mode']!='smoke' or gate['method_signature']!=signature or gate['manifest_sha256']!=manifest_digest:p.error('NABirds smoke gate differs or failed')
        if gate['checkpoint_sha256']['imagenet']!=cub_checks['checkpoint_sha256']['imagenet']:p.error('NABirds smoke used different weights')
    count=600 if args.mode=='full' else args.smoke_episodes
    if count>600:p.error('Smoke exceeds frozen plan')
    names=list(RUNS) if args.mode=='full' else ['imagenet']
    out.mkdir(parents=True);write_json(out/'status.json',dict(state='running',run_token=token,phase='audit'))
    logger=logging.getLogger('nabirds_transfer');logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(out/'run.log'));logger.addHandler(logging.StreamHandler())
    device=choose_device(args.device,logger);device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU'
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True);set_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    logger.info('Stage06 %s ACTUAL device=%s (%s), FP32, Batch=%d chunk=%d',args.mode,device,device_name,args.batch_size,args.query_chunk)
    started=time.perf_counter();protected=folder_hashes(cub);protocol_before=folder_hashes(protocol)
    sources={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').rglob('*.py'))}
    dataset=AutomaticNABirdsDataset(args.data_root,manifest_path,ROOT/'data/splits/cub_clean_v1/train_classes.txt',args.taxonomy,
                                   get_cub_transforms(224,False),allow_automatic=True)
    catalog=[]
    for i,((path,label),row) in enumerate(zip(dataset.samples,dataset.records)):
        if sha(path)!=row['sha256']:raise ValueError('Target image bytes changed since pool was frozen')
        catalog.append(dict(dataset_index=i,image_id=row['image_id'],dataset_label=label,class_name=dataset.classes[label],
            relative_path=row['path'],sha256=row['sha256'],species_code=row['class_id'],visual_class_id=row['visual_class_id'],official_split=0))
    if canonical_hash(catalog)!=read(protocol/'audit_summary.json')['dataset_content_sha256']:raise ValueError('Target catalog differs')
    plans={k:read(protocol/f'plan_{k}shot.json') for k in (1,5)};validate_paired_plans(dataset,plans[1],plans[5])
    if any(plan_hash(plans[k])!=read(protocol/'plan_checks.json')['both_plan_hashes'][str(k)] for k in (1,5)):raise ValueError('Frozen NABirds plans changed')
    if args.mode=='full' and gate['plan_hashes']!={str(k):plan_hash(v) for k,v in plans.items()}:raise ValueError('NABirds plans differ from smoke')
    shutil.copytree(protocol,out/'protocol')
    selected=sorted({i for k in (1,5) for batch in plans[k]['batches'][:count] for i in batch})
    positions={i:row for row,i in enumerate(selected)}
    config=dict(mode=args.mode,episodes=count,backbones=names,seed=42,device=str(device),device_name=device_name,
        requested_device=args.device,method_signature=signature,manifest_sha256=manifest_digest,
        target_protocol=dataset.manifest_metadata['protocol'],automatic_screening=True,human_reviewed=False,
        image_split='official_test_only',target_species=len(dataset.classes),target_images=len(dataset),
        feature_batch_size=args.batch_size,query_chunk=args.query_chunk,workers=args.workers,
        used_episode_ids=list(range(count)),plan_hashes={str(k):plan_hash(v) for k,v in plans.items()},
        cub_reference=str(cub),taxonomy_sha256=sha(args.taxonomy),validation_report=str(args.validation_report) if args.validation_report else None)
    write_json(out/'config.json',config);write_json(out/'command.json',dict(argv=sys.argv if argv is None else argv))
    write_json(out/'source_sha256.json',sources)
    results={};performance={};digests={}
    for name in names:
        begin=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        write_json(out/'status.json',dict(state='running',run_token=token,phase='feature_extraction',backbone=name))
        path=getattr(args,f'{name}_checkpoint');digest=sha(path)
        if digest!=cub_checks['checkpoint_sha256'][name]:raise ValueError('Checkpoint differs from frozen CUB experiment')
        model=FewShotModel(method='global_local',metric='cosine',backbone_weights=path,backbone_prefix='backbone.').to(device).eval().requires_grad_(False)
        before=state_hash(model)
        binding=dict(schema='stage06-raw-layer4-gap-v1',dataset='nabirds',manifest_sha256=manifest_digest,
            dataset_content_sha256=read(protocol/'audit_summary.json')['dataset_content_sha256'],
            checkpoint_sha256=digest,encoder_state_sha256=before,sample_indices=selected,ordered_sample_ids=[catalog[i]['image_id'] for i in selected],
            preprocessing=repr(dataset.transform),mode='eval',frozen=True,dtype='float32',shape=[512,7,7],device=str(device),
            device_name=device_name,batch_size=args.batch_size,torch=str(torch.__version__),tf32=False,
            versions={n:importlib.metadata.version(n) for n in ['torchvision','Pillow','numpy']},
            feature_source_sha256={f:h for f,h in sources.items() if f.endswith(('backbones/resnet.py','data/cub_dataset.py','models/few_shot.py','data/nabirds_automatic.py'))})
        name_out=out/name;name_out.mkdir();logger.info('%s extracting needed %d/%d NABirds images',name,len(selected),len(dataset))
        local,cache=collect_local(model,dataset,selected,None,binding,args.cache_dir.resolve(),args.batch_size,args.workers,device,logger)
        payload=torch.load(cache['path'],map_location='cpu',weights_only=True,mmap=True)
        gap=payload['gap'];assert tensor_hash(gap)==cache['gap_sha256']
        global_features=torch.zeros(len(dataset),512,dtype=torch.float32);global_features[selected]=gap
        with torch.inference_mode():
            probe_images=torch.stack([dataset[i][0] for i in selected[:min(args.batch_size,len(selected))]]).to(device)
            maps=model.encode(probe_images);g=model.backbone(probe_images)
            torch.testing.assert_close(g,model.backbone.avgpool(maps).flatten(1),rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(g.cpu(),global_features[selected[:len(g)]],rtol=1e-4,atol=5e-5)
            cache['forward_probe']=dict(device=str(device),images=list(probe_images.shape),layer4=list(maps.shape),gap=list(g.shape))
            del probe_images,maps,g
        write_json(name_out/'cache.json',cache)
        results[name]={rep:{} for rep in REPS};classification_start=time.perf_counter();shot_seconds={}
        for k in (1,5):
            shot_start=time.perf_counter();case_out=name_out/f'{k}shot';case_out.mkdir()
            write_json(out/'status.json',dict(state='running',run_token=token,phase='classification',backbone=name,shot=k))
            rows={rep:[] for rep in REPS}
            with (case_out/'predictions.jsonl').open('w') as stream,torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
                labels=torch.arange(5,device=device).repeat_interleave(5)
                for episode,batch in enumerate(plans[k]['batches'][:count]):
                    if len({catalog[i]['sha256'] for i in batch})!=len(batch):raise ValueError('Episode support/query content duplicates')
                    g,s,q=episode_inputs(batch,k,positions,local,global_features,device)
                    scores=representation_scores(g,s,q,5,k,signature['local_k'],args.query_chunk,signature['calibration_eps'])
                    record=dict(episode_id=episode,class_mapping=[dict(episode_label=j,dataset_label=dataset.samples[batch[j*k]][1],
                        species_code=dataset.species_codes[dataset.samples[batch[j*k]][1]],name=dataset.classes[dataset.samples[batch[j*k]][1]]) for j in range(5)],
                        support_ids=[catalog[i]['image_id'] for i in batch[:5*k]],query_ids=[catalog[i]['image_id'] for i in batch[5*k:]],
                        true_labels=labels.cpu().tolist(),score_device=str(device),methods={})
                    for rep in REPS:
                        z=scores[f'{rep}_scores'];pred=z.argmax(-1);prob=z.softmax(-1)
                        correct=int((pred==labels).sum());loss=float(F.cross_entropy(z,labels))
                        rows[rep].append(dict(episode_id=episode,correct=correct,queries=25,accuracy=correct/25,cross_entropy=loss))
                        record['methods'][rep]=dict(logits=z.cpu().tolist(),probabilities=prob.cpu().tolist(),predictions=pred.cpu().tolist())
                    record['global_calibrated']=scores['global_calibrated'].cpu().tolist();record['local_calibrated']=scores['local_calibrated'].cpu().tolist()
                    stream.write(json.dumps(record)+'\n')
                    if episode==0:local_diagnostics(case_out,k,g,s,q,scores,device,signature['local_k'])
                    if (episode+1)%100==0:logger.info('%s %d-shot %d/%d scored on %s',name,k,episode+1,count,device)
            for rep in REPS:
                write_rows(case_out/f'{rep}_episodes.csv',rows[rep]);accuracies=[r['accuracy'] for r in rows[rep]]
                results[name][rep][k]=dict(summary=accuracy_summary(accuracies),accuracies=accuracies)
            if device.type=='cuda':torch.cuda.synchronize(device)
            shot_seconds[str(k)]=time.perf_counter()-shot_start
            logger.info('%s %d-shot NABirds accuracy Global=%.4f Local=%.4f Fusion=%.4f',name,k,*[results[name][rep][k]['summary']['mean_accuracy'] for rep in REPS])
        if state_hash(model)!=before or sha(path)!=digest:raise ValueError('Backbone weights or BN state changed')
        performance[name]=dict(feature_cache=cache,classification_seconds=time.perf_counter()-classification_start,
            shot_seconds=shot_seconds,total_seconds=time.perf_counter()-begin,
            cuda_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,
            cuda_peak_reserved_mb=torch.cuda.max_memory_reserved(device)/2**20 if device.type=='cuda' else None)
        write_json(out/'performance.json',performance);digests[name]=digest
        del model,local,payload,gap,global_features
        if device.type=='cuda':torch.cuda.empty_cache()
    if folder_hashes(cub)!=protected or folder_hashes(protocol)!=protocol_before:raise ValueError('Frozen input results/protocol changed')
    if sources!={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'src').rglob('*.py'))}:raise ValueError('Source changed while evaluating')
    checks=dict(status='PASS',mode=args.mode,device=str(device),device_name=device_name,method_signature=signature,
        manifest_sha256=manifest_digest,automatic_screening=True,human_reviewed=False,episodes_per_shot=count,
        plan_hashes=config['plan_hashes'],
        checkpoint_sha256=digests,weights_unchanged=True,old_results_unchanged=True,same_episodes_all_methods=True,
        support_query_content_disjoint=True,protocol_unchanged=True,covariance_or_parameter_fitting=False,
        total_method_evaluations=len(names)*2*count*3,total_query_predictions=len(names)*2*count*3*25,
        training=False,seconds=time.perf_counter()-started)
    write_json(out/'checks.json',checks);stage06_report(out,results,checks,config,dataset.manifest_metadata,cub)
    checks['seconds']=time.perf_counter()-started;write_json(out/'checks.json',checks)
    write_json(out/'status.json',dict(state='completed',run_token=token,seconds=checks['seconds']))
    logger.info('PASS Stage06 %s. Report: %s',args.mode,out/'report.md')
    return checks


def main(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv);token=uuid.uuid4().hex;output=None
    for i,value in enumerate(arguments):
        if value=='--output-dir' and i+1<len(arguments):output=Path(arguments[i+1]).resolve()
        elif value.startswith('--output-dir='):output=Path(value.split('=',1)[1]).resolve()
    try:return _run(argv,token)
    except BaseException as error:
        if output and (output/'status.json').is_file() and read(output/'status.json').get('run_token')==token:
            status=read(output/'status.json')
            write_json(output/'failure.json',dict(exception_type=type(error).__name__,message=str(error),phase=status,traceback=traceback.format_exc()))
            write_json(output/'status.json',dict(state='failed',run_token=token,phase=status.get('phase')))
            logging.getLogger('nabirds_transfer').exception('Stage06 failed/interrupted')
            if (output/'checks.json').exists():
                checks=read(output/'checks.json');checks['status']='FAILED';write_json(output/'checks.json',checks)
        raise


if __name__=='__main__':main()
