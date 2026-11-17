"""Supervised CUB base-class training, with existing unseen-class few-shot validation.

Classification-head Train Accuracy and episodic Val Accuracy are distinct tasks.
No CUB test or NABirds image is used. Resume is supported at epoch boundaries.
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset, get_cub_transforms
from pattern_recognition_coursework.data.episodes import load_or_create_plan, plan_hash
from pattern_recognition_coursework.engine.experiment import run_episodes, set_seed
from pattern_recognition_coursework.models.backbones.resnet import SupervisedResNet18
from pattern_recognition_coursework.models.few_shot import FewShotModel

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = dict(epochs=50, batch_size=8, lr=0.01, momentum=0.9, weight_decay=1e-4,
                min_lr=0.0, image_size=224, seed=42, workers=0, amp=True,
                grad_clip=5.0, device='auto', n_way=5, k_shot=1, n_query=5,
                val_episodes=100, data_root=str(ROOT/'cub2002011/versions/7/CUB_200_2011'),
                splits_dir=str(ROOT/'data/splits/cub_clean_v1'),
                weights=str(ROOT/'resnet18-f37072fd.pth'), deterministic=True)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def code_version(root):
    """Record actual source hashes even when distributed without Git metadata."""
    root = Path(root)
    result = dict(head=None, files={str(f.relative_to(root)): sha(f)
                                   for f in sorted((root/'src').rglob('*.py'))})
    try:
        result['head'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.PIPE).strip()
        result['git_available'] = True
    except (OSError, subprocess.CalledProcessError) as error:
        result['git_available'] = False
        result['git_error'] = str(error)
        metadata = root/'SERVER_TRANSFER_METADATA.json'
        if metadata.is_file():
            try:
                transfer = json.loads(metadata.read_text(encoding='utf-8'))
                result['transfer_source_commit'] = transfer.get('git_commit')
                result['transfer_source_branch'] = transfer.get('git_branch')
            except (OSError, ValueError, AttributeError) as metadata_error:
                result['transfer_metadata_error'] = str(metadata_error)
    return result


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def atomic_save(value, path):
    temporary = Path(str(path)+'.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def capture_rng():
    state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=(state[0], state[1].tolist(), state[2], state[3], state[4]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, device):
    random.setstate(state['python']); torch.set_rng_state(state['torch'])
    n = state['numpy']; np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    if device.type == 'cuda' and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def dataset_audit(train, val, splits):
    """Reject overlapping classes, IDs, resolved paths and indexed image content."""
    def ids(split):
        return {line.split(maxsplit=1)[0] for line in (splits/f'{split}_images.txt').read_text().splitlines() if line.strip()}
    if set(train.classes) & set(val.classes) or ids('train') & ids('val'):
        raise ValueError('Train/val classes or image IDs overlap')
    if {p for p,_ in train.samples} & {p for p,_ in val.samples}:
        raise ValueError('Train/val image paths overlap')
    with (splits/'image_hashes.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    hashes = [row['sha256'] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError('Clean index still contains exact duplicate image content')
    # Also prove actual bytes still match the clean audit, rather than trust a stale CSV.
    root = train.data_root/'images'
    for row in rows:
        if row['split'] in ('train', 'val') and sha(root/row['relative_path']) != row['sha256']:
            raise ValueError(f"Image changed since clean audit: {row['relative_path']}")
    return {'train_images':len(train), 'train_classes':len(train.classes),
            'val_images':len(val), 'val_classes':len(val.classes),
            'class_id_path_content_overlap':False, 'train_val_image_bytes_rehashed':True,
            'test_or_nabirds_images_read':False}


def make_validation_scorer(model, device):
    """Reuse existing scoring code; share the trained encoder, never its 100-class head."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        scorer = FewShotModel(method='protonet', metric='cosine')
    scorer.backbone = model.backbone
    return scorer.to(device)


def train_epoch(model, loader, optimizer, scaler, device, amp, clip, step_writer, tb, epoch, step):
    model.train()
    loss_sum = correct = samples = 0
    norms = []; skipped = 0
    loss_fn = nn.CrossEntropyLoss()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)
            loss = loss_fn(logits, labels)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite supervised loss/logits')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        params = [p for p in model.parameters() if p.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(params, clip if clip > 0 else float('inf'))
        finite = bool(torch.isfinite(norm))
        if not finite and not amp:
            raise FloatingPointError('Non-finite gradients without AMP')
        previous_scale = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        did_skip = scaler.get_scale() < previous_scale
        skipped += int(did_skip)
        if finite: norms.append(float(norm))
        count = labels.numel(); samples += count
        loss_sum += float(loss.detach())*count
        batch_correct = int((logits.detach().argmax(1)==labels).sum()); correct += batch_correct
        step += 1
        row = dict(epoch=epoch, step=step, loss=float(loss.detach()), accuracy=batch_correct/count,
                   lr=optimizer.param_groups[0]['lr'], grad_norm=float(norm) if finite else None,
                   optimizer_step_skipped=did_skip)
        step_writer.writerow(row)
        for tag in ('loss','accuracy','lr','grad_norm'):
            if row[tag] is not None: tb.add_scalar('train_step/'+tag,row[tag],step)
    return dict(loss=loss_sum/samples, accuracy=correct/samples,
                grad_norm=float(np.mean(norms)) if norms else None,
                amp_skipped_steps=skipped, samples=samples, seconds=time.perf_counter()-start,
                gpu_memory_mb=torch.cuda.memory_allocated(device)/2**20 if device.type=='cuda' else None,
                gpu_peak_memory_mb=torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None), step


def export_backbone(checkpoint, output):
    output = Path(output)
    if output.exists() or output.with_suffix(output.suffix+'.json').exists():
        raise FileExistsError('Backbone export already exists')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = {k.removeprefix('backbone.'):v for k,v in saved['model'].items() if k.startswith('backbone.')}
    if not state: raise ValueError('No encoder state in supervised checkpoint')
    output.parent.mkdir(parents=True,exist_ok=True)
    atomic_save(state, output)
    write_json(output.with_suffix(output.suffix+'.json'),dict(source=str(Path(checkpoint).resolve()),
        source_sha256=sha(checkpoint), output_sha256=sha(output), epoch=saved['epoch'],
        classifier_removed=True, validation_kind=saved['validation_kind']))
    return output


def render_report(run, history, result):
    write_json(run/'history.json', history)
    with (run/'logs/epochs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    for name,keys in [('loss',['train_loss','val_loss']),('accuracy',['train_accuracy','val_accuracy']),
                      ('learning_rate',['lr']),('gradient_norm',['grad_norm'])]:
        series=[(k,[(h['epoch'],h[k]) for h in history if h[k] is not None]) for k in keys]
        values=[float(y) for _,points in series for _,y in points]
        if not values: continue
        low=min(values);high=max(values);span=max(high-low,1e-9)
        lines=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="450"><rect width="900" height="450" fill="white"/>',
               f'<text x="75" y="25">{name}: train=100-class CE; val=unseen-class few-shot (different tasks)</text>']
        for i in range(5):
            y=350-i*65;lines.append(f'<path d="M75 {y}H820" stroke="#ddd"/><text x="5" y="{y}">{low+span*i/4:.4g}</text>')
        for i,(key,points) in enumerate(series):
            coords=' '.join(f'{75+(x-1)/max(history[-1]["epoch"]-1,1)*745},{350-(y-low)/span*260}' for x,y in points)
            color=['blue','red'][i];lines.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/><text x="75" y="{385+i*22}" fill="{color}">{key}</text>')
        lines.append('</svg>');(run/f'figures/{name}.svg').write_text(''.join(lines))
    write_json(run/'result.json',result)
    text='# Stage04监督训练结果\n\n训练使用100类监督分类头；验证复用未见类cosine原型评分。Train/Val Loss与Accuracy属于不同任务，不可当作同一个分类头的训练/验证差距。验证不使用NABirds或CUB test。\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n\n'
    text+='[配置](config.json) · [逐epoch CSV](logs/epochs.csv) · [逐step CSV](logs/steps.csv) · [运行日志](logs/run.log)\n\n'
    for name in ['loss','accuracy','learning_rate','gradient_norm']:text+=f'![{name}](figures/{name}.svg)\n\n'
    text+='Checkpoint为epoch边界恢复，包含分类头、optimizer、scheduler和AMP scaler。修改batch size恢复会改变训练轨迹。未承诺跨硬件位级一致或中途batch精确续训。\n'
    (run/'report.md').write_text(text)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True,help='New directory, including for resumed runs; never overwrite old experiments')
    p.add_argument('--resume',type=Path)
    for name in ['epochs','batch-size','image-size','seed','workers','n-way','k-shot','n-query','val-episodes']:
        p.add_argument('--'+name,type=int,default=None)
    for name in ['lr','momentum','weight-decay','min-lr','grad-clip']:
        p.add_argument('--'+name,type=float,default=None)
    for name in ['data-root','splits-dir','weights']:p.add_argument('--'+name,default=None)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--deterministic',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default=None)
    p.add_argument('--random-init',action='store_true',help='Explicit diagnostic alternative to default local ImageNet initialization')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    checkpoint=torch.load(args.resume,map_location='cpu',weights_only=True) if args.resume else None
    supplied={k:v for k,v in vars(args).items() if k in DEFAULTS and v is not None}
    c={**DEFAULTS,**(checkpoint['config'] if checkpoint else {}),**supplied}
    if args.random_init:c['weights']=None
    if checkpoint:
        mutable={'batch_size','workers','device','data_root','splits_dir'}
        if any(c[k]!=checkpoint['config'].get(k,DEFAULTS[k]) for k in DEFAULTS if k not in mutable):
            p.error('Resume must preserve epochs/schedule, seed and protocol; batch size/device/workers may change and are recorded')
        if args.random_init:p.error('--random-init cannot change a resumed initialization')
    for k in ['epochs','batch_size','image_size','n_way','k_shot','n_query','val_episodes']:
        if c[k]<1:p.error(k+' must be positive')
    if c['workers']<0 or c['seed']<0 or c['lr']<=0 or not all(math.isfinite(c[k]) for k in ['lr','momentum','weight_decay','min_lr','grad_clip']):p.error('Invalid optimizer/worker settings')
    if c['momentum']<0 or c['weight_decay']<0 or c['min_lr']<0 or c['min_lr']>c['lr'] or c['grad_clip']<0:p.error('Invalid scheduler/gradient clipping settings')
    if c['device']=='cuda' and not torch.cuda.is_available():p.error('CUDA not available; no silent CPU fallback')
    device=torch.device('cuda' if c['device']=='cuda' or c['device']=='auto' and torch.cuda.is_available() else 'cpu')
    run=args.run_dir.resolve()
    if not run.is_relative_to(ROOT):p.error('Run directory must be inside project')
    if run.exists():p.error('Run directory already exists; choose a new name')
    set_seed(c['seed']);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(c['deterministic'])
    # No downloading: default initialization must use the already supplied local file.
    if not checkpoint and c['weights'] and not Path(c['weights']).is_file():p.error('Local ImageNet weights missing; no automatic download')
    splits=Path(c['splits_dir']);data=Path(c['data_root'])
    train=CUBFewShotDataset(data,splits/'train_images.txt',get_cub_transforms(c['image_size'],True),require_clean_index=True)
    val=CUBFewShotDataset(data,splits/'val_images.txt',get_cub_transforms(c['image_size'],False),require_clean_index=True)
    audit=dataset_audit(train,val,splits)
    split_hashes={f'{s}_{kind}.txt':sha(splits/f'{s}_{kind}.txt') for s in ['train','val','test'] for kind in ['images','classes']}
    if checkpoint and (split_hashes!=checkpoint['split_hashes'] or train.class_to_idx!=checkpoint['class_to_idx']):
        raise ValueError('Resume dataset manifests/class mapping differ')
    if checkpoint and checkpoint['epoch']>=c['epochs']:p.error('Checkpoint already reached configured final epoch')
    model=SupervisedResNet18(len(train.classes),None if checkpoint else c['weights']).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=c['epochs'],eta_min=c['min_lr'])
    amp=c['amp'] and device.type=='cuda'
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    scorer=make_validation_scorer(model,device)
    start_epoch=1;step=0;history=[];best=-1.;best_epoch=0
    run.mkdir(parents=True)
    for d in ['logs','checkpoints','figures']:(run/d).mkdir()
    logger=logging.getLogger(str(run));logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(run/'logs/run.log'));logger.addHandler(logging.StreamHandler())
    writer=SummaryWriter(str(run/'logs/tensorboard'))
    try:
        plan=load_or_create_plan(run/'validation_plan.json',val,c['n_way'],c['k_shot'],c['n_query'],c['val_episodes'],c['seed']+10000)
        if checkpoint:
            if plan_hash(plan)!=checkpoint['validation_plan_hash']:raise ValueError('Resume validation tasks changed')
            model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
            if amp and checkpoint['amp_enabled']:scaler.load_state_dict(checkpoint['scaler'])
            restore_rng(checkpoint['rng'],device)
            start_epoch=checkpoint['epoch']+1;step=checkpoint['step'];history=checkpoint['history'];best=checkpoint['best_accuracy'];best_epoch=checkpoint['best_epoch']
            # If last.pth itself is the best, it is sufficient even if a crash
            # occurred between atomic last.pth and best.pth publication.
            old_best=args.resume if checkpoint['epoch']==best_epoch else args.resume.parent/'best.pth'
            best_saved=checkpoint if old_best==args.resume else torch.load(old_best,map_location='cpu',weights_only=True)
            if best_saved['epoch']!=best_epoch or best_saved['validation']['accuracy']!=best:raise ValueError('Original best checkpoint missing or inconsistent with resume state')
            shutil.copyfile(old_best,run/'checkpoints/best.pth')
        write_json(run/'config.json',c);write_json(run/'data_audit.json',audit)
        import torchvision,platform
        write_json(run/'environment.json',dict(python=sys.version,torch=torch.__version__,torchvision=torchvision.__version__,platform=platform.platform(),device=str(device),amp_effective=amp,gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None))
        write_json(run/'command.json',dict(argv=[sys.executable,*sys.argv] if argv is None else argv,resume=str(args.resume) if args.resume else None,validation_kind='unseen-class episodic cosine prototypes',train_kind='supervised class-head CrossEntropy',class_to_idx=train.class_to_idx,initialization_info=model.backbone.backbone_load_info,initial_weights_sha256=sha(c['weights']) if c['weights'] and Path(c['weights']).exists() else None))
        write_json(run/'code_version.json',code_version(ROOT))
        with (run/'logs/steps.csv').open('w',newline='') as stream:
            sw=csv.DictWriter(stream,fieldnames=['epoch','step','loss','accuracy','lr','grad_norm','optimizer_step_skipped']);sw.writeheader()
            for epoch in range(start_epoch,c['epochs']+1):
                set_seed(c['seed']+epoch)
                generator=torch.Generator().manual_seed(c['seed']+epoch)
                loader=DataLoader(train,batch_size=c['batch_size'],shuffle=True,generator=generator,num_workers=c['workers'],pin_memory=device.type=='cuda')
                used_lr=optimizer.param_groups[0]['lr']
                tm,step=train_epoch(model,loader,optimizer,scaler,device,amp,c['grad_clip'],sw,writer,epoch,step);stream.flush()
                vm,_=run_episodes(scorer,DataLoader(val,batch_sampler=plan['batches'],num_workers=c['workers']),c['n_way'],c['k_shot'],c['n_query'],device)
                scheduler.step()
                row=dict(epoch=epoch,train_loss=tm['loss'],train_accuracy=tm['accuracy'],val_loss=vm['loss'],val_accuracy=vm['accuracy'],lr=used_lr,next_lr=optimizer.param_groups[0]['lr'],grad_norm=tm['grad_norm'],amp_skipped_steps=tm['amp_skipped_steps'],gpu_memory_mb=tm['gpu_memory_mb'],gpu_peak_memory_mb=tm['gpu_peak_memory_mb'],train_samples=tm['samples'],seconds=tm['seconds'])
                history.append(row);improved=vm['accuracy']>best
                if improved:best=vm['accuracy'];best_epoch=epoch
                state=dict(schema='supervised-cub-v1',model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),amp_enabled=amp,epoch=epoch,step=step,config=c,num_classes=len(train.classes),class_to_idx=train.class_to_idx,validation=vm,validation_kind='unseen-class episodic cosine prototypes',validation_plan_hash=plan_hash(plan),split_hashes=split_hashes,rng=capture_rng(),history=history,best_accuracy=best,best_epoch=best_epoch)
                atomic_save(state,run/'checkpoints/last.pth')
                if improved:atomic_save(state,run/'checkpoints/best.pth')
                for key in ['train_loss','train_accuracy','val_loss','val_accuracy','lr','grad_norm','gpu_memory_mb','gpu_peak_memory_mb']:
                    if row[key] is not None:writer.add_scalar(key,row[key],epoch)
                writer.flush();logger.info(json.dumps(row))
                render_report(run,history,dict(status='running',best_epoch=best_epoch,best_validation_accuracy=best))
        saved=torch.load(run/'checkpoints/best.pth',map_location='cpu',weights_only=True)
        # Reuse model allocation: avoid a second ResNet+SGD on the MX450.
        model.load_state_dict(saved['model'],strict=True)
        repeated,_=run_episodes(scorer,DataLoader(val,batch_sampler=plan['batches'],num_workers=0),c['n_way'],c['k_shot'],c['n_query'],device)
        for key in ['loss','accuracy','nll','macro_f1','ece']:
            if not math.isclose(repeated[key],saved['validation'][key],rel_tol=1e-6,abs_tol=1e-7):raise ValueError('Best checkpoint revalidation differs')
        export_backbone(run/'checkpoints/best.pth',run/'checkpoints/backbone_best.pth')
        result=dict(status='completed',best_epoch=best_epoch,best_validation_accuracy=best,best_validation=repeated,best_reload_validation_reproduced=True,best_checkpoint=str(run/'checkpoints/best.pth'),last_checkpoint=str(run/'checkpoints/last.pth'),backbone_export=str(run/'checkpoints/backbone_best.pth'),epochs_completed=c['epochs'],validation_kind=saved['validation_kind'],train_kind='supervised 100-base-class head (actual count in config)',amp_effective=amp)
        render_report(run,history,result)
        return result
    except BaseException as exc:
        logger.exception('Supervised training failed/interrupted')
        write_json(run/'failure.json',dict(error=repr(exc),last_completed_epoch=history[-1]['epoch'] if history else None,note='No automatic OOM retry or batch-size change. Resume last.pth into a new directory.'))
        raise
    finally:
        writer.flush();writer.close()
        for handler in list(logger.handlers):handler.close();logger.removeHandler(handler)


def export_main(argv=None):
    p=argparse.ArgumentParser(description='Remove supervised head and export encoder-only state_dict')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv);print(export_backbone(a.checkpoint,a.output))


if __name__=='__main__':main()
