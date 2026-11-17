"""Tiny real-code training/export/resume regression, never full CUB training."""
import csv
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pattern_recognition_coursework.data.cub_dedup import audit_and_clean
from pattern_recognition_coursework.models.backbones.resnet import make_backbone, SupervisedResNet18
from pattern_recognition_coursework.engine.supervised import main, export_backbone, train_epoch


@pytest.fixture
def clean_data(tmp_path):
    raw=tmp_path/'raw';splits=tmp_path/'splits';splits.mkdir();raw.mkdir()
    cr=[];ir=[];lr=[]
    for split,cids in [('train',[1,2]),('val',[3,4]),('test',[5,6])]:
        crows=[];irows=[]
        for cid in cids:
            name=f'{cid:03d}.Bird_{cid}';(raw/'images'/name).mkdir(parents=True)
            cr.append(f'{cid} {name}\n');crows.append(f'{cid} {name}\n')
            for i in range(3):
                image_id=cid*10+i;relative=f'{name}/{i}.png'
                Image.new('RGB',(72,68),(cid*30,i*65,11)).save(raw/'images'/relative)
                ir.append(f'{image_id} {relative}\n');lr.append(f'{image_id} {cid}\n')
                irows.append(f'{image_id} {name} {relative}\n')
        (splits/f'{split}_classes.txt').write_text(''.join(crows));(splits/f'{split}_images.txt').write_text(''.join(irows))
    for name,rows in [('classes.txt',cr),('images.txt',ir),('image_class_labels.txt',lr)]:
        (raw/name).write_text(''.join(rows))
    clean=tmp_path/'clean';audit_and_clean(raw,splits,clean,tmp_path/'audit')
    return raw,clean


def arguments(clean_data,run):
    raw,splits=clean_data
    return ['--data-root',str(raw),'--splits-dir',str(splits),'--run-dir',str(run),
            '--random-init','--epochs','2','--batch-size','2','--image-size','64',
            '--n-way','2','--k-shot','1','--n-query','1','--val-episodes','2',
            '--lr','0.01','--workers','0','--device','cpu']


def test_supervised_full_cycle_and_epoch_boundary_resume(clean_data,tmp_path):
    torch.use_deterministic_algorithms(True)
    # Local tests may use /tmp, while the production CLI restricts outputs to project.
    import pattern_recognition_coursework.engine.supervised as module
    with patch.object(module,'ROOT',tmp_path):
        continuous=tmp_path/'continuous';args=arguments(clean_data,continuous)
        complete=main(args)
        assert complete['status']=='completed' and complete['best_reload_validation_reproduced']
        assert 'unseen-class' in complete['validation_kind']
        last=torch.load(continuous/'checkpoints/last.pth',map_location='cpu',weights_only=True)
        assert last['epoch']==2 and last['scheduler']['last_epoch']==2
        assert {'classifier.weight','classifier.bias'}<=last['model'].keys()
        assert last['optimizer']['state'] and 'scaler' in last
        with (continuous/'logs/epochs.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)==2 and float(rows[1]['lr'])<float(rows[0]['lr'])
        assert all(int(r['train_samples'])==6 for r in rows)
        events=EventAccumulator(str(continuous/'logs/tensorboard')).Reload()
        assert {'train_loss','val_loss','train_accuracy','val_accuracy','lr','grad_norm'}<=set(events.Tags()['scalars'])
        exported=make_backbone(backbone_weights=continuous/'checkpoints/backbone_best.pth')
        assert not any(k.startswith('classifier') for k in exported.state_dict())
        # Interrupt only after epoch 1 checkpoint is fully saved; do not change T_max.
        interrupted=tmp_path/'interrupted';original=module.render_report
        def stop_after_epoch_one(run,history,result):
            original(run,history,result)
            if history[-1]['epoch']==1:raise KeyboardInterrupt('synthetic epoch-boundary interruption')
        with patch.object(module,'render_report',side_effect=stop_after_epoch_one):
            with pytest.raises(KeyboardInterrupt):main(arguments(clean_data,interrupted))
        resumed=tmp_path/'resumed'
        result=main(['--resume',str(interrupted/'checkpoints/last.pth'),'--run-dir',str(resumed),'--device','cpu'])
        after=torch.load(resumed/'checkpoints/last.pth',map_location='cpu',weights_only=True)
        for key,tensor in last['model'].items():torch.testing.assert_close(after['model'][key],tensor,rtol=0,atol=0)
        assert after['scheduler']==last['scheduler']
        assert after['step']==last['step']
        assert after['history'][1]['val_accuracy']==last['history'][1]['val_accuracy']
        assert result['status']=='completed'
        with pytest.raises(SystemExit):main(arguments(clean_data,continuous))
        with pytest.raises(FileExistsError):export_backbone(continuous/'checkpoints/best.pth',continuous/'checkpoints/backbone_best.pth')


def test_feature_interface_keeps_classifier():
    torch.set_num_threads(2)
    model=SupervisedResNet18(7).eval();x=torch.rand(2,3,64,64)
    with torch.inference_mode():
        assert model(x).shape==(2,7)
        assert model.forward_features(x).shape==(2,512)


def test_cuda_amp_and_clipping_two_small_batches(tmp_path):
    if not torch.cuda.is_available():pytest.skip('GPU unavailable in this execution context')
    torch.use_deterministic_algorithms(True)
    from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset,get_cub_transforms
    root=Path(__file__).resolve().parents[1]
    if not (root/'resnet18-f37072fd.pth').exists():pytest.skip('Local ImageNet weights required')
    model=SupervisedResNet18(100,root/'resnet18-f37072fd.pth').cuda()
    ds=CUBFewShotDataset(root/'cub2002011/versions/7/CUB_200_2011',root/'data/splits/cub_clean_v1/train_images.txt',get_cub_transforms(224,True),require_clean_index=True)
    loader=DataLoader(Subset(ds,[0,1,2,3]),batch_size=2)
    opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    before=model.classifier.weight.detach().clone()
    writer=SummaryWriter(str(tmp_path/'tb'))
    with (tmp_path/'steps.csv').open('w',newline='') as f:
        sw=csv.DictWriter(f,fieldnames=['epoch','step','loss','accuracy','lr','grad_norm','optimizer_step_skipped']);sw.writeheader()
        metrics,step=train_epoch(model,loader,opt,scaler,torch.device('cuda'),True,1.,sw,writer,1,0)
    writer.flush();writer.close()
    assert step==2 and metrics['samples']==4 and metrics['gpu_peak_memory_mb']>0
    assert not torch.equal(before,model.classifier.weight)
    assert scaler.state_dict()
    assert sum(float(p.grad.norm())**2 for p in model.parameters() if p.grad is not None)**.5<=1.001
    file=tmp_path/'cuda_state.pth'
    torch.save({'optimizer':opt.state_dict(),'scaler':scaler.state_dict()},file)
    saved=torch.load(file,map_location='cpu',weights_only=True)
    restored=torch.amp.GradScaler('cuda');restored.load_state_dict(saved['scaler'])
    assert restored.state_dict()==scaler.state_dict()
    other=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
    other.load_state_dict(saved['optimizer'])
    for key,state in opt.state_dict()['state'].items():
        torch.testing.assert_close(other.state_dict()['state'][key]['momentum_buffer'],state['momentum_buffer'],rtol=0,atol=0)
    evidence=dict(gpu=torch.cuda.get_device_name(),amp_enabled=True,init_scale=128,
                  input_resolution=224,batch_size=2,batches=2,metrics=metrics,
                  grad_clip=1.,parameter_updated=True,amp_state_restore_equal=True,
                  sgd_momentum_restore_equal=True,training_images_used=4,full_training=False)
    destination=root/'outputs/stage04_training/supervised_implementation/gpu_supervised_evidence.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(evidence,indent=2)+'\n')
