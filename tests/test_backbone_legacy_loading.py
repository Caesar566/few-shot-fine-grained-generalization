"""Legacy BN-counter compatibility only; no training or network downloads."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import warnings

import pytest
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from pattern_recognition_coursework.models.backbones.resnet import load_backbone_weights, make_backbone, spatial_features
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset, get_cub_transforms


def tiny():
    return nn.Sequential(nn.Conv2d(3,4,3,bias=False), nn.BatchNorm2d(4))


def test_only_missing_bn_counter_is_zeroed_and_logged(tmp_path):
    model=tiny();model[1].num_batches_tracked.fill_(99)
    state={k:v.clone() for k,v in model.state_dict().items() if k!='1.num_batches_tracked'}
    file=tmp_path/'legacy.pt';torch.save(state,file)
    with pytest.warns(UserWarning,match='1.num_batches_tracked'):
        info=load_backbone_weights(model,file)
    assert info['initialized_buffers']==['1.num_batches_tracked']
    assert info['loaded_keys']==len(state)
    assert int(model[1].num_batches_tracked)==0
    for k,v in state.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


@pytest.mark.parametrize('key',['0.weight','1.weight','1.bias','1.running_mean','1.running_var'])
def test_critical_missing_keys_still_rejected(tmp_path,key):
    model=tiny();before={k:v.clone() for k,v in model.state_dict().items()}
    state=dict(before);state.pop(key);state.pop('1.num_batches_tracked')
    file=tmp_path/'missing.pt';torch.save(state,file)
    with pytest.raises(ValueError,match='missing='):
        load_backbone_weights(model,file)
    for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)


@pytest.mark.parametrize('defect',['unknown','shape','dtype','counter_dtype'])
def test_unknown_shape_dtype_errors_rejected(tmp_path,defect):
    model=tiny();state=dict(model.state_dict())
    if defect=='unknown':state['mystery.weight']=torch.zeros(1)
    elif defect=='shape':state['0.weight']=torch.zeros(1)
    elif defect=='dtype':state['1.running_mean']=state['1.running_mean'].double()
    else:state['1.num_batches_tracked']=state['1.num_batches_tracked'].float()
    file=tmp_path/'invalid.pt';torch.save(state,file)
    with pytest.raises(ValueError,match='不兼容'):load_backbone_weights(model,file)


def test_same_named_non_bn_buffer_is_not_eligible(tmp_path):
    model=tiny();model[0].register_buffer('num_batches_tracked',torch.tensor(0))
    state=dict(model.state_dict());state.pop('0.num_batches_tracked')
    file=tmp_path/'not_bn.pt';torch.save(state,file)
    with pytest.raises(ValueError,match='0.num_batches_tracked'):load_backbone_weights(model,file)


def test_existing_counter_is_preserved(tmp_path):
    model=tiny();state=dict(model.state_dict());state['1.num_batches_tracked']=torch.tensor(17)
    file=tmp_path/'modern.pt';torch.save(state,file)
    with warnings.catch_warnings(record=True) as logs:
        info=load_backbone_weights(model,file)
    assert not logs and info['initialized_buffers']==[]
    assert int(model[1].num_batches_tracked)==17


def test_prefixed_container_and_counter_compatibility(tmp_path):
    model=tiny();state={'encoder.'+k:v for k,v in model.state_dict().items() if not k.endswith('num_batches_tracked')}
    file=tmp_path/'prefix.pt';torch.save({'state_dict':state},file)
    with pytest.warns(UserWarning,match='1.num_batches_tracked'):
        info=load_backbone_weights(model,file,'encoder.')
    assert info['initialized_buffers']==['1.num_batches_tracked'] and info['container']=='state_dict'


def test_real_official_checkpoint_native_equals_custom_and_real_cub_forward():
    root=Path(__file__).resolve().parents[1]
    weight=root/'resnet18-f37072fd.pth'
    cache=root/'outputs/stage02_backbone/weights_verification/local_torch_cache'
    data=root/'cub2002011/versions/7/CUB_200_2011'
    if not weight.exists() or not (cache/'checkpoints'/weight.name).exists() or not data.exists():
        pytest.skip('Local official checkpoint/cache and real CUB required; never download')
    torch.set_num_threads(2)
    def digest():return hashlib.sha256(weight.read_bytes()).hexdigest()
    before=digest()
    assert before=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
    with patch('torch.hub.get_dir',return_value=str(cache)),patch('torch.hub.download_url_to_file',side_effect=AssertionError('downloads forbidden')):
        native=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        native.fc=nn.Identity();native.eval()
        with warnings.catch_warnings(record=True) as logs:
            warnings.simplefilter('always')
            custom=make_backbone(backbone_weights=weight).eval()
    filled=custom.backbone_load_info['initialized_buffers']
    assert len(filled)==20
    warning_text='\n'.join(str(w.message) for w in logs)
    assert all(k in warning_text for k in filled)
    assert native.state_dict().keys()==custom.state_dict().keys()
    for k,v in native.state_dict().items():
        assert v.dtype==custom.state_dict()[k].dtype
        torch.testing.assert_close(v,custom.state_dict()[k],rtol=0,atol=0)
    ds=CUBFewShotDataset(data,root/'data/splits/cub_clean_v1/val_images.txt',get_cub_transforms(224,False),require_clean_index=True)
    samples=[r for r in json.loads((root/'outputs/stage01_data/sample_manifest.json').read_text()) if r['dataset']=='cub_val']
    x=torch.stack([ds[r['sample_index']][0] for r in samples])
    with torch.inference_mode():
        a=native(x);b=custom(x)
        sa=spatial_features(native,x);sb=spatial_features(custom,x)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(sa,sb,rtol=0,atol=0)
    assert torch.isfinite(a).all() and torch.isfinite(sa).all()
    assert before==digest()
    out=root/'outputs/stage02_backbone/loader_compatibility';out.mkdir(parents=True,exist_ok=True)
    evidence={'official_sha256':before,'source_file_unchanged':True,
        'encoder_state_entries_compared':len(native.state_dict()),'all_state_entries_exactly_equal':True,
        'initialized_buffers':filled,'load_info':custom.backbone_load_info,
        'warnings':[str(w.message) for w in logs],'images':[r['path'] for r in samples],
        'input_shape':list(x.shape),'global_shape':list(a.shape),'spatial_shape':list(sa.shape),
        'global_max_abs_difference':float((a-b).abs().max()),'spatial_max_abs_difference':float((sa-sb).abs().max()),
        'training_executed':False,'downloads_performed':False}
    (out/'real_checkpoint_comparison.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
