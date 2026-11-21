"""Paired tasks, episode-level CI and stale cache rejection without training."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import t
import torch

from pattern_recognition_coursework.analysis.episodic_statistics import accuracy_summary
from pattern_recognition_coursework.data.paired_episodes import make_paired_plans, validate_paired_plans
from pattern_recognition_coursework.engine.episodic_evaluation import cache_load, cached_features, tensor_hash


class Tiny:
    def __init__(self,root):
        self.data_root=root
        self.samples=[(root/f'{c}_{i}.jpg',c) for c in range(7) for i in range(14)]
        self.class_to_indices={c:list(range(c*14,(c+1)*14)) for c in range(7)}

    def __len__(self):return len(self.samples)


def test_600_paired_plans_share_queries_and_nested_support(tmp_path):
    dataset=Tiny(tmp_path)
    plans=make_paired_plans(dataset,600,42)
    assert plans==make_paired_plans(dataset,600,42)
    assert plans!=make_paired_plans(dataset,600,43)
    for a,b in zip(plans[1]['batches'],plans[5]['batches']):
        assert len(a)==30 and len(b)==50
        assert a[5:]==b[25:] and a[:5]==b[:25:5]
        assert len(set(b))==50
        assert not set(b[:25])&set(b[25:])
    plans[1]['batches'][0][0]=plans[5]['batches'][0][1]
    with pytest.raises(ValueError,match='nested support'):
        validate_paired_plans(dataset,plans[1],plans[5])


def test_sample_std_and_student_t_ci():
    summary=accuracy_summary([0.,.5,1.])
    assert summary['mean_accuracy']==.5 and summary['sample_std']==.5
    expected=float(t.ppf(.975,2)*.5/np.sqrt(3))
    np.testing.assert_allclose(summary['ci95'],[.5-expected,.5+expected])
    assert accuracy_summary([.8]*600)['ci95_half_width']<1e-14


@pytest.mark.parametrize('values',[[1.],[1.,float('nan')],[-.1,.5],[1.1,.2]])
def test_invalid_episode_statistics_rejected(values):
    with pytest.raises(ValueError):accuracy_summary(values)


@pytest.mark.parametrize('field',['checkpoint_sha256','preprocessing','mode','actual_content_sha256'])
def test_stale_cache_binding_rejected(tmp_path,field):
    binding=dict(checkpoint_sha256='trainedA',preprocessing='center224',mode='eval',actual_content_sha256='dataA')
    features=torch.arange(8*512,dtype=torch.float32).reshape(8,512)
    path=tmp_path/'cache.pt'
    torch.save(dict(binding=binding,features=features,feature_sha256=tensor_hash(features)),path)
    assert torch.equal(cache_load(path,binding,8),features)
    changed=dict(binding);changed[field]='different'
    with pytest.raises(ValueError,match='binding mismatch'):cache_load(path,changed,8)


def test_corrupt_feature_cache_rejected(tmp_path):
    features=torch.randn(8,512);binding={'weights':'a'};path=tmp_path/'cache.pt'
    checksum=tensor_hash(features);features[0,0]+=1
    torch.save(dict(binding=binding,features=features,feature_sha256=checksum),path)
    with pytest.raises(ValueError,match='checksum mismatch'):cache_load(path,binding,8)
    torch.save(dict(binding=binding,features=features.double(),feature_sha256=tensor_hash(features.double())),path)
    with pytest.raises(ValueError,match='float32'):cache_load(path,binding,8)


def test_real_cache_creation_reuse_and_eval_guard(tmp_path):
    class Encoder(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def encode(self,images):
            self.calls+=1
            return images.flatten(1).mean(1,keepdim=True)+torch.arange(512,dtype=torch.float32)[None,:]
    images=torch.arange(8*3*4*4,dtype=torch.float32).reshape(8,3,4,4)
    dataset=torch.utils.data.TensorDataset(images,torch.arange(8))
    model=Encoder().eval()
    logger=SimpleNamespace(info=lambda *args:None)
    args=(model,dataset,{'weights':'verifiedA','mode':'eval'},tmp_path,2,0,torch.device('cpu'),True,logger)
    features,first=cached_features(*args)
    assert model.calls==4 and not first['reused']
    loaded,second=cached_features(*args)
    assert model.calls==4 and second['reused'] and torch.equal(features,loaded)
    model.train()
    with pytest.raises(ValueError,match='eval mode'):cached_features(*args)
