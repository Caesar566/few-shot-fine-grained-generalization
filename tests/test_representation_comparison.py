"""Top-k, chunking, calibration and CUDA/CPU parity; no model training."""
import pytest
import torch
import json
from types import SimpleNamespace

from pattern_recognition_coursework.engine.representation_comparison import collect_local,choose_device

from pattern_recognition_coursework.models.metrics.local import local_descriptor_scores
from pattern_recognition_coursework.models.metrics.representation import (
    chunked_local_scores,representation_scores,row_score_standardize,
)


def test_hand_calculated_top3_and_patch_mean():
    support=torch.tensor([[[[1.,1.,0.]],[[0.,0.,1.]]],
                          [[[0.,0.,0.]],[[1.,1.,1.]]]])
    query=torch.tensor([[[[1.,1.,1.]],[[0.,0.,0.]]]])
    score=chunked_local_scores(support,query,2,1,3,1)
    torch.testing.assert_close(score,torch.tensor([[2/3,0.]]))


@pytest.mark.parametrize('k',[1,5])
def test_chunked_dn4_equals_existing_full_query_function(k):
    torch.manual_seed(42);torch.set_num_threads(2)
    support=torch.rand(5*k,512,7,7);query=torch.rand(25,512,7,7)
    expected=local_descriptor_scores(support,query,5,k,3)
    for chunk in (1,4,25):
        actual=chunked_local_scores(support,query,5,k,3,chunk)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        assert torch.equal(actual.argmax(-1),expected.argmax(-1))


def test_row_calibration_fixed_scale_and_degenerate_rows():
    scores=torch.tensor([[1.,2.,3.,4.,5.],[8.,8.,8.,8.,8.]])
    z=row_score_standardize(scores)
    torch.testing.assert_close(z[0].mean(),torch.tensor(0.))
    torch.testing.assert_close(z[0].square().mean(),torch.tensor(1.))
    assert torch.equal(z[1],torch.zeros(5))
    torch.testing.assert_close(z,row_score_standardize(scores*7+20))
    assert torch.equal(scores[0].argmax(),z[0].argmax())


def test_fusion_is_exact_fixed_half_of_calibrated_scores():
    torch.manual_seed(3)
    global_features=torch.rand(30,512);s=torch.rand(5,512,7,7);q=torch.rand(25,512,7,7)
    before=global_features.clone()
    with torch.inference_mode():r=representation_scores(global_features,s,q,5,1)
    assert torch.equal(r['fusion_scores'],.5*r['global_calibrated']+.5*r['local_calibrated'])
    assert torch.equal(before,global_features)
    assert all(value.device.type=='cpu' and value.dtype==torch.float32 for value in r.values())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA inaccessible in current environment')
@pytest.mark.parametrize('k',[1,5])
def test_cuda_cpu_same_matching_and_fusion_logic(k):
    torch.manual_seed(11)
    g=torch.rand(5*k+25,512);s=torch.rand(5*k,512,7,7);q=torch.rand(25,512,7,7)
    with torch.inference_mode():
        a=representation_scores(g,s,q,5,k)
        b=representation_scores(g.cuda(),s,q,5,k)
    for value in b.values():assert value.device.type=='cuda'
    bounds={}
    for branch in ('global','local'):
        raw_a=a[branch+'_scores'];raw_b=b[branch+'_scores'].cpu()
        torch.testing.assert_close(raw_a,raw_b,rtol=1e-5,atol=1e-6)
        # Nearly equal random class scores amplify FP32 rounding after z-score.
        # Bound propagation from STRICTLY checked raw errors and FP32 mean error,
        # rather than incorrectly requiring calibrated vectors to be bitwise equal.
        scale=torch.minimum(raw_a.std(-1,unbiased=False),raw_b.std(-1,unbiased=False)).clamp_min(1e-6)
        rounding=2*torch.finfo(torch.float32).eps*torch.maximum(raw_a.abs().amax(-1),raw_b.abs().amax(-1))
        bounds[branch]=8*((raw_a-raw_b).abs().amax(-1)+rounding)/scale+1e-5
        error=(a[branch+'_calibrated']-b[branch+'_calibrated'].cpu()).abs()
        assert (error<=bounds[branch][:,None]).all()
    fusion_error=(a['fusion_scores']-b['fusion_scores'].cpu()).abs()
    assert (fusion_error<=(.5*bounds['global']+.5*bounds['local'])[:,None]+1e-6).all()
    for name in ('global_scores','local_scores','fusion_scores'):
        assert torch.equal(a[name].argmax(-1),b[name].cpu().argmax(-1))


def test_local_cache_ids_and_weights_cannot_be_mixed(tmp_path):
    class Spatial(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=SimpleNamespace(avgpool=torch.nn.AdaptiveAvgPool2d(1))
        def encode(self,images):
            return images.mean((1,2,3))[:,None,None,None].expand(-1,512,7,7).contiguous()
    model=Spatial().eval();images=torch.rand(4,3,14,14)
    dataset=torch.utils.data.TensorDataset(images,torch.arange(4))
    global_features=model.encode(images).mean((-1,-2))
    logger=SimpleNamespace(info=lambda *args:None)
    binding={'checkpoint':'A','sample_ids':['3','1']}
    options=(model,dataset,[3,1],global_features,binding,tmp_path,2,0,torch.device('cpu'),logger)
    local,first=collect_local(*options);loaded,second=collect_local(*options)
    assert not first['hit'] and second['hit'] and torch.equal(local,loaded)
    changed=list(options);changed[2]=[1,3]
    with pytest.raises(ValueError,match='identity mismatch'):collect_local(*changed)
    payload=torch.load(first['path'],weights_only=True);payload['binding']['checkpoint']='B'
    torch.save(payload,first['path'])
    with pytest.raises(ValueError,match='identity mismatch'):collect_local(*options)


def test_cuda_unavailable_has_explicit_warning(monkeypatch,caplog):
    import logging
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert choose_device('cuda',logging.getLogger('test_fallback')).type=='cpu'
    assert 'explicitly falling back to CPU' in caplog.text


def test_smoke_cache_rows_reused_in_full_cache(tmp_path):
    class Spatial(torch.nn.Module):
        def __init__(self):
            super().__init__();self.backbone=SimpleNamespace(avgpool=torch.nn.AdaptiveAvgPool2d(1));self.seen=[]
        def encode(self,images):
            self.seen.extend(images[:,0,0,0].tolist())
            return images.mean((1,2,3))[:,None,None,None].expand(-1,512,7,7).contiguous()
    model=Spatial().eval();images=torch.arange(4,dtype=torch.float32)[:,None,None,None].expand(4,3,14,14).clone()
    dataset=torch.utils.data.TensorDataset(images,torch.arange(4))
    global_features=model.encode(images).mean((-1,-2));model.seen=[]
    logger=SimpleNamespace(info=lambda *args:None)
    common=dict(checkpoint='A',preprocessing='fixed',mode='eval',frozen=True)
    subset=dict(common,sample_indices=[0,2],ordered_sample_ids=['id0','id2'])
    collect_local(model,dataset,[0,2],global_features,subset,tmp_path,2,0,torch.device('cpu'),logger)
    model.seen=[]
    full=dict(common,sample_indices=[0,1,2,3],ordered_sample_ids=['id0','id1','id2','id3'])
    local,meta=collect_local(model,dataset,[0,1,2,3],global_features,full,tmp_path,2,0,torch.device('cpu'),logger)
    assert meta['reused_images']==2 and meta['extracted_images']==2
    assert model.seen==[1.,3.]
    torch.testing.assert_close(local.mean((-1,-2)),global_features)
    other=dict(full,checkpoint='B');model.seen=[]
    _,different=collect_local(model,dataset,[0,1,2,3],global_features,other,tmp_path,2,0,torch.device('cpu'),logger)
    assert different['reused_images']==0 and different['extracted_images']==4


def test_failure_record_and_existing_run_protection(tmp_path,monkeypatch):
    import pattern_recognition_coursework.engine.representation_comparison as module
    output=tmp_path/'new'
    def failing(argv,token):
        output.mkdir()
        module.write_json(output/'status.json',dict(run_token=token,state='running',phase='synthetic_probe'))
        raise RuntimeError('synthetic diagnostic')
    monkeypatch.setattr(module,'_run',failing)
    with pytest.raises(RuntimeError):module.main(['--output-dir',str(output)])
    failure=json.loads((output/'failure.json').read_text())
    assert failure['phase']['phase']=='synthetic_probe' and failure['exception_type']=='RuntimeError'
    previous=(output/'status.json').read_bytes()
    def existing(argv,token):raise SystemExit(2)
    monkeypatch.setattr(module,'_run',existing)
    with pytest.raises(SystemExit):module.main(['--output-dir',str(output)])
    assert (output/'status.json').read_bytes()==previous


def test_new_target_gap_can_be_collected_without_old_global_cache(tmp_path):
    class Spatial(torch.nn.Module):
        def __init__(self):super().__init__();self.backbone=SimpleNamespace(avgpool=torch.nn.AdaptiveAvgPool2d(1))
        def encode(self,images):return images.mean((1,2,3))[:,None,None,None].expand(-1,512,7,7).contiguous()
    images=torch.rand(3,3,14,14);dataset=torch.utils.data.TensorDataset(images,torch.arange(3))
    logger=SimpleNamespace(info=lambda *args:None)
    local,meta=collect_local(Spatial().eval(),dataset,[0,1,2],None,{'dataset':'new_target'},tmp_path,2,0,torch.device('cpu'),logger)
    saved=torch.load(meta['path'],weights_only=True)
    torch.testing.assert_close(saved['gap'],local.mean((-1,-2)))
    assert meta['gap_vs_original_global_max_abs'] is None
