"""Independent math and member-order checks for the forward-only audit."""
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from scripts.check_stage05_protonet import audit_members, checked_math, toy_check
from pattern_recognition_coursework.models.metrics.episode import prototype_scores


def test_hand_computed_five_way_one_and_five_shot():
    records = toy_check()
    assert [r['k_shot'] for r in records] == [1, 5]
    assert all(r['status']=='PASS' for r in records)


def test_missing_feature_dimension_scaling_is_detected():
    support=torch.tensor([[0.,0.],[2.,0.]])
    query=torch.tensor([[1.,0.],[3.,0.]])
    actual=prototype_scores(support,query,2,1)
    checked_math(torch.cat([support,query]),2,1,1,actual)
    with pytest.raises(AssertionError):
        checked_math(torch.cat([support,query]),2,1,1,actual*2)


@pytest.mark.parametrize('k',[1,5])
def test_member_local_mapping_and_content_duplicates(tmp_path,k):
    samples=[];ids=[];classes=[f'Bird{i}' for i in range(5)]
    support=[];query=[]
    for label in [4,1,3,0,2]:
        for position in range(k+5):
            index=len(samples);path=tmp_path/f'{index}.png'
            Image.new('RGB',(4,4),(label*35,position*15,33)).save(path)
            samples.append((path,label));ids.append(str(index))
            (support if position<k else query).append(index)
    dataset=SimpleNamespace(samples=samples,image_ids=ids,classes=classes,data_root=tmp_path)
    indices=support+query
    rows,names=audit_members(dataset,indices,k,5)
    assert names==['Bird4','Bird1','Bird3','Bird0','Bird2']
    assert [r['episode_label'] for r in rows[5*k:]]==[i for i in range(5) for _ in range(5)]
    # Different ID and path, but exactly the same bytes: must still be rejected.
    samples[query[0]][0].write_bytes(samples[support[0]][0].read_bytes())
    with pytest.raises(AssertionError):audit_members(dataset,indices,k,5)
