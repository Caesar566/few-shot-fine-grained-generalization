"""Logical-index deduplication regression tests; no network or model training."""
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest
from torch.utils.data import DataLoader

from pattern_recognition_coursework.data.cub_dataset import (
    CUBFewShotDataset, FewShotBatchSampler, get_cub_transforms,
)
from pattern_recognition_coursework.data.cub_dedup import (
    audit_and_clean, classify_duplicate_groups, sha256_file,
)
from pattern_recognition_coursework.data.episodes import (
    dataset_fingerprint, load_or_create_plan,
)


@pytest.fixture
def tiny(tmp_path):
    root=tmp_path/'raw';splits=tmp_path/'splits'
    root.mkdir();splits.mkdir()
    classes=[];images=[];labels=[];locations={}
    for split,cids in [('train',[1,2]),('val',[3,4]),('test',[5,6])]:
        cr=[];ir=[]
        for cid in cids:
            cname=f'{cid:03d}.Bird_{cid}'
            classes.append(f'{cid} {cname}\n');cr.append(f'{cid} {cname}\n')
            folder=root/'images'/cname;folder.mkdir(parents=True)
            for n in range(3):
                image_id=cid*10+n;relative=f'{cname}/{n}.png'
                path=root/'images'/relative
                Image.new('RGB',(18,16),(cid*30,n*60,7)).save(path)
                locations[(cid,n)]=path
                images.append(f'{image_id} {relative}\n');labels.append(f'{image_id} {cid}\n')
                ir.append(f'{image_id} {cname} {relative}\n')
        (splits/f'{split}_classes.txt').write_text(''.join(cr))
        (splits/f'{split}_images.txt').write_text(''.join(ir))
    (root/'classes.txt').write_text(''.join(classes))
    (root/'images.txt').write_text(''.join(images))
    (root/'image_class_labels.txt').write_text(''.join(labels))
    return root,splits,locations


def make_clean(tiny,tmp_path):
    root,splits,paths=tiny
    clean=tmp_path/'clean';report=tmp_path/'report'
    result=audit_and_clean(root,splits,clean,report)
    return clean,report,result


def test_same_class_dedup_preserves_sources_and_labels(tiny,tmp_path):
    root,splits,paths=tiny
    shutil.copyfile(paths[(1,0)],paths[(1,1)])
    before={str(p):sha256_file(p) for base in (root,splits) for p in base.rglob('*') if p.is_file()}
    clean,report,result=make_clean(tiny,tmp_path)
    assert result['duplicate_groups']==1 and result['redundant_disk_images']==1
    assert result['excluded_official_index_images']==1
    assert result['clean_index_images']==17
    assert result['ready_for_training_and_evaluation'] is True
    manifest=json.loads((clean/'dedup_manifest.json').read_text())
    assert manifest['excluded_image_ids']==['11']
    ds=CUBFewShotDataset(root,clean/'train_images.txt',get_cub_transforms(32,False),require_clean_index=True)
    original=CUBFewShotDataset(root,splits/'train_images.txt')
    assert ds.index_kind=='clean' and len(ds)==5 and len(original)==6
    assert ds.classes==original.classes and ds.class_to_idx==original.class_to_idx
    assert paths[(1,0)] in [p for p,_ in ds.samples]
    assert paths[(1,1)] not in [p for p,_ in ds.samples]
    assert all(sha256_file(Path(p))==h for p,h in before.items())
    images,labels=next(iter(DataLoader(ds,batch_size=2)))
    assert list(images.shape)==[2,3,32,32] and len(labels)==2


@pytest.mark.parametrize('target',[(2,0),(3,0)])
def test_cross_class_or_split_is_retained_and_blocked(tiny,tmp_path,target):
    root,splits,paths=tiny
    shutil.copyfile(paths[(1,0)],paths[target])
    clean,report,result=make_clean(tiny,tmp_path)
    assert result['review_required_groups']==1
    assert result['cross_class_groups']==1
    assert result['cross_split_groups']==(1 if target[0]==3 else 0)
    assert result['excluded_official_index_images']==0
    assert result['clean_index_images']==18
    assert result['ready_for_training_and_evaluation'] is False
    for split in ['train','val','test']:
        assert (clean/f'{split}_images.txt').read_bytes()==(splits/f'{split}_images.txt').read_bytes()
    with pytest.raises(ValueError,match='requiring review'):
        CUBFewShotDataset(root,clean/'train_images.txt')
    assert 'retained_pending_review' in (report/'review_required.csv').read_text()


def test_numeric_id_keeper_is_independent_of_inventory_order():
    rows=[{'image_id':'10','class_name':'Bird','split':'test','relative_path':'Bird/a.png','sha256':'same'},
          {'image_id':'2','class_name':'Bird','split':'test','relative_path':'Bird/z.png','sha256':'same'}]
    groups,excluded=classify_duplicate_groups(rows)
    reverse_groups,reverse_excluded=classify_duplicate_groups(list(reversed(rows)))
    assert groups==reverse_groups
    assert groups[0]['keeper_image_id']=='2'
    assert excluded==reverse_excluded=={'10'}


def test_unindexed_image_is_audited_but_never_added(tiny,tmp_path):
    root,splits,paths=tiny
    extra=paths[(1,0)].parent/'unindexed.png'
    shutil.copyfile(paths[(1,0)],extra)
    clean,report,result=make_clean(tiny,tmp_path)
    assert result['physical_images_scanned']==19
    assert result['unindexed_images']==1
    assert result['official_indexed_images']==result['clean_index_images']==18
    assert result['excluded_official_index_images']==0
    assert 'unindexed_not_added' in (report/'duplicate_samples.csv').read_text()
    assert 'unindexed.png' not in (clean/'train_images.txt').read_text()


def test_clean_indexes_are_deterministic(tiny,tmp_path):
    root,splits,paths=tiny
    shutil.copyfile(paths[(1,0)],paths[(1,1)])
    clean,_,_=make_clean(tiny,tmp_path)
    other=tmp_path/'other'
    audit_and_clean(root,splits,other,tmp_path/'report_other')
    for p in clean.glob('*.txt'):
        assert p.read_bytes()==(other/p.name).read_bytes()
    assert (clean/'image_hashes.csv').read_bytes()==(other/'image_hashes.csv').read_bytes()


@pytest.mark.parametrize('changed',['train_images.txt','test_classes.txt','image_hashes.csv'])
def test_changed_derived_index_is_rejected(tiny,tmp_path,changed):
    root,_,_=tiny
    clean,_,_=make_clean(tiny,tmp_path)
    path=clean/changed
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='hash mismatch'):
        CUBFewShotDataset(root,clean/'train_images.txt')


def test_changed_source_metadata_is_rejected(tiny,tmp_path):
    root,_,_=tiny
    clean,_,_=make_clean(tiny,tmp_path)
    (root/'classes.txt').write_bytes((root/'classes.txt').read_bytes()+b'\n')
    with pytest.raises(ValueError,match='source metadata'):
        CUBFewShotDataset(root,clean/'train_images.txt')


def test_explicit_requirement_rejects_raw_indexes(tiny):
    root,splits,_=tiny
    assert CUBFewShotDataset(root,splits/'train_images.txt').index_kind=='original'
    with pytest.raises(ValueError,match='verified clean index'):
        CUBFewShotDataset(root,splits/'train_images.txt',require_clean_index=True)


def test_publishing_marker_is_rejected(tiny,tmp_path):
    root,splits,_=tiny
    folder=tmp_path/'publishing';folder.mkdir()
    (folder/'dedup_manifest.json').write_text(json.dumps({'schema':'cub-exact-dedup-v1','state':'publishing'}))
    with pytest.raises(ValueError,match='incomplete'):
        CUBFewShotDataset(root,folder/'train_images.txt')


def test_refuse_overwrite_or_raw_destinations(tiny,tmp_path):
    root,splits,_=tiny
    clean,report,_=make_clean(tiny,tmp_path)
    with pytest.raises(FileExistsError):audit_and_clean(root,splits,clean,report)
    with pytest.raises(ValueError):audit_and_clean(root,splits,root/'images'/'clean',tmp_path/'new_report')
    with pytest.raises(ValueError):audit_and_clean(root,splits,splits,tmp_path/'new_report')


def test_removed_duplicate_changes_plan_identity(tiny,tmp_path):
    root,splits,paths=tiny
    shutil.copyfile(paths[(1,0)],paths[(1,1)])
    clean,_,_=make_clean(tiny,tmp_path)
    raw=CUBFewShotDataset(root,splits/'train_images.txt')
    derived=CUBFewShotDataset(root,clean/'train_images.txt')
    assert dataset_fingerprint(raw)!=dataset_fingerprint(derived)
    plan_path=tmp_path/'episodes.json'
    load_or_create_plan(plan_path,raw,2,1,1,2,42)
    with pytest.raises(ValueError,match='清单不一致'):
        load_or_create_plan(plan_path,derived,2,1,1,2,42)


@pytest.mark.parametrize('shot',[1,5])
def test_real_clean_cub_episode_content_disjoint_and_repeatable(shot):
    root=Path(__file__).resolve().parents[1]
    data=root/'cub2002011/versions/7/CUB_200_2011'
    clean=root/'data/splits/cub_clean_v1'
    if not (data/'classes.txt').exists() or not (clean/'dedup_manifest.json').exists():
        pytest.skip('Local CUB and separately generated clean indexes are required')
    with (clean/'image_hashes.csv').open(encoding='utf-8-sig') as stream:
        import csv
        hashes={r['relative_path']:r['sha256'] for r in csv.DictReader(stream)}
    for split in ['train','val','test']:
        ds=CUBFewShotDataset(data,clean/f'{split}_images.txt',require_clean_index=True)
        sampler=FewShotBatchSampler(ds,300,5,shot,15,seed=42)
        batches=list(sampler)
        assert batches==list(sampler)
        assert batches!=list(FewShotBatchSampler(ds,300,5,shot,15,seed=43))
        for batch in batches:
            assert len(batch)==5*(shot+15)
            content=[hashes[ds.samples[i][0].relative_to(data/'images').as_posix()] for i in batch]
            assert len(set(content))==len(content)
            labels=[ds.samples[i][1] for i in batch]
            classes=labels[:5*shot:shot]
            assert len(set(classes))==5
            assert labels[:5*shot]==[c for c in classes for _ in range(shot)]
            assert labels[5*shot:]==[c for c in classes for _ in range(15)]
