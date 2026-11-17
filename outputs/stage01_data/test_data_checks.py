"""Data-only stage acceptance; two real failures are deliberately not xfailed.

No model imports. Missing formal species review is an acceptance blocker,
not permission to fabricate species assignments for actual NABirds images.
"""
import json
from pathlib import Path

from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from check_data import (
    CUB_ROOT, ROOT, OUT, MEAN, STD, denormalize, make_datasets, sha,
)
from pattern_recognition_coursework.data.cub_dataset import FewShotBatchSampler, get_cub_transforms
from pattern_recognition_coursework.data.nabirds import NABirdsDataset


@pytest.fixture(scope='session')
def datasets():
    torch.set_num_threads(2)
    return make_datasets()


@pytest.fixture(scope='session')
def samples():
    return json.loads((OUT/'sample_manifest.json').read_text())


@pytest.mark.parametrize('name,images,classes',[
    ('cub_train',5890,100),('cub_val',2947,50),('cub_test',2951,50),('nabirds_raw',48562,555),
])
def test_counts_and_label_namespace(datasets,name,images,classes):
    dataset=datasets[name]
    assert len(dataset)==images
    assert len(dataset.classes)==classes
    assert set(dataset.class_to_indices)==set(range(classes))
    assert all(0<=label<classes for _,label in dataset.samples)
    assert sum(len(v) for v in dataset.class_to_indices.values())==images
    for label,indices in dataset.class_to_indices.items():
        assert all(dataset.samples[index][1]==label for index in indices)


@pytest.mark.parametrize('name',['cub_train','cub_val','cub_test','nabirds_raw'])
def test_actual_image_loading_and_tensor_shape(datasets,name):
    d=datasets[name]
    for index in [0,len(d)//2,len(d)-1]:
        image,label=d[index]
        assert image.shape==(3,224,224)
        assert image.dtype==torch.float32
        assert torch.isfinite(image).all()
        assert 0<=label<len(d.classes)
        assert float(image.min())>=-2.12
        assert float(image.max())<=2.65


@pytest.mark.parametrize('name',['cub_train','cub_val','nabirds_raw'])
def test_same_visualized_batch_actual_dataloader(datasets,samples,name):
    chosen=[r for r in samples if r['dataset']==name]
    images,labels=next(iter(DataLoader(Subset(datasets[name],[r['sample_index'] for r in chosen]),batch_size=4,num_workers=0)))
    assert images.shape==(4,3,224,224)
    assert labels.dtype==torch.int64 and labels.shape==(4,)
    assert labels.tolist()==[r['label'] for r in chosen]
    assert torch.isfinite(images).all()


def test_cub_split_id_name_path_disjointness_and_seed():
    s=json.loads((OUT/'statistics.json').read_text())
    assert all(all(v==0 for v in pair.values()) for pair in s['split_intersections'].values())
    assert all(s['seed42_split_matches'].values())


@pytest.mark.parametrize('shot',[1,5])
def test_episode_index_order_reproducibility_and_labels(datasets,shot):
    for name,d in datasets.items():
        sampler=FewShotBatchSampler(d,5,5,shot,5,seed=42)
        batches=list(sampler)
        assert batches==list(sampler)
        for batch in batches:
            assert len(batch)==5*(shot+5)
            assert len(batch)==len(set(batch))
            support=batch[:5*shot];query=batch[5*shot:]
            assert set(support).isdisjoint(query)
            class_order=[d.samples[i][1] for i in support[::shot]]
            assert len(set(class_order))==5
            assert [d.samples[i][1] for i in support]==[c for c in class_order for _ in range(shot)]
            assert [d.samples[i][1] for i in query]==[c for c in class_order for _ in range(5)]


def test_transform_order_and_actual_parameters():
    train=get_cub_transforms(224,True).transforms
    evaluation=get_cub_transforms(224,False).transforms
    assert [type(x).__name__ for x in train]==['RandomResizedCrop','RandomHorizontalFlip','ColorJitter','ToTensor','Normalize']
    assert [type(x).__name__ for x in evaluation]==['Resize','CenterCrop','ToTensor','Normalize']
    assert train[0].scale==(.2,1.)
    assert train[0].ratio==(.75,4/3)
    assert train[0].size==(224,224) and train[0].antialias is True
    assert train[1].p==.5
    assert train[2].brightness==(.85,1.15)
    assert train[2].contrast==(.85,1.15)
    assert train[2].saturation==(.85,1.15)
    assert train[2].hue is None
    assert evaluation[0].size==257
    assert evaluation[1].size==(224,224)
    assert evaluation[-1].mean==[.485,.456,.406]
    assert evaluation[-1].std==[.229,.224,.225]


def test_eval_deterministic_under_changed_seed(samples):
    transform=get_cub_transforms(224,False)
    for row in samples:
        with Image.open(ROOT/row['path']) as src:rgb=src.convert('RGB')
        torch.manual_seed(1);a=transform(rgb)
        torch.manual_seed(99);b=transform(rgb)
        assert torch.equal(a,b)


def test_augmentation_sizes_diversity_and_denormalization(samples):
    assert len(samples)>=10
    for row in samples:
        assert row['train_views_distinct']
        assert row['eval_repeat_equal']
        assert row['original_sha256']==sha(OUT/row['original_copy'])
        for view in row['train_views']:
            assert view['tensor']['shape']==[3,224,224]
            assert view['tensor']['dtype']=='torch.float32'
            assert view['tensor']['finite']
            assert view['denormalized']['min']>=-1e-6
            assert view['denormalized']['max']<=1+1e-6
            with Image.open(OUT/view['file']) as im:assert im.size==(224,224)
    image=torch.rand(3,224,224)
    normalized=T.Normalize(MEAN.flatten().tolist(),STD.flatten().tolist())(image)
    torch.testing.assert_close(denormalize(normalized),image,atol=1e-6,rtol=0)


def test_no_cross_split_or_cross_domain_exact_content_duplicates():
    audit=json.loads((OUT/'evidence/content_audit.json').read_text())
    assert audit['cross_cub_split_duplicate_groups']==[]
    assert audit['cub_nabirds_exact_duplicate_groups']==[]


def test_support_query_content_disjointness_acceptance():
    """Expected to FAIL until the actual data/sampler defect is repaired."""
    repro=json.loads((OUT/'evidence/content_episode_reproduction.json').read_text())
    assert repro['status']!='FAIL', (
        f"Known real-data content collision: seed=42, episode_zero_based={repro['episode_zero_based']}, "
        f"support indices={repro['duplicate_support']}, query indices={repro['duplicate_query']}. "
        'Distinct indices alone do not guarantee distinct image content.'
    )


def test_formal_nabirds_species_dataset_acceptance():
    """Expected to FAIL until an actual human-reviewed manifest is supplied."""
    review=ROOT/'data/splits/nabirds_review.json'
    manifest=ROOT/'data/splits/nabirds_test.json'
    assert review.exists() and manifest.exists(), (
        'BLOCKED: formal NABirds review/manifest absent; raw visual-category loading '
        'is not a substitute for approved non-overlapping species evaluation.'
    )
    ds=NABirdsDataset(ROOT/'nabirds',manifest,get_cub_transforms(224,False),
        review_file=review,cub_train_classes=ROOT/'data/splits/train_classes.txt')
    image,label=ds[0]
    assert image.shape==(3,224,224) and 0<=label<len(ds.classes)


def test_source_and_split_files_unchanged():
    before=json.loads((OUT/'evidence/protected_files_before.json').read_text())
    assert all(sha(ROOT/path)==value for path,value in before.items())
