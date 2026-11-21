"""Automatic screening is explicit, species-level, test-only and fails closed."""
import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from pattern_recognition_coursework.data.nabirds import sha256_file
from pattern_recognition_coursework.data.nabirds_automatic import (
    AutomaticNABirdsDataset,build_automatic_manifest,resolve_cub_train,
)
from pattern_recognition_coursework.data.paired_episodes import make_paired_plans


@pytest.fixture
def fixture(tmp_path):
    root=tmp_path/'raw';(root/'images').mkdir(parents=True)
    classes=['0 Birds\n','1 Family group\n'];hierarchy=['1 0\n'];images=[];labels=[];splits=[]
    taxonomy=tmp_path/'taxonomy.csv'
    with taxonomy.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['COMMON_NAME','SCIENTIFIC_NAME','SPECIES_CODE','CATEGORY','REPORT_AS']);w.writeheader()
        for species in range(7):
            name='Training Bird' if species==0 else f'Novel Bird {species}'
            w.writerow(dict(COMMON_NAME=name,SCIENTIFIC_NAME=f'Fixture bird{species}',SPECIES_CODE=f'sp{species}',CATEGORY='species',REPORT_AS=''))
            sid=100+species;classes.append(f'{sid} {name}\n');hierarchy.append(f'{sid} 1\n')
            for sex in (0,1):
                visual=200+species*2+sex
                classes.append(f'{visual} {name} ({"Male" if sex==0 else "Female"})\n');hierarchy.append(f'{visual} {sid}\n')
                for photo in range(7):
                    image_id=f'uuid-{species}-{sex}-{photo}';relative=f'{image_id}.png'
                    Image.new('RGB',(12,12),(species*30,sex*70,photo*25)).save(root/'images'/relative)
                    images.append(f'{image_id} {relative}\n');labels.append(f'{image_id} {visual}\n')
                    splits.append(f'{image_id} {1 if photo==6 else 0}\n')
    for name,content in [('classes.txt',classes),('hierarchy.txt',hierarchy),('images.txt',images),('image_class_labels.txt',labels),('train_test_split.txt',splits)]:
        (root/name).write_text(''.join(content))
    cub=tmp_path/'cub.txt';cub.write_text('1 001.Training_Bird\n')
    hashes=tmp_path/'hash.csv';hashes.write_text('sha256\n'+sha256_file(root/'images/uuid-0-0-0.png')+'\n')
    return root,cub,taxonomy,hashes


def test_automatic_pool_merges_sexes_excludes_training_species_and_train_images(fixture,tmp_path):
    root,cub,taxonomy,hashes=fixture
    m=build_automatic_manifest(root,cub,taxonomy,hashes)
    assert m['human_reviewed'] is False and m['automatic_screening'] is True
    assert len(m['classes'])==6 and len(m['samples'])==72
    assert all(c['image_count']==12 and len(c['visual_class_ids'])==2 for c in m['classes'])
    assert all(s['class_id']!='sp0' and s['official_split']==0 for s in m['samples'])
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='consent'):AutomaticNABirdsDataset(root,path,cub,taxonomy)
    dataset=AutomaticNABirdsDataset(root,path,cub,taxonomy,allow_automatic=True)
    plans=make_paired_plans(dataset,600,42)
    assert all(a[5:]==b[25:] and a[:5]==b[:25:5] for a,b in zip(plans[1]['batches'],plans[5]['batches']))


def test_unresolved_cub_class_cannot_be_approved(fixture):
    root,cub,taxonomy,hashes=fixture;cub.write_text('1 Unknown_Unresolvable_Taxon\n')
    with pytest.raises(ValueError,match='Unresolved CUB'):build_automatic_manifest(root,cub,taxonomy,hashes)


def test_duplicate_content_is_removed_only_in_derived_index(fixture):
    root,cub,taxonomy,hashes=fixture
    a=root/'images/uuid-1-0-0.png';b=root/'images/uuid-1-0-1.png';b.write_bytes(a.read_bytes())
    before={p.name:sha256_file(p) for p in (root/'images').iterdir()}
    m=build_automatic_manifest(root,cub,taxonomy,hashes)
    assert len(m['logical_content_exclusions'])==1
    assert any(r['reason']=='same_species_exact_duplicate' for r in m['logical_content_exclusions'])
    assert before=={p.name:sha256_file(p) for p in (root/'images').iterdir()}


def test_train_test_metadata_tampering_rejected(fixture,tmp_path):
    root,cub,taxonomy,hashes=fixture;m=build_automatic_manifest(root,cub,taxonomy,hashes)
    path=tmp_path/'manifest.json';path.write_text(json.dumps(m))
    split=root/'train_test_split.txt';split.write_text(split.read_text().replace('uuid-1-0-0 0','uuid-1-0-0 1'))
    with pytest.raises(ValueError,match='metadata changed'):AutomaticNABirdsDataset(root,path,cub,taxonomy,allow_automatic=True)
