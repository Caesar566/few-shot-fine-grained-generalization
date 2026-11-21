"""Freeze an explicitly automatic species protocol and 600 paired NABirds tasks."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from pattern_recognition_coursework.data.nabirds import load_nabirds_metadata
from pattern_recognition_coursework.data.nabirds_automatic import build_automatic_manifest,AutomaticNABirdsDataset
from pattern_recognition_coursework.data.paired_episodes import make_paired_plans
from pattern_recognition_coursework.engine.episodic_evaluation import ROOT,save_plans,sha,write_json,canonical_hash
from pattern_recognition_coursework.engine.distance_comparison import write_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=ROOT/'nabirds')
    p.add_argument('--taxonomy',type=Path,default=ROOT/'outputs/stage06_nabirds/preflight/ebird_taxonomy_v2026.csv')
    p.add_argument('--protocol-dir',type=Path,default=ROOT/'outputs/stage06_nabirds/protocol')
    p.add_argument('--allow-automatic-screening',action='store_true')
    args=p.parse_args()
    if not args.allow_automatic_screening:p.error('Explicit consent required; this is not the manual review protocol')
    out=args.protocol_dir.resolve()
    if not out.is_relative_to(ROOT/'outputs/stage06_nabirds'):p.error('Protocol output must stay in Stage06')
    if (out/'plan_1shot.json').exists():p.error('Protocol already frozen; refusing overwrite')
    out.mkdir(parents=True,exist_ok=True)
    cub=ROOT/'data/splits/cub_clean_v1/train_classes.txt';hashes=ROOT/'data/splits/cub_clean_v1/image_hashes.csv'
    manifest_path=out/'manifest.json'
    if not manifest_path.exists():
        write_json(manifest_path,build_automatic_manifest(args.data_root,cub,args.taxonomy,hashes,10))
    dataset=AutomaticNABirdsDataset(args.data_root,manifest_path,cub,args.taxonomy,allow_automatic=True)
    catalog=[]
    for index,((path,label),row) in enumerate(zip(dataset.samples,dataset.records)):
        if sha(path)!=row['sha256']:raise ValueError('Actual NABirds image content changed')
        catalog.append(dict(dataset_index=index,image_id=row['image_id'],dataset_label=label,
            class_name=dataset.classes[label],relative_path=row['path'],sha256=row['sha256'],
            species_code=row['class_id'],visual_class_id=row['visual_class_id'],official_split=0))
    plans=make_paired_plans(dataset,600,42)
    assert plans==make_paired_plans(dataset,600,42)
    plan_checks=save_plans(out,plans,catalog);write_json(out/'plan_checks.json',plan_checks)
    manifest=dataset.manifest_metadata;metadata=load_nabirds_metadata(args.data_root)
    split={line.split()[0]:int(line.split()[1]) for line in (args.data_root/'train_test_split.txt').read_text().splitlines()}
    raw=Counter(metadata['labels'].values());test=Counter(metadata['labels'][i] for i,value in split.items() if value==0)
    kept=Counter(row['visual_class_id'] for row in dataset.records)
    write_rows(out/'visual_mapping_audit.csv',[dict(visual_class_id=row['visual_class_id'],name=row['visual_name'],
        raw_images=raw[row['visual_class_id']],official_test_images=test[row['visual_class_id']],retained_images=kept[row['visual_class_id']],
        species_code=row['species_code'],decision=row['decision']) for row in manifest['visual_audit']])
    write_rows(out/'species_mapping.csv',[dict(local_label=i,species_code=row['class_id'],name=row['name'],
        scientific_name=row['scientific_name'],official_test_images=row['image_count'],visual_class_ids=json.dumps(row['visual_class_ids'])) for i,row in enumerate(manifest['classes'])])
    summary=dict(protocol=manifest['protocol'],human_reviewed=False,image_split='official_test_only',
        species=len(dataset.classes),images=len(dataset),minimum_images_per_species=min(row['image_count'] for row in manifest['classes']),
        maximum_images_per_species=max(row['image_count'] for row in manifest['classes']),
        semantic_visual_decisions=dict(Counter(r['decision'] for r in manifest['visual_audit'])),
        resolved_cub_train_classes=len(manifest['cub_semantic_resolution']),excluded_taxon_superset_size=len(manifest['excluded_cub_taxon_codes']),
        actual_selected_bytes_verified=True,content_exclusions=len(manifest['logical_content_exclusions']),
        manifest_sha256=sha(manifest_path),taxonomy_sha256=sha(args.taxonomy),dataset_content_sha256=canonical_hash(catalog),
        protocol_frozen_before_target_performance=True,seed=42,episodes_per_shot=600,
        plan_hashes=plan_checks['both_plan_hashes'])
    write_json(out/'audit_summary.json',summary)
    write_json(out/'authorization.json',dict(authorization='User explicitly selected automatic conservative screening',
        human_reviewed=False,uncertain_target_species_excluded=True,taxonomy_authority_version=2026,
        source=manifest['taxonomy_source'],frozen_before_model_evaluation=True))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
