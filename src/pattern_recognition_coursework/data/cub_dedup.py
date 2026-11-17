"""Exact-byte CUB audit and opt-in derived indexes; never mutate source images.

Index schema stays compatible with existing --splits-dir entry points.
Ambiguous cross-class/cross-split duplicate groups are retained and block use.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

SCHEMA = 'cub-exact-dedup-v1'
SPLITS = ('train', 'val', 'test')
METADATA = ('classes.txt', 'images.txt', 'image_class_labels.txt')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.gif', '.ppm'}


def sha256_file(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _table(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        if key in rows:
            raise ValueError(f'Duplicate ID in {path}: {key}')
        rows[key] = value
    if not rows:
        raise ValueError(f'Empty metadata: {path}')
    return rows


def _inside_images(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or '..' in Path(relative).parts or '\\' in relative:
        raise ValueError(f'Unsafe image path: {relative}')
    path = (root / 'images' / relative).resolve()
    if not path.is_relative_to((root / 'images').resolve()):
        raise ValueError(f'Image path escapes images root: {relative}')
    return path


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_duplicate_groups(inventory: list[dict]) -> tuple[list[dict], set[str]]:
    """Pure deterministic policy; conflicting indexed rows are never dropped."""
    by_hash = defaultdict(list)
    for row in inventory:
        by_hash[row['sha256']].append(row)
    groups = []
    excluded = set()
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda r: r['relative_path'])
        indexed = [r for r in members if r['split'] in SPLITS]
        cross_class = len({r['class_name'] for r in members}) > 1
        cross_split = len({r['split'] for r in indexed}) > 1
        review = cross_class or cross_split
        keeper = None
        if not review and indexed:
            keeper = min(indexed, key=lambda r: (int(r['image_id']), r['relative_path']))
            excluded.update(r['image_id'] for r in indexed if r is not keeper)
        groups.append({'group_id': len(groups)+1, 'sha256': digest, 'members': members,
                       'cross_class': cross_class, 'cross_split': cross_split,
                       'requires_review': review,
                       'keeper_image_id': keeper['image_id'] if keeper else None,
                       'keeper_relative_path': keeper['relative_path'] if keeper else None})
    return groups, excluded


def audit_and_clean(data_root: Path, source_splits: Path, clean_splits: Path,
                    report_dir: Path) -> dict:
    """Read all disk images, then publish new indexes with source hashes.

    Destinations must not contain any files. Both are outside the raw data tree
    and outside the source split directory. No overwrite/removal API is used.
    """
    data_root, source_splits = Path(data_root).resolve(), Path(source_splits).resolve()
    clean_splits, report_dir = Path(clean_splits).resolve(), Path(report_dir).resolve()
    if not data_root.is_dir() or not source_splits.is_dir():
        raise FileNotFoundError('Missing data root or source split directory')
    for destination in (clean_splits, report_dir):
        if (destination.is_relative_to(data_root) or destination == source_splits
                or source_splits.is_relative_to(destination)):
            raise ValueError('Destination must not overwrite or contain source data/splits')
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f'Refusing to overwrite existing output: {destination}')
    if clean_splits == report_dir or clean_splits.is_relative_to(report_dir) or report_dir.is_relative_to(clean_splits):
        raise ValueError('Clean indexes and report must have separate directories')

    classes = _table(data_root/'classes.txt')
    images = _table(data_root/'images.txt')
    image_labels = _table(data_root/'image_class_labels.txt')
    if images.keys() != image_labels.keys():
        raise ValueError('Official images and labels do not have identical IDs')
    source_hashes = {f'{s}_{kind}.txt': sha256_file(source_splits/f'{s}_{kind}.txt')
                     for s in SPLITS for kind in ('classes', 'images')}
    metadata_hashes = {name: sha256_file(data_root/name) for name in METADATA}
    indexed = {}
    seen_ids, seen_classes = set(), set()
    raw_lines = {}
    for split in SPLITS:
        split_classes = _table(source_splits/f'{split}_classes.txt')
        if seen_classes.intersection(split_classes):
            raise ValueError('Original class manifests overlap across splits')
        seen_classes.update(split_classes)
        if any(classes.get(cid) != name for cid, name in split_classes.items()):
            raise ValueError('Split class manifest disagrees with official classes')
        raw_lines[split] = (source_splits/f'{split}_images.txt').read_bytes().splitlines(keepends=True)
        actual_classes = set()
        for line in raw_lines[split]:
            if not line.strip():
                continue
            parts = line.decode('utf-8').strip().split(maxsplit=2)
            if len(parts) != 3:
                raise ValueError(f'Malformed {split} image row')
            image_id, cname, relative = parts
            if not image_id.isdecimal() or image_id in seen_ids or relative in indexed:
                raise ValueError('Repeated/invalid official image ID or path')
            cid = image_labels.get(image_id)
            if (images.get(image_id) != relative or split_classes.get(cid) != cname
                    or relative.split('/')[0] != cname):
                raise ValueError('Image manifest disagrees with official metadata')
            path = _inside_images(data_root, relative)
            if not path.is_file():
                raise FileNotFoundError(path)
            indexed[relative] = {'image_id': image_id, 'class_id': cid, 'class_name': cname, 'split': split}
            seen_ids.add(image_id)
            actual_classes.add(cid)
        if actual_classes != split_classes.keys():
            raise ValueError('Empty/missing class in image manifests')
    if seen_ids != images.keys() or seen_classes != classes.keys():
        raise ValueError('Original manifests must cover all official images/classes')

    inventory, skipped = [], []
    for path in sorted((data_root/'images').rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(data_root/'images').as_posix()
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            skipped.append(relative)
            continue
        _inside_images(data_root, relative)
        row = indexed.get(relative)
        if row is None:
            cname = relative.split('/')[0]
            row = {'image_id': '', 'class_id': next((cid for cid,n in classes.items() if n == cname), ''),
                   'class_name': cname, 'split': 'unindexed'}
        inventory.append({**row, 'relative_path': relative, 'original_path': str(path),
                          'sha256': sha256_file(path), 'size_bytes': path.stat().st_size})
    found = {row['relative_path'] for row in inventory}
    if indexed.keys() - found:
        raise ValueError('Indexed images were not included in the full image scan')
    groups, excluded = classify_duplicate_groups(inventory)
    unresolved = [g['group_id'] for g in groups if g['requires_review']]
    clean_rows = [r for r in inventory if r['split'] in SPLITS and r['image_id'] not in excluded]
    clean_hashes = Counter(r['sha256'] for r in clean_rows)
    ready = not unresolved and max(clean_hashes.values()) == 1
    stats = {}
    for split in SPLITS:
        original = [r for r in inventory if r['split'] == split]
        cleaned = [r for r in clean_rows if r['split'] == split]
        before_counts = Counter(r['class_name'] for r in original)
        after_counts = Counter(r['class_name'] for r in cleaned)
        if before_counts.keys() != after_counts.keys():
            raise ValueError('Logical dedup unexpectedly removed a whole class')
        stats[split] = {'original_images': len(original), 'clean_images': len(cleaned),
                        'excluded_images': len(original)-len(cleaned), 'classes': len(after_counts),
                        'min_images_per_class_before': min(before_counts.values()),
                        'min_images_per_class_after': min(after_counts.values())}

    clean_splits.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    # Fail closed while publishing, even if generation is interrupted.
    marker = clean_splits/'dedup_manifest.json'
    marker.write_text(json.dumps({'schema': SCHEMA, 'ready_for_training_and_evaluation': False,
                                 'state': 'publishing'}, indent=2)+'\n', encoding='utf-8')
    for split in SPLITS:
        (clean_splits/f'{split}_classes.txt').write_bytes((source_splits/f'{split}_classes.txt').read_bytes())
        kept_lines = [line for line in raw_lines[split]
                      if not line.strip() or line.decode('utf-8').split(maxsplit=1)[0] not in excluded]
        (clean_splits/f'{split}_images.txt').write_bytes(b''.join(kept_lines))
    content_fields = ['image_id', 'class_name', 'split', 'relative_path', 'sha256']
    _write_csv(clean_splits/'image_hashes.csv',
               [{k:r[k] for k in content_fields} for r in clean_rows], content_fields)
    output_hashes = {p.name: sha256_file(p) for p in sorted(clean_splits.iterdir()) if p.name != marker.name}
    manifest = {'schema': SCHEMA, 'state': 'complete',
                'ready_for_training_and_evaluation': ready,
                'policy': 'same-class same-split: minimum numeric official image_id then POSIX path; conflicts retained for human review',
                'source_splits_path': str(source_splits), 'data_root_at_audit': str(data_root),
                'source_split_sha256': source_hashes, 'source_metadata_sha256': metadata_hashes,
                'output_sha256': output_hashes, 'excluded_image_ids': sorted(excluded,key=int),
                'unresolved_review_group_ids': unresolved, 'split_statistics': stats}
    marker.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    duplicate_rows = []
    for group in groups:
        for row in group['members']:
            if row['split'] == 'unindexed':
                decision = 'unindexed_not_added'
            elif group['requires_review']:
                decision = 'retained_pending_review'
            elif row['image_id'] in excluded:
                decision = 'excluded_from_derived_index'
            else:
                decision = 'keep'
            duplicate_rows.append({'group_id': group['group_id'], **row,
                'cross_class': group['cross_class'], 'cross_split': group['cross_split'],
                'requires_review': group['requires_review'], 'decision': decision,
                'keeper_image_id': group['keeper_image_id'], 'keeper_relative_path': group['keeper_relative_path']})
    inventory_fields = ['image_id','class_id','class_name','split','relative_path','original_path','sha256','size_bytes']
    duplicate_fields = ['group_id']+inventory_fields+['cross_class','cross_split','requires_review','decision','keeper_image_id','keeper_relative_path']
    _write_csv(report_dir/'full_inventory.csv', inventory, inventory_fields)
    _write_csv(report_dir/'duplicate_samples.csv', duplicate_rows, duplicate_fields)
    _write_csv(report_dir/'review_required.csv', [r for r in duplicate_rows if r['requires_review']], duplicate_fields)
    _write_csv(report_dir/'unindexed_images.csv', [r for r in inventory if r['split']=='unindexed'], inventory_fields)
    class_rows = []
    for split in SPLITS:
        before_counts = Counter(r['class_name'] for r in inventory if r['split']==split)
        after_counts = Counter(r['class_name'] for r in clean_rows if r['split']==split)
        class_rows.extend({'split': split, 'class_name': c, 'original_images': n,
                           'clean_images': after_counts[c], 'excluded_images': n-after_counts[c]}
                          for c,n in sorted(before_counts.items()))
    _write_csv(report_dir/'class_counts_comparison.csv', class_rows,
               ['split','class_name','original_images','clean_images','excluded_images'])
    summary = {'schema': SCHEMA, 'physical_images_scanned': len(inventory),
        'official_indexed_images': len(indexed), 'unindexed_images': len(inventory)-len(indexed),
        'unique_disk_image_contents': len({r['sha256'] for r in inventory}),
        'duplicate_groups': len(groups), 'redundant_disk_images': sum(len(g['members'])-1 for g in groups),
        'affected_classes': sorted({r['class_name'] for g in groups for r in g['members']}),
        'cross_class_groups': sum(g['cross_class'] for g in groups),
        'cross_split_groups': sum(g['cross_split'] for g in groups),
        'review_required_groups': len(unresolved), 'excluded_official_index_images': len(excluded),
        'clean_index_images': len(clean_rows), 'clean_index_unique_contents': len(clean_hashes),
        'ready_for_training_and_evaluation': ready, 'split_statistics': stats,
        'clean_splits_dir': str(clean_splits), 'skipped_non_image_files': skipped,
        'original_images_modified_moved_or_deleted': False}
    (report_dir/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return summary


def validate_clean_index(data_root: Path, split_file: Path, *, required: bool = False) -> dict | None:
    """Recognize a derived directory and reject unready/stale index metadata.

    Original plain-text manifests remain opt-in compatible and are not silently
    redirected. Metadata/CSV integrity is checked; this does not rehash all image
    bytes on every Dataset construction (the audit records that snapshot).
    """
    split_file = Path(split_file)
    marker = split_file.parent/'dedup_manifest.json'
    if not marker.exists():
        if required:
            raise ValueError('A verified clean index with dedup_manifest.json is required')
        return None
    manifest = json.loads(marker.read_text(encoding='utf-8'))
    if (manifest.get('schema') != SCHEMA or manifest.get('state') != 'complete'
            or manifest.get('ready_for_training_and_evaluation') is not True
            or manifest.get('unresolved_review_group_ids')):
        raise ValueError('Clean CUB index is incomplete or has unresolved duplicate groups requiring review')
    output_hashes = manifest.get('output_sha256', {})
    required_names = [f'{s}_{kind}.txt' for s in SPLITS for kind in ('classes','images')]+['image_hashes.csv']
    if split_file.name not in [f'{s}_images.txt' for s in SPLITS]:
        raise ValueError('Clean CUB index requires a named train/val/test image manifest')
    for name in required_names:
        path = split_file.parent/name
        if not path.is_file() or output_hashes.get(name) != sha256_file(path):
            raise ValueError(f'Clean CUB index hash mismatch: {name}')
    for name in METADATA:
        path = Path(data_root)/name
        if not path.is_file() or manifest.get('source_metadata_sha256',{}).get(name) != sha256_file(path):
            raise ValueError(f'Clean CUB index does not match source metadata: {name}')
    return manifest
