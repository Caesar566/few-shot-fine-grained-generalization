"""Explicit AUTOMATIC species screening; never claims a human-reviewed manifest.

Keep the existing manual NABirds protocol unchanged. Unknown target names and
ambiguous hierarchy assignments are excluded. Ambiguous CUB names are mapped
to conservative supersets; unresolved CUB training names fail closed.
"""
import csv
import json
import re
from collections import Counter,defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from pattern_recognition_coursework.data.nabirds import (
    load_nabirds_metadata,normalize_bird_name,_ancestors,_safe_image_path,sha256_file,
)

PROTOCOL='nabirds-auto-conservative-ebird2026-officialtest-v1'


def name_keys(name):
    # Explicit English spelling/possessive handling, not fuzzy approval.
    plain=re.sub(r'^\d+\.','',name).replace('_',' ')
    folded=re.sub(r"['’]s\b",'',plain,flags=re.IGNORECASE)
    return {normalize_bird_name(x) for x in (plain,folded)}


def taxonomy_index(path):
    with Path(path).open(encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
    species={r['SPECIES_CODE']:r for r in rows if r['CATEGORY']=='species'}
    index=defaultdict(set)
    for code,row in species.items():
        for key in name_keys(row['COMMON_NAME']):index[key].add(code)
    return rows,species,index


def resolve_cub_train(path,taxonomy_path):
    rows,species,index=taxonomy_index(taxonomy_path);resolved=[];exclusions=set()
    genus_rules={'greatgreyshrike':'Lanius','greenvioletear':'Colibri',
                 'housewren':'Troglodytes','marshwren':'Cistothorus'}
    aliases={'floridajay':'Florida Scrub-Jay','nelsonsharptailedsparrow':"Nelson's Sparrow",
             'capeglossystarling':'Cape Starling','artictern':'Arctic Tern'}
    for line in Path(path).read_text().splitlines():
        if not line.strip():continue
        class_id,name=line.split(maxsplit=1);key=normalize_bird_name(name)
        codes=set();reason='exact/possessive common-name species mapping'
        if key in genus_rules:
            genus=genus_rules[key];codes={c for c,r in species.items() if r['SCIENTIFIC_NAME'].split()[0]==genus}
            reason=f'conservative entire genus {genus}: historical name/split uncertainty'
        else:
            for candidate in name_keys(name):codes.update(index.get(candidate,set()))
        if not codes and key in aliases:
            for candidate in name_keys(aliases[key]):codes.update(index.get(candidate,set()))
            reason='explicit historical/spelling alias: '+aliases[key]
        if not codes:
            # Dataset labels Sayornis/Geococcyx are genera, not species names.
            codes={c for c,r in species.items() if normalize_bird_name(r['SCIENTIFIC_NAME'].split()[0])==key}
            if codes:reason='conservative scientific-genus exclusion'
        if not codes:
            # Herring Gull, White Pelican, Cardinal, Whip-poor-will, House Wren,
            # Marsh Wren etc may have historical splits or generic qualifiers.
            codes={c for c,r in species.items() if any(k.endswith(key) for k in name_keys(r['COMMON_NAME']))}
            if codes:reason='conservative all species with generic/historical name suffix'
        if codes and key=='cardinal':
            genera={species[c]['SCIENTIFIC_NAME'].split()[0] for c in codes}
            codes|={c for c,r in species.items() if r['SCIENTIFIC_NAME'].split()[0] in genera}
            reason='conservative generic cardinal name and all matching genera'
        if not codes:
            # Myrtle Warbler is an eBird infraspecific group of Yellow-rumped.
            for row in rows:
                if row['CATEGORY'] not in ('issf','form','intergrade'):continue
                match=re.match(r'^(.*?)\s*\(([^()]*)\)$',row['COMMON_NAME'])
                if not match:continue
                variant=match.group(2)+' '+match.group(1).split()[-1]
                if name_keys(variant)&name_keys(name):
                    parent=row.get('REPORT_AS','')
                    if parent in species:codes.add(parent)
            if codes:reason='infraspecific name mapped to REPORT_AS species; whole species excluded'
        if not codes:raise ValueError(f'Unresolved CUB Train class: {class_id} {name}; cannot assert semantic non-overlap')
        exclusions.update(codes)
        resolved.append(dict(cub_class_id=int(class_id),cub_name=name,reason=reason,
            excluded_species=[dict(species_code=c,common_name=species[c]['COMMON_NAME'],scientific_name=species[c]['SCIENTIFIC_NAME']) for c in sorted(codes)]))
    return resolved,exclusions,species,index


def build_automatic_manifest(root,cub_train_classes,taxonomy_path,cub_hash_csv,min_images=10):
    if min_images<10:raise ValueError('5-shot + 5-query requires at least ten test images')
    root=Path(root);metadata=load_nabirds_metadata(root)
    cub,resolved_codes,species,index=resolve_cub_train(cub_train_classes,taxonomy_path)
    split={}
    for line in (root/'train_test_split.txt').read_text().splitlines():
        image_id,value=line.split()
        if image_id in split or value not in ('0','1'):raise ValueError('Invalid official NABirds split')
        split[image_id]=int(value)
    if set(split)!=set(metadata['images']):raise ValueError('Official split IDs differ from image index')
    visuals=sorted(set(metadata['labels'].values()));assignments={};audit=[]
    for visual in visuals:
        ancestry=_ancestors(visual,metadata['parents']);matches=[]
        for node in ancestry:
            codes=set()
            for key in name_keys(metadata['classes'][node]):codes.update(index.get(key,set()))
            for code in sorted(codes):matches.append((node,code))
        codes={c for _,c in matches}
        if len(codes)==1:
            code=next(iter(codes));node=matches[-1][0]
            decision='exclude_cub_train_semantic_overlap' if code in resolved_codes else 'candidate_nonoverlap'
            if decision=='candidate_nonoverlap':assignments[visual]=(code,node)
        else:code=None;node=None;decision='exclude_unknown_or_ambiguous_species'
        audit.append(dict(visual_class_id=visual,visual_name=metadata['classes'][visual],
            ancestry=[dict(class_id=n,name=metadata['classes'][n]) for n in ancestry],
            species_code=code,species_ancestor_id=node,decision=decision))
    # No selective inclusion of just one easy sex/age subclass of a species.
    for visual,(code,node) in list(assignments.items()):
        descendants=[v for v in visuals if node in _ancestors(v,metadata['parents'])]
        if any(assignments.get(v,(None,None))[0]!=code for v in descendants):
            for v in descendants:assignments.pop(v,None)
    for row in audit:
        if row['decision']=='candidate_nonoverlap' and row['visual_class_id'] not in assignments:
            row['decision']='exclude_incomplete_or_conflicting_species_descendants'
    with Path(cub_hash_csv).open(encoding='utf-8-sig') as stream:cub_hashes={r['sha256'] for r in csv.DictReader(stream)}
    samples=[]
    for image_id,path in sorted(metadata['images'].items()):
        visual=metadata['labels'][image_id]
        if split[image_id]!=0 or visual not in assignments:continue
        physical=_safe_image_path(root,path)
        if not physical.is_file():raise FileNotFoundError(physical)
        code,node=assignments[visual];digest=sha256_file(physical)
        samples.append(dict(image_id=image_id,path=path,class_id=code,visual_class_id=visual,
                            species_ancestor_id=node,official_split=0,sha256=digest))
    by_hash=defaultdict(list)
    for row in samples:by_hash[row['sha256']].append(row)
    keep=[];removed=[]
    for digest,group in sorted(by_hash.items()):
        if digest in cub_hashes:reason='content_overlaps_any_CUB_split'
        elif len({r['class_id'] for r in group})>1:reason='ambiguous_cross_species_duplicate'
        else:reason=None
        group=sorted(group,key=lambda r:r['image_id'])
        if reason:removed.extend(dict(r,reason=reason) for r in group)
        else:
            keep.append(group[0]);removed.extend(dict(r,reason='same_species_exact_duplicate') for r in group[1:])
    counts=Counter(r['class_id'] for r in keep);included={c for c,n in counts.items() if n>=min_images}
    samples=sorted([r for r in keep if r['class_id'] in included],key=lambda r:r['image_id'])
    if len(included)<5:raise ValueError('Fewer than five eligible automatically screened species')
    classes=[dict(class_id=c,name=species[c]['COMMON_NAME'],scientific_name=species[c]['SCIENTIFIC_NAME'],
                  image_count=counts[c],visual_class_ids=sorted({r['visual_class_id'] for r in samples if r['class_id']==c})) for c in sorted(included)]
    return dict(protocol=PROTOCOL,dataset='nabirds',evaluation_only=True,automatic_screening=True,human_reviewed=False,
        class_unit='automatically taxonomy-matched species',image_split='official_test_only',official_split_value=0,
        taxonomy_version=2026,taxonomy_sha256=sha256_file(taxonomy_path),
        taxonomy_source='https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&locale=en&version=2026',
        cub_train_classes_sha256=sha256_file(cub_train_classes),cub_content_index_sha256=sha256_file(cub_hash_csv),
        source_metadata_sha256={**metadata['hashes'],'train_test_split.txt':sha256_file(root/'train_test_split.txt')},
        protocol_source_sha256=sha256_file(__file__),min_images=min_images,classes=classes,samples=samples,
        cub_semantic_resolution=cub,excluded_cub_taxon_codes=sorted(resolved_codes),visual_audit=audit,
        logical_content_exclusions=removed,insufficient_test_species=[dict(species_code=c,count=n) for c,n in sorted(counts.items()) if c not in included],
        raw_statistics=dict(images=len(metadata['images']),visual_categories=len(visuals),hierarchy_nodes=len(metadata['classes']),
            official_train_images=sum(v==1 for v in split.values()),official_test_images=sum(v==0 for v in split.values())),
        limitations='Automatic conservative name/taxonomy screening, not expert human verification. Unknown target taxa excluded; broad CUB labels exclude supersets. No model results used to define pool.')


class AutomaticNABirdsDataset(Dataset):
    """Separate explicit protocol: never masquerade as manual NABirdsDataset."""
    def __init__(self,root,manifest_path,cub_train_classes,taxonomy_path,transform=None,*,allow_automatic=False):
        if not allow_automatic:raise ValueError('Explicit consent to automatic screening is required')
        manifest=json.loads(Path(manifest_path).read_text());metadata=load_nabirds_metadata(root)
        expected={**metadata['hashes'],'train_test_split.txt':sha256_file(Path(root)/'train_test_split.txt')}
        if manifest.get('protocol')!=PROTOCOL or manifest.get('human_reviewed') is not False or manifest.get('automatic_screening') is not True:
            raise ValueError('Expected explicitly automatic, not human-reviewed protocol')
        if manifest['source_metadata_sha256']!=expected or manifest['taxonomy_sha256']!=sha256_file(taxonomy_path):raise ValueError('NABirds/taxonomy metadata changed')
        if manifest['cub_train_classes_sha256']!=sha256_file(cub_train_classes) or manifest['protocol_source_sha256']!=sha256_file(__file__):raise ValueError('CUB Train/protocol version changed')
        self.data_root=Path(root);self.transform=transform;self.manifest_metadata=manifest
        self.manifest_hash=sha256_file(manifest_path)
        self.classes=[c['name'] for c in manifest['classes']];self.species_codes=[c['class_id'] for c in manifest['classes']]
        self.class_to_idx={c:i for i,c in enumerate(self.species_codes)}
        if set(self.species_codes)&set(manifest['excluded_cub_taxon_codes']):raise ValueError('Semantically overlapping CUB species included')
        self.class_to_indices={i:[] for i in range(len(self.classes))};self.samples=[];self.records=[]
        official={line.split()[0]:int(line.split()[1]) for line in (Path(root)/'train_test_split.txt').read_text().splitlines()}
        audit={r['visual_class_id']:r for r in manifest['visual_audit']};seen=set();content=set()
        for row in manifest['samples']:
            image_id=row['image_id'];visual=row['visual_class_id'];code=row['class_id']
            if image_id in seen or row['sha256'] in content:raise ValueError('Duplicate sample/content in automatic manifest')
            if official[image_id]!=0 or metadata['images'][image_id]!=row['path'] or metadata['labels'][image_id]!=visual:raise ValueError('Official test/label mapping mismatch')
            if audit[visual]['decision']!='candidate_nonoverlap' or audit[visual]['species_code']!=code:raise ValueError('Unapproved automatic species mapping')
            if row['species_ancestor_id'] not in _ancestors(visual,metadata['parents']):raise ValueError('Invalid species hierarchy')
            label=self.class_to_idx[code];path=_safe_image_path(root,row['path'])
            if not path.is_file():raise FileNotFoundError(path)
            seen.add(image_id);content.add(row['sha256']);self.class_to_indices[label].append(len(self.samples))
            self.samples.append((path,label));self.records.append(row)
        for label,row in enumerate(manifest['classes']):
            if len(self.class_to_indices[label])!=row['image_count'] or row['image_count']<10:raise ValueError('Species sample count invalid')

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        path,label=self.samples[index]
        with Image.open(path) as source:image=source.convert('RGB')
        return (self.transform(image) if self.transform else image),label
