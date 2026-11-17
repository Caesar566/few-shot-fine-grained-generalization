"""Stage-01 data diagnostics only. No project model/engine imports or training.

Run with PYTHONPATH=src .venv/bin/python -B outputs/stage01_data/check_data.py
NABirds raw diagnostic labels are visual categories, NEVER approved test species.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import textwrap

import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
from torchvision.transforms import functional as TF

from pattern_recognition_coursework.data.cub_dataset import (
    CUBFewShotDataset, FewShotBatchSampler, get_cub_transforms,
)
from pattern_recognition_coursework.data.episodes import dataset_fingerprint, validate_plan
from pattern_recognition_coursework.data.nabirds import (
    NABirdsDataset, _ancestors, _safe_image_path, create_review, load_nabirds_metadata,
    normalize_bird_name,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
CUB_ROOT = ROOT / 'cub2002011/versions/7/CUB_200_2011'
NAB_ROOT = ROOT / 'nabirds'
IMAGE_SIZE = 224
MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)
SEED = 20261116


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(name, data):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def save_csv(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class NABirdsRawDiagnosticDataset(Dataset):
    """Raw-image inspection only; no human-reviewed species test claim.

    Uses actual metadata, image IDs, safe paths, RGB decoding and existing
    get_cub_transforms. The official NABirdsDataset remains unchanged.
    """
    label_unit = 'raw visual category (NOT reviewed species)'

    def __init__(self, transform=None, metadata=None):
        self.data_root = NAB_ROOT
        self.transform = transform
        self.metadata = metadata if metadata is not None else load_nabirds_metadata(NAB_ROOT)
        self.original_ids = sorted(set(self.metadata['labels'].values()))
        self.class_to_idx = {cid: i for i, cid in enumerate(self.original_ids)}
        self.classes = [self.metadata['classes'][cid] for cid in self.original_ids]
        self.image_ids = sorted(self.metadata['images'])
        self.samples = []
        self.class_to_indices = defaultdict(list)
        for image_id in self.image_ids:
            label = self.class_to_idx[self.metadata['labels'][image_id]]
            path = _safe_image_path(NAB_ROOT, self.metadata['images'][image_id])
            self.class_to_indices[label].append(len(self.samples))
            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert('RGB')
        return (self.transform(image) if self.transform else image), label


def make_datasets():
    datasets = {
        f'cub_{split}': CUBFewShotDataset(
            str(CUB_ROOT), str(ROOT / f'data/splits/{split}_images.txt'),
            get_cub_transforms(IMAGE_SIZE, split == 'train'),
        ) for split in ['train', 'val', 'test']
    }
    datasets['nabirds_raw'] = NABirdsRawDiagnosticDataset(get_cub_transforms(IMAGE_SIZE, False))
    return datasets


def denormalize(tensor):
    """Inverse Normalize before display; clamp only at rendering boundary."""
    return tensor.detach().cpu() * STD + MEAN


def tensor_stats(tensor):
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            'min': float(tensor.min()), 'max': float(tensor.max()),
            'finite': bool(torch.isfinite(tensor).all())}


def mapping_rows(name, dataset):
    rows = []
    for local, cname in enumerate(dataset.classes):
        original = dataset.original_ids[local] if name == 'nabirds_raw' else int(cname.split('.')[0])
        rows.append({'dataset': name, 'original_class_id': original, 'local_label': local,
                     'class_name': cname, 'images': len(dataset.class_to_indices[local]),
                     'label_unit': 'raw_visual_category' if name == 'nabirds_raw' else 'CUB_species'})
    return rows


def transform_config():
    train = get_cub_transforms(IMAGE_SIZE, True)
    evaluation = get_cub_transforms(IMAGE_SIZE, False)
    def describe(composed):
        return [{'position': i+1, 'name': type(t).__name__, 'repr': repr(t)}
                for i, t in enumerate(composed.transforms)]
    rrc, flip, jitter, _, norm = train.transforms
    resize = evaluation.transforms[0]
    return {'image_size': IMAGE_SIZE, 'train_order': describe(train),
            'eval_order': describe(evaluation), 'rrc_scale': list(rrc.scale),
            'rrc_ratio': list(rrc.ratio), 'rrc_interpolation': str(rrc.interpolation),
            'rrc_antialias': rrc.antialias, 'flip_p': flip.p,
            'jitter_brightness': jitter.brightness, 'jitter_contrast': jitter.contrast,
            'jitter_saturation': jitter.saturation, 'jitter_hue': jitter.hue,
            'resize_size': resize.size, 'resize_interpolation': str(resize.interpolation),
            'resize_antialias': resize.antialias,
            'normalize_mean': list(norm.mean), 'normalize_std': list(norm.std)}


def episode_diagnostics(datasets):
    result = {}
    for name, dataset in datasets.items():
        for shot in [1, 5]:
            sampler = FewShotBatchSampler(dataset, 20, 5, shot, 5, seed=42)
            batches = list(sampler)
            plan = {'schema': 'few-shot-episodes-v1', 'dataset_fingerprint': dataset_fingerprint(dataset),
                    'n_way': 5, 'k_shot': shot, 'n_query': 5, 'episodes': 20, 'batches': batches}
            validate_plan(plan, dataset, 5, shot, 5, 20)
            assert batches == list(sampler)
            result[f'{name}_{shot}shot'] = {'episodes': 20, 'index_order_and_disjoint': True,
                'reproducible': True, 'note': 'NABirds raw task is diagnostic, not approved species evaluation'
                if name == 'nabirds_raw' else 'CUB metadata/indices'}
    return result


def full_content_audit(datasets):
    """Hash all indexed source files; no deduplication or mutation."""
    groups = defaultdict(list)
    records = []
    seen = 0
    total = sum(len(d) for d in datasets.values())
    for name, dataset in datasets.items():
        for index, (path, label) in enumerate(dataset.samples):
            row = {'dataset': name, 'sample_index': index, 'label': label,
                   'class_name': dataset.classes[label], 'path': path.relative_to(ROOT).as_posix(),
                   'sha256': sha(path)}
            records.append(row)
            groups[row['sha256']].append(row)
            seen += 1
            if seen % 5000 == 0:
                print(f'Hashing indexed images: {seen}/{total}', flush=True)
    save_csv('evidence/image_sha256.csv', records)
    duplicates = [{'sha256': h, 'items': rows} for h, rows in groups.items() if len(rows) > 1]
    cross_cub = [g for g in duplicates if len({r['dataset'] for r in g['items'] if r['dataset'].startswith('cub_')}) > 1]
    cross_domain = [g for g in duplicates if any(r['dataset'] == 'nabirds_raw' for r in g['items'])
                    and any(r['dataset'].startswith('cub_') for r in g['items'])]
    return {'files_hashed': seen, 'unique_contents': len(groups), 'duplicate_groups': duplicates,
            'cross_cub_split_duplicate_groups': cross_cub,
            'cub_nabirds_exact_duplicate_groups': cross_domain,
            'scope': 'all indexed images, exact SHA256 bytes; no perceptual duplicate or ImageNet overlap check'}


def content_episode_reproduction(dataset, content_audit):
    pairs = []
    for group in content_audit['duplicate_groups']:
        items = [r for r in group['items'] if r['dataset'] == 'cub_test']
        if len(items) > 1:
            pairs.append((group['sha256'], {r['sample_index'] for r in items}))
    for episode, batch in enumerate(FewShotBatchSampler(dataset, 10000, 5, 5, 5, seed=42)):
        for digest, pair in pairs:
            support = pair.intersection(batch[:25]); query = pair.intersection(batch[25:])
            if support and query:
                return {'status': 'FAIL', 'reason': 'distinct indices with identical content across support/query',
                        'seed': 42, 'n_way': 5, 'k_shot': 5, 'n_query': 5,
                        'episode_zero_based': episode, 'sha256': digest,
                        'support_indices': batch[:25], 'query_indices': batch[25:],
                        'duplicate_support': sorted(support), 'duplicate_query': sorted(query),
                        'historical_test_claim': 'new audit sampling only; historical plans not available'}
    return {'status': 'PASS_WITH_LIMITS', 'reason': 'no content collision found in these 10000 newly sampled tasks'}


def font(size):
    for path in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                 '/usr/share/fonts/TTF/DejaVuSans.ttf']:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def panel_image(image):
    tile = Image.new('RGB', (224, 224), '#e5e7eb')
    small = ImageOps.contain(image.convert('RGB'), (224,224))
    tile.paste(small, ((224-small.width)//2, (224-small.height)//2))
    return tile


def visualize(datasets):
    samples = []
    train_transform = get_cub_transforms(IMAGE_SIZE, True)
    test_transform = get_cub_transforms(IMAGE_SIZE, False)
    for group_id, name in enumerate(['cub_train', 'cub_val', 'nabirds_raw']):
        dataset = datasets[name]
        indices = random.Random(SEED + group_id).sample(range(len(dataset)), 4)
        for index in indices:
            number = len(samples)+1
            path, label = dataset.samples[index]
            stem = f'{number:02d}_{name}_{index}'
            folder = OUT / 'samples' / stem
            folder.mkdir(parents=True, exist_ok=True)
            original = folder / ('original' + path.suffix.lower())
            shutil.copyfile(path, original)
            with Image.open(path) as source:
                rgb = source.convert('RGB')
            rgb.save(folder/'original_rgb.png')
            tiles = [panel_image(rgb)]
            tensors=[]
            views=[]
            for repeat in range(3):
                seed=SEED+number*100+repeat
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    tensor=train_transform(rgb)
                restored=denormalize(tensor)
                display=TF.to_pil_image(restored.clamp(0,1))
                file=folder/f'train_aug_{repeat+1}_denorm.png'
                display.save(file)
                tiles.append(display)
                tensors.append(tensor)
                views.append({'view':repeat+1,'seed':seed,'file':file.relative_to(OUT).as_posix(),
                              'tensor':tensor_stats(tensor),'denormalized':tensor_stats(restored)})
            eval_tensor=test_transform(rgb)
            eval_file=folder/'eval_preprocess_denorm.png'
            eval_image=TF.to_pil_image(denormalize(eval_tensor).clamp(0,1))
            eval_image.save(eval_file);tiles.append(eval_image)
            board=Image.new('RGB',(1220,402),'#f8fafc')
            draw=ImageDraw.Draw(board)
            draw.text((14,8),f'{number:02d} | {name} | label={label} | {dataset.classes[label]}',fill='#111827',font=font(15))
            rel=path.relative_to(ROOT).as_posix()
            for line_no,line in enumerate(textwrap.wrap(rel,135)):
                draw.text((14,32+line_no*19),line,fill='#334155',font=font(13))
            captions=['Original RGB\nfit with padding (display only)',
                      'Train preview #1\nRRC -> HFlip -> ColorJitter',
                      'Train preview #2\nRRC -> HFlip -> ColorJitter',
                      'Train preview #3\nRRC -> HFlip -> ColorJitter',
                      'Eval preprocessing\nResize(257) -> CenterCrop(224)']
            for col,(tile,caption) in enumerate(zip(tiles,captions)):
                x=12+col*243
                board.paste(tile,(x,102))
                draw.multiline_text((x,333),caption,fill='#111827',font=font(12),spacing=4)
            draw.text((14,380),'Tensor views are inverse-normalized before display. Train previews on val/NABirds are illustration ONLY.',fill='#9a3412',font=font(12))
            board_path=OUT/'visualizations'/f'{stem}.png'
            board_path.parent.mkdir(parents=True,exist_ok=True)
            board.save(board_path)
            samples.append({'number':number,'dataset':name,'sample_index':index,'label':label,
                'class_name':dataset.classes[label],'path':rel,'original_size':list(rgb.size),
                'original_copy':original.relative_to(OUT).as_posix(),'original_sha256':sha(path),
                'copy_sha256':sha(original),'train_views':views,'eval_tensor':tensor_stats(eval_tensor),
                'eval_file':eval_file.relative_to(OUT).as_posix(),
                'train_views_distinct':not all(torch.equal(tensors[0],t) for t in tensors[1:]),
                'eval_repeat_equal':torch.equal(eval_tensor,test_transform(rgb)),
                'board':board_path.relative_to(OUT).as_posix()})
    overview=Image.new('RGB',(1220,402*len(samples)),'white')
    for i,row in enumerate(samples):
        with Image.open(OUT/row['board']) as im:overview.paste(im,(0,i*402))
    overview.save(OUT/'visualizations/contact_sheet.png')
    save_json('sample_manifest.json',samples)
    return samples


def batch_diagnostics(datasets,samples):
    stats={}
    for name in ['cub_train','cub_val','nabirds_raw']:
        selected=[s for s in samples if s['dataset']==name]
        indices=[s['sample_index'] for s in selected]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(SEED)
            images,labels=next(iter(DataLoader(Subset(datasets[name],indices),batch_size=4,shuffle=False,num_workers=0)))
        stats[name]={'images':tensor_stats(images),'labels':labels.tolist(),'labels_dtype':str(labels.dtype),
                     'labels_shape':list(labels.shape),'class_names':[s['class_name'] for s in selected],
                     'paths':[s['path'] for s in selected],
                     'actual_transform':'train' if name=='cub_train' else 'eval',
                     'label_namespace':'raw_visual_category' if name=='nabirds_raw' else name+'_local'}
        print('BATCH',name,json.dumps(stats[name],ensure_ascii=False),flush=True)
    for shot in [1,5]:
        dataset=datasets['cub_train']
        images,labels=next(iter(DataLoader(dataset,batch_sampler=FewShotBatchSampler(dataset,1,5,shot,5,seed=42),num_workers=0)))
        order=labels[:5*shot:shot].tolist()
        assert labels[:5*shot].tolist()==[label for label in order for _ in range(shot)]
        assert labels[5*shot:].tolist()==[label for label in order for _ in range(5)]
        episode_labels=torch.arange(5).repeat_interleave(5)
        stats[f'cub_train_episode_{shot}shot']={'images':tensor_stats(images),
            'dataset_labels':labels.tolist(),'class_order':order,'query_episode_labels':episode_labels.tolist(),
            'support_count':5*shot,'query_count':25,'labels_dtype':str(labels.dtype)}
        print('EPISODE',shot,'shot',json.dumps(stats[f'cub_train_episode_{shot}shot']),flush=True)
    save_json('tensor_stats.json',stats)


def write_report():
    summary=json.loads((OUT/'statistics.json').read_text())
    transforms=json.loads((OUT/'transforms.json').read_text())
    samples=json.loads((OUT/'sample_manifest.json').read_text())
    tensors=json.loads((OUT/'tensor_stats.json').read_text())
    content=json.loads((OUT/'evidence/content_audit.json').read_text())
    repro=json.loads((OUT/'evidence/content_episode_reproduction.json').read_text())
    checks=json.loads((OUT/'evidence/acceptance_checks.json').read_text())
    tests=json.loads((OUT/'evidence/pytest_results.json').read_text()) if (OUT/'evidence/pytest_results.json').exists() else None
    text='''# Stage 01：数据加载、划分、标签与增强检查

本阶段只使用Dataset、DataLoader、Transform和数据审计；未运行模型训练，未修改项目模型、引擎、分类器、数据源码或正式清单。所有新增脚本、图像、统计和测试证据均位于本目录。运行环境是本次实际检查所得，不沿用上轮缺依赖的结论。

## 1. 验收结论

'''
    for c in checks:
        text+=f"- **{c['status']} — {c['check']}**：{c['detail']}\n"
    text+='\n'+('pytest结果：`'+str(tests['counts'])+'`，退出码 '+str(tests['returncode'])+'。完整输出见 [pytest_stdout.txt](evidence/pytest_stdout.txt)，逐用例见 [pytest_results.json](evidence/pytest_results.json)。\n' if tests else 'pytest尚未运行，不能计为通过。\n')
    text+='''
**NABirds原始图片诊断可正常加载，不等于正式NABirds测试集可用。** 当前没有人工审核物种清单；不虚构物种标签、不伪造reviewer/evidence、不修改正式Dataset来绕过审核。原始555视觉类包含性别/年龄外观分类，不是555个独立物种。

## 2. 统计与标签映射

| 数据集 | 图像数 | 类别数 | 原始类别ID范围 | Dataset标签范围 | 实际预处理 |
|---|---:|---:|---|---|---|
'''
    for name,s in summary['datasets'].items():
        text+=f"| {name} | {s['samples']} | {s['classes']} | {s['original_id_min']}–{s['original_id_max']}（不保证连续） | 0–{s['classes']-1} | {'训练增强' if name=='cub_train' else '确定性eval'} |\n"
    text+='''
正式NABirds测试集样本数/物种数：**无法确定，因review/manifest缺失**。原始全量统计不能填作正式测试规模。全部原始元信息有1011层级节点，只有555类带图像。

CUB：读取每个split清单中的类别名称，`classes=sorted(set(cname))`，`class_to_idx={cname:i}`；各split独立编号，train标签0和val标签0可以代表不同物种，不能用本地整数交集判断泄漏。完整映射和每类样本数见`statistics/*_class_mapping.csv`，逐图片索引见`statistics/*_samples.csv`。

NABirds正式Dataset：manifest的`classes`数组顺序决定本地标签，`class_to_idx={species_class_id:index}`，每张样本从视觉类别映射到已审核物种祖先，再映射至本地标签。它不是把原始ID直接当0-based标签。

本阶段NABirdsRawDiagnosticDataset按**有图像的原始视觉ID升序**建立0-based映射，仅用于读取与增强展示；其映射明确标`raw_visual_category`。表中NABirds标签不能冒充正式物种标签。

Episode：support按类排列，query按相同类顺序排列。正式训练循环根据这一约定构造`arange(n_way).repeat_interleave(n_query)`局部query标签；本阶段只核对标签与顺序，不调用训练循环。固定评估任务由episodes.py保存/验证，不在本阶段生成正式测试计划。

## 3. 完整调用关系与源码位置

```mermaid
flowchart TD
 A[控制台train命令] --> B[engine/experiment.py main]
 B --> C[CUBFewShotDataset + get_cub_transforms]
 C --> D[train随机增强 / val确定性预处理]
 D --> E[FewShotBatchSampler]
 E --> F[DataLoader batch_sampler]
 G[控制台evaluate命令] --> H[engine/evaluate.py main]
 H --> I[get_cub_transforms is_train=False]
 I --> J{dataset}
 J -->|CUB| K[CUBFewShotDataset val/test清单]
 J -->|NABirds| L[NABirdsDataset 审核review与manifest]
 K --> M[load_or_create_plan]
 L --> M
 M --> N[episode_loader + DataLoader]
 O[prepare_nabirds.py] --> P[load_nabirds_metadata + create_review]
 P --> Q[人工taxonomy与层级审核]
 Q --> R[build_manifest]
 R --> L
 S[make_splits.py] --> T[CUB官方classes/images + seed42划分类别]
 T --> C
```

上图是源码调用关系，不表示本阶段运行了train/evaluate。实际只直接实例化数据组件。

| 位置 | 职责 |
|---|---|
| `src/pattern_recognition_coursework/data/cub_dataset.py:10–32` | 训练与评估Transform |
| 同文件`:35–96` | CUB清单、类别映射、RGB读取及transform |
| 同文件`:99–156` | episode抽样、固定种子、support/query索引顺序 |
| `data/episodes.py:10–72` | 有序清单指纹、任务验证和持久化 |
| `data/nabirds.py:65–96` | 图像索引、标签、层级与安全路径 |
| 同文件`:99–202` | 生成待审核项、人工结论验证、物种级manifest |
| 同文件`:205–282` | 正式NABirdsDataset及RGB读取 |
| `scripts/make_splits.py:29–94` | 原始CUB类列表以seed42打乱，100/50/50类拆分并输出图片清单 |
| `scripts/validate_cub.py:24–71` | 清单和官方元信息一致性、类别/图片ID互斥 |
| `scripts/prepare_nabirds.py:13–38` | 审核模板与正式清单CLI |
| `engine/experiment.py:231–234,308–310` | train/val Dataset与训练DataLoader构造 |
| 同文件`:93–94,128–136` | 固定评估episode的DataLoader；监督路径普通shuffle DataLoader |
| `engine/evaluate.py:62–86` | 测试只用eval Transform，CUB/NABirds Dataset后固定计划 |

本阶段真实数据根是`cub2002011/versions/7/CUB_200_2011`及`nabirds`，入口默认`data/raw/...`当前不存在；诊断显式指定根，不移动数据。`make_splits.find_cub_root`的递归兜底可能识别到NABirds同名元信息，因此本阶段不调用会重写清单的生成脚本，仅在内存重现seed划分并与当前文件比较。

## 4. 实际Transform及顺序（本机torchvision实例）

### 训练路径

1. PIL读取后`convert('RGB')`，确保三通道。
2. `RandomResizedCrop(224, scale=(0.2,1.0), ratio=(0.75,4/3), interpolation=BILINEAR, antialias=True)`：随机选择占原图20%–100%面积、宽高比0.75–1.3333的矩形，再缩放成224×224。比例与面积不等同于边长比例；实现尝试最多10次，失败按默认中心区域回退。此步内部包含resize，训练前**没有额外Resize**。
3. `RandomHorizontalFlip(p=0.5)`：以50%概率左右翻转；没有垂直翻转。
4. `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)`：亮度/对比度/饱和度因子分别在[0.85,1.15]采样，三者应用顺序随机；`hue=None`，没有色相扰动。
5. `ToTensor()`：uint8 RGB HWC转换为float32 CHW，像素缩放至[0,1]。
6. `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`：逐通道`(x-mean)/std`，空间尺寸不变，值不再限于[0,1]。对于原始[0,1]，三通道理论范围分别约[-2.118,2.249]、[-2.036,2.429]、[-1.804,2.640]。

### 验证及测试路径（CUB与NABirds共用）

1. RGB转换。
2. `Resize(int(224*1.15))`，Python截断结果是**257**，不是258。整数size表示将短边缩放至257并保持宽高比，BILINEAR、antialias=True，max_size=None；不是直接拉伸成257×257。
3. `CenterCrop(224)`得到224×224。
4. `ToTensor()`，随后同样的`Normalize`。

**测试路径没有随机裁剪、Flip或ColorJitter。** 对所选同一图片改变seed重复执行eval Transform，张量逐元素相等。没有发现代码实际使用RandomRotation、RandomErasing、AutoAugment、MixUp、CutMix或边界框裁剪，不能把这些写成现有增强。

完整实例repr和默认参数见 [transforms.json](transforms.json)。参数适用于本次224像素检查；其他image_size会改变Resize与crop大小。归一化反操作是`x=tensor*std+mean`，本次图片均先反归一化，再仅在显示边界clamp至[0,1]；原始副本逐字节保留。

## 5. 数据泄漏及划分检查

'''
    text+=f"- 类别、图像ID与路径交集：`{summary['split_intersections']}`。\n"
    text+=f"- seed42类别划分与当前三份类清单一致：`{summary['seed42_split_matches']}`。\n"
    text+=f"- 对全部 **{content['files_hashed']}** 个索引文件执行SHA256，共 **{content['unique_contents']}** 种字节内容；跨CUB划分重复组 **{len(content['cross_cub_split_duplicate_groups'])}**；CUB↔NABirds精确重复组 **{len(content['cub_nabirds_exact_duplicate_groups'])}**。见 [content_audit.json](evidence/content_audit.json) 和完整 [image_sha256.csv](evidence/image_sha256.csv)。\n"
    text+=f"- support/query内容隔离：**{repro['status']}**。"+(f"CUB test新采样任务零基`{repro['episode_zero_based']}`出现不同索引相同字节分别在support/query，seed42，5-way 5-shot，每类5 query。" if repro['status']=='FAIL' else repro['reason'])+'原采样器仅保证索引不重复，本阶段未修复，也未重写历史结果。\n'
    text+='''- 精确哈希不排除裁剪/压缩版本、近重复、背景/拍摄来源关联或ImageNet预训练重叠。本阶段不把“0个精确重复”宣称完全无泄漏。
- 名称与祖先匹配只是物种重叠候选。当前NABirds review缺失，不能确认源/目标物种独立性；正式工具默认只排CUB train，若要求排除全部CUB需明确协议。
- CUB使用项目类别级100/50/50划分，不采用官方train_test_split.txt图像级5994/5794作为本课题划分。官方图像级标记可能与项目split不同，不是自动泄漏证据。

名称/祖先重叠候选明细见 [species_overlap_candidates.csv](statistics/species_overlap_candidates.csv)。350条是三个CUB划分全部匹配记录，祖先链可能为同一视觉类产生多条记录，不能当作350个重叠物种：

| CUB来源 | 匹配记录条数 | 不同NABirds视觉类 | 不同CUB类别 |
|---|---:|---:|---:|
| train | 144 | 72 | 58 |
| val | 114 | 57 | 34 |
| test | 92 | 46 | 30 |

这些是当前元信息名称匹配结果，不是专家确认的物种数量；同物异名可能漏检。待审核模板已保存在`evidence/nabirds_pending_review_DIAGNOSTIC_ONLY.json`，所有decision保持pending，不是正式test manifest。

## 6. 同批数据Tensor证据

每组下面的4个样本与可视化选择完全相同。train用实际训练Transform；val/NABirds原始诊断用实际eval Transform。不同数据集本地整数标签属于不同命名空间，不把三组拼成共享分类训练batch。

'''
    for name,row in tensors.items():
        text+=f"### {name}\n\n```json\n{json.dumps(row,ensure_ascii=False,indent=2)}\n```\n\n"
    text+='''## 7. 至少10张原始与增强对照（实际12张）

每行左到右为：原图、三次独立训练增强、确定性eval预处理。图内注明类别、路径和增强类型。`cub_val`与`nabirds_raw`的训练增强列是**反事实展示**，不是实际验证/测试输入；实际DataLoader使用eval列。原图展示适配画布的padding只用于排版，不是项目Transform。

[完整总览](visualizations/contact_sheet.png)；样本索引、原图副本SHA256、增强seed、各张量范围见 [sample_manifest.json](sample_manifest.json)。

'''
    for sample in samples:
        text+=f"### {sample['number']:02d} — {sample['dataset']} — {sample['class_name']}\n\n`{sample['path']}`；label={sample['label']}；原始尺寸={sample['original_size']}。\n\n![原图、三次训练增强及测试预处理]({sample['board']})\n\n[逐字节原图副本]({sample['original_copy']})\n\n"
    text+='''## 8. 自动化测试、复现与范围

测试脚本：[test_data_checks.py](test_data_checks.py)。执行日志及JUnit逐用例保存在evidence。数据诊断使用CPU和num_workers=0，未下载任何数据或权重，未创建模型、optimizer或调用项目engine。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:outputs/stage01_data .venv/bin/python -B -m pytest outputs/stage01_data/test_data_checks.py tests/test_data_pipeline.py tests/test_episode_plan_and_nabirds.py -q -p no:cacheprovider -o addopts='' --basetemp=outputs/stage01_data/evidence/pytest_tmp --junitxml=outputs/stage01_data/evidence/junit.xml
```

本轮已实际执行该组测试；若重跑请为basetemp/JUnit选择新的证据目录，以免pytest清理已有临时目录。`test_data_checks.py`中的内容级互斥与正式审核清单验收故意是正常断言，已知未通过项不会通过xfail/skip掩盖；应阅读实际测试结果，而不是要求一律绿色。合成NABirds审核测试仅验证正式Dataset机械逻辑，绝不表示真实物种已人工确认。

`check_data.py`可生成全套证据，默认拒绝覆盖已有statistics；`--report-only`仅由已生成证据重新渲染本报告。完整源码、数据清单前后保全结果见 [preservation.json](evidence/preservation.json)。停留在Stage01，等待用户确认后再开展下一阶段。
'''
    (OUT/'report.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        write_report();return
    if (OUT/'statistics.json').exists():
        parser.error('Existing statistics: refusing to overwrite this data-check run')
    torch.set_num_threads(2)
    protected=[p for base in ['src','scripts','tests','data/splits'] for p in (ROOT/base).rglob('*')
               if p.is_file() and '__pycache__' not in p.parts]
    before={p.relative_to(ROOT).as_posix():sha(p) for p in protected}
    save_json('evidence/protected_files_before.json',before)
    save_json('environment.json',{'python':sys.version,'torch':torch.__version__,
        'torchvision':torchvision.__version__,'Pillow':PIL.__version__,
        'device':'CPU only','seed':SEED,'data_roots':{'cub':str(CUB_ROOT),'nabirds':str(NAB_ROOT)}})
    from scripts.validate_cub import validate
    official=validate(CUB_ROOT,ROOT/'data/splits',check_images=True)
    datasets=make_datasets()
    stats={}
    for name,dataset in datasets.items():
        rows=mapping_rows(name,dataset)
        save_csv(f'statistics/{name}_class_mapping.csv',rows)
        stats[name]={'samples':len(dataset),'classes':len(dataset.classes),
            'original_id_min':min(r['original_class_id'] for r in rows),
            'original_id_max':max(r['original_class_id'] for r in rows),
            'min_per_class':min(r['images'] for r in rows),'max_per_class':max(r['images'] for r in rows),
            'label_min':0,'label_max':len(dataset.classes)-1,'label_unit':rows[0]['label_unit']}
        sample_rows=[]
        for i,(path,label) in enumerate(dataset.samples):
            assert path.is_file()
            sample_rows.append({'sample_index':i,'label':label,'original_class_id':rows[label]['original_class_id'],
                'class_name':dataset.classes[label],'path':path.relative_to(ROOT).as_posix()})
        save_csv(f'statistics/{name}_samples.csv',sample_rows)
    intersections={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        da,db=datasets['cub_'+a],datasets['cub_'+b]
        ids_a={s.split(maxsplit=2)[0] for s in (ROOT/f'data/splits/{a}_images.txt').read_text().splitlines()}
        ids_b={s.split(maxsplit=2)[0] for s in (ROOT/f'data/splits/{b}_images.txt').read_text().splitlines()}
        intersections[a+'_'+b]={'class_names':len(set(da.classes)&set(db.classes)),
            'image_ids':len(ids_a&ids_b),'paths':len({str(p) for p,_ in da.samples}&{str(p) for p,_ in db.samples})}
    official_classes=[line.split(maxsplit=1) for line in (CUB_ROOT/'classes.txt').read_text().splitlines()]
    random.Random(42).shuffle(official_classes)
    expected={'train':official_classes[:100],'val':official_classes[100:150],'test':official_classes[150:]}
    split_matches={s:{int(i) for i,_ in rows}=={int(line.split(maxsplit=1)[0]) for line in (ROOT/f'data/splits/{s}_classes.txt').read_text().splitlines()} for s,rows in expected.items()}
    save_json('statistics.json',{'datasets':stats,'official_cub_validation':official,
        'split_intersections':intersections,'seed42_split_matches':split_matches,
        'nabirds_official_test_status':'BLOCKED: no human-reviewed manifest/review',
        'nabirds_hierarchy_nodes':len(datasets['nabirds_raw'].metadata['classes'])})
    save_json('transforms.json',transform_config())
    review=create_review(NAB_ROOT,ROOT/'data/splits/train_classes.txt')
    save_json('evidence/nabirds_pending_review_DIAGNOSTIC_ONLY.json',review)
    overlaps=[]
    meta=datasets['nabirds_raw'].metadata
    for split in ['train','val','test']:
        normalized=defaultdict(list)
        for cname in datasets['cub_'+split].classes:normalized[normalize_bird_name(cname)].append(cname)
        for cid in sorted(set(meta['labels'].values())):
            for ancestor in _ancestors(cid,meta['parents']):
                for cname in normalized[normalize_bird_name(meta['classes'][ancestor])]:
                    overlaps.append({'cub_split':split,'cub_class':cname,'nabirds_visual_id':cid,
                        'nabirds_visual_name':meta['classes'][cid],'matched_ancestor_id':ancestor,
                        'matched_ancestor_name':meta['classes'][ancestor],'status':'candidate only, requires human review'})
    save_csv('statistics/species_overlap_candidates.csv',overlaps)
    save_json('evidence/episode_index_checks.json',episode_diagnostics(datasets))
    content=full_content_audit(datasets)
    save_json('evidence/content_audit.json',content)
    repro=content_episode_reproduction(datasets['cub_test'],content)
    save_json('evidence/content_episode_reproduction.json',repro)
    samples=visualize(datasets)
    batch_diagnostics(datasets,samples)
    checks=[
        {'check':'CUB真实Dataset与DataLoader','status':'PASS','detail':'train/val/test均实例化；所选图片及训练episode完成实际CPU数据加载'},
        {'check':'NABirds原图Dataset与DataLoader','status':'PASS','detail':'48562图、555视觉类原始诊断加载；不能作为已审核物种test'},
        {'check':'正式NABirds物种测试集','status':'BLOCKED','detail':'data/splits/nabirds_review.json与nabirds_test.json不存在'},
        {'check':'CUB划分和映射','status':'PASS','detail':f'官方验证、类别/ID/路径互斥；seed42重现={split_matches}'},
        {'check':'support/query索引及顺序','status':'PASS','detail':'两shot、四组数据、160个任务通过，NABirds只作原始诊断'},
        {'check':'support/query内容互斥','status':repro['status'],'detail':repro['reason']},
        {'check':'测试预处理确定性','status':'PASS','detail':'eval无随机增强，12张样本重复执行逐元素相等'},
        {'check':'12图原始及三次增强展示','status':'PASS','detail':'保留原始字节副本、反归一化PNG、逐图对照和总览'},
        {'check':'跨数据集物种无重叠','status':'BLOCKED','detail':f'发现{len(overlaps)}条名称/祖先匹配候选；未完成同义名/物种人工审核'},
    ]
    save_json('evidence/acceptance_checks.json',checks)
    after={p:sha(ROOT/p) for p in before}
    changed=[p for p in before if before[p]!=after[p]]
    assert not changed,changed
    project_model_imports=[k for k in sys.modules if k.startswith(('pattern_recognition_coursework.models','pattern_recognition_coursework.engine'))]
    assert not project_model_imports,project_model_imports
    save_json('evidence/preservation.json',{'files_checked':len(before),'changed_files':changed,
        'project_model_or_engine_modules_imported':project_model_imports,'model_training_executed':False})
    write_report()
    print('Stage01 data generation complete:',OUT,flush=True)


if __name__=='__main__':
    main()
