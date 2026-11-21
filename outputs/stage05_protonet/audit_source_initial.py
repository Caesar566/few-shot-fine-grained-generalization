"""Read-only, reproducible ProtoNet inspection using completed Stage 04 models.

No optimizer, training, test split images, or NABirds evaluation is performed.
Figures use Pillow so the audit needs no additional plotting dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import random
import sys
import textwrap
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset, get_cub_transforms
from pattern_recognition_coursework.data.episodes import load_or_create_plan, plan_hash
from pattern_recognition_coursework.engine.experiment import run_episodes, set_seed
from pattern_recognition_coursework.models.few_shot import FewShotModel
from pattern_recognition_coursework.models.metrics.episode import prototype_scores

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'imagenet': 'supervised_imagenet_seed42_server_bs128_fresh',
    'random': 'supervised_random_seed42_bs128',
    'simclr': 'supervised_simclr_seed42_bs128',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(value):
    t = value.detach().cpu().float()
    return dict(shape=list(value.shape), dtype=str(value.dtype), finite=bool(t.isfinite().all()),
                minimum=float(t.min()), maximum=float(t.max()), mean=float(t.mean()),
                std=float(t.std(unbiased=False)),
                mean_l2_norm=float(t.flatten(1).norm(dim=1).mean()) if t.ndim > 1 else None)


def tensor_image(value):
    # The source tensor is ImageNet-normalized: undo Normalize before displaying.
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    rgb = (value.detach().cpu()*std+mean).clamp(0, 1)
    return Image.fromarray((rgb.permute(1, 2, 0).numpy()*255).round().astype('uint8'))


def font(size=13):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def checked_math(features, n_way, k_shot, n_query, actual_logits):
    """Compare production scores with independent broadcast arithmetic and NLL."""
    support = features[:n_way*k_shot]
    query = features[n_way*k_shot:]
    prototypes = torch.stack([support[i*k_shot:(i+1)*k_shot].mean(0) for i in range(n_way)])
    # No torch.cdist and no call to prototype_scores in the reference calculation.
    raw_squared = ((query[:, None, :]-prototypes[None, :, :])**2).sum(-1)
    dimension = features.shape[-1]
    expected_logits = -raw_squared/dimension
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1e-6, atol=1e-6)
    probabilities = actual_logits.softmax(-1)
    stable = (expected_logits-expected_logits.max(-1, keepdim=True).values).exp()
    stable = stable/stable.sum(-1, keepdim=True)
    torch.testing.assert_close(probabilities, stable, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(n_way*n_query, device=features.device))
    labels = torch.arange(n_way, device=features.device).repeat_interleave(n_query)
    ce = F.cross_entropy(actual_logits, labels)
    reference_ce = (torch.logsumexp(expected_logits, -1)-expected_logits[torch.arange(len(labels)), labels]).mean()
    torch.testing.assert_close(ce, reference_ce, rtol=1e-6, atol=1e-6)
    assert raw_squared.shape == (n_way*n_query, n_way)
    assert raw_squared.min() >= 0 and probabilities.isfinite().all()
    assert torch.equal(actual_logits.argmax(-1), raw_squared.argmin(-1))
    unscaled_logits = -raw_squared
    return dict(support=support, query=query, prototypes=prototypes, raw_squared=raw_squared,
                scaled_squared=raw_squared/dimension, logits=actual_logits,
                probabilities=probabilities, labels=labels,
                predictions=probabilities.argmax(-1),
                unscaled_probabilities=unscaled_logits.softmax(-1),
                loss=float(ce), reference_loss=float(reference_ce),
                unscaled_loss=float(F.cross_entropy(unscaled_logits, labels)))


def toy_check():
    """Five classes in 2D, hand-specified means and squared distances."""
    centers = torch.tensor([[0., 0.], [10., 0.], [0., 10.], [10., 10.], [20., 0.]])
    query = centers+torch.tensor([0., 1.])
    expected = torch.tensor([[1., 101., 81., 181., 401.],
                             [101., 1., 181., 81., 101.],
                             [121., 221., 1., 101., 521.],
                             [221., 121., 101., 1., 221.],
                             [401., 101., 481., 181., 1.]])
    records = []
    for k in (1, 5):
        offsets = torch.zeros(1, 2) if k == 1 else torch.tensor([[-2., 0.], [-1., 0.], [0., 0.], [1., 0.], [2., 0.]])
        support = (centers[:, None, :]+offsets[None, :, :]).reshape(5*k, 2)
        actual = prototype_scores(support, query, 5, k, 'euclidean')
        result = checked_math(torch.cat([support, query]), 5, k, 1, actual)
        torch.testing.assert_close(result['prototypes'], centers, rtol=0, atol=0)
        torch.testing.assert_close(result['raw_squared'], expected, rtol=0, atol=0)
        assert torch.equal(result['predictions'], torch.arange(5))
        records.append(dict(k_shot=k, support=support.tolist(), expected_prototypes=centers.tolist(),
                            query=query.tolist(), expected_squared_distances=expected.tolist(),
                            actual_logits=actual.tolist(), probabilities=result['probabilities'].tolist(),
                            ce=result['loss'], status='PASS'))
    return records


def audit_members(dataset, indices, k, q):
    assert len(indices) == 5*(k+q) and len(set(indices)) == len(indices)
    class_order = [dataset.samples[indices[i*k]][1] for i in range(5)]
    assert len(set(class_order)) == 5
    mapping = {label: slot for slot, label in enumerate(class_order)}
    rows = []
    for position, index in enumerate(indices):
        path, original_label = dataset.samples[index]
        support = position < 5*k
        local_label = mapping[original_label]
        expected = position//k if support else (position-5*k)//q
        assert local_label == expected
        rows.append(dict(position=position, role='support' if support else 'query',
                         dataset_index=index, official_image_id=dataset.image_ids[index],
                         dataset_label=original_label, episode_label=local_label,
                         class_name=dataset.classes[original_label], path=str(path.relative_to(dataset.data_root.resolve())),
                         sha256=sha(path)))
    # Checks all roles/classes as well as just the support/query intersection.
    assert len({r['path'] for r in rows}) == len(rows)
    assert len({r['official_image_id'] for r in rows}) == len(rows)
    assert len({r['sha256'] for r in rows}) == len(rows)
    for slot in range(5):
        assert sum(r['role']=='support' and r['episode_label']==slot for r in rows) == k
        assert sum(r['role']=='query' and r['episode_label']==slot for r in rows) == q
    return rows, [dataset.classes[label] for label in class_order]


def card(tensor, row, prediction=None, probability=None):
    canvas = Image.new('RGB', (270, 318), 'white')
    canvas.paste(tensor_image(tensor).resize((224, 224)), (23, 0))
    d = ImageDraw.Draw(canvas)
    title = f"{row['role'].upper()} | true={row['episode_label']}"
    color = 'black'
    if prediction is not None:
        title += f' pred={prediction} p={probability:.3f}'
        color = '#137333' if prediction == row['episode_label'] else '#b3261e'
    d.text((4, 226), title, fill=color, font=font(12))
    lines = textwrap.wrap(row['class_name'], 36)+textwrap.wrap(row['path'].removeprefix('images/'), 38)
    for i, line in enumerate(lines[:5]):d.text((4, 244+i*14), line, fill='black', font=font(11))
    return canvas


def episode_visuals(out, images, rows, classes, math_result, k, q):
    pred = math_result['predictions'].cpu().tolist()
    probabilities = math_result['probabilities'].cpu()
    canvas = Image.new('RGB', (5*270, (k+q)*318+115), '#e8edf2')
    d = ImageDraw.Draw(canvas)
    d.text((12, 8), f'5-way {k}-shot {q}-query | columns=episode classes; rows=support then query', fill='black', font=font(18))
    for slot, name in enumerate(classes):
        for j, line in enumerate(textwrap.wrap(f'{slot}: {name}', 32)):
            d.text((slot*270+5, 44+j*16), line, fill='black', font=font(12))
    errors = []
    for i, row in enumerate(rows):
        query_index = i-5*k
        prediction = pred[query_index] if query_index >= 0 else None
        confidence = float(probabilities[query_index].max()) if query_index >= 0 else None
        tile = card(images[i], row, prediction, confidence)
        line = i % k if i < 5*k else k+(i-5*k) % q
        canvas.paste(tile, (row['episode_label']*270, 115+line*318))
        if prediction is not None and prediction != row['episode_label']:
            error = dict(row, query_index=query_index, predicted_label=prediction,
                         predicted_class=classes[prediction], confidence=confidence)
            errors.append(error)
            # Complete true/predicted names accompany every error image in errors.json.
            tile.save(out/'errors'/f'query_{query_index:02d}.png')
    canvas.save(out/'episode.png')
    if errors:
        gallery = Image.new('RGB', (5*270, math.ceil(len(errors)/5)*360+40), 'white')
        draw = ImageDraw.Draw(gallery)
        draw.text((8, 8), 'Classification errors (predicted class names in errors.json)', fill='black', font=font(16))
        for i, row in enumerate(errors):
            tile = Image.open(out/'errors'/f"query_{row['query_index']:02d}.png")
            x, y = (i % 5)*270, 40+(i//5)*360
            gallery.paste(tile, (x, y))
            for j, line in enumerate(textwrap.wrap('Pred: '+row['predicted_class'], 34)):
                draw.text((x+4, y+320+j*13), line, fill='#b3261e', font=font(11))
        gallery.save(out/'errors.png')
    else:
        (out/'errors/README.md').write_text('This episode has no misclassified query samples.\n')
    write_json(out/'errors.json', errors)


def heatmap(path, tensor, classes, title):
    matrix = tensor.detach().cpu().numpy()
    h, w = matrix.shape
    canvas = Image.new('RGB', (450+w*135, 160+h*28), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill='black', font=font(17))
    for j, name in enumerate(classes):
        draw.text((10, 38+j*17), f'Class {j}: {name}', fill='black', font=font(12))
    lo, hi = float(matrix.min()), float(matrix.max())
    for j in range(w):draw.text((455+j*135, 135), f'prototype {j}', fill='black', font=font(12))
    for i in range(h):
        draw.text((8, 162+i*28), f'Q{i:02d} true={i//5} {classes[i//5]}', fill='black', font=font(11))
        for j in range(w):
            value = float(matrix[i, j]); fraction = (value-lo)/max(hi-lo, 1e-12)
            color = (int(245-190*fraction), int(248-145*fraction), int(255-70*fraction))
            box = (450+j*135, 158+i*28, 584+j*135, 185+i*28)
            draw.rectangle(box, fill=color, outline='white')
            draw.text((455+j*135, 162+i*28), f'{value:.5f}', fill='white' if fraction>.65 else 'black', font=font(12))
    canvas.save(path)


def save_matrix(path, tensor, classes):
    array = tensor.detach().cpu().tolist()
    write_csv(path, [dict(query_index=i, **{f'class_{j}': value for j, value in enumerate(row)}) for i, row in enumerate(array)])


def check_episode(model, dataset, plan, out, seed, repeats, logger):
    k, q = plan['k_shot'], plan['n_query']
    out.mkdir();(out/'errors').mkdir()
    indices = plan['batches'][0]
    rows, classes = audit_members(dataset, indices, k, q)
    images, original_labels = next(iter(DataLoader(dataset, batch_sampler=plan['batches'], num_workers=0)))
    assert original_labels.tolist() == [r['dataset_label'] for r in rows]
    labels = torch.tensor([r['episode_label'] for r in rows[5*k:]])
    torch.testing.assert_close(labels, torch.arange(5).repeat_interleave(q), rtol=0, atol=0)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        features = model.encode(images)
        actual_logits = model.classify_embeddings(features, 5, k, q)
        result = checked_math(features, 5, k, q, actual_logits)
        # Compare independent support/query calls with the concatenated call.
        separately = torch.cat([model.encode(images[:5*k]), model.encode(images[5*k:])])
        torch.testing.assert_close(features, separately, rtol=1e-4, atol=1e-5)
        repeat_evidence = []
        for repetition in range(repeats):
            set_seed(seed)
            from pattern_recognition_coursework.data.cub_dataset import FewShotBatchSampler
            sampled = next(iter(FewShotBatchSampler(dataset, 1, 5, k, q, seed=seed)))
            assert sampled == indices
            fresh_images, _ = next(iter(DataLoader(dataset, batch_sampler=[sampled], num_workers=0)))
            assert torch.equal(images, fresh_images)
            fresh_logits = model(fresh_images, 5, k, q)
            assert torch.equal(actual_logits, fresh_logits)
            assert torch.equal(result['predictions'], fresh_logits.argmax(-1))
            repeat_evidence.append(dict(repetition=repetition+1, indices_equal=True, inputs_bitwise_equal=True,
                                        logits_bitwise_equal=True, predictions_equal=True,
                                        max_logit_abs_diff=float((actual_logits-fresh_logits).abs().max())))
        # Execute the actual evaluation engine too: it generates local query labels.
        engine_metrics, engine_predictions = run_episodes(model, DataLoader(dataset, batch_sampler=plan['batches']),
                                                        5, k, q, torch.device('cpu'), collect_predictions=True)
    assert engine_predictions[0]['predicted_slots'] == result['predictions'].tolist()
    assert abs(engine_metrics['loss']-result['loss']) < 1e-6
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert features.shape == (5*(k+q), 512) and result['prototypes'].shape == (5, 512)
    assert features.isfinite().all() and features.std() > 0
    write_csv(out/'members.csv', rows)
    predictions = []
    for i, row in enumerate(rows[5*k:]):
        slot = int(result['predictions'][i])
        predictions.append(dict(row, query_index=i, predicted_label=slot, predicted_class=classes[slot],
                                correct=slot==row['episode_label'], confidence=float(result['probabilities'][i, slot])))
    write_csv(out/'predictions.csv', predictions)
    for key in ['raw_squared', 'scaled_squared', 'logits', 'probabilities', 'unscaled_probabilities']:
        save_matrix(out/f'{key}.csv', result[key], classes)
    for key, tensor in [('features', features), ('prototypes', result['prototypes'])]:
        write_csv(out/f'{key}.csv', [dict(index=i, **{f'd{j:03d}': float(v) for j, v in enumerate(row)}) for i, row in enumerate(tensor)])
    prototype_statistics = [dict(episode_label=i, class_name=name, l2_norm=float(result['prototypes'][i].norm()),
                                 **stats(result['prototypes'][i])) for i, name in enumerate(classes)]
    write_json(out/'prototype_statistics.json', prototype_statistics)
    evidence = dict(k_shot=k, n_way=5, n_query=q, classes=classes, plan_sha256=plan_hash(plan),
                    images=stats(images), support=stats(result['support']), query=stats(result['query']),
                    prototypes=stats(result['prototypes']), logits=stats(actual_logits),
                    squared_distance=stats(result['raw_squared']), probabilities=stats(result['probabilities']),
                    original_dataset_labels=original_labels.tolist(), episode_query_labels=labels.tolist(),
                    accuracy=float((result['predictions']==labels).float().mean()), loss=result['loss'],
                    independent_reference_loss=result['reference_loss'], unscaled_negative_distance_loss=result['unscaled_loss'],
                    repeats=repeat_evidence, engine_loss=engine_metrics['loss'],
                    separate_encoding_max_abs_diff=float((features-separately).abs().max()),
                    checks=dict(support_query_ids_paths_and_bytes_disjoint=True, counts_and_local_labels=True,
                                means_distances_logits_softmax_ce=True, one_shared_frozen_encoder=True,
                                backbone_parameters_and_bn_buffers_unchanged=True, engine_agrees=True,
                                repeated_sampling_transform_and_predictions_identical=True))
    write_json(out/'evidence.json', evidence)
    episode_visuals(out, images, rows, classes, result, k, q)
    heatmap(out/'distance_heatmap.png', result['raw_squared'], classes, 'Raw SQUARED Euclidean distance (smaller = nearer)')
    heatmap(out/'probability_heatmap.png', result['probabilities'], classes, 'Production probabilities: softmax(-squared_distance / 512)')
    logger.info('%s | images=%s support=%s query=%s prototypes=%s distance=%s logits=%s | loss=%.8f accuracy=%.4f errors=%d | all checks PASS',
                out.name, list(images.shape), list(result['support'].shape), list(result['query'].shape),
                list(result['prototypes'].shape), list(result['raw_squared'].shape), list(actual_logits.shape),
                result['loss'], evidence['accuracy'], sum(not p['correct'] for p in predictions))
    logger.info('Tensor statistics and labels: %s', json.dumps(evidence, ensure_ascii=False))
    return evidence


def report(out, results, checkpoints):
    lines = ['# Stage 05：ProtoNet 数据流与数学核验', '',
             '本阶段只做前向验证。使用阶段4三组已完成50轮的最优Checkpoint，在CUB clean Val上执行固定seed=42的5-way 1-shot和5-shot任务，每类5个Query。不同初始化使用同一份任务计划；不读取CUB Test或NABirds图片，不训练或选择新Checkpoint。', '',
             '**所有核心检查通过。每个设置仅1个Episode，下面准确率用于展示流程，不是正式泛化成绩；这些Checkpoint原本也由CUB Val选择，因此不能将本检查称为独立最终测试。**', '',
             '## 代码调用关系', '',
             '- `data/cub_dataset.py:CUBFewShotDataset/get_cub_transforms`：清洗索引、分区本地标签、确定性预处理。',
             '- `data/cub_dataset.py:FewShotBatchSampler.__iter__`：先采5类，每类无放回抽K+Q张，最后返回按类分组的全部Support，再返回按同一类序排列的Query。',
             '- `data/episodes.py:load_or_create_plan/validate_plan`：持久化索引、数据指纹、数量与分组顺序验证。',
             '- `models/few_shot.py:FewShotModel.forward → encode → classify_embeddings`：共用一个ResNet-18编码器，按位置切分Support/Query。',
             '- `models/metrics/episode.py:prototype_scores`：Support按`[5,K,512]`重排，沿K维求均值，然后计算距离分数。',
             '- `engine/experiment.py:run_episodes`：eval模式，无优化器，不求梯度；生成局部Query标签`arange(5).repeat_interleave(5)`，计算CE与预测。',
             '- `scripts/check_stage05_protonet.py`：独立公式、内容哈希、重复前向、完整引擎交叉核验和可视化。', '',
             '## 数据、标签与Tensor Shape', '',
             '数据来自2947张/50类的CUB清洗验证集，与5890张/100类的训练集在类别、图像ID、路径、内容哈希上互斥。每张选中图片的SHA256会重新计算并核对清洗索引。`members.csv`记录原始官方图像ID、Val本地标签、Episode局部0～4标签、路径和内容哈希。局部类别顺序来自采样顺序，不是鸟类名称排序。', '',
             '实际预处理：RGB → Resize短边257（默认bilinear）→ CenterCrop224 → ToTensor(float32,0～1) → ImageNet Normalize(mean=.485/.456/.406,std=.229/.224/.225)。测试没有随机裁剪、翻转或ColorJitter。可视化先反归一化到RGB，不把归一化Tensor直接当图片。', '',
             '| 阶段 | 5-way 1-shot | 5-way 5-shot |', '|---|---|---|',
             '| Support图片 | `[5,3,224,224]` | `[25,3,224,224]` |',
             '| Query图片（每类5张） | `[25,3,224,224]` | `[25,3,224,224]` |',
             '| 拼接输入 | `[30,3,224,224]` | `[50,3,224,224]` |',
             '| Conv1 | `[30,64,112,112]` | `[50,64,112,112]` |',
             '| Layer1（经maxpool） | `[30,64,56,56]` | `[50,64,56,56]` |',
             '| Layer2 | `[30,128,28,28]` | `[50,128,28,28]` |',
             '| Layer3 | `[30,256,14,14]` | `[50,256,14,14]` |',
             '| Layer4 | `[30,512,7,7]` | `[50,512,7,7]` |',
             '| GAP + Flatten | `[30,512]` | `[50,512]` |',
             '| Support重排 | `[5,1,512]` | `[5,5,512]` |',
             '| 类别均值原型 | `[5,512]` | `[5,512]` |',
             '| Query-原型广播差 | `[25,5,512]` | `[25,5,512]` |',
             '| 距离 / logits / 概率 | 均为`[25,5]` | 均为`[25,5]` |',
             '| Query标签 / 预测 | 均为`[25]`，int64 | 均为`[25]`，int64 |', '',
             '上述主干中间形状由运行时hook实际记录在各初始化的`weight_loading.json`。所有特征和分数为float32，输入范围、特征范围/均值/标准差/L2范数及标签完整保存在`evidence.json`和运行日志。', '',
             '```mermaid', 'flowchart TD', ' A[清洗CUB Val索引] --> B[seed42采5类 每类K+5张]',
             ' B --> C[前5K张Support 后25张Query]', ' C --> D[确定性预处理 224 RGB]',
             ' D --> E[同一个冻结ResNet18 eval]', ' E --> F[全局512维特征]',
             ' F --> G[Support重排 5×K×512 沿K求均值]', ' G --> H[5×512原型]',
             ' F --> I[25×512 Query]', ' H --> J[广播平方差并对512维求和]',
             ' I --> J', ' J --> K[25×5平方距离]', ' K --> L[Logits等于负距离除512]',
             ' L --> M[Softmax概率 argmax预测]', ' L --> N[与Episode局部标签计算CE]', '```', '',
             '## 数学计算及关键缩放', '',
             r'原型：`p_c = (1/K) Σ_k f(x_support[c,k])`。1-shot时原型就是该类唯一Support的特征；5-shot时是该类5张Support特征的均值。原型不使用Query，也不使用Query标签。', '',
             r'平方欧氏距离：`D[i,c] = Σ_d (q[i,d] - p[c,d])²`，形状`[25,5]`。', '',
             '**现有实现实际返回`logits = -D / 512 / T`，本阶段T=1；没有特征L2归一化、Episode标准化、Adapter或局部融合。它与直接`-D`的argmax相同，但Softmax概率与CE不同。没有悄悄删除除512，也没有把它称作未经缩放的原始距离。**', '',
             r'概率：`P[i,c] = exp(logits[i,c]) / Σ_j exp(logits[i,j])`。每行概率和为1；预测为`argmax(P)`，等价于`argmin(D)`。损失：`CE = mean(logsumexp(logits[i,:]) - logits[i,y_i])`。PyTorch CE接收logits，不先传Softmax。', '',
             '`raw_squared.csv`保存D；`scaled_squared.csv`保存D/512；`logits.csv`保存-D/512；`probabilities.csv`保存实际概率。`unscaled_probabilities.csv`及evidence中的unscaled loss仅用于说明如果直接使用-D会产生怎样的不同，不作为项目实际预测概率。', '',
             '## Checkpoint与结果', '',
             '| 初始化 | Stage04最佳轮次 | 1-shot正确数/25 | 5-shot正确数/25 |', '|---|---:|---:|---:|']
    for name in RUNS:
        one, five = results[name]
        lines.append(f"| {name} | {checkpoints[name]['epoch']} | {round(one['accuracy']*25)} | {round(five['accuracy']*25)} |")
    lines += ['', '完整监督Checkpoint只加载`model`中的`backbone.`前缀，明确排除`classifier.weight/bias`两个张量。与Checkpoint内120个编码器state逐项精确相等，参数和BN缓冲区在核验前后均未改变；权重文件的SHA256也未改变。未使用监督100类分类头，也没有把未训练的新编码器冒充阶段4模型。', '',
              '## 可视化入口', '']
    for name in RUNS:
        for k in (1, 5):
            base=f'{name}/5way_{k}shot'
            lines += [f'- **{name} / {k}-shot**：[完整Episode]({base}/episode.png) · [平方距离热力图]({base}/distance_heatmap.png) · [预测概率热力图]({base}/probability_heatmap.png) · [所有Query预测]({base}/predictions.csv) · [原型统计]({base}/prototype_statistics.json) · [错误清单]({base}/errors.json)']
            errors=json.loads((out/base/'errors.json').read_text())
            lines += [f'  错分{len(errors)}张。'+(f'[错误图集]({base}/errors.png)' if errors else '本Episode没有错分，errors目录内有说明。')]
    lines += ['', '完整Episode图列为Episode类别，先显示K行Support，再显示5行Query。每张图显示角色、真实局部类别、Query预测与置信度，以及真实类别名称和样本相对路径；顶部给出0～4类别名称对照。错误图集补充预测类别名称，CSV/JSON提供未截断路径。', '',
              '## 自动核验与可复现性', '',
              '- 人工5类二维向量同时验证1-shot/5-shot，原型与手写25项平方距离逐项相等；实际分数需除以特征维数2，而非真实图像特征的512。见`toy_vectors.json`。',
              '- 真实数据每类Support/Query数量正确；索引、官方ID、路径、SHA256均唯一；所选数据内容与清洗CSV一致。',
              '- 独立均值/广播平方距离/稳定Softmax/logsumexp损失与现有分类分支、完整`run_episodes`引擎一致。',
              '- 每个设置重复3次：重新seed、采样、读取与预处理、完整前向；索引、输入、logits和预测均逐位相同。另一次独立进程重跑结果见`rerun_verification.json`。',
              '- 模型eval、所有参数requires_grad=False、inference_mode、无优化器；参数及BN state逐位不变。',
              '- 这是当前CPU环境内的确定性验证；不承诺跨GPU/跨硬件逐位一致。', '',
              '运行配置、环境、命令、源码哈希、权重加载清单和任务计划保存在本目录。`checks.json`为总验收，`run.log`为实际运行日志，`tests/`保存小规模回归证据。']
    (out/'report.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/stage05_protonet')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT/'outputs/stage05_protonet'):
        parser.error('Audit outputs must stay inside outputs/stage05_protonet')
    if out.exists() and any(out.iterdir()):parser.error('Output directory is not empty; refusing to overwrite')
    if args.repeats < 1:parser.error('repeats must be positive')
    out.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('stage05');logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(out/'run.log'));logger.addHandler(logging.StreamHandler())
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);set_seed(args.seed)
    source_paths = sorted((ROOT/'src').rglob('*.py'))
    source_before = {str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    split_dir = ROOT/'data/splits/cub_clean_v1';raw = ROOT/'cub2002011/versions/7/CUB_200_2011'
    dataset = CUBFewShotDataset(raw, split_dir/'val_images.txt', get_cub_transforms(224, False), require_clean_index=True)
    # Keep stable official IDs independently of any Dataset implementation details.
    dataset.image_ids = [line.split()[0] for line in dataset.split_file.read_text().splitlines() if line.strip()]
    train = [line.split(maxsplit=2) for line in (split_dir/'train_images.txt').read_text().splitlines() if line.strip()]
    val = [line.split(maxsplit=2) for line in dataset.split_file.read_text().splitlines() if line.strip()]
    for col in (0,1,2):assert not {r[col] for r in train}&{r[col] for r in val}
    with (split_dir/'image_hashes.csv').open(encoding='utf-8-sig') as stream:hash_rows=list(csv.DictReader(stream))
    train_hashes={r['sha256'] for r in hash_rows if r['split']=='train'}
    val_hashes={r['sha256'] for r in hash_rows if r['split']=='val'}
    assert not train_hashes&val_hashes
    expected_hash={r['relative_path']:r['sha256'] for r in hash_rows if r['split']=='val'}
    plans={k:load_or_create_plan(out/f'plan_5way_{k}shot.json',dataset,5,k,5,1,args.seed) for k in (1,5)}
    for plan in plans.values():
        for index in plan['batches'][0]:
            path=dataset.samples[index][0]
            assert sha(path)==expected_hash[str(path.relative_to(raw.resolve()/'images'))]
    config=dict(seed=args.seed, n_way=5,k_shots=[1,5],n_query_per_class=5,repeats=args.repeats,
                split='CUB clean Val',image_size=224,device='cpu',dtype='float32',metric='euclidean',
                logits='-sum_squared_distance / 512',temperature=1,feature_standardization=False,
                feature_l2_normalization=False,adapter=False,training=False,optimizer=None)
    write_json(out/'config.json',config);write_json(out/'toy_vectors.json',toy_check())
    write_json(out/'environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
                                          platform=platform.platform(),threads=torch.get_num_threads()))
    write_json(out/'command.json',dict(argv=sys.argv if argv is None else argv))
    results={};checkpoints={}
    started=time.perf_counter()
    for name, run_name in RUNS.items():
        run=ROOT/'outputs/stage04_training'/run_name
        completion=json.loads((run/'result.json').read_text())
        assert completion['status']=='completed' and completion['epochs_completed']==50
        path=run/'checkpoints/best.pth';before_hash=sha(path)
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        assert checkpoint['epoch']==completion['best_epoch']
        assert checkpoint['split_hashes']['val_images.txt']==sha(dataset.split_file)
        assert checkpoint['validation']['accuracy']==completion['best_validation_accuracy']
        model=FewShotModel(method='protonet',metric='euclidean',temperature=1,
                           backbone_weights=path,backbone_prefix='backbone.').eval()
        model.requires_grad_(False)
        state=model.backbone.state_dict()
        for key,value in state.items():assert torch.equal(value,checkpoint['model']['backbone.'+key])
        name_out=out/name;name_out.mkdir()
        shape_record={};hooks=[]
        for label,module in [('conv1',model.backbone.conv1),('layer1',model.backbone.layer1),
                             ('layer2',model.backbone.layer2),('layer3',model.backbone.layer3),
                             ('layer4',model.backbone.layer4),('gap',model.backbone.avgpool)]:
            def hook(m, inputs, output, label=label):
                shape_record.setdefault(label, {})[str(inputs[0].shape[0])]=list(output.shape)
            hooks.append(module.register_forward_hook(hook))
        logger.info('Loading %s checkpoint: %s | epoch=%d sha256=%s',name,path,checkpoint['epoch'],before_hash)
        results[name]=[check_episode(model,dataset,plans[k],name_out/f'5way_{k}shot',args.seed,args.repeats,logger) for k in (1,5)]
        for hook in hooks:hook.remove()
        for batch in (30,50):
            for label,tail in [('conv1',[64,112,112]),('layer1',[64,56,56]),
                               ('layer2',[128,28,28]),('layer3',[256,14,14]),
                               ('layer4',[512,7,7]),('gap',[512,1,1])]:
                assert shape_record[label][str(batch)]==[batch,*tail]
        assert before_hash==sha(path)
        checkpoints[name]=dict(path=str(path.relative_to(ROOT)),sha256=before_hash,epoch=checkpoint['epoch'],
                               source_validation_accuracy=checkpoint['validation']['accuracy'],
                               initialization=checkpoint['config']['weights'],
                               loaded_encoder_tensors=len(state),state_matches_checkpoint=True,
                               weight_file_unchanged=True,load_info=model.backbone_load_info,
                               runtime_shapes=shape_record)
        write_json(name_out/'weight_loading.json',checkpoints[name])
        del model,checkpoint
    assert source_before=={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    write_json(out/'source_sha256.json',dict(package=source_before,audit_script_sha256=sha(__file__)))
    write_json(out/'checkpoints.json',checkpoints)
    write_json(out/'summary.json',results)
    write_json(out/'checks.json',dict(status='PASS',real_episodes=6,shots=[1,5],queries_per_class=5,
                                     toy_cases=2,repeats_per_episode=args.repeats,
                                     package_source_unchanged=True,checkpoint_files_unchanged=True,
                                     train_val_classes_ids_paths_content_disjoint=True,
                                     actual_episode_bytes_match_clean_manifest=True,
                                     training_performed=False,seconds=time.perf_counter()-started))
    report(out,results,checkpoints)
    logger.info('PASS: all 6 real episodes and both hand-calculated vector cases. Report: %s',out/'report.md')


if __name__=='__main__':
    main()
