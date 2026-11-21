"""Finalize Stage05 evidence after real forward runs, replay, and pytest complete."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import math
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(name):
    return json.loads((OUT/name).read_text())


def rows(name):
    with (OUT/name).open() as stream:
        return list(csv.DictReader(stream))


replay = read('rerun_verification.json')
assert replay['status']=='PASS' and len(replay['episodes'])==6
suite = ET.parse(OUT/'tests/junit.xml').getroot()
cases = suite.findall('.//testcase')
assert len(cases)==11
assert not suite.findall('.//failure') and not suite.findall('.//error')
source = read('source_sha256.json')
assert hashlib.sha256((OUT/'audit_source_initial.py').read_bytes()).hexdigest()==source['audit_script_sha256']
assert hashlib.sha256((ROOT/'scripts/check_stage05_protonet.py').read_bytes()).hexdigest()==source['replay_script_sha256']
for path, expected in source['package'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected
for k in (1,5):
    members = (OUT/f'imagenet/5way_{k}shot/members.csv').read_bytes()
    for name in ('random','simclr'):
        assert (OUT/f'{name}/5way_{k}shot/members.csv').read_bytes()==members

checks = read('checks.json')
checks.update(independent_process_replay='PASS', independent_process_episodes=6,
              repeated_full_forward_passes=18, automated_tests_passed=len(cases),
              same_episode_members_across_initializations=True,
              original_audit_source_snapshot_hash_verified=True,
              replay_source_hash_verified=True)
(OUT/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
environment = read('environment.json')
environment['package_versions'] = {name:importlib.metadata.version(name) for name in
                                   ['torch','torchvision','numpy','Pillow','pytest']}
(OUT/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')

dist = rows('imagenet/5way_1shot/raw_squared.csv')[0]
logits = rows('imagenet/5way_1shot/logits.csv')[0]
probabilities = rows('imagenet/5way_1shot/probabilities.csv')[0]
prediction = rows('imagenet/5way_1shot/predictions.csv')[0]
extra = ['','## 一个真实Query的具体数值','',
         '以ImageNet初始化Checkpoint的1-shot任务第0张Query为例：', '',
         f"- 真实类别：`{prediction['class_name']}`，Episode标签`{prediction['episode_label']}`。",
         f"- 图片路径：`{prediction['path']}`。", '',
         '| 原型局部类别 | 原始平方距离D | 实际logit=-D/512 | Softmax概率 |',
         '|---|---:|---:|---:|']
for c in range(5):
    key = f'class_{c}'
    extra.append(f'| {c} | {float(dist[key]):.6f} | {float(logits[key]):.6f} | {float(probabilities[key]):.6f} |')
extra += ['',f"最近原型/最大概率对应类别`{prediction['predicted_label']}`（`{prediction['predicted_class']}`）。"
          f"真实类别的本张Query负对数概率为`{-math.log(float(probabilities['class_'+prediction['episode_label']])):.6f}`；"
          'Episode CE是25张Query对应损失的平均值。这个样本预测错误，说明流程检查通过不意味着所有图片都会分类正确。', '',
          '## 最终验证证据与复跑命令', '',
          '- 11项自动化测试全部通过：[pytest日志](tests/pytest.log)、[JUnit](tests/junit.xml)。测试包括人工向量、错误距离缩放拒绝、不同标签/不同路径但内容相同的样本拒绝，以及现有few-shot/指标回归。',
          '- 新进程重新加载三组Checkpoint、重新生成seed42任务，对六个Episode的特征、logits、概率和预测逐位核对，最大差值全部为0：[复跑证据](rerun_verification.json)、[复跑日志](replay.log)。',
          '- 三种初始化在同一shot设置下的members.csv逐字节一致。参数、BN缓冲区、权重文件、原有src模块均未改变。',
          '- 初次运行脚本的精确源码快照：[audit_source_initial.py](audit_source_initial.py)。源码与后续带`--replay`入口的脚本分别记录SHA256，避免将两个版本混为一谈。', '',
          '```bash',
          '# 首次审计（已有默认目录时拒绝覆盖；复跑请指定新的空目录）',
          'uv run python scripts/check_stage05_protonet.py \\',
          '  --output-dir outputs/stage05_protonet/recheck_seed42', '',
          '# 新进程对照该次保存的完整特征、分数、概率和预测',
          'uv run python scripts/check_stage05_protonet.py --replay \\',
          '  --output-dir outputs/stage05_protonet/recheck_seed42',
          '```', '',
          '以上执行不创建优化器、不更新权重、不下载模型；使用本地阶段4完整best.pth。目录须保持同一工程布局，数据和Checkpoint需可读。']
report = OUT/'report.md'
text = report.read_text()
marker = '\n## 一个真实Query的具体数值\n'
if marker in text:text = text.split(marker)[0]
report.write_text(text+'\n'.join(extra)+'\n')
print(json.dumps(checks, indent=2))
print('Report:', report)
