"""Run CUDA regression/replay, export figures, and finish the full-run report."""
from pathlib import Path
import csv
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from pattern_recognition_coursework.engine.distance_comparison import folder_hashes
from pattern_recognition_coursework.engine.episodic_evaluation import sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/stage05_protonet/representation_full_seed42_600_cuda'


def read(path):return json.loads(Path(path).read_text())


def main():
    start=time.perf_counter();config=read(OUT/'config.json')
    protected={p:folder_hashes(Path(p)) for p in [config['source_dir'],config['global_baseline_dir']]}
    tests=[sys.executable,'-m','pytest','tests/test_representation_comparison.py',
           'tests/test_advanced_models.py','tests/test_few_shot_baselines.py','-q',
           f'--junitxml={OUT}/cuda_tests.xml']
    with (OUT/'cuda_tests.log').open('w') as stream:subprocess.run(tests,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    suite=ET.parse(OUT/'cuda_tests.xml').getroot()
    assert len(suite.findall('.//testcase'))==19
    assert not suite.findall('.//failure') and not suite.findall('.//error') and not suite.findall('.//skipped')
    replay=[sys.executable,str(ROOT/'outputs/stage05_protonet/verify_representation_full.py'),
            '--run-dir',str(OUT),'--device','cuda']
    with (OUT/'independent_verification.log').open('w') as stream:subprocess.run(replay,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    for path,original in protected.items():assert folder_hashes(Path(path))==original
    checkpoint_info=read(Path(config['source_dir'])/'checkpoints.json')
    checks=read(OUT/'checks.json')
    for name,record in checkpoint_info.items():assert sha(record['path'])==checks['checkpoint_sha256'][name]
    perf=read(OUT/'performance.json')
    lines=['','## 完整运行性能与验收','',
        f'实际设备：**{checks["device_name"]} / CUDA**。完整评估（含核对、缓存、推理、保存和图表）耗时 **{checks["seconds"]:.2f} 秒**。'
        '下面显存为PyTorch峰值allocated/reserved，不包含驱动上下文或其他进程。','',
        '| 初始化 | 缓存复用图片 | 新提取图片 | 提取/缓存秒 | GPU前向探针秒 | 1-shot秒 | 5-shot秒 | 峰值allocated MiB | 峰值reserved MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,p in perf.items():
        cache=p['feature_extraction']
        lines.append(f'| {name} | {cache["reused_images"]} | {cache["extracted_images"]} | {cache["extraction_seconds"]:.2f} | {p["forward_probe_seconds"]:.2f} | {p["shot_seconds"]["1"]:.2f} | {p["shot_seconds"]["5"]:.2f} | {p["cuda_peak_allocated_mb"]:.2f} | {p["cuda_peak_reserved_mb"]:.2f} |')
    lines += ['', '固定5-way、每类5 Query，1-shot/5-shot各600个Episode，三种Backbone共10800次方法评分；Global 3600项复用旧Cosine统计，Local/Fusion新增7200项。完整结果共270000条Query方法预测。', '',
        '- **19项CUDA回归测试全部通过**，包含分块等价、固定融合、不同权重/ID缓存隔离、smoke→full部分缓存复用及异常记录保护。[日志](cuda_tests.log)。',
        '- 独立进程在CUDA上复核全部3600个Episode记录中的三路分数、校准、概率与270000个预测；对18个首/中/末Episode重新计算GPU Global/Local/Fusion原始公式，逐位复现。统计与18组配对CI独立重算一致。[证据](independent_verification.json)。',
        '- 旧Episode和距离实验目录逐文件哈希未改变，三份训练权重SHA256保持不变；没有训练、OOM、CPU回退或异常。',
        '- 提取Batch Size 8，Query chunk 4，FP32、关闭AMP/TF32。完整缓存位于CPU，三组缓存通过不同Checkpoint SHA256隔离。ImageNet复用了150张smoke特征，未重复提取。', '',
        '## 配对结果解读','']
    with (OUT/'paired_comparison.csv').open() as stream:paired=list(csv.DictReader(stream))
    for r in paired:
        if r['first']=='global' and r['second']=='fusion':
            lines.append(f'- {r["initialization"]} {r["shot"]}-shot：Fusion−Global = **{float(r["mean_delta_pp"]):+.3f} pp**，配对95% CI [{float(r["ci95_low_pp"]):+.3f}, {float(r["ci95_high_pp"]):+.3f}] pp。')
    lines += ['', '固定0.5融合没有稳定超过Global。本实验如实保留该结果，没有根据这些Val标签重新调整权重、Top-k、归一化或校准参数。Local较弱或Fusion未提升的原因尚未通过额外实验定位，不据此推断具体鸟类器官或背景因果。', '',
              '结果为曾用于选择Checkpoint的CUB Val对比，不是独立最终测试，也不能外推为NABirds成绩。95% CI仅反映固定模型/数据下的Episode采样波动，配对分析未做多重比较校正。']
    marker='\n## 完整运行性能与验收\n';text=(OUT/'report.md').read_text()
    if marker in text:text=text.split(marker)[0]
    (OUT/'report.md').write_text(text+'\n'.join(lines)+'\n')
    for filename in ['accuracy_comparison','accuracy_bars','paired_lifts']:
        subprocess.run(['rsvg-convert','-o',str(OUT/f'{filename}.png'),str(OUT/f'{filename}.svg')],check=True)
    checks.update(cuda_regression_tests_passed=19,old_input_directories_hash_reverified=True,
                  all_checkpoints_sha256_reverified=True,oom_count=0,cpu_fallback=False)
    (OUT/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (OUT/'finalization.json').write_text(json.dumps(dict(status='PASS',finalization_seconds=time.perf_counter()-start,
        evaluation_seconds=checks['seconds'],main_process_device=checks['device_name'],verification_device='cuda'),indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    print('Complete report:',OUT/'report.md')


if __name__=='__main__':main()
