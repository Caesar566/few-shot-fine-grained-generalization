"""Render existing statistics and append verified evidence; no inference or tuning."""
import csv
import json
import subprocess
from html import escape
from pathlib import Path

root = Path(__file__).resolve().parent
run = root / 'full_seed42_600_cuda'
verification = json.loads((run / 'independent_verification.json').read_text())
assert verification['status'] == 'PASS'
performance = json.loads((run / 'performance.json').read_text())
status = json.loads((run / 'status.json').read_text())
rows = list(csv.DictReader((run / 'cub_vs_nabirds.csv').open()))

# Draw from saved summaries, never recompute classifier scores or select parameters.
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1150" height="850" viewBox="0 0 1150 850">',
       '<rect width="1150" height="850" fill="white"/>',
       '<g font-family="sans-serif" font-size="14" fill="#18212b">',
       '<text x="30" y="30" font-size="21">CUB Val versus NABirds: frozen representation comparison</text>',
       '<text x="30" y="55">Mean accuracy and 95% CI; independent task pools, descriptive comparison</text>',
       '<text x="800" y="55" fill="#3666aa">CUB Val</text>',
       '<text x="925" y="55" fill="#ca6927">NABirds</text>']
x0, width, top = 330, 750, 100
for tick in range(40, 101, 10):
    x = x0 + (tick - 40) / 60 * width
    svg += [f'<line x1="{x}" y1="80" x2="{x}" y2="790" stroke="#e0e5eb"/>',
            f'<text x="{x-12}" y="820">{tick}%</text>']
for i, row in enumerate(rows):
    y = top + i * 38
    label = f"{row['initialization']} / {row['representation']} / {row['shot']}-shot"
    svg.append(f'<text x="20" y="{y+5}">{escape(label)}</text>')
    for prefix, color, offset in [('cub_val', '#3666aa', -5), ('nabirds', '#ca6927', 5)]:
        mean = float(row[f'{prefix}_mean_percent'])
        low = float(row[f'{prefix}_ci95_low_percent'])
        high = float(row[f'{prefix}_ci95_high_percent'])
        a, b, c = [x0 + (v - 40) / 60 * width for v in (low, high, mean)]
        yy = y + offset
        svg += [f'<line x1="{a}" y1="{yy}" x2="{b}" y2="{yy}" stroke="{color}" stroke-width="2"/>',
                f'<circle cx="{c}" cy="{yy}" r="3.5" fill="{color}"/>']
svg += ['</g></svg>']
(run / 'cub_vs_nabirds.svg').write_text('\n'.join(svg))
for name in ('accuracy_bars', 'paired_lifts', 'cub_vs_nabirds'):
    subprocess.run(['rsvg-convert', str(run / f'{name}.svg'), '-o', str(run / f'{name}.png')], check=True)

marker = '\n## 最终独立验收与运行性能\n'
report_path = run / 'report.md'
report = report_path.read_text().split(marker)[0]
report += marker
report += '\n独立进程验收 **PASS**；27项CUDA回归测试全部通过。独立复核全部19307张选中图片的物理SHA256、317个物种的自动规则一致性、三份Checkpoint的CUB Train类别哈希与类别名称、10800次方法评估及270000条预测。对18个任务探针重新执行CUDA评分，三路logits逐位相等；18项配对统计和18项跨数据集描述性差值均复算一致。自动规则一致性不等于人工物种审核。\n'
report += '\n实际设备MX450 CUDA，FP32、Batch8、Query chunk4；未发生OOM或CPU回退。主评估总耗时 %.2f 秒（不包含先行数据准备、smoke与事后独立复核）。\n' % status['seconds']
report += '\n| 初始化 | 特征阶段耗时(s) | 分类耗时(s) | 已复用图片 | 新提取图片 | CUDA峰值allocated/reserved(MiB) |\n|---|---:|---:|---:|---:|---:|\n'
for name, item in performance.items():
    cache = item['feature_cache']
    report += f"| {name} | {cache['extraction_seconds']:.2f} | {item['classification_seconds']:.2f} | {cache['reused_images']} | {cache['extracted_images']} | {item['cuda_peak_allocated_mb']:.2f}/{item['cuda_peak_reserved_mb']:.2f} |\n"
report += '\n每个Backbone只缓存600对任务实际涉及的14352张图像；每份Local缓存约1.44GB，保存在CPU侧。ImageNet安全复用smoke的150张，Random/SimCLR首次提取。上述特征阶段耗时包含该阶段缓存处理；PyTorch显存统计不包含驱动上下文。\n'
report += '\n配对分析中Local在六组配置下均低于Global，差值95% CI均不含0。Fusion对ImageNet的两种Shot低于Global；Random 1-shot/5-shot、SimCLR 1-shot/5-shot的Fusion−Global区间均含0。因此Random、SimCLR 5-shot的小幅正均值差不能解释为稳定提升。六组Fusion均高于Local，其配对区间不含0；这些均为未做多重比较校正的描述性区间。\n'
report += '\n![CUB与NABirds并列均值与区间](cub_vs_nabirds.png)\n\n[独立验收JSON](independent_verification.json) · [独立验收日志](independent_verification.log) · [CUDA回归测试日志](../tests/cuda_tests.log) · [PNG准确率图](accuracy_bars.png) · [PNG配对图](paired_lifts.png)\n'
report += '\n原Stage04权重、Stage05结果及原始图片/划分保持不变，前后哈希检查通过。复现命令及筛选细节见仓库`docs/stage06_nabirds_automatic.md`；`command.json`保存实际主运行命令。\n'
report_path.write_text(report)
print(json.dumps({'status': 'PASS', 'report': str(report_path), 'png_charts': 3}, ensure_ascii=False))
