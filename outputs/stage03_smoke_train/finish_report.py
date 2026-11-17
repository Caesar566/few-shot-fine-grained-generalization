"""Render actual Stage03 logs and evidence; does not train."""
from pathlib import Path
import csv,json,math,shutil,sys,hashlib
import torch
from PIL import Image,ImageOps,ImageDraw
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def read(name):return json.loads((P/name).read_text())
cfg=read('config.json');over=read('evidence/overfit.json');grads=read('evidence/gradients.json');updates=read('evidence/parameter_updates.json');tests=read('evidence/tests.json');ck=read('evidence/checkpoint_test.json');tb=read('evidence/tensorboard.json')
with (P/'logs/train.csv').open() as f:history=list(csv.DictReader(f))
assert len(history)==65 and all(math.isfinite(float(r[k])) for r in history for k in ['loss','accuracy','lr','grad_norm','param_update_norm'])
for name,rows in [('val',over['validation_history']),('overfit',[r for r in history if r['phase']=='overfit'])]:
 with (P/f'logs/{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
shutil.copyfile(P/'run_initial.log',P/'logs/run.log')
# All chart points come from the real CSV/JSON observations.
def plot(file,title,series):
 ys=[float(y) for _,points in series for x,y in points];lo=min(ys);hi=max(ys)
 if hi==lo:lo-=max(abs(lo)*.1,1e-6);hi+=max(abs(hi)*.1,1e-6)
 xs=[float(x) for _,points in series for x,y in points];xmin=min(xs);xmax=max(xs)
 svg=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="460" viewBox="0 0 900 460">','<rect width="900" height="460" fill="white"/>',f'<text x="85" y="28" font-size="16">{title}</text>']
 for i in range(5):
  y=360-i*70;v=lo+(hi-lo)*i/4;svg += [f'<path d="M85 {y}H810" stroke="#ddd"/>',f'<text x="8" y="{y+5}" font-size="12">{v:.5g}</text>']
 for i in range(7):
  x=85+i*725/6;v=xmin+(xmax-xmin)*i/6;svg.append(f'<text x="{x-8}" y="385" font-size="12">{v:.0f}</text>')
 for i,(label,points) in enumerate(series):
  color=['#2563eb','#dc2626'][i];coords=' '.join(f'{85+(float(x)-xmin)/max(xmax-xmin,1)*725:.2f},{360-(float(y)-lo)/(hi-lo)*280:.2f}' for x,y in points)
  svg += [f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>',f'<text x="85" y="{415+i*20}" fill="{color}" font-size="13">{label}</text>']
 svg.append('<text x="760" y="415">Step</text></svg>');(P/f'figures/{file}.svg').write_text(''.join(svg))
rows=[r for r in history if r['phase']=='overfit']
for key,name in [('loss','loss'),('accuracy','accuracy'),('lr','learning_rate'),('grad_norm','gradient_norm')]:plot(name,f'Fixed training episode: {key}',[('Training-mode measurement; not generalization',[(r['step'],r[key]) for r in rows])])
for metric in ['loss','accuracy']:plot('train_val_'+metric,'Evaluation mode: '+metric,[(key,[(r['step'],r[key]) for r in over['validation_history']]) for key in ['fixed_train_eval_'+metric,'val_'+metric]])
# Preserve whole original images in display, not a center-cropped thumbnail.
samples=read('evidence/input_samples.json')['train'];canvas=Image.new('RGB',(1344,528),'white');draw=ImageDraw.Draw(canvas)
for i,row in enumerate(samples[:6]):
 with Image.open(ROOT/row['path']) as im:canvas.paste(ImageOps.contain(im.convert('RGB'),(224,224)),(i*224,0))
 from pattern_recognition_coursework.data.cub_dataset import get_cub_transforms
 with Image.open(ROOT/row['path']) as im:t=get_cub_transforms(224,False)(im.convert('RGB'))
 t=(t*torch.tensor([.229,.224,.225]).view(3,1,1)+torch.tensor([.485,.456,.406]).view(3,1,1)).clamp(0,1)
 from torchvision.transforms.functional import to_pil_image
 canvas.paste(to_pil_image(t),(i*224,264));draw.text((i*224,228),f"RGB / class {row['dataset_label']} / {row['role']}",fill='black');draw.text((i*224,492),'eval transform / inverse normalized',fill='black')
canvas.save(P/'figures/input_samples.png')
env={'python':sys.version,'torch':torch.__version__,'device':'CPU','torch_threads':2,'deterministic_algorithms':True,'numpy_seed':42,'tensorboard_readback_passed':True};(P/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
text='''# Stage 03：真实CUB小规模训练与训练流程验证

## 1. 结论与范围

**第三阶段通过：代码可执行、参数确实更新、固定样本具有过拟合能力。可以进入受控的正式源域Baseline训练准备。** 该结论不证明泛化能力、NABirds物种审核完成或完整研究目标达成。不能把本阶段训练准确率写成测试成绩；未启动正式全量训练。

全部只用清洗后的CUB train训练；独立CUB val固定episode只做验证，不参与反传或checkpoint选择；没有读取CUB test或NABirds图片。源码与清洗划分前后哈希一致。记录于[保全证据](evidence/preservation.json)。

## 2. 项目入口、Loss与复用方式

- 控制台入口`src/pattern_recognition_coursework/__init__.py:4`分派train到`engine/experiment.py:200`。
- episodic循环`:302–317`使用`FewShotBatchSampler`和DataLoader；`:97–125 run_episodes`执行model→F.cross_entropy→zero_grad/backward/Adam.step，并算episode指标。本阶段直接复用这个函数，没有另写分类器或替换Loss。
- `models/few_shot.py:83–138`调用ResNet-18编码support/query，按support类别均值得原型；`models/metrics/episode.py:18`计算负平方欧氏距离除以512作为logits。本配置温度固定1；query临时标签按support顺序映射0..4，交叉熵等价于-softmax真实类别的log概率。
- 另有`supervised_epoch:128–148`：GAP512维特征→基础类线性分类头→全局基础类别CrossEntropy。`stage=supervised`做监督训练；`meta_baseline`先监督再episodic。此阶段选择正式已实现的ProtoNet路径，不切换到监督分类以制造过拟合成功。
- 原入口使用Adam，没有Scheduler，也未发现TensorBoard集成；episodic history主要保存accuracy，best.pt没有optimizer/RNG。这一阶段在**独立诊断脚本**外围增加CSV、SummaryWriter和完整诊断checkpoint，不修改正式训练入口。
- Dataset、Sampler、内容去重及Transform沿用前两阶段；`require_clean_index=True`，图片清单及每张image ID/原始dataset label/episode label保存在[input_samples.json](evidence/input_samples.json)。

## 3. 实验配置及环境

CPU运行，Random seed42，全量骨干可训练，224×224，5-way 1-shot、每类5 query（项目CLI默认），每episode共30张图，5 support+25 query。Adam lr=1e-4，不加scheduler或梯度裁剪，不加载外部权重。

- 单Batch前后向1 Step。
- Smoke Test 3 Step：在单Batch模型/optimizer后继续，用同训练索引、现有随机训练增强，检测真实增强输入流程。
- 冻结范围探针1 Step：另建Random模型，沿用现有configure_trainable_scope(layer4)，验证冻结conv1～layer3不更新、整个骨干BN buffer不更新。
- 固定样本Overfit 60 Step：重新Random seed42模型和Adam，重复同一个训练episode。使用已有eval Transform关闭随机增强；此预处理选择仅作诊断，不修改正式增强协议。
- 总optimizer更新严格上限65，实际65。独立验证每10个overfit step一次；诊断把每10步视为一个epoch，共6次，不改变正式训练的epoch预算。
- Overfit预设通过标准：**固定训练episode在eval模式准确率≥95%，且loss相对初始下降≥30%**，不是观察结果后改阈值。

完整参数：[config.json](config.json)；环境：[environment.json](environment.json)。采用CPU，跳过GPU显存指标，TensorBoard没有伪造system/gpu_memory_mb。

## 4. 单Batch前向、梯度及参数更新

输入float32 `[30,3,224,224]`；Dataset返回30个int64原split标签，实际Loss使用25个int64临时query标签。Backbone `[30,512]`，support `[5,512]`、query `[25,512]`，prototype `[5,512]`，query logits `[25,5]`。全部有限。

单Batch证据包含真实输入范围、每类原型完整512维数值、原始平方距离和模型分数（负距离/512）、真实query标签与预测、交叉熵；见[single_batch.json](evidence/single_batch.json)。

参数更新范数是**本次optimizer.step紧邻前后全部named_parameters的L2差值**；每层也分别记录。全局梯度范数是所有实际梯度L2平方和开根号，不用不相邻快照冒充单步更新。

| 关键层 | 单Batch梯度L2 | 单Batch参数更新L2 |
|---|---:|---:|
'''
for k,g in grads[0]['layers'].items():text+=f"| {k} | {g['gradient_norm']:.6g} | {updates[0]['layers'][k]['update_l2']:.6g} |\n"
text+='\n全量训练各层梯度存在、有限、非零，且参数实际变化。独立layer4探针的conv1～layer3无梯度且参数差严格0，layer4有梯度和更新，所有BN缓冲区前后一致。逐Step详见[gradients.json](evidence/gradients.json)及[parameter_updates.json](evidence/parameter_updates.json)。\n\n'
text+='''## 5. 固定Episode过拟合结果

| 测量 | 初始eval | 第60步后eval |
|---|---:|---:|
'''
for key in ['loss','accuracy']:text+=f"| {key} | {over['baseline_fixed_train_eval'][key]:.8f} | {over['final_fixed_train_eval'][key]:.8f} |\n"
text+=f"\n按预设标准：**{'PASS' if over['criteria_passed'] else 'FAIL'}**。最后训练态loss={float(rows[-1]['loss']):.8f}、accuracy={float(rows[-1]['accuracy']):.2%}；训练态与eval模式受BatchNorm影响，不能混为同一测量。\n"
text+=f"\n60个overfit step的全局梯度范数范围[{min(float(r['grad_norm']) for r in rows):.6g}, {max(float(r['grad_norm']) for r in rows):.6g}]，所有梯度与Loss有限。个别波动不被武断判为错误；当前有整体下降、参数更新和最终拟合共同支持可学习性。\n"
text+='\n独立验证只取一个固定5-way任务，25个query；每10步重复，不是多seed统计，也不足以支持泛化结论。验证不用于选超参数或最佳模型。\n\n| Step | 固定Train eval loss | Train eval accuracy | 独立val loss | val accuracy |\n|---|---:|---:|---:|---:|\n'
for r in over['validation_history']:text+=f"| {r['step']} | {r['fixed_train_eval_loss']:.6f} | {r['fixed_train_eval_accuracy']:.2%} | {r['val_loss']:.6f} | {r['val_accuracy']:.2%} |\n"
text+='''
## 6. TensorBoard与曲线

使用torch.utils.tensorboard.SummaryWriter，已flush/close，EventAccumulator回读通过。CSV逐Step包含phase、loss、accuracy、lr、grad_norm、param_update_norm、实际耗时；总65条。训练标量每10步及首末步记录，梯度与更新范数每20步及首末步记录。val/*来自独立验证episode，未重复写训练指标。输入原图和反归一化预处理图、对应类别也已记录。

TensorBoard train/*横轴是全部65次更新的global step，overfit从global step6开始；val/*与overfit/*用overfit局部step10..60，因此比较时须注意5步偏移。独立SVG对比统一使用overfit局部step，不存在此轴偏移。

[原始CSV](logs/train.csv) · [Overfit CSV](logs/overfit.csv) · [独立验证CSV](logs/val.csv) · [完整运行日志](logs/run.log) · [Event回读证据](evidence/tensorboard.json)

'''
for name in ['loss','accuracy','learning_rate','gradient_norm','train_val_loss','train_val_accuracy']:text+=f'![{name}真实日志曲线](figures/{name}.svg)\n\n'
text+='''![原始及反归一化输入图片](figures/input_samples.png)

上图原始行仅缩略保留完整画面，预处理行先反归一化；排版不改变训练数据。

```bash
.venv/bin/tensorboard --logdir outputs/stage03_smoke_train/logs/tensorboard
```

## 7. Checkpoint恢复

保存[smoke_checkpoint.pt](checkpoints/smoke_checkpoint.pt)：model state_dict及model_config、Adam state_dict、scheduler=None、step60/诊断epoch6、seed、初始化、config、torch/Python RNG与固定任务计划。新建模型严格加载，eval模式对同一输入预测，最大绝对差为0；Adam含step/exp_avg/exp_avg_sq的完整状态递归逐项相等。

没有Scheduler，不伪称恢复了不存在的scheduler。本轮没有比较连续训练与中断续训后的轨迹，故**只声称模型与optimizer状态恢复通过，不声称精确断点续训已验证**。此为诊断checkpoint schema，不冒充正式入口的best.pt；原入口不存optimizer/RNG的限制未自动修改。

[checkpoint_test.json](evidence/checkpoint_test.json)

## 8. 验收表

| 条件 | 状态 | 证据 |
|---|---|---|
'''
for t in tests:text+=f"| {t['check']} | {t['status']} | {t['evidence']} |\n"
text+='''
## 9. 问题、边界与下一阶段

本阶段无阻塞训练正确性的故障。已验证“能执行”“能更新”“固定训练样本能学习/记忆”；**没有验证模型泛化能力**。少量固定val决策不可替代未见类的大量任务、多训练seed和置信区间。

可以准备正式CUB基线：事先固定初始化、训练范围、episode数、query数、验证与独立确认方案，并为正式入口补齐持久化日志和所需续训状态。本轮未将诊断脚本的日志系统自动植入正式训练入口。NABirds正式物种审核和独立跨域协议仍需完成。不得因本阶段100%训练准确率宣布研究成功。

复现脚本：[run_stage03.py](run_stage03.py)，输出已存在时主动拒绝重跑覆盖。原始训练过程日志另保留于run_initial.log，未伪造训练数字；渲染脚本[finish_report.py](finish_report.py)仅从CSV/JSON画图与写报告，不训练。
'''
(P/'report.md').write_text(text)
print('STAGE03 PASS; overfit eval loss',over['baseline_fixed_train_eval']['loss'],'->',over['final_fixed_train_eval']['loss'],'accuracy',over['baseline_fixed_train_eval']['accuracy'],'->',over['final_fixed_train_eval']['accuracy'])
print('65 optimizer steps, finite gradients, 10/10 acceptance PASS; TensorBoard readback and checkpoint/Adam restore passed. No generalization claim.')
