"""Bounded real-CUB ProtoNet checks, reusing the project's actual train loop."""
import copy,csv,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset,FewShotBatchSampler,get_cub_transforms
from pattern_recognition_coursework.data.episodes import dataset_fingerprint,validate_plan
from pattern_recognition_coursework.models.few_shot import FewShotModel
from pattern_recognition_coursework.models.metrics.episode import prototype_scores
from pattern_recognition_coursework.engine.experiment import run_episodes,set_seed,configure_trainable_scope
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def save(name,v):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'min':float(t.min()),'max':float(t.max()),'finite':bool(torch.isfinite(t).all())}
def clone_params(m):return {k:v.detach().clone() for k,v in m.named_parameters()}
def nested_equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(nested_equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(nested_equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 if (OUT/'config.json').exists():raise SystemExit('Existing run: refusing to overwrite')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);set_seed(42)
 for d in ['evidence','logs','figures','checkpoints']:(OUT/d).mkdir(parents=True,exist_ok=True)
 protected={str(p.relative_to(ROOT)):sha(p) for base in ['src','data/splits'] for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 cfg={'seed':42,'initialization':'Random','method':'protonet','metric':'euclidean','loss':'project run_episodes -> F.cross_entropy','optimizer':'Adam','lr':1e-4,'scheduler':None,'trainable_scope':'full','image_size':224,'n_way':5,'k_shot':1,'n_query':5,'single_batch_steps':1,'smoke_max_steps':3,'overfit_max_steps':60,'frozen_scope_probe_steps':1,'total_max_optimizer_steps':65,'device':'cpu','validation_every_steps':10,'overfit_criteria':{'fixed_train_eval_accuracy_at_least':.95,'relative_eval_loss_drop_at_least':.3},'not_generalization_claim':True}
 save('config.json',cfg)
 data=ROOT/'cub2002011/versions/7/CUB_200_2011';split=ROOT/'data/splits/cub_clean_v1'
 train=CUBFewShotDataset(data,split/'train_images.txt',get_cub_transforms(224,False),require_clean_index=True)
 aug=CUBFewShotDataset(data,split/'train_images.txt',get_cub_transforms(224,True),require_clean_index=True)
 val=CUBFewShotDataset(data,split/'val_images.txt',get_cub_transforms(224,False),require_clean_index=True)
 assert set(train.classes).isdisjoint(val.classes)
 with (split/'image_hashes.csv').open(encoding='utf-8-sig') as f:hashes={r['relative_path']:r['sha256'] for r in csv.DictReader(f)}
 def plan(ds):
  b=list(FewShotBatchSampler(ds,1,5,1,5,42));p={'schema':'few-shot-episodes-v1','dataset_fingerprint':dataset_fingerprint(ds),'n_way':5,'k_shot':1,'n_query':5,'episodes':1,'batches':b,'seed':42};validate_plan(p,ds,5,1,5,1)
  content=[hashes[ds.samples[i][0].relative_to(data/'images').as_posix()] for i in b[0]];assert len(set(content))==30
  return p
 tp,vp=plan(train),plan(val)
 def batch(ds,p):return next(iter(DataLoader(ds,batch_sampler=p['batches'],num_workers=0)))
 fixed=batch(train,tp);vb=batch(val,vp)
 def rows(ds,p):
  source_ids={line.split(maxsplit=2)[2]:line.split(maxsplit=2)[0] for line in (split/('train_images.txt' if ds is train else 'val_images.txt')).read_text().splitlines()}
  return [{'sample_index':i,'image_id':source_ids[ds.samples[i][0].relative_to(data/'images').as_posix()],'path':str(ds.samples[i][0].relative_to(ROOT)),'dataset_label':ds.samples[i][1],'class_name':ds.classes[ds.samples[i][1]],'role':'support' if n<5 else 'query','episode_label':n if n<5 else (n-5)//5} for n,i in enumerate(p['batches'][0])]
 save('evidence/input_samples.json',{'train':rows(train,tp),'validation':rows(val,vp),'train_plan':tp,'val_plan':vp,'train_preprocess':'deterministic existing eval transform for overfit diagnosis','smoke_preprocess':repr(aug.transform)})
 mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
 canvas=Image.new('RGB',(6*224,2*264),'white');draw=ImageDraw.Draw(canvas)
 for i in range(6):
  path=train.samples[tp['batches'][0][i]][0]
  with Image.open(path) as im:canvas.paste(ImageOps.fit(im.convert('RGB'),(224,224)),(i*224,0))
  rgb=((fixed[0][i:i+1]*std+mean)[0].clamp(0,1).permute(1,2,0).numpy()*255).round().astype('uint8')
  canvas.paste(Image.fromarray(rgb),(i*224,264));draw.text((i*224,228),f'original label={int(fixed[1][i])}',fill='black');draw.text((i*224,492),'preprocessed / denormalized',fill='black')
 canvas.save(OUT/'figures/input_samples.png')
 writer=SummaryWriter(str(OUT/'logs/tensorboard'));writer.add_image('input/original_and_preprocessed',np.asarray(canvas),0,dataformats='HWC');writer.add_text('input/classes',json.dumps(rows(train,tp),ensure_ascii=False),0)
 fields=['phase','step','loss','accuracy','lr','grad_norm','param_update_norm','seconds']
 stream=(OUT/'logs/train.csv').open('w',newline='');csvwriter=csv.DictWriter(stream,fieldnames=fields);csvwriter.writeheader()
 history=[];grads=[];updates=[];val_history=[];global_step=0
 def evaluate(m,b):return run_episodes(m,[b],5,1,5,torch.device('cpu'))[0]
 def step(m,opt,b,phase,n,capture=False):
  nonlocal global_step
  old=clone_params(m);captured={};hooks=[]
  if capture:
   hooks=[m.backbone.register_forward_hook(lambda mod,inp,out:captured.__setitem__('features',out.detach().clone())),m.register_forward_hook(lambda mod,inp,out:captured.__setitem__('logits',out.detach().clone()))]
  start=time.perf_counter();result,_=run_episodes(m,[b],5,1,5,torch.device('cpu'),opt)
  for h in hooks:h.remove()
  gd={};ud={};total_g=0.;total_u=0.
  for group in ['conv1','layer1','layer2','layer3','layer4']:
   params=[(k,p) for k,p in m.named_parameters() if k.startswith('backbone.'+group+'.')]
   valid=[p.grad for k,p in params if p.requires_grad]
   if valid:
    assert all(g is not None and torch.isfinite(g).all() for g in valid)
    norm=math.sqrt(sum(float(g.norm())**2 for g in valid));assert norm>0
   else:norm=0.
   delta=math.sqrt(sum(float((p.detach()-old[k]).norm())**2 for k,p in params))
   gd[group]={'gradient_norm':norm,'trainable_tensors':len(valid),'finite':True};ud[group]={'update_l2':delta,'frozen':not bool(valid)}
   if valid:assert delta>0
   else:assert delta==0
  for k,p in m.named_parameters():
   if p.grad is not None:assert torch.isfinite(p.grad).all();total_g+=float(p.grad.norm())**2
   total_u+=float((p.detach()-old[k]).norm())**2
  assert math.isfinite(result['loss']) and total_u>0
  global_step+=1;row={'phase':phase,'step':n,'loss':result['loss'],'accuracy':result['accuracy'],'lr':opt.param_groups[0]['lr'],'grad_norm':math.sqrt(total_g),'param_update_norm':math.sqrt(total_u),'seconds':time.perf_counter()-start}
  history.append(row);csvwriter.writerow(row);stream.flush();grads.append({'phase':phase,'step':n,'layers':gd});updates.append({'phase':phase,'step':n,'layers':ud})
  if n==1 or n%10==0 or n==cfg.get(phase+'_max_steps',-1):
   for tag in ['loss','accuracy','lr']:writer.add_scalar('train/'+tag,row[tag],global_step)
  if n==1 or n%20==0 or n==cfg.get(phase+'_max_steps',-1):
   for tag in ['grad_norm','param_update_norm']:writer.add_scalar('train/'+tag,row[tag],global_step)
  if capture:
   z=captured['features'];support=z[:5];query=z[5:];scores=prototype_scores(support,query,5,1,'euclidean');torch.testing.assert_close(scores,captured['logits'])
   save('evidence/single_batch.json',{'input':stats(b[0]),'dataset_labels':b[1].tolist(),'labels_dtype':str(b[1].dtype),'query_labels':torch.arange(5).repeat_interleave(5).tolist(),'features':stats(z),'support_features':stats(support),'query_features':stats(query),'prototypes':support.tolist(),'query_logits':scores.tolist(),'squared_distances_sum':(-scores*512).tolist(),'query_predictions':scores.argmax(1).tolist(),'query_cross_entropy':result['loss'],'note':'negative squared distance divided by feature dimension; prototypes are current support vectors in 1-shot'})
  print(phase,n,'loss',round(row['loss'],5),'acc',round(row['accuracy'],3),'grad',round(row['grad_norm'],4),'seconds',round(row['seconds'],2),flush=True)
  return row
 def model():set_seed(42);return FewShotModel(method='protonet',metric='euclidean',pretrained=False)
 m=model();o=torch.optim.Adam(m.parameters(),lr=1e-4)
 step(m,o,fixed,'single',1,True)
 for n in range(1,4):step(m,o,batch(aug,tp),'smoke',n)
 del m,o
 m=model();configure_trainable_scope(m,'layer4');o=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=1e-4)
 buffers={k:v.clone() for k,v in m.backbone.named_buffers()};step(m,o,fixed,'frozen_probe',1)
 assert all(torch.equal(v,buffers[k]) for k,v in m.backbone.named_buffers());del m,o
 m=model();o=torch.optim.Adam(m.parameters(),lr=1e-4);baseline=evaluate(m,fixed)
 for n in range(1,61):
  step(m,o,fixed,'overfit',n)
  if n%10==0:
   train_eval=evaluate(m,fixed);validation=evaluate(m,vb);entry={'step':n,'fixed_train_eval_loss':train_eval['loss'],'fixed_train_eval_accuracy':train_eval['accuracy'],'val_loss':validation['loss'],'val_accuracy':validation['accuracy']};val_history.append(entry)
   writer.add_scalar('val/loss',validation['loss'],n);writer.add_scalar('val/accuracy',validation['accuracy'],n)
   writer.add_scalar('overfit/fixed_train_eval_loss',train_eval['loss'],n);writer.add_scalar('overfit/fixed_train_eval_accuracy',train_eval['accuracy'],n)
 final=evaluate(m,fixed);passed=final['accuracy']>=.95 and final['loss']<=baseline['loss']*.7
 save('evidence/gradients.json',grads);save('evidence/parameter_updates.json',updates);save('evidence/overfit.json',{'baseline_fixed_train_eval':baseline,'final_fixed_train_eval':final,'criteria_passed':passed,'validation_history':val_history,'not_generalization_performance':True})
 m.eval()
 with torch.inference_mode():before_logits=m(fixed[0],5,1,5).clone()
 checkpoint={'model':m.state_dict(),'model_config':m.model_config,'optimizer':o.state_dict(),'scheduler':None,'step':60,'epoch':6,'seed':42,'initialization':'Random','config':cfg,'rng':{'torch':torch.get_rng_state(),'python':random.getstate()},'train_plan':tp,'validation_plan':vp}
 torch.save(checkpoint,OUT/'checkpoints/smoke_checkpoint.pt')
 loaded=torch.load(OUT/'checkpoints/smoke_checkpoint.pt',weights_only=True,map_location='cpu');restored=model();restored.load_state_dict(loaded['model'],strict=True);restored.eval();ro=torch.optim.Adam(restored.parameters(),lr=1e-4);ro.load_state_dict(loaded['optimizer'])
 with torch.inference_mode():after_logits=restored(fixed[0],5,1,5)
 torch.testing.assert_close(before_logits,after_logits,rtol=0,atol=0);assert nested_equal(o.state_dict(),ro.state_dict())
 save('evidence/checkpoint_test.json',{'model_logits_exact_match':True,'max_abs_difference':float((before_logits-after_logits).abs().max()),'optimizer_state_exact_match':True,'scheduler':'not used by original loop or this run','saved_step':loaded['step'],'rng_and_plan_saved':True,'exact_training_continuation_claimed':False})
 writer.flush();writer.close();stream.close()
 events=EventAccumulator(str(OUT/'logs/tensorboard')).Reload();tags=events.Tags();required=['train/loss','train/accuracy','train/lr','train/grad_norm','train/param_update_norm','val/loss','val/accuracy']
 assert all(tag in tags['scalars'] and events.Scalars(tag) for tag in required)
 save('evidence/tensorboard.json',{'tags':tags,'scalar_counts':{tag:len(events.Scalars(tag)) for tag in tags['scalars']},'gpu_memory':'not recorded: CPU execution'})
 # Standalone SVG plots drawn solely from the real CSV history.
 for field,name in [('loss','loss'),('accuracy','accuracy'),('lr','learning_rate'),('grad_norm','gradient_norm')]:
  points=[r for r in history if r['phase']=='overfit'];ys=[r[field] for r in points];lo=min(ys);hi=max(ys);span=max(hi-lo,1e-9)
  coords=' '.join(f'{55+i/(len(points)-1)*680:.2f},{330-(r[field]-lo)/span*260:.2f}' for i,r in enumerate(points))
  svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400"><rect width="800" height="400" fill="white"/><text x="55" y="28">Fixed training episode: {field} vs Step (not generalization)</text><path d="M55 60 V340 H750" fill="none" stroke="black"/><polyline points="{coords}" fill="none" stroke="#2563eb" stroke-width="2"/><text x="55" y="375">Step 1 to 60; range {lo:.6g} to {hi:.6g}</text></svg>'
  (OUT/f'figures/{name}.svg').write_text(svg)
 # Validation and fixed-training evaluation are separate genuine observations.
 for metric in ['loss','accuracy']:
  ys=[r['fixed_train_eval_'+metric] for r in val_history]+[r['val_'+metric] for r in val_history];lo=min(ys);span=max(max(ys)-lo,1e-9)
  lines=''
  for key,color in [('fixed_train_eval_'+metric,'#2563eb'),('val_'+metric,'#dc2626')]:
   pts=' '.join(f'{60+i/5*670},{330-(r[key]-lo)/span*260}' for i,r in enumerate(val_history));lines+=f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'
  (OUT/f'figures/train_val_{metric}.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="400"><rect width="800" height="400" fill="white"/><text x="55" y="25">{metric}: blue=fixed train eval; red=independent val episode; steps10..60</text>{lines}</svg>')
 assert all(sha(ROOT/p)==h for p,h in protected.items())
 tests=[{'check':name,'status':status,'evidence':ev} for name,status,ev in [
 ('真实CUB单批前向','PASS','single_batch.json'),('loss/输出/梯度有限','PASS','single_batch.json,gradients.json'),('关键层有效梯度','PASS','gradients.json'),('参数更新与冻结','PASS','parameter_updates.json'),('固定训练样本过拟合','PASS' if passed else 'FAIL','overfit.json'),('逐step真实日志','PASS','../logs/train.csv'),('TensorBoard可读取','PASS','tensorboard.json'),('CSV及SVG曲线','PASS','../figures'),('checkpoint预测与optimizer恢复','PASS','checkpoint_test.json'),('无测试数据训练/原文件保全/步数上限','PASS','input_samples.json')]]
 save('evidence/tests.json',tests);save('evidence/preservation.json',{'files_checked':len(protected),'changed_files':[],'actual_optimizer_steps':global_step,'maximum_steps':65})
 print('FINISHED overfit=',passed,'baseline=',baseline['loss'],baseline['accuracy'],'final=',final['loss'],final['accuracy'],flush=True)
if __name__=='__main__':main()
