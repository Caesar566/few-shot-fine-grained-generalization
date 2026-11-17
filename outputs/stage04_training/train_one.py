"""One explicitly selected Stage04 experiment; never run all configs in parallel."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,csv,fcntl,json,logging,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset,FewShotBatchSampler,get_cub_transforms
from pattern_recognition_coursework.data.episodes import load_or_create_plan,plan_hash
from pattern_recognition_coursework.engine.experiment import run_episodes,set_seed,model_from_config,sha256_file
from pattern_recognition_coursework.models.few_shot import FewShotModel
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
def atomic_save(obj,path):
 temp=path.with_suffix('.tmp');torch.save(obj,temp);os.replace(temp,path)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 c=json.loads(a.config.read_text());assert c['schema']=='stage04-episodic-v1'
 assert c['stage']=='episodic' and c['method']=='protonet' and c['scheduler'] is None
 assert c['device'] in ['cuda','cpu'] and c['epochs']*c['train_episodes_per_epoch']<=1000
 if not a.execute:
  print('Configuration parsed. No training. Requires explicit user approval and --execute.');return
 if c['device']=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA unavailable: no training started; CPU fallback prohibited')
 # Process-wide serial exclusion, including separate shell invocations.
 lock=(OUT/'.training.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit('Another Stage04 experiment is active')
 run=OUT/'runs'/c['experiment_id']
 run.mkdir(parents=True,exist_ok=False)
 for name in ['logs','checkpoints','figures']:(run/name).mkdir()
 log=logging.getLogger('stage04');log.setLevel(logging.INFO);log.addHandler(logging.FileHandler(run/'logs/run.log'));log.addHandler(logging.StreamHandler())
 writer=None
 try:
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);set_seed(c['seed'])
  device=torch.device(c['device']);source=c['initialization_source'];weights=ROOT/source['weights'] if source['weights'] else None
  if weights and sha256_file(weights)!=source['sha256']:raise ValueError('Initialization file hash changed')
  data=ROOT/c['data_root'];splits=ROOT/c['splits_dir']
  train=CUBFewShotDataset(data,splits/'train_images.txt',get_cub_transforms(c['image_size'],True),require_clean_index=True)
  val=CUBFewShotDataset(data,splits/'val_images.txt',get_cub_transforms(c['image_size'],False),require_clean_index=True)
  assert set(train.classes).isdisjoint(val.classes)
  for ds in [train,val]:assert min(map(len,ds.class_to_indices.values()))>=c['k_shot']+c['n_query']
  plan=load_or_create_plan(run/'validation_plan.json',val,c['n_way'],c['k_shot'],c['n_query'],c['validation_episodes'],c['seed']+10000)
  m=FewShotModel(method=c['method'],metric=c['metric'],backbone_weights=weights).to(device)
  opt=torch.optim.Adam(m.parameters(),lr=c['learning_rate']);scheduler=None
  training={**c,'eval_episodes':c['validation_episodes'],'split_hashes':{s:sha256_file(splits/f'{s}_images.txt') for s in ['train','val','test']},'train_classes_sha256':sha256_file(splits/'train_classes.txt')}
  write(run/'config.json',c);write(run/'command.json',{'argv':[sys.executable,*sys.argv],'cwd':str(ROOT),'initialization_info':m.backbone_load_info})
  import torchvision,platform,subprocess
  write(run/'environment.json',{'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'platform':platform.platform(),'cuda':torch.version.cuda,'device':str(device),'gpu_name':torch.cuda.get_device_name() if device.type=='cuda' else None})
  write(run/'code_version.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'status':subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True),'source_files':{str(f.relative_to(ROOT)):sha256_file(f) for f in (ROOT/'src').rglob('*.py')},'runner_sha256':sha256_file(Path(__file__))})
  writer=SummaryWriter(str(run/'logs/tensorboard'));history=[];best=-1.;step=0
  with (run/'logs/steps.csv').open('w',newline='') as stream:
   csvwriter=csv.DictWriter(stream,fieldnames=['epoch','step','train_loss','train_accuracy','learning_rate']);csvwriter.writeheader()
   for epoch in range(1,c['epochs']+1):
    set_seed(c['seed']+100000+epoch)
    sampler=FewShotBatchSampler(train,c['train_episodes_per_epoch'],c['n_way'],c['k_shot'],c['n_query'],seed=c['seed']+epoch)
    losses=[];accuracies=[];start=time.perf_counter()
    for batch in DataLoader(train,batch_sampler=sampler,num_workers=c['workers']):
     before_probe={k:par.detach().clone() for k,par in m.named_parameters()} if step<2 else None
     metrics,_=run_episodes(m,[batch],c['n_way'],c['k_shot'],c['n_query'],device,opt)
     assert np.isfinite(metrics['loss']);assert all(torch.isfinite(par.grad).all() for par in m.parameters() if par.grad is not None)
     if before_probe is not None:
      assert all(par.grad is not None for par in m.parameters() if par.requires_grad)
      assert any(not torch.equal(par.detach(),before_probe[k]) for k,par in m.named_parameters())
      write(run/'gpu_smoke.json',{'steps_checked':step+1,'finite_loss_and_gradients':True,'actual_parameter_update':True,'device':str(device),'allocated_mb':torch.cuda.memory_allocated()/2**20 if device.type=='cuda' else None,'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,'note':'first two steps are part of the approved 1000-step budget; no extra training'})
     step+=1;losses.append(metrics['loss']);accuracies.append(metrics['accuracy'])
     csvwriter.writerow({'epoch':epoch,'step':step,'train_loss':metrics['loss'],'train_accuracy':metrics['accuracy'],'learning_rate':opt.param_groups[0]['lr']});stream.flush()
     if step==1 or step%10==0:
      writer.add_scalar('train/loss',metrics['loss'],step);writer.add_scalar('train/accuracy',metrics['accuracy'],step)
    vm,_=run_episodes(m,DataLoader(val,batch_sampler=plan['batches'],num_workers=c['workers']),c['n_way'],c['k_shot'],c['n_query'],device)
    row={'epoch':epoch,'step':step,'train_loss':float(np.mean(losses)),'train_accuracy':float(np.mean(accuracies)),'validation_loss':vm['loss'],'validation_accuracy':vm['accuracy'],'learning_rate':opt.param_groups[0]['lr'],'seconds':time.perf_counter()-start}
    history.append(row)
    with (run/'logs/epochs.csv').open('w',newline='') as s:
     w=csv.DictWriter(s,fieldnames=list(row));w.writeheader();w.writerows(history)
    for name in ['validation_loss','validation_accuracy','learning_rate']:writer.add_scalar(name,row[name],epoch)
    checkpoint={'model':m.state_dict(),'model_config':m.model_config,'training_config':training,'backbone':m.backbone.state_dict(),'optimizer':opt.state_dict(),'scheduler':None,'epoch':epoch,'step':step,'validation_metrics':vm,'validation_plan_hash':plan_hash(plan),'seed':c['seed'],'rng':{'torch':torch.get_rng_state(),'python':random.getstate(),'numpy':list(np.random.get_state()[1].tolist()),'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else []}}
    atomic_save(checkpoint,run/'checkpoints/last.pth')
    if vm['accuracy']>best:
     best=vm['accuracy'];atomic_save(checkpoint,run/'checkpoints/best.pth')
    log.info(json.dumps(row));writer.flush()
  selected=torch.load(run/'checkpoints/best.pth',map_location=device,weights_only=True)
  fresh=model_from_config(selected['model_config'],load_initial_weights=False).to(device);fresh.load_state_dict(selected['model'],strict=True)
  repeated,_=run_episodes(fresh,DataLoader(val,batch_sampler=plan['batches'],num_workers=0),c['n_way'],c['k_shot'],c['n_query'],device)
  assert repeated==selected['validation_metrics'],'Independent reload validation differs'
  atomic_save({'backbone':fresh.backbone.state_dict(),'config':c,'selected_epoch':selected['epoch'],'validation':repeated},run/'checkpoints/best_backbone.pth')
  for name in ['loss','accuracy','learning_rate']:
   keys=['learning_rate'] if name=='learning_rate' else ['train_'+name,'validation_'+name]
   values=[h[k] for h in history for k in keys];lo=min(values);span=max(max(values)-lo,1e-8);svg='<svg xmlns="http://www.w3.org/2000/svg" width="850" height="400"><rect width="850" height="400" fill="white"/>'
   for i,k in enumerate(keys):
    coords=' '.join(f'{65+j/max(len(history)-1,1)*700},{330-(h[k]-lo)/span*260}' for j,h in enumerate(history));color=['blue','red'][i];svg+=f'<polyline points="{coords}" fill="none" stroke="{color}"/><text x="65" y="{25+i*20}" fill="{color}">{k}</text>'
   svg+=f'<text x="65" y="380">Epoch 1..{len(history)}; value range {lo:.6g}..{max(values):.6g}</text></svg>';(run/f'figures/{name}.svg').write_text(svg)
  write(run/'result.json',{'status':'completed','experiment':c['experiment_id'],'initialization':c['initialization'],'seed':c['seed'],'best_epoch':selected['epoch'],'best_validation':repeated,'last_validation':history[-1],'independent_reload_validation_exact':True,'best_checkpoint':str((run/'checkpoints/best.pth').relative_to(ROOT)),'last_checkpoint':str((run/'checkpoints/last.pth').relative_to(ROOT))})
 except Exception as e:
  log.exception('Run failed');write(run/'result.json',{'status':'failed','experiment':c['experiment_id'],'error':repr(e)});raise
 finally:
  if writer:writer.flush();writer.close()
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  from summarize_runs import refresh
  refresh()
if __name__=='__main__':main()
