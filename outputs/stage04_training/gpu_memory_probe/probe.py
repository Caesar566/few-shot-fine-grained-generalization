"""Exactly the approved two-step FP32 GPU probe; no full training or retries."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import json,time,traceback,sys
import torch
from torch.utils.data import DataLoader
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset,FewShotBatchSampler,get_cub_transforms
from pattern_recognition_coursework.models.few_shot import FewShotModel
from pattern_recognition_coursework.engine.experiment import set_seed,run_episodes
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
result={'status':'STARTING','approved_max_steps':2,'completed_optimizer_steps':0,'full_training_started':False,'amp_enabled':False,'config_modified':False,'steps':[]}
phase='initialization'
def memory():
 free,total=torch.cuda.mem_get_info()
 return {'total_mib':total/2**20,'free_mib':free/2**20,'allocated_mib':torch.cuda.memory_allocated()/2**20,'reserved_mib':torch.cuda.memory_reserved()/2**20,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
def snapshot(): (OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
def main():
 global phase
 if (OUT/'result.json').exists():raise SystemExit('Existing probe: refusing overwrite')
 try:
  result['cuda_available']=torch.cuda.is_available()
  if not result['cuda_available']:result['status']='BLOCKED_CUDA_UNAVAILABLE';return
  result['gpu_name']=torch.cuda.get_device_name(0);result['torch']=torch.__version__
  result['initial_memory']=memory();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);set_seed(42)
  c=json.loads((ROOT/'outputs/stage04_training/configs/random_seed42.json').read_text());result['config']=c
  ds=CUBFewShotDataset(ROOT/c['data_root'],ROOT/c['splits_dir']/'train_images.txt',get_cub_transforms(224,True),require_clean_index=True)
  model=FewShotModel(method='protonet',metric='euclidean').cuda()
  assert all(p.dtype==torch.float32 for p in model.parameters())
  optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
  def forward_start(*args):
   global phase
   phase='forward'
  def forward_end(*args):
   global phase
   phase='loss_or_backward'
  def optimizer_start(*args):
   global phase
   phase='adam_step'
  model.register_forward_pre_hook(forward_start);model.register_forward_hook(forward_end)
  optimizer.register_step_pre_hook(optimizer_start)
  set_seed(c['seed']+100000+1)
  loader=DataLoader(ds,batch_sampler=FewShotBatchSampler(ds,2,5,1,5,seed=43),num_workers=0)
  result['after_model_memory']=memory()
  for index,batch in enumerate(loader,1):
   phase='parameter_snapshot';result['attempted_step']=index;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   before={k:p.detach().clone() for k,p in model.named_parameters()}
   start=time.perf_counter()
   metrics,_=run_episodes(model,[batch],5,1,5,torch.device('cuda'),optimizer)
   torch.cuda.synchronize();phase='post_step_checks'
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
   changed=any(not torch.equal(p.detach(),before[k]) for k,p in model.named_parameters());assert changed
   result['completed_optimizer_steps']=index
   result['steps'].append({'step':index,'input_shape':list(batch[0].shape),'loss':metrics['loss'],'accuracy':metrics['accuracy'],'finite_gradients':True,'parameters_changed':changed,'seconds':time.perf_counter()-start,'memory':memory()})
   print(json.dumps(result['steps'][-1]),flush=True);del before;snapshot()
  result['status']='PASS_TWO_STEPS'
 except torch.cuda.OutOfMemoryError as e:
  result['status']='OOM';result['failure_phase']=phase;result['error']=str(e);result['memory_at_failure']=memory();result['traceback']=traceback.format_exc();print('OOM at',phase,str(e),flush=True)
 except Exception as e:
  result['status']='ERROR';result['failure_phase']=phase;result['error']=repr(e);result['traceback']=traceback.format_exc();print(result['traceback'],flush=True)
 finally:snapshot()
 print('PROBE RESULT',result['status'],'completed steps',result['completed_optimizer_steps'],flush=True)
if __name__=='__main__':main()
