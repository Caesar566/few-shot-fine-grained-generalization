"""Read-only data/weight/environment preflight. Does not train."""
from pathlib import Path
import csv,hashlib,json,platform,statistics,subprocess,sys
import torch,torchvision
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset,get_cub_transforms
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def main():
 if (OUT/'plan.json').exists():raise SystemExit('Existing plan: refusing overwrite')
 data=ROOT/'cub2002011/versions/7/CUB_200_2011';splits=ROOT/'data/splits/cub_clean_v1'
 ds={s:CUBFewShotDataset(data,splits/f'{s}_images.txt',get_cub_transforms(224,s=='train'),require_clean_index=True) for s in ['train','val']}
 assert len(ds['train'])==5890 and len(ds['val'])==2947
 assert set(ds['train'].classes).isdisjoint(ds['val'].classes)
 paths={'ImageNet':ROOT/'resnet18-f37072fd.pth','SimCLR':ROOT/'outputs/stage02_backbone/weights_verification/simclr_resnet18_backbone.pt'}
 prov=json.loads((ROOT/'reports/records/simclr_weight_provenance.json').read_text())
 source=ROOT/'checkpoint_100-resnet18-simclr-imagenet1k.ckpt'
 assert sha(paths['ImageNet'])=='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
 assert sha(source)==prov['source_sha256']
 verified=json.loads((ROOT/'outputs/stage02_backbone/weights_verification/results.json').read_text())
 assert sha(paths['SimCLR'])==verified['SimCLR_checkpoint']['export_sha256']
 from pattern_recognition_coursework.models.backbones.resnet import make_backbone
 # Preflight actual local deserialization; no forward, gradient, or optimizer.
 for p in paths.values():make_backbone(backbone_weights=p)
 cmd=subprocess.run(['nvidia-smi'],text=True,capture_output=True)
 env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'nvidia_smi_exit':cmd.returncode,'nvidia_smi_stdout':cmd.stdout,'nvidia_smi_stderr':cmd.stderr}
 save('preflight/environment.json',env)
 save('preflight/data.json',{'datasets':{s:{'images':len(d),'classes':len(d.classes)} for s,d in ds.items()},'classes_disjoint':True,'clean_marker':json.loads((splits/'dedup_manifest.json').read_text()),'test_or_nabirds_images_read':False})
 sources={
 'Random':{'pretraining_data':None,'algorithm':'torchvision random initialization, weights=None','weights':None,'sha256':None,'compatibility':'verified Stage03 Random full-backbone episodic smoke'},
 'ImageNet':{'pretraining_data':'ImageNet-1K','algorithm':'torchvision IMAGENET1K_V1 supervised','weights':str(paths['ImageNet'].relative_to(ROOT)),'sha256':sha(paths['ImageNet']),'url':'https://download.pytorch.org/models/resnet18-f37072fd.pth','compatibility':'strict local loader with 20 BN counters zero-initialized, exact native state/output equality verified'},
 'SimCLR':{'pretraining_data':'ImageNet-1K / derived set per recorded external model card, exact upstream recipe not independently audited','algorithm':'DM-Diaz SimCLR ResNet-18','weights':str(paths['SimCLR'].relative_to(ROOT)),'sha256':sha(paths['SimCLR']),'source_checkpoint':str(source.relative_to(ROOT)),'source_sha256':sha(source),'url':prov['source'],'compatibility':'120 encoder entries extracted, projection head excluded, exact source tensors verified','upstream_recipe_difference':'historical report states 100 epochs, batch32; source checkpoint has no hyper_parameters field; supervised vs SSL comparison is between specific weights, not isolated pretraining algorithms'}}
 save('preflight/weight_sources.json',sources)
 common={'schema':'stage04-episodic-v1','stage':'episodic','method':'protonet','metric':'euclidean','temperature':1.,'learnable_temperature':False,'adapter':False,'trainable_scope':'full','n_way':5,'k_shot':1,'n_query':5,'image_size':224,'epochs':10,'train_episodes_per_epoch':100,'validation_episodes':200,'learning_rate':1e-4,'optimizer':'Adam','scheduler':None,'lr_policy':'constant, as verified Stage03; no unverified scheduler introduced','workers':0,'device':'cuda','validation_seed_offset':10000,'train_episode_seed_rule':'seed + epoch','training_rng_rule':'seed + 100000 + epoch','selection':'highest fixed CUB val episode accuracy; strict >; earliest epoch on ties','data_root':str(data.relative_to(ROOT)),'splits_dir':str(splits.relative_to(ROOT)),'train_transform':repr(ds['train'].transform),'validation_transform':repr(ds['val'].transform),'read_cub_test':False,'read_nabirds':False,'model_selection_uses':'CUB val only','independent_revalidation':'reload best.pth and same fixed val plan; exact metrics comparison; reproducibility check, NOT unbiased independent test','approval_status':'PENDING_USER_APPROVAL','gpu_configuration_status':'GPU smoke not yet validated; CPU Stage03 validated','budget_status':'proposed fixed budget, not a convergence guarantee'}
 configs=[]
 for init in sources:
  for seed in [42,43,44]:
   config={**common,'experiment_id':f'{init.lower()}_seed{seed}','initialization':init,'seed':seed,'initialization_source':sources[init]}
   name=f'configs/{config["experiment_id"]}.json';save(name,config);configs.append(name)
 code={str(p.relative_to(ROOT)):sha(p) for base in ['src','scripts'] for p in (ROOT/base).rglob('*.py')}
 head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 status=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True)
 save('preflight/code_version.json',{'git_head':head,'tracked_worktree_status':status,'source_sha256':code,'note':'dirty workspace: git HEAD alone does not identify validated source'})
 with (ROOT/'outputs/stage03_smoke_train/logs/train.csv').open() as f:timings=[float(r['seconds']) for r in csv.DictReader(f) if r['phase']=='overfit']
 save('plan.json',{'status':'PENDING_APPROVAL_AND_GPU','configs':configs,'run_order_suggested':['random_seed42','imagenet_seed42','simclr_seed42','remaining seeds only after separate approval'],'parallel_training_allowed':False,'max_steps_per_experiment':1000,'number_proposed_experiments':9,'max_total_training_steps_if_all_approved':9000,'training_experiments_executed':0,'cpu_stage03_median_step_seconds':statistics.median(timings),'estimated_cpu_training_only_minutes_per_experiment':statistics.median(timings)*1000/60,'estimate_excludes_validation_io_and_checkpointing':True,'cuda_available':env['cuda_available']})
 with (OUT/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['experiment','initialization','seed','status','best_epoch','val_loss','val_accuracy','best_checkpoint','last_checkpoint']);w.writeheader()
  for name in configs:
   c=json.loads((OUT/name).read_text());w.writerow({'experiment':c['experiment_id'],'initialization':c['initialization'],'seed':c['seed'],'status':'NOT_RUN_PENDING_APPROVAL_AND_GPU'})
 print('Preflight complete. 9 proposed independent configs; 0 training runs. CUDA:',env['cuda_available'])
if __name__=='__main__':main()
