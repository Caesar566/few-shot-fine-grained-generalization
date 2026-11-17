"""CPU-only backbone checks. Network downloads forbidden; no training."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import warnings
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision
from pattern_recognition_coursework.models.backbones import resnet as module
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset,get_cub_transforms
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def stats(t):
 t=t.detach().cpu();return {'shape':list(t.shape),'dtype':str(t.dtype),'finite':bool(torch.isfinite(t).all()),'min':float(t.min()),'max':float(t.max()),'mean':float(t.mean()),'std':float(t.std()),'nonzero_fraction':float((t!=0).float().mean())}
def fingerprint(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 torch.set_num_threads(2)
 protected={str(p.relative_to(ROOT)):sha(p) for base in ['src','scripts','data/splits'] for p in (ROOT/base).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 tests=[]
 def check(name,fn):
  try:detail=fn();tests.append({'name':name,'status':'PASS','detail':detail})
  except Exception as e:tests.append({'name':name,'status':'FAIL','error':repr(e)})
  print(name,tests[-1]['status'],flush=True)
 seed=42
 factory=module.resnet18
 with patch.object(module,'resnet18',wraps=factory) as construct,patch('torch.load',side_effect=AssertionError('Random must not load checkpoint')),patch('torch.hub.load_state_dict_from_url',side_effect=AssertionError('download forbidden')):
  torch.manual_seed(seed);model=module.make_backbone()
  assert construct.call_args.kwargs=={'weights':None}
  assert model.backbone_load_info is None
 tests.append({'name':'random_never_loads_pretraining','status':'PASS','detail':'weights=None; torch.load and URL loaders forbidden'})
 torch.manual_seed(seed);repeat=module.make_backbone();assert fingerprint(model)==fingerprint(repeat)
 torch.manual_seed(43);other=module.make_backbone();assert fingerprint(model)!=fingerprint(other)
 tests.append({'name':'random_seed_reproducibility','status':'PASS'})
 del repeat,other
 data=ROOT/'cub2002011/versions/7/CUB_200_2011'
 ds=CUBFewShotDataset(data,ROOT/'data/splits/cub_clean_v1/val_images.txt',get_cub_transforms(224,False),require_clean_index=True)
 sample=json.loads((ROOT/'outputs/stage01_data/sample_manifest.json').read_text())
 indices=[r['sample_index'] for r in sample if r['dataset']=='cub_val']
 batch=torch.stack([ds[i][0] for i in indices]);save('evidence/input_batch.json',{'indices':indices,'images':[str(ds.samples[i][0].relative_to(ROOT)) for i in indices],'labels':[ds[i][1] for i in indices],'tensor':stats(batch),'preprocessing':repr(ds.transform)})
 shapes={};handles=[]
 for name in ['conv1','bn1','relu','maxpool','layer1','layer2','layer3','layer4','avgpool','fc']:
  handles.append(getattr(model,name).register_forward_hook(lambda m,i,o,n=name:shapes.__setitem__(n,list(o.shape))))
 model.eval();before=fingerprint(model)
 with torch.inference_mode():global_features=model(batch)
 for h in handles:h.remove()
 with torch.inference_mode():spatial=module.spatial_features(model,batch)
 assert list(spatial.shape)==[4,512,7,7] and list(global_features.shape)==[4,512]
 torch.testing.assert_close(spatial.mean((-1,-2)),global_features)
 assert torch.isfinite(spatial).all() and torch.isfinite(global_features).all()
 assert float(global_features.std())>1e-6 and float((global_features[0]-global_features[1]).abs().max())>1e-6
 assert before==fingerprint(model)
 tests.append({'name':'224_shapes_finite_nonconstant_and_gap_equivalence','status':'PASS','detail':shapes})
 additional={}
 with torch.inference_mode():
  for size in [84,64]:
   x=torch.nn.functional.interpolate(batch,size=(size,size),mode='bilinear',align_corners=False)
   additional[str(size)]={'spatial':stats(module.spatial_features(model,x)),'global':stats(model(x))}
 results={'Random':{'status':'VERIFIED','weight_source':None,'seed':42,'load_info':model.backbone_load_info,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'state_dict_entries':len(model.state_dict()),'fingerprint':before,'spatial':stats(spatial),'global':stats(global_features),'different_image_max_abs_difference':float((global_features[0]-global_features[1]).abs().max()),'eval_preserves_state':True}}
 (OUT/'network.txt').write_text(str(model))
 save('shapes.json',{'input':list(batch.shape),'stages':shapes,'other_sizes':additional})
 # Actual local weight availability. Never invoke pretrained=True without a cache.
 weights=ResNet18_Weights.IMAGENET1K_V1
 caches=[ROOT/'data/cache/torch/checkpoints/resnet18-f37072fd.pth',Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth']
 available=[p for p in caches if p.is_file()]
 results['ImageNet Supervised']={'status':'BLOCKED_MISSING_WEIGHTS','source':weights.url,'expected_enum':str(weights),'searched_paths':[str(p) for p in caches],'missing_keys':None,'unexpected_keys':None,'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters()),'feature_stats':None,'note':'architecture parameter counts only; no pretrained instance claimed'}
 if available:
  path=available[0]
  if not sha(path).startswith('f37072fd'):raise ValueError('ImageNet file digest does not match official filename hash')
  with patch('torch.hub.load_state_dict_from_url',side_effect=AssertionError('download forbidden')):
   actual=module.make_backbone(backbone_weights=path).eval()
  source=torch.load(path,map_location='cpu',weights_only=True)
  for k,v in actual.state_dict().items():torch.testing.assert_close(v,source[k],rtol=0,atol=0)
  with torch.inference_mode():g=actual(batch);s=module.spatial_features(actual,batch)
  results['ImageNet Supervised'].update(status='VERIFIED_LOCAL_OFFICIAL_HASH',load_info=actual.backbone_load_info,missing_keys=[],unexpected_keys=[],global_stats=stats(g),spatial_stats=stats(s))
 provenance=json.loads((ROOT/'reports/records/simclr_weight_provenance.json').read_text())
 simpaths=[ROOT/'data/cache/weights'/provenance['checkpoint_filename'],ROOT/'data/cache/weights/simclr_resnet18_backbone.pt']
 results['SimCLR']={'status':'BLOCKED_MISSING_WEIGHTS','recorded_source':provenance,'searched_paths':[str(p) for p in simpaths],'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters()),'feature_stats':None,'missing_keys':None,'unexpected_keys':None,'note':'counts describe the common architecture only; provenance claims not externally verified'}
 # Extract original conversion function without importing the training engine.
 tree=ast.parse((ROOT/'scripts/convert_simclr_resnet18.py').read_text())
 ns={'torch':torch,'nn':nn,'resnet18':resnet18,'STAGES':{'0':'conv1','1':'bn1','4':'layer1','5':'layer2','6':'layer3','7':'layer4'}}
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_encoder'],type_ignores=[]),'original_extract_encoder','exec'),ns)
 extract=ns['extract_encoder']
 if simpaths[0].exists() or simpaths[1].exists():
  if simpaths[0].exists():
   assert sha(simpaths[0])==provenance['source_sha256'];state=extract(torch.load(simpaths[0],map_location='cpu',weights_only=True));path=OUT/'evidence/verified_simclr_encoder.pt';torch.save(state,path)
  else:
   path=simpaths[1];assert sha(path)==provenance['output_sha256'];state=torch.load(path,map_location='cpu',weights_only=True)
  actual=module.make_backbone(backbone_weights=path).eval()
  for k,v in actual.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
  with torch.inference_mode():g=actual(batch);s=module.spatial_features(actual,batch)
  results['SimCLR'].update(status='VERIFIED_MATCHES_RECORDED_SHA256',load_info=actual.backbone_load_info,missing_keys=[],unexpected_keys=[],global_stats=stats(g),spatial_stats=stats(s))
 # Synthetic round-trip tests are NOT ImageNet or SimCLR initialization evidence.
 fixture=OUT/'evidence/synthetic_random_encoder.pt';torch.save(model.state_dict(),fixture)
 def roundtrip():
  loaded=module.make_backbone(backbone_weights=fixture)
  assert fingerprint(loaded)==before
  return loaded.backbone_load_info
 check('synthetic_local_loader_roundtrip_NOT_pretrained',roundtrip)
 def negative(name,state):
  try:
   with patch('torch.load',return_value=state):module.load_backbone_weights(model,fixture)
  except ValueError:return 'rejected as required'
  raise AssertionError(name+' wrongly accepted')
 source=model.state_dict()
 bad=dict(source);bad.pop('conv1.weight');check('reject_missing_encoder_key',lambda:negative('missing',bad))
 extra=dict(source);extra['unknown.weight']=torch.zeros(1);check('reject_unexpected_key',lambda:negative('extra',extra))
 wrong=dict(source);wrong['conv1.weight']=torch.zeros(1);check('reject_wrong_shape',lambda:negative('shape',wrong))
 head=dict(source);head.update({'fc.weight':torch.zeros(3,512),'fc.bias':torch.zeros(3)})
 def headcheck():
  with patch('torch.load',return_value=head),warnings.catch_warnings(record=True):info=module.load_backbone_weights(model,fixture)
  assert info['ignored_keys']==['fc.bias','fc.weight'];return info
 check('explicit_classifier_head_exclusion_synthetic',headcheck)
 def convertcheck():
  stages={'conv1':'0','bn1':'1','layer1':'4','layer2':'5','layer3':'6','layer4':'7'}
  sequential={'backbone.'+stages[k.split('.')[0]]+'.'+k.split('.',1)[1]:v for k,v in source.items()}
  sequential['projection_head.0.weight']=torch.zeros(7,512)
  converted=extract({'state_dict':sequential})
  assert converted.keys()==source.keys()
  for k in source:torch.testing.assert_close(source[k],converted[k],rtol=0,atol=0)
  del sequential['backbone.0.weight']
  try:extract({'state_dict':sequential})
  except ValueError:return '120 encoder entries round-trip; synthetic projection ignored; incomplete encoder rejected'
  raise AssertionError('missing encoder key accepted')
 check('simclr_key_translation_SYNTHETIC_ONLY',convertcheck)
 save('initializations.json',results);save('evidence/tests.json',tests)
 assert all(t['status']=='PASS' for t in tests)
 assert all(sha(ROOT/p)==h for p,h in protected.items())
 save('evidence/preservation.json',{'files_checked':len(protected),'changed_files':[],'training_executed':False,'downloads_performed':False})
 save('environment.json',{'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'device':'CPU','mode':'eval + inference_mode'})
 print('FINISHED',len(tests),'checks;',[(k,v['status']) for k,v in results.items()],flush=True)
if __name__=='__main__':main()
