"""Verify supplied local weights using safe tensor loading and CPU inference."""
import ast,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
import torchvision
from torchvision.models import resnet18
from pattern_recognition_coursework.models.backbones.resnet import make_backbone,spatial_features
from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset,get_cub_transforms
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def fingerprint(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.cpu().numpy().tobytes())
 return h.hexdigest()
def stats(t):
 return {'shape':list(t.shape),'dtype':str(t.dtype),'finite':bool(torch.isfinite(t).all()),'min':float(t.min()),'max':float(t.max()),'mean':float(t.mean()),'std':float(t.std())}
def main():
 if (OUT/'results.json').exists():raise SystemExit('Refusing to overwrite existing verification')
 torch.set_num_threads(2)
 paths={'ImageNet':ROOT/'resnet18-f37072fd.pth','SimCLR':ROOT/'checkpoint_100-resnet18-simclr-imagenet1k.ckpt'}
 hashes={k:sha(p) for k,p in paths.items()}
 expected_im='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'
 provenance=json.loads((ROOT/'reports/records/simclr_weight_provenance.json').read_text())
 assert hashes['ImageNet']==expected_im,'ImageNet SHA256 mismatch'
 assert hashes['SimCLR']==provenance['source_sha256'],'SimCLR source SHA256 mismatch'
 im=torch.load(paths['ImageNet'],map_location='cpu',weights_only=True)
 ckpt=torch.load(paths['SimCLR'],map_location='cpu',weights_only=True)
 tree=ast.parse((ROOT/'scripts/convert_simclr_resnet18.py').read_text())
 ns={'torch':torch,'nn':nn,'resnet18':resnet18,'STAGES':{'0':'conv1','1':'bn1','4':'layer1','5':'layer2','6':'layer3','7':'layer4'}}
 exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='extract_encoder'],type_ignores=[]),'original_encoder_extractor','exec'),ns)
 encoder=ns['extract_encoder'](ckpt)
 exported=OUT/'simclr_resnet18_backbone.pt'
 if exported.exists():
  cached=torch.load(exported,map_location='cpu',weights_only=True)
  assert cached.keys()==encoder.keys()
  for k in encoder:torch.testing.assert_close(cached[k],encoder[k],rtol=0,atol=0)
 else:torch.save(encoder,exported)
 source_state=ckpt['state_dict']
 projection={k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in source_state.items() if not k.startswith('backbone.') and isinstance(v,torch.Tensor)}
 data=ROOT/'cub2002011/versions/7/CUB_200_2011'
 ds=CUBFewShotDataset(data,ROOT/'data/splits/cub_clean_v1/val_images.txt',get_cub_transforms(224,False),require_clean_index=True)
 selected=[r for r in json.loads((ROOT/'outputs/stage01_data/sample_manifest.json').read_text()) if r['dataset']=='cub_val']
 x=torch.stack([ds[r['sample_index']][0] for r in selected])
 result={'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'device':'CPU','training':False,'downloads':False,'input':stats(x),'images':[r['path'] for r in selected],'source_hashes':hashes,'SimCLR_checkpoint':{'epoch':ckpt.get('epoch'),'global_step':ckpt.get('global_step'),'encoder_state_entries':len(encoder),'non_encoder_tensors':projection,'original_top_level_keys':list(ckpt),'export_path':str(exported.relative_to(ROOT)),'export_sha256':sha(exported),'historical_export_sha256':provenance['output_sha256'],'note':'torch.save serialization digest may differ from historical export; exact tensors checked against source'} ,'initializations':{}}
 global_outputs={}
 cache=OUT/'local_torch_cache/checkpoints';cache.mkdir(parents=True,exist_ok=True)
 link=cache/paths['ImageNet'].name
 if not link.exists():link.symlink_to(paths['ImageNet'])
 torch.hub.set_dir(str(cache.parent))
 try:make_backbone(backbone_weights=paths['ImageNet'])
 except ValueError as e:result['ImageNet_direct_local_loader_error']=str(e)
 with patch('torch.hub.download_url_to_file',side_effect=AssertionError('No downloads allowed')):
  for name,path in [('Random',None),('ImageNet',paths['ImageNet']),('SimCLR',exported)]:
   torch.manual_seed(42)
   model=(make_backbone(pretrained=True) if name=='ImageNet' else make_backbone(backbone_weights=path)).eval()
   expected=im if name=='ImageNet' else encoder if name=='SimCLR' else None
   if expected is not None:
    assert all(k in expected or k.endswith('.num_batches_tracked') for k in model.state_dict())
    for k,v in model.state_dict().items():
     if k in expected:torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
     else:assert k.endswith('.num_batches_tracked') and int(v)==0
   before=fingerprint(model)
   with torch.inference_mode():
    global_=model(x);local=spatial_features(model,x)
   assert local.shape==(4,512,7,7) and global_.shape==(4,512)
   torch.testing.assert_close(local.mean((-1,-2)),global_)
   assert torch.isfinite(global_).all() and torch.isfinite(local).all()
   assert float(global_.std())>1e-6 and float((global_[0]-global_[1]).abs().max())>1e-6
   assert fingerprint(model)==before
   result['initializations'][name]={'status':'PASS','parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'load_info':model.backbone_load_info,'missing_encoder_keys':[],'unexpected_encoder_keys':[],'state_fingerprint':before,'global_features':stats(global_),'spatial_features':stats(local),'different_images_max_abs_diff':float((global_[0]-global_[1]).abs().max()),'eval_state_unchanged':True,'compatibility_initialized_buffers':[k for k in model.state_dict() if expected is not None and k not in expected]}
   global_outputs[name]=global_.clone()
   print(name,'PASS',result['initializations'][name]['global_features'],flush=True)
 result['pairwise_max_abs_feature_difference']={f'{a}_vs_{b}':float((global_outputs[a]-global_outputs[b]).abs().max()) for a,b in [('Random','ImageNet'),('Random','SimCLR'),('ImageNet','SimCLR')]}
 assert len({r['state_fingerprint'] for r in result['initializations'].values()})==3
 assert all(v>1e-6 for v in result['pairwise_max_abs_feature_difference'].values())
 assert hashes=={k:sha(p) for k,p in paths.items()}
 result['supplied_weight_files_unchanged']=True
 (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('All three initializations verified; source files unchanged; no training/downloads.')
if __name__=='__main__':main()
