"""Fixed-budget local baseline trainer with atomic epoch checkpoints and resume."""
import argparse,hashlib,json,os,random,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from warprec.common import initialize_datasets,standard_optimizer
from warprec.data.reader import ReaderFactory
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import load_train_configuration
from warprec.utils.registry import model_registry


def atomic_json(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=str));tmp.replace(path)


def mapping_digest(dataset):
 return hashlib.sha256(json.dumps(dataset.get_inverse_mappings(),sort_keys=True,default=str).encode()).hexdigest()


def make_loader(model,dataset,params,epoch,generator):
 if model.__class__.__name__=='gSASRec':
  from semantic_training import CompactHistoryDataset,HistoryCollator
  from inference_utils import get_train_histories
  samples=CompactHistoryDataset(get_train_histories(dataset.train_session),model.max_seq_len,model.n_items)
  return torch.utils.data.DataLoader(samples,batch_size=params['batch_size'],shuffle=True,generator=generator,collate_fn=HistoryCollator(model.n_items,model.neg_samples),num_workers=0)
 if model.__class__.__name__=='BERT4Rec':
  # Refresh cloze masks/negatives each epoch; never use held-out interactions.
  key=f"cloze_len_{model.max_seq_len}_prob_{model.mask_prob}_neg_{model.neg_samples}"
  dataset.train_session._cached_dataset.pop(key,None)
  loader=dataset.train_session.get_cloze_mask_dataloader(max_seq_len=model.max_seq_len,mask_prob=model.mask_prob,
   mask_token_id=model.mask_token_id,neg_samples=model.neg_samples,batch_size=params['batch_size'],seed=42+epoch,low_memory=False)
 else:loader=model.get_dataloader(interactions=dataset.train_set,sessions=dataset.train_session,low_memory=False)
 if model.__class__.__name__=='DuoRec':
  from semantic_training import IndexDataset,SemanticCollator
  return torch.utils.data.DataLoader(IndexDataset(loader.dataset),batch_size=params['batch_size'],shuffle=True,generator=generator,collate_fn=SemanticCollator(loader.dataset),num_workers=0)
 return torch.utils.data.DataLoader(loader.dataset,batch_size=params['batch_size'],shuffle=True,generator=generator,collate_fn=loader.collate_fn,num_workers=0)


def train(config,dataset,name,out,smoke=False,stop_after=300):
 out.mkdir(parents=True,exist_ok=True)
 if smoke and (out/'SMOKE_SUCCESS').exists():return
 params={k:(v[-1] if isinstance(v,list) else v) for k,v in dict(config.models[name]).items() if k not in ('meta','optimization','early_stopping','lr_scheduler')}
 params['corrected_training']=True
 target_epochs=int(params['epochs']);assert target_epochs>0
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 model=model_registry.get_class(name)(params=params,interactions=dataset.train_set,info=dataset.info(),seed=42).to(config.general.device)
 if name in ['BERT4Rec','gSASRec','CL4SRec','DuoRec']:assert model.corrected_training
 print('CLASS',name,type(model).__module__,'corrected',getattr(model,'corrected_training',False),flush=True)
 device=config.general.device;optimizer=standard_optimizer(model);generator=torch.Generator().manual_seed(42)
 digest=mapping_digest(dataset);start_epoch=0;history=[]
 # Initialize cached eager samples before restoring RNG state.
 loader=make_loader(model,dataset,params,0,generator)
 if not len(loader):raise ValueError('No training batches')
 checkpoint=out/'checkpoint.pt'
 if checkpoint.exists():
  state=torch.load(checkpoint,map_location='cpu',weights_only=False)
  assert state['full_epoch'] or smoke, 'Cannot resume a smoke checkpoint as full training'
  assert state['params']==params and state['mapping_digest']==digest and state['model_name']==name
  model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
  random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
  torch.cuda.set_rng_state_all(state['cuda_rng']);generator.set_state(state['loader_rng'])
  start_epoch=state['epoch'];history=state['history']
 atomic_json(out/'params.json',params)
 atomic_json(out/'training_metadata.json',dict(model=name,target_epochs=target_epochs,seed=42,mapping_digest=digest,
  model_class=type(model).__module__+'.'+type(model).__name__,samples_per_epoch=len(loader.dataset),batches_per_epoch=len(loader),learning_rate=params['learning_rate'],
  initialization='from scratch unless resuming this same output',gradient_clip_norm=1.0,
  bert_masks='refreshed each epoch',other_negative_samples='gSASRec: dynamic prefix-filtered negatives; other models: framework eager samples',
  checkpoint_selection=f'final epoch {target_epochs}; no test evaluation during training',smoke=smoke))
 print(f'TRAIN {name} samples={len(loader.dataset)} batches={len(loader)} resume_epoch={start_epoch}',flush=True)
 end_epoch=1 if smoke else min(target_epochs,stop_after)
 for epoch in range(start_epoch,end_epoch):
  if epoch and name=='BERT4Rec':loader=make_loader(model,dataset,params,epoch,generator)
  model.train();beg=time.monotonic();total=0.;count=0;steps=0
  for step,batch in enumerate(loader,1):
   batch=[x.to(device) for x in batch];optimizer.zero_grad(set_to_none=True)
   before=next(model.parameters()).detach().clone() if step==1 else None
   loss=model.train_step(batch,epoch)
   if not torch.isfinite(loss):raise ValueError(f'Nonfinite loss {name} epoch={epoch+1} step={step}')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
   if step==1 and not norm>0:raise ValueError('Zero gradients on first batch')
   optimizer.step()
   if step==1 and torch.equal(before,next(model.parameters()).detach()):raise ValueError('Weights did not update')
   size=len(batch[0]);total+=loss.item()*size;count+=size;steps+=1
   if step==1 or step%250==0:print(f'{name} epoch={epoch+1}/{target_epochs} batch={step}/{len(loader)} loss={total/count:.6f} sec={time.monotonic()-beg:.1f}',flush=True)
   if smoke and step>=2:break
  if not smoke:assert count==len(loader.dataset) and steps==len(loader)
  if any(not torch.isfinite(v).all() for v in model.parameters()):raise ValueError('Nonfinite parameters')
  history.append(dict(epoch=epoch+1,loss=total/count,samples=count,batches=steps,seconds=time.monotonic()-beg))
  state=dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),params=params,info=dataset.info(),
   model_name=name,mapping_digest=digest,epoch=epoch+1,history=history,full_epoch=not smoke,
   python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=generator.get_state())
  tmp=out/'checkpoint.tmp';torch.save(state,tmp);tmp.replace(checkpoint);atomic_json(out/'training_log.json',history)
  print(f'EPOCH_COMPLETE {name} {epoch+1}/{target_epochs}',flush=True)
 if smoke:
  state=torch.load(checkpoint,map_location='cpu',weights_only=False)
  model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
  model.eval();print('SMOKE_PASS',name,flush=True);(out/'SMOKE_SUCCESS').write_text('Up to 2 optimizer steps and checkpoint reload verified\n')
 elif end_epoch==target_epochs:
  assert history[-1]['epoch']==target_epochs and len(history)==target_epochs
  (out/'SUCCESS').write_text(f'{target_epochs} complete epochs; final checkpoint.pt\n')
 del model,optimizer,loader;torch.cuda.empty_cache()
 # Release eager sample tensors between models; cached user histories remain available.
 dataset.train_session._cached_dataset.clear()


def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--model');p.add_argument('--output-dir',required=True);p.add_argument('--preflight-all',action='store_true');p.add_argument('--stop-after-epoch',type=int,default=300);a=p.parse_args()
 cfg=load_train_configuration(a.config);ds,_,_=initialize_datasets(reader=ReaderFactory.get_reader(config=cfg),callback=WarpRecCallback(),config=cfg)
 for n in (list(cfg.models) if a.preflight_all else [a.model]):train(cfg,ds,n,Path(a.output_dir)/n if a.preflight_all else Path(a.output_dir),a.preflight_all,a.stop_after_epoch)
if __name__=='__main__':main()
