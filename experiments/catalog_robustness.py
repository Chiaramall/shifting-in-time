"""Centroid-farthest catalog injection, preserving the historical relocation protocol."""
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import numpy as np
from frbo import finite_rank_biased_overlap


def inject_catalog(sequence, ascending_positions, catalog_ranked, intensity, destination):
    n=len(sequence)
    if sorted(ascending_positions)!=list(range(n)) or n<2 or not 0<intensity<1:
        raise ValueError('Invalid sequence, ranking or intensity')
    count=min(n-1,max(1,math.floor(n*intensity+.5)))
    new=list(catalog_ranked[:count])
    if len(new)!=count or len(set(new))!=count or set(new)&set(sequence):
        raise ValueError('Insufficient distinct unseen catalog items')
    removed=sorted(ascending_positions[-count:])
    remaining=[v for i,v in enumerate(sequence) if i not in set(removed)]
    if destination not in ('old','middle','recent'):raise ValueError(destination)
    start={'old':0,'middle':len(remaining)//2,'recent':len(remaining)}[destination]
    result=remaining[:start]+new+remaining[start:]
    assert len(result)==n
    return result,removed,list(range(start,start+count)),new


def farthest_catalog(embeddings,histories,count,device='cpu'):
    """Exact full-catalog Euclidean search; exclude full training profile.
    Rank descending distance, ties ascending internal item ID. Float64 arithmetic.
    """
    import torch
    e=torch.as_tensor(embeddings,dtype=torch.float64,device=device)
    centers=torch.stack([e[sorted(set(h))].mean(0) for h in histories])
    d2=(centers.square().sum(1)[:,None]+e.square().sum(1)[None,:]-2*centers@e.T).clamp_min(0)
    for j,h in enumerate(histories):d2[j,sorted(set(h))]=-torch.inf
    order=torch.argsort(d2,dim=1,descending=True,stable=True)[:,:count]
    values=torch.gather(d2,1,order)
    if not torch.isfinite(values).all():raise ValueError('Not enough unseen catalog items')
    return order.cpu().tolist(),values.sqrt().cpu().tolist()


def run(args):
    import torch
    from inference_utils import batched,choose_users,get_train_histories,load_model,recommend_batch,resolve_checkpoint
    from embedding_utils import extract_embeddings,distance_matrix
    from warprec.common import initialize_datasets
    from warprec.data.reader import ReaderFactory
    from warprec.utils.callback import WarpRecCallback
    from warprec.utils.config import load_train_configuration
    torch.set_num_threads(8)
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=True)
    config=load_train_configuration(args.config);names=args.models or list(config.models)
    meta=dict(vars(args),models=names,top_k=10,p=.9,protocol='catalog_farthest_relocation_v1',
        centroid='mean of distinct full-training-history input embeddings',
        candidates='entire model item catalog excluding full training history; no test filtering',
        removal='same as previous relocation: farthest visible occurrences from original centroid',
        insertion='replace removed block by unseen catalog items; insert old/middle/recent',
        block_order='decreasing distance, ties ascending internal ID; most distant first',
        masking='original full training history fixed in every condition; injected items remain eligible',
        cohort='at least one original hit@20; all eligible test users unless max-users specified',
        intensities=[.05,.1,.2,.3],metric='FRBO@10 finite normalized, p=0.9')
    mp=output/'metadata.json'
    if mp.exists() and json.loads(mp.read_text())!=meta:raise ValueError('Resume parameters differ')
    mp.write_text(json.dumps(meta,indent=2))
    dataset,_,_=initialize_datasets(reader=ReaderFactory.get_reader(config=config),callback=WarpRecCallback(),config=config)
    histories=get_train_histories(dataset.train_session);ev=dataset.eval_set.get_sparse().tocsr()
    users=choose_users(histories,ev,2,args.max_users,42,True)
    um,im=dataset.get_inverse_mappings();positives={u:set(ev.indices[ev.indptr[u]:ev.indptr[u+1]]) for u in users}
    device=torch.device(args.device)
    for name in names:
        folder=output/name;folder.mkdir(exist_ok=True)
        if (folder/'SUCCESS').exists():continue
        checkpoint=resolve_checkpoint(config.models[name]['meta']['load_from'])
        if not checkpoint or not checkpoint.exists() or not (checkpoint.parent/'SUCCESS').exists():
            (folder/'BLOCKED.json').write_text(json.dumps(dict(reason='Configured checkpoint missing or training incomplete',checkpoint=str(checkpoint)),indent=2));continue
        (folder/'BLOCKED.json').unlink(missing_ok=True)
        try:
            print('START',name,flush=True)
            model=load_model(name,dict(config.models[name]),dataset,device,strict=True)
            checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            fingerprint=dict(path=str(checkpoint),sha256=checkpoint_hash)
            prior=folder/'checkpoint.json'
            if prior.exists() and json.loads(prior.read_text())!=fingerprint:
                raise ValueError('Checkpoint changed since previous evaluation; use a new output directory')
            prior.write_text(json.dumps(fingerprint,indent=2))
            seqs={u:list(map(int,histories[u][-model.max_seq_len:])) for u in users}
            def recommend(batch,seq):
                return recommend_batch(model,batch,seq,[histories[u] for u in batch],positives,20,device,compute_metrics=False)
            original={};cohort=[]
            for batch in batched(users,args.batch_size):
                for u,r in zip(batch,recommend(batch,[seqs[u] for u in batch])):
                    hit=len(set(r.top_items)&positives[u]);original[u]=r.top_items[:10]
                    cohort.append(dict(user_idx=u,user_id=um[u],selected=bool(hit),hits_at20=hit,original_top_items=json.dumps(r.top_items[:10])))
            with (folder/'cohort.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(cohort[0]));w.writeheader();w.writerows(cohort)
            chosen=[r['user_idx'] for r in cohort if r['selected']]
            print(name,'COHORT',len(chosen),'/',len(users),flush=True)
            embeddings=extract_embeddings(model,list(range(model.n_items)),device)
            ranks={};catalog={};distances={}
            for batch in batched(chosen,args.batch_size):
                cs,ds=farthest_catalog(embeddings,[histories[u] for u in batch],max(1,math.floor(model.max_seq_len*.3+.5)),device)
                for u,items,dd in zip(batch,cs,ds):
                    center=embeddings[sorted(set(histories[u]))].mean(0)
                    d=distance_matrix(embeddings[seqs[u]],center[None,:],'euclidean')[:,0]
                    ranks[u]=sorted(range(len(seqs[u])),key=lambda i:(d[i],seqs[u][i],i))
                    catalog[u]=items;distances[u]=dd
                    assert not set(items)&set(histories[u])
            with (folder/'catalog_selection.csv').open('w') as f:
                w=csv.writer(f);w.writerow(['user_idx','user_id','rank','item_idx','item_id','distance'])
                for u in chosen:
                    for j,(i,d) in enumerate(zip(catalog[u],distances[u]),1):w.writerow([u,um[u],j,i,im[i],d])
            aggregates=[]
            for destination in ['old','middle','recent']:
                for intensity in [.05,.1,.2,.3]:
                    path=folder/f'{destination}_{int(round(intensity*100)):02d}.csv'
                    if path.exists():
                        with path.open() as f:records=list(csv.DictReader(f))
                        assert {int(r['user_idx']) for r in records}==set(chosen) and len(records)==len(chosen)
                        values=[float(r['frbo10']) for r in records]
                    else:
                        values=[];tmp=path.with_suffix('.tmp')
                        with tmp.open('w') as f:
                            fields=['model','user_idx','user_id','destination','intensity','selected_count','history_len','original_sequence','perturbed_sequence','removed_positions','inserted_positions','new_items','original_top_items','perturbed_top_items','frbo10']
                            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                            for batch in batched(chosen,args.batch_size):
                                changed=[inject_catalog(seqs[u],ranks[u],catalog[u],intensity,destination) for u in batch]
                                results=recommend(batch,[v[0] for v in changed])
                                for u,r,(s,removed,inserted,new) in zip(batch,results,changed):
                                    assert not set(new)&set(histories[u])
                                    frbo=finite_rank_biased_overlap(original[u],r.top_items[:10],.9);values.append(frbo)
                                    row=dict(model=name,user_idx=u,user_id=um[u],destination=destination,intensity=intensity,selected_count=len(new),history_len=len(histories[u]),original_sequence=seqs[u],perturbed_sequence=s,removed_positions=removed,inserted_positions=inserted,new_items=new,original_top_items=original[u],perturbed_top_items=r.top_items[:10],frbo10=frbo)
                                    w.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
                        tmp.replace(path)
                    aggregates.append(dict(model=name,destination=destination,intensity=intensity,n_users=len(values),frbo10=float(np.mean(values)) if values else None))
                    (folder/'aggregate.json').write_text(json.dumps(aggregates,indent=2))
                    print(name,destination,intensity,'FRBO',aggregates[-1]['frbo10'],flush=True)
            (folder/'SUCCESS').write_text('All 12 conditions complete; FRBO@10 p=.9\n')
            (folder/'ERROR.json').unlink(missing_ok=True)
            del model;torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            (folder/'ERROR.json').write_text(json.dumps(dict(error=str(e),traceback=traceback.format_exc()),indent=2));traceback.print_exc()
    if all((output/n/'SUCCESS').exists() for n in names):(output/'SUCCESS').write_text('All requested models completed\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--models',nargs='+');p.add_argument('--max-users',type=int);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--device',default='cuda');run(p.parse_args())
