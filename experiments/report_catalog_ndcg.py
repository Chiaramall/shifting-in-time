"""Compute nDCG@10 at 5/30% from saved catalog perturbation rankings.
No model inference/training. Uses the original test split and fixed hit@20 cohorts.
"""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
from inference_utils import get_train_histories,ranking_metrics
from frbo import finite_rank_biased_overlap
from warprec.common import initialize_datasets
from warprec.data.reader import ReaderFactory
from warprec.utils.callback import WarpRecCallback
from warprec.utils.config import load_train_configuration

MODELS=['BERT4Rec','BSARec','DuoRec','CL4SRec','GRU4Rec','LightSANs','SASRec','eSASRec','gSASRec']
DATASETS=['lastfm1k','movielens10m']
DESTINATIONS=['recent','middle','old']

def rows(p):
 with p.open() as f:yield from csv.DictReader(f)

def write_csv(path,values):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)

def render(out,aggregate,baselines):
 lookup={(r['dataset'],r['model'],r['destination'],r['intensity']):r for r in aggregate}
 base={(r['dataset'],r['model']):r for r in baselines}
 for metric in ['ndcg10','frbo10']:
  title='nDCG@10' if metric=='ndcg10' else 'FRBO@10'
  lines=[f'# {title} — nuovi item dal catalogo, 5% e 30%', '',
         'Coorti fisse specifiche del modello: almeno un hit nella top-20 originale. Originale e perturbato valutati sugli stessi utenti e positivi test. nDCG@10 con rilevanza binaria e IDCG basato su min(10, numero di positivi test distinti). FRBO@10 p=0.9. Nessuna nuova inferenza.', '',
         '| Dataset | Modello | Utenti | nDCG@10 originale | Recent 5% | Recent 30% | Middle 5% | Middle 30% | Old 5% | Old 30% |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
  for ds in DATASETS:
   for model in MODELS:
    b=base[ds,model]
    vals=[lookup[ds,model,d,a][metric] for d in DESTINATIONS for a in [.05,.3]]
    lines.append('| '+' | '.join([ds,model,str(b['n_selected_users']),f'{b["original_ndcg10"]:.6f}']+[f'{v:.6f}' for v in vals])+' |')
  (out/f'{metric}.md').write_text('\n'.join(lines)+'\n')
  caption=(r'Recommendation effectiveness (nDCG@10) under catalog perturbations at 5\% and 30\% intensity. '
           if metric=='ndcg10' else r'Original recommendation effectiveness (nDCG@10) and ranking stability (FRBO@10, $p=0.9$) under catalog perturbations at 5\% and 30\% intensity. ')
  caption+=r'Each model is evaluated on its fixed cohort of users with at least one relevant item in the original top-20. New items are selected by maximum distance from the training-profile centroid and replace equally many history occurrences before insertion. Higher values are better.'
  tex=[r'\begin{table*}[!t]',r'\centering',r'\caption{'+caption+'}',r'\label{tab:catalog_'+metric+'}',r'\scriptsize',r'\setlength{\tabcolsep}{2.5pt}',r'\renewcommand{\arraystretch}{1.05}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{@{}l|c|cc|cc|cc|c|cc|cc|cc@{}}',r'\toprule',r'& \multicolumn{7}{c|}{\textbf{LastFM}} & \multicolumn{7}{c}{\textbf{ML-10M}} \\',r'\cmidrule(lr){2-8}\cmidrule(lr){9-15}',r'\textbf{Model} & \textbf{Original} & \multicolumn{2}{c|}{\textbf{Recent}} & \multicolumn{2}{c|}{\textbf{Middle}} & \multicolumn{2}{c|}{\textbf{Old}} & \textbf{Original} & \multicolumn{2}{c|}{\textbf{Recent}} & \multicolumn{2}{c|}{\textbf{Middle}} & \multicolumn{2}{c}{\textbf{Old}} \\',r'& \textbf{nDCG@10} & \textbf{5\%} & \textbf{30\%} & \textbf{5\%} & \textbf{30\%} & \textbf{5\%} & \textbf{30\%} & \textbf{nDCG@10} & \textbf{5\%} & \textbf{30\%} & \textbf{5\%} & \textbf{30\%} & \textbf{5\%} & \textbf{30\%} \\',r'\midrule']
  for model in MODELS:
   values=[]
   for ds in DATASETS:
    values.append(base[ds,model]['original_ndcg10'])
    values += [lookup[ds,model,d,a][metric] for d in DESTINATIONS for a in [.05,.3]]
   tex.append(model+' & '+' & '.join(f'{v:.4f}' for v in values)+r' \\')
  tex += [r'\bottomrule',r'\end{tabular}%',r'}',r'\end{table*}']
  (out/f'main_results_{metric}.tex').write_text('\n'.join(tex)+'\n')
 (out/'preview.tex').write_text(r'''\documentclass[10pt]{article}
\usepackage[a4paper,landscape,margin=14mm]{geometry}
\usepackage{booktabs,graphicx,array}
\begin{document}
\input{main_results_ndcg10.tex}
\clearpage
\input{main_results_frbo10.tex}
\end{document}
''')

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',default='results/catalog');p.add_argument('--output-dir',default='results/catalog/ndcg_05_30');a=p.parse_args()
 source=Path(a.source);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 aggregate=[];baselines=[]
 for ds in DATASETS:
  cfg=load_train_configuration(f'config/{ds}_eval.yml')
  dataset,_,_=initialize_datasets(reader=ReaderFactory.get_reader(config=cfg),callback=WarpRecCallback(),config=cfg)
  histories=get_train_histories(dataset.train_session);ev=dataset.eval_set.get_sparse().tocsr();um,im=dataset.get_inverse_mappings()
  positives={u:set(ev.indices[ev.indptr[u]:ev.indptr[u+1]]) for u in histories}
  for model in MODELS:
   folder=source/ds/model;assert (folder/'SUCCESS').exists()
   cohort={int(r['user_idx']):r for r in rows(folder/'cohort.csv')}
   for u,r in cohort.items():assert str(um[u])==r['user_id']
   # Check saved catalog IDs against this dataset's mappings.
   for r in rows(folder/'catalog_selection.csv'):assert str(im[int(r['item_idx'])])==r['item_id']
   selected={u for u,r in cohort.items() if r['selected']=='True'}
   clean={u:json.loads(r['original_top_items']) for u,r in cohort.items()}
   original={u:ranking_metrics(r,positives[u],10)[0] for u,r in clean.items()}
   b=dict(dataset=ds,model=model,n_selected_users=len(selected),n_eligible_users=len(cohort),original_ndcg10=math.fsum(original[u] for u in selected)/len(selected),original_ndcg10_all_eligible=math.fsum(original.values())/len(original))
   baselines.append(b)
   dest=out/ds/model;dest.mkdir(parents=True,exist_ok=True)
   write_csv(dest/'baseline.csv',[dict(user_idx=u,user_id=um[u],selected=u in selected,n_test_positives=len(positives[u]),original_ndcg10=original[u]) for u in cohort])
   for position in DESTINATIONS:
    for pct,intensity in [('05',.05),('30',.3)]:
     vals=[];frbos=[];seen=set();injected_relevant=[]
     with (dest/f'{position}_{pct}.csv').open('w') as f:
      fields=['user_idx','user_id','original_ndcg10','perturbed_ndcg10','delta_ndcg10','frbo10','injected_test_positives']
      w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
      for r in rows(folder/f'{position}_{pct}.csv'):
       u=int(r['user_idx']);assert u in selected and u not in seen;seen.add(u)
       before=json.loads(r['original_top_items']);after=json.loads(r['perturbed_top_items'])
       assert before==clean[u] and len(before)==len(after)==10 and len(set(after))==10
       if position=='recent' and pct=='05':
        seq=json.loads(r['original_sequence']);assert list(histories[u][-len(seq):])==seq
       ndcg=ranking_metrics(after,positives[u],10)[0]
       frbo=finite_rank_biased_overlap(before,after,.9);assert abs(frbo-float(r['frbo10']))<1e-12
       new=json.loads(r['new_items']);assert not set(new)&set(histories[u])
       nt=len(set(new)&positives[u]);injected_relevant.append(nt)
       vals.append(ndcg);frbos.append(frbo)
       w.writerow(dict(user_idx=u,user_id=um[u],original_ndcg10=original[u],perturbed_ndcg10=ndcg,delta_ndcg10=ndcg-original[u],frbo10=frbo,injected_test_positives=nt))
     assert seen==selected
     aggregate.append(dict(dataset=ds,model=model,destination=position,intensity=intensity,n_users=len(seen),original_ndcg10=b['original_ndcg10'],ndcg10=math.fsum(vals)/len(vals),delta_ndcg10=math.fsum(vals)/len(vals)-b['original_ndcg10'],frbo10=math.fsum(frbos)/len(frbos),users_with_injected_test_positive=sum(n>0 for n in injected_relevant)))
   write_csv(out/'aggregate.csv',aggregate);write_csv(out/'baselines.csv',baselines)
   print('COMPLETE',ds,model,'users',len(selected),'original_ndcg10',b['original_ndcg10'],flush=True)
  del dataset,histories,ev,positives
 render(out,aggregate,baselines)
 (out/'metadata.json').write_text(json.dumps(dict(source=str(source),datasets=DATASETS,models=MODELS,top_k=10,intensities=[.05,.3],relevance='binary membership in original held-out test positives; no rating threshold',cohort='fixed per-model original hit@20',aggregation='macro-average across selected users',normalization='IDCG sums the first min(10, number of distinct test positives) discounts',mask='original full training history, unchanged; injected items can be recommended and can be test positives',inference=False),indent=2))
 (out/'SUCCESS').write_text('18 models x 6 conditions: nDCG and FRBO recomputed and fixed cohorts verified\n')
if __name__=='__main__':main()
