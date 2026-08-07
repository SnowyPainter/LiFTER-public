#!/opt/conda/bin/python
"""Common-ratio explanation comparison on frozen/trainable native CTDG models."""
from __future__ import annotations
import argparse, json, math, random, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(ROOT))
from evaluation.evaluate import (CTDGHistoryIndex, LiFTERHistoryIndex, RunSpec, binary_average_precision,
    build_benchmark_model, build_lifter_causal_fact_context, evaluate_run, read_csv_tail,
    sample_historical_destinations, split_indices)
from models.explainability import TGNNExplainer, TempME

DATASET='wikipedia'; RATIOS=(.05,.10,.15,.20,.25,.30); FIXED_K=(1,2,3,5,10)
HISTORY_PER_ENDPOINT=10

def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def train_checkpoint(model_name, seed, model_config, device):
 root=HERE/'checkpoints'/f'{model_name}_seed{seed}'; path=root/f'{DATASET}_{model_name}_seed{seed}.pt'
 if model_name=='TGN':
  main_path=ROOT/f'experiments/main_benchmark/checkpoints/{DATASET}_TGN_seed{seed}.pt'
  if main_path.exists(): return main_path
 if path.exists():
  saved=torch.load(path,map_location='cpu',weights_only=False)
  if model_name!='TGN' or any(key.endswith('memory_updater.weight_ih_l0') for key in saved['model']): return path
 config=yaml.safe_load((ROOT/'experiments/main_benchmark/config_lifter.yaml').read_text()); config.update(model=model_name,dataset=DATASET,seed=seed,model_config=model_config)
 config['training'].update(epochs=10,max_examples=32768,history_len=20,batch_size=512,historical_negative_strategy='uniform',progress=False)
 config['paths']['checkpoints']=str(root); config['paths'].pop('load_checkpoint',None); config['explanation_evaluation']['max_queries']=0; config['lifter_diagnostics']['enabled']=False
 spec=RunSpec('ctdg',DATASET,ROOT/f'data/processed/ctdg/{DATASET}/events.csv',model_name,'base',seed,'reference',model_config,False)
 evaluate_run(spec,config,device); return path

def make_index_and_batches(frame, rows, destinations, history_len, lifter=False, fact_context_len=8, device=None, index=None, raw_features=None):
 n=int(max(frame.src.max(),frame.dst.max()))+1
 if index is not None:
  pass
 elif lifter:
  context=build_lifter_causal_fact_context(frame,fact_context_len)
  raw_features=np.empty((len(frame),0),np.float32) if raw_features is None else raw_features
  index=LiFTERHistoryIndex(frame,history_len,n,context,raw_features)
 else: index=CTDGHistoryIndex(frame,history_len,n)
 sources=frame.iloc[rows].src.to_numpy(np.int64); timestamps=frame.iloc[rows].timestamp.to_numpy(np.float32)
 src=index.gather(sources,rows); dst=index.gather(destinations,rows)
 if lifter: sn,st,sf,sm=src; dn,dt,df,dm=dst
 else:
  sn,st,sm=src; dn,dt,dm=dst; sf=np.zeros((*sn.shape,1),np.float32); df=np.zeros((*dn.shape,1),np.float32)
 T=lambda x,d=None:torch.tensor(x,dtype=d,device=device)
 return (T(sources),T(destinations),T(timestamps,torch.float32),T(sn),T(st,torch.float32),T(sf,torch.float32),T(sm,torch.bool),T(dn),T(dt,torch.float32),T(df,torch.float32),T(dm,torch.bool)),index

def dictionary(batch):
 names=('src','dst','timestamp','history_nodes','history_times','history_edge_feats','history_mask','dst_history_nodes','dst_history_times','dst_history_edge_feats','dst_history_mask')
 return dict(zip(names,batch))

def apply_mask(batch, combined):
 out=list(batch); width=batch[6].shape[1]; out[6]=batch[6]&combined[:,:width]; out[10]=batch[10]&combined[:,width:]; return tuple(out)

@torch.no_grad()
def evaluate_explanations(predictor, positive, negative, importance_fn, device, *, compute_stability=True):
 predictor.eval()
 # Capture the shared evaluation universe before any model forward.  Some
 # executors derive internal masks from these tensors; the benchmark budget
 # must remain tied to the immutable input bank.
 pv=torch.cat((positive[6],positive[10]),1).clone()
 nv=torch.cat((negative[6],negative[10]),1).clone()
 po=predictor(*positive).logits; no=predictor(*negative).logits
 started=time.perf_counter(); pi=importance_fn(positive); ni=importance_fn(negative); torch.cuda.synchronize() if device.type=='cuda' else None
 runtime=1000*(time.perf_counter()-started)/(2*len(po))
 curves=[]
 def evaluate_budget(ratio=None, fixed_k=None):
  masks=[]
  for scores,valid in ((pi,pv),(ni,nv)):
   selected=torch.zeros_like(valid)
   for row in range(len(scores)):
    ids=torch.nonzero(valid[row]).flatten()
    if not len(ids): continue
    keep=min(len(ids),fixed_k if fixed_k is not None else max(1,int(math.ceil(ratio*len(ids)))))
    selected[row,ids[scores[row,ids].topk(keep).indices]]=True
   masks.append(selected)
  pe=predictor(*apply_mask(positive,masks[0])).logits; ne=predictor(*apply_mask(negative,masks[1])).logits
  pd=predictor(*apply_mask(positive,pv&~masks[0])).logits; nd=predictor(*apply_mask(negative,nv&~masks[1])).logits
  original=torch.cat((po,no)); explanation=torch.cat((pe,ne)); deleted=torch.cat((pd,nd)); labels=[1.]*len(po)+[0.]*len(no)
  original_ap=binary_average_precision(labels,torch.sigmoid(original).cpu().tolist()); deleted_ap=binary_average_precision(labels,torch.sigmoid(deleted).cpu().tolist())
  return {"ratio":ratio,"k":fixed_k,"accuracy":float(((explanation>=0)==(original>=0)).float().mean()),
   "sufficiency_probability_delta":float((torch.sigmoid(original)-torch.sigmoid(explanation)).abs().mean()),
   "necessity_probability_delta":float((torch.sigmoid(original)-torch.sigmoid(deleted)).abs().mean()),
   "deletion_fidelity_ap":original_ap-deleted_ap,"avg_selected_events":float(torch.cat(masks).sum(1).float().mean()),
   "avg_selected_fraction":float((torch.cat(masks).sum(1)/torch.cat((pv,nv)).sum(1).clamp_min(1)).float().mean())}
 for ratio in RATIOS: curves.append(evaluate_budget(ratio=ratio))
 fixed_k_curves=[evaluate_budget(fixed_k=k) for k in FIXED_K]
 x=np.asarray(RATIOS); width=x[-1]-x[0]
 area=lambda key:float(np.trapz([r[key] for r in curves],x)/width)
 # Stability to a small perturbation: remove the least-important valid event.
 def perturb(batch,scores,valid):
  mask=valid.clone()
  for row in range(len(scores)):
   ids=torch.nonzero(valid[row]).flatten()
   if len(ids)>1: mask[row,ids[scores[row,ids].argmin()]]=False
  return apply_mask(batch,mask)
 top_overlap=[]
 if compute_stability:
  p2=importance_fn(perturb(positive,pi,pv))
  for row in range(len(pi)):
   ids=torch.nonzero(pv[row]).flatten()
   if not len(ids): top_overlap.append(1.0); continue
   keep=max(1,int(math.ceil(.1*len(ids)))); a=set(ids[pi[row,ids].topk(keep).indices].cpu().tolist()); b=set(ids[p2[row,ids].topk(keep).indices].cpu().tolist()); top_overlap.append(len(a&b)/len(a|b))
 return {"base_ap":binary_average_precision([1.]*len(po)+[0.]*len(no),torch.sigmoid(torch.cat((po,no))).cpu().tolist()),
  "acc_auc":area('accuracy'),"deletion_aufsc":area('deletion_fidelity_ap'),"sufficiency_auc":area('sufficiency_probability_delta'),
  "necessity_auc":area('necessity_probability_delta'),"mean_selected_events":float(np.mean([r['avg_selected_events'] for r in curves])),
  "mean_available_events":float(torch.cat((pv,nv)).sum(1).float().mean()),
  "mean_selected_fraction":float(np.mean([r['avg_selected_fraction'] for r in curves])),
  "explanation_ms_per_query":runtime,"stability_jaccard":float(np.mean(top_overlap)) if top_overlap else None,"curves":curves,
  "fixed_k_curves":fixed_k_curves}

def main():
 global DATASET
 parser=argparse.ArgumentParser(); parser.add_argument('--dataset',choices=('wikipedia','reddit','mooc','lastfm'),default='wikipedia'); parser.add_argument('--queries',type=int,default=256); args=parser.parse_args(); DATASET=args.dataset
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); frame=read_csv_tail(ROOT/f'data/processed/ctdg/{DATASET}/events.csv',32768)
 frame['src']=pd.to_numeric(frame.src).astype('int64'); frame['dst']=pd.to_numeric(frame.dst).astype('int64'); frame['timestamp']=pd.to_numeric(frame.timestamp).astype('float32'); frame=frame.sort_values('timestamp',kind='stable').reset_index(drop=True)
 train,ev=split_indices(len(frame),.15); rows=ev[np.linspace(0,len(ev)-1,args.queries,dtype=np.int64)]; pool=frame.dst.drop_duplicates().to_numpy(np.int64)
 raw_columns=[column for column in frame.columns if str(column).startswith('feat_')]
 raw_features=frame[raw_columns].apply(pd.to_numeric,errors='coerce').fillna(0).to_numpy(np.float32)
 # Build a temporary history index solely to sample identical historical alternatives.
 temp=CTDGHistoryIndex(frame,128,int(max(frame.src.max(),frame.dst.max()))+1); shn,_,shm=temp.gather(frame.src.to_numpy(np.int64),np.arange(len(frame),dtype=np.int64))
 positive_dst=frame.iloc[rows].dst.to_numpy(np.int64); negative_dst,_=sample_historical_destinations(frame.iloc[rows].src.to_numpy(np.int64),positive_dst,shn[rows],shm[rows],pool,strategy='uniform')
 shared_index=CTDGHistoryIndex(frame,HISTORY_PER_ENDPOINT,int(max(frame.src.max(),frame.dst.max()))+1)
 outputs=[]
 for seed in (7,17,27):
  seed_all(seed)
  tgn_cfg={"num_nodes":"auto","edge_feat_dim":1,"hidden_dim":64,"time_dim":32,"dropout":.1}; tgn_path=train_checkpoint('TGN',seed,tgn_cfg,device)
  pos,_=make_index_and_batches(frame,rows,positive_dst,HISTORY_PER_ENDPOINT,device=device,index=shared_index); neg,_=make_index_and_batches(frame,rows,negative_dst,HISTORY_PER_ENDPOINT,device=device,index=shared_index)
  spec=RunSpec('ctdg',DATASET,ROOT/f'data/processed/ctdg/{DATASET}/events.csv','TGN','base',seed,'reference',tgn_cfg,False); tgn=build_benchmark_model(spec,frame,{"model_config":tgn_cfg,"attention":{}},device); tgn.load_state_dict(torch.load(tgn_path,map_location=device,weights_only=False)['model']); tgn.eval()
  # Actual explorer--navigator T-GNNExplainer.  The frozen TGN supplies all
  # coalition rewards; the navigator only learns an action-ordering prior.
  tgnn=TGNNExplainer(tgn,rollout=40,min_events=1,exploration=2.0,sparsity_weight=.05).to(device)
  nav_optimizer=torch.optim.AdamW(tgnn.navigator.parameters(),lr=2e-3)
  for start in range(0,min(1024,len(train)),64):
   tr=train[start:start+64]; pdst=frame.iloc[tr].dst.to_numpy(np.int64)
   nav_batch,_=make_index_and_batches(frame,tr,pdst,HISTORY_PER_ENDPOINT,device=device,index=shared_index)
   nav_loss=tgnn.navigator_training_loss(dictionary(nav_batch)); nav_optimizer.zero_grad(); nav_loss.backward(); nav_optimizer.step()
  tgnn.eval()
  def tgnn_importance(batch):
   return tgnn.explain(dictionary(batch),top_k=1).event_scores
  outputs.append({"seed":seed,"model":"TGN + T-GNNExplainer",**evaluate_explanations(tgn,pos,neg,tgnn_importance,device)})
  # TempME is a post-hoc explainer: its predictor must remain frozen.
  for parameter in tgn.parameters(): parameter.requires_grad_(False)
  tempme=TempME(tgn,num_nodes=int(max(frame.src.max(),frame.dst.max()))+1,edge_feat_dim=1,hidden_dim=64).to(device)
  optimizer=torch.optim.AdamW((p for p in tempme.parameters() if p.requires_grad),lr=2e-3)
  for _ in range(2):
   for start in range(0,min(8192,len(train)),256):
    tr=train[start:start+256]; pdst=frame.iloc[tr].dst.to_numpy(np.int64); ndst=np.roll(pdst,1)
    positive_batch,_=make_index_and_batches(frame,tr,pdst,HISTORY_PER_ENDPOINT,device=device,index=shared_index); negative_batch,_=make_index_and_batches(frame,tr,ndst,HISTORY_PER_ENDPOINT,device=device,index=shared_index)
    batch=tuple(torch.cat((left,right),0) for left,right in zip(positive_batch,negative_batch)); labels=torch.cat((torch.ones(len(tr),device=device),torch.zeros(len(tr),device=device)))
    loss=tempme.training_loss(dictionary(batch),labels,beta=.01); optimizer.zero_grad(); loss.backward(); optimizer.step()
  tempme.eval()
  def temp_importance(batch):
   inputs=dictionary(batch); src=tempme(**inputs,hard=True).event_scores; dst_inputs={**inputs,'history_nodes':inputs['dst_history_nodes'],'history_times':inputs['dst_history_times'],'history_edge_feats':inputs['dst_history_edge_feats'],'history_mask':inputs['dst_history_mask']}; dst=tempme(**dst_inputs,hard=True).event_scores; return torch.cat((src,dst),1)
  outputs.append({"seed":seed,"model":"TGN + TempME",**evaluate_explanations(tgn,pos,neg,temp_importance,device)})
  for name in ('TGIB','SIG'):
   cfg={"num_nodes":"auto","edge_feat_dim":1,"hidden_dim":64,"time_dim":32,"dropout":.1}; path=train_checkpoint(name,seed,cfg,device); spec=RunSpec('ctdg',DATASET,ROOT/f'data/processed/ctdg/{DATASET}/events.csv',name,'base',seed,'reference',cfg,False); model=build_benchmark_model(spec,frame,{"model_config":cfg,"attention":{}},device); model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model']); model.eval()
   outputs.append({"seed":seed,"model":name,**evaluate_explanations(model,pos,neg,lambda b,m=model:m(*b).aux['event_importance'],device)})
  saved=torch.load(ROOT/f'experiments/jodie_predicate_validation/checkpoints/{DATASET}_k1_seed{seed}/{DATASET}_LiFTER_seed{seed}.pt',map_location=device,weights_only=False); cfg=saved['model_config']; lindex=LiFTERHistoryIndex(frame,128,int(max(frame.src.max(),frame.dst.max()))+1,build_lifter_causal_fact_context(frame,cfg['fact_context_len']),raw_features); lpos,_=make_index_and_batches(frame,rows,positive_dst,128,True,cfg['fact_context_len'],device,index=lindex); lneg,_=make_index_and_batches(frame,rows,negative_dst,128,True,cfg['fact_context_len'],device,index=lindex); spec=RunSpec('ctdg',DATASET,ROOT/f'data/processed/ctdg/{DATASET}/events.csv','LiFTER','base',seed,'reference',cfg,False); model=build_benchmark_model(spec,frame,{"model_config":cfg,"attention":{}},device); model.load_state_dict(saved['model']); model.eval()
  # Align batches to the exact H-slot executor database.
  def align(batch):
   out=list(batch); h=model.max_grounding_facts
   for i in range(3,11): out[i]=out[i][:,-h:]
   return tuple(out)
  lpos,lneg=align(lpos),align(lneg)
  for common, symbolic in ((pos,lpos),(neg,lneg)):
   for slot in (3,4,6,7,8,10):
    if not torch.equal(common[slot],symbolic[slot]):
     raise RuntimeError(f'LiFTER evidence bank differs at tensor slot {slot}')
  outputs.append({"seed":seed,"model":"LiFTER",**evaluate_explanations(model,lpos,lneg,lambda b,m=model:m(*b).aux['event_importance'],device)})
  print('completed seed',seed,flush=True)
 summary={}
 for name in sorted({r['model'] for r in outputs}):
  rs=[r for r in outputs if r['model']==name]; summary[name]={k:{"mean":float(np.mean([r[k] for r in rs])),"std":float(np.std([r[k] for r in rs],ddof=1))} for k in ('base_ap','acc_auc','deletion_aufsc','sufficiency_auc','necessity_auc','mean_available_events','mean_selected_events','mean_selected_fraction','explanation_ms_per_query','stability_jaccard')}
 out=HERE/'results'/f'{DATASET}_summary.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"dataset":DATASET,"queries_per_seed":args.queries,"history_per_endpoint":HISTORY_PER_ENDPOINT,"ratios":RATIOS,"fixed_k":FIXED_K,"records":outputs,"summary":summary},indent=2)+'\n'); print(out)
if __name__=='__main__': main()
