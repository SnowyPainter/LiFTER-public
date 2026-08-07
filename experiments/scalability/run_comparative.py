#!/opt/conda/bin/python
"""Profile predictor throughput under one shared local-history workload."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(ROOT))
from models.factory import build_model
from experiments.scalability.run_experiment import make_batch, make_model, timed

MODELS=("TGN","TGAT","DyGFormer","GraphMixer","CRAFT","LiFTER")

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]
    for batch_size in (32,128,512):
        lifter_batch=make_batch(batch_size,10,device)
        neural_batch=list(lifter_batch); neural_batch[5]=neural_batch[5][...,:1]; neural_batch[9]=neural_batch[9][...,:1]; neural_batch=tuple(neural_batch)
        for name in MODELS:
            if name=='LiFTER': model=make_model(10,1,True,device); batch=lifter_batch
            elif name=='CRAFT': model=build_model(name,num_nodes=12000,hidden_dim=64,num_heads=2,max_neighbors=10,dropout=.1).to(device).eval(); batch=neural_batch
            else: model=build_model(name,num_nodes=12000,edge_feat_dim=1,hidden_dim=64,time_dim=32,neural_history_len=10,dropout=.1).to(device).eval(); batch=neural_batch
            with torch.no_grad(): latency=timed(lambda:model(*batch),warmup=3,repeats=12)
            rows.append({'model':name,'batch':batch_size,'history_per_endpoint':10,'latency_ms':latency,'queries_per_second':1000*batch_size/latency})
            del model
    explanation={}
    for d in ('wikipedia','reddit','mooc','lastfm'):
        summary=json.loads((ROOT/f'experiments/explanation_quality/results/{d}_summary.json').read_text())['summary']
        for model,values in summary.items(): explanation.setdefault(model,[]).append(values['explanation_ms_per_query']['mean'])
    result={'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','protocol':{'history_per_endpoint':10,'random_weights_for_latency_only':True},'forecasting':rows,'explanation_macro_ms_per_query':{m:sum(v)/len(v) for m,v in explanation.items()}}
    out=HERE/'results'/'comparative_summary.json'; out.write_text(json.dumps(result,indent=2)+'\n'); print(out)

if __name__=='__main__': main()
