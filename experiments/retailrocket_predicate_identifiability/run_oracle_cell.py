#!/opt/conda/bin/python
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch, yaml

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from evaluation.evaluate import RunSpec,evaluate_run

def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--seed',type=int,required=True); args=pa.parse_args()
    config=yaml.safe_load((ROOT/'experiments/main_benchmark/config_lifter.yaml').read_text())
    edge_dim=52
    centres=np.zeros((3,edge_dim),dtype=np.float32); centres[:,1:4]=np.eye(3,dtype=np.float32)
    scale=np.full(edge_dim,1e6,dtype=np.float32); scale[1:4]=1.0
    config.update(dataset='retailrocket_oracle',model='LiFTER',seed=args.seed)
    config['training'].update(max_examples=262144,query_filter={'column':'action_type','value':'transaction'},historical_negative_strategy='uniform',progress=False)
    config['model_config'].update(predicate_count=3,predicate_assignment_mode='kmeans_fixed',predicate_execution_mode='hard',fixed_predicate_centroids=centres.tolist(),fixed_predicate_mean=np.zeros(edge_dim).tolist(),fixed_predicate_scale=scale.tolist())
    config['explanation_evaluation']['max_queries']=0; config['lifter_diagnostics']['enabled']=False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    spec=RunSpec('ctdg','retailrocket_oracle',ROOT/'data/processed/ctdg/retailrocket_oracle/events.csv','LiFTER','base',args.seed,'reference',dict(config['model_config']),False)
    metrics=evaluate_run(spec,config,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    out=HERE/'outputs'/f'oracle_seed{args.seed}.json'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'seed':args.seed,'metrics':metrics},indent=2)+'\n'); print(f'saved {out}')
if __name__=='__main__': main()
