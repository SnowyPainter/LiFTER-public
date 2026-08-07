#!/opt/conda/bin/python
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch, yaml

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; sys.path.insert(0,str(ROOT))
from evaluation.evaluate import RunSpec, evaluate_run

VARIANTS={
 "full": {},
 "no_exact_binding": {"exact_entity_binding":False},
 "randomized_binding": {"binding_role_permutation":[2,5,4,0,3,1]},
 "no_renewal": {"include_renewal_rules":False},
 "no_transitions": {"latent_transition_rule":False,"latent_second_order_transition_rule":False},
 "q_only": {"program_rules_enabled":False,"positioned_recurrence_rules":False},
 "symbolic_without_q": {"latent_transition_rule":False,"latent_second_order_transition_rule":False},
}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['mooc','lastfm'],required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--variant',choices=[*VARIANTS,'matched_mlp'],required=True); p.add_argument('--full-scale',action='store_true'); a=p.parse_args()
 config=yaml.safe_load((ROOT/'experiments/main_benchmark/config_lifter.yaml').read_text()); config.update(dataset=a.dataset,seed=a.seed)
 config['training'].update(epochs=10 if a.full_scale else 3,max_examples=32768,train_max_examples=0 if a.full_scale else 8192,train_sampling_strategy='uniform_horizon',progress=False,historical_negative_strategy='uniform')
 config['explanation_evaluation']['max_queries']=0; config['lifter_diagnostics']['enabled']=False; config['paths'].pop('checkpoints',None); config['paths'].pop('load_checkpoint',None)
 model='LiFTER'; model_config=dict(config['model_config'])
 if a.variant=='matched_mlp':
  model='GraphMixer'; model_config={"num_nodes":"auto","edge_feat_dim":1,"hidden_dim":64,"time_dim":32,"neural_history_len":20,"dropout":0.1}
 else: model_config.update(VARIANTS[a.variant])
 config['model']=model; config['model_config']=model_config
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
 spec=RunSpec('ctdg',a.dataset,ROOT/f'data/processed/ctdg/{a.dataset}/events.csv',model,a.variant,a.seed,'reference',model_config,False)
 metrics=evaluate_run(spec,config,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
 result={"dataset":a.dataset,"seed":a.seed,"variant":a.variant,"scale":"full" if a.full_scale else "pilot","historical_auc":metrics['historical_negative_auc'],"historical_ap":metrics['historical_negative_ap'],"random_auc":metrics['auc'],"random_ap":metrics['average_precision'],"train_seconds":metrics.get('train_seconds'),"trainable_parameters":metrics.get('trainable_parameters',metrics.get('attention_audit',{}).get('trainable_parameters'))}
 out=HERE/'outputs'/f"{a.dataset}_{a.variant}_seed{a.seed}_{result['scale']}.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
if __name__=='__main__': main()
