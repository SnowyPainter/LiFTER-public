#!/opt/conda/bin/python
import argparse, subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
HERE=Path(__file__).resolve().parent; PY='/opt/conda/bin/python'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--workers',type=int,default=3); p.add_argument('--full-scale',action='store_true')
 p.add_argument('--datasets',nargs='+',default=['mooc','lastfm']); p.add_argument('--variants',nargs='+',default=['full','no_exact_binding','randomized_binding','no_renewal','no_transitions','q_only','symbolic_without_q','matched_mlp']); p.add_argument('--seeds',nargs='+',type=int,default=[7,17,27]); a=p.parse_args()
 cells=[(d,v,s) for d in a.datasets for v in a.variants for s in a.seeds]
 def run(c):
  d,v,s=c; cmd=[PY,str(HERE/'run_cell.py'),'--dataset',d,'--variant',v,'--seed',str(s)];
  if a.full_scale: cmd.append('--full-scale')
  subprocess.run(cmd,check=True); return c
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  for future in as_completed([pool.submit(run,c) for c in cells]): print('DONE',future.result(),flush=True)
 subprocess.run([PY,str(HERE/'summarize.py'),'--scale','full' if a.full_scale else 'pilot'],check=True)
if __name__=='__main__': main()
