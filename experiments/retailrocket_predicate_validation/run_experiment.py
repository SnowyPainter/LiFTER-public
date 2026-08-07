#!/opt/conda/bin/python
from __future__ import annotations
import argparse, json, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = "/opt/conda/bin/python"

def complete(path: Path) -> bool:
    try:
        value=json.loads(path.read_text())
        return "action_recovery" in value and "historical_negative_auc" in value["global_predicate_shuffle"]
    except Exception: return False

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--workers",type=int,default=3); parser.add_argument("--force",action="store_true"); parser.add_argument("--progress",action="store_true"); args=parser.parse_args()
    cells=[]
    for k in (1,2,3,4,8):
        for seed in (7,17,27):
            path=HERE/"outputs"/f"k{k}_seed{seed}.json"
            if not args.force and complete(path): print(f"SKIP K={k}/seed{seed}",flush=True)
            else: cells.append((k,seed))
    def run(cell):
        k,seed=cell; print(f"START K={k}/seed{seed}",flush=True)
        command=[PYTHON,str(HERE/"run_cell.py"),"--k",str(k),"--seed",str(seed)]
        if args.progress: command.append("--progress")
        subprocess.run(command,check=True); print(f"DONE K={k}/seed{seed}",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,cell) for cell in cells]
        for future in as_completed(futures): future.result()
    subprocess.run([PYTHON,str(HERE/"summarize.py")],check=True)
if __name__=="__main__": main()
