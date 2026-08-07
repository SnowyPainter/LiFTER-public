#!/opt/conda/bin/python
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent; PY='/opt/conda/bin/python'
subprocess.run([PY,str(HERE/'prepare_oracle_data.py')],check=True)
with ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(lambda seed: subprocess.run([PY,str(HERE/'run_oracle_cell.py'),'--seed',str(seed)],check=True),(7,17,27)))
subprocess.run([PY,str(HERE/'run_probe.py')],check=True)
subprocess.run([PY,str(HERE/'summarize.py')],check=True)
