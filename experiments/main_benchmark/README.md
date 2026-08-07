# Main CTDG benchmark

LiFTER와 EdgeBank, TGN, TGAT, DyGFormer, GraphMixer, CRAFT, CRAFT-R을 동일한
continuous-time link forecasting protocol로 평가한다.

## Protocol

- datasets: Wikipedia, Reddit, MOOC, LastFM
- events: dataset별 32,768
- chronological train/test split: 85/15
- learned models: 10 epochs, seeds 7/17/27
- evaluation candidates: uniform historical negatives with random fallback, and random negatives
- reported metrics: Historical AUC/AP and Random AUC/AP

LiFTER는 모든 interactions을 하나의 observed predicate로 표현하는 $K=1$ program,
grounding capacity $H=10$, hidden dimension 64와 transition rank 32를 사용한다. $H$는
main test를 제거한 development prefix의 chronological validation에서
`{10,20,40,80}` 중 하나를 고르며, 네 datasets의 macro Historical AUC를 selection
criterion으로 사용한다.

## Run

저장소 루트에서 실행한다.

```bash
/opt/conda/bin/python experiments/main_benchmark/run_uniform_historical.py \
  --models EdgeBank TGN TGAT DyGFormer GraphMixer CRAFT CRAFT-R LiFTER \
  --datasets wikipedia reddit mooc lastfm \
  --seeds 7 17 27 \
  --progress

/opt/conda/bin/python experiments/main_benchmark/summarize_uniform_historical.py
```

완료된 cells는 자동으로 건너뛴다. 다시 실행하려면 `--force`, 서로 독립인 cells를
병렬 실행하려면 `--workers N`을 추가한다. LiFTER만 실행하는 명령은 다음과 같다.

```bash
/opt/conda/bin/python experiments/main_benchmark/run_uniform_historical.py \
  --models LiFTER \
  --datasets wikipedia reddit mooc lastfm \
  --seeds 7 17 27 \
  --workers 3 --progress
```

## Results

- raw cells: `outputs/{dataset}_{model}_base_seed{seed}.json`
- complete comparison: `results/uniform_historical_summary.json`
- final single-predicate LiFTER result: `results/lifter_k1_summary.json`
- human-readable LiFTER table: `results/lifter_k1_summary.md`
- TGN causality audit: `results/tgn_causality_audit.json`

The summarizer validates that every LiFTER cell uses $K=1$, $H=10$, 32,768 events,
10 epochs and uniform historical negatives before writing the dedicated K=1 artifacts.
