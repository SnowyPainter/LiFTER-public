# LiFTER

LiFTER(Link-Fact Temporal Rule Inducer)는 CTDG interaction을 grounded temporal
facts로 보존하고 finite temporal rule language의 실행으로 future link를 예측하는
neuro-symbolic model이다. 연구 주장과 결과는 [`docs/claim.md`](docs/claim.md)에
정리되어 있다. 아래 명령은 모두 저장소 루트에서 실행한다.

## 1. 환경과 코드 확인

공식 runner는 `/opt/conda/bin/python`을 사용한다. 실험 환경은 Python 3.10,
PyTorch 2.12.0, CUDA 13.0, NumPy 1.26.3, pandas 2.3.3과
scikit-learn 1.7.2다. 전체 benchmark에는 CUDA GPU를 권장한다.

```bash
cd /workspace/nsctdg
PYTHON=/opt/conda/bin/python
$PYTHON -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
$PYTHON -m unittest discover -s tests -q
```

## 2. 데이터 준비

```bash
$PYTHON etl/scripts/download_datasets.py \
  --domain ctdg \
  --dataset wikipedia,reddit,mooc,lastfm \
  --materialize
```

Materialized inputs는 `data/processed/ctdg/{dataset}/events.csv`에 저장된다. 모든
main experiments는 timestamp로 정렬된 최근 최대 32,768 events, chronological
85/15 split, 10 epochs와 seeds 7/17/27을 사용한다.

## 3. Grounding capacity 선택

Main test와 분리된 development prefix의 validation split에서 공통
grounding capacity를 선택한다.

```bash
$PYTHON experiments/grounding_budget_validation/run_validation.py \
  --workers 3 --progress
```

- Raw cells: `experiments/grounding_budget_validation/outputs/`
- Selection: `experiments/grounding_budget_validation/results/summary.json`
- `claim.md`에서 선택된 값: `H=10`

## 4. Main forecasting benchmark

LiFTER와 EdgeBank, TGN, TGAT, DyGFormer, GraphMixer, CRAFT, CRAFT-R을 동일한
uniform-historical-negative protocol로 평가한다.

```bash
$PYTHON experiments/main_benchmark/run_uniform_historical.py \
  --models EdgeBank TGN TGAT DyGFormer GraphMixer CRAFT CRAFT-R LiFTER \
  --datasets wikipedia reddit mooc lastfm \
  --seeds 7 17 27 \
  --progress

$PYTHON experiments/main_benchmark/summarize_uniform_historical.py
$PYTHON experiments/main_benchmark/audit_tgn_causality.py
```

- Raw cells: `experiments/main_benchmark/outputs/{dataset}_{model}_base_seed{seed}.json`
- Summary: `experiments/main_benchmark/results/uniform_historical_summary.json`
- Final K=1 LiFTER summary: `experiments/main_benchmark/results/lifter_k1_summary.json`, `lifter_k1_summary.md`
- TGN causality audit: `experiments/main_benchmark/results/tgn_causality_audit.json`

LiFTER만 실행하려면 다음처럼 제한한다.

```bash
$PYTHON experiments/main_benchmark/run_uniform_historical.py \
  --models LiFTER \
  --datasets wikipedia reddit mooc lastfm \
  --seeds 7 17 27 \
  --progress
```

완료된 cell은 자동으로 건너뛴다. 다시 학습하려면 `--force`를 추가한다. 개별
cell 실행법은 [`experiments/main_benchmark/README.md`](experiments/main_benchmark/README.md)에
있다.

## 5. Predicate vocabulary 분석

### 5.1 JODIE에서 K 선택

`K=1,2,4,8`과 predicate-assignment shuffle을 비교한다. 이 단계의 K=1
checkpoints는 이후 mechanism, explanation과 execution 분석에서 재사용된다.

```bash
$PYTHON experiments/jodie_predicate_validation/run_experiment.py \
  --datasets wikipedia reddit mooc lastfm \
  --ks 1 2 4 8 \
  --seeds 7 17 27 \
  --workers 3 --progress
```

- Raw cells: `experiments/jodie_predicate_validation/outputs/`
- Checkpoints: `experiments/jodie_predicate_validation/checkpoints/`
- Summary: `experiments/jodie_predicate_validation/results/summary.json`

### 5.2 Typed predicate가 유효해지는 조건

```bash
$PYTHON experiments/predicate_phase_diagram/run_experiment.py
```

- Summary: `experiments/predicate_phase_diagram/results/summary.json`
- Report: `experiments/predicate_phase_diagram/results/report.md`

이 controlled stream은 hidden event type이 이후 destination distribution을 바꾸는
정도와 event 이전 observations에서 type을 구별할 수 있는 정도를 독립적으로
조절한다.

## 6. Predictive-mechanism diagnosis

Frozen K=1 checkpoints에서 일곱 execution components의 `2^7=128` combinations,
Shapley allocation, query regimes와 matched grounded-fact deletion을 계산한다.

```bash
$PYTHON experiments/mechanism_decomposition/run_experiment.py \
  --workers 3 --progress
```

- Raw cells: `experiments/mechanism_decomposition/outputs/`
- Summary/report: `experiments/mechanism_decomposition/results/summary.json`, `report.md`
- Deep analysis: `experiments/mechanism_decomposition/results/deep_summary.json`, `deep_report.md`

이 결과는 learned predictor의 score를 분해하며 실제 data-generating mechanism의
causal identification을 의미하지 않는다.

## 7. Neural-symbolic responsibility

MOOC와 LastFM에서 exact binding, randomized binding, renewal, transition,
transition-only와 same-input MLP variants를 모두 처음부터 다시 학습한다.

```bash
$PYTHON experiments/neural_symbolic_responsibility/run_experiment.py \
  --full-scale \
  --datasets mooc lastfm \
  --seeds 7 17 27 \
  --workers 3
```

- Raw cells: `experiments/neural_symbolic_responsibility/outputs/`
- Summary: `experiments/neural_symbolic_responsibility/results/full_summary.json`
- Report: `experiments/neural_symbolic_responsibility/results/report.md`

## 8. Explanation evaluation

TGN + T-GNNExplainer, TGN + TempME, TGIB, SIG와 LiFTER를 동일한 256
chronological queries와 seeds 7/17/27에서 비교한다. 모든 방법에는 query마다
최근 source events 10개와 destination events 10개로 구성된 동일한 evidence bank를
제공한다. Explanation-ratio grid 5--30%와 query마다 선택하는 event 수를 고정한
`k = {1, 2, 3, 5, 10}`을 함께 평가한다.

```bash
for dataset in wikipedia reddit mooc lastfm; do
  $PYTHON experiments/explanation_quality/run_experiment.py \
    --dataset "$dataset" --queries 256
done
```

- Dataset results: `experiments/explanation_quality/results/{dataset}_summary.json`
- Combined report: `experiments/explanation_quality/results/report.md`

Dataset JSON을 재생성해 수치가 바뀌면 combined report도 함께 갱신해야 한다.

## 9. Execution agreement

`--queries 0`은 네 chronological test splits의 전체 queries를 의미한다.

```bash
for dataset in wikipedia reddit mooc lastfm; do
  $PYTHON experiments/execution_certificate/run_experiment.py \
    --dataset "$dataset" --queries 0
done
```

- Dataset results: `experiments/execution_certificate/results/{dataset}_summary.json`
- Report: `experiments/execution_certificate/results/report.md`

이 검사는 prediction과 explanation의 execution agreement를 확인하는 implementation
verification이며 forecasting 또는 robustness metric이 아니다.

## 10. Full-stream scalability

Reddit 32k/128k/512k/full stream의 fact index와 `H`, `K`, transition rank, batch,
training throughput, trace serialization과 fact-deletion re-execution을 측정한다.

```bash
$PYTHON experiments/scalability/run_experiment.py
```

- Summary: `experiments/scalability/results/summary.json`
- Report: `experiments/scalability/results/report.md`

## 11. `claim.md` 대응표

| Section | Reproduction artifact |
|---|---|
| 5.1 Experimental setup | `grounding_budget_validation/results/summary.json`, benchmark configs |
| 5.2 Forecasting results | `main_benchmark/results/uniform_historical_summary.json` |
| 5.3 Predicate analysis | `jodie_predicate_validation/results/summary.json`, `predicate_phase_diagram/results/` |
| 5.4 Predictive-mechanism decomposition | `mechanism_decomposition/results/summary.json`, `deep_summary.json` |
| 5.5 Neural-symbolic responsibilities | `neural_symbolic_responsibility/results/full_summary.json` |
| 5.6 Explanation quality | `explanation_quality/results/{dataset}_summary.json` |
| 5.6 Execution agreement | `execution_certificate/results/{dataset}_summary.json` |
| 5.7 Scalability | `scalability/results/summary.json` |

## 12. 재현 규칙

- 표의 `±`는 seeds 7/17/27의 sample standard deviation이다.
- 모든 split은 chronological하며 query 이후 events를 history에 넣지 않는다.
- `outputs/`는 per-cell raw results, `results/`는 aggregation과 reports,
  `checkpoints/`는 downstream frozen-model analyses에 사용된다.
- Orchestration runner의 `--force`는 검증된 기존 cell도 다시 계산한다.
- Workers는 같은 GPU memory를 공유한다. OOM이 발생하면 `--workers 1`을 사용한다.

마지막으로 다음을 실행한다.

```bash
$PYTHON -m unittest discover -s tests -q
git diff --check
```
