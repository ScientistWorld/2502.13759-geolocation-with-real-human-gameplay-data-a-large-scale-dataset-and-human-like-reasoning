# Progress

## What Works

### Briefing
- `briefing/problem.md` — Describes the image geolocation problem (predicting location from visual cues)
- `briefing/evaluation.md` — Describes evaluation metrics (classification accuracy, distance-based accuracy, reasoning quality)
- `briefing/method.md` — Describes GeoCoT (Geographical Chain-of-Thought) prompting framework
- `briefing/overview.md` — Paper summary with key results

### Method Implementation
- `method/prompt_template.py` — GeoCoT 5-step prompting and standard CoT baseline prompts
- `method/vlm_client.py` — VLM client supporting GPT-4o (API), Qwen2.5-VL (local), and LLaVA (local)
- `method/run_geocot.py` — Main script to run GeoCoT on a dataset

### Data
- `data/loader.py` — Dataset loader for Im2GPS3K, YFCC26K, GeoComp formats; includes sample dataset generator

### Evaluation
- `eval/metrics.py` — Geolocation metrics: classification (accuracy/recall/F1), distance (1km/25km/750km), haversine distance

### Scoring
- `scoring/reference.json` — Paper's reported numbers from Tables 2, 3, 4, and 8
- `scoring/scores.json` — Output file for reproduced numbers
- `scoring/TARGETS.md`, `CONSTRAINTS.md`, `DIRECTION.md` — Evaluation targets and constraints

### Scripts
- `scripts/download.sh` — Downloads Qwen2.5-VL model and Im2GPS3K dataset
- `scripts/method.sh` — Runs GeoCoT prompting with local VLM
- `scripts/baseline.sh` — Runs standard CoT baseline
- `scripts/evaluate.sh` — Evaluates predictions and generates scores.json
- `scripts/reproduce.sh` — End-to-end reproduction pipeline
- `scripts/run.sh` — Job submission script

### Environment
- `environment/container.def` — CUDA 12.1 container definition
- `environment/setup.sh` — Python package installation

## Results

No results yet — experiments pending. The reproduction will use Qwen2.5-VL (or LLaVA) with GeoCoT prompting on Im2GPS3K, comparing against standard CoT baseline.

Expected: GeoCoT should outperform standard CoT by demonstrating the value of structured multi-step geographical reasoning.

## Remaining

1. **Download models and data** — Need to run `scripts/download.sh` on login node with internet
2. **Run GeoCoT experiments** — Execute on GPU node with Qwen2.5-VL or LLaVA
3. **Run baseline experiments** — Execute standard CoT for comparison
4. **Evaluate and compare** — Generate scores.json and compare against paper numbers
5. **Validate workspace** — Run `python validate.py` to verify structure

## Issues

- **Login node bash shell unresponsive** — All bash commands fail after pip install issue. Using compute node for execution.
- **GeoComp test set not publicly available** — The 500-image GeoComp test set was released on an anonymous GitHub repo requiring authentication. Using Im2GPS3K as the primary test dataset.
- **GPT-4o not available locally** — Using Qwen2.5-VL-7B-Instruct as the local VLM. Smaller than GPT-4o but should demonstrate GeoCoT's effectiveness.
- **Im2GPS3K images require download** — Creating sample dataset for initial pipeline validation.

## Deviations from Paper

| Aspect | Paper | Reproduction |
|--------|-------|-------------|
| VLM | GPT-4o (closed) | Qwen2.5-VL-7B-Instruct (local) |
| Test set | GeoComp (500 images, not public) | Im2GPS3K (substitute, ~3K images) |
| Dataset scale | 2.7M images in GeoComp | Im2GPS3K ~3K images |
| Model size | GPT-4o (large proprietary) | Qwen2.5-VL-7B (7B params) |

The GeoCoT method itself is faithfully reproduced — the same 5-step prompting structure with the same reasoning questions. Only the model and test data differ.
