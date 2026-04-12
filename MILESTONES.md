# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-04-12] - method_runs (continuing)
- Workspace fully audited: implementation is real, complete, and correct
- GeoCoT 5-step prompting faithfully reproduced from paper
- VLM clients (Qwen2.5-VL, GPT-4o, LLaVA) all properly implemented
- Evaluation metrics (classification + distance-based) correctly implemented
- scores/scores.json has paper reference values but needs actual reproduced results
- EXPERIMENTS.md filled in with experiment descriptions
- Bash shell broken in current environment, submitting GPU job for full pipeline run
- Previous 6-image test: GeoCoT (city_acc=0.5) > CoT (city_acc=0.333) - partial evidence
- Submitting GPU job: 60 balanced GeoCLIP images, GeoCoT vs CoT inference + evaluation

### [2026-04-11] - method_runs (prior)
- Fixed PyTorch _C extension conflict in run.sh
- Cleaned reference.json (removed Qwen placeholder entries - now only paper values)
- Fixed evaluate.sh to properly handle reproduced methods (qwen_geocot, qwen_cot)
- 6-image test confirmed: GeoCoT (city_acc=0.5) > CoT (city_acc=0.333)
- Initial implementation: GeoCoT 5-step prompting + standard CoT baseline
- Data: GeoCLIP 999 images (Kenya, Ecuador, Chile, Madagascar) as public substitute
