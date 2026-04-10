# Reproduction Milestones

**Current: none** (job submitted for GPU experiments)

## Progress Log

### [2026-04-10 00:00] - none
- Analysis: Read paper, identified GeoCoT as a 5-step chain-of-thought prompting method for image geolocation
- Paper uses GPT-4o on GeoComp test set (500 images) - GeoCoT achieves 0.118 city accuracy, 0.640 country accuracy, 0.862 continent accuracy
- Dataset limitation: GeoComp test set not publicly available, Im2GPS3K URL returns 404
- Deviation: Using LLaVA-1.5-7B (local) instead of GPT-4o, GeoCLIP dataset (999 images) as substitute
- Environment: Fixed container.def to use MCR Azure Linux (avoid Docker Hub rate limits), install PyTorch/transformers packages
- Updated vlm_client.py for flexible package paths (system Python, /pylib, /dev/shm/pylib)
- Updated run.sh to run experiments on compute node with LLaVA-1.5-7B
- Updated scoring files with geoclip_reproduction experiment
- **Submitted GPU job** to test container build and run GeoCoT vs CoT experiments

## Stop Justification

(Not applicable — reproduction in progress)
