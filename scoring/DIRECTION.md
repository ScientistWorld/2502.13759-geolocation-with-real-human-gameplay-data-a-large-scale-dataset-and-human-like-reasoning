# Research Direction

## Research Problem
This benchmark advances image geolocation using Vision Language Models (VLMs). The core question is whether better inference-time visual reasoning can improve a VLM's ability to infer geographical location from image evidence. The task requires predicting continent, country, and city from street-view or ground-level images.

## Approach Family
Scientists should work within inference-time VLM reasoning methods: text prompts, explanation scaffolds, self-checking, prediction parsing, and lightweight post-processing that do not require geolocation training. The research direction is to improve how a VLM extracts and uses image-grounded geographic evidence.

The scientist must not replace the problem with a fundamentally different paradigm such as supervised geolocation training or a pure image-retrieval system.

## Approach Scope
The scientist may:
- Improve or extend the explanation structure used at inference time
- Design better prompts for specific geographical regions or image types
- Improve the prediction parsing or post-processing pipeline
- Use multi-turn dialogue to refine predictions
- Combine explanatory VLM inference with lightweight retrieval or map priors, as long as the VLM's image-grounded reasoning remains central
- Evaluate on additional datasets or geographical settings

## Out of Bounds
- Replacing inference-time reasoning with fine-tuning or training on geolocation data
- Switching to purely retrieval-based methods
- Using a geolocation-specific model as the primary comparison (must compare prompting strategies on the same base VLM)
- Using additional training data beyond the prompt itself
