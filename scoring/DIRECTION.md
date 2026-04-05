# Research Direction

## Research Problem
This paper advances image geolocation using Vision Language Models (VLMs). The core question is whether structured chain-of-thought prompting can improve a VLM's ability to reason about geographical location from visual cues. The task requires predicting continent, country, and city from street-view or ground-level images using multi-step visual reasoning.

## Core Contribution
The paper's novelty is GeoCoT (Geographical Chain-of-Thought): a structured five-step prompting strategy that guides VLMs through progressively finer-grained geographical reasoning:
1. Climate/continental features
2. Cultural and architectural markers
3. Infrastructure cues (driving side, license plates)
4. Landmark-based verification
5. Micro-level validation (sidewalk patterns, clothing)

The scientist must work on improving this prompting-based approach — not replacing it with a fundamentally different paradigm.

## Approach Scope
The scientist should focus on the prompting/reasoning strategy for geolocation VLMs. They may:
- Improve or extend the GeoCoT prompt structure (add/modify reasoning steps)
- Design better prompts for specific geographical regions or image types
- Improve the prediction parsing or post-processing pipeline
- Use multi-turn dialogue to refine predictions
- Combine GeoCoT with retrieval-based methods for improved precision
- Evaluate on additional datasets or geographical settings

## Out of Bounds
- Replacing the prompting approach with fine-tuning or training on geolocation data
- Switching to purely retrieval-based methods (bypasses the reasoning contribution)
- Using a geolocation-specific model as the primary comparison (must compare prompting strategies on the same base VLM)
- Using additional training data beyond the prompt itself
