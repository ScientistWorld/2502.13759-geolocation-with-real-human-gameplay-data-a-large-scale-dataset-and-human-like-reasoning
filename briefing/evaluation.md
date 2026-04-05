# Evaluation

How success is measured. Describe only the metrics, targets, and evaluation procedure — do NOT describe the paper's method or how they achieved these results. Think of this as a general evaluation protocol for other scientists trying to solve the same problem.

## Metrics

Performance is measured at three levels of geographical granularity:

1. **Classification-based metrics** (at city, country, and continent levels):
   - **Accuracy**: proportion of correct predictions out of all predictions
   - **Recall**: proportion of true positive predictions out of all actual positive cases
   - **F1 Score**: harmonic mean of precision and recall

2. **Distance-based metrics** (thresholds define correct prediction):
   - **Street-level (1km)**: fraction of predictions within 1km of ground truth
   - **City-level (25km)**: fraction of predictions within 25km of ground truth
   - **Country-level (750km)**: fraction of predictions within 750km of ground truth

3. **Reasoning quality metrics** (for methods that produce textual reasoning):
   - **GPTScore**: similarity between generated reasoning and ground-truth reasoning
   - **Completeness of Extraction (CE)**: whether all key clues in ground truth are covered (0-5 scale)
   - **Accuracy of Extraction (AE)**: correctness of identified attributes (0-5 scale)
   - **Accuracy of Correspondence (AC)**: whether conclusions match ground-truth logic (0-5 scale)
   - **Logical Coherence (LC)**: consistency and flow of reasoning chain (0-5 scale)

4. **Hallucination metrics** (for reasoning quality):
   - **Object Hallucination (OH)**: count of objects mentioned that don't exist in the image
   - **Fact Hallucination (FH)**: count of incorrect factual information
   - **Attribution Hallucination (AH)**: count of incorrectly attributed properties

## Baselines and Targets

The paper evaluates several baseline approaches:
- **LLaVA-1.6**: general-purpose vision-language model
- **Llama-3.2-Vision**: general-purpose vision-language model
- **Qwen-VL**: general-purpose vision-language model
- **GeoCLIP**: CLIP-inspired geolocation-specific model
- **GeoReasoner**: reasoning-based geolocation method using LVLMs
- **Kimi-latest**: closed-source vision-language model
- **Kimi-latest(CoT)**: Kimi with standard chain-of-thought prompting
- **GPT-4o**: closed-source vision-language model
- **GPT-4o(CoT)**: GPT-4o with standard chain-of-thought prompting

Target performance (paper's reported results on GeoComp test set with 500 images):
- GPT-4o(CoT) achieves ~0.094 city accuracy, 0.623 country accuracy, 0.819 continent accuracy
- GPT-4o achieves ~0.045 street accuracy, 0.147 city accuracy, 0.678 country accuracy
- On Im2GPS3K, GPT-4o(CoT) achieves ~0.14 street, 0.45 city, 0.69 country accuracy

## Evaluation Protocol

1. **Test set**: 500 geo-tagged locations selected via stratified sampling across continents (20 mainstream countries, 6 continents). Each image has ground-truth continent, country, and city labels, plus GPS coordinates.
2. **Model input**: Each model receives the image and produces either a direct location prediction or a reasoning chain followed by a location prediction.
3. **For classification metrics**: Parse the model's city/country/continent prediction and compare against ground truth.
4. **For distance metrics**: Convert city/country predictions to GPS coordinates (using the center of the predicted city/country) and compute haversine distance to ground truth.
5. **Statistical significance**: Use two-tailed paired t-test with p-value < 0.05 to determine significant improvements over baselines.
