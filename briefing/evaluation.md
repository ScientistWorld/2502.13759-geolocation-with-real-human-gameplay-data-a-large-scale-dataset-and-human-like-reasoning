# Evaluation

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

3. **Reasoning quality metrics** (for systems that produce textual explanations):
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

Reference results for prior systems and the reproduced system are stored in `scoring/reference.json`. The primary benchmark target is to improve fine-grained geolocation without relying solely on a single metric: city, country, and continent classification metrics should be considered alongside distance-threshold accuracy and inference overhead.

For the main 500-image benchmark, strong reference systems reach roughly 0.09-0.12 city accuracy, 0.62-0.64 country accuracy, and 0.82-0.86 continent accuracy. Distance-threshold reference results are substantially lower for strict thresholds, with street-level accuracy below 0.10 and city-level accuracy below 0.20.

## Evaluation Protocol

1. **Test set**: 500 geo-tagged locations selected via stratified sampling across continents (20 mainstream countries, 6 continents). Each image has ground-truth continent, country, and city labels, plus GPS coordinates.
2. **Model input**: Each system receives the image and produces a parseable location prediction. Systems may also emit textual explanations.
3. **For classification metrics**: Parse the model's city/country/continent prediction and compare against ground truth.
4. **For distance metrics**: Convert city/country predictions to GPS coordinates (using the center of the predicted city/country) and compute haversine distance to ground truth.
5. **Statistical significance**: Use two-tailed paired t-test with p-value < 0.05 to determine significant improvements over baselines.
