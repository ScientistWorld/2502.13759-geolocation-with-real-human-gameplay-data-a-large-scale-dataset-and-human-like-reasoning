# Scientist Constraints

## Model
The scientist must use a Vision Language Model (VLM) for geolocation. The benchmark is intended to compare inference-time visual reasoning strategies rather than model training. Scientists may:
- Use any VLM (open-source or API-based) that can process images and text
- Use different VLMs for different experiments
- Improve the text interface, explanation scaffold, parsing, or post-processing around a fixed VLM

Scientists must NOT:
- Fine-tune or train the VLM on geolocation data for the primary comparison
- Use a VLM specifically trained on geolocation data for the primary comparison (use general-purpose VLMs)

**Rationale**: The paper's claim is about improving geolocation through inference-time reasoning with existing VLM capabilities. Fine-tuning would change the comparison into a training-data benchmark.

## Data
The scientist must use the provided test set format. For the GeoComp experiment, use:
- GeoComp test set (500 images with city/country/continent labels) if available
- Im2GPS3K dataset as a substitute if GeoComp is unavailable
- YFCC26K or Im2GPS as additional generalization benchmarks

No external geolocation datasets may be used to train or validate the method. Test data must not be used for training.

## Evaluation Protocol
- Evaluate at three granularity levels: city, country, continent
- Use both classification metrics (accuracy, recall, F1) and distance metrics (1km, 25km, 750km thresholds)
- Report inference-time and generated-token overhead for the same prediction files when timing information is available
- Report results on the same test set as the reproduction for fair comparison

## Compliance Checklist
- [ ] Method is prompting-based (no fine-tuning on geolocation data)
- [ ] Evaluation uses the same test set as the reproduction
- [ ] All three granularity levels (city/country/continent) are reported
- [ ] Both classification and distance metrics are reported
- [ ] Inference-time overhead is reported or explicitly marked unavailable
- [ ] The method stays within inference-time image reasoning rather than supervised geolocation training
