# Scientist Constraints

## Model
The scientist must use a Vision Language Model (VLM) for geolocation. The contribution of the paper is the GeoCoT prompting strategy, not the model architecture. Scientists may:
- Use any VLM (open-source or API-based) that can process images and text
- Use different VLMs for different experiments
- Improve the prompting strategy, prompt engineering, or reasoning framework

Scientists must NOT:
- Fine-tune or train the VLM on geolocation data (the paper's method requires no training)
- Use a VLM specifically trained on geolocation data for the primary comparison (use general-purpose VLMs)

**Rationale**: GeoCoT is a prompting strategy. Its value is in showing that structured reasoning improves performance over generic prompting. Fine-tuning would bypass this contribution.

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
- [ ] GeoCoT structured prompting is the method being improved
