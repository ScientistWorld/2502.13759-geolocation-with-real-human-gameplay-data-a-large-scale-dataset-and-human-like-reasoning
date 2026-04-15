# Evaluation Targets

## Primary: geocomp_classification
Geolocation accuracy at city, country, and continent levels on the main geolocation test set.
- **Metric**: city_accuracy, city_recall, city_f1, country_accuracy, country_recall, country_f1, continent_accuracy, continent_recall, continent_f1 (higher is better)
- **Reference target**: the proposed system in the paper reaches 0.118 city accuracy, 0.640 country accuracy, and 0.862 continent accuracy; the full recall/F1 values are encoded in `scoring/reference.json`.

## Primary: geocomp_distance
Distance-based accuracy: fraction of predictions within 1km (street), 25km (city), 750km (country).
- **Metric**: street_1km, city_25km, country_750km (higher is better)
- **Reference target**: the proposed system reaches 0.073 street, 0.157 city, and 0.711 country accuracy.

## Constraint: geocomp_efficiency
Inference overhead should remain modest. A scientist should not win the benchmark only by producing much longer outputs or spending substantially more inference time per image.
- **Metric**: avg_generated_tokens and avg_inference_time_s (lower is better, treated as constraint metrics)
- **Reference target**: the proposed system averages 173.28 generated tokens and 8.88 seconds per example, compared with 141.58 tokens and 7.57 seconds for a simpler explanatory baseline.

## Diagnostic: ablation_explanation_scaffolds
Contribution of progressively richer explanation scaffolds.
- **Metric**: city/country/continent accuracy, recall, and F1 (higher is better when all useful scaffolding is present)
- **Reference target**: the full proposed configuration reaches 0.118 city accuracy on the paper's main test set.
