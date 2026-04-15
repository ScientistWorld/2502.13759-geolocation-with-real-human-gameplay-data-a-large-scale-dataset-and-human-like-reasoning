# Evaluation Targets

## Primary: geolocation_classification
Geolocation accuracy at city, country, and continent levels on the GeoComp test set.
- **Metric**: city_accuracy, city_recall, city_f1, country_accuracy, country_recall, country_f1, continent_accuracy, continent_recall, continent_f1 (higher is better)
- **Paper's result**: GeoCoT achieves 0.118 city accuracy, 0.640 country accuracy, and 0.862 continent accuracy; the full recall/F1 values are encoded in `scoring/reference.json`.

## Primary: geolocation_distance
Distance-based accuracy: fraction of predictions within 1km (street), 25km (city), 750km (country).
- **Metric**: street_1km, city_25km, country_750km (higher is better)
- **Paper's result**: GeoCoT achieves 0.073 street, 0.157 city, 0.711 country accuracy

## Ablation: reasoning_steps
Contribution of each GeoCoT reasoning step — removing steps should degrade performance.
- **Metric**: city/country/continent accuracy, recall, and F1 (higher is better when all steps present)
- **Paper's result**: Single steps achieve ~0.11 city accuracy; full 5-step GeoCoT achieves 0.118

## Constraint: geocomp_efficiency
Inference overhead should remain modest because the paper's method is a deployment-time prompting strategy with no training or fine-tuning.
- **Metric**: avg_generated_tokens and avg_inference_time_s (lower is better, treated as constraint metrics)
- **Paper's result**: GeoCoT averages 173.28 generated tokens and 8.88 seconds per example, compared with 141.58 tokens and 7.57 seconds for standard CoT.
