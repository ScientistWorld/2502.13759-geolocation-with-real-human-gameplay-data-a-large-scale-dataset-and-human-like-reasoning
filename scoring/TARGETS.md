# Evaluation Targets

## Primary: geolocation_classification
Geolocation accuracy at city, country, and continent levels on the GeoComp test set.
- **Metric**: city_accuracy, country_accuracy, continent_accuracy (higher is better)
- **Paper's result**: GeoCoT achieves 0.118 city, 0.640 country, 0.862 continent accuracy

## Primary: geolocation_distance
Distance-based accuracy: fraction of predictions within 1km (street), 25km (city), 750km (country).
- **Metric**: street_1km, city_25km, country_750km (higher is better)
- **Paper's result**: GeoCoT achieves 0.073 street, 0.157 city, 0.711 country accuracy

## Ablation: reasoning_steps
Contribution of each GeoCoT reasoning step — removing steps should degrade performance.
- **Metric**: city_accuracy, country_accuracy, continent_accuracy (higher is better when all steps present)
- **Paper's result**: Single steps achieve ~0.11 city accuracy; full 5-step GeoCoT achieves 0.118

## Constraint: model_type
The scientist must use a Vision Language Model (VLM) with chain-of-thought reasoning. The method is the prompting strategy, not the model architecture.
- **Metric**: (informational — method must be prompting-based, not fine-tuned models)
