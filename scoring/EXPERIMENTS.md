# Experiment Overview

This file describes the scientific purpose of each experiment in `scores.json`. The headings exactly match the experiment-name keys.

## geocomp_classification
Tests geolocation prediction quality at three geographical granularity levels (city, country, continent) on a 500-image test set. Reports accuracy, recall, and F1 so improvements cannot be reduced to a single matching-rate metric.

## geocomp_distance
Tests distance-based geolocation accuracy: the fraction of predictions that fall within 1km (street level), 25km (city level), and 750km (country level) of the ground truth location. Evaluates on the same 500-image test set.

## geocomp_efficiency
Measures the inference-time cost of producing geolocation reasoning. This guards against trivial improvements that only work by generating much longer outputs or using substantially more inference time.

## ablation_explanation_scaffolds
Tests whether adding more structured explanation stages changes geolocation performance. This diagnostic helps distinguish real improvements in visual localization from gains that only appear in the main aggregate metrics.
