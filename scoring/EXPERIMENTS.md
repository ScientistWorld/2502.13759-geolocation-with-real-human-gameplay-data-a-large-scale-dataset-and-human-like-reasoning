# Experiment Overview

This file describes the scientific purpose of each experiment in `scores.json`. The headings exactly match the experiment-name keys.

## geocomp_classification
Tests geolocation prediction quality at the available geographical granularity levels. The paper reference covers city, country, and continent labels; the local reproduced sample has country and continent labels, so `scores.json` omits city-level classification metrics instead of treating missing city labels as errors. Accuracy, recall, and F1 are reported for each available level so improvements cannot be reduced to a single matching-rate metric.

## geocomp_distance
Tests distance-based geolocation accuracy: the fraction of predictions that fall within 1km (street level), 25km (city level), and 750km (country level) of the ground truth coordinates. The split also reports `median_error_km` so future agents cannot game the benchmark when the thresholded metrics saturate near zero. Missing predicted coordinates count as failures when ground-truth coordinates are available.

## geocomp_efficiency
Measures the inference-time cost of producing geolocation reasoning. This guards against trivial improvements that only work by generating much longer outputs or using substantially more inference time.

## ablation_explanation_scaffolds
Tests controlled variants of the reproduced geolocation system under the same classification metrics. This diagnostic checks whether changes that appear useful in the main comparison also change localization outcomes under closely related settings.
