# Experiment Overview

This file describes the scientific purpose of each experiment in `scores.json`. The headings exactly match the experiment-name keys.

## geocomp_classification
Tests geolocation prediction accuracy at three geographical granularity levels (city, country, continent) on a 500-image test set. Compares a structured prompting strategy against standard chain-of-thought prompting using the same vision-language model.

## geocomp_distance
Tests distance-based geolocation accuracy: the fraction of predictions that fall within 1km (street level), 25km (city level), and 750km (country level) of the ground truth location. Evaluates on the same 500-image test set.

## im2gps_generalization
Tests whether the prompting strategy generalizes to a different dataset (Im2GPS/Im2GPS3K). Measures distance-based accuracy on this held-out dataset to evaluate robustness across geographic distribution shifts.

## ablation_geocot_steps
Tests the contribution of individual reasoning steps by progressively adding them. Compares performance with 1 step, 2 steps, 3 steps, and all 5 steps to determine whether each additional step provides incremental benefit.
