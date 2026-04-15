# Problem Definition

## Research Question

Image geolocation is the task of determining where a ground-level image was taken. A solution must infer the continent, country, and city from visual evidence such as architecture, road signs, natural landscape, vegetation, terrain, language, infrastructure, and cultural markers.

The benchmark focuses on fine-grained localization rather than only coarse regional prediction. It also values explanations that identify image evidence supporting the predicted location, because many geolocation errors come from plausible but unsupported visual interpretations.

## Why It Matters

Accurate image geolocation has important applications in crime tracking, navigation, fact-checking, and cultural exploration. A model that can accurately determine the location of an image from visual cues alone would be valuable for many real-world applications. Furthermore, geolocation serves as a benchmark for evaluating the reasoning and world knowledge capabilities of vision models.

## Success Criteria

A successful solution should:
- Predict geographical locations (continent, country, city) from street-view or ground-level images
- Achieve high accuracy at multiple levels of geographical granularity (continent, country, city)
- Produce interpretable, image-grounded reasoning when textual explanations are emitted
- Generalize to diverse geographic regions and image types
- Be evaluated using distance-based metrics (e.g., fraction of predictions within 1km, 25km, 750km of ground truth) and classification-based metrics (accuracy, recall, F1 at city/country/continent levels)
