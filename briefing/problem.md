# Problem Definition

What problem does this paper address? Write this so that someone with no knowledge of the paper can understand what needs to be solved and how success is measured.

Do NOT describe the paper's method or solution here — only the problem itself. Think about if a future scientist agent tries to solve this problem from scratch without knowing what approach the paper took. If you leak the solution, you compromise the benchmark.

## Research Question

Image geolocation is the task of determining the geographical location (continent, country, city) where an image was taken, based on visual cues in the image. This is a challenging task that requires complex reasoning about contextual clues such as architectural styles, road signs, natural landscapes, vegetation patterns, and cultural markers. Prior approaches have relied on either classification-based methods (partitioning Earth into grid cells and classifying images into cells) or retrieval-based methods (matching images against a database of geo-tagged images). These approaches face limitations in precision, scalability, and interpretability.

## Why It Matters

Accurate image geolocation has important applications in crime tracking, navigation, fact-checking, and cultural exploration. A model that can accurately determine the location of an image from visual cues alone would be valuable for many real-world applications. Furthermore, geolocation serves as a benchmark for evaluating the reasoning and world knowledge capabilities of vision models.

## Success Criteria

A successful solution should:
- Predict geographical locations (continent, country, city) from street-view or ground-level images
- Achieve high accuracy at multiple levels of geographical granularity (continent, country, city)
- Produce interpretable reasoning that explains the prediction based on visual cues
- Generalize to diverse geographic regions and image types
- Be evaluated using distance-based metrics (e.g., fraction of predictions within 1km, 25km, 750km of ground truth) and classification-based metrics (accuracy, recall, F1 at city/country/continent levels)
