#!/usr/bin/env python3
"""
Run GeoCoT on a geolocation dataset.

Usage:
    python -m method.run_geocot --dataset im2gps3k --output results/geocot_predictions.json

The script runs GeoCoT prompting on images from a geolocation dataset,
parses the model's location predictions, and saves results.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from method.prompt_template import GEO_COT_USER_PROMPT, build_baseline_prompt
from method.vlm_client import get_vlm_client


COUNTRY_TO_CONTINENT = {
    "kenya": ("Kenya", "Africa"),
    "madagascar": ("Madagascar", "Africa"),
    "ecuador": ("Ecuador", "South America"),
    "chile": ("Chile", "South America"),
    "brazil": ("Brazil", "South America"),
    "argentina": ("Argentina", "South America"),
    "peru": ("Peru", "South America"),
    "colombia": ("Colombia", "South America"),
    "united states": ("United States", "North America"),
    "usa": ("United States", "North America"),
    "us": ("United States", "North America"),
    "canada": ("Canada", "North America"),
    "mexico": ("Mexico", "North America"),
    "germany": ("Germany", "Europe"),
    "france": ("France", "Europe"),
    "united kingdom": ("United Kingdom", "Europe"),
    "uk": ("United Kingdom", "Europe"),
    "england": ("United Kingdom", "Europe"),
    "spain": ("Spain", "Europe"),
    "italy": ("Italy", "Europe"),
    "japan": ("Japan", "Asia"),
    "china": ("China", "Asia"),
    "india": ("India", "Asia"),
    "russia": ("Russia", "Europe"),
    "south korea": ("South Korea", "Asia"),
    "thailand": ("Thailand", "Asia"),
    "australia": ("Australia", "Oceania"),
    "south africa": ("South Africa", "Africa"),
    "egypt": ("Egypt", "Africa"),
    "nigeria": ("Nigeria", "Africa"),
    "turkey": ("Turkey", "Asia"),
    "indonesia": ("Indonesia", "Asia"),
    "uganda": ("Uganda", "Africa"),
    "tanzania": ("Tanzania", "Africa"),
    "morocco": ("Morocco", "Africa"),
}

CONTINENTS = {
    "africa": "Africa",
    "asia": "Asia",
    "europe": "Europe",
    "north america": "North America",
    "south america": "South America",
    "oceania": "Oceania",
    "australia": "Oceania",
}


def parse_location_prediction(text: str) -> Dict[str, Optional[str]]:
    """
    Parse a location prediction from model output text.

    Looks for the pattern "Location: [City], [Country], [Continent]"
    and extracts the components.

    Returns:
        Dict with keys: city, country, continent, raw_prediction
    """
    result = {"city": None, "country": None, "continent": None, "raw_prediction": text}

    if not text:
        return result

    # Try explicit "Location: City, Country, Continent" style outputs first.
    patterns = [
        r"Location:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"location:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"Location\s*(?:is\s*)?:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"Predicted\s*Location:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"(?:taken|located)\s+in\s+(.+?),\s*(.+?),\s*(Africa|Asia|Europe|North America|South America|Oceania)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["city"] = match.group(1).strip()
            result["country"] = match.group(2).strip()
            result["continent"] = match.group(3).strip()
            return result

    # GeoCoT often returns a paragraph rather than a strict label. Fall back to
    # country and continent mentions so the reusable runner can score real VLM
    # responses without requiring method-specific evaluation code.
    text_lower = text.lower()
    for key, (country, continent) in sorted(COUNTRY_TO_CONTINENT.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(key)}\b", text_lower):
            result["country"] = country
            result["continent"] = continent
            break
    if result["continent"] is None:
        for key, continent in CONTINENTS.items():
            if re.search(rf"\b{re.escape(key)}\b", text_lower):
                result["continent"] = continent
                break

    return result


def run_geocot(
    dataset_path: str,
    output_path: str,
    vlm_type: str = "auto",
    max_images: Optional[int] = None,
    batch_size: int = 1,
    method: str = "geocot",
) -> List[Dict]:
    """
    Run GeoCoT on a geolocation dataset.

    Args:
        dataset_path: Path to the dataset (CSV with image_path, lat, lon, city, country, continent)
        output_path: Path to save predictions
        vlm_type: VLM client type ("gpt4o", "qwen_vl", "llava", "auto")
        max_images: Maximum number of images to process (None for all)
        batch_size: Number of images to process in a batch
        method: "geocot" or "cot" (standard chain-of-thought)

    Returns:
        List of prediction results
    """
    # Load VLM client
    print(f"Initializing VLM client ({vlm_type})...")
    client = get_vlm_client(vlm_type)

    # Load dataset
    import pandas as pd
    df = pd.read_csv(dataset_path)
    if max_images:
        df = df.head(max_images)

    print(f"Processing {len(df)} images with method={method}...")

    # Select prompt
    prompt = GEO_COT_USER_PROMPT if method == "geocot" else build_baseline_prompt()

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="GeoCoT"):
        image_path = row["image_path"]
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue

        try:
            response = client.predict(image_path, prompt)
            prediction = parse_location_prediction(response)

            result = {
                "idx": idx,
                "image_path": image_path,
                "ground_truth_city": row.get("city", None),
                "ground_truth_country": row.get("country", None),
                "ground_truth_continent": row.get("continent", None),
                "ground_truth_lat": row.get("lat", None),
                "ground_truth_lon": row.get("lon", None),
                "predicted_city": prediction["city"],
                "predicted_country": prediction["country"],
                "predicted_continent": prediction["continent"],
                "model_response": response,
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                "idx": idx,
                "image_path": image_path,
                "ground_truth_city": row.get("city", None),
                "ground_truth_country": row.get("country", None),
                "ground_truth_continent": row.get("continent", None),
                "ground_truth_lat": row.get("lat", None),
                "ground_truth_lon": row.get("lon", None),
                "predicted_city": None,
                "predicted_country": None,
                "predicted_continent": None,
                "model_response": None,
                "error": str(e),
            })

        # Save intermediate results
        if idx > 0 and idx % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    # Save final results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run GeoCoT on geolocation dataset")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset CSV or name (im2gps3k)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save predictions JSON")
    parser.add_argument("--vlm", type=str, default="auto",
                        choices=["gpt4o", "qwen_vl", "llava", "auto"],
                        help="VLM client type")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to process")
    parser.add_argument("--method", type=str, default="geocot",
                        choices=["geocot", "cot"],
                        help="Prompting method: geocot or cot")
    args = parser.parse_args()

    results = run_geocot(
        dataset_path=args.dataset,
        output_path=args.output,
        vlm_type=args.vlm,
        max_images=args.max_images,
        method=args.method,
    )

    # Print summary
    successful = sum(1 for r in results if r.get("predicted_city") is not None)
    print(f"\nSummary: {successful}/{len(results)} predictions generated successfully")


if __name__ == "__main__":
    main()
