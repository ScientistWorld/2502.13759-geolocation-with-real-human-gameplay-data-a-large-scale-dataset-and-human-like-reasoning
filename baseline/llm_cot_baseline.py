#!/usr/bin/env python3
"""
Simple LLM Chain-of-Thought baseline for geolocation.

This baseline uses a generic CoT prompt without the GeoCoT-specific
geographical reasoning steps. It asks the VLM to predict location directly.
"""

import json
from typing import Dict, List, Optional
from method.prompt_template import COT_PROMPT


def parse_location_prediction(text: str) -> Dict[str, Optional[str]]:
    """
    Parse a location prediction from model output text.

    Looks for "Location: [City], [Country], [Continent]"
    pattern and extracts components.

    Returns:
        Dict with keys: city, country, continent, raw_prediction
    """
    result = {"city": None, "country": None, "continent": None, "raw_prediction": text}

    # Try to find "Location: ..." pattern
    import re
    patterns = [
        r"Location:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"location:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"Location\s*(?:is\s*)?:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
        r"Predicted\s*Location:\s*(.+?),\s*(.+?),\s*(.+?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["city"] = match.group(1).strip()
            result["country"] = match.group(2).strip()
            result["continent"] = match.group(3).strip()
            return result

    return result


def run_baseline(
    dataset_path: str,
    output_path: str,
    vlm_type: str = "auto",
    max_images: Optional[int] = None,
    batch_size: int = 1,
) -> List[Dict]:
    """
    Run CoT baseline on a geolocation dataset.

    Args:
        dataset_path: Path to the dataset (CSV with image_path, lat, lon, city, country, continent)
        output_path: Path to save predictions
        vlm_type: VLM client type ("gpt4o", "qwen_vl", "llava", "auto")
        max_images: Maximum number of images to process (None for all)
        batch_size: Number of images to process in a batch
        method: "cot" (standard chain-of-thought baseline)

    Returns:
        List of prediction results
    """
    import sys
    sys.path.insert(0, "/home/user")
    from method.vlm_client import get_vlm_client

    print(f"Initializing VLM client ({vlm_type}) for baseline...")
    client = get_vlm_client(vlm_type)

    import pandas as pd
    df = pd.read_csv(dataset_path)
    if max_images:
        df = df.head(max_images)

    print(f"Processing {len(df)} images with CoT baseline...")

    results = []
    for idx, row in df.iterrows():
        image_path = row["image_path"]

        try:
            response = client.predict(image_path, COT_PROMPT)
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

        # Save intermediate results every 10 images
        if (idx + 1) % 10 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    # Save final results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run CoT baseline for geolocation")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset CSV (e.g., /home/user/data/geoclip/geoclip.csv)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save predictions JSON")
    parser.add_argument("--vlm", type=str, default="llava",
                        choices=["gpt4o", "qwen_vl", "llava", "auto"],
                        help="VLM client type")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Maximum number of images to process")

    args = parser.parse_args()

    results = run_baseline(
        dataset_path=args.dataset,
        output_path=args.output,
        vlm_type=args.vlm,
        max_images=args.max_images,
    )

    successful = sum(1 for r in results if r.get("predicted_city") is not None)
    print(f"\nSummary: {successful}/{len(results)} predictions generated successfully")
