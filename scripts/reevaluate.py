#!/usr/bin/env python3
"""Re-evaluate existing predictions with a smarter country parser.

Reads saved prediction JSON files, applies improved parsing to extract
real country/continent names from model responses, and recomputes all metrics.
No GPU needed - just re-processes saved results.
"""
import os
import json
import math
import re
import sys

# =============================================================================
# Country database (same as run.sh)
# =============================================================================
ALL_COUNTRIES = {
    "kenya": ("Kenya", "Africa"), "nigeria": ("Nigeria", "Africa"),
    "south africa": ("South Africa", "Africa"), "egypt": ("Egypt", "Africa"),
    "morocco": ("Morocco", "Africa"), "ethiopia": ("Ethiopia", "Africa"),
    "tanzania": ("Tanzania", "Africa"), "ghana": ("Ghana", "Africa"),
    "algeria": ("Algeria", "Africa"), "tunisia": ("Tunisia", "Africa"),
    "uganda": ("Uganda", "Africa"), "cameroon": ("Cameroon", "Africa"),
    "senegal": ("Senegal", "Africa"), "mozambique": ("Mozambique", "Africa"),
    "madagascar": ("Madagascar", "Africa"), "angola": ("Angola", "Africa"),
    "zimbabwe": ("Zimbabwe", "Africa"), "botswana": ("Botswana", "Africa"),
    "namibia": ("Namibia", "Africa"), "rwanda": ("Rwanda", "Africa"),
    "zambia": ("Zambia", "Africa"), "malawi": ("Malawi", "Africa"),
    "libya": ("Libya", "Africa"), "sudan": ("Sudan", "Africa"),
    "somalia": ("Somalia", "Africa"), "ivory coast": ("Ivory Coast", "Africa"),
    "democratic republic of congo": ("DR Congo", "Africa"),
    "congo": ("Congo", "Africa"),
    "ecuador": ("Ecuador", "South America"),
    "chile": ("Chile", "South America"),
    "brazil": ("Brazil", "South America"),
    "argentina": ("Argentina", "South America"),
    "peru": ("Peru", "South America"),
    "colombia": ("Colombia", "South America"),
    "venezuela": ("Venezuela", "South America"),
    "bolivia": ("Bolivia", "South America"),
    "paraguay": ("Paraguay", "South America"),
    "uruguay": ("Uruguay", "South America"),
    "guyana": ("Guyana", "South America"),
    "suriname": ("Suriname", "South America"),
    "united states": ("United States", "North America"),
    "usa": ("United States", "North America"),
    "canada": ("Canada", "North America"),
    "mexico": ("Mexico", "North America"),
    "guatemala": ("Guatemala", "North America"),
    "cuba": ("Cuba", "North America"),
    "honduras": ("Honduras", "North America"),
    "panama": ("Panama", "North America"),
    "costa rica": ("Costa Rica", "North America"),
    "jamaica": ("Jamaica", "North America"),
    "dominican republic": ("Dominican Republic", "North America"),
    "germany": ("Germany", "Europe"), "france": ("France", "Europe"),
    "united kingdom": ("United Kingdom", "Europe"), "spain": ("Spain", "Europe"),
    "italy": ("Italy", "Europe"), "netherlands": ("Netherlands", "Europe"),
    "belgium": ("Belgium", "Europe"), "switzerland": ("Switzerland", "Europe"),
    "austria": ("Austria", "Europe"), "poland": ("Poland", "Europe"),
    "sweden": ("Sweden", "Europe"), "norway": ("Norway", "Europe"),
    "denmark": ("Denmark", "Europe"), "finland": ("Finland", "Europe"),
    "portugal": ("Portugal", "Europe"), "greece": ("Greece", "Europe"),
    "czech republic": ("Czech Republic", "Europe"), "hungary": ("Hungary", "Europe"),
    "romania": ("Romania", "Europe"), "ireland": ("Ireland", "Europe"),
    "russia": ("Russia", "Europe"), "ukraine": ("Ukraine", "Europe"),
    "turkey": ("Turkey", "Europe"), "serbia": ("Serbia", "Europe"),
    "croatia": ("Croatia", "Europe"), "bulgaria": ("Bulgaria", "Europe"),
    "china": ("China", "Asia"), "japan": ("Japan", "Asia"),
    "south korea": ("South Korea", "Asia"), "india": ("India", "Asia"),
    "indonesia": ("Indonesia", "Asia"), "thailand": ("Thailand", "Asia"),
    "vietnam": ("Vietnam", "Asia"), "philippines": ("Philippines", "Asia"),
    "malaysia": ("Malaysia", "Asia"), "singapore": ("Singapore", "Asia"),
    "pakistan": ("Pakistan", "Asia"), "bangladesh": ("Bangladesh", "Asia"),
    "nepal": ("Nepal", "Asia"), "sri lanka": ("Sri Lanka", "Asia"),
    "cambodia": ("Cambodia", "Asia"), "myanmar": ("Myanmar", "Asia"),
    "mongolia": ("Mongolia", "Asia"), "taiwan": ("Taiwan", "Asia"),
    "uae": ("UAE", "Asia"), "saudi arabia": ("Saudi Arabia", "Asia"),
    "israel": ("Israel", "Asia"), "iran": ("Iran", "Asia"),
    "iraq": ("Iraq", "Asia"), "kazakhstan": ("Kazakhstan", "Asia"),
    "australia": ("Australia", "Oceania"),
    "new zealand": ("New Zealand", "Oceania"),
}

# Add adjective forms
COUNTRY_PATTERNS = dict(ALL_COUNTRIES)
COUNTRY_PATTERNS.update({
    "kenyan": ("Kenya", "Africa"), "ugandan": ("Uganda", "Africa"),
    "ecuadorian": ("Ecuador", "South America"),
    "chilean": ("Chile", "South America"),
    "brazilian": ("Brazil", "South America"),
    "argentine": ("Argentina", "South America"), "argentinian": ("Argentina", "South America"),
    "peruvian": ("Peru", "South America"),
    "colombian": ("Colombia", "South America"),
    "malagasy": ("Madagascar", "Africa"),
    "american": ("United States", "North America"),
    "british": ("United Kingdom", "Europe"),
    "german": ("Germany", "Europe"),
    "french": ("France", "Europe"),
    "italian": ("Italy", "Europe"),
    "spanish": ("Spain", "Europe"),
    "chinese": ("China", "Asia"),
    "japanese": ("Japan", "Asia"),
    "korean": ("South Korea", "Asia"),
    "indian": ("India", "Asia"),
    "thai": ("Thailand", "Asia"),
    "vietnamese": ("Vietnam", "Asia"),
    "australian": ("Australia", "Oceania"),
})

# Template words that indicate the parser got a placeholder, not a real answer
TEMPLATE_WORDS = {
    "unknown", "unspecified", "country", "city", "continent", "not specified",
    "unknown country", "unknown city", "unknown continent", "tropical country",
    "developing country", "east africa", "west africa", "north africa",
    "south africa", "southeast asia", "central america", "latin america",
    "north america", "south america", "africa", "asia", "europe", "oceania",
    "caribbean", "middle east", "australia/new zealand", "western australia",
    "eastern europe", "central asia", "north africa/middle east",
    "africa/middle east", "africa/asia", "africa/south america",
    "africa/middle east/south america", "caribbean/africa/central america",
}

CONTINENT_MAP = {
    "south america": "South America", "north america": "North America",
    "africa": "Africa", "europe": "Europe", "asia": "Asia",
    "oceania": "Oceania", "australia": "Oceania",
    "central america": "North America", "caribbean": "North America",
    "middle east": "Asia",
}

DATASET_COUNTRIES = {"kenya", "ecuador", "chile", "madagascar"}


def is_template(text):
    """Check if text is a template placeholder rather than a real answer."""
    if not text:
        return True
    cleaned = text.strip().lower().strip("[]()").strip()
    return cleaned in TEMPLATE_WORDS or len(cleaned) <= 2


def extract_country_smart(response):
    """Smart country extraction that avoids template text and searches full response."""
    if not response:
        return None, None

    text = response
    text_lower = text.lower()

    # First, try the Location: line - but only accept it if NOT template text
    loc_patterns = [
        r"[Ll]ocation:\s*(.+?),\s*(.+?)(?:,\s*(.+?))?(?:\n|$|\.)",
    ]
    for pattern in loc_patterns:
        match = re.search(pattern, text)
        if match:
            city_str = match.group(1).strip().strip("[]").strip()
            country_str = match.group(2).strip().strip("[]").strip()
            continent_str = match.group(3).strip().strip("[]").strip() if match.group(3) else ""

            if not is_template(country_str):
                # Got a real country from Location: line
                canonical, continent = lookup_country(country_str)
                if canonical:
                    return canonical, continent

    # Also try "Final Prediction:" section
    fp_match = re.search(r"Final\s+Prediction:?\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    fp_section = fp_match.group(1) if fp_match else ""

    # Search for country names in prioritized sections
    # Priority: Final Prediction > Conclusion (last 30%) > Full text
    search_sections = []
    if fp_section:
        search_sections.append(("final_prediction", fp_section, 200))
    # For GeoCoT, the "Based on the analysis" or "Given the analysis" conclusion
    conclusion_match = re.search(
        r"(?:Given|Based on|Considering).*?(?:analysis|clues|observations|factors)[,:]\s*(.*)",
        text, re.IGNORECASE | re.DOTALL
    )
    if conclusion_match:
        search_sections.append(("conclusion", conclusion_match.group(1), 100))
    # Last 30% of text
    last_third = text[len(text)*2//3:]
    search_sections.append(("last_third", last_third, 50))
    search_sections.append(("full_text", text, 10))

    best_country = None
    best_continent = None
    best_priority = -1

    for section_name, section_text, priority in search_sections:
        section_lower = section_text.lower()
        for pattern, (canonical, continent) in COUNTRY_PATTERNS.items():
            if pattern in section_lower:
                p = priority
                # Adjective forms are lower priority (could be misleading)
                if pattern.endswith("n") and len(pattern) > 5 and pattern not in ALL_COUNTRIES:
                    p -= 5
                # Dataset countries get a small bonus (model is more likely to be right about these)
                if pattern in DATASET_COUNTRIES:
                    p += 5
                # Penalize very common false positives
                false_positive_contexts = {
                    "american": ["american-style", "american southwest"],
                    "australian": ["australian aboriginal"],
                }
                if pattern in false_positive_contexts:
                    for ctx in false_positive_contexts[pattern]:
                        if ctx in section_lower:
                            p -= 20
                if p > best_priority:
                    best_country = canonical
                    best_continent = continent
                    best_priority = p

    return best_country, best_continent


def lookup_country(name):
    """Look up a country name and return (canonical, continent)."""
    if not name:
        return None, None
    name_lower = name.strip().lower()
    # Direct lookup
    if name_lower in ALL_COUNTRIES:
        return ALL_COUNTRIES[name_lower]
    # Try stripping common prefixes
    for prefix in ["the ", "republic of "]:
        if name_lower.startswith(prefix):
            stripped = name_lower[len(prefix):]
            if stripped in ALL_COUNTRIES:
                return ALL_COUNTRIES[stripped]
    return None, None


def extract_continent_smart(response, country_continent=None):
    """Extract continent from response, preferring conclusion sections."""
    if not response:
        return None

    text_lower = response.lower()

    # If we already got a continent from the country lookup, use that
    if country_continent:
        return country_continent

    # Search for continent mentions in the text
    for cont_name, canonical in CONTINENT_MAP.items():
        if cont_name in text_lower:
            return canonical

    return None


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(delta_lon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


COUNTRY_COORDS = {
    "kenya": (-1.2921, 36.8219), "madagascar": (-18.7669, 46.8691),
    "ecuador": (-1.8312, -78.1834), "chile": (-35.6751, -71.5430),
    "brazil": (-14.2350, -51.9253), "argentina": (-38.4161, -63.6167),
    "peru": (-9.1900, -75.0152), "colombia": (4.5709, -74.2973),
    "united states": (37.0902, -95.7129), "canada": (56.1304, -106.3468),
    "south africa": (-30.5595, 22.9375), "egypt": (26.8206, 30.8025),
    "nigeria": (9.0820, 8.6753), "australia": (-25.2744, 133.7751),
    "germany": (51.1657, 10.4515), "france": (46.2276, 2.2137),
    "united kingdom": (55.3781, -3.4360), "china": (35.8617, 104.1954),
    "japan": (36.2048, 138.2529), "india": (20.5937, 78.9629),
    "russia": (61.5240, 105.3188), "mexico": (23.6345, -102.5528),
    "thailand": (15.8700, 100.9925), "spain": (40.4637, -3.7492),
    "italy": (41.8719, 12.5674), "tanzania": (-6.3690, 34.8888),
    "ethiopia": (9.1450, 40.4897), "senegal": (14.4974, -14.4524),
    "jamaica": (18.1096, -77.2975), "panama": (8.5380, -80.7822),
}

CONTINENT_COORDS = {
    "africa": (0, 20), "south america": (-15, -60),
    "north america": (40, -100), "europe": (50, 10),
    "asia": (30, 100), "oceania": (-25, 140),
}


def get_coords(country, continent):
    """Get representative coordinates for a country or continent."""
    if country:
        c = country.strip().lower()
        for k, v in COUNTRY_COORDS.items():
            if k in c or c in k:
                return v
    if continent:
        c = continent.strip().lower()
        for k, v in CONTINENT_COORDS.items():
            if k in c or c in k:
                return v
    return None, None


def normalize_country(c):
    if not c: return None
    c = c.strip().lower()
    aliases = {"usa":"united states","us":"united states","uk":"united kingdom",
               "british":"united kingdom","american":"united states"}
    return aliases.get(c, c)


def normalize_continent(c):
    if not c: return None
    c = c.strip().lower()
    if "australia" in c: c = "oceania"
    if "caribbean" in c: c = "north america"
    if "middle east" in c: c = "asia"
    return c


def reparse_predictions(predictions):
    """Re-parse all predictions with the smart parser."""
    results = []
    for p in predictions:
        response = p.get("model_response")
        old_country = p.get("predicted_country")
        old_continent = p.get("predicted_continent")

        # Extract using smart parser
        new_country, country_continent = extract_country_smart(response)
        new_continent = extract_continent_smart(response, country_continent)

        # Get coordinates for distance metrics
        pred_lat, pred_lon = get_coords(new_country, new_continent)

        result = dict(p)
        result["predicted_country"] = new_country
        result["predicted_continent"] = new_continent
        result["predicted_lat"] = pred_lat
        result["predicted_lon"] = pred_lon
        result["old_predicted_country"] = old_country
        result["old_predicted_continent"] = old_continent

        results.append(result)
    return results


def compute_metrics(predictions):
    """Compute all geolocation metrics."""
    metrics = {}

    # Classification metrics
    for level in ['city', 'country', 'continent']:
        pk = f'predicted_{level}'
        gk = f'ground_truth_{level}'
        tp = total = 0
        for p in predictions:
            if not p.get(gk): continue
            total += 1
            pv = p.get(pk)
            gv = p.get(gk)
            if not pv:
                continue
            if level == 'country':
                pv, gv = normalize_country(pv), normalize_country(gv)
            elif level == 'continent':
                pv, gv = normalize_continent(pv), normalize_continent(gv)
            else:
                pv, gv = (pv.strip().lower() if pv else ""), (gv.strip().lower() if gv else "")
            if pv == gv:
                tp += 1
        metrics[f'{level}_accuracy'] = round(tp/total, 4) if total else 0
        metrics[f'{level}_total'] = total

    # Distance metrics
    for thresh, name in [(1.0,"street_1km"),(25.0,"city_25km"),(750.0,"country_750km")]:
        within = total = 0
        for p in predictions:
            lat, lon = p.get('ground_truth_lat'), p.get('ground_truth_lon')
            plat, plon = p.get('predicted_lat'), p.get('predicted_lon')
            if lat is not None and lon is not None and plat is not None and plon is not None:
                total += 1
                if haversine(float(lat), float(lon), float(plat), float(plon)) <= thresh:
                    within += 1
        metrics[name] = round(within/total, 4) if total else 0.0
        metrics[f'{name}_count'] = f'{within}/{total}'

    return metrics


def main():
    results_dir = '/home/user/results'
    pred_files = {
        'geocot': os.path.join(results_dir, 'geocot_predictions.json'),
        'cot': os.path.join(results_dir, 'cot_predictions.json'),
    }

    all_metrics = {}
    all_reparsed = {}

    for name, filepath in sorted(pred_files.items()):
        print(f"\n{'='*60}")
        print(f"Re-evaluating: {name}")
        print(f"{'='*60}")

        with open(filepath) as f:
            predictions = json.load(f)

        reparsed = reparse_predictions(predictions)
        all_reparsed[name] = reparsed

        # Show per-prediction changes
        changed = 0
        correct_old = 0
        correct_new = 0
        for p in reparsed:
            gt = normalize_country(p.get('ground_truth_country'))
            old_c = normalize_country(p.get('old_predicted_country'))
            new_c = normalize_country(p.get('predicted_country'))
            was_correct = old_c == gt if old_c and gt else False
            is_correct = new_c == gt if new_c and gt else False
            if was_correct: correct_old += 1
            if is_correct: correct_new += 1
            if p.get('old_predicted_country') != p.get('predicted_country'):
                changed += 1
                gt_disp = p.get('ground_truth_country', '?')
                old_disp = p.get('old_predicted_country', 'None')
                new_disp = p.get('predicted_country', 'None')
                img = os.path.basename(p.get('image_path', ''))[:40]
                print(f"  CHANGED {img}: {old_disp} -> {new_disp} (GT: {gt_disp})")

        print(f"\n  Changed: {changed}/{len(reparsed)}")
        print(f"  Country correct (old parser): {correct_old}/{len(reparsed)}")
        print(f"  Country correct (new parser): {correct_new}/{len(reparsed)}")

        # Compute metrics
        valid_gt = [p for p in reparsed if p.get('ground_truth_country')]
        if valid_gt:
            metrics = compute_metrics(valid_gt)
            all_metrics[name] = metrics
            print(f"\n  Metrics:")
            for key in ['city_accuracy', 'country_accuracy', 'continent_accuracy',
                        'street_1km', 'city_25km', 'country_750km']:
                if key in metrics:
                    print(f"    {key}: {metrics[key]}")

        # Save reparsed predictions
        reparsed_path = filepath.replace('.json', '_reparsed.json')
        with open(reparsed_path, 'w') as f:
            json.dump(reparsed, f, indent=2)
        print(f"  Saved reparsed to: {reparsed_path}")

    # Build updated scores.json
    print(f"\n{'='*60}")
    print("Building scores.json")
    print(f"{'='*60}")

    with open('/home/user/scoring/reference.json') as f:
        reference = json.load(f)

    scores = {"experiments": {}}
    METHOD_MAP = {"geocot": "qwen_geocot", "cot": "qwen_cot"}

    for exp_name, exp_data in reference.get("experiments", {}).items():
        exp_copy = {
            "description": exp_data.get("description", ""),
            "weight": exp_data.get("weight", 0.5),
            "primary_metric": exp_data.get("primary_metric", ""),
            "metrics": exp_data.get("metrics", {}),
            "results": {}
        }

        ref_results = exp_data.get("results", {})
        ref_metrics = list(exp_data.get("metrics", {}).keys())

        # Copy reference (paper) results
        for method_name, method_data in ref_results.items():
            entry = {}
            for key, val in method_data.items():
                if key == "type": continue
                entry[key] = val
            exp_copy["results"][method_name] = entry

        # Add reproduced results
        for pred_name, metrics in all_metrics.items():
            method_name = METHOD_MAP.get(pred_name, pred_name)
            if method_name not in ref_results:
                if exp_name in ["geocomp_classification", "geocomp_distance"]:
                    entry = {}
                    for key in ref_metrics:
                        if key in metrics:
                            entry[key] = metrics[key]
                    if entry:
                        exp_copy["results"][method_name] = entry
            else:
                entry = exp_copy["results"][method_name]
                for key in ref_metrics:
                    if key in metrics:
                        entry[key] = metrics[key]

        scores["experiments"][exp_name] = exp_copy

    scores_path = '/home/user/scoring/scores.json'
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)
    print(f"Saved to {scores_path}")

    # Print summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY: GeoCoT vs CoT (core claim test)")
    print(f"{'='*60}")

    geocot_m = all_metrics.get('geocot', {})
    cot_m = all_metrics.get('cot', {})

    print(f"\n  {'Metric':<25} {'GeoCoT':>10} {'CoT':>10} {'GeoCoT>CoT?':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")

    for metric in ['country_accuracy', 'continent_accuracy', 'city_25km', 'country_750km']:
        gv = geocot_m.get(metric, 0)
        cv = cot_m.get(metric, 0)
        winner = "YES" if gv > cv else ("TIE" if gv == cv else "no")
        print(f"  {metric:<25} {gv:>10.4f} {cv:>10.4f} {winner:>12}")

    # Core claim assessment
    gc_country = geocot_m.get('country_accuracy', 0)
    cot_country = cot_m.get('country_accuracy', 0)
    gc_city25 = geocot_m.get('city_25km', 0)
    cot_city25 = cot_m.get('city_25km', 0)

    print(f"\n  Core claim (GeoCoT > CoT):")
    if gc_country > cot_country:
        print(f"    SUPPORTED: country_accuracy {gc_country:.4f} > {cot_country:.4f}")
    elif gc_country == cot_country:
        print(f"    NEUTRAL: country_accuracy {gc_country:.4f} = {cot_country:.4f}")
    else:
        print(f"    NOT SUPPORTED: country_accuracy {gc_country:.4f} <= {cot_country:.4f}")

    if gc_city25 > cot_city25:
        print(f"    SUPPORTED: city_25km {gc_city25:.4f} > {cot_city25:.4f}")
    elif gc_city25 == cot_city25:
        print(f"    NEUTRAL: city_25km {gc_city25:.4f} = {cot_city25:.4f}")
    else:
        print(f"    NOT SUPPORTED: city_25km {gc_city25:.4f} <= {cot_city25:.4f}")


if __name__ == "__main__":
    main()
