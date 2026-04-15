"""
GeoCoT Prompt Templates.

Implements the five-step geographical chain-of-thought prompting from the paper.
Paper reference: Appendix B of "Geolocation with Real Human Gameplay Data" (arXiv: 2502.13759)
"""

# GeoCoT: Full 5-step prompt from paper's Appendix B
# This is the paper's actual method - 5 sequential reasoning steps
GEO_COT_PROMPT = """Question 1: Are there prominent natural features, such as specific types of vegetation, landforms (e.g., mountains, hills, plains), or soil characteristics, that provide clues about the geographical region?

Question 2: Are there any culturally, historically, or architecturally significant landmarks, buildings, or structures, or are there any inscriptions or signs in a specific language or script that could help determine the country or region?

Question 3: Are there distinctive road-related features, such as traffic direction (e.g., left-hand or right-hand driving), specific types of bollards, unique utility pole designs, or license plate colors and styles, which countries are known to have these characteristics?

Question 4: Are there observable urban or rural markers (e.g., street signs, fire hydrants, guideposts), or other infrastructure elements, that can provide more specific information about the country or city?

Question 5: Are there identifiable patterns in sidewalks (e.g., tile shapes, colors, or arrangements), clothing styles worn by people, or other culturally specific details that can help narrow down the city or area?

Let's think step by step. Based on the question I provided, locate the location of the picture as accurately as possible. Identify the continent, country, and city, and summarize it into a paragraph. For example: the presence of tropical rainforests, palm trees, and red soil indicates a tropical climate, most likely located in Southeast Asia. The signs are in Thai and the hydrant is a green circle, indicating that it is in Thailand. Right-side traffic, cylindrical bollards with blue markings and license plates with black lettering on a white background meet Thai standards. Traditional Thai architecture, such as pitched roofs and wooden structures, further points to a specific location in Thailand. Square gray sidewalk tiles and non-motorized lanes marked with red asphalt are specific urban design features that help narrow down the location. Combining tropical vegetation, Thai-language road signs, traditional architecture, and specific urban design features, this image was most likely taken in a city in Bangkok, Thailand, Asia."""

# Standard CoT baseline prompt (from paper's inference_example.py)
COT_PROMPT = """Analyze the following image. Use geographic elements such as landmarks, architecture, language, or other visual clues to determine the location.
If you cannot determine the location, please guess the answer from continent to city. Do not answer you cannot determine the location.
Output the chain of reasoning."""

# Direct prediction (no reasoning) - for comparison
DIRECT_PROMPT = """What is the geographical location shown in this image? Provide the city, country, and continent."""


# Keep backward-compatible aliases
GEO_COT_SYSTEM_PROMPT = ""
GEO_COT_USER_PROMPT = GEO_COT_PROMPT


def build_geocot_prompt(image_path: str = None) -> str:
    """
    Build the full GeoCoT prompt for a given image.

    Args:
        image_path: Path to the image file (used for VLM input)

    Returns:
        The user prompt string for GeoCoT analysis
    """
    return GEO_COT_PROMPT


def build_baseline_prompt(image_path: str = None) -> str:
    """Build a standard chain-of-thought prompt (no GeoCoT structure)."""
    return COT_PROMPT
