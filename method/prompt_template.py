"""
GeoCoT Prompt Templates.

Implements the five-step geographical chain-of-thought prompting from the paper.
"""

GEO_COT_SYSTEM_PROMPT = """You are an expert geolocation analyst. Your task is to determine the geographical location (continent, country, and city) of a given image through careful visual analysis and multi-step reasoning.

Think step by step about the geographical clues visible in the image. Consider natural features, cultural markers, architectural styles, infrastructure elements, and other visually distinguishable details that can help narrow down the location.

Output your final answer in this format:
Location: [City], [Country], [Continent]

For example:
Location: Bangkok, Thailand, Asia
"""


GEO_COT_USER_PROMPT = """Analyze this image and determine its geographical location using the following structured reasoning steps:

Step 1 - Continental/Climate Zone Identification:
Are there prominent natural features, such as specific types of vegetation, landforms (e.g., mountains, hills, plains), or soil characteristics, that provide clues about the geographical region?

Step 2 - Country-Level Localization:
Are there any culturally, historically, or architecturally significant landmarks, buildings, or structures, or are there any inscriptions or signs in a specific language or script that could help determine the country or region?

Step 3 - City-Level Refinement:
Are there distinctive road-related features, such as traffic direction (e.g., left-hand or right-hand driving), specific types of bollards, unique utility pole designs, or license plate colors and styles, which countries are known to have these characteristics?

Step 4 - Landmark-Based Verification:
Are there observable urban or rural markers (e.g., street signs, fire hydrants, guideposts), or other infrastructure elements, that can provide more specific information about the country or city?

Step 5 - Fine-Grained Micro-Level Validation:
Are there identifiable patterns in sidewalks (e.g., tile shapes, colors, or arrangements), clothing styles worn by people, or other culturally specific details that can help narrow down the city or area?

Let's think step by step. Based on the questions I provided, locate the location of the picture as accurately as possible. Identify the continent, country, and city, and summarize it into a paragraph. For example: the presence of tropical rainforests, palm trees, and red soil indicates a tropical climate... Signs in Thai, right-side traffic, and traditional Thai architecture further suggest it is in Thailand... Combining these clues, this image was likely taken in a city in Thailand, Asia.

Please provide your analysis and final location prediction.
"""


def build_geocot_prompt(image_path: str = None) -> str:
    """
    Build the full GeoCoT prompt for a given image.

    Args:
        image_path: Path to the image file (used for VLM input)

    Returns:
        The user prompt string for GeoCoT analysis
    """
    return GEO_COT_USER_PROMPT


# Standard chain-of-thought prompt for comparison baseline
COT_PROMPT = """Let's think step by step about the geographical location where this image was taken. Identify any visible clues about the location, then provide your best guess of the city, country, and continent.

Output format: Location: [City], [Country], [Continent]
"""


def build_baseline_prompt(image_path: str = None) -> str:
    """Build a standard chain-of-thought prompt (no GeoCoT structure)."""
    return COT_PROMPT
