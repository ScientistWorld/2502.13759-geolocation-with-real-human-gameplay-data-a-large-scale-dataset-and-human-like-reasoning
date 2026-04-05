"""
GeoCoT (Geographical Chain-of-Thought) implementation.

This module implements the GeoCoT prompting framework from the paper:
"Geolocation with Real Human Gameplay Data" (arXiv: 2502.13759)

GeoCoT is a structured prompting strategy that guides VLMs through five
sequential reasoning steps for image geolocation:
  1. Continental/Climate Zone Identification
  2. Country-Level Localization
  3. City-Level Refinement
  4. Landmark-Based Verification
  5. Fine-Grained Micro-Level Validation

No training is required — this is purely a prompting strategy applied to
off-the-shelf Vision Language Models.
"""

from .prompt_template import GEO_COT_SYSTEM_PROMPT, GEO_COT_USER_PROMPT

__all__ = ["GEO_COT_SYSTEM_PROMPT", "GEO_COT_USER_PROMPT", "GeoCoTEngine"]
