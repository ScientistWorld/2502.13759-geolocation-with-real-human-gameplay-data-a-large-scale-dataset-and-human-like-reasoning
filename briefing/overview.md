# Overview

- **Paper ID:** 2502.13759
- **Title:** Geolocation with Real Human Gameplay Data: A Large-Scale Dataset and Human-Like Reasoning Framework
- **Domain:** Computer Vision / Geolocation / Vision-Language Models
- **TL;DR:** Introduces GeoCoT, a geographical chain-of-thought prompting framework that guides vision-language models through five structured reasoning steps to predict image locations, improving accuracy by up to 25% over baselines.

## Short Summary

This paper addresses the image geolocation task — predicting where a photo was taken from visual cues — using a novel prompting framework called GeoCoT (Geographical Chain-of-Thought). Unlike prior approaches that use classification into grid cells or retrieval from image databases, GeoCoT guides a Large Vision Model through five sequential reasoning steps: (1) identifying the continent/climate zone from natural features, (2) narrowing to country using cultural markers, (3) refining to city via infrastructure cues, (4) verifying with landmarks, and (5) confirming with micro-level details. The method requires no training — it is purely a prompting strategy applied to off-the-shelf VLMs. Evaluated on a 500-image test set across 20 countries, GeoCoT achieves the best results across all nine classification metrics (accuracy, recall, F1 at city/country/continent levels) and all three distance thresholds (1km, 25km, 750km).

## Key Results

- **City-level accuracy**: GeoCoT (0.118) vs GPT-4o(CoT) (0.094) vs LLaVA-1.6 (0.002)
- **Country-level accuracy**: GeoCoT (0.640) vs GPT-4o(CoT) (0.623) vs LLaVA-1.6 (0.041)
- **Street-level distance (1km)**: GeoCoT (0.073) vs GPT-4o (0.045) vs GeoCLIP (0.035)
- **Country-level distance (750km)**: GeoCoT (0.711) vs GPT-4o(CoT) (0.701) vs LLaVA-1.6 (0.082)
