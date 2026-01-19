---
title: AMR Biosafety Dashboard
emoji: 🦠
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.31.1
app_file: amr.py
pinned: false
license: mit
---

# AI-Driven AMR Biosafety Dashboard

This dashboard leverages phylogeny-weighted machine learning to provide enhanced surveillance of Antimicrobial Resistance (AMR).

## Features
- **Phylogeny-Weighted Prediction**: Addresses outbreak bias in surveillance data.
- **Risk Assessment**: Biosafety risk scoring based on resistance trends.
- **Novelty Detection**: Identifies statistical and ML-based anomalies.
- **Explainable AI**: SHAP values for transparent model predictions.
- **Policy Simulation**: Simulate antibiotic substitution scenarios.

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run amr.py
   ```
