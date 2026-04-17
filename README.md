Author: Khalid Adnan Alsayed

# Fairness Metric Disagreement in Machine Learning

This repository accompanies the paper:

**When Fairness Metrics Disagree: Evaluating the Reliability of Demographic Fairness Assessment in Machine Learning**

## Paper

arXiv: Pending

## Overview

This project investigates whether commonly used fairness metrics produce consistent conclusions when applied to the same machine learning model. It introduces the Fairness Disagreement Index (FDI), a quantitative measure of disagreement across fairness metrics.

## Key Contribution

- Introduces the Fairness Disagreement Index (FDI)
- Quantifies inconsistency across fairness metrics
- Demonstrates that fairness conclusions can vary significantly depending on metric choice

## Repository Structure

- src/ — source scripts for the full pipeline
- data/raw/ — expected location for raw datasets
- data/metadata/ — metadata files such as constructed pairs
- outputs/tables/ — generated result tables
- outputs/figures/ — generated figures

## Datasets

Raw datasets are not included in this repository.

Expected dataset locations:
- data/raw/lfw/
- data/raw/rfw/
- data/raw/rfw_pairs/

This project’s final experimental pipeline is based on LFW with proxy-defined group partitions.

## Installation

Create and activate a virtual environment, then install dependencies:

pip install -r requirements.txt

## How to Run

FaceNet pipeline:

python src/build_pairs.py
python src/extract_embeddings.py
python src/score_pairs.py
python src/evaluate_metrics.py
python src/compute_fdi.py

ArcFace pipeline:

python src/extract_embeddings_arcface.py
python src/score_pairs_arcface.py
python src/evaluate_metrics_arcface.py
python src/compute_fdi_arcface.py

## Outputs

- Fairness metrics across thresholds
- FDI results
- Comparative figures across models

## Citation

@article{alsayed2026fdi,
  title={When Fairness Metrics Disagree: Evaluating the Reliability of Demographic Fairness Assessment in Machine Learning},
  author={Alsayed, Khalid Adnan},
  year={2026}
}

