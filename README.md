# Fairness Metric Disagreement in Machine Learning

This repository accompanies the paper:

**When Fairness Metrics Disagree: Evaluating the Reliability of Demographic Fairness Assessment in Machine Learning**

## Overview

This project investigates whether commonly used fairness metrics produce consistent conclusions when applied to the same machine learning model. It introduces the **Fairness Disagreement Index (FDI)**, a quantitative measure of disagreement across fairness metrics.

The repository includes:
- pair construction from LFW
- embedding extraction using FaceNet
- embedding extraction using ArcFace
- similarity scoring
- fairness metric evaluation across thresholds
- FDI computation
- plotting utilities for figures

## Repository Structure

- `src/` — source scripts for the full pipeline
- `data/raw/` — expected location for raw datasets
- `data/metadata/` — metadata files such as constructed pairs
- `outputs/tables/` — generated result tables
- `outputs/figures/` — generated figures

## Datasets

Raw datasets are **not included** in this repository.

Expected dataset locations:
- `data/raw/lfw/`
- `data/raw/rfw/`
- `data/raw/rfw_pairs/`

This project’s final experimental pipeline is based on LFW with proxy-defined group partitions.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt

