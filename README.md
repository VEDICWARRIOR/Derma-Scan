# DERMA-SCAN: Melanoma Classification using Deep Learning

## Overview
This project implements a deep learning pipeline for binary melanoma classification using the HAM10000 dataset.

The system:
- Loads dermoscopic images
- Performs binary classification (melanoma vs non-melanoma)
- Uses transfer learning (MobileNetV2)
- Outputs confusion matrix and classification report

## Dataset
HAM10000 (ISIC Archive)

Download from:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

Place images inside:
tele_dermatology/dataset/images/

## Installation

```bash
pip install -r requirements.txt