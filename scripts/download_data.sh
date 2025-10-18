#!/usr/bin/env bash
set -e

echo "Downloading datasets..."
mkdir -p data/raw/mri_alz
mkdir -p data/raw/2d_mri
mkdir -p data/raw/oasis_2d_optional

echo "-> jboysen/mri-and-alzheimers (metadata)"
kaggle datasets download -d jboysen/mri-and-alzheimers -p data/raw/mri_alz --unzip

echo "-> marcopinamonti/alzheimer-mri-4-classes-dataset (MRI images)"
kaggle datasets download -d marcopinamonti/alzheimer-mri-4-classes-dataset -p data/raw/2d_mri --unzip

echo "Done. Files saved in data/raw/"