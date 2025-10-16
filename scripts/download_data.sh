#!/usr/bin/env bash
set -e

mkdir -p data/raw/mri_alz
mkdir -p data/raw/oasis_2d_optional

kaggle datasets download -d jboysen/mri-and-alzheimers -p data/raw/mri_alz --unzip

#  для зовнішнього тесту на потім
# kaggle datasets download -d <owner>/<alzheimer-mri-4classes> -p data/raw/oasis_2d_optional --unzip

echo "Done. Raw data in data/raw/"
