#!/usr/bin/env bash
# Download the dataset (CC0-1.0) from Kaggle. Requires ~/.kaggle/kaggle.json or
# KAGGLE_USERNAME / KAGGLE_KEY in the environment.
set -euo pipefail
mkdir -p data
kaggle datasets download -d umeradnaan/remote-sensing-satellite-images -p data --unzip
rm -f data/*.zip
ls data
