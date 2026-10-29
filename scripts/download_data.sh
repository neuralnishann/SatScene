#!/usr/bin/env bash
# Download the dataset from Kaggle. The Roboflow export identifies its license as CC BY 4.0.
# Requires ~/.kaggle/kaggle.json or
# KAGGLE_USERNAME / KAGGLE_KEY in the environment.
set -euo pipefail
mkdir -p data
kaggle datasets download -d umeradnaan/remote-sensing-satellite-images -p data --unzip
rm -f data/*.zip
ls data
