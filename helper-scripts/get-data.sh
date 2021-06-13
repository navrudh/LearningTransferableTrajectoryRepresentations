#!/usr/bin/env bash

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPTS_DIR/../data" || exit
wget -nc https://archive.ics.uci.edu/ml/machine-learning-databases/00339/train.csv.zip
7z x "train.csv.zip" -aos
