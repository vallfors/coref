#!/bin/sh
python3 train.py config/config_lite.json
python3 main.py config/config_lite.json
perl ../reference-coreference-scorers/scorer.pl bcub evaluation/goldClusters/all evaluation/predictedClusters/all none