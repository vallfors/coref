#!/bin/sh
python3 train.py config/config.json
python3 main.py config/config.json
perl ../reference-coreference-scorers/scorer.pl bcub evaluation/goldClusters/all evaluation/predictedClusters/all none