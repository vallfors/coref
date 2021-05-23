#!/bin/sh
python3 train.py config/predictedML.json
python3 main.py config/predictedML.json
perl ../reference-coreference-scorers/scorer.pl bcub evaluation/goldClusters/all evaluation/predictedClusters/all none