import json
from typing import List

# A configuration for the sieve training
class TrainingConfig:
    inputFile: str

    def __init__(self, filename: str):
        with open(filename) as f:
            configDict = json.load(f)
        self.inputFile = configDict['inputFile']
