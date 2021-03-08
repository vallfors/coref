import json

# A configuration for the coreference resolution.
# This object should contain information about which algorithm to run,
# and any specific settings for the algorithm. The config is read from a json file.
class Config:
    useGoldMentions: bool
    algorithm: str
    inputFile: str

    def __init__(self, filename: str):
        with open(filename) as f:
            configDict = json.load(f)
        self.useGoldMentions = configDict['useGoldMentions']
        self.algorithm = configDict['algorithm']
        self.inputFile = configDict['inputFile']
