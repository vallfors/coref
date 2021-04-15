import json
from typing import List

# A configuration for the coreference resolution.
# This object should contain information about which algorithm to run,
# and any specific settings for the algorithm. The config is read from a json file.
class Config:
    useGoldMentions: bool
    algorithm: str
    inputFile: str
    trainingInputFile: str
    wordVectorFile: str
    useAllDocs: bool
    docId: int
    multipassSieves: List[str]
    writeForScoring: bool
    debugMentionDetection: bool
    compareClusters: bool
    maxDepth: int
    debugFeatureSelection: bool
    minimalMutualInformation: float
    allowedFeatureRarity: int

    def __init__(self, filename: str):
        with open(filename) as f:
            configDict = json.load(f)
        self.useGoldMentions = configDict['useGoldMentions']
        self.algorithm = configDict['algorithm']
        self.inputFile = configDict['inputFile']
        self.useAllDocs = configDict['useAllDocs']
        if not self.useAllDocs:
            self.docId = configDict['docId']
        if configDict['algorithm'] == 'multipass':
            self.multipassSieves = configDict['multipassSieves']
        if configDict['algorithm'] == 'hcoref':
            self.scaffoldingSieves = configDict['scaffoldingSieves']
            self.wordVectorFile = configDict['wordVectorFile']
            self.debugFeatureSelection = configDict['debugFeatureSelection']
        self.writeForScoring = configDict['writeForScoring']
        self.debugMentionDetection = configDict['debugMentionDetection']
        self.compareClusters = configDict['compareClusters']
        if 'trainingInputFile' in configDict:
            self.trainingInputFile = configDict['trainingInputFile']
            self.maxDepth = configDict['maxDepth']
            self.minimalMutualInformation = configDict['minimalMutualInformation']
            self.allowedFeatureRarity = configDict['allowedFeatureRarity']
