from preprocessing import config
from preprocessing.document import Document, Mention
from preprocessing.config import Config
from hcoref.feature_vector import getFeatureVector, getStringFeatureVector

import operator
from typing import List
import joblib
from gensim.models import KeyedVectors
import numpy as np

class Sieve:
    name: str
    sentenceLimit: int
    threshold: float
    model = None
    encoder = None
    selectedFeatures: List[int]
    infoDict = {}

    def __init__(self, name, sentenceLimit, threshold, model, encoder, selectedFeatures, infoDict):
        self.name = name
        self.sentenceLimit = sentenceLimit
        self.threshold = threshold
        self.model = model
        self.encoder = encoder
        self.selectedFeatures = selectedFeatures
        self.infoDict = infoDict

def useSieve(config: Config, sieve, doc: Document, wordVectors, mention: Mention, antecedents: List[Mention]) -> float:
    if sieve.infoDict['mentionPos'] != 'ANY' and mention.features.upos != sieve.infoDict['mentionPos']:
        return None
    mentionDistance = 0
    featureVectors = []
    if len(antecedents) == 0:
        return None
    results = []
    for antecedent in antecedents:
        nonStringFeatures = getFeatureVector(doc, wordVectors, mention, antecedent, mentionDistance, config.features)
        stringFeatures = getStringFeatureVector(doc, wordVectors, mention, antecedent, mentionDistance, config.stringFeatures)
        stringFeatures = sieve.encoder.transform([stringFeatures]).toarray()
        featureVector = np.concatenate((nonStringFeatures, stringFeatures[0]), 0)
        mentionDistance += 1
        selectedFeatures = []
        for i in range(0, len(featureVector)):
            if sieve.selectedFeatures[i]:
                selectedFeatures.append(featureVector[i])
        featureVectors.append(selectedFeatures)
    results = sieve.model.predict_proba(featureVectors)
    bestValue = 0.0
    bestAntecedentIndex = -1
    for antecedentIndex in range(0, len(antecedents)):
        if sieve.infoDict['antecedentPos'] != 'ANY' and antecedents[antecedentIndex].features.upos != sieve.infoDict['antecedentPos']:
            continue
        if results[antecedentIndex][1] > bestValue:
            bestValue = results[antecedentIndex][1]
            bestAntecedentIndex = antecedentIndex
    if bestValue > sieve.threshold and bestAntecedentIndex != -1:
        return antecedents[bestAntecedentIndex]
    else:
        return None

# Moves all mentions in the mention cluster to the antecedent cluster.
def link(doc: Document, mention: Mention, antecedent: Mention):
    mentionCluster = mention.predictedCluster
    antecedentCluster = antecedent.predictedCluster
    if mentionCluster == antecedentCluster:
        return
    for m in doc.predictedClusters[mentionCluster]:
        doc.predictedMentions[m].predictedCluster = antecedentCluster
    doc.predictedClusters[antecedentCluster] += doc.predictedClusters[mentionCluster]
    del doc.predictedClusters[mentionCluster]


# Returns an ordered list of candidate antecedents for a mention,
# in the order they should be considered in.
# TODO: Should use the tree structure for the previous two sentences
def getCandidateAntecedents(doc: Document, mention: Mention, sieve: Sieve) -> List[Mention]:
    x = list(doc.predictedMentions.values())
    x.sort(key=operator.attrgetter('startPos'))
    antecedents = []
    for idx, a in enumerate(x):
        if mention.stanzaSentence - a.stanzaSentence >= sieve.infoDict['sentenceLimit']:
            continue
        if a.stanzaSentence >= mention.stanzaSentence:
            break
        antecedents.append(a)
    sameSentence = []
    while idx < len(x) and x[idx].stanzaSentence == mention.stanzaSentence:
        if x[idx].id == mention.id:
            break
        sameSentence.append(x[idx])
        idx += 1
    antecedents.reverse()
    antecedents = sameSentence + antecedents
    
    return antecedents

def doSievePasses(config: Config, doc: Document, wordVectors, sieves: List[Sieve]):
    for sieve in sieves:
        for mention in doc.predictedMentions.values():
            candidateAntecedents = getCandidateAntecedents(doc, mention, sieve)
            bestAntecedent = useSieve(config, sieve, doc, wordVectors, mention, candidateAntecedents)
            if bestAntecedent != None:
                link(doc, mention, bestAntecedent)

def scaffoldingAlgorithm(doc: Document, config: Config): 
    sieves = []
    for s in config.scaffoldingSieves:
        name = s['name']
        model = joblib.load(f'models/{name}Model.joblib')
        encoder = joblib.load(f'models/{name}OneHotEncoder.joblib')
        selectedFeatures = joblib.load(f'models/{name}SelectedFeatures.joblib')
        sieves.append(Sieve(s['name'], s['sentenceLimit'], s['threshold'], model, encoder, selectedFeatures, s))
    # Create a cluster for each mention, containing only that mention
    doc.predictedClusters = {}
    for mention in doc.predictedMentions.values():
         doc.predictedClusters[mention.id] = [mention.id]
         mention.predictedCluster = mention.id

    doc.eligibleMentions = []
    for mention in doc.predictedMentions.values():
        doc.eligibleMentions.append(mention)
    # Probably should have some different ordering
    doc.eligibleMentions.sort(key=operator.attrgetter('startPos'))
    wordVectors = KeyedVectors.load_word2vec_format(config.wordVectorFile, binary=True)
    doSievePasses(config, doc, wordVectors, sieves)
