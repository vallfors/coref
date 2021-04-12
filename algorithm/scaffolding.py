from preprocessing.document import Document, Mention
from preprocessing.config import Config
from hcoref.feature_vector import getFeatureVector

import operator
from typing import List
import joblib
from gensim.models import KeyedVectors

class Sieve:
    name: str
    sentenceLimit: int
    threshold: float
    model = None

    def __init__(self, name, sentenceLimit, threshold, model):
        self.name = name
        self.sentenceLimit = sentenceLimit
        self.threshold = threshold
        self.model = model

def useSieve(sieve, doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int) -> float:
    if sieve.name == 'proper':
        if mention.features.upos != 'PROPN' or antecedent.features.upos != 'PROPN':
            return 0.0
    if sieve.name == 'common':
        if mention.features.upos != 'NOUN' or antecedent.features.upos != 'NOUN':
            return 0.0
    if sieve.name == 'properCommon':
        if mention.features.upos != 'NOUN' or antecedent.features.upos != 'PROPN':
            return 0.0 
    if sieve.name == 'pronoun':
        if mention.features.upos != 'PRON' or antecedent.features.upos != 'PRON':
            return 0.0 
    featureVector = getFeatureVector(doc, wordVectors, mention, antecedent, mentionDistance)
    results = sieve.model.predict_proba([featureVector])
    return results[0][1]

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
def getCandidateAntecedents(doc: Document, mention: Mention, sentenceLimit: int) -> List[Mention]:
    x = list(doc.predictedMentions.values())
    x.sort(key=operator.attrgetter('startPos'))
    antecedents = []
    for idx, a in enumerate(x):
        if mention.stanzaSentence - a.stanzaSentence >= sentenceLimit:
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

def doSievePasses(doc: Document, wordVectors, sieves: List[Sieve]):
    for sieve in sieves:
        for mention in doc.predictedMentions.values():
            bestValue = 0.0
            bestAntecedent = None
            mentionDistance = 0
            for candidateAntecedent in getCandidateAntecedents(doc, mention, sieve.sentenceLimit):
                val = useSieve(sieve, doc, wordVectors, mention, candidateAntecedent, mentionDistance)
                mentionDistance += 1
                if val > bestValue:
                    bestValue = val
                    bestAntecedent = candidateAntecedent
            if bestValue > sieve.threshold and bestAntecedent != None:
                link(doc, mention, bestAntecedent)

def scaffoldingAlgorithm(doc: Document, config: Config): 
    sieves = []
    for s in config.scaffoldingSieves:
        name = s['name']
        model = joblib.load(f'models/{name}Model.joblib')
        sieves.append(Sieve(s['name'], s['sentenceLimit'], s['threshold'], model))
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
    wordVectors = KeyedVectors.load_word2vec_format("../model.bin", binary=True)
    doSievePasses(doc, wordVectors, sieves)
