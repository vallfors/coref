from preprocessing.document import Document, Mention
from preprocessing.config import Config
from hcoref.hcoref import getFeatureVector

import operator
from typing import List
import joblib

def useSieve(sieveModel, doc: Document, mention: Mention, antecedent: Mention) -> float:
    featureVector = getFeatureVector(doc, mention, antecedent)
    results = sieveModel.predict_proba([featureVector])
    return results[0][1]

# Moves all mentions in the mention cluster to the antecedent cluster.
def link(doc: Document, mention: Mention, antecedent: Mention):
    mentionCluster = mention.cluster
    antecedentCluster = antecedent.cluster
    if mentionCluster == antecedentCluster:
        return
    for m in doc.predictedClusters[mentionCluster]:
        doc.predictedMentions[m].cluster = antecedentCluster
    doc.predictedClusters[antecedentCluster] += doc.predictedClusters[mentionCluster]
    del doc.predictedClusters[mentionCluster]


# Returns an ordered list of candidate antecedents for a mention,
# in the order they should be considered in.
# TODO: Should use the tree structure for the previous two sentences
def getCandidateAntecedents(doc: Document, mention:Mention) -> List[Mention]:
    x = list(doc.predictedMentions.values())
    x.sort(key=operator.attrgetter('startPos'))
    antecedents = []
    for idx, a in enumerate(x):
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

def doSievePasses(doc: Document, sieves):
    threshold = 0.5
    for sieve in sieves:
        for mention in doc.predictedMentions.values():
            bestValue = 0.0
            bestAntecedent = None
            for candidateAntecedent in getCandidateAntecedents(doc, mention):
                val = useSieve(sieve, doc, mention, candidateAntecedent)
                if val > bestValue:
                    bestValue = val
                    bestAntecedent = candidateAntecedent
            if bestValue > threshold and bestAntecedent != None:
                link(doc, mention, bestAntecedent)

def scaffoldingAlgorithm(doc: Document, config: Config):
    testSieveModel = joblib.load('testsieve.joblib') 
    sieveMapping = {'testSieve': testSieveModel}
    sieves = []
    for s in config.multipassSieves:
        if not s in sieveMapping:
            raise Exception(f'Invalid multipass sieve name: {s}')
        sieves.append(sieveMapping[s])
    # Create a cluster for each mention, containing only that mention
    doc.predictedClusters = {}
    for mention in doc.predictedMentions.values():
         doc.predictedClusters[mention.id] = [mention.id]
         mention.cluster = mention.id

    doc.eligibleMentions = []
    for mention in doc.predictedMentions.values():
        doc.eligibleMentions.append(mention)
    # Probably should have some different ordering
    doc.eligibleMentions.sort(key=operator.attrgetter('startPos'))
    doSievePasses(doc, sieves)
