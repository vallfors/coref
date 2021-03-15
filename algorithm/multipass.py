from preprocessing.document import Document, Mention
from preprocessing.config import Config

import operator
from typing import List

# Pass 1 from Raghunathan et al. 
def exactMatch(mention: Mention, candidateAntecedent: Mention) -> bool:
    return mention.text == candidateAntecedent.text 

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
    for a in x:
        if a.startPos >= mention.startPos:
            break
        antecedents.append(a)
    return antecedents

# Takes first possible antecedent that satisfies the sieve. Return value indicates whether any
# antecedent was found.
def pickAntecedent(doc: Document, sieve,  mention: Mention) -> bool:
    for antecedent in getCandidateAntecedents(doc, mention):
        if sieve(mention, antecedent):
            link(doc, mention, antecedent)
            return True
    return False

def doSievePass(doc: Document, sieve):
    newEligibleMentions = []
    for mention in doc.eligibleMentions:
        antecedentFound = pickAntecedent(doc, sieve, mention)
        if not antecedentFound:
            newEligibleMentions.append(mention)
    doc.eligibleMentions = newEligibleMentions

def multiPassSieve(doc: Document, sieves):
    for sieve in sieves:
        doSievePass(doc, sieve)

def multiPass(doc: Document, config: Config):
    sieves = [exactMatch]
    
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
    multiPassSieve(doc, sieves)
