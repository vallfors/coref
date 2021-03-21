from preprocessing.document import Document, Mention
from preprocessing.config import Config

import operator
from typing import List

# Pass 2 from Lee et al 2013
def exactMatch(doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    return mention.text == candidateAntecedent.text 

# Pass 3 from Lee et al 2013
# Returns true if the strings are identical up to and including their headwords.
def relaxedStringMatch(doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    mentionTextUntilHead = ''
    for id in mention.stanzaIds:
        word = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[id-1]
        mentionTextUntilHead += word.text + ' '
        if id == mention.features.headWord:
            break
    antecedentTextUntilHead = ''
    for id in candidateAntecedent.stanzaIds:
        word = doc.stanzaAnnotation.sentences[candidateAntecedent.stanzaSentence].words[id-1]
        antecedentTextUntilHead += word.text + ' '
        if id == candidateAntecedent.features.headWord:
            break
    if mentionTextUntilHead == antecedentTextUntilHead:
        print('Oh wow!')
    return mentionTextUntilHead == antecedentTextUntilHead

# Pass 10 from Lee et al 2013
def pronounResolution(doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos != 'PRON':
        return False
    # Sentence distance is at most 3 for pronoun resolution
    if mention.stanzaSentence - candidateAntecedent.stanzaSentence > 3:
        return False

    if mention.features.gender not in ['UNKNOWN', 'NEU/UTR'] and candidateAntecedent.features.gender not in ['UNKNOWN', 'NEU/UTR']:
        if mention.features.gender != candidateAntecedent.features.gender:
            return False
    if mention.features.number != 'UNKNOWN' and candidateAntecedent.features.number != 'UNKNOWN':
        if mention.features.number != candidateAntecedent.features.number:
            return False
    
    return True

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
    antecedents.reverse()
    return antecedents

# Takes first possible antecedent that satisfies the sieve. Return value indicates whether any
# antecedent was found.
def pickAntecedent(doc: Document, sieve,  mention: Mention) -> bool:
    for antecedent in getCandidateAntecedents(doc, mention):
        if sieve(doc, mention, antecedent):
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
    sieves = [exactMatch, relaxedStringMatch, pronounResolution]
    
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
