from preprocessing.document import Document, Mention
from preprocessing.config import Config

import operator
from typing import List

# Pass 2 from Lee et al 2013
def exactMatch(doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    return mention.text.lower() == candidateAntecedent.text.lower() 

def lemmaHeadWordMatch(doc: Document, mention: Mention, ca: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    for id in mention.stanzaIds:
        word = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[id-1]
        if word.id != mention.features.headWord and word.upos in ['NOUN', 'PROPN', 'ADJ']:
            return False
    for id in ca.stanzaIds:
        word = doc.stanzaAnnotation.sentences[ca.stanzaSentence].words[id-1]
        if word.id != ca.features.headWord and word.upos in ['NOUN', 'PROPN', 'ADJ']:
            return False
    if mention.features.number != 'UNKNOWN' and ca.features.number != 'UNKNOWN':
        if mention.features.number != ca.features.number:
            return False
    if mention.features.headWordLemma == ca.features.headWordLemma:

        return True
    return False


def headWordMatch(doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if not mention.features.definite:
        return False
    if mention.features.upos == 'PRON':
        return False
    antecedentHeadWord = doc.stanzaAnnotation.sentences[candidateAntecedent.stanzaSentence].words[candidateAntecedent.features.headWord-1].text
    mentionHeadWord = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[mention.features.headWord-1].text
    return antecedentHeadWord.lower() == mentionHeadWord.lower() 

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
    return mentionTextUntilHead.lower() == antecedentTextUntilHead.lower()

# Pass 5 from Lee et al 2013
def strictHeadMatch(doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    headWords = []
    for m in doc.predictedClusters[candidateAntecedent.cluster]:
        headWords.append(doc.stanzaAnnotation.sentences[doc.predictedMentions[m].stanzaSentence].words[doc.predictedMentions[m].features.headWord-1].text)
    mentionHeadWord = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[mention.features.headWord-1].text
    if mentionHeadWord not in headWords:
        return False

    return True

# Pass 10 from Lee et al 2013
def pronounResolution(doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos != 'PRON':
        return False
    if len(mention.stanzaIds) > 1:
        return False
    # Sentence distance is at most 3 for pronoun resolution
    if mention.stanzaSentence - candidateAntecedent.stanzaSentence > 3:
        return False
    if mention.features.number == 'SING' and mention.features.animacy == 'ANIMATE':
        if candidateAntecedent.features.nerTag not in ['UNKNOWN', 'PER']:         
            return False
    if mention.features.number == 'PLU' and mention.features.animacy == 'ANIMATE':
        if candidateAntecedent.features.nerTag not in ['UNKNOWN', 'ORG']:
            return False
    if mention.features.animacy == 'ANIMATE':
        if candidateAntecedent.features.nerTag not in ['UNKNOWN', 'ORG', 'PER']:
            return False
    if candidateAntecedent.features.nerTag  in ['TME', 'EVN']:
        return False
    
    if mention.features.gender != 'UNKNOWN' and candidateAntecedent.features.gender != 'UNKNOWN':
        if mention.features.gender != candidateAntecedent.features.gender:
            return False
    if mention.features.naturalGender != 'UNKNOWN' and candidateAntecedent.features.naturalGender != 'UNKNOWN':
        if mention.features.naturalGender != candidateAntecedent.features.naturalGender:
            return False
    if mention.features.number != 'UNKNOWN' and candidateAntecedent.features.number != 'UNKNOWN':
        if mention.features.number != candidateAntecedent.features.number:
            return False
    if mention.features.animacy != 'UNKNOWN' and candidateAntecedent.features.animacy != 'UNKNOWN':
        if mention.features.animacy != candidateAntecedent.features.animacy:
            return False

    return True

def genetiveResolution(doc: Document, mention: Mention, ca: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    if mention.text[len(mention.text)-1] == 's' and mention.text[:len(mention.text)-1].lower() == ca.text.lower():
        return True
    if ca.text[len(ca.text)-1] == 's' and ca.text[:len(ca.text)-1].lower() == mention.text.lower():
        return True
    return False

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
    sieveMapping = {'exactMatch': exactMatch,'genetiveResolution': genetiveResolution, 'lemmaHeadWordMatch': lemmaHeadWordMatch, 'pronounResolution': pronounResolution}
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
    multiPassSieve(doc, sieves)
