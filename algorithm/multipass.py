from preprocessing.document import Document, Mention
from preprocessing.config import Config
from algorithm.candidate_antecedents import getCandidateAntecedents

from typing import List
import operator

# Pass 2 from Lee et al 2013
def exactMatch(config: Config, doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    return mention.text.lower() == candidateAntecedent.text.lower() 

def lemmaHeadWordMatch(config: Config, doc: Document, mention: Mention, ca: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    for id in mention.stanzaIds:
        word = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[id-1]
        if word.id != mention.features.headWordId and word.upos in ['NOUN', 'PROPN', 'ADJ']:
            return False
    for id in ca.stanzaIds:
        word = doc.stanzaAnnotation.sentences[ca.stanzaSentence].words[id-1]
        if word.id != ca.features.headWordId and word.upos in ['NOUN', 'PROPN', 'ADJ']:
            return False
    if mention.features.number != 'UNKNOWN' and ca.features.number != 'UNKNOWN':
        if mention.features.number != ca.features.number:
            return False
    if mention.features.headWordLemma == ca.features.headWordLemma:

        return True
    return False


def headWordMatch(config: Config, doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if not mention.features.definite:
        return False
    if mention.features.upos == 'PRON':
        return False
    return candidateAntecedent.features.headWord.lower() == mention.features.headWord.lower() 

# Pass 3 from Lee et al 2013
# Returns true if the strings are identical up to and including their headwords.
def relaxedStringMatch(config: Config, doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    mentionTextUntilHead = ''
    for id in mention.stanzaIds:
        word = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[id-1]
        mentionTextUntilHead += word.text + ' '
        if id == mention.features.headWordId:
            break
    antecedentTextUntilHead = ''
    for id in candidateAntecedent.stanzaIds:
        word = doc.stanzaAnnotation.sentences[candidateAntecedent.stanzaSentence].words[id-1]
        antecedentTextUntilHead += word.text + ' '
        if id == candidateAntecedent.features.headWordId:
            break        
    return mentionTextUntilHead.lower() == antecedentTextUntilHead.lower()

# Pass 5 from Lee et al 2013
def strictHeadMatch(config: Config, doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    headWords = []
    for m in doc.predictedClusters[candidateAntecedent.predictedCluster]:
        headWords.append(doc.predictedMentions[m].features.headWord)
    if mention.features.headWord not in headWords:
        return False

    return True

# Pass 10 from Lee et al 2013
def pronounResolution(config: Config, doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos != 'PRON':
        return False
    if len(mention.stanzaIds) > 1:
        return False
    # Sentence distance is at most 3 for pronoun resolution
    if mention.stanzaSentence - candidateAntecedent.stanzaSentence > 3:
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
    if mention.features.person != 'UNKNOWN' and candidateAntecedent.features.person != 'UNKNOWN':
        if mention.features.person != candidateAntecedent.features.person:
            return False

    return True

# Pass 10 from Lee et al 2013
def pronounResolutionWithNerRules(config: Config, doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
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
    if mention.features.person != 'UNKNOWN' and candidateAntecedent.features.person != 'UNKNOWN':
        if mention.features.person != candidateAntecedent.features.person:
            return False

    return True

# Pass 10 from Lee et al 2013
def pronounResolutionClusterBased(config: Config, doc: Document, mention: Mention, candidateAntecedent: Mention) -> bool:
    if mention.features.upos != 'PRON':
        return False
    if len(mention.stanzaIds) > 1:
        return False
    # Sentence distance is at most 3 for pronoun resolution
    if mention.stanzaSentence - candidateAntecedent.stanzaSentence > 3:
        return False
    
    # Calculate cluster features for the mention
    mentionNumber = 'UNKNOWN'
    mentionAnimacy = 'UNKNOWN'
    mentionNaturalGender = 'UNKNOWN'
    mentionNerTag = 'UNKNOWN'
    for mId in doc.predictedClusters[mention.predictedCluster]:
        m = doc.predictedMentions[mId]
        if m.features.naturalGender != 'UNKNOWN':
            mentionNaturalGender = m.features.naturalGender
        if m.features.number != 'UNKNOWN':
            mentionNumber = m.features.number
        if m.features.animacy != 'UNKNOWN':
            mentionAnimacy = m.features.animacy
        if m.features.nerTag != 'UNKNOWN':
            mentionNerTag = m.features.nerTag
    
    # Cluster features for the antecedent
    antecedentNumber = 'UNKNOWN'
    antecedentAnimacy = 'UNKNOWN'
    antecedentNaturalGender = 'UNKNOWN'
    antecedentNerTag = 'UNKNOWN'
    for m in doc.predictedClusters[candidateAntecedent.predictedCluster]:
        m = doc.predictedMentions[mId]
        if m.features.naturalGender != 'UNKNOWN':
            antecedentNaturalGender = m.features.naturalGender
        if m.features.number != 'UNKNOWN':
            antecedentNumber = m.features.number
        if m.features.animacy != 'UNKNOWN':
            antecedentAnimacy = m.features.animacy
        if m.features.nerTag != 'UNKNOWN':
            antecedentNerTag = m.features.nerTag
    
    if mentionNaturalGender != 'UNKNOWN' and antecedentNaturalGender != 'UNKNOWN':
        if mentionNaturalGender != antecedentNaturalGender:
            return False
    if mentionNumber != 'UNKNOWN' and antecedentNumber != 'UNKNOWN':
        if mentionNumber != antecedentNumber:
            return False
    if mentionAnimacy != 'UNKNOWN' and antecedentAnimacy != 'UNKNOWN':
        if mentionAnimacy != antecedentAnimacy:
            return False
    if mentionNerTag != 'UNKNOWN' and antecedentNerTag != 'UNKNOWN':
        if mentionNerTag != antecedentNerTag:
            return False

    # Grammatical gender is not cluster based, since the entire cluster does not have to agree
    if mention.features.gender != 'UNKNOWN' and candidateAntecedent.features.gender != 'UNKNOWN':
        if mention.features.gender != candidateAntecedent.features.gender:
            return False
    return True

def genetiveResolution(config: Config, doc: Document, mention: Mention, ca: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    if mention.text[len(mention.text)-1] == 's' and mention.text[:len(mention.text)-1].lower() == ca.text.lower():
        return True
    if ca.text[len(ca.text)-1] == 's' and ca.text[:len(ca.text)-1].lower() == mention.text.lower():
        return True
    return False

def wordVectorDistance(config: Config, doc: Document, mention: Mention, ca: Mention) -> bool:
    if mention.features.upos == 'PRON':
        return False
    if mention.features.headWord in config.wordVectors.key_to_index and ca.features.headWord in config.wordVectors.key_to_index:
        wordvecHeadwordDistance = config.wordVectors.distance(mention.features.headWord.lower(), ca.features.headWord.lower())
        if wordvecHeadwordDistance < 0.2:
            print(mention.features.headWord)
            print(ca.features.headWord)
            print(wordvecHeadwordDistance)
            print()
            return True
        else:
            return False
    
    else:
        return False

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

# Takes first possible antecedent that satisfies the sieve. Return value indicates whether any
# antecedent was found.
def pickAntecedent(config: Config, doc: Document, sieve,  mention: Mention) -> bool:
    for antecedent in getCandidateAntecedents(config, doc, mention, 10000):
        if sieve(config, doc, mention, antecedent):
            link(doc, mention, antecedent)
            return True
    return False

def doSievePass(config: Config, doc: Document, sieve):
    newEligibleMentions = []
    for mention in doc.eligibleMentions:
        if not config.allowIndefiniteAnaphor:
            if mention.features.upos == 'PRON' and mention.features.definite == 'IND':
                print("Indef pron: " + mention.text)
                continue
            if mention.text[:2] == 'en' or mention.text[:3] == 'ett':
                print("Indef noun: " + mention.text)
                continue
        antecedentFound = pickAntecedent(config, doc, sieve, mention)
        if not antecedentFound:
            newEligibleMentions.append(mention)
    doc.eligibleMentions = newEligibleMentions

def multiPassSieve(config: Config, doc: Document, sieves):
    for sieve in sieves:
        doSievePass(config, doc, sieve)

def multiPass(doc: Document, config: Config):
    sieveMapping = {'exactMatch': exactMatch,'genetiveResolution': genetiveResolution,
                    'headWordMatch': headWordMatch, 'lemmaHeadWordMatch': lemmaHeadWordMatch, 
                    'pronounResolution': pronounResolution, 
                    'pronounResolutionClusterBased': pronounResolutionClusterBased,
                    'pronounResolutionWithNerRules': pronounResolutionWithNerRules,
                    'wordVectorDistance': wordVectorDistance}
    sieves = []
    for s in config.multipassSieves:
        if not s in sieveMapping:
            raise Exception(f'Invalid multipass sieve name: {s}')
        sieves.append(sieveMapping[s])
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
    multiPassSieve(config, doc, sieves)
