from preprocessing.document import Mention, Document


def sentenceDistance(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    return mention.stanzaSentence-antecedent.stanzaSentence

def identicalHeadWords(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    if mention.features.headWord.lower() == antecedent.features.headWord.lower():
        return 1
    else:
        return 0

def identicalHeadWordsAndProper(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    sameHeadWords = identicalHeadWords(doc, wordVectors, mention, antecedent, mentionDistance)
    if sameHeadWords and mention.features.upos == 'PROPN' and antecedent.features.upos == 'PROPN':
        return 1
    else:
        return 0

def exactStringMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    if mention.text.lower() == antecedent.text.lower():
        return 1
    else:
        return 0

def genderMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    value = 0.5
    if mention.features.gender != 'UNKNOWN' and antecedent.features.gender != 'UNKNOWN':
        value = 1
        if mention.features.gender != antecedent.features.gender:
            value = 0
    return value

def naturalGenderMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    value = 0.5
    if mention.features.naturalGender != 'UNKNOWN' and antecedent.features.naturalGender != 'UNKNOWN':
        value = 1
        if mention.features.naturalGender != antecedent.features.naturalGender:
            value = 0
    return value

def numberMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    value = 0.5
    if mention.features.number != 'UNKNOWN' and antecedent.features.number != 'UNKNOWN':
        value = 1
        if mention.features.number != antecedent.features.number:
            value = 0
    return value

def animacyMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    value = 0.5
    if mention.features.animacy != 'UNKNOWN' and antecedent.features.animacy != 'UNKNOWN':
        value = 1
        if mention.features.animacy != antecedent.features.animacy:
            value = 0
    return value

def nerMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    value = 0.5
    if mention.features.nerTag != 'UNKNOWN' and antecedent.features.nerTag != 'UNKNOWN':
        value = 1
        if mention.features.nerTag != antecedent.features.nerTag:
            value = 0
    return value

def subjectObjectRelation(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    value = 0
    if mention.features.headWordHead == antecedent.features.headWordHead:
        if mention.features.isObject and antecedent.features.isSubject:
            value = 1
        if mention.features.isSubject and antecedent.features.isObject:
            value = 1
    return value

def mentionDistance(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    return mentionDistance

def wordvecHeadWordDistance(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    if mention.features.headWord in wordVectors.key_to_index and antecedent.features.headWord in wordVectors.key_to_index:
        value = wordVectors.distance(mention.features.headWord.lower(), antecedent.features.headWord.lower())
    else:
        value = -1
    return value

def minimumClusterDistance(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    mentionCluster = doc.predictedClusters[mention.predictedCluster]
    antecedentCluster = doc.predictedClusters[antecedent.predictedCluster]

    value = 10000
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            value = min(value, abs(m.stanzaSentence-a.stanzaSentence))
    return value

def mentionClusterSize(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    mentionCluster = doc.predictedClusters[mention.predictedCluster]
    return len(mentionCluster)

def antecedentClusterSize(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    antecedentCluster = doc.predictedClusters[antecedent.predictedCluster]
    return len(antecedentCluster)

def clusterHeadWordMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    mentionCluster = doc.predictedClusters[mention.predictedCluster]
    antecedentCluster = doc.predictedClusters[antecedent.predictedCluster]
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            if m.features.headWord.lower() == a.features.headWord.lower():
                if m.features.upos == 'PRON' or a.features.upos == 'PRON':
                    continue
                return True
    return False

def clusterProperHeadWordMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    mentionCluster = doc.predictedClusters[mention.predictedCluster]
    antecedentCluster = doc.predictedClusters[antecedent.predictedCluster]
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            if m.features.upos == 'PROPN' and a.features.upos == 'PROPN':
                if m.features.headWord.lower() == a.features.headWord.lower():
                    return True
    return False
    
def clusterLemmaHeadWordMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    mentionCluster = doc.predictedClusters[mention.predictedCluster]
    antecedentCluster = doc.predictedClusters[antecedent.predictedCluster]
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            if m.features.upos == 'PRON' or a.features.upos == 'PRON':
                continue
            if a.features.headWordLemma.lower() == m.features.headWordLemma.lower():
                return True
    return False

def clusterGenitiveHeadWordMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    mentionCluster = doc.predictedClusters[mention.predictedCluster]
    antecedentCluster = doc.predictedClusters[antecedent.predictedCluster]
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            if m.features.upos == 'PRON' or a.features.upos == 'PRON':
                continue
            if a.text.lower() +'s' == m.text.lower() or m.text.lower() +'s' == a.text.lower():
                return True
    return False

featureFunction = {'sentenceDistance': sentenceDistance, 'identicalHeadWords': identicalHeadWords,
                    'identicalHeadWordsAndProper': identicalHeadWordsAndProper, 'exactStringMatch': exactStringMatch,
                    'genderMatch': genderMatch, 'naturalGenderMatch': naturalGenderMatch,
                    'numberMatch': numberMatch, 'animacyMatch': animacyMatch, 'nerMatch': nerMatch,
                    'subjectObjectRelation': subjectObjectRelation, 'mentionDistance': mentionDistance,
                    'wordvecHeadWordDistance': wordvecHeadWordDistance, 'minimumClusterDistance': minimumClusterDistance,
                    'mentionClusterSize': mentionClusterSize, 'antecedentClusterSize': antecedentClusterSize,
                    'clusterHeadWordMatch': clusterHeadWordMatch, 'clusterProperHeadWordMatch': clusterProperHeadWordMatch,
                    'clusterLemmaHeadWordMatch': clusterLemmaHeadWordMatch, 'clusterGenitiveHeadWordMatch': clusterGenitiveHeadWordMatch}

def getFeatureVector(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int, features):
    featureVector = []
    for feature in features:
        featureVector.append(featureFunction[feature](doc, wordVectors, mention, antecedent, mentionDistance))
    return featureVector
def getStringFeatureVector(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    lastId = mention.stanzaIds[-1] # stanza ids start at 1!
    if lastId < len(doc.stanzaAnnotation.sentences[mention.stanzaSentence].words):
        mentionNextWord = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[lastId]
    elif mention.stanzaSentence+1 < len(doc.stanzaAnnotation.sentences):
        mentionNextWord = doc.stanzaAnnotation.sentences[mention.stanzaSentence+1].words[0]
    else:
        mentionNextWord = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[0] # Should happen rarely, not an actual fix
    return [mention.features.headWordDeprel, mention.features.headWord, mentionNextWord.upos, mentionNextWord.text]
