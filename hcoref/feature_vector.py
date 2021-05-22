from preprocessing.document import Mention, Document
from typing import List


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

def personMatch(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    value = 0.5
    if mention.features.person != 'UNKNOWN' and antecedent.features.person != 'UNKNOWN':
        value = 1
        if mention.features.person != antecedent.features.person:
            value = 0
    return value

def numberMatchClusterBased(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    # Calculate cluster features for the mention
    mentionNumber = set()
    for mId in doc.predictedClusters[mention.predictedCluster]:
        m = doc.predictedMentions[mId]
        mentionNumber.add(m.features.number)
    mentionNumber.discard('UNKNOWN')
    
    # Cluster features for the antecedent
    antecedentNumber = set()

    for mId in doc.predictedClusters[antecedent.predictedCluster]:
        m = doc.predictedMentions[mId]
        antecedentNumber.add(m.features.number)
    antecedentNumber.discard('UNKNOWN')

    if len(mentionNumber) > 0 and len(antecedentNumber) > 0:
        if len(mentionNumber.intersection(antecedentNumber)) < 1:
            return 0
        return 1
    return 0.5

def genderMatchClusterBased(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    # Calculate cluster features for the mention
    mentionAttribute = set()
    for mId in doc.predictedClusters[mention.predictedCluster]:
        m = doc.predictedMentions[mId]
        mentionAttribute.add(m.features.gender)
    mentionAttribute.discard('UNKNOWN')
    
    # Cluster features for the antecedent
    antecedentAttribute = set()

    for mId in doc.predictedClusters[antecedent.predictedCluster]:
        m = doc.predictedMentions[mId]
        antecedentAttribute.add(m.features.gender)
    antecedentAttribute.discard('UNKNOWN')

    if len(mentionAttribute) > 0 and len(antecedentAttribute) > 0:
        if len(mentionAttribute.intersection(antecedentAttribute)) < 1:
            return 0
        return 1
    return 0.5

def animacyMatchClusterBased(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    # Calculate cluster features for the mention
    mentionAttribute = set()
    for mId in doc.predictedClusters[mention.predictedCluster]:
        m = doc.predictedMentions[mId]
        mentionAttribute.add(m.features.animacy)
    mentionAttribute.discard('UNKNOWN')
    
    # Cluster features for the antecedent
    antecedentAttribute = set()

    for mId in doc.predictedClusters[antecedent.predictedCluster]:
        m = doc.predictedMentions[mId]
        antecedentAttribute.add(m.features.animacy)
    antecedentAttribute.discard('UNKNOWN')

    if len(mentionAttribute) > 0 and len(antecedentAttribute) > 0:
        if len(mentionAttribute.intersection(antecedentAttribute)) < 1:
            return 0
        return 1
    return 0.5

def personMatchClusterBased(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    # Calculate cluster features for the mention
    mentionAttribute = set()
    for mId in doc.predictedClusters[mention.predictedCluster]:
        m = doc.predictedMentions[mId]
        mentionAttribute.add(m.features.person)
    mentionAttribute.discard('UNKNOWN')
    
    # Cluster features for the antecedent
    antecedentAttribute = set()

    for mId in doc.predictedClusters[antecedent.predictedCluster]:
        m = doc.predictedMentions[mId]
        antecedentAttribute.add(m.features.person)
    antecedentAttribute.discard('UNKNOWN')

    if len(mentionAttribute) > 0 and len(antecedentAttribute) > 0:
        if len(mentionAttribute.intersection(antecedentAttribute)) < 1:
            return 0
        return 1
    return 0.5

def nerMatchClusterBased(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    # Calculate cluster features for the mention
    mentionAttribute = set()
    for mId in doc.predictedClusters[mention.predictedCluster]:
        m = doc.predictedMentions[mId]
        mentionAttribute.add(m.features.nerTag)
    mentionAttribute.discard('UNKNOWN')
    
    # Cluster features for the antecedent
    antecedentAttribute = set()

    for mId in doc.predictedClusters[antecedent.predictedCluster]:
        m = doc.predictedMentions[mId]
        antecedentAttribute.add(m.features.nerTag)
    antecedentAttribute.discard('UNKNOWN')

    if len(mentionAttribute) > 0 and len(antecedentAttribute) > 0:
        if len(mentionAttribute.intersection(antecedentAttribute)) < 1:
            return 0
        return 1
    return 0.5

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

def iWithinIClusterCheck(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    mentionCluster = doc.predictedClusters[mention.predictedCluster]
    antecedentCluster = doc.predictedClusters[antecedent.predictedCluster]
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            if m.stanzaSentence != a.stanzaSentence:
                continue
            aFirst = a.stanzaIds[0]
            aLast = a.stanzaIds[-1]
            mFirst = m.stanzaIds[0]
            mLast = m.stanzaIds[-1]
            if aFirst >= mFirst and aLast <= mLast:
                return True
            if mFirst >= aFirst and mLast <= aLast:
                return True
    return False

    
# From Hybrid method (Nilsson 2010)

def antecedentFirstInSentence(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    return antecedent.stanzaIds[0] == 1

def anaphorFirstInSentence(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    return mention.stanzaIds[0] == 1

def dependentOnSame(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    if mention.stanzaSentence != antecedent.stanzaSentence:
        return False
    if mention.features.headWordHead == antecedent.features.headWordHead:
        return True
    return False

def anaphorLength(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    return len(mention.stanzaIds)

def antecedentLength(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    return len(antecedent.stanzaIds)

def antecedentClusterIncludesAnaphorCluster(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int):
    mentionCluster = doc.predictedClusters[mention.predictedCluster]
    antecedentCluster = doc.predictedClusters[antecedent.predictedCluster]
    antecedentClusterWords = set()
    for id in antecedentCluster:
        m = doc.predictedMentions[id].text.split(' ')
        for word in m:
            if len(word) > 3:
                antecedentClusterWords.add(word)

    for mentionId in mentionCluster:
        m = doc.predictedMentions[mentionId].text.split(' ')
        for word in m:
            if len(word) > 3:
                if word not in antecedentClusterWords:
                    return False
    return True



    

featureFunction = {'sentenceDistance': sentenceDistance, 'identicalHeadWords': identicalHeadWords,
                    'identicalHeadWordsAndProper': identicalHeadWordsAndProper, 'exactStringMatch': exactStringMatch,
                    'genderMatch': genderMatch, 'naturalGenderMatch': naturalGenderMatch,
                    'numberMatch': numberMatch, 'animacyMatch': animacyMatch, 'nerMatch': nerMatch,
                    'subjectObjectRelation': subjectObjectRelation, 'mentionDistance': mentionDistance,
                    'wordvecHeadWordDistance': wordvecHeadWordDistance, 'minimumClusterDistance': minimumClusterDistance,
                    'mentionClusterSize': mentionClusterSize, 'antecedentClusterSize': antecedentClusterSize,
                    'clusterHeadWordMatch': clusterHeadWordMatch, 'clusterProperHeadWordMatch': clusterProperHeadWordMatch,
                    'clusterLemmaHeadWordMatch': clusterLemmaHeadWordMatch, 'clusterGenitiveHeadWordMatch': clusterGenitiveHeadWordMatch,
                    'antecedentFirstInSentence': antecedentFirstInSentence,'anaphorFirstInSentence':anaphorFirstInSentence, 'dependentOnSame': dependentOnSame,
                    'antecedentLength': antecedentLength,'anaphorLength': anaphorLength,'numberMatchClusterBased':numberMatchClusterBased,'personMatch': personMatch,
                    'nerMatchClusterBased':nerMatchClusterBased,'personMatchClusterBased': personMatchClusterBased,'genderMatchClusterBased': genderMatchClusterBased,'animacyMatchClusterBased': animacyMatchClusterBased,'iWithinIClusterCheck': iWithinIClusterCheck,'antecedentClusterIncludesAnaphorCluster': antecedentClusterIncludesAnaphorCluster}

def getFeatureVector(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int, features):
    featureVector = []
    for feature in features:
        featureVector.append(featureFunction[feature](doc, wordVectors, mention, antecedent, mentionDistance))
    return featureVector

def mentionDeprel(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.headWordDeprel

def mentionHeadWord(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.headWord #obs fel

def mentionNextWordPos(doc: Document, mention: Mention, antecedent: Mention) -> str:
    lastId = mention.stanzaIds[-1] # stanza ids start at 1!
    if lastId < len(doc.stanzaAnnotation.sentences[mention.stanzaSentence].words):
        return doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[lastId].upos
    elif mention.stanzaSentence +1 < len(doc.stanzaAnnotation.sentences):
        return doc.stanzaAnnotation.sentences[mention.stanzaSentence+1].words[0].upos
    else:
        return 'UNDEFINED' # If mention is the last word in the text

def mentionNextWordText(doc: Document, mention: Mention, antecedent: Mention) -> str:
    lastId = mention.stanzaIds[-1] # stanza ids start at 1!
    if lastId < len(doc.stanzaAnnotation.sentences[mention.stanzaSentence].words):
        return doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[lastId].text
    elif mention.stanzaSentence +1 < len(doc.stanzaAnnotation.sentences):
        return doc.stanzaAnnotation.sentences[mention.stanzaSentence+1].words[0].text
    else:
        return 'UNDEFINED' # If mention is the last word in the text

def antecedentDeprel(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.headWordDeprel

def antecedentHeadWord(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.headWord

def antecedentNextWordPos(doc: Document, mention: Mention, antecedent: Mention) -> str:
    lastId = antecedent.stanzaIds[-1] # stanza ids start at 1!
    if lastId < len(doc.stanzaAnnotation.sentences[antecedent.stanzaSentence].words):
        return doc.stanzaAnnotation.sentences[antecedent.stanzaSentence].words[lastId].upos
    elif antecedent.stanzaSentence +1 < len(doc.stanzaAnnotation.sentences):
        return doc.stanzaAnnotation.sentences[antecedent.stanzaSentence+1].words[0].upos
    else:
        return 'UNDEFINED' # If mention is the last word in the text

def antecedentNextWordText(doc: Document, mention: Mention, antecedent: Mention) -> str:
    lastId = antecedent.stanzaIds[-1] # stanza ids start at 1!
    if lastId < len(doc.stanzaAnnotation.sentences[antecedent.stanzaSentence].words):
        return doc.stanzaAnnotation.sentences[antecedent.stanzaSentence].words[lastId].text
    elif antecedent.stanzaSentence +1 < len(doc.stanzaAnnotation.sentences):
        return doc.stanzaAnnotation.sentences[antecedent.stanzaSentence+1].words[0].text
    else:
        return 'UNDEFINED' # If mention is the last word in the text

# From Hybrid method (Nilsson 2010)
def antecedentGrammaticalGender(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.gender

def anaphorGrammaticalGender(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.gender
    
def anaphorDefiniteness(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.definite

def antecedentDefiniteness(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.definite

def anaphorPronounType(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.pronounType

def antecedentPronounType(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.pronounType

def anaphorCase(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.case

def antecedentCase(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.case

def anaphorAnimacy(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.animacy

def antecedentAnimacy(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.animacy

def anaphorNaturalGender(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.naturalGender

def antecedentNaturalGender(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.naturalGender

def anaphorNerTag(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.nerTag

def antecedentNerTag(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.nerTag

def anaphorPerson(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.person

def antecedentPerson(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.person

def anaphorNumber(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return mention.features.number

def antecedentNumber(doc: Document, mention: Mention, antecedent: Mention) -> str:
    return antecedent.features.number

def antecedentPronounText(doc: Document, mention: Mention, antecedent: Mention) -> str:
    if antecedent.features.upos != 'PRON':
        return 'UNDEFINED'
    return antecedent.text

def anaphorPronounText(doc: Document, mention: Mention, antecedent: Mention) -> str:
    if mention.features.upos != 'PRON':
        return 'UNDEFINED'
    return mention.text

stringFeatureFunction = {'mentionDeprel': mentionDeprel, 'mentionHeadWord': mentionHeadWord,
                         'mentionNextWordPos': mentionNextWordPos, 'mentionNextWordText': mentionNextWordText,
                         'antecedentDeprel': antecedentDeprel, 'antecedentHeadWord': antecedentHeadWord,
                         'antecedentNextWordPos': antecedentNextWordPos, 'antecedentNextWordText': antecedentNextWordText,
                         'antecedentGrammaticalGender': antecedentGrammaticalGender,'anaphorGrammaticalGender':anaphorGrammaticalGender,
                         'anaphorDefiniteness': anaphorDefiniteness,'antecedentDefiniteness': antecedentDefiniteness,
                         'antecedentPronounType': antecedentPronounType,'anaphorPronounType' :anaphorPronounType,'anaphorCase': anaphorCase,'antecedentCase': antecedentCase,
                         'antecedentAnimacy':antecedentAnimacy,'anaphorAnimacy': anaphorAnimacy, 'antecedentNaturalGender': antecedentNaturalGender,'anaphorNaturalGender': anaphorNaturalGender,'antecedentNerTag': antecedentNerTag,'anaphorNerTag': anaphorNerTag,'antecedentPerson': antecedentPerson,'anaphorPerson': anaphorPerson,'antecedentNumber': antecedentNumber,'anaphorNumber': anaphorNumber,'antecedentPronounText': antecedentPronounText, 'anaphorPronounText': anaphorPronounText}

def getStringFeatureVector(doc: Document, wordVectors, mention: Mention, antecedent: Mention, mentionDistance: int, stringFeatures: List[str]):
    featureVector = []
    for feature in stringFeatures:
        featureVector.append(stringFeatureFunction[feature](doc, mention, antecedent))
    return featureVector