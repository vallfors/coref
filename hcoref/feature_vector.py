from preprocessing.document import Mention, Document

def getFeatureVector(doc: Document, mention: Mention, antecedent: Mention, mentionDistance: int):
    sentenceDistance = mention.stanzaSentence-antecedent.stanzaSentence
    mentionHeadWord = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[mention.features.headWord-1].text
    antecedentHeadWord = doc.stanzaAnnotation.sentences[antecedent.stanzaSentence].words[antecedent.features.headWord-1].text
    if mentionHeadWord.lower() == antecedentHeadWord.lower():
        identicalHeadWords = 1
    else:
        identicalHeadWords = 0
    if identicalHeadWords and mention.features.upos == 'PROPN' and antecedent.features.upos == 'PROPN':
        identicalHeadWordsAndProper = 1
    else:
        identicalHeadWordsAndProper = 0
    if mention.text.lower() == antecedent.text.lower():
        exactStringMatch = 1
    else:
        exactStringMatch = 0
    genderMatch = 0.5
    if mention.features.gender != 'UNKNOWN' and antecedent.features.gender != 'UNKNOWN':
        genderMatch = 1
        if mention.features.gender != antecedent.features.gender:
            genderMatch = 0
    naturalGenderMatch = 0.5
    if mention.features.naturalGender != 'UNKNOWN' and antecedent.features.naturalGender != 'UNKNOWN':
        naturalGenderMatch = 1
        if mention.features.naturalGender != antecedent.features.naturalGender:
            naturalGenderMatch = 0
    numberMatch = 0.5
    if mention.features.number != 'UNKNOWN' and antecedent.features.number != 'UNKNOWN':
        numberMatch = 1
        if mention.features.number != antecedent.features.number:
            numberMatch = 0
    animacyMatch = 0.5
    if mention.features.animacy != 'UNKNOWN' and antecedent.features.animacy != 'UNKNOWN':
        animacyMatch = 1
        if mention.features.animacy != antecedent.features.animacy:
            animacyMatch = 0
    nerMatch = 0.5
    if mention.features.nerTag != 'UNKNOWN' and antecedent.features.nerTag != 'UNKNOWN':
        nerMatch = 1
        if mention.features.nerTag != antecedent.features.nerTag:
            nerMatch = 0

    # Cluster features
    mentionCluster = doc.predictedClusters[mention.predictedCluster]
    antecedentCluster = doc.predictedClusters[antecedent.predictedCluster]
    minimumClusterDistance = 10000
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            minimumClusterDistance = min(minimumClusterDistance, abs(m.stanzaSentence-a.stanzaSentence))
    
    antecedentClusterSize = len(antecedentCluster)
    mentionClusterSize = len(mentionCluster)

    return [sentenceDistance, mentionDistance, minimumClusterDistance, antecedentClusterSize, mentionClusterSize, exactStringMatch, identicalHeadWords, identicalHeadWordsAndProper, numberMatch, genderMatch, naturalGenderMatch, animacyMatch, nerMatch]
