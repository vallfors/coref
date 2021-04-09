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

    # Minimum cluster distance
    minimumClusterDistance = 10000
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            minimumClusterDistance = min(minimumClusterDistance, abs(m.stanzaSentence-a.stanzaSentence))
    
    # Cluster sizes
    antecedentClusterSize = len(antecedentCluster)
    mentionClusterSize = len(mentionCluster)

    # Any two mentions from each cluster have the same headword?
    clusterHeadwordMatch = 0
    clusterProperHeadwordMatch = 0
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            mHeadword = doc.stanzaAnnotation.sentences[m.stanzaSentence].words[m.features.headWord-1].text
            aHeadword = doc.stanzaAnnotation.sentences[a.stanzaSentence].words[a.features.headWord-1].text
            if mHeadword.lower() == aHeadword.lower():
                if m.features.upos == 'PRON' or a.features.upos == 'PRON':
                    continue
                clusterHeadwordMatch = 1
                if m.features.upos == 'PROPN' and a.features.upos == 'PROPN':
                    clusterProperHeadwordMatch = 1

    # Any two mentions from each cluster are the genitive of the other? Lemmas match?
    clusterGenitiveMatch = 0
    clusterLemmaHeadMatch = 0
    for id in mentionCluster:
        m = doc.predictedMentions[id]
        for antecedentId in antecedentCluster:
            a = doc.predictedMentions[antecedentId]
            if m.features.upos == 'PRON' or a.features.upos == 'PRON':
                continue
            if a.text.lower() +'s' == m.text.lower() or m.text.lower() +'s' == a.text.lower():
                clusterGenitiveMatch = 1
            if a.features.headWordLemma.lower() == m.features.headWordLemma.lower():
                clusterLemmaHeadMatch = 1

    return [sentenceDistance, mentionDistance, minimumClusterDistance,
        antecedentClusterSize, mentionClusterSize, exactStringMatch, identicalHeadWords, 
        identicalHeadWordsAndProper, numberMatch, genderMatch, naturalGenderMatch,
        animacyMatch, nerMatch, clusterHeadwordMatch, clusterProperHeadwordMatch,
        clusterGenitiveMatch, clusterLemmaHeadMatch]
