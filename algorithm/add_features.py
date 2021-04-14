from typing import Dict

from algorithm.features import Features
from preprocessing.document import Document, Mention

animatePronouns = ['han', 'hon', 'jag', 'honom', 'henne', 'hans', 'hennes', 'min', 'mitt', 'din', 'ditt', 'er', 'ert', 'eran', 'vår', 'våran', 'vårt', 'ni', 'vi']
animateNerTags = ['PER', 'ORG']
inanimateNerTags = ['TME', 'LOC', 'EVN']

singularPronouns = ['han', 'hon', 'jag', 'du', 'honom', 'henne', 'hans', 'hennes', 'dess', 'din', 'ditt', 'min', 'mitt']
pluralPronouns = ['deras', 'vår', 'våran', 'vårt', 'vi', 'de', 'dem']
pluralOrSingularPronouns = ['sin', 'sig', 'er', 'ert', 'eran', 'ni']

def extractAnimacy(headWord, nerTag):
    if headWord.text in animatePronouns:
        return 'ANIMATE'
    if nerTag in animateNerTags:
        return 'ANIMATE'
    if nerTag in inanimateNerTags:
        return 'INANIMATE'
    return 'UNKNOWN'

def extractNaturalGender(headWord):
    if headWord.text in ['han', 'honom', 'hans']:
        return 'MALE'
    if headWord.text in ['hon', 'henne', 'hennes']:
        return 'FEMALE'
    return 'UNKNOWN'

def extractGender(headWord):
    # I deliberatily skip the masculine case, which is very unusual
    if 'NEU' in headWord.xpos and 'UTR' in headWord.xpos:
        return 'UNKNOWN'
    if 'NEU' in headWord.xpos:
        return 'NEU'
    if 'UTR' in headWord.xpos:
       return 'UTR'
    return 'UNKNOWN'

def extractNumber(headWord, nerTag):
    if headWord.text in singularPronouns:
        return 'SIN'
    if headWord.text in pluralPronouns:
        return 'PLU'
    if headWord.text in pluralOrSingularPronouns:
        return 'UNKNOWN'
    if nerTag in ['ORG']:
        return 'UNKNOWN' # Organizations can be referred to both by plural and singular references
    if 'SIN' in headWord.xpos and 'PLU' in headWord.xpos:
        return 'UNKNOWN'
    if 'SIN' in headWord.xpos:
        return 'SIN'
    if 'PLU' in headWord.xpos:
        return'PLU'
    return 'UNKNOWN'

def extractDefiniteness(headWord):
    if 'DEF' in headWord.xpos:
        return 'DEF'
    if 'IND' in headWord.xpos:
        return 'IND'
    else:
        return 'UNKNOWN'

def addFeaturesToMentionDict(doc: Document, nerPipeline, mentions: Dict[int, Mention]):
    for mention in mentions.values():
        mention.features = Features()

        # Finding head word
        ids = mention.stanzaIds
        sentence = mention.stanzaSentence
        for id in ids:
            word = doc.stanzaAnnotation.sentences[sentence].words[id-1]
            if word.head not in ids:
                headWord = word
        mention.features.headWordId = headWord.id
        mention.features.headWord = headWord.text

        # Named entity recognition
        nerResult = nerPipeline(mention.text)
        if len(nerResult) < 1:
            mention.features.nerTag = 'UNKNOWN'
        else:
            mention.features.nerTag = nerResult[0]['entity']

        mention.features.upos = headWord.upos        
        mention.features.headWordDeprel = headWord.deprel
        mention.features.headWordLemma = headWord.lemma
        mention.features.headWordHead = headWord.head

        mention.features.isSubject = 'nsubj' in headWord.deprel
        mention.features.isObject = 'obj' in headWord.deprel # Includes indirect objects, iobj
        
        mention.features.number = extractNumber(headWord, mention.features.nerTag)
        mention.features.animacy = extractAnimacy(headWord, mention.features.nerTag)
        mention.features.naturalGender = extractNaturalGender(headWord)
        mention.features.gender = extractGender(headWord)
        mention.features.definite = extractDefiniteness(headWord)

def addFeatures(doc: Document, nerPipeline):
    addFeaturesToMentionDict(doc, nerPipeline, doc.goldMentions)
    addFeaturesToMentionDict(doc, nerPipeline, doc.predictedMentions)