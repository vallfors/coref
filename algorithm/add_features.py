from typing import Dict

from algorithm.features import Features
from preprocessing.document import Document, Mention

animateNerTags = ['PER', 'ORG']
inanimateNerTags = ['TME', 'LOC', 'EVN']
animatePronouns = ['jag', 'mig', 'min', 'mitt', 'mina', 'du', 'dig', 'din', 'ditt', 'dina', 'han',
                    'honom', 'hans', 'hon', 'henne', 'hennes','ni', 'er', 'ert', 'era', 'vi',
                    'oss', 'vår', 'vårt', 'våra', 'ni', 'er', 'ert', 'era']

singularPronouns = ['jag', 'mig', 'min', 'mitt', 'mina', 'du', 'dig', 'din', 'ditt', 'dina', 'han',
                    'honom', 'hans', 'hon', 'henne', 'hennes', 'den', 'dess', 'det', 'man', 'en', 'ens',
                    'sig', 'sin', 'sitt', 'sina', 'ni', 'er', 'ert', 'era']

pluralPronouns = ['vi', 'oss', 'vår', 'vårt', 'våra', 'ni', 'er', 'ert', 'era', 'de', 'dem', 'deras',
                  'sig', 'sin', 'sitt', 'sina']
                  

firstPersonPronouns = ['jag', 'mig', 'min', 'mitt', 'mina', 'vi', 'oss', 'vår', 'vårt', 'våra']
secondPersonPronouns = ['du', 'dig', 'din', 'ditt', 'dina',	'ni', 'er', 'ert', 'era']
thirdPersonPronouns = ['han', 'honom', 'hans', 'de', 'dem', 'deras','hon', 'henne', 'hennes',
                        'den', 'dess', 'det', 'man', 'en', 'ens', 'sig', 'sin', 'sitt', 'sina']

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
    if headWord.text in singularPronouns and headWord.text in pluralPronouns:
        return 'UNKNOWN'
    if headWord.text in singularPronouns:
        return 'SIN'
    if headWord.text in pluralPronouns:
        return 'PLU'
    
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

def extractPerson(headWord):
    if headWord.text in firstPersonPronouns:
        return 'first'
    if headWord.text in secondPersonPronouns:
        return 'second'
    if headWord.text in thirdPersonPronouns:
        return 'third'
    return 'UNKNOWN'

def extractPronounType(headWord):
    if headWord.feats == None:
        return 'UNKNOWN'
    feats = headWord.feats.split('|')
    for feat in feats:
        if feat.startswith('PronType'):
            return feat.split('=')[1]
    return 'UNKNOWN'

def extractCase(headWord):
    if headWord.feats == None:
        return 'UNKNOWN'
    feats = headWord.feats.split('|')
    for feat in feats:
        if feat.startswith('Case'):
            return feat.split('=')[1]
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
        mention.features.person = extractPerson(headWord)
        mention.features.pronounType = extractPronounType(headWord)
        mention.features.case = extractCase(headWord)
def addFeatures(doc: Document, nerPipeline):
    addFeaturesToMentionDict(doc, nerPipeline, doc.goldMentions)
    addFeaturesToMentionDict(doc, nerPipeline, doc.predictedMentions)