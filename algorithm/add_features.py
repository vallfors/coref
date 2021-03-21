from algorithm.features import Features
from preprocessing.document import Document
import re

def addFeatures(doc: Document):
    for mention in doc.predictedMentions.values():
        mention.features = Features()

        # Finding head word
        ids = mention.stanzaIds
        sentence = mention.stanzaSentence
        for id in ids:
            word = doc.stanzaAnnotation.sentences[sentence].words[id-1]
            if word.head not in ids:
                headWord = word
        mention.features.headWord = headWord.id

        # POS
        mention.features.upos = headWord.upos

        # Plural/singular
        if 'SIN' in headWord.xpos:
            mention.features.number = 'SIN'
        elif 'PLU' in headWord.xpos:
            mention.features.number = 'PLU'
        else:
            mention.features.number = 'UNKNOWN'

        # Definitiness
        if 'DEF' in headWord.xpos:
            mention.features.definite = 'DEF'
        elif 'IND' in headWord.xpos:
            mention.features.definite = 'IND'
        else:
            mention.features.definite = 'UNKNOWN'
        
        # Gender
        # I deliberatily skip the masculine case, which is very unusual
        if 'NEU' in headWord.xpos and 'UTR' in headWord.xpos:
            mention.features.gender = 'NEU/UTR'
        elif 'NEU' in headWord.xpos:
            mention.features.gender = 'NEU'
        elif 'UTR' in headWord.xpos:
            mention.features.gender = 'UTR'
        else:
            mention.features.gender = 'UNKNOWN'

