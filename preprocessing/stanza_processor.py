import stanza

from preprocessing.document import Document

class StanzaAnnotator:
    def __init__(self):
        stanza.download('sv')
        self.nlp = stanza.Pipeline(lang='sv')

    def annotateDocument(self, doc: Document):
        doc.stanzaAnnotation = self.nlp(doc.text)

def addStanzaLinksToGoldMentions(doc: Document):
    positionToSentence = {}
    for sentenceId, sentence in enumerate(doc.stanzaAnnotation.sentences):
        for word in sentence.words:
            # The misc field is formated as "start_char=4388|end_char=4392"
            startPos = int(word.misc.split('|')[0].split('=')[1])
            positionToSentence[startPos] = sentenceId
    
    for mention in doc.goldMentions.values():
        if mention.startPos not in positionToSentence:
            raise Exception('Gold mention "{mention.text}" has no corresponding Stanza token')
        mention.stanzaSentence = positionToSentence[mention.startPos]
        mention.stanzaIds = []
        for word in doc.stanzaAnnotation.sentences[mention.stanzaSentence].words:
            wordStartPos = int(word.misc.split('|')[0].split('=')[1])
            wordEndPos = int(word.misc.split('|')[1].split('=')[1])

            if wordStartPos < mention.startPos and wordEndPos > mention.startPos:
                raise Exception(f'Stanza word "{word.text}" is partially overlapped with gold mention "{mention.text}"')
            if wordStartPos < mention.endPos and wordEndPos > mention.endPos:
                raise Exception(f'Stanza word "{word.text}" is partially overlapped with gold mention "{mention.text}"')
            if wordStartPos >= mention.startPos and wordEndPos <= mention.endPos:
                mention.stanzaIds.append(word.id)