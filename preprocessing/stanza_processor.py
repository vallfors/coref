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
    broken = []
    for mentionId, mention in doc.goldMentions.items():
        if mention.startPos not in positionToSentence:
            broken.append((mention.id, mention.cluster))
            continue
            #mention.stanzaSentence = 1 # Temporary 'fix', don't check in!
            #raise Exception(f'Gold mention {mention.text} has no corresponding Stanza token')
        else:
            mention.stanzaSentence = positionToSentence[mention.startPos]
        mention.stanzaIds = []
        for word in doc.stanzaAnnotation.sentences[mention.stanzaSentence].words:
            wordStartPos = int(word.misc.split('|')[0].split('=')[1])
            wordEndPos = int(word.misc.split('|')[1].split('=')[1])
            if wordStartPos >= mention.startPos and wordEndPos <= mention.endPos:
                mention.stanzaIds.append(word.id)
        if len(mention.stanzaIds) == 0:
            broken.append((mention.id, mention.cluster))
    if len(broken) > 0:
        print('WARNING! Some gold mentions do not match token limits:')
    for t in broken:
        (mentionId, clusterId) = t
        print(doc.goldMentions[mentionId].text)
        doc.goldClusters[clusterId].remove(mentionId)
        del doc.goldMentions[mentionId]