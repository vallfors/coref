from preprocessing.document import Document, Mention
from collections import deque
import operator

def getSubtree(sentence, word):
    children = {}
    for w in sentence.words:
        if w.head not in children:
            children[w.head] = []
        children[w.head].append(w)
    results = []
    q = deque()
    q.append(word)
    while len(q) > 0:
        v = q.popleft()
        results.append(v)
        if v.id not in children:
            continue
        for c in children[v.id]:
            q.append(c)
    results.sort(key=operator.attrgetter('id'))
    if results[0].deprel == 'case':
        del results[0]
    return results


def candidatePos(word) -> bool:
    if word.upos not in ['NOUN', 'PROPN', 'DET', 'PRON']:
        return False
    if word.deprel in ['det', 'nmod', 'flat:name']:
        return False
    return True

def createMention(wordList, id: int, sentenceId: int) -> Mention:
    mention = Mention()
    mention.id = id

    firstWord = wordList[0]
    lastWord = wordList[len(wordList)-1]
    # The misc field is formated as "start_char=4388|end_char=4392"
    # And we take the start_char from the first word, and end_char from the last word
    mention.startPos = int(firstWord.misc.split('|')[0].split('=')[1])
    mention.endPos = int(lastWord.misc.split('|')[1].split('=')[1])

    mention.text = ''
    for word in wordList:
        mention.text += word.text + ' '

    mention.stanzaIds = []
    for word in wordList:
        mention.stanzaIds.append(word.id)
    mention.stanzaSentence = sentenceId
    return mention

# Detects mentions and adds them to the predicted mentions of the document
def mentionDetection(doc: Document):
    doc.predictedMentions = {}
    currentMentionId = 0
    for sentenceId, sentence in enumerate(doc.stanzaAnnotation.sentences):
        for word in sentence.words:
            if candidatePos(word):
                wordList = getSubtree(sentence, word)
                doc.predictedMentions[currentMentionId] = createMention(wordList, currentMentionId, sentenceId)
                currentMentionId += 1
