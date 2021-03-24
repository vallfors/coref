from preprocessing.document import Document, Mention
from collections import deque
import operator

def getSubtrees(sentence, word):
    children = {}
    for w in sentence.words:
        if w.deprel in ['orphan', 'appos']:
            continue
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

    allResults = {}
    for id, w in enumerate(results):
        if w.deprel == 'cop':
            allResults[str(results[id+1:])] = results[id+1:]

    for id, w in enumerate(results):
        if w.text == 'och':
            allResults[str(results[:id])] = results[:id]
    allResults[str(results)] = results

    for r in allResults.values():
        if len(r) == 0:
            continue
        while r[0].deprel in ['case', 'cc', 'punct', 'mark']:
            del r[0]

        while r[len(r)-1].deprel == 'punct' and r[len(r)-1].text != "'":
            del r[len(r)-1]

        for id, w in enumerate(r):
            if w.text == ',':
                r = r[:id]

    return allResults.values()


def candidatePos(word) -> bool:
    xpos = word.xpos.split('|')[0]
    # Possibly 'DT' should also be added
    if xpos not in ['PS', 'PN', 'PM', 'NN']:
        return False
 
    if word.deprel in ['det', 'flat:name', 'expl', 'fixed', 'nummod', 'flat', 'goeswith', 'compound', 'appos']:
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
    mention.text = mention.text[:-1]
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
                wordLists = getSubtrees(sentence, word)
                for wordList in wordLists:
                    if len(wordList) == 0:
                        continue
                    doc.predictedMentions[currentMentionId] = createMention(wordList, currentMentionId, sentenceId)
                    currentMentionId += 1
