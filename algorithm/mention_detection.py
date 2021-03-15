from preprocessing.document import Document, Mention
from collections import deque
import operator

def printSubtree(sentence, word):
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
    print()
    if results[0].deprel == 'case':
        del results[0]
    for w in results:
        print(f'{w.text} ', end='')
    print()


def candidatePos(word) -> bool:
    if word.upos not in ['NOUN', 'PROPN', 'DET', 'PRON']:
        return False
    if word.deprel in ['det', 'nmod', 'flat:name']:
        return False
    return True

# Detects mentions
def mentionDetection(doc: Document):
    for sentence in doc.stanzaAnnotation.sentences:
        for word in sentence.words:
            if candidatePos(word):
                printSubtree(sentence, word)
