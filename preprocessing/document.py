from typing import Dict, List
from read_conll import CoNLLFile, loadFromFile

class Mention:
    id: int
    text: str
    startPos: int
    endPos: int
    cluster: int


class Document:
    docName: str
    text: str
    goldMentions: Dict[int, Mention]
    goldClusters: Dict[int, List[int]]

    predictedMentions: Dict[int, Mention]
    predictedClusters: Dict[int, List[int]]

    def __init__(self, conll: CoNLLFile):
        self.docName = conll.fileName

        self.text = ""
        finishedMentions = []
        mentionsInProgress = {}
        for row in conll.rows:
            self.text += row.form + " "
            for mention in mentionsInProgress.values():
                    mention.text += row.form + " "
            if row.coreference == '-':
                continue
            corefs = row.coreference.split('|')
            for coref in corefs:
                first = False
                last = False
                if coref[0] == '(':
                    first = True
                if coref[len(coref)-1] == ')':
                    last = True
                clusterId = int(coref.strip(' ()'))
                if first:
                    mention = Mention()
                    mention.cluster = clusterId
                    mention.text = row.form + " "
                    mention.startPos = len(self.text)-len(row.form)-1
                    mentionsInProgress[clusterId] = mention
                if last:
                    mention = mentionsInProgress[clusterId]
                    mention.endPos = len(self.text)-1
                    finishedMentions.append(mention)
                    del mentionsInProgress[clusterId]
        self.goldMentions = {}
        self.goldClusters = {}
        idCounter = 0
        for mention in finishedMentions:
            mention.id = idCounter
            self.goldMentions[idCounter] = mention
            if mention.cluster not in self.goldClusters:
                self.goldClusters[mention.cluster] = []
            self.goldClusters[mention.cluster].append(mention.id)

            idCounter += 1


def main():
    obj = loadFromFile('./data/suc-core-conll/aa05_fixed.conll')
    doc = Document(obj)
    for id, mention in doc.goldMentions.items():
        print("{} {}".format(mention.id, mention.text))
    for id, cluster in doc.goldClusters.items():
        print(id)
        for mentionId in cluster:
            print(doc.goldMentions[mentionId].text)

if __name__ == "__main__":
    main()