from typing import List
from read_conll import CoNLLFile, loadFromFile

class Mention:
    id: int
    text: str
    startPos: int
    endPos: int
    cluster: int

class Cluster:
    mentionIds: List[int]

class Document:
    docName: str
    text: str
    goldMentions: List[Mention]
    goldClusters: List[Cluster]

    predictedMentions: List[Mention]
    predictedClusters: List[Cluster]

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
        self.goldMentions = finishedMentions


            

def main():
    obj = loadFromFile('./data/suc-core-conll/aa05_fixed.conll')
    doc = Document(obj)
    for mention in doc.goldMentions:
        print("{} {}".format(mention.cluster, mention.text))

if __name__ == "__main__":
    main()