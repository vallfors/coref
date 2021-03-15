from typing import Dict, List
import stanza
import json

from preprocessing.read_conll import CoNLLFile

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Mention:
    id: int
    text: str
    startPos: int
    endPos: int
    cluster: int

# Prints one cluster at a time, by displaying the text with characters that belong to the cluster highlighted.
# The text shown begins 30 characters before the first mention in the cluster, and ends 30 characters after the last.
# Waits for user input between each cluster
def printClusters(mentions: Dict[int, Mention], clusters: Dict[int, List[int]], text: str):
    contrastColor = bcolors.OKGREEN
    for cluster in clusters.values():
        clusterPositions = set()
        minStart = 1000000
        maxEnd = 0
        for mentionId in cluster:
            start = mentions[mentionId].startPos
            end = mentions[mentionId].endPos
            minStart = min(minStart, start)
            maxEnd = max(maxEnd, end)
            for i in range(start, end):
                clusterPositions.add(i)
        minStart = max(0, minStart-30)
        maxEnd = min(len(text), maxEnd+30)
        for pos in range(minStart, maxEnd):
            if pos in clusterPositions:
                print( contrastColor + text[pos] + bcolors.ENDC, end='')
            else:
                print(text[pos], end='')
        input()
        print()
        print('################################################')
        print()


class Document:
    docName: str
    text: str
    stanzaAnnotation: stanza.models.common.doc.Document

    goldMentions: Dict[int, Mention]
    goldClusters: Dict[int, List[int]]

    predictedMentions: Dict[int, Mention]
    predictedClusters: Dict[int, List[int]]

    # A list of eligible mentions in the order they should be processed.
    # An eligible mention is a mention that is first in its cluster. 
    eligibleMentions: List[Mention]
    
    def printGold(self):
        printClusters(self.goldMentions, self.goldClusters, self.text)

    def printPredicted(self):
        printClusters(self.predictedMentions, self.predictedClusters, self.text)

def documentFromRawText(name: str, text: str) -> Document:
    doc = Document()
    doc.docName = name
    doc.text = text

def documentFromConll(conll: CoNLLFile) -> Document:
    doc = Document()
    doc.docName = conll.fileName

    doc.text = ""
    finishedMentions = []
    mentionsInProgress = {}
    for row in conll.rows:
        doc.text += row.form + " "
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
                mention.startPos = len(doc.text)-len(row.form)-1
                if clusterId in mentionsInProgress:
                    raise Exception("i within i!")
                mentionsInProgress[clusterId] = mention
            if last:
                mention = mentionsInProgress[clusterId]
                mention.endPos = len(doc.text)-1
                finishedMentions.append(mention)
                del mentionsInProgress[clusterId]
    doc.goldMentions = {}
    doc.goldClusters = {}
    idCounter = 0
    for mention in finishedMentions:
        mention.id = idCounter
        doc.goldMentions[idCounter] = mention
        if mention.cluster not in doc.goldClusters:
            doc.goldClusters[mention.cluster] = []
        doc.goldClusters[mention.cluster].append(mention.id)

        idCounter += 1

    return doc

def documentsFromTextinatorFile(filename: str) -> List[Document]:
    with open(filename) as f:
        jsonData = json.load(f)
    docs = []
    for jsonDocument in jsonData['data']:
        doc = Document()
        doc.text = jsonDocument['context']
        doc.docName = filename

        # Assign id:s to mentions and clusters and add them to the document
        clusterId = 0
        mentionId = 0
        doc.goldMentions = {}
        doc.goldClusters = {}
        for relation in jsonDocument['relations'].values():
            cluster = []
            for node in relation['nodes']:
                mention = Mention()
                mention.cluster = clusterId
                mention.startPos = node['start']
                mention.endPos = node['end']
                mention.id = mentionId
                mention.text = node['text']
                doc.goldMentions[mentionId] = mention
                cluster.append(mentionId)
                mentionId += 1
            doc.goldClusters[clusterId] = cluster
            clusterId += 1
        docs.append(doc)
    return docs