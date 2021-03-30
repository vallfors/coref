from typing import DefaultDict
from preprocessing.document import Document, Mention

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

def printCluster(cluster, mentions, text, contrastColor):
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

def compareClusters(doc: Document):
    correspondingPredictedClusters = DefaultDict(set)
    correspondingGoldClusters = DefaultDict(set)
    for goldId, goldCluster in doc.goldClusters.items():
        for predictedId, predictedCluster in doc.predictedClusters.items():
            for mentionId in goldCluster:
                if mentionId in doc.goldToPredicted and doc.goldToPredicted[mentionId] in predictedCluster:
                    correspondingPredictedClusters[goldId].add(predictedId)
                    correspondingGoldClusters[predictedId].add(goldId)
    
    for goldId, goldCluster in doc.goldClusters.items():
        print( bcolors.OKGREEN + 'GOLD CLUSTER:' + bcolors.ENDC)
        printCluster(goldCluster, doc.goldMentions, doc.text, bcolors.OKCYAN)
        print( bcolors.OKGREEN + 'THE FOLLOWING PREDICTED CLUSTERS HAVE OVERLAP:' + bcolors.ENDC)
        for predictedClusterId in correspondingPredictedClusters[goldId]:
            printCluster(doc.predictedClusters[predictedClusterId], doc.predictedMentions, doc.text, bcolors.WARNING)
