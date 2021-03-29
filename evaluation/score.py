from typing import DefaultDict
from preprocessing.document import Document

def writeConllForScoring(doc: Document):
    writeGoldOrPredictedForScoring(doc, True)
    writeGoldOrPredictedForScoring(doc, False)

def writeGoldOrPredictedForScoring(doc: Document, gold: bool):
    startsAtWord = DefaultDict(list)
    endsAtWord = DefaultDict(list)
    startsAndEndsAtWord = DefaultDict(list)
    if gold:
        clusters = doc.goldClusters
        mentions = doc.goldMentions
        outFilePath = f'evaluation/goldClusters/{doc.docName}'
    else:
        clusters = doc.predictedClusters
        mentions = doc.predictedMentions
        outFilePath = f'evaluation/predictedClusters/{doc.docName}'
    
    for clusterId, cluster in clusters.items():
        if len(cluster) <= 1:
            continue
        for mentionId in cluster:
            mention = mentions[mentionId]
            if len(mention.stanzaIds) == 1:
                startsAndEndsAtWord[(mention.stanzaSentence, mention.stanzaIds[0])].append(clusterId)
            else:
                startsAtWord[(mention.stanzaSentence, mention.stanzaIds[0])].append(clusterId)
                endsAtWord[(mention.stanzaSentence, mention.stanzaIds[len(mention.stanzaIds)-1])].append(clusterId)
    
    with open(outFilePath, 'w') as outFile:
        outFile.write(f'#begin document ({doc.docName});\n')
        for sentenceId, sentence in enumerate(doc.stanzaAnnotation.sentences):
            for word in sentence.words:
                corefString = ''
                for clusterId in startsAndEndsAtWord[(sentenceId, word.id)]:
                    corefString += '(' + str(clusterId) + ')' + '|'
                for clusterId in startsAtWord[(sentenceId, word.id)]:
                    corefString += '(' + str(clusterId) + '|'
                for clusterId in endsAtWord[(sentenceId, word.id)]:
                    corefString += str(clusterId) + ')' + '|'
                if corefString == '':
                    corefString = '-'
                corefString = corefString.strip('|')
                outFile.write(f'{doc.docName}\t{sentenceId}\t{word.id-1}\t{word.text}\t{corefString}\n')
        outFile.write('#end document\n')