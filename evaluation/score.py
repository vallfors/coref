from typing import DefaultDict, List

from preprocessing.document import Document

def writeConllForScoring(docs: List[Document]):
    writeGoldOrPredictedForScoring(docs, True)
    writeGoldOrPredictedForScoring(docs, False)

def writeGoldOrPredictedForScoring(docs: List[Document], gold: bool):
    if gold:
        outFilePath = f'evaluation/goldClusters/all'
    else:
        outFilePath = f'evaluation/predictedClusters/all'
    with open(outFilePath, 'w') as outFile:
        for doc in docs:
            startsAtWord = DefaultDict(list)
            endsAtWord = DefaultDict(list)
            startsAndEndsAtWord = DefaultDict(list)
            if gold:
                clusters = doc.goldClusters
                mentions = doc.goldMentions
            else:
                clusters = doc.predictedClusters
                mentions = doc.predictedMentions
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