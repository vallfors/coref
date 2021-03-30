from preprocessing.document import Document
from preprocessing.config import Config

def matchMentions(doc: Document, config: Config):
    doc.predictedToGold = {}
    doc.goldToPredicted = {}

    goldMentionFromPositions = {}
    for goldMention in doc.goldMentions.values():
        goldMentionFromPositions[goldMention.startPos, goldMention.endPos] = goldMention.id
    for predictedMention in doc.predictedMentions.values():
        if (predictedMention.startPos, predictedMention.endPos) in goldMentionFromPositions:
            goldMentionId = goldMentionFromPositions[predictedMention.startPos, predictedMention.endPos]
            doc.predictedToGold[predictedMention.id] = goldMentionId
            doc.goldToPredicted[goldMentionId] = predictedMention.id

    correct = 0
    falseNegatives = 0
    falsePositives = 0

    for predicted, gold in doc.predictedToGold.items():
        correct += 1
    if config.debugMentionDetection:
        print('-----------Printing missed mentions----------')
    for goldMention in doc.goldMentions.values():
        if goldMention.id not in doc.goldToPredicted:
            falseNegatives +=1
            if config.debugMentionDetection:
                print(goldMention.text)
                for word in doc.stanzaAnnotation.sentences[goldMention.stanzaSentence].words:
                    print(word.text, end = ' ')
                print('\n----------')

    if config.debugMentionDetection:
        print('-----------Printing extra mentions----------')
    for predictedMention in doc.predictedMentions.values():
        if predictedMention.id not in doc.predictedToGold:
            falsePositives += 1
            if config.debugMentionDetection:
                print(predictedMention.text)
                for word in doc.stanzaAnnotation.sentences[predictedMention.stanzaSentence].words:
                    print(word.text, end = ' ')

                print('\n----------')
    if config.debugMentionDetection:
        precision = float(correct)/(float(correct)+float(falsePositives))
        recall = float(correct)/(float(correct)+float(falseNegatives))
        print(f'Correct mentions: {correct}')
        print(f'Missed mentions: {falseNegatives}')
        print(f'Extra mentions: {falsePositives}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')