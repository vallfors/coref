from preprocessing.document import Document

def matchMentions(doc: Document):
    predictedToGold = {}
    goldToPredicted = {}

    goldMentionFromPositions = {}
    for goldMention in doc.goldMentions.values():
        goldMentionFromPositions[goldMention.startPos, goldMention.endPos] = goldMention.id
    for predictedMention in doc.predictedMentions.values():
        if (predictedMention.startPos, predictedMention.endPos) in goldMentionFromPositions:
            goldMentionId = goldMentionFromPositions[predictedMention.startPos, predictedMention.endPos]
            predictedToGold[predictedMention.id] = goldMentionId
            goldToPredicted[goldMentionId] = predictedMention.id

    correct = 0
    falseNegatives = 0
    falsePositives = 0

    for predicted, gold in predictedToGold.items():
        correct += 1

    for goldMention in doc.goldMentions.values():
        if goldMention.id not in goldToPredicted:
            print(goldMention.text)

    for predictedMention in doc.predictedMentions.values():
        if predictedMention.id not in predictedToGold:
            falsePositives += 1
            
    precision = float(correct)/(float(correct)+float(falsePositives))
    recall = float(correct)/(float(correct)+float(falseNegatives))
    print(f'Correct mentions: {correct}')
    print(f'Missed mentions: {falseNegatives}')
    print(f'Extra mentions: {falsePositives}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')