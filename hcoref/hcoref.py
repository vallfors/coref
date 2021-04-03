from sklearn.ensemble import RandomForestClassifier
import operator
from typing import List
import joblib
from random import randint

from preprocessing.document import Mention, Document
from hcoref.training_config import TrainingConfig

def getFeatureVector(doc: Document, mention: Mention, antecedent: Mention):
    sentenceDistance = mention.stanzaSentence-antecedent.stanzaSentence
    mentionHeadWord = doc.stanzaAnnotation.sentences[mention.stanzaSentence].words[mention.features.headWord-1].text
    antecedentHeadWord = doc.stanzaAnnotation.sentences[antecedent.stanzaSentence].words[antecedent.features.headWord-1].text
    if mentionHeadWord.lower() == antecedentHeadWord.lower():
        identicalHeadWords = 1
    else:
        identicalHeadWords = 0
    if mention.text.lower() == antecedent.text.lower():
        exactStringMatch = 1
    else:
        exactStringMatch = 0
    return [sentenceDistance, identicalHeadWords, exactStringMatch]

# Returns an ordered list of candidate antecedents for a mention,
# in the order they should be considered in.
def getCandidateAntecedents(doc: Document, mention:Mention) -> List[Mention]:
    x = list(doc.goldMentions.values())
    x.sort(key=operator.attrgetter('startPos'))
    antecedents = []
    sentenceThreshold = 30
    for idx, a in enumerate(x):
        if a.stanzaSentence >= mention.stanzaSentence:
            break
        if a.stanzaSentence - mention.stanzaSentence >= sentenceThreshold:
            continue
        antecedents.append(a)
    sameSentence = []
    while idx < len(x) and x[idx].stanzaSentence == mention.stanzaSentence:
        if x[idx].id == mention.id:
            break
        sameSentence.append(x[idx])
        idx += 1
    antecedents.reverse()
    antecedents = sameSentence + antecedents
    
    return antecedents

def trainSieves(docs: List[Document]):
    properX = []
    properY = []
    commonX = []
    commonY = []
    properCommonX = []
    properCommonY = []
    pronounX = []
    pronounY = []
    for doc in docs:
        for mention in doc.goldMentions.values():
            for antecedent in getCandidateAntecedents(doc, mention):
                if mention.features.upos == 'PROPN' and antecedent.features.upos == 'PROPN':
                    X = properX
                    y = properY
                elif mention.features.upos == 'NOUN' and antecedent.features.upos == 'NOUN':
                    X = commonX
                    y = commonY
                elif mention.features.upos == 'NOUN' and antecedent.features.upos == 'PROPN':
                    X = properCommonX
                    y = properCommonY
                elif mention.features.upos == 'PRON':
                    X = pronounX
                    y = pronounY
                else:
                    continue
                if mention.cluster == antecedent.cluster:
                    y.append(1)
                else:
                    y.append(0)
                X.append(getFeatureVector(doc, mention, antecedent))

    commonModel = RandomForestClassifier(max_depth=2, random_state=0)
    commonModel.fit(commonX, commonY)
    print(commonModel.feature_importances_)
    joblib.dump(commonModel, 'models/commonModel.joblib') 

    properModel = RandomForestClassifier(max_depth=2, random_state=0)
    properModel.fit(properX, properY)
    print(properModel.feature_importances_)
    joblib.dump(properModel, 'models/properModel.joblib')

    properCommonModel = RandomForestClassifier(max_depth=2, random_state=0)
    properCommonModel.fit(properCommonX, properCommonY)
    print(properCommonModel.feature_importances_)
    joblib.dump(properCommonModel, 'models/properCommonModel.joblib')

    pronounModel = RandomForestClassifier(max_depth=2, random_state=0)
    pronounModel.fit(pronounX, pronounY)
    print(pronounModel.feature_importances_)
    joblib.dump(pronounModel, 'models/pronounModel.joblib') 

def trainAll(docs: List[Document], config: TrainingConfig):
    trainSieves(docs)