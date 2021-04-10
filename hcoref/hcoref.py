from algorithm.scaffolding import doSievePasses
from sklearn.ensemble import RandomForestClassifier
import operator
from typing import List
import joblib
import numpy as np
from gensim.models import KeyedVectors

from preprocessing.document import Mention, Document
from hcoref.training_config import TrainingConfig
from hcoref.feature_vector import getFeatureVector

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

def trainSieves(docs: List[Document], wordVectors):
    properMentionPairs = []
    properY = []
    commonMentionPairs = []
    commonY = []
    properCommonMentionPairs = []
    properCommonY = []
    pronounMentionPairs = []
    pronounY = []
    for doc in docs:
        # Create a cluster for each mention, containing only that mention
        doc.predictedMentions = doc.goldMentions
        doc.predictedClusters = {}
        for mention in doc.predictedMentions.values():
            doc.predictedClusters[mention.id] = [mention.id]
            mention.predictedCluster = mention.id
        for mention in doc.goldMentions.values():
            mentionDistance = 0
            for antecedent in getCandidateAntecedents(doc, mention):
                if mention.features.upos == 'PROPN' and antecedent.features.upos == 'PROPN':
                    mentionPairs = properMentionPairs
                    y = properY
                elif mention.features.upos == 'NOUN' and antecedent.features.upos == 'NOUN':
                    mentionPairs = commonMentionPairs
                    y = commonY
                elif mention.features.upos == 'NOUN' and antecedent.features.upos == 'PROPN':
                    mentionPairs = properCommonMentionPairs
                    y = properCommonY
                elif mention.features.upos == 'PRON':
                    mentionPairs = pronounMentionPairs
                    y = pronounY
                else:
                    continue
                if mention.cluster == antecedent.cluster:
                    y.append(1)
                else:
                    y.append(0)
                mentionPairs.append((doc, wordVectors, mention, antecedent, mentionDistance))
                mentionDistance += 1
    
    np.set_printoptions(suppress=True)
    commonModel = RandomForestClassifier(max_depth=2, random_state=0)
    commonX = []
    for mp in commonMentionPairs:
        commonX.append(getFeatureVector(*mp))
        doc = mp[0]
    commonModel.fit(commonX, commonY)
    print(commonModel.feature_importances_)
    joblib.dump(commonModel, 'models/commonModel.joblib') 
    doSievePasses(doc, wordVectors, [('commonSieve', commonModel)])
    
    properModel = RandomForestClassifier(max_depth=2, random_state=0)
    properX = []
    for mp in properMentionPairs:
        properX.append(getFeatureVector(*mp))
        doc = mp[0]
    properModel.fit(properX, properY)
    print(properModel.feature_importances_)
    joblib.dump(properModel, 'models/properModel.joblib')
    doSievePasses(doc, wordVectors, [('properSieve', properModel)])

    properCommonModel = RandomForestClassifier(max_depth=2, random_state=0)
    properCommonX = []
    for mp in properCommonMentionPairs:
        properCommonX.append(getFeatureVector(*mp))
        doc = mp[0]
    properCommonModel.fit(properCommonX, properCommonY)
    print(properCommonModel.feature_importances_)
    joblib.dump(properCommonModel, 'models/properCommonModel.joblib')
    doSievePasses(doc, wordVectors, [('properCommonSieve', properCommonModel)])

    pronounModel = RandomForestClassifier(max_depth=2, random_state=0)
    pronounX = []
    for mp in pronounMentionPairs:
        pronounX.append(getFeatureVector(*mp))
        doc = mp[0]
    pronounModel.fit(pronounX, pronounY)
    print(pronounModel.feature_importances_)
    joblib.dump(pronounModel, 'models/pronounModel.joblib') 

def trainAll(docs: List[Document], config: TrainingConfig):
    wordVectors = KeyedVectors.load_word2vec_format("../model.bin", binary=True)
    trainSieves(docs, wordVectors)