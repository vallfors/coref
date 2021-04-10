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

def trainSieveAndUse(wordVectors, mentionPairs, Y, name):
    model = RandomForestClassifier(max_depth=2, random_state=0)
    X = []
    for mp in mentionPairs:
        X.append(getFeatureVector(*mp))
        doc = mp[0]
    model.fit(X, Y)
    print(model.feature_importances_)
    joblib.dump(model, f'models/{name}Model.joblib')
    doSievePasses(doc, wordVectors, [('commonSieve', model)])

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
    trainSieveAndUse(wordVectors, commonMentionPairs, commonY, 'common')
    trainSieveAndUse(wordVectors, properMentionPairs, properY, 'proper')
    trainSieveAndUse(wordVectors, properCommonMentionPairs, properCommonY, 'properCommon')
    trainSieveAndUse(wordVectors, pronounMentionPairs, pronounY, 'pronoun')
    
def trainAll(docs: List[Document], config: TrainingConfig):
    wordVectors = KeyedVectors.load_word2vec_format("../model.bin", binary=True)
    trainSieves(docs, wordVectors)