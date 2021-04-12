from sklearn.ensemble import RandomForestClassifier
import operator
from typing import List
import joblib
import numpy as np
from gensim.models import KeyedVectors

from preprocessing.document import Mention, Document
from preprocessing.config import Config
from hcoref.feature_vector import getFeatureVector
from algorithm.scaffolding import doSievePasses, Sieve

# Returns an ordered list of candidate antecedents for a mention,
# in the order they should be considered in.
def getCandidateAntecedents(doc: Document, mention:Mention) -> List[Mention]:
    x = list(doc.goldMentions.values())
    x.sort(key=operator.attrgetter('startPos'))
    antecedents = []
    sentenceThreshold = 15
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

def trainSieveAndUse(config: Config, wordVectors, mentionPairs, Y, name):
    if config.maxDepth == -1: # No max depth
        model = RandomForestClassifier(random_state=0)
    else:
        model = RandomForestClassifier(max_depth=config.maxDepth, random_state=0)
    X = []
    for mp in mentionPairs:
        X.append(getFeatureVector(*mp))
        doc = mp[0]
    model.fit(X, Y)
    print(model.feature_importances_)
    joblib.dump(model, f'models/{name}Model.joblib')
    for s in config.scaffoldingSieves:
        if s['name'] == name:
            sieve = s
    doSievePasses(doc, wordVectors, [Sieve(sieve['name'], sieve['sentenceLimit'], sieve['threshold'], model)])

def trainSieves(config: Config, docs: List[Document], wordVectors):
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
    trainSieveAndUse(config, wordVectors, properMentionPairs, properY, 'proper')
    trainSieveAndUse(config, wordVectors, commonMentionPairs, commonY, 'common')
    trainSieveAndUse(config, wordVectors, properCommonMentionPairs, properCommonY, 'properCommon')
    trainSieveAndUse(config, wordVectors, pronounMentionPairs, pronounY, 'pronoun')

def trainAll(docs: List[Document], config: Config):
    wordVectors = KeyedVectors.load_word2vec_format(config.wordVectorFile, binary=True)
    trainSieves(config, docs, wordVectors)