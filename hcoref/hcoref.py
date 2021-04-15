from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import operator
from typing import List
import joblib
import numpy as np
from gensim.models import KeyedVectors

from preprocessing.document import Mention, Document
from preprocessing.config import Config
from hcoref.feature_vector import getFeatureVector, getStringFeatureVector
from algorithm.scaffolding import doSievePasses, Sieve

# Returns an ordered list of candidate antecedents for a mention,
# in the order they should be considered in.
def getCandidateAntecedents(doc: Document, mention:Mention) -> List[Mention]:
    x = list(doc.predictedMentions.values())
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

def trainSieveAndUse(docs: List[Document], config: Config, wordVectors, mentionPairs, Y, name):
    if config.maxDepth == -1: # No max depth
        model = RandomForestClassifier(random_state=0)
    else:
        model = RandomForestClassifier(max_depth=config.maxDepth, random_state=0)
    X = []
    stringFeatureVectors = []
    for mp in mentionPairs:
        X.append(getFeatureVector(*mp))
        stringFeatureVectors.append(getStringFeatureVector(*mp))
    
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(stringFeatureVectors)
    stringFeatures = encoder.transform(stringFeatureVectors).toarray()
    X = np.concatenate((X, stringFeatures), 1)
    
    manualFeatures = ['sentenceDistance', 'mentionDistance', 'minimumClusterDistance',
        'antecedentClusterSize', 'mentionClusterSize', 'exactStringMatch', 'identicalHeadWords', 
        'identicalHeadWordsAndProper', 'numberMatch', 'genderMatch', 'naturalGenderMatch',
        'animacyMatch', 'nerMatch', 'clusterHeadwordMatch', 'clusterProperHeadwordMatch',
        'clusterGenitiveMatch', 'clusterLemmaHeadMatch', 'wordvecHeadwordDistance']
    allFeatureNames = np.concatenate((manualFeatures, encoder.get_feature_names(['mention_deprel', 'mention_headWord', 'mentionNextWordUpos', 'mentionNextWordText'])), 0)
    
    selectedFeatures = []
    mutualInfo = mutual_info_classif(X, Y)
    if config.debugFeatureSelection:
        print(f'Selected features for {name} sieve')
    for i, featureName in enumerate(allFeatureNames):
        binary = True
        zeroes = 0
        ones = 0
        for j in range(0,len(X)):
            if X[j][i] == 0:
                zeroes+=1
            elif X[j][i] == 1:
                ones+=1
            else:
                binary = False
        if binary and (ones < config.allowedFeatureRarity or zeroes < config.allowedFeatureRarity):
            rareBinary = True
        else:
            rareBinary = False
        if not rareBinary and mutualInfo[i] > config.minimalMutualInformation:
            selectedFeatures.append(1)
            if config.debugFeatureSelection:
                print(f'{featureName} {mutualInfo[i]}')
        else:
            selectedFeatures.append(0)
    if config.debugFeatureSelection:
        print()

    # This part removes all features that are not selected from X
    X = np.transpose(X)
    tempX = []
    for i in range(0, len(X)):
        if selectedFeatures[i]:
            tempX.append(X[i])
    X = np.transpose(np.array(tempX))

    model.fit(X, Y)

    joblib.dump(model, f'models/{name}Model.joblib')
    joblib.dump(encoder, f'models/{name}OneHotEncoder.joblib')
    joblib.dump(selectedFeatures, f'models/{name}SelectedFeatures.joblib')

    for s in config.scaffoldingSieves:
        if s['name'] == name:
            sieve = s
    for doc in docs:
        doSievePasses(doc, wordVectors, [Sieve(sieve['name'], sieve['sentenceLimit'], sieve['threshold'], model, encoder, selectedFeatures)])

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
        doc.predictedClusters = {}
        for mention in doc.predictedMentions.values():
            doc.predictedClusters[mention.id] = [mention.id]
            mention.predictedCluster = mention.id
        for mention in doc.predictedMentions.values():
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
                if mention.cluster == antecedent.cluster and mention.cluster != -1:
                    y.append(1)
                else:
                    y.append(0)
                mentionPairs.append((doc, wordVectors, mention, antecedent, mentionDistance))
                mentionDistance += 1
    
    np.set_printoptions(suppress=True)
    trainSieveAndUse(docs, config, wordVectors, properMentionPairs, properY, 'proper')
    trainSieveAndUse(docs, config, wordVectors, commonMentionPairs, commonY, 'common')
    trainSieveAndUse(docs, config, wordVectors, properCommonMentionPairs, properCommonY, 'properCommon')
    trainSieveAndUse(docs, config, wordVectors, pronounMentionPairs, pronounY, 'pronoun')

def trainAll(docs: List[Document], config: Config):
    wordVectors = KeyedVectors.load_word2vec_format(config.wordVectorFile, binary=True)
    trainSieves(config, docs, wordVectors)