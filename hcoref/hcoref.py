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
def getCandidateAntecedents(doc: Document, mention: Mention, maxSentenceDistance: int) -> List[Mention]:
    x = list(doc.predictedMentions.values())
    x.sort(key=operator.attrgetter('startPos'))
    antecedents = []
    for idx, a in enumerate(x):
        if a.stanzaSentence >= mention.stanzaSentence:
            break
        if mention.stanzaSentence - a.stanzaSentence >= maxSentenceDistance:
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

def trainSieveAndUse(docs: List[Document], config: Config, mentionPairs, Y, sieve):
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
    mutualInfo = mutual_info_classif(X, Y, random_state=0)

    if config.debugFeatureSelection:
        print(f'Selected features for {sieve["name"]} sieve')
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

    joblib.dump(model, f'models/{sieve["name"]}Model.joblib')
    joblib.dump(encoder, f'models/{sieve["name"]}OneHotEncoder.joblib')
    joblib.dump(selectedFeatures, f'models/{sieve["name"]}SelectedFeatures.joblib')

    for doc in docs:
        doSievePasses(doc, config.wordVectors, [Sieve(sieve['name'], sieve['sentenceLimit'], sieve['threshold'], model, encoder, selectedFeatures)])

def trainSieves(config: Config, docs: List[Document], wordVectors):
    sieves = config.scaffoldingSieves
    mentionPairs = {}
    Y = {}
    maxSentenceDistance = 0
    for sieve in sieves:
        mentionPairs[sieve['name']] = []
        Y[sieve['name']] = []
        maxSentenceDistance = max(maxSentenceDistance, sieve['sentenceLimit'])
    for doc in docs:
        # Create a cluster for each mention, containing only that mention
        doc.predictedClusters = {}
        for mention in doc.predictedMentions.values():
            doc.predictedClusters[mention.id] = [mention.id]
            mention.predictedCluster = mention.id
        for mention in doc.predictedMentions.values():
            mentionDistance = 0
            for antecedent in getCandidateAntecedents(doc, mention, maxSentenceDistance):
                sievesMatching = 0
                for sieve in sieves:
                    if mention.stanzaSentence - antecedent.stanzaSentence >= sieve['sentenceLimit']:
                        continue
                    if sieve['mentionPos'] == mention.features.upos or sieve['mentionPos'] == 'ANY':
                        if sieve['antecedentPos'] == antecedent.features.upos or sieve['antecedentPos'] == 'ANY':
                            if mention.cluster == antecedent.cluster and mention.cluster != -1:
                                Y[sieve['name']].append(1)
                            else:
                                Y[sieve['name']].append(0)
                            sievesMatching += 1
                            mentionPairs[sieve['name']].append((doc, wordVectors, mention, antecedent, mentionDistance))
                if sievesMatching > 0: # This check makes it perform better, but I don't know why
                    mentionDistance += 1

    np.set_printoptions(suppress=True)
    for sieve in sieves:
        trainSieveAndUse(docs, config, mentionPairs[sieve['name']], Y[sieve['name']], sieve)

def trainAll(docs: List[Document], config: Config):
    wordVectors = KeyedVectors.load_word2vec_format(config.wordVectorFile, binary=True)
    trainSieves(config, docs, wordVectors)