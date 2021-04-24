from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import operator
from typing import List
import joblib
import numpy as np
from gensim.models import KeyedVectors
import random
import time

from preprocessing.document import Mention, Document
from preprocessing.config import Config
from hcoref.feature_vector import getFeatureVector, getStringFeatureVector
from algorithm.scaffolding import doSievePasses, Sieve, useSieve

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
    print(sieve['name'])
    print(f'Length of mentionPairs: {len(mentionPairs)}')
    print(f'Len of Y: {len(Y)}')
    startTime = time.time()
    print(sieve['name'])
    if config.maxDepth == -1: # No max depth
        model = RandomForestClassifier(random_state=0)
    else:
        model = RandomForestClassifier(max_depth=config.maxDepth, random_state=0)
    X = []
    stringFeatureVectors = []
    for mp in mentionPairs:
        X.append(getFeatureVector(*mp))
        stringFeatureVectors.append(getStringFeatureVector(*mp))
    
    print(f'One hot encoding {time.time() - startTime}')
    startTime = time.time()
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
    
    print(f'Doing feature selection {time.time() - startTime}')
    startTime = time.time()
    selectedFeatures = []
    mutualInfo = mutual_info_classif(X, Y, random_state=0)

    if config.debugFeatureSelection:
        print(f'Selected features for {sieve["name"]} sieve')
    toPrint = []
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
                toPrint.append((mutualInfo[i], featureName))
        else:
            selectedFeatures.append(0)
    if config.debugFeatureSelection:
        toPrint.sort()
        toPrint.reverse()
        for mi, featureName in toPrint:
            print(f'{featureName} {mi}')
        print()

    # This part removes all features that are not selected from X
    X = np.transpose(X)
    tempX = []
    for i in range(0, len(X)):
        if selectedFeatures[i]:
            tempX.append(X[i])
    X = np.transpose(np.array(tempX))

    print(f'Fitting model {time.time() - startTime}')
    startTime = time.time()
    model.fit(X, Y)

    joblib.dump(model, f'models/{sieve["name"]}Model.joblib')
    joblib.dump(encoder, f'models/{sieve["name"]}OneHotEncoder.joblib')
    joblib.dump(selectedFeatures, f'models/{sieve["name"]}SelectedFeatures.joblib')

    print(f'Applying sieve {time.time() - startTime}')
    startTime = time.time()
    for doc in docs:
        doSievePasses(doc, config.wordVectors, [Sieve(sieve['name'], sieve['sentenceLimit'], sieve['threshold'], model, encoder, selectedFeatures)])

def trainSieves(config: Config, docs: List[Document], wordVectors):
    sieves = config.scaffoldingSieves
    positiveMentionPairs = {}
    negativeMentionPairs = {}
    maxSentenceDistance = 0
    for sieve in sieves:
        positiveMentionPairs[sieve['name']] = []
        negativeMentionPairs[sieve['name']] = []
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
                                positiveMentionPairs[sieve['name']].append((doc, wordVectors, mention, antecedent, mentionDistance))
                            else:
                                negativeMentionPairs[sieve['name']].append((doc, wordVectors, mention, antecedent, mentionDistance))
                            sievesMatching += 1
                if sievesMatching > 0: # This check makes it perform better, but I don't know why
                    mentionDistance += 1

    np.set_printoptions(suppress=True)
    if not config.useSubsampling:
        for sieve in sieves:
            mentionPairs = negativeMentionPairs[sieve['name']] + positiveMentionPairs[sieve['name']]
            Y = [0] * len(negativeMentionPairs[sieve['name']]) + [1] * len(positiveMentionPairs[sieve['name']])
            trainSieveAndUse(docs, config, mentionPairs, Y, sieve)
            
    if config.useSubsampling:
        # First train with a random subset of negative examples
        subsampledY = {}
        subsampledPairs = {}
        random.seed(0)
        for sieve in sieves:
            numSamples = int(0.2*len(negativeMentionPairs[sieve['name']])) 
            samples = random.sample(negativeMentionPairs[sieve['name']], numSamples)
            subsampledPairs[sieve['name']] = samples + positiveMentionPairs[sieve['name']]
            subsampledY[sieve['name']] = [0] * len(samples) + [1] * len(positiveMentionPairs[sieve['name']])
            print(sieve['name'])
            print(f'Negative: {len(samples)} Positive: {len(positiveMentionPairs[sieve["name"]])}')
            print(len(subsampledPairs[sieve['name']]))
        for sieve in sieves:
            trainSieveAndUse(docs, config, subsampledPairs[sieve['name']], subsampledY[sieve['name']], sieve)

        # Reset all the clusters, and retrain with the most difficult negative examples.
        for doc in docs:
            # Create a cluster for each mention, containing only that mention
            doc.predictedClusters = {}
            for mention in doc.predictedMentions.values():
                doc.predictedClusters[mention.id] = [mention.id]
                mention.predictedCluster = mention.id
        difficultPairs = {}
        difficultY = {}
        for sieve in sieves:
            difficulties = []
            name = sieve['name']
            model = joblib.load(f'models/{name}Model.joblib')
            encoder = joblib.load(f'models/{name}OneHotEncoder.joblib')
            selectedFeatures = joblib.load(f'models/{name}SelectedFeatures.joblib')
            for idx, mentionPair in enumerate(negativeMentionPairs[sieve['name']]):
                positiveLikelihood = useSieve(Sieve(sieve['name'], sieve['sentenceLimit'], sieve['threshold'], model, encoder, selectedFeatures), *mentionPair)
                difficulties.append((1-positiveLikelihood, idx))
            difficulties.sort()
            numSamples = int(0.2*len(negativeMentionPairs[sieve['name']]))
            difficultNegatives = []
            for i in range(0, numSamples):
                difficultNegatives.append(negativeMentionPairs[sieve['name']][difficulties[i][1]])
            difficultPairs[sieve['name']] = difficultNegatives + positiveMentionPairs[sieve['name']]
            difficultY[sieve['name']] = [0] * len(difficultNegatives) + [1] * len(positiveMentionPairs[sieve['name']])
        for sieve in sieves:
            trainSieveAndUse(docs, config, difficultPairs[sieve['name']], difficultY[sieve['name']], sieve)

def trainAll(docs: List[Document], config: Config):
    wordVectors = KeyedVectors.load_word2vec_format(config.wordVectorFile, binary=True)
    trainSieves(config, docs, wordVectors)