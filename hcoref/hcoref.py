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
import os, psutil


from preprocessing.document import Mention, Document
from preprocessing.config import Config
from hcoref.feature_vector import getFeatureVector, getStringFeatureVector
from algorithm.scaffolding import doSievePasses, Sieve, useSieve
from algorithm.candidate_antecedents import getCandidateAntecedents

def trainSieveAndUse(docs: List[Document], config: Config, mentionPairs, Y, sieve):
    name = sieve['name']
    print(f'Started training sieve {name}')
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 
    if config.maxDepth == -1: # No max depth
        model = RandomForestClassifier(random_state=0)
    else:
        model = RandomForestClassifier(max_depth=config.maxDepth, random_state=0)
    X = []
    stringFeatureVectors = []
    print('Before string feature vectors')
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 
    for mp in mentionPairs:
        X.append(getFeatureVector(*mp, config.features))
        stringFeatureVectors.append(getStringFeatureVector(*mp, config.stringFeatures))
    
    print('Before onehot encoder')
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(stringFeatureVectors)
    stringFeatures = encoder.transform(stringFeatureVectors).toarray()
    print('Len after feature encoding:')
    print(len(stringFeatures))
    print(len(stringFeatures[0]))
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss)
    X = np.concatenate((X, stringFeatures), 1)
    print('After onehot encoder')
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 
    allFeatureNames = np.concatenate((config.features, encoder.get_feature_names(config.stringFeatures)), 0)
    
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

    print('After feature selection')
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 

    model.fit(X, Y)
    print('After fitting')
    process = psutil.Process(os.getpid())
    print(process.memory_info().rss) 

    joblib.dump(model, f'models/{sieve["name"]}Model.joblib')
    joblib.dump(encoder, f'models/{sieve["name"]}OneHotEncoder.joblib')
    joblib.dump(selectedFeatures, f'models/{sieve["name"]}SelectedFeatures.joblib')

    for doc in docs:
        doSievePasses(config, doc, config.wordVectors, [Sieve(sieve['name'], sieve['sentenceLimit'], sieve['threshold'], model, encoder, selectedFeatures, sieve)])

def trainSieves(config: Config, docs: List[Document], wordVectors):
    sieves = config.scaffoldingSieves
    positiveMentionPairs = {}
    negativeMentionPairs = {}
    maxSentenceDistance = 0
    for sieve in sieves:
        positiveMentionPairs[sieve['name']] = []
        negativeMentionPairs[sieve['name']] = []
        maxSentenceDistance = max(maxSentenceDistance, sieve['sentenceLimit'])
    process = psutil.Process(os.getpid())
    print('Before sample collection:')
    print(process.memory_info().rss) 
    for doc in docs:
        # Create a cluster for each mention, containing only that mention
        doc.predictedClusters = {}
        for mention in doc.predictedMentions.values():
            doc.predictedClusters[mention.id] = [mention.id]
            mention.predictedCluster = mention.id
        for mention in doc.predictedMentions.values():
            mentionDistance = 0
            positiveFoundForThisMention = False
            for antecedent in getCandidateAntecedents(config, doc, mention, maxSentenceDistance):
                if positiveFoundForThisMention:
                    break
                if mention.cluster == antecedent.cluster and mention.cluster != -1:
                    positiveFoundForThisMention = True
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
                mentionDistance += 1

    if not config.useSubsampling:
        for sieve in sieves:
            mentionPairs = negativeMentionPairs[sieve['name']] + positiveMentionPairs[sieve['name']]
            Y = [0] * len(negativeMentionPairs[sieve['name']]) + [1] * len(positiveMentionPairs[sieve['name']])
            trainSieveAndUse(docs, config, mentionPairs, Y, sieve)

    process = psutil.Process(os.getpid())
    print('Collected all samples:')
    print(process.memory_info().rss)        
    if config.useSubsampling:
        # First train with a random subset of negative examples
        subsampledY = {}
        subsampledPairs = {}
        random.seed(0)
        for sieve in sieves:
            if sieve['subsample']:
                numSamples = int(0.2*len(negativeMentionPairs[sieve['name']])) 
                samples = random.sample(negativeMentionPairs[sieve['name']], numSamples)
            else:
                samples = negativeMentionPairs[sieve['name']]
            subsampledPairs[sieve['name']] = samples + positiveMentionPairs[sieve['name']]
            subsampledY[sieve['name']] = [0] * len(samples) + [1] * len(positiveMentionPairs[sieve['name']])
        for sieve in sieves:
            trainSieveAndUse(docs, config, subsampledPairs[sieve['name']], subsampledY[sieve['name']], sieve)

        process = psutil.Process(os.getpid())
        print('First pass done:')
        print(process.memory_info().rss) 
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
            featureVectors = []
            for mentionPair in negativeMentionPairs[sieve['name']]:
                nonStringFeatures = getFeatureVector(*mentionPair, config.features)
                stringFeatures = getStringFeatureVector(*mentionPair, config.stringFeatures)
                stringFeatures = encoder.transform([stringFeatures]).toarray()
                featureVector = np.concatenate((nonStringFeatures, stringFeatures[0]), 0)
                featuresToUse = []
                for i in range(0, len(featureVector)):
                    if selectedFeatures[i]:
                        featuresToUse.append(featureVector[i])
                featureVectors.append(featuresToUse)
            results = model.predict_proba(featureVectors)
            for i in range(0, len(results)):
                difficulties.append((results[i][0], i))
            difficulties.sort()
            numSamples = int(0.2*len(negativeMentionPairs[sieve['name']]))
            difficultNegatives = []

            for i in range(0, numSamples):
                difficultNegatives.append(negativeMentionPairs[sieve['name']][difficulties[i][1]])
            
            if not sieve['subsample']:
                difficultNegatives = negativeMentionPairs[sieve['name']]
            difficultPairs[sieve['name']] = difficultNegatives + positiveMentionPairs[sieve['name']]
            difficultY[sieve['name']] = [0] * len(difficultNegatives) + [1] * len(positiveMentionPairs[sieve['name']])
        process = psutil.Process(os.getpid())
        print('Collected all difficult samples:')
        print(process.memory_info().rss) 
        for sieve in sieves:
            trainSieveAndUse(docs, config, difficultPairs[sieve['name']], difficultY[sieve['name']], sieve)
        process = psutil.Process(os.getpid())
        print('Finished second training:')
        print(process.memory_info().rss) 

def trainAll(docs: List[Document], config: Config):
    wordVectors = KeyedVectors.load_word2vec_format(config.wordVectorFile, binary=True)
    trainSieves(config, docs, wordVectors)