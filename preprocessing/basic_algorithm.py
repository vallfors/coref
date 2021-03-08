from preprocessing.document import *
from preprocessing.config import *
from preprocessing.read_conll import *

# Placeholder coreference algorithm that clusters all mentions that have exactly the same string
# and no others.
def exactStringMatch(doc: Document, config: Config):
    clusters = {}
    for mention in doc.predictedMentions.values():
        if mention.text not in clusters:
            clusters[mention.text] = []
        clusters[mention.text].append(mention.id)
    
    idCounter = 0
    doc.predictedClusters = {}
    for cluster in clusters.values():
        doc.predictedClusters[idCounter] = cluster
        idCounter += 1


# Takes a coreference document that has predicted mentions already set.
# Runs a place holder "coreference resolution" to fill in predicted clusters
def predictCoreference(doc: Document, config: Config):
    if config.useGoldMentions:
        if doc.goldMentions == None:
            raise Exception("Configured to use gold mentions, but no gold mentions set!")
        doc.predictedMentions = doc.goldMentions
    elif doc.predictedMentions == None:
        raise Exception("Predicted mentions have not been set!")

    if config.algorithm == "exactstringmatch":
        exactStringMatch(doc, config)
    else:
        raise Exception("Configured to use algorithm {}, but no such algorithm is implemented".format(config.algorithm))