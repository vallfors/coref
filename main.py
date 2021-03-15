from argparse import ArgumentParser

from preprocessing.document import *
from preprocessing.config import *
from algorithm.basic_algorithm import *
from preprocessing.stanza_processor import *

def main():
    parser = ArgumentParser()
    parser.add_argument('configFile', help='Path to the config file')

    args = parser.parse_args()
    config = Config(args.configFile)
    conllObj = loadFromFile(config.inputFile)
    doc = documentFromConll(conllObj)
    predictCoreference(doc, config)
    doc.printPredicted()
    #stanzaAnnotator = StanzaAnnotator()
    #stanzaAnnotator.annotateDocument(doc)
    #print(doc.stanzaAnnotation)
    for id, mention in doc.goldMentions.items():
        print("{} {}".format(mention.id, mention.text))
    for id, cluster in doc.goldClusters.items():
         print(id)
         for mentionId in cluster:
             print(doc.goldMentions[mentionId].text)

if __name__ == "__main__":
    main()