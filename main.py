from argparse import ArgumentParser

from preprocessing.document import *
from preprocessing.config import *
from algorithm.basic_algorithm import *
from preprocessing.stanza_processor import *
from algorithm.mention_detection import mentionDetection
from evaluation.mention_matching import *
from evaluation.cluster_matching import *
from evaluation.score import writeConllForScoring
from algorithm.add_features import addFeatures

def main():
    parser = ArgumentParser()
    parser.add_argument('configFile', help='Path to the config file')

    args = parser.parse_args()
    config = Config(args.configFile)

    stanzaAnnotator = StanzaAnnotator()
    docs = documentsFromTextinatorFile(config.inputFile)
    if not config.useAllDocs:
        if config.docId >= len(docs):
            raise Exception(f'Document id {config.docId} out of bounds, check config')
        docs = [docs[config.docId]]
    for doc in docs:
        stanzaAnnotator.annotateDocument(doc)
        if not config.useGoldMentions:
            mentionDetection(doc)
        else:
            doc.predictedMentions = doc.goldMentions
        addStanzaLinksToGoldMentions(doc)
        addFeatures(doc)
        predictCoreference(doc, config)
        
        matchMentions(doc, config)
        if config.compareClusters:
            compareClusters(doc)
    if config.writeForScoring:
        writeConllForScoring(docs)
    
if __name__ == "__main__":
    main()