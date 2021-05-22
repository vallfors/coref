from argparse import ArgumentParser

from transformers import pipeline
from transformers.utils.dummy_pt_objects import load_tf_weights_in_bert_generation

from preprocessing.document import *
from preprocessing.config import *
from algorithm.basic_algorithm import *
from preprocessing.stanza_processor import *
from algorithm.mention_detection import mentionDetection
from evaluation.mention_matching import *
from evaluation.cluster_matching import *
from evaluation.score import writeConllForScoring
from algorithm.add_features import addFeatures

def logGreen(message: str):
    print('\033[92m' + message + '\033[0m')

def main():
    logGreen('Starting coreference prediction procedure')
    parser = ArgumentParser()
    parser.add_argument('configFile', help='Path to the config file')

    args = parser.parse_args()
    config = Config(args.configFile)

    stanzaAnnotator = StanzaAnnotator()
    nerPipeline = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
    docs = documentsFromTextinatorFile(config.inputFile)
    if not config.useAllDocs:
        if config.docId >= len(docs):
            raise Exception(f'Document id {config.docId} out of bounds, check config')
        docs = [docs[config.docId]]

    for id, doc in enumerate(docs):
        logGreen(f'Processing document {id}')
        load_tf_weights_in_bert_generation('Adding stanza annotation')
        stanzaAnnotator.annotateDocument(doc)
        if not config.useGoldMentions:
            logGreen('Doing mention detection')
            mentionDetection(doc)
        else:
            doc.predictedMentions = doc.goldMentions
        logGreen('Preprocessing')
        addStanzaLinksToGoldMentions(doc)
        addFeatures(doc, nerPipeline)
        logGreen('Coreference prediction')
        predictCoreference(doc, config)
        
        logGreen('Evaluation')
        matchMentions(doc, config)
        if config.compareClusters:
           compareClusters(doc)
        print()

    #printStatistics(docs)
    if config.writeForScoring:
        writeConllForScoring(docs)
    
if __name__ == "__main__":
    main()