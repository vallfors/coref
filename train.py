from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
from transformers import pipeline

from preprocessing.document import documentsFromTextinatorFile
from preprocessing.config import Config
from preprocessing.stanza_processor import StanzaAnnotator, addStanzaLinksToGoldMentions
from algorithm.add_features import addFeatures
from hcoref.hcoref import trainAll

def main():
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
    for doc in docs:
        stanzaAnnotator.annotateDocument(doc)
        doc.predictedMentions = doc.goldMentions
        addStanzaLinksToGoldMentions(doc)
        addFeatures(doc, nerPipeline)
    
    trainAll(docs, config)
    
        
if __name__ == "__main__":
    main()