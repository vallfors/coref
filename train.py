from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
from transformers import pipeline

from preprocessing.document import documentsFromTextinatorFile
from preprocessing.config import Config
from preprocessing.stanza_processor import StanzaAnnotator, addStanzaLinksToGoldMentions
from algorithm.add_features import addFeatures
from hcoref.hcoref import trainAll
from hcoref.training_config import TrainingConfig

def main():
    parser = ArgumentParser()
    parser.add_argument('configFile', help='Path to the training config file')

    args = parser.parse_args()
    trainingConfig = TrainingConfig(args.configFile)

    stanzaAnnotator = StanzaAnnotator()
    nerPipeline = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
    docs = documentsFromTextinatorFile(trainingConfig.inputFile)

    for doc in docs:
        stanzaAnnotator.annotateDocument(doc)
        doc.predictedMentions = doc.goldMentions
        addStanzaLinksToGoldMentions(doc)
        addFeatures(doc, nerPipeline)
    
    trainAll(docs, trainingConfig)
        
if __name__ == "__main__":
    main()