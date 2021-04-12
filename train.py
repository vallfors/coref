from evaluation.mention_matching import matchMentions
from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
from transformers import pipeline

from preprocessing.document import documentsFromTextinatorFile
from preprocessing.config import Config
from preprocessing.stanza_processor import StanzaAnnotator, addStanzaLinksToGoldMentions
from algorithm.add_features import addFeatures
from hcoref.hcoref import trainAll
from algorithm.mention_detection import mentionDetection

def main():
    parser = ArgumentParser()
    parser.add_argument('configFile', help='Path to the training config file')

    args = parser.parse_args()
    config = Config(args.configFile)

    stanzaAnnotator = StanzaAnnotator()
    nerPipeline = pipeline('ner', model='KB/bert-base-swedish-cased-ner', tokenizer='KB/bert-base-swedish-cased-ner')
    docs = documentsFromTextinatorFile(config.trainingInputFile)

    for doc in docs:
        stanzaAnnotator.annotateDocument(doc)
        if not config.useGoldMentions:
            mentionDetection(doc)
        else:
            doc.predictedMentions = doc.goldMentions
        addStanzaLinksToGoldMentions(doc)
        matchMentions(doc, config)
        for mention in doc.predictedMentions.values():
            if mention.id in doc.predictedToGold:
                mention.cluster = doc.goldMentions[doc.predictedToGold[mention.id]].cluster
            else:
                mention.cluster = -1 # Mention belongs to no gold cluster, since it corresponds to no gold mention.
        addFeatures(doc, nerPipeline)
    
    trainAll(docs, config)
        
if __name__ == "__main__":
    main()