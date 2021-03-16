from argparse import ArgumentParser
from evaluate.mention_matching import matchMentions

from preprocessing.document import *
from preprocessing.config import *
from algorithm.basic_algorithm import *
from preprocessing.stanza_processor import *
from algorithm.mention_detection import mentionDetection
from evaluate.mention_matching import *

def main():
    parser = ArgumentParser()
    parser.add_argument('configFile', help='Path to the config file')

    args = parser.parse_args()
    config = Config(args.configFile)

    doc = getDocumentFromFile(config.inputFile)
    stanzaAnnotator = StanzaAnnotator()
    stanzaAnnotator.annotateDocument(doc)
    if not config.useGoldMentions:
        mentionDetection(doc)
    predictCoreference(doc, config)
    matchMentions(doc)
    doc.printPredicted()
    
if __name__ == "__main__":
    main()