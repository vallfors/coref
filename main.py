from preprocessing.document import *
from preprocessing.config import *
from algorithm.basic_algorithm import *
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('configFile', help='Path to the config file')

    args = parser.parse_args()
    config = Config(args.configFile)
    obj = loadFromFile(config.inputFile)
    doc = Document(obj)
    predictCoreference(doc, config)
    for id, mention in doc.predictedMentions.items():
        print("{} {}".format(mention.id, mention.text))
    for id, cluster in doc.predictedClusters.items():
        print(id)
        for mentionId in cluster:
            print(doc.predictedMentions[mentionId].text)

if __name__ == "__main__":
    main()