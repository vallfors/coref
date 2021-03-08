from preprocessing.document import *
from preprocessing.config import *
from preprocessing.basic_algorithm import *

def main():
    obj = loadFromFile('./data/suc-core-conll/aa05_fixed.conll')
    doc = Document(obj)
    config = Config("preprocessing/config.json")
    predictCoreference(doc, config)
    for id, mention in doc.predictedMentions.items():
        print("{} {}".format(mention.id, mention.text))
    for id, cluster in doc.predictedClusters.items():
        print(id)
        for mentionId in cluster:
            print(doc.predictedMentions[mentionId].text)

if __name__ == "__main__":
    main()