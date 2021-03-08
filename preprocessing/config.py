import json

# A configuration for the coreference resolution.
# This object should contain information about which algorithm to run,
# and any specific settings for the algorithm. The config is read from a json file.
class Config:
    useGoldMentions: bool

    def __init__(self, filename: str):
        with open(filename) as f:
            configDict = json.load(f)
        self.useGoldMentions = configDict['useGoldMentions']

if __name__ == "__main__":
    config = Config("preprocessing/config.json")
    print(config.useGoldMentions)