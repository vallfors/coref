# coref

Coref is a tool for performing coreference resolution for Swedish texts. It implements two different algorithms, one machine learning based and one rule-based. The algorithms are inspired by two algorithms for English, described in the following papers:

Lee, Heeyoung, Mihai Surdeanu, and Dan Jurafsky. "A scaffolding approach to coreference resolution integrating statistical and rule-based models." Natural Language Engineering 23.5 (2017): 733-762.

Lee, Heeyoung, et al. "Deterministic coreference resolution based on entity-centric, precision-ranked rules." Computational linguistics 39.4 (2013): 885-916.

## Usage

In order to try the tool, run `python3 main.py <path/to/your/config/file>`. For example `python3 main.py config/goldRuleBest.json` to run one version of the rule-based algorithm. The configuration file controls which algorithm is used and the parameters for it, as well as which input file is used. No pretrained models are provided, so in order to predict using the ML algorithm it must be trained first, see below.

### Input Format

The input file should be provided in the Textinator format, with a .json ending, or as a raw txt file (only for prediction). The path to the input file must be given in the config file.

### Training
To train new models for the ML algorithm, run `python3 train.py <path/to/configfile>`, for example `python3 main.py config/goldMLBest.json`. The models will be stored in the correct directory and with the correct file names. The same config file can be used for training and evaluating the ML algorithm, and doing so will ensure that the algorithm is trained and evaluated using the same settings. The only thing that needs to be changed is the path to your data files, but many more settings can be tweaked. For a complete description of all the options, see Configuration.

### Configuration
| Option        | Possible Values           | Description  |
| ------------- |:-------------| -----|
| algorithm      | "hcoref", "multipass" | The algorithm used. "hcoref" is the ML based algorithm, and "multipass" the rule-based algorithm. |
| useGoldMentions      | boolean      |   If true, the mentions in the input will be used when predicting clusters, rather than using a mention prediction algorithm |
| inputFile | a file path, for instance  "data/input.json"     |    The input file used for evaluation |
| trainingInputFile | a file path, for instance  "data/training.json"     |    The input file used for training |
| useAllDocs | boolean     |    Whether all documents in the file should be evaluated, or only one |
| docId | int    |    The index (starting at 0) of the file to be evaluated. Used if useAllDocs = false |
| compareClusters | boolean    |    If true, you will get a walk through comparison between gold and predicted clusters, after each evaluated document |
| debugMentionDetection | boolean     |    If true, all missed and extra mentions will be printed, as well as mention detection statistics |
| writeForScoring | boolean  |    If true, the results of the prediction will be printed in a format that can be used by the official ConLL-2011 scorer |
| scaffoldingSieves | a list of objects, each with name, sentenceLimit and threshold    |    Descriptions of the scaffolding sieves. |
| multipassSieves |    a list of strings    |    The names of the sieves to be used in the multipass algorithm. They will be applied in the order they are given. See below for a list of all available sieves. |
| debugFeatureSelection | boolean | If true, debugging info will be shown while doing feature selection. |
| minimalMutualInformation | float | Feature selection parameter, see the paper for more details. |
| allowedFeatureRarity | int | Feature selection parameter, see the paper for more details. |
| useSubsampling | boolean | Whether to use all training examples when training, or only a sample. This can be necessary to enable when a large amount of training samples are available. |
| orderAntecedentsBySubject| boolean | Regulates the order mentions are considered for coreference. If true, subjects will be prioritized. |
| features | list of strings | The features (except features with string values) to be used in the ML algorithm. See the example configs or the code for available features. |
| stringFeatures | list of strings | The features with string values to be used in the ML algorithm. See the example configs or the code for available features. |
| wordVectorFile | path as string, e.g. "../model.bin" | If word vector features are enabled, this must be provided. |
| allowIndefiniteAnaphor | boolean | If true, allows indefinite mentions to be marked as coreferent with previous mentions. |
| maxDepth | int | The maximum depth of the decision trees in the ML algorithm. If set to -1, no maximum depth is used. |


## Navigating the Repository

The core of the algorithm implementations are in the algorithm directory. `multipass.py` contains the core of the rule-based algorithm, while `scaffolding.py` (name taken from the original paper) contains the prediction procedure for the ML algorithm. Training and features for the ML algorithm can be found in the `hcoref` directory. Due to the exploratory nature of this project, there is a lot of code that was not used in the final solution, but that can be reenabled by changing config files, or that may be in incomplete condition (such as support for alternative file formats).
