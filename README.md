# coref

Coref is a tool for doing coreference resolution for Swedish texts. It's currently under development.

## Usage

In order to try the tool, run `python3 main.py <path/to/your/config/file>`. For example `python3 main.py config/config.json` to use the example config file. The configuration file controls which algorithm is used and the parameters for it, as well as which input file is used. If the input is raw text, it should be saved with the .txt extension to be processed correctly.

### Training
To train new models for the ML algorithm, run `python3 train.py <path/to/configfile>`. The models will be stored in the correct directory and with the correct file names. The same config file can be used for training and evaluating the ML algorithm, and doing so will ensure that the algorithm is trained and evaluated using the same settings. To try it out you can start with the config file `config/trainAndEvaluate`. The only thing that needs to be changed is the path to your data files. This config is a good starting point to change to try different settings, such as tree depth, merging thresholds etc. For a complete description of all the options, see Configuration.

### Configuration
| Option        | Possible Values           | Description  |
| ------------- |:-------------| -----|
| algorithm      | "hcoref", "multipass" | The algorithm used |
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







