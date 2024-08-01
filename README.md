# Rock4ML-VLDB25

Artifacts of Paper "When Does Conflict Resolution Help Downstream ML Models?"

## 0 Full-Version Paper

`FullVersion.pdf` is the full version of our submitted paper.

## 1 Artifacts Overview

### 1.1 Organization

In the main directory `Artiacts`, you can see the file tree organization as follows

```bash
.
├── data
│   ├── adult
│   │   ├── adult_clean.csv
│   │   ├── adult_dirty.csv
│   │   ├── exp
│   │   │   └── ...
│   │   └── repaired
│   │       ├── adult_clean.csv
│   │       ├── adult_dirty.csv
│   │       └── adult_repaired_rock.csv
│   ├── default
│   │   └── ...
│   ├── german
│   │   └── ...
│   ├── nursery
│   │   └── ...
│   └── road_safety
│       └── ...
├── rock4ml.yml
└── rock4ml
    ├── rock4ml.py
    ├── run.py
    └── tools
        └── ...
```

### 1.2 Auxiliary File

`rock4ml.yml` contains the conda environment for quick start.

### 1.3 Data

The `./data` directory contains the data used in our experiments, with each dataset in a separate subdirectory `./data/{DATASET_NAME}`.

#### 1.3.1 Two Original Files

File `./data/{DATASET_NAME}/{DATASET_NAME}_clean.csv` contains the clean version of a dataset, while file `./data/{DATASET_NAME}/{DATASET_NAME}_dirty.csv` contains the dirty one. Note that they can be further splitted as training and validation data, for the usage of some baselines, e.g., CPClean.

#### 1.3.2 `exp` directory

For each dataset, the `exp` directory is aligned with the workspace in our efficiency baseline [CPClean](https://github.com/chu-data-lab/CPClean/?tab=readme-ov-file#2-construct-cpclean-space), so that we can run it easily. You can refer to the provided repo for more information. 

In our experiments, for those baselines that need the validation data (e.g., CPClean, BoostClean, DiffPrep), they clean the data / train the model on the training data `X_train_dirty.csv` with the help of the validation data `X_val_dirty.csv` (both are dirty in our settings). For our method and some other baselines (e.g., Picket), they clean the data / train the model on the concatenation of the training and validation data, so they will solely use file `X_train_val_dirty.csv`. 

#### 1.3.3 `repaired` directory

In the `repaired` directory of each dataset, there are three files used for training and evaluation in our method, namely dirty version, clean version and the version repaired by CRMethod Rock. They are the splitted files containting both the training and the validation data, without test data.

Here we use Rock as the representative of CRMethod, and other CRMethods have been public and can also be referred to and test easily. Note that Rock is a commercial software so we do not have its code and we are not authorized by its authors to provide its usage. Here we just provide the the version that has only been fixed once by Rock and experimental results show that Rock4ML still outperforms.

### 1.4 Codes

The `./rock4ml` directory contains the data used in our experiments.

#### 1.4.1 `rock4ml.py`

This file contains the main codes of our method. For simplicity, we directly print the training results of CRMethod, Dirty, and Rock4ML respectively in the `.fit()` method, so we pass the `X_test` and `y_test` into it, and test data are not involved in our method. The `.fit()` method also returns the model which we use to report the effectivenss of Rock4ML. Here we do not implement the `.predict()` method. Since our method will finally converge, we set the max iterations to 15 for quick verfications.

#### 1.4.2 `run.py`

This file can help you conduct a quick run of our method. It takes some parameters as input and you can set their values based on the comments in the codes.

#### 1.4.3 `tools` directory

This directory contains many codes used as tools for our method such as Data Preprocess (aligned with CPClean), Genetic Algorithm, FTTransformer, and so on.

## 2 Quick Start

### 2.1 Set the Environment

First, make sure you have conda, and run the following command in the terminal:

```bash
conda env create -f rock4ml.yml
```

After intalling all required dependencies, activate the conda environment:

```bash
conda activate Rock4ML
```

Now you have an established environment.

### 2.2 Run our method

> We recommend to use CPU to reproduce

In `./rock4ml` directory, run the following examples:

```bash
python3 run.py -r ../data/ -d german -c rock -m LogisticRegression
```

It will first train two models, one with the dirty data and the other with the data repaired by Rock; we record their accuracy as reference/baselines. Then, it will run Rock4ML, identifying critical attributes and pinpointing the influential tuples for refinement. Finally, it gets the refined model and report the final results like this:

```bash
Rock4ML get the F1 score: 0.6215704824976348
Dirty get the F1 score: 0.48051948051948057
rock get the F1 score: 0.5832232690557952
```

------

You can change the dataset by doing:

```bash
python3 run.py -r ../data/ -d nursery -c rock -m LogisticRegression
# change dataset to nursery
```

and get

```bash
Rock4ML get the F1 score: 0.7480440366136221
Dirty get the F1 score: 0.7266642070930794
rock get the F1 score: 0.559842087667984
```
or

```bash
python3 run.py -r ../data/ -d adult -c rock -m LogisticRegression
# change dataset to adult
```

and get

```bash
Rock4ML get the F1 score: 0.7201480303441687
Dirty get the F1 score: 0.6577534791718532
rock get the F1 score: 0.7124421958428682
```

------

You can change the model by doing:

```bash
python3 run.py -r ../data/ -d german -c rock -m MLPClassifier 
# change the model to MLPClassifier (2-nn deep network)
```

and get

```bash
Rock4ML get the F1 score: 0.6567545795243599
Dirty get the F1 score: 0.5820024448913601
rock get the F1 score: 0.5685958574807037
```

or

```bash
python3 run.py -r ../data/ -d german -c rock -m FTTransformer -pi 240 -e 20
# change the model to FTTransformer
```

and get

```bash
Rock4ML get the F1 score: 0.5180111618467782
Dirty get the F1 score: 0.4658259993870135
rock get the F1 score: 0.51338199513382
```

