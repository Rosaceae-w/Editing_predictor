# RNA editing predictor

Predicts whether the center site (position 51) of a 101-nt sequence is edited, and its editing level.

## Setup (run it once)

Tested with Python 3.13, older Python versions may or may not work. Runs on CPU. Creating a clean new environment is recommended. Use either option below.

**Option A: conda**

```
conda create -n editing_predictor python=3.13 -y
conda activate editing_predictor
pip install -r requirements.txt
```

**Option B: Python venv**

```
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Activate the environment again in each new terminal before running `predict.py`.

To remove the environment when no longer needed:

```
# conda
conda deactivate
conda env remove -n editing_predictor

# venv
deactivate
rm -rf venv                       # Windows: rmdir /s /q venv
```

## Usage

```
# one or more sequences
python predict.py -s GGAAATGTTTGATCAGTTGTGTATGCTGAAGAAGTTCGTGCTCTCGCACTAGAGAACTGCATATCCTTTGCGTGCCACCGTTGACGATGAAACGGATGTCG

# FASTA file -> TSV
python predict.py -f input.fa -o predictions.tsv

# choose a regression model (default: c3)
python predict.py -f input.fa -o predictions.tsv -m c5
```

## Regression models

The classifier is the same for all options; `-m` only changes the model used for `Editing_level`.

The regression models differ only in the RNA-seq read-count threshold used to select training sites. `cN` was trained on sites with ≥ N reads.

| `-m` | Description |
|---|---|
| `c3` (default) | read count >= 3 |
| `c5` | read count >= 5 |
| `c6` | read count >= 6 |
| `c10` | read count >= 10 |

## Output columns

| Column | Meaning |
|---|---|
| `Edited_Prediction` | 1 if `Confidence_score` > 0.98, else 0. 1 means edited, 0 means non-edited |
| `Confidence_score` | Classifier's confidence level that the site is edited |
| `Editing_level` | Predicted editing level from the regression model, on the same scale as the training data |

The classifier and the regression model are trained separately. The regression model predicts editing level regardless of the classifier call. Interpret it only for sites predicted as edited.

## Input notes

- Sequences should be exactly 101 nt, centered on the site of interest (A).
- Both DNA and RNA sequences are accepted.
- Longer sequences are center-cropped to 101 nt; shorter ones are padded with N. A warning will be printed in both cases.
- Non-ACGTU characters are encoded as all-zero (treated as unknown).
