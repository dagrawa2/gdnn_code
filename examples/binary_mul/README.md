# Binary multiplication
Devanshu Agrawal & James Ostrowski


### Description

This subrepository includes code and scripts to reproduce the results in Section 4.1 of the associated paper.


### Requirements

These requirements are in addition to the ones listed in the [main README](../../README.md) for the GDNN module.

- scikit-learn


### Reproducing results

First, we generate the dataset:

    python generate_data.py

Next, we generate the signed perm-irreps described in Example 1b in the paper:

    python generate_reps.py

Based on these signed perm-irreps, we generate the weightsharing patterns to be imported into our GDNN models:

    bash generate_patterns.sh

We then train all models:

    bash train.sh

Note the included script `train.sh` runs the 24 initialization seeds sequentially.

Finally, to reproduce Table 2 in the paper, run the following:

    python table.py

The LaTeX source for table 2 will be written to `table_loss.txt`.
