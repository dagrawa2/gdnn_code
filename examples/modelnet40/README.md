# 3D object classification
Devanshu Agrawal & James Ostrowski


### Description

This subrepository includes code and scripts to reproduce the results in Section 4.3 of the associated paper.


### Requirements

These requirements are in addition to the ones listed in the [main README](../../README.md) for the GDNN module.

- torchvision
- trimesh
- rtree
- pandas and matplotlib


### Reproducing results

First, we download and preprocess the ModelNet40 dataset:

    bash download_data.sh
    python preprocess_data.py

Next, we generate the icosahedral meshgrids needed to construct the icosahedral symmetry group:

    python generate_mesh.py

Based on these meshgrids, we generate the weightsharing patterns to be imported into our GDNN models:

    bash generate_patterns.sh

We then train all models:

    bash train.sh

Note the included script `train.sh` runs the 24 initialization seeds sequentially.

Finally, to reproduce Figure 2 in the paper, run the following:

    python plot.py

The plot will be written to `plot.png`.
