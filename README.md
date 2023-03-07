# Densely Connected G-Invariant Deep Neural Networks with Signed Permutation Representations
Devanshu Agrawal & James Ostrowski

### Abstract

We introduce and investigate, for finite groups G, G-invariant deep neural network (G-DNN) architectures with ReLU activation that are densely connected-- 
i.e., include all possible skip connections. 
In contrast to other G-invariant architectures in the literature, the preactivations of theG-DNNs presented here are able to transform by *signed* permutation representations (signed perm-reps) of G. 
Moreover, the individual layers of the G-DNNs are not required to be G-equivariant; 
instead, the preactivations are constrained to be G-equivariant functions of the network input in a way that couples weights across all layers. 
The result is a richer family of G-invariant architectures never seen previously. 
We derive an efficient implementation of G-DNNs after a reparameterization of weights, 
as well as necessary and sufficient conditions for an architecture to be "admissible"-- 
i.e., nondegenerate and inequivalent to smaller architectures. 
We include code that allows a user to build a G-DNN interactively layer-by-layer, 
with the final architecture guaranteed to be admissible. 
Finally, we apply G-DNNs to two example problems -- 
(1) multiplication in \{-1, 1\} (with theoretical guarantees) and (2) 3D object classification -- 
finding that the inclusion of signed perm-reps significantly boosts predictive performance compared to baselines with only ordinary (i.e., unsigned) perm-reps.


### Description

This repository includes code and scripts to reproduce the results presented in the associated paper.


### Requirements

These are the minimal requirements to use the GDNN module. 
For additional requirements needed to run the examples, see their respective README files.

- python >= 3.8 (along with standard Anaconda packages including numpy and scipy)
- pytorch >= 1.13
- [Gappy](https://github.com/embray/gappy)
- [igl](https://libigl.github.io/libigl-python-bindings)
- tabulate


### Installation

To download and install the GDNN module, run the following:

    git clone https://github.com/dagrawa2/gdnn.git
    cd gdnn
    python setup.py


### Reproducing results

To reproduce Table 1 in the paper, run the following:

    python admissible_architectures.py

The LaTeX source for Table 1 will be written to `admissible_architectures.txt`.

To reproduce the results in Section 4 of the paper, see the respective README files of the [binary multiplication](examples/binary_mul/README.md) and [3D object classification](examples/modelnet40/README.md) examples.
