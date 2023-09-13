"""Generate binary multiplication dataset and associated symmetry group."""

import os
import itertools
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split

import gdnn

# fix the random seed
random.seed(123)
np.random.seed(456)

def generate(train_size, output_dir):
	"""Generate binary multiplication dataset.

	This is a binary classification problem with input set {-1, 1}^16, 
	and where each input is labeled according to its parity.

	Args:
		train_size (int): Number of training points. 
			The remaining points are taken as the test set.
		output_dir (str): .npz file to which to write the dataset.
	"""
	# input set {-1, 1}^16
	x = itertools.product([1, -1], repeat=16)
	x = np.array([list(v) for v in x], dtype=int)
	# parity of products
	y = np.prod(x, 1)

	# onehot-encode +-1 and flatten
	x = np.stack([x, -x], 2).reshape((-1, 1, 32))
	x = ((x+1)/2).astype(np.float32)

	# set class labels to 0 vs 1
	y = np.expand_dims((y+1)/2, 1).astype(np.float32)

	# train-test split
	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, stratify=y)

	# symmetry group generators
	generators = []
	for i in range(15):
		v = [j+1 for j in range(32)]
		# flip last bit
		v[30] += 1
		v[31] -= 1
		# flip ith bit
		v[		2*i] += 1
		v[2*i+1] -= 1
		generators.append( (f"g_{i+1:d}", v) )

	# express generators in format that GDNN expects
	generators = [(name, gen, gen) for (name, gen) in generators]

	# write to disk
	os.makedirs(output_dir, exist_ok=True)
	np.savez(os.path.join(output_dir, "dataset.npz"), x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
	with open(os.path.join(output_dir, "generators.pkl"), "wb") as f:
		pickle.dump(generators, f)

	return 


if __name__ == "__main__":
	train_size = 0.2
	output_dir = "data"

	generate(train_size, output_dir)

	print("Done!")
