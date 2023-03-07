import os
import itertools
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split

import gdnn

random.seed(123)
np.random.seed(456)

def generate(train_size, output_dir):
	x = itertools.product([1, -1], repeat=16)
	x = np.array([list(v) for v in x], dtype=int)
	y = np.prod(x, 1)

	x = np.stack([x, -x], 2).reshape((-1, 1, 32))
	x = ((x+1)/2).astype(np.float32)

	y = np.expand_dims((y+1)/2, 1).astype(np.float32)

	x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, stratify=y)

	generators = []
	for i in range(15):
		v = [i+1 for i in range(32)]
		v[30] += 1
		v[31] -= 1
		v[		2*i] += 1
		v[2*i+1] -= 1
		generators.append( (f"g_{i+1:d}", v) )

	generators = [(name, gen, gen) for (name, gen) in generators]

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
