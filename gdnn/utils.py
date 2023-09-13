"""Miscellaneous utility functions."""

import numpy as np

def subsample(X, Y, percentage):
	"""Subsample a classification dataset without replacement and with class stratification.

		Args:
			X (np.ndarray): Data inputs with first dimension indexing the data.
Y (np.ndarray): 1D array of class labels of the data points. 
				If there are C classes, then the labels must be ints in [0 . . . C-1].
			percentage (int): Reduce input dataset down to this percentage.

		Returns:
			Tuple of subsampled X and Y.
		"""
	# get unique class labels
	classes = np.unique(Y)

	# subsample with stratification
	idxs_all = np.arange(len(Y))
	idxs = []
	for c in classes:
		idxs_class = idxs_all[Y==c]
		idxs.append( idxs_class[:round(percentage/100*len(idxs_class))] )

	idxs = np.concatenate(idxs, 0)
	return X[idxs], Y[idxs]
