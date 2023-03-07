import numpy as np

def subsample(X, Y, percentage):
	classes = np.unique(Y)

	idxs_all = np.arange(len(Y))
	idxs = []
	for c in classes:
		idxs_class = idxs_all[Y==c]
		idxs.append( idxs_class[:round(percentage/100*len(idxs_class))] )

	idxs = np.concatenate(idxs, 0)
	return X[idxs], Y[idxs]
