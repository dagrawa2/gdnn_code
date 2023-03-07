import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

def rotmat(v, theta):
	c = np.cos(theta/2)
	s = np.sin(theta/2)
	xs, ys, zs = tuple(s*v)
	R = Rotation.from_quat([xs, ys, zs, c]).as_matrix()
	return np.asarray(R)


def group_generators(mesh_dir, level=0, generator_names=["R", "S"]):
	with open(os.path.join(mesh_dir, "icosphere_0.pkl"), "rb") as f:
		mesh_dict = pickle.load(f)
	v = mesh_dict["V"][0]
	n = mesh_dict["N"][6]

	with open(os.path.join(mesh_dir, f"icosphere_{level:d}.pkl"), "rb") as f:
		mesh_dict = pickle.load(f)
	V = mesh_dict["V"]

	R = rotmat(v, 2*np.pi/5)
	S = rotmat(n, 2*np.pi/3)

	generators_rot = [R, S]
	generator_reps = []
	for gen in generators_rot:
		perm_dots = V.dot(gen.T).dot(V.T)
		generator_reps.append(list(perm_dots.argmax(1)+1))

	generators = [
		[2, 3, 4, 5, 1], 
		[2, 3, 1]
	]

	return list( zip(generator_names, generators, generator_reps) )
