"""The icosahedral group."""

import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

def rotmat(v, theta):
	"""Construct a 3x3 rotation matrix given an axis and angle.

	Args:
		v (np.ndarray): 3D unit vector about which to rotate.
		theta (float): Angle (in radians) to rotate about v (right-handed).

	Returns:
		3x3 np.ndarray rotation matrix.
	"""
	# build quaternian
	c = np.cos(theta/2)
	s = np.sin(theta/2)
	xs, ys, zs = tuple(s*v)
	# get rotation matrix
	R = Rotation.from_quat([xs, ys, zs, c]).as_matrix()
	return np.asarray(R)


def group_generators(mesh_dir, level=0, generator_names=["R", "S"]):
	"""Generators for a rep of the icosahedral group.

	The group is the set of isometric permutations on the vertices of an icosahedron. 
	One generator is a rotation about a vertex, 
	and another is a rotation about the normal vector to a face.

	Args:
		mesh_dir (str): Directory containing the precomputed spherical meshes.
		level (int): Resolution level of the mesh.
		generator_names (list): List of names (str) of generators.

	Returns:
		List of group generators given as (name, permutation, permutation_rep) where
			* name (str) is the generator name, 
			* permutation (list) is a permutation implementing the generator, and
			* permutation_rep (list) is a permutation implementing the action of the generator on the spherical mesh of the specified resolution level.
	"""
	# load a vertex and a normal face vector
	with open(os.path.join(mesh_dir, "icosphere_0.pkl"), "rb") as f:
		mesh_dict = pickle.load(f)
	v = mesh_dict["V"][0]
	n = mesh_dict["N"][6]

	# load vertices of the spherical mesh
	with open(os.path.join(mesh_dir, f"icosphere_{level:d}.pkl"), "rb") as f:
		mesh_dict = pickle.load(f)
	V = mesh_dict["V"]

	# rotations generating the isometries
	R = rotmat(v, 2*np.pi/5)
	S = rotmat(n, 2*np.pi/3)

	generators_rot = [R, S]
	generator_reps = []
	for gen in generators_rot:
		# rotate the mesh vertices
		perm_dots = V.dot(gen.T).dot(V.T)
		# match between original and rotated vertices to obtain permutation
		generator_reps.append(list(perm_dots.argmax(1)+1))

	# intrinsic generators of icosahedral group (isomorphic to A_5)
	generators = [
		[2, 3, 4, 5, 1], 
		[2, 3, 1, 4, 5]
	]

	return list( zip(generator_names, generators, generator_reps) )
