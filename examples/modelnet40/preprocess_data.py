"""Preprocessing script for ModelNet40 adapted from https://github.com/maxjiang93/ugscnn"""

import os
import pickle
import numpy as np
import torch
import torchvision
import trimesh


def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
	"""
	Create a rotation matrix with an optional fourth homogeneous coordinate

	:param a, b, c: ZYZ-Euler angles
	"""
	def z(a):
		return np.array([[np.cos(a), np.sin(a), 0, 0],
						 [-np.sin(a), np.cos(a), 0, 0],
						 [0, 0, 1, 0],
						 [0, 0, 0, 1]])

	def y(a):
		return np.array([[np.cos(a), 0, np.sin(a), 0],
						 [0, 1, 0, 0],
						 [-np.sin(a), 0, np.cos(a), 0],
						 [0, 0, 0, 1]])

	r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
	if hom_coord:
		return r
	else:
		return r[:3, :3]


def render_model(mesh, sgrid):
	# Cast rays
	# triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
	index_tri, index_ray, loc = mesh.ray.intersects_id(
		ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
	loc = loc.reshape((-1, 3))  # fix bug if loc is empty

	# Each ray is in 1-to-1 correspondence with a grid point. Find the position of these points
	grid_hits = sgrid[index_ray]
	grid_hits_normalized = grid_hits / np.linalg.norm(grid_hits, axis=1, keepdims=True)

	# Compute the distance from the grid points to the intersection points
	dist = np.linalg.norm(grid_hits - loc, axis=-1)

	# For each intersection, look up the normal of the triangle that was hit
	normals = mesh.face_normals[index_tri]
	normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

	# Construct spherical images
	dist_im = np.ones(sgrid.shape[0])
	dist_im[index_ray] = dist
	# dist_im = dist_im.reshape(theta.shape)

	# shaded_im = np.zeros(sgrid.shape[0])
	# shaded_im[index_ray] = normals.dot(light_dir)
	# shaded_im = shaded_im.reshape(theta.shape) + 0.4

	n_dot_ray_im = np.zeros(sgrid.shape[0])
	# n_dot_ray_im[index_ray] = np.abs(np.einsum("ij,ij->i", normals, grid_hits_normalized))
	n_dot_ray_im[index_ray] = np.einsum("ij,ij->i", normalized_normals, grid_hits_normalized)

	nx, ny, nz = normalized_normals[:, 0], normalized_normals[:, 1], normalized_normals[:, 2]
	gx, gy, gz = grid_hits_normalized[:, 0], grid_hits_normalized[:, 1], grid_hits_normalized[:, 2]
	wedge_norm = np.sqrt((nx * gy - ny * gx) ** 2 + (nx * gz - nz * gx) ** 2 + (ny * gz - nz * gy) ** 2)
	n_wedge_ray_im = np.zeros(sgrid.shape[0])
	n_wedge_ray_im[index_ray] = wedge_norm

	# Combine channels to construct final image
	# im = dist_im.reshape((1,) + dist_im.shape)
	im = np.stack((dist_im, n_dot_ray_im, n_wedge_ray_im), axis=0)

	return im


def rnd_rot():
	a = np.random.rand() * 2 * np.pi
	z = np.random.rand() * 2 - 1
	c = np.random.rand() * 2 * np.pi
	rot = rotmat(a, np.arccos(z), c, True)
	return rot



class ToMesh:

	def __init__(self, random_rotations=False, random_translation=0):
		self.rot = random_rotations
		self.tr = random_translation


	def __call__(self, path):
		mesh = trimesh.load_mesh(path)
		mesh.remove_degenerate_faces()
		mesh.fix_normals()
		mesh.fill_holes()
		mesh.remove_duplicate_faces()
		mesh.remove_infinite_values()
		mesh.remove_unreferenced_vertices()

		mesh.apply_translation(-mesh.centroid)

		r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
		mesh.apply_scale(1 / r)

		if self.tr > 0:
			tr = np.random.rand() * self.tr
			rot = rnd_rot()
			mesh.apply_transform(rot)
			mesh.apply_translation([tr, 0, 0])

			if not self.rot:
				mesh.apply_transform(rot.T)

		if self.rot:
			mesh.apply_transform(rnd_rot())

		r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
		mesh.apply_scale(0.99 / r)

		return mesh


	def __repr__(self):
		return self.__class__.__name__ + '(rotation={0}, translation={1})'.format(self.rot, self.tr)



class ProjectOnSphere:

	def __init__(self, meshfile, dataset, normalize=True):
		self.meshfile = meshfile
		pkl = pickle.load(open(meshfile, "rb"))
		self.sgrid = pkl["V"]
		self.level = int(meshfile.split('_')[-1].split('.')[0])
		self.pts = self.sgrid.shape[0]
		self.normalize = normalize
		assert(dataset in ["modelnet10", "modelnet40"])
		self.dataset = dataset

	def __call__(self, mesh):
		im = render_model(mesh, self.sgrid)  # shape 3_channels x #v

		from scipy.spatial.qhull import QhullError  # pylint: disable=E0611
		try:
			convex_hull = mesh.convex_hull
		except QhullError:
			convex_hull = mesh

		hull_im = render_model(convex_hull, self.sgrid)

		im = np.concatenate([im, hull_im], axis=0)
		assert len(im) == 6

		# take absolute value of normal
		im[1] = np.absolute(im[1])
		im[4] = np.absolute(im[4])

		if self.normalize and self.dataset == 'modelnet10':
			im[0] -= 0.7203571
			im[0] /= 0.2807092
			im[1] -= 0.6721025
			im[1] /= 0.2561926
			im[2] -= 0.6199647
			im[2] /= 0.26200315
			im[3] -= 0.49367973
			im[3] /= 0.19068004
			im[4] -= 0.7766791
			im[4] /= 0.17894566
			im[5] -= 0.55923575
			im[5] /= 0.22804247
		elif self.normalize and self.dataset == 'modelnet40':
			im[0] -= 0.7139052
			im[0] /= 0.27971452
			im[1] -= 0.6935615
			im[1] /= 0.2606435
			im[2] -= 0.5850884
			im[2] /= 0.27366385
			im[3] -= 0.53355956
			im[3] /= 0.21440032
			im[4] -= 0.76255935
			im[4] /= 0.19869797
			im[5] -= 0.5651189
			im[5] /= 0.24401328
		im = im.astype(np.float32)  # pylint: disable=E1101

		return im


	def __repr__(self):
		return self.__class__.__name__ + '(level={0}, points={1})'.format(self.level, self.pts)



class CacheNPY(object):

	def __init__(self, prefix, transform, sp_mesh_dir, sp_mesh_level=5):
		self.transform = transform
		self.prefix = prefix


	def check_trans(self, file_path):
		try:
			return self.transform(file_path)
		except:
			raise


	def __call__(self, file_path):
		head, tail = os.path.split(file_path)
		root, _ = os.path.splitext(tail)
		npy_path = os.path.join(head, self.prefix + root + '.npy')

		if not os.path.exists(npy_path):
			img = self.check_trans(file_path)
			np.save(npy_path, img)



### key functions used in main body

def preprocess_data(sp_mesh_dir, sp_mesh_level, data_dir, partition):
	"""Preprocess each ModelNet40 file and save in .npy format.

	Args:
		sp_mesh_dir (str): Directory containing spherical meshes generated from "generate_mesh.py".
		sp_mesh_level (int): Resolution level of spherical mesh.
		data_dir (str): Root directory of raw ModelNet40 data files.
		partition (str): Either "train" or "test" split.
	"""
	sp_mesh_file = os.path.join(sp_mesh_dir, f"icosphere_{sp_mesh_level}.pkl")  # spherical mesh

	# transform that will do the preprocessing per file
	transform = CacheNPY(prefix=f"sp{sp_mesh_level}_", transform=torchvision.transforms.Compose(
		[
			ToMesh(random_rotations=False, random_translation=0),
			ProjectOnSphere(meshfile=sp_mesh_file, dataset="modelnet40", normalize=True)
		]
	), sp_mesh_dir=sp_mesh_dir, sp_mesh_level=sp_mesh_level)

	dir = os.path.join(data_dir, f"modelnet40_{partition}")  # dir of data files
	files = os.listdir(dir)
	files = [f for f in files if ".off" in f]
	for f in files:
		try:
			transform( os.path.join(dir, f) )
		except:
			# print failed files to stdout (sometimes due to OOM)
			print(f)


def aggregate_dataset(data_dir, partition):
	"""Aggregate preprocessed .npy files into numpy array.

	Args:
		data_dir (str): Root directory of ModelNet40 data files.
		partition (str): Either "train" or "test" split.
	"""
	# 40 class labels
	classes = ['airplane', 'bowl', 'desk', 'keyboard', 'person', 'sofa', 'tv_stand', 'bathtub', 'car', 'door', 
		'lamp', 'piano', 'stairs', 'vase', 'bed', 'chair', 'dresser', 'laptop', 'plant', 'stool', 
		'wardrobe', 'bench', 'cone', 'flower_pot', 'mantel', 'radio', 'table', 'xbox', 'bookshelf', 'cup', 
		'glass_box', 'monitor', 'range_hood', 'tent', 'bottle', 'curtain', 'guitar', 'night_stand', 'sink', 'toilet']

	dir = os.path.join(data_dir, f"modelnet40_{partition}")  # dir of data files
	files = [f for f in os.listdir(dir) if "sp2_" in f]

	x, y = [], []
	for f in files:
		label = "_".join(f.split("_")[1:-1])  # extract from filename
		y.append( classes.index(label) )
		x.append( np.load(os.path.join(dir, f)) )

	# build arrays
	x = np.stack(x, 0)
	y = np.array(y)
	return x, y


if __name__ == "__main__":
	import gdnn

	level = 2  # resolution
	partitions = ["train", "test"]

	print("Preprocessing data . . . ")
	data_dir = "data/raw"
	for partition in partitions:
		print(f"Partition: {partition}")
		preprocess_data("mesh_files", level, data_dir, partition)

	print("Gathering data into arrays . . . ")
	dataset = {}
	for partition in partitions:
		dataset[f"x_{partition}"], dataset[f"y_{partition}"] = aggregate_dataset(data_dir, partition)

	# write processed data to disk
	data_dir = "data"
	np.savez(os.path.join(data_dir, "dataset.npz"), **dataset)

	# construct and save icosahedral symmetry group
	print("Saving icosahedral group generators . . . ")
	generators = gdnn.icosahedron.generators("mesh_files", level=level)
	with open(os.path.join(data_dir, "generators.pkl"), "wb") as f:
		pickle.dump(generators, f)

	print("Done!")
