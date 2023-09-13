"""The GDNN model."""

import os
import sys
import pickle
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from gappy import gap
from tabulate import tabulate

from .gapfunctions import *


class BatchNormFlattened(nn.BatchNorm1d):
	"""Batch normalization for flattened inputs.

	If an input minibatch has shape (batch_size, channels*features), 
	then it is reshaped to (batch_size, channels, features), batchnorm is applied, and finally the minibatch is reflattened.
	"""

	def __init__(self, *args, **kwargs):
		"""Class constructor."""
		super(BatchNormFlattened, self).__init__(*args, **kwargs)


	def forward(self, X):
		"""Forward pass.

		The input is reshaped to expose its channel dimensions, along which batchnorm is broadcasted, 
		after which the output is reflattened.

		Args:
			X (torch.Tensor): Input minibatch with flattened channel and feature dimensions.

		Returns:
			Normalized minibatch of the same shape as the input.
		"""
		return super(BatchNormFlattened, self).forward(X \
			.reshape(X.shape[0], -1, self.num_features) \
			.permute(0, 2, 1) \
			).permute(0, 2, 1) \
			.reshape(X.shape[0], -1)


	def slope(self):
		"""Slope coefficient of the batchnorm affine transformation."""
		return self.weight.detach()/torch.sqrt(self.running_var.detach() + self.eps)



class EquivariantAll2IrrepFlattened(nn.Module):
	"""Equivariant linear layer whose output transforms by a signed perm-irrep.

	This layer takes as input the concatenation of the activations of all previous layers together with the network input; 
	i.e., the layer includes skip connections to all previous layers.
	"""

	def __init__(self, in_channels_list, out_channels, weight_patterns, abs_patterns=False):
		"""Class constructor.

		Args:
			in_channels_list (list): List of input channels from all previous layers.
			out_channels (int): Output channels of this layer.
			weight_patterns (list): List of weightsharing patterns (np.ndarray objects) for all previous layers.
			abs_patterns (bool): Ignore this option.
		"""
		super(EquivariantAll2IrrepFlattened, self).__init__()
		self.in_channels_list = in_channels_list
		self.out_channels = out_channels
		self.weight_patterns = [torch.as_tensor(pattern) for pattern in weight_patterns]
		self.abs_patterns = abs_patterns

		self.degree = self.weight_patterns[0].shape[0]  # rep degree
		self.signed = self.weight_patterns[0].min() < 0  # whether output rep is type 2
		if self.abs_patterns:
			self.weight_patterns = [torch.abs(pattern) for pattern in self.weight_patterns]

		# init free (unconstrained) weights
		self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.out_channels, in_channels, pattern.max())) for (in_channels, pattern) in zip(self.in_channels_list, self.weight_patterns)])
		for weights in self.weights:
			nn.init.kaiming_uniform_(weights)

		if not self.signed:
			self.biases = nn.Parameter(torch.zeros(self.out_channels))
		else:
			# type 2 reps require zero bias
			self.register_parameter("biases", None)


	def forward(self, X):
		"""Forward pass.

		Args:
			X (torch.Tensor): Of size (batch_size, agg_ch_feats) where
				agg_ch_feats is the sum of channels*features for all previous layers.

		Returns:
			Output of size (batch_size, self.out_channels*self.degree).
		"""
		# construct equivariant weight matrix
		weight_matrix = self.weight_matrix()

		# affine transformation
		out = X.mm(weight_matrix.T)
		if self.biases is not None:
			bias_vector = self.bias_vector()
			out = out + bias_vector.unsqueeze(0)
		return out


	def weight_matrix(self):
		"""Build equivariant weight matrix.

		The weight matrix should be visualized as a block matrix with each block having input and output channels. 
		Certain permutations of the block-columns are equivalent to permutations of block-rows.

		Returns:
			Weight matrix.
		"""
		return torch.cat([
			(torch.sign(pattern)[None,None,...]*weights[...,torch.abs(pattern)-1]) \
			.permute(2, 0, 3, 1) \
			.reshape(self.out_channels*self.degree, -1) \
			if weights.numel() > 0 else \
			torch.zeros(self.out_channels*self.degree, weights.shape[1]*pattern.shape[1], dtype=torch.float32) \
			for (weights, pattern) in zip(self.weights, self.weight_patterns)], 1)


	def bias_vector(self):
		"""Build bias vector.

		Repeats so that bias is constant across features.

		Returns:
			Bias vector.
		"""
		if self.biases is not None:
			return self.biases.repeat(self.degree)
		# bias is zero for type 2 reps
		return torch.zeros(self.out_channels*self.degree)



class EquivariantAll2RepFlattened(nn.Module):
	"""Equivariant linear layer whose output transforms by a signed perm-rep.

	This layer is just a concatenation of EquivariantAll2IrrepFlattened layers along the output dimension.
	"""

	def __init__(self, in_channels_list, out_channels, weight_patterns, **kwargs):
		"""Class constructor.

		Args:
			in_channels_list (list): List of input channels from all previous layers.
			out_channels (int): Output channels of this layer.
			weight_patterns (list): Nested list of weightsharing patterns, 
				where each sublist is as expected by EquivariantAll2IrrepFlattened.
		"""
		super(EquivariantAll2RepFlattened, self).__init__()
		self.all2irrep_modules = nn.ModuleList([EquivariantAll2IrrepFlattened(in_channels_list, out_channels, patterns, **kwargs) for patterns in weight_patterns])


	def forward(self, X):
		"""Forward pass.

		Args:
			X (torch.Tensor): Of size (batch_size, agg_ch_feats) where
				agg_ch_feats is the sum of channels*features for all previous layers.

		Returns:
			Concatenation of outputs of EquivariantAll2IrrepFlattened layers.
		"""
		return torch.cat([module(X) for module in self.all2irrep_modules], 1)


	def weight_matrix(self):
		"""Build equivariant weight matrix.

		Returns:
			Concatenation of weight matrices of the EquivariantAll2IrrepFlattened layers.
		"""
		return torch.cat([module.weight_matrix() for module in self.all2irrep_modules], 0)


	def bias_vector(self):
		"""Build bias vector.

		Returns:
			Concatenation of bias vectors of the EquivariantAll2IrrepFlattened layers.
		"""
		return torch.cat([module.bias_vector() for module in self.all2irrep_modules], 0)



class GDNN(nn.Module):
	"""The GDNN model."""

	def __init__(self, generators, channels, batch_norm=False, abs_patterns=False, tunnel_reps=False, unravel_reps=False, load_reps=None, load_patterns=None, save_patterns=None):
		"""Class constructor.

		Args:
			generators (list): List of tuples (generator_name, generator, generator_rep), where
				* generator_name (str) is the generator name, 
				* generator (list) is a permutation such that the generators togehter generate the group, and
				* generator_rep (list) is another permutation such that all generator_reps generate the group rep by which the group acts.
			channels (list): List of tuples (in_ch, out_ch, idx) for each layer where
				* in_ch (int) is the number of input channels, 
				* out_ch (int) is the number of output channels, and
				* idx (int) is the subgroup index for this layer.
			batch_norm (bool): Whether to use batch normalization.
			abs_patterns (bool): Ignore this option.
			tunnel_reps (bool): Whether to tunnel from type 2 to type 1 (see associated paper).
			unravel_reps (bool): Whether to unravel from type 2 to type 1 (see associated paper).
			load_reps (str): Load precomputed reps from this location.
			load_patterns (str): Load weightsharing patterns from this location.
			save_patterns (str): Save weightsharing patterns to this location.

		super(GDNN, self).__init__()
		self.generator_names, self.generators, self.generator_reps = list(zip(*generators))
		self.in_channels, self.out_channels, self.subgroup_idxs = list(zip(*channels))
		self.batch_norm = batch_norm
		self.abs_patterns = abs_patterns
		self.tunnel_reps = tunnel_reps
		self.unravel_reps = unravel_reps
		self.load_reps = load_reps
		self.load_patterns = load_patterns
		self.save_patterns = save_patterns

		if self.tunnel_reps and self.unravel_reps:
			raise ValueError("tunnel_reps and unravel_reps cannot both be set to True.")

		# more convenient names
		EquivariantLayer = EquivariantAll2RepFlattened
		BatchNormLayer = BatchNormFlattened

		if self.load_patterns is not None:
			# load precomputed weightsharing patterns
			with open(self.load_patterns, "rb") as f:
				patterns = pickle.load(f)
		else:
			# build weightsharing patterns
			patterns = self.build_patterns()

		if self.save_patterns is not None:
			# save weightsharing patterns
			with open(self.save_patterns, "wb") as f:
				pickle.dump(patterns, f)

		# build layers
		self.layers = nn.ModuleList([])
		for (l, (out_channels_l, patterns_l)) in enumerate(zip(self.out_channels, patterns)):
			in_channels_list = list(reversed(self.in_channels[:l+1]))
			self.layers.append( EquivariantLayer(in_channels_list, out_channels_l, patterns_l, abs_patterns=self.abs_patterns) )

		# build batchnorm layers
		if self.batch_norm:
			self.bn_layers = nn.ModuleList([BatchNormLayer(out_channels_l, momentum=None) for out_channels_l in self.out_channels[:-1]])
		else:
			self.bn_layers = nn.ModuleList([nn.Identity() for out_channels_l in self.out_channels[:-1]])


	def forward(self, X):
		"""Forward pass. See associated paper for details.

		Args:
			X (torch.Tensor): Input batch with shape (batch_size, input_channels, input_features).

		Returns:
			Output of shape (batch_size, output_channels).
		"""
		out1 = X.permute(0, 2, 1).reshape(X.shape[0], -1)  # flatten
		out2 = self.layers[0](out1)
		bias = self.layers[0].bias_vector()

		for (bn, layer) in zip(self.bn_layers, self.layers[1:]):
			out1 = torch.cat([bn(F.relu(out2)-(out2-bias)/2), out1], 1)
			out2 = layer(out1)
			bias = layer.bias_vector()

		return out2


	def build_patterns(self):
		"""Build weightsharing patterns."""
		G = group_from_generators(self.generators)  # GAP group

		Gamma = group_from_generators(self.generator_reps)  # GAP group rep
		degree = len(self.generator_reps[0])
		# decompose group action into perm-irreps
		Gamma_decomposition = permgroup_decomposition(G, Gamma, degree).python()
		Gamma_stabilizers, Gamma_perm = Gamma_decomposition["stabilizers"], np.array(Gamma_decomposition["conjugator"])-1

		if self.load_reps is not None:
			# load precomputed reps
			with open(self.load_reps, "rb") as f:
				reps = pickle.load(f)
			reps = [[(group_from_generators(H), group_from_generators(K)) for (H, K) in reps_l] for reps_l in reps]
		else:
			# build reps
			reps = self.build_reps(G, Gamma_stabilizers)

		if self.tunnel_reps:
			# tunnel from type 1 to type 2
			reps = [[(H, H) for (H, K) in reps_l] for reps_l in reps]

		if self.unravel_reps:
			# unravel from type 1 to type 2
			reps = [[(K, K) for (H, K) in reps_l] for reps_l in reps]

		# build weightsharing patterns
		patterns = []
		Js = [Gamma_stabilizers]
		for (l, (reps_l, out_channels_l)) in enumerate(zip(reps, self.out_channels)):
			in_channels_list = list(reversed(self.in_channels[:l+1]))
			patterns_l = [[weight_pattern(G, H, K, Js_l) for Js_l in Js] for (H, K) in reps_l]
			for row in patterns_l:
				row[-1] = row[-1][...,Gamma_perm]
			patterns.append(patterns_l)
			Js.insert(0, [H for (H, K) in reps_l])

		return patterns


	def build_reps(self, G, Gamma_stabilizers):
		"""Build reps interactively layer-by-layer.

		Based on the user-provided subgroup indexes to be used at each layer, 
		i.e., number of hidden neurons per irrep in each layer, 
		a list of possible irreps is generated and listed in stdout, 
		and the user is prompted to select the desired irreps.

		Args:
			G (gap.Group): The symmetry group.
			Gamma_stabilizers (list): List of stabilizer subgroups (gap.Group objects) corresponding to the irreps comprising the input rep of G.

		Returns:
			List of reps given as tuples (H, K), 
			where H and K are as described in the associated paper.
		"""
		# build the theta table
		subgroup_pairs = SubgroupPairsHK(G, self.subgroup_idxs)
		theta_table = {}
		for (H, K) in subgroup_pairs.HKs():
			for J in subgroup_pairs.Hs():
				theta_table[(H, K, J)] = theta(G, H, K, J)

		# init stabilizers
		stabilizers = {}
		for (H, K) in subgroup_pairs.HKs():
			stabilizers[(H, K)] = gap.Intersection([theta(G, H, K, J) for J in Gamma_stabilizers])

		# build reps interactively
		reps = []
		for (layer, idx) in enumerate(self.subgroup_idxs, start=1):
			print(f"===== Layer {layer:d} (subgroup index {idx:d}) =====".center(os.get_terminal_size().columns))
			print()
			options = [(H, K) for (H, K) in subgroup_pairs.HKs() \
				if gap.Index(G, H) == idx \
				and K == stabilizers[(H, K)] ]
			if len(options) == 0:  # no available reps
				print(f"Error: No reps of index {idx:d} available for this layer.")
				sys.exit()
			options_str = [[f"{i:d}", subgroup_generators_str(G, H, self.generator_names), subgroup_generators_str(G, K, self.generator_names)] \
				for (i, (H, K)) in enumerate(options) ]
			table = tabulate(options_str, headers=[" ", "H", "K"], tablefmt="plain")
			print(table)
			print()
			rep_idxs = input("Select reps (row idxs separated by commas): ")
			rep_idxs = list( map(int, rep_idxs.replace(" ", "").split(",")) )
			selected_reps = [options[i] for i in rep_idxs]
			reps.append(selected_reps)
			Js = map(lambda x: x[0], selected_reps)
			for (H, K) in subgroup_pairs.HKs():
				stabilizers[(H, K)] = gap.Intersection([stabilizers[(H, K)]] + [theta_table[(H, K, J)] for J in Js])

		return reps
