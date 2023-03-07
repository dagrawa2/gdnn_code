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

	def __init__(self, *args, **kwargs):
		super(BatchNormFlattened, self).__init__(*args, **kwargs)

	def forward(self, X):
		return super(BatchNormFlattened, self).forward(X \
			.reshape(X.shape[0], -1, self.num_features) \
			.permute(0, 2, 1) \
			).permute(0, 2, 1) \
			.reshape(X.shape[0], -1)

	def slope(self):
		return self.weight.detach()/torch.sqrt(self.running_var.detach() + self.eps)



class EquivariantAll2IrrepFlattened(nn.Module):

	def __init__(self, in_channels_list, out_channels, weight_patterns, abs_patterns=False):
		super(EquivariantAll2IrrepFlattened, self).__init__()
		self.in_channels_list = in_channels_list
		self.out_channels = out_channels
		self.weight_patterns = [torch.as_tensor(pattern) for pattern in weight_patterns]
		self.abs_patterns = abs_patterns

		self.degree = self.weight_patterns[0].shape[0]
		self.signed = self.weight_patterns[0].min() < 0
		if self.abs_patterns:
			self.weight_patterns = [torch.abs(pattern) for pattern in self.weight_patterns]

		self.weights = nn.ParameterList([nn.Parameter(torch.zeros(self.out_channels, in_channels, pattern.max())) for (in_channels, pattern) in zip(self.in_channels_list, self.weight_patterns)])
		for weights in self.weights:
			nn.init.kaiming_uniform_(weights)

		if not self.signed:
			self.biases = nn.Parameter(torch.zeros(self.out_channels))
		else:
			self.register_parameter("biases", None)

	def forward(self, X):
		weight_matrix = self.weight_matrix()

		out = X.mm(weight_matrix.T)
		if self.biases is not None:
			bias_vector = self.bias_vector()
			out = out + bias_vector.unsqueeze(0)
		return out

	def weight_matrix(self):
		return torch.cat([
			(torch.sign(pattern)[None,None,...]*weights[...,torch.abs(pattern)-1]) \
			.permute(2, 0, 3, 1) \
			.reshape(self.out_channels*self.degree, -1) \
			if weights.numel() > 0 else \
			torch.zeros(self.out_channels*self.degree, weights.shape[1]*pattern.shape[1], dtype=torch.float32) \
			for (weights, pattern) in zip(self.weights, self.weight_patterns)], 1)

	def bias_vector(self):
		if self.biases is not None:
			return self.biases.repeat(self.degree)
		return torch.zeros(self.out_channels*self.degree)



class EquivariantAll2RepFlattened(nn.Module):

	def __init__(self, in_channels_list, out_channels, weight_patterns, **kwargs):
		super(EquivariantAll2RepFlattened, self).__init__()
		self.all2irrep_modules = nn.ModuleList([EquivariantAll2IrrepFlattened(in_channels_list, out_channels, patterns, **kwargs) for patterns in weight_patterns])

	def forward(self, X):
		return torch.cat([module(X) for module in self.all2irrep_modules], 1)

	def weight_matrix(self):
		return torch.cat([module.weight_matrix() for module in self.all2irrep_modules], 0)

	def bias_vector(self):
		return torch.cat([module.bias_vector() for module in self.all2irrep_modules], 0)



class GDNN(nn.Module):

	def __init__(self, generators, channels, batch_norm=False, abs_patterns=False, tunnel_reps=False, unravel_reps=False, load_reps=None, load_patterns=None, save_patterns=None):
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

		EquivariantLayer = EquivariantAll2RepFlattened
		BatchNormLayer = BatchNormFlattened

		if self.load_patterns is not None:
			with open(self.load_patterns, "rb") as f:
				patterns = pickle.load(f)
		else:
			patterns = self.build_patterns()

		if self.save_patterns is not None:
			with open(self.save_patterns, "wb") as f:
				pickle.dump(patterns, f)

		self.layers = nn.ModuleList([])
		for (l, (out_channels_l, patterns_l)) in enumerate(zip(self.out_channels, patterns)):
			in_channels_list = list(reversed(self.in_channels[:l+1]))
			self.layers.append( EquivariantLayer(in_channels_list, out_channels_l, patterns_l, abs_patterns=self.abs_patterns) )

		if self.batch_norm:
			self.bn_layers = nn.ModuleList([BatchNormLayer(out_channels_l, momentum=None) for out_channels_l in self.out_channels[:-1]])
		else:
			self.bn_layers = nn.ModuleList([nn.Identity() for out_channels_l in self.out_channels[:-1]])


	def forward(self, X):
		out1 = X.permute(0, 2, 1).reshape(X.shape[0], -1)
		out2 = self.layers[0](out1)
		bias = self.layers[0].bias_vector()

		for (bn, layer) in zip(self.bn_layers, self.layers[1:]):
			out1 = torch.cat([bn(F.relu(out2)-(out2-bias)/2), out1], 1)
			out2 = layer(out1)
			bias = layer.bias_vector()

		return out2


	def build_patterns(self):
		G = group_from_generators(self.generators)

		Gamma = group_from_generators(self.generator_reps)
		degree = len(self.generator_reps[0])
		Gamma_decomposition = permgroup_decomposition(G, Gamma, degree).python()
		Gamma_stabilizers, Gamma_perm = Gamma_decomposition["stabilizers"], np.array(Gamma_decomposition["conjugator"])-1

		if self.load_reps is not None:
			with open(self.load_reps, "rb") as f:
				reps = pickle.load(f)
			reps = [[(group_from_generators(H), group_from_generators(K)) for (H, K) in reps_l] for reps_l in reps]
		else:
			reps = self.build_reps(G, Gamma_stabilizers)

		if self.tunnel_reps:
			reps = [[(H, H) for (H, K) in reps_l] for reps_l in reps]

		if self.unravel_reps:
			reps = [[(K, K) for (H, K) in reps_l] for reps_l in reps]

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
		subgroup_pairs = SubgroupPairsHK(G, self.subgroup_idxs)
		theta_table = {}
		for (H, K) in subgroup_pairs.HKs():
			for J in subgroup_pairs.Hs():
				theta_table[(H, K, J)] = theta(G, H, K, J)

		stabilizers = {}
		for (H, K) in subgroup_pairs.HKs():
			stabilizers[(H, K)] = gap.Intersection([theta(G, H, K, J) for J in Gamma_stabilizers])

		reps = []
		for (layer, idx) in enumerate(self.subgroup_idxs, start=1):
			print(f"===== Layer {layer:d} (subgroup index {idx:d}) =====".center(os.get_terminal_size().columns))
			print()
			options = [(H, K) for (H, K) in subgroup_pairs.HKs() \
				if gap.Index(G, H) == idx \
				and K == stabilizers[(H, K)] ]
			if len(options) == 0:
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
