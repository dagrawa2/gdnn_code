import functools
import numpy as np


def direct_product(G, H):
	names_G, G = zip(*G)
	names_G, G = list(names_G), list(G)
	names_H, H = zip(*H)
	names_H, H = list(names_H), list(H)
	G = np.array(G)
	H = np.array(H)
	xtrivial = np.tile( np.arange(H.shape[1])[None,:] + G.shape[1]+1, [G.shape[0], 1])
	trivialx = np.tile( np.arange(G.shape[1])[None,:]+1, [H.shape[0], 1])
	G = np.concatenate([G, xtrivial], 1)
	H = np.concatenate([trivialx, H+trivialx.shape[1]], 1)
	GxH = np.concatenate([G, H], 0)
	generators = GxH.tolist()
	names = names_G + names_H
	return list( zip(names, generators) )

def group(group):
	if "x" not in group:
		letter, degree = tuple(group.split("_"))
		degree = int(degree)
		generators = group_dict[letter+"_n"](degree)
		return generators
	factors = group.split("x")
	factor_generators = []
	for (i, factor) in enumerate(factors, start=1):
		letter, degree = tuple(factor.split("_"))
		degree = int(degree)
		generators = group_dict[letter+"_n"](degree)
		generators = [(f"{n}_{i:d}", g) for (n, g) in generators]
		factor_generators.append(generators)
	generators = functools.reduce(direct_product, factor_generators)
	return generators


def alternating(n):
	gens = []
	for (i, j) in enumerate(range(3, n+1), start=1):
		gen = list(range(1, n+1))
		gen[0] = 2
		gen[1] = j
		gen[j-1] = 1
		gens.append((f"g_{i:d}", gen))
	return gens

def cyclic(n):
	rho = list(range(2, n+2))
	rho[-1] = 1
	return [("r", rho)]

def dihedral(n):
	rho = list(range(2, n+2))
	rho[-1] = 1
	tau = list(range(1, n+1))
	tau.reverse()
	return [("r", rho), ("t", tau)]

def quaternian(n):
	assert n==8, "The quaternian group currently supports only order 8."
	i = [3, 4, 2, 1, 7, 8, 6, 5]
	j = [5, 6, 8, 7, 2, 1, 3, 4]
	k = [7, 8, 5, 6, 4, 3, 2, 1]
	return [("i", i), ("j", j), ("k", k)]

def symmetric(n):
	rho = list(range(2, n+2))
	rho[-1] = 1
	tau = list(range(1, n+1))
	tau[0] = 2
	tau[1] = 1
	return [("r", rho), ("t", tau)]


group_dict = {
	"A_n": alternating, 
	"C_n": cyclic, 
	"D_n": dihedral, 
	"Q_n": quaternian, 
	"S_n": symmetric, 
	"Z_n": cyclic
}
