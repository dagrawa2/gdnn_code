import itertools
import numpy as np
from gappy import gap

import gdnn
from gdnn.gapfunctions import *


def count_admissible_architectures(generators):
	generator_names, generators, generator_reps = list(zip(*generators))

	G = group_from_generators(generators)
	subgroup_pairs = SubgroupPairsHK(G, [i+1 for i in range(G.Order())])

	theta_table = {}
	for (H, K) in subgroup_pairs.HKs():
		for J in subgroup_pairs.Hs():
			theta_table[(H, K, J)] = theta(G, H, K, J)

	Gamma = group_from_generators(generator_reps)
	degree = len(generator_reps[0])
	Gamma_decomposition = permgroup_decomposition(G, Gamma, degree).python()
	Gamma_stabilizers, Gamma_perm = Gamma_decomposition["stabilizers"], np.array(Gamma_decomposition["conjugator"])-1
	stabilizers = {}
	for (H, K) in subgroup_pairs.HKs():
		stabilizers[(H, K)] = gap.Intersection([theta(G, H, K, J) for J in Gamma_stabilizers])

	all_idxs = np.unique(np.array([gap.Index(G, H) for H in subgroup_pairs.Hs()]))
	idx_chains = itertools.chain.from_iterable(itertools.combinations(all_idxs[1:], r) for r in range(len(all_idxs)))
	idx_chains = [np.flip(np.array([1] + list(chain))) for chain in idx_chains if len(chain) > 0]

	D = {}
	for idx_chain in idx_chains:
		if f"depth_{len(idx_chain):d}" not in D.keys():
			D[f"depth_{len(idx_chain):d}"] = {"depth": len(idx_chain), "architectures": 0, "admissible_architectures": 0}
		all_reps = [[(H, K) for (H, K) in subgroup_pairs.HKs() if gap.Index(G, H) == i] for i in idx_chain]
		for reps in itertools.product(*all_reps):
			if gap.Index(reps[-1][0], reps[-1][1]) != 1:
				continue
			D[f"depth_{len(idx_chain):d}"]["architectures"] += 1
			admissible = True
			for (l, (H, K)) in enumerate(reps):
				Js = [reps[i][0] for i in range(l)]
				stabilizer_rep = gap.Intersection([stabilizers[(H, K)]] + [theta_table[(H, K, J)] for J in Js])
				if stabilizer_rep != K:
					admissible = False
					break
			if admissible:
				D[f"depth_{len(idx_chain):d}"]["admissible_architectures"] += 1

	return D


def build_table(groups, headers, filename="admissible_architectures.txt"):
	Ds = []
	for group in groups:
		generators = gdnn.groups.group(group)
		generators = [(name, g, g) for (name, g) in generators]
		Ds.append( count_admissible_architectures(generators) )

	max_depth = max([D_depth["depth"] for D in Ds for (key, D_depth) in D.items()])
	n_cols = len(groups) + 1

	with open(filename, "w") as f:
		f.write("\\begin{tabular}{"+"c"*n_cols+"}\n")
		f.write("\\toprule\n")
		f.write("Depth & " + " & ".join(headers) + " \\\\\n")
		f.write("\\midrule\n")
		for depth in range(2, max_depth+1):
			f.write(f"{depth:d}")
			for D in Ds:
				values = D[f"depth_{depth:d}"]
				architectures, admissible_architectures = values["architectures"], values["admissible_architectures"]
				f.write(f" & {admissible_architectures:d}/{architectures:d}")
			f.write(" \\\\\n")
		f.write("\\bottomrule\n")
		f.write("\\end{tabular}")


if __name__ == "__main__":
	groups = ["C_8", "C_2xC_4", "C_2xC_2xC_2", "D_4", "Q_8"]
	headers = ["$C_8$", "$C_2\\times C_4$", "$C_2^3$", "$D_4$", "$Q_8$"]

	build_table(groups, headers, filename="admissible_architectures.txt")

	print("Done!")
