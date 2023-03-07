import os
import json
import itertools
import numpy as np


def get_stats(architecture, seeds=24):
	dirname = f"results/{architecture}"
	D = {}
	for (split, metric, endpoint) in itertools.product(["train", "val"], ["loss", "accuracy"], ["i", "f"]):
		D[f"{split}_{metric}_{endpoint}"] = []
	for seed in range(seeds):
		with open(os.path.join(dirname, f"seed{seed:d}/results.json"), "r") as f:
			results = json.load(f)["train"]
		for (split, metric, endpoint) in itertools.product(["train", "val"], ["loss", "accuracy"], ["i", "f"]):
			i = 0 if endpoint=="i" else -1
			D[f"{split}_{metric}_{endpoint}"].append(results[f"{split}_{metric}"][i])
	for (key, value) in D.items():
		value = np.array(value)
		D[key] = (value.mean(), value.std())
	return D


def build_table(architectures, metric, filename, seeds=24):
	with open(filename, "w") as f:
		f.write("\\begin{tabular}{ccccc}\n")
		f.write("\\toprule\n")
		f.write("Architecture & Initial train & Initial val & Final train & Final val \\\\\n")
		f.write("\\midrule\n")
		for architecture in architectures:
			f.write(f"{architecture}")
			D = get_stats(architecture, seeds=seeds)
			for (endpoint, split) in itertools.product(["i", "f"], ["train", "val"]):
				mean, std = D[f"{split}_{metric}_{endpoint}"]
				f.write(f" & {mean:.3f}\\pm {std:.3f}")
			f.write(" \\\\\n")
		f.write("\\bottomrule\n")
		f.write("\\end{tabular}")


if __name__ == "__main__":
	architectures = ["type1", "type2", "unraveled_init-random", "unraveled_init-type2"]
	metrics = ["loss", "accuracy"]

	for metric in metrics:
		build_table(architectures, metric, f"table_{metric}.txt")

	print("Done!")
