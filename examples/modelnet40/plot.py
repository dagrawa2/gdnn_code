"""Plotting script."""

import os
import json
import itertools
import numpy as np
import pandas as pd

import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=16)
matplotlib.rc("ytick", labelsize=16)

figsize = plt.rcParams['figure.figsize']


### utilities

def y_minmax(means, stds, padding=0.05):
    """Determine y-axis limits for bar plot."""
    y_min = (means-stds*np.less(means, 0).astype(int)).min()
    y_max = (means+stds*np.greater(means, 0).astype(int)).max()
    y_range = y_max-y_min
    y_min = y_min - padding*y_range
    y_max = y_max + padding*y_range
    return y_min, y_max


def bar_width_shifts(n_bars):
    """Determine bar widths and positions in bar plot."""
    total_width = 0.7
    width = total_width/n_bars
    shifts = np.array([-total_width/2 + total_width/(2*n_bars)*(2*m+1) for m in range(n_bars)])
    return width, shifts


def bar_yerrs(ys, errs):
    """Determine if error bars in bar plot should be above or below bars."""
    yerrs = []
    for (y, err) in zip(ys, errs):
        if y >= 0:
            yerrs.append([0, err])  # error above bar
        else:
            yerrs.append([err, 0])  # error below bar
    yerrs = np.array(yerrs).T
    return yerrs


def get_unique_legend_handles_labels(fig):
    """Get unique handles to labels in figure object."""
    tuples = [(h, l) for ax in fig.get_axes() for (h, l) in zip(*ax.get_legend_handles_labels())]
    handles, labels = zip(*tuples)
    unique = [(h, l) for (i, (h, l)) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    handles, labels = zip(*unique)
    return list(handles), list(labels)


### gather scores

def gather_acc(output_dir, num_seeds=None):
	"""Gather accuracy scores across seeds."""
	if num_seeds is None:
		num_seeds = len(list(filter(lambda x: "seed" in x, os.listdir(output_dir))))  # all seeds

	accuracies = []
	losses = []
	epochs = []
	times = []
	train_loss = []
	val_loss = []
	val_acc = []
	# collect metrics
	for seed in range(num_seeds):
		with open(os.path.join(output_dir, f"seed{seed:d}/results.json"), "r") as f:
			D = json.load(f)
		accuracies.append( np.array(D["train"]["val_accuracy"]).max() )
		losses.append( np.array(D["train"]["loss"]).min() )
		epochs.append( np.array(D["train"]["val_accuracy"]).argmax() )
		times.append( D["time"] )
		train_loss.append(D["train"]["loss"])
		val_loss.append(D["train"]["val_loss"])
		val_acc.append(D["train"]["val_accuracy"])

	accuracies = np.array(accuracies)*100
	losses = np.array(losses)
	epochs = np.array(epochs)
	times = np.array(times)/60

	train_loss = np.array(train_loss)
	val_loss = np.array(val_loss)
	val_acc = 100*np.array(val_acc)

	# reduce
	train_loss_mean = train_loss.mean(0)
	train_loss_std = train_loss.std(0)
	val_loss_mean = val_loss.mean(0)
	val_loss_std = val_loss.std(0)
	val_acc_mean = val_acc.mean(0)
	val_acc_std = val_acc.std(0)

	# write metrics
	os.makedirs(os.path.join(output_dir, "gathered"), exist_ok=True)
	with open(os.path.join(output_dir, "gathered/score.txt"), "w") as f:
		f.write(f"accuracy: {accuracies.mean():.1f} +- {accuracies.std():.1f}\n")
		f.write(f"loss: {losses.mean():.3f} +- {losses.std():.3f}\n")
		f.write(f"epoch: {epochs.mean():.1f} +- {epochs.std():.1f}\n")
		f.write(f"time: {times.mean():.1f} +- {times.std():.1f}\n")

	# write training loss
	with open(os.path.join(output_dir, "gathered/train_loss.txt"), "w") as f:
		f.write("epoch,train_loss_mean,train_loss_std\n")
		for (i, (m, s)) in enumerate(zip(train_loss_mean, train_loss_std), start=1):
			f.write(f"{i:d},{m:.3f},{s:.3f}\n")

	# write validation loss
	with open(os.path.join(output_dir, "gathered/val_loss.txt"), "w") as f:
		f.write("epoch,val_loss_mean,val_loss_std\n")
		for (i, (m, s)) in enumerate(zip(val_loss_mean, val_loss_std), start=1):
			f.write(f"{i:d},{m:.3f},{s:.3f}\n")

	# write validation accuracy
	with open(os.path.join(output_dir, "gathered/val_acc.txt"), "w") as f:
		f.write("epoch,val_acc_mean,val_acc_std\n")
		for (i, (m, s)) in enumerate(zip(val_acc_mean, val_acc_std), start=1):
			f.write(f"{i:d},{m:.1f},{s:.1f}\n")

	return


def gather_scores(results_dir, Ns, architectures):
	"""Gather metrics for various architectures and training data sizes."""
	D = {"N": np.array(Ns, dtype=int)}  # init dict
	for architecture in architectures:
		D[f"{architecture}_mean"] = []
		D[f"{architecture}_std"] = []
	for (N, architecture) in itertools.product(Ns, architectures):
		with open(os.path.join(results_dir, f"N{N:d}/{architecture}/gathered/score.txt"), "r") as f:
			text = f.readlines()[0]
		chunks = text.strip("\n").split(" ")
		mean, std = chunks[1], chunks[3]
		D[f"{architecture}_mean"].append(mean)
		D[f"{architecture}_std"].append(std)
	headers = ["N"]
	for (architecture, stat) in itertools.product(architectures, ["mean", "std"]):
		D[f"{architecture}_{stat}"] = np.array(D[f"{architecture}_{stat}"], dtype=float)
		headers.append(f"{architecture}_{stat}")
	# convert dict to pandas dataframe
	df = pd.DataFrame.from_dict(D)[headers]
	return df


### plot scores

def subplot_scores(results_dir, Ns, architectures, labels, colors, title=None):
	"""Bar graph of scores across architectures and training data sizes."""
	data = gather_scores(results_dir, Ns, architectures)
	all_means = np.concatenate([getattr(data, f"{architecture}_mean").values for architecture in architectures], 0)
	all_stds = np.concatenate([getattr(data, f"{architecture}_std").values for architecture in architectures], 0)
	y_min, y_max = y_minmax(all_means, all_stds)
	x = np.arange(len(Ns))
	width, shifts = bar_width_shifts(len(architectures))
	for (architecture, shift) in zip(architectures, shifts):
		plt.bar(x+shift, getattr(data, f"{architecture}_mean").values, width, yerr=bar_yerrs(getattr(data, f"{architecture}_mean").values, getattr(data, f"{architecture}_std").values), capsize=2, ecolor=colors[architecture], color=colors[architecture], label=labels[architecture])
	plt.xticks(x, Ns)
	plt.ylim(y_min, y_max)
	plt.xlabel("Training data used (%)", fontsize=16)
	plt.ylabel("Val accuracy (%)", fontsize=16)
	if title is not None:
		plt.title(title, fontsize=16)


def plot_scores(results_dir, Ns, architectures, labels, colors):
	"""Bar graph of scores across architectures and training data sizes. Wraps around subplot_scores."""
	plt.figure(figsize=(figsize[0],figsize[0]))
	plt.subplot(1, 1, 1)
	subplot_scores(results_dir, Ns, architectures, labels, colors, title=None)
	handles, labels = get_unique_legend_handles_labels(plt.gcf())
	plt.figlegend(handles, labels, ncol=3, loc=(0.1,0.9), fancybox=True, fontsize=16)
	plt.tight_layout()
	plt.subplots_adjust(top=0.9, wspace=0.18)
	plt.savefig("plot.png")
	plt.close()


if __name__ == "__main__":
	results_dir = "results"
	Ns = [25, 50, 75, 100]  # training subsampling percentages
	architectures = ["type1", "mixed", "unraveled"]

	labels = ["Type 1", "Mixed", "Unraveled"]
	colors = ["red", "green", "blue"]

	labels = {key: value for (key, value) in zip(architectures, labels)}
	colors = {key: value for (key, value) in zip(architectures, colors)}

	print("Gathering scores . . . ")
	for (N, architecture) in itertools.product(Ns, architectures):
		gather_acc(os.path.join(results_dir, f"N{N:d}", architecture))

	print("Plotting . . . ")
	plot_scores(results_dir, Ns, architectures, labels, colors)

	print("Done!")
