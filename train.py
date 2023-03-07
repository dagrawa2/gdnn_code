import os
import json
import pickle
import time
import argparse
import warnings
warnings.filterwarnings("ignore")  # due to no GPU

import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import gdnn


# command-line arguments
parser=argparse.ArgumentParser()
# dataset
parser.add_argument('--data_dir', '-dd', default='data', type=str, help='Data directory.')
parser.add_argument('--percentage', '-p', default=0, type=int, help='Percentage of modelnet40 training set to use.')
# network architecture
parser.add_argument('--abs_patterns', '-a', action="store_true", help='Use absolute value of weight patterns.')
parser.add_argument('--tunnel_reps', '-t', action="store_true", help='Tunnel type 2 irreps into type 1 of same degree.')
parser.add_argument('--unravel_reps', '-u', action="store_true", help='Unravel type 2 irreps into larger type 1.')
parser.add_argument('--channels', '-c', required=True, type=int, nargs='+', help='Output channels in each layer.')
parser.add_argument('--subgroup_idxs', '-i', required=True, type=int, nargs='+', help='Index of subgroups modded out in each layer.')
parser.add_argument('--batch_norm', '-bn', action="store_true", help='Turn on batch normalization.')
# SGD hyperparameters
parser.add_argument('--batch_size', '-b', default=128, type=int, help='Minibatch size.')
parser.add_argument('--epochs', '-e', default=10, type=int, help='Number of epochs for training.')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='Learning rate.')
parser.add_argument('--lr_decay', '-ld', default=0.5, type=float, help='LR scheduler decay rate if lr_step > 0.')
parser.add_argument('--lr_step', '-ls', default=0, type=int, help='LR scheduler step size.')
# validation
parser.add_argument('--val_train', '-vt', action="store_true", help='Evaluate on training set along with validation set.')
parser.add_argument('--val_batch_size','-vb', default=512, type=int, help='Minibatch size during validation/testing.')
parser.add_argument('--val_interval','-vi', default=0, type=int, help='Epoch interval at which to record validation metrics. If 0, test metrics are not recorded.')
# output options
parser.add_argument('--output_dir', '-o', required=True, type=str, help='Output directory.')
parser.add_argument('--exist_ok', '-ok', action="store_true", help='Allow overwriting the output directory.')
parser.add_argument('--save_initial', '-si', action="store_true", help='Save initial parameters.')
parser.add_argument('--save_params', '-sp', action="store_true", help='Save parameters.')
parser.add_argument('--save_patterns', '-cp', default="", type=str, help='Save weight patterns to this file.')
# loading options
parser.add_argument('--load_reps', '-lrp', default="", type=str, help='Load reps from this file.')
parser.add_argument('--load_patterns', '-lp', default="", type=str, help='Load weight patterns from this file.')
parser.add_argument('--special_init', '-spi', default="", type=str, help='Load type 2 weights into unraveled architecture from this file.')
# misc
parser.add_argument('--device', '-dv', default="cpu", type=str, help='Device.')
parser.add_argument('--seed','-s', default=0, type=int, help='RNG seed.')
args=parser.parse_args()


# fix the random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# record initial time
time_start = time.time()

# create output directory
output_dir = args.output_dir
if os.path.exists(output_dir) \
	and not args.exist_ok:
	print("Output directory already exists; terminating script.")
	import sys; sys.exit(0)
os.makedirs(output_dir, exist_ok=args.exist_ok)

# dir for saving patterns
save_patterns = args.save_patterns if args.save_patterns != "" else None
if save_patterns is not None:
	os.makedirs(os.path.dirname(save_patterns), exist_ok=True)

# loading files
load_reps = args.load_reps if args.load_reps != "" else None
load_patterns = args.load_patterns if args.load_patterns != "" else None

# group generators
with open(os.path.join(args.data_dir, "generators.pkl"), "rb") as f:
	generators = pickle.load(f)

# data loaders
with np.load(os.path.join(args.data_dir, "dataset.npz")) as dataset:
	x_train, y_train = dataset["x_train"], dataset["y_train"]
	if args.percentage > 0 and args.percentage < 100:
		x_train, y_train = gdnn.utils.subsample(x_train, y_train, args.percentage)
	train_loader = DataLoader(TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train)), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=8)
	if args.val_train:
		train4val_loader = DataLoader(TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train)), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)
	val_loader = DataLoader(TensorDataset(torch.as_tensor(dataset["x_test"]), torch.as_tensor(dataset["y_test"])), batch_size=args.val_batch_size, shuffle=False, drop_last=False, num_workers=8)

# loss function and metrics
if y_train.max() >= 2:
	loss_fn = F.cross_entropy
	accuracy_fn = lambda y, targets: y.argmax(1).eq(targets).float().mean()
else:
	loss_fn = F.binary_cross_entropy_with_logits
	accuracy_fn = lambda y, targets: y.squeeze(1).ge(0).long().eq(targets.squeeze(1).long()).float().mean()
metrics = {"accuracy": accuracy_fn}

# channels
in_channels = x_train.shape[1]
channels = []
for (out_channels, subgroup_idx) in zip(args.channels, args.subgroup_idxs):
	channels.append( (in_channels, out_channels, subgroup_idx) )
	in_channels = out_channels

# build model
model = gdnn.models.GDNN(generators, channels, batch_norm=args.batch_norm, tunnel_reps=args.tunnel_reps, unravel_reps=args.unravel_reps, load_reps=load_reps, load_patterns=load_patterns, save_patterns=save_patterns)

# special init for unraveled architecture on binary multiplication example
if args.special_init != "":
	with torch.no_grad():
		weights = torch.load(args.special_init)
		for (n, p) in model.named_parameters():
			p.data = torch.zeros_like(p)
		for l in range(4):
			for i in range(8//2**l):
				for j in range(4):
					model.layers[l].all2irrep_modules[i].weights[0].data[...,2*i+j] = (-1)**j*weights[f"layers.{l:d}.all2irrep_modules.{i:d}.weights.0"][...,j//2]
		for j in range(5):
			model.layers[4].all2irrep_modules[0].weights[j].data = weights[f"layers.4.all2irrep_modules.0.weights.{j:d}"]
		model.layers[4].all2irrep_modules[0].biases.data = weights[f"layers.4.all2irrep_modules.0.biases"]
		for (n, p) in model.named_parameters():
			if "layers.4" not in n:
				p.data *= 0.5
	import gc; del weights; gc.collect()

# save initial parameters
if args.save_initial:
	torch.save(model.state_dict(), os.path.join(output_dir, "initial.pth"))

# create trainer and callbacks
trainer = gdnn.trainers.Trainer(model, loss_fn, metrics=metrics, epochs=args.epochs, lr=args.lr, lr_decay_rate=args.lr_decay, lr_step_size=args.lr_step, device=args.device)
callbacks = [gdnn.callbacks.Training()]
if args.val_interval > 0:
	if args.val_train:
		callbacks.append( gdnn.callbacks.Validation(trainer, train4val_loader, epoch_interval=args.val_interval, prefix="train", print_loss=False) )
	callbacks.append( gdnn.callbacks.Validation(trainer, val_loader, epoch_interval=args.val_interval) )

# train model
trainer.fit(train_loader, callbacks)

# function to convert np array to list of python numbers
ndarray2list = lambda arr, dtype: [getattr(__builtins__, dtype)(x) for x in arr]

# collect results
results_dict = {
#	"data_shapes": {name: list(A.shape) for (name, A) in [("X_1", X_1), ("X_2", X_2)]}, 
	"train": {key: ndarray2list(value, "float") for cb in callbacks for (key, value) in cb.history.items()}, 
}

# add command-line arguments, number of model parameters, and script execution time to results
results_dict["args"] = dict(vars(args))
results_dict["num_params"] = sum([p.numel() for p in model.parameters()])
results_dict["time"] = time.time()-time_start

# save results
print("Saving results ...")
with open(os.path.join(output_dir, "results.json"), "w") as fp:
	json.dump(results_dict, fp, indent=2)
if args.save_params:
	trainer.save_model(os.path.join(output_dir, "model.pth"))


print("Done!")
