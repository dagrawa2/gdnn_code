"""Context managers."""

from contextlib import contextmanager

@contextmanager
def evaluating(net):
	"""Sets a model to eval mode within this scope.

		Args:
			net (torch.nn.Module): Model.
		"""
	istrain = net.training
	try:
		net.eval()
		yield net
	finally:
		if istrain:
			net.train()
