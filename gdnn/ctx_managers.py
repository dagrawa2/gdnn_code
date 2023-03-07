from contextlib import contextmanager

@contextmanager
def evaluating(net):
	istrain = net.training
	try:
		net.eval()
		yield net
	finally:
		if istrain:
			net.train()
