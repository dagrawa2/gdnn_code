import time
import numpy as np


class Callback(object):
	"""Base class for callbacks to monitor training."""

	def __init__(self, *args, **kwargs):
		"""Class constructor."""
		pass

	def start_of_training(self):
		"""Called before training loop."""
		pass

	def start_of_epoch(self, epoch):
		"""Called at the start of each epoch.

		Args:
			epoch: Type int, number of epochs that have already completed.
		"""
		pass

	def end_of_epoch(self, epoch, batch_logs):
		"""Called at the end of each epoch.
			epochs: Type int, number of epochs that have already completed,
				not including the current one.
			batch_logs: Dict of metrics on the training minibatches of this epoch.
		"""
		pass

	def end_of_training(self):
		"""Called after the training loop."""
		pass


class Training(Callback):
	"""Monitor metrics on training set based on minibatches."""

	def __init__(self):
		super(Training, self).__init__()
		self.history = {"epoch": [], "time": [], "loss": []}

	def start_of_training(self):
		self._initial_epochs = self.history["epoch"][-1] if len(self.history["epoch"]) > 0 else 0
		if isinstance(self.history["epoch"], np.ndarray):
			self.history = {key: list(value) for (key, value) in self.history.items()}

	def start_of_epoch(self, epoch):
		self._initial_time = time.time()

	def end_of_epoch(self, epoch, batch_logs):
		delta_time = time.time() - self._initial_time
		loss = np.mean(np.asarray(batch_logs["loss"]))
		self.history["epoch"].append(self._initial_epochs+epoch+1)
		self.history["time"].append( self.history["time"][-1]+delta_time ) if len(self.history["time"]) > 0 else self.history["time"].append(delta_time)
		self.history["loss"].append(loss)
		print(f"Epoch {self._initial_epochs+epoch+1:d} loss {loss:.3f}")

	def end_of_training(self):
		self.history = {key: np.array(value) for (key, value) in self.history.items()}


class Validation(Callback):
	"""Monitor metrics on validation set."""

	def __init__(self, trainer, val_loader, epoch_interval=10, prefix="val", print_loss=True):
		"""Args:
			trainer: trainer.Trainer object, 
				used to access trainer and model methods and parameters.
			val_loader: DataLoader object,
				Validation data loader.
			epoch_interval: Int, 
				Interval between epochs when validation metrics should be calculated.
			prefix: Str, 
				Prefix used in history keys.
			print_loss: Bool, 
				Print validation loss to stdout.
		"""
		super(Validation, self).__init__()
		self.trainer = trainer
		self.val_loader = val_loader
		self.epoch_interval = epoch_interval
		self.prefix = prefix
		self.print_loss = print_loss
		self.history = {f"{self.prefix}_epoch": [], f"{self.prefix}_loss": []}
		for metric_name in self.trainer.metrics.keys():
			self.history[f"{self.prefix}_{metric_name}"] = []

	def start_of_training(self):
		self.history[f"{self.prefix}_epoch"].append(0)
		metric_dict = self.trainer.evaluate(self.val_loader)
		for (metric_name, score) in metric_dict.items():
			self.history[f"{self.prefix}_{metric_name}"].append(score)

	def end_of_epoch(self, epoch, batch_logs):
		if (epoch+1)%self.epoch_interval != 0:
			pass
		else:
			self.history[f"{self.prefix}_epoch"].append(epoch+1)
			metric_dict = self.trainer.evaluate(self.val_loader)
			for (metric_name, score) in metric_dict.items():
				self.history[f"{self.prefix}_{metric_name}"].append(score)
			if self.print_loss:
				loss = metric_dict["loss"]
				print(f". . . val loss {loss:.3f}")

	def end_of_training(self):
		self.history = {key: np.array(value) for (key, value) in self.history.items()}
