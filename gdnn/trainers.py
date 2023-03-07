import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from .ctx_managers import evaluating


class Trainer(object):

	def __init__(self, model, loss_fn, metrics={}, epochs=1, lr=1e-3, lr_decay_rate=0.5, lr_step_size=0, device="cpu"):
		self.model = model.to(device)
		self.loss_fn = loss_fn
		self.metrics = metrics
		self.epochs = epochs
		self.lr = lr
		self.lr_decay_rate = lr_decay_rate
		self.lr_step_size = lr_step_size
		self.device = device

		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.lr_step_size, gamma=self.lr_decay_rate) \
			if self.lr_step_size > 0 else None

	def fit(self, train_loader, callbacks):
		self.model.train()
		# callbacks before training
		callbacks = callbacks if isinstance(callbacks, list) else [callbacks]
		for cb in callbacks:
			cb.start_of_training()
		print("Training model ...")
		print("---")
		for epoch in range(self.epochs):
			# callbacks at the start of the epoch
			for cb in callbacks:
				cb.start_of_epoch(epoch)
			batch_logs = {"loss": []}
			for (X_batch, Y_batch) in train_loader:
				X_batch = X_batch.to(self.device)
				Y_batch = Y_batch.to(self.device)
				preds_batch = self.model(X_batch)
				loss = self.loss_fn(preds_batch, Y_batch)
				loss.backward()
				self.optimizer.step()
				self.optimizer.zero_grad()
				batch_logs["loss"].append(loss.item())
			if self.lr_scheduler is not None:
				self.lr_scheduler.step()
			# callbacks at the end of the epoch
			for cb in callbacks:
				cb.end_of_epoch(epoch, batch_logs)

		# callbacks at the end of training
		for cb in callbacks:
			cb.end_of_training()
		print("---")

	def evaluate(self, val_loader):
		preds, targets = [], []
		with evaluating(self.model), torch.no_grad():
			for (X_batch, Y_batch) in val_loader:
				X_batch = X_batch.to(self.device)
				pred_batch = self.model(X_batch)
				preds.append(pred_batch)
				targets.append(Y_batch)

			preds = torch.cat(preds)
			targets = torch.cat(targets)

			loss = self.loss_fn(preds, targets).item()
			metrics_dict = {"loss": loss}

			for (metric_name, metric) in self.metrics.items():
				metrics_dict[metric_name] = metric(preds, targets).item()

		return metrics_dict


	def predict(self, data_loader):
		preds = []
		with evaluating(self.model), torch.no_grad():
			for (X_batch, ) in data_loader:
				X_batch = X_batch.to(self.device)
				preds_batch = self.model(X_batch)
				preds.append( preds_batch.cpu().numpy() )
		preds = np.concatenate(preds, 0)

		return preds


	def save_model(self, filename):
		torch.save(self.model.state_dict(), filename)

	def load_model(self, filename):
		self.model.load_state_dict(torch.load(filename))
