from torch.utils.data import TensorDataset
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
import time

import copy
import math

from differentiable_rmpflow.rmpflow.utils import * # generate_trajectories, ContextDatasetMomentum, TrajectoryTrackingTimeDependentController


class Params(object):
	def __init__(self, **kwargs):
		super(Params, self).__init__()
		self.__dict__.update(kwargs)


# ----------------------------------------
def find_data_range(torch_datasets, idx=0):
	x_tensor = torch.tensor([])
	for dataset in torch_datasets:
		x_tensor = torch.cat((x_tensor, dataset.tensors[idx]), dim=0)

	x_min = torch.min(x_tensor, dim=0)[0].reshape(1,-1)
	x_max = torch.max(x_tensor, dim=0)[0].reshape(1,-1)

	return x_min, x_max

# ----------------------------------------
def train(model, loss_fn, opt, train_dataset,
		  n_epochs=500, batch_size=None, shuffle=True,
		  clip_gradient=True, clip_value_grad=0.1,
		  clip_weight=False, clip_value_weight=2,
		  log_freq=5, logger=None, loss_clip=1e3, stop_threshold=float('inf')):
	'''
	train the torch model with the given parameters
	:param model (torch.nn.Module): the model to be trained
	:param loss_fn (callable): loss = loss_fn(y_pred, y_target)
	:param opt (torch.optim): optimizer
	:param x_train (torch.Tensor): training data (position/position + velocity)
	:param y_train (torch.Tensor): training label (velocity/control)
	:param n_epochs (int): number of epochs
	:param batch_size (int): size of minibatch, if None, train in batch
	:param shuffle (bool): whether the dataset is reshuffled at every epoch
	:param clip_gradient (bool): whether the gradients are clipped
	:param clip_value_grad (float): the threshold for gradient clipping
	:param clip_weight (bool): whether the weights are clipped (not implemented)
	:param clip_value_weight (float): the threshold for weight clipping (not implemented)
	:param log_freq (int): the frequency for printing loss and saving results on tensorboard
	:param logger: the tensorboard logger
	:return: None
	'''

	# if batch_size is None, train in batch
	n_samples = len(train_dataset)
	if batch_size is None:
		train_loader = DataLoader(
			dataset=train_dataset,
			batch_size=n_samples,
			shuffle=shuffle
		)
		batch_size = n_samples
	else:
		train_loader = DataLoader(
			dataset=train_dataset,
			batch_size=batch_size,
			shuffle=shuffle
		)

	# record time elasped
	ts = time.time()

	if hasattr(loss_fn, 'reduction'):
		if loss_fn.reduction == 'mean':
			mean_flag = True
		else:
			mean_flag = False
	else:
		mean_flag = True 		# takes the mean by default

	best_train_loss = float('inf')
	best_train_epoch = 0
	best_model = model

	# train the model
	model.train()
	for epoch in range(n_epochs):
		# iterate over minibatches
		train_loss = 0.
		for x_batch, y_batch in train_loader:
			# forward pass
			if isinstance(x_batch, torch.Tensor):
				y_pred = model(x_batch)
			elif isinstance(x_batch, dict):
				y_pred = model(**x_batch)
			else:
				raise ValueError
			# compute loss
			loss = loss_fn(y_pred, y_batch)
			train_loss += loss.item()

			if loss > loss_clip:
				print('loss too large, skip')
				continue

			# backward pass
			opt.zero_grad()
			loss.backward()

			# clip gradient based on norm
			if clip_gradient:
				# torch.nn.utils.clip_grad_value_(
				#     model.parameters(),
				#     clip_value_grad
				# )
				torch.nn.utils.clip_grad_norm_(
					model.parameters(),
					clip_value_grad
				)
			# update parameters
			opt.step()

		if mean_flag:   # fix for taking mean over all data instead of mini batch!
			train_loss = float(batch_size)/float(n_samples)*train_loss

		if epoch - best_train_epoch >= stop_threshold:
			break

		if train_loss < best_train_loss:
			best_train_epoch = epoch
			best_train_loss = train_loss
			best_model = copy.deepcopy(model)

		# report loss in command line and tensorboard every log_freq epochs
		if epoch % log_freq == (log_freq - 1):
			print('    Epoch [{}/{}]: current loss is {}, time elapsed {} second'.format(
				epoch + 1, n_epochs,
				train_loss,
				time.time()-ts)
			)

			if logger is not None:
				info = {'Training Loss': train_loss}

				# log scalar values (scalar summary)
				for tag, value in info.items():
					logger.scalar_summary(tag, value, epoch + 1)

				# log values and gradients of the parameters (histogram summary)
				for tag, value in model.named_parameters():
					tag = tag.replace('.', '/')
					logger.histo_summary(
						tag,
						value.data.cpu().numpy(),
						epoch + 1
					)
					logger.histo_summary(
						tag + '/grad',
						value.grad.data.cpu().numpy(),
						epoch + 1
					)
	return best_model, best_train_loss


# -----------------------------------
def train_dagger(model, loss_fn, opt, concat_dataset, dt,
				 n_epochs=500, batch_size=None, shuffle=True,
				 clip_gradient=True, clip_value_grad=0.1,
				 clip_weight=False, clip_value_weight=2,
				 log_freq=5, logger=None, loss_clip=1e3, stop_threshold=float('inf'),
				 n_dagger_iterations=3, tracker_kp=4.0, split_ratio=0.32):
	'''
	train the torch model with the given parameters
	:param model (torch.nn.Module): the model to be trained
	:param loss_fn (callable): loss = loss_fn(y_pred, y_target)
	:param opt (torch.optim): optimizer
	:param x_train (torch.Tensor): training data (position/position + velocity)
	:param y_train (torch.Tensor): training label (velocity/control)
	:param n_epochs (int): number of epochs
	:param batch_size (int): size of minibatch, if None, train in batch
	:param shuffle (bool): whether the dataset is reshuffled at every epoch
	:param clip_gradient (bool): whether the gradients are clipped
	:param clip_value_grad (float): the threshold for gradient clipping
	:param clip_weight (bool): whether the weights are clipped (not implemented)
	:param clip_value_weight (float): the threshold for weight clipping (not implemented)
	:param log_freq (int): the frequency for printing loss and saving results on tensorboard
	:param logger: the tensorboard logger
	:param tracker_kp: p-gain for the trajectory tracking expert
	:param split_ratio: ratio of original vs new data

	:return:
	'''

	assert batch_size is None, NotImplementedError('Doesnt work for minibatches!')

	batch_size = len(concat_dataset)

	p = 0.05**(1./(n_dagger_iterations-1 + 1e-14))  # the probabilty at the last iteration should be low (0.05 here)
	traj_criterion = nn.MSELoss()

	datasets = concat_dataset.datasets
	n_experts = len(datasets)

	experts = []
	expert_traj_list = []
	expert_time_list = []
	s0_list = []
	x_experts = torch.tensor([])
	xd_experts = torch.tensor([])

	for n in range(n_experts):
		dataset = datasets[n]

		x_expert = dataset.tensors[0]
		xd_expert = dataset.tensors[1]
		t_expert = np.arange(0, x_expert.shape[0])*dt

		x_experts = torch.cat((x_experts, x_expert), dim=0)
		xd_experts = torch.cat((xd_experts, xd_expert), dim=0)
		expert_traj_list.append(x_expert)
		expert_time_list.append(t_expert)
		s0_list.append(x_expert[0].numpy())

		expert_model = TrajectoryTrackingTimeDependentController(x_expert,
																 xd_expert,
																 dt=dt, kp=tracker_kp, kd=1.0)
		experts.append(expert_model)

	dataset_expert = TensorDataset(x_experts, xd_experts)

	learner_time_dependent = TimeWrapperNet(model)  # just adding in an additional dummy time variable to the learner model
	mixture_model = MixtureNet(model_1=experts[0], model_2=learner_time_dependent, beta=1.)

	x_new = torch.tensor([])
	xd_new = torch.tensor([])

	best_mse = float('inf')
	best_iteration = 0
	best_model = model

	for i in range(n_dagger_iterations):
		beta = p**i
		mixture_model.set_beta(beta)

		print('----------------------------------------------------------')
		print('------------------- Dagger Iteration : {} -----------------'.format(i))

		if i == 0:
			print('---------------- Training on observed data ---------------')
			print('----------------------------------------------------------')
			dataset = dataset_expert
		else:
			print('-------------------- Aggregating data --------------------')
			print('------------ mixing_rate: {}, tracker_kp: {} ----------'.format(beta, tracker_kp))
			print('----------------------------------------------------------')
			for n in range(n_experts):
				s0 = s0_list[n]
				t_eval = expert_time_list[n]
				t_final = t_eval[-1]
				expert = experts[n]
				mixture_model.set_model_1(expert)

				mixture_model.eval()
				# NOTE: We have a time-dependent mixture here now!
				x_n = generate_trajectories(mixture_model, s0, time_dependent=True,
											order=1, return_label=False, t_step=dt, t_final=t_final, method='euler')

				xd_n_expert = expert(t_eval, x_n)

				x_new = torch.cat((x_new, x_n), dim=0)
				xd_new = torch.cat((xd_new, xd_n_expert), dim=0)

			dataset_new = TensorDataset(x_new, xd_new)

			batch_size_expert = int(math.ceil(split_ratio * batch_size))
			batch_size_new = int(batch_size - batch_size_expert)

			data_loader_expert = DataLoader(dataset_expert, batch_size=batch_size_expert, shuffle=True)
			data_loader_new = DataLoader(dataset_new, batch_size=batch_size_new, shuffle=True)

			x_train_expert, xd_train_expert = next(iter(data_loader_expert))
			x_train_new, xd_train_new = next(iter(data_loader_new))

			x_train = torch.cat((x_train_expert, x_train_new), dim=0)
			xd_train = torch.cat((xd_train_expert, xd_train_new), dim=0)

			dataset = TensorDataset(x_train, xd_train)

		model.train()
		train(model=model, loss_fn=loss_fn, opt=opt, train_dataset=dataset, n_epochs=n_epochs, batch_size=batch_size,
			  shuffle=shuffle,
			  clip_gradient=clip_gradient, clip_value_grad=clip_value_grad,
			  clip_weight=clip_weight, clip_value_weight=clip_value_weight,
			  log_freq=log_freq, logger=logger, loss_clip=loss_clip, stop_threshold=stop_threshold)

		model.eval()

		traj_mse = 0.
		for n in range(n_experts):
			s0 = s0_list[n]
			t_eval = expert_time_list[n]
			t_final = t_eval[-1]
			learner_traj = generate_trajectories(model, s0, time_dependent=False,
										order=1, return_label=False, t_step=dt, t_final=t_final, method='euler')

			traj_mse = traj_mse + traj_criterion(learner_traj, expert_traj_list[n]).item()
		traj_mse = traj_mse/n_experts

		if traj_mse < best_mse:
			best_mse = traj_mse
			best_iteration = i
			best_model = copy.deepcopy(model)

		print('Current Trajectory MSE: {}'.format(traj_mse))
		print('Best Trajectory MSE: {}, Best Iteration: {}'.format(best_mse, best_iteration))

	return best_model, best_mse

# ----------------------------------------------------------------------------------------------------


def train_dagger_context(context_model, loss_fn, opt, context_dataset, dt, context_rmptree_list, subtask_maps,
				 n_epochs=500, batch_size=None, shuffle=True,
				 clip_gradient=True, clip_value_grad=0.1,
				 clip_weight=False, clip_value_weight=2,
				 log_freq=5, logger=None, loss_clip=1e3, stop_threshold=float('inf'),
				 n_dagger_iterations=3, tracker_kp=4.0, split_ratio=0.32):
	'''
	train the torch model with the given parameters
	:param model (torch.nn.Module): the model to be trained
	:param loss_fn (callable): loss = loss_fn(y_pred, y_target)
	:param opt (torch.optim): optimizer
	:param x_train (torch.Tensor): training data (position/position + velocity)
	:param y_train (torch.Tensor): training label (velocity/control)
	:param n_epochs (int): number of epochs
	:param batch_size (int): size of minibatch, if None, train in batch
	:param shuffle (bool): whether the dataset is reshuffled at every epoch
	:param clip_gradient (bool): whether the gradients are clipped
	:param clip_value_grad (float): the threshold for gradient clipping
	:param clip_weight (bool): whether the weights are clipped (not implemented)
	:param clip_value_weight (float): the threshold for weight clipping (not implemented)
	:param log_freq (int): the frequency for printing loss and saving results on tensorboard
	:param logger: the tensorboard logger
	:param tracker_kp: p-gain for the trajectory tracking expert
	:param split_ratio: ratio of original vs new data

	:return:
	'''

	assert batch_size is None, NotImplementedError('Doesnt work for minibatches!')


	batch_size = len(context_dataset)

	p_ = 0.05**(1./(n_dagger_iterations-1 + 1e-14))  # the probabilty at the last iteration should be low (0.05 here)
	traj_criterion = nn.MSELoss()

	datasets = context_dataset.datasets
	n_experts = len(datasets)


	assert len(context_rmptree_list) == n_experts, ValueError('Rmptree list should have length equal to number of demos!')

	experts = []
	expert_traj_list = []
	expert_time_list = []
	expert_dataset_list = []
	q0_list = []

	k=0
	for n in range(n_experts):
		dataset = datasets[n]

		q_expert = dataset.state
		qd_expert = dataset.qd_config
		t_expert = np.arange(0, q_expert.shape[0])*dt

		expert_traj_list.append(q_expert)
		expert_time_list.append(t_expert)
		expert_dataset_list.append(TensorDataset(q_expert, qd_expert))
		q0_list.append(q_expert[0].numpy())

		expert_model = TrajectoryTrackingController(q_expert.numpy(), qd_expert.numpy(),
																 dt=dt, kp=tracker_kp, kd=1.0)

		experts.append(expert_model)

	cspace_dim = q_expert.shape[1]
	subtask_nets = context_model.lagrangian_vel_nets

	full_rmptree_list = []
	for context_rmptree in context_rmptree_list:
		n = 0
		for subtask_map in subtask_maps:
			subtask_node = context_rmptree.add_task_space(subtask_map, name='subtask_' + str(n))
			subtask_net = subtask_nets[n + 1]
			subtask_momentum_controller = LearnedMomentumController(momentum_net=subtask_net, metric_scaling=1.)
			subtask_rmp = MomentumControllerRmp(subtask_momentum_controller)
			subtask_node.add_rmp(subtask_rmp)
			n+=1
		full_rmptree_list.append(context_rmptree)

	best_mse = float('inf')
	best_iteration = 0
	best_model = context_model

	for i in range(n_dagger_iterations):
		beta = p_**i

		print('----------------------------------------------------------')
		print('------------------- Dagger Iteration : {} -----------------'.format(i))

		if i == 0:
			print('---------------- Training on observed data ---------------')
			print('----------------------------------------------------------')
			dataset = context_dataset
		else:
			print('-------------------- Aggregating data --------------------')
			print('------------ mixing_rate: {}, tracker_kp: {} ----------'.format(beta, tracker_kp))
			print('----------------------------------------------------------')

			context_model.eval()

			dataset_list_new = []
			for n in range(n_experts):
				t_eval = expert_time_list[n]
				t_final = t_eval[-1]
				expert = experts[n]
				learner = full_rmptree_list[n]
				context_rmptree = context_rmptree_list[n]

				# NOTE: We have a time-dependent mixture here now!
				N = int(t_final/dt + 1)

				q = q0_list[n].reshape(1, -1)
				q_traj = np.zeros((N, cspace_dim))
				qd_traj = np.zeros((N, cspace_dim))
				qd_traj_expert = np.zeros((N, cspace_dim))

				x_list = []
				J_list = []
				p_list = []
				m_list = []

				p = np.zeros((N, cspace_dim))
				m = np.zeros((N, cspace_dim, cspace_dim))

				for i in range(N):
					alpha = float(np.random.binomial(1, beta))
					qd_learner = learner.eval_canonical(t=i*dt, s=State(q.T, order=1)).xd.T
					qd_expert = expert(t=i*dt, x=q)
					qd = alpha*qd_expert + (1.-alpha)*qd_learner

					q_traj[i, :] = q
					qd_traj[i, :] = qd
					qd_traj_expert[i, :] = qd_expert

					q = q + dt * qd

					context_rmp_eval = context_rmptree.eval_natural(t=i*dt, s=State(q.T, order=1))
					p[i] = context_rmp_eval.p.T
					m[i] = context_rmp_eval.M

				x_list.append(torch.from_numpy(q_traj).float())
				J_list.append(torch.eye(cspace_dim, dtype=torch.float).repeat((N, 1, 1)))
				p_list.append(torch.from_numpy(p).float())
				m_list.append(torch.from_numpy(m).float())

				for subtask_map in subtask_maps:
					workspace_dim = subtask_map.num_outputs
					x = np.zeros((N, workspace_dim))
					J = np.zeros((N, workspace_dim, cspace_dim))

					for n in range(N):
						x[n] = subtask_map.psi(q_traj[n].reshape(-1, 1)).flatten()
						J[n] = subtask_map.J(q_traj[n].reshape(-1, 1))

					x_list.append(torch.from_numpy(x).float())
					J_list.append(torch.from_numpy(J).float())
					p_list.append(torch.zeros(N, 1))
					m_list.append(torch.zeros(N, 1))

				dataset = ContextDatasetMomentum(
					torch.from_numpy(q_traj).float(),
					torch.from_numpy(qd_traj_expert).float(),
					x_list, J_list, p_list, m_list)

				dataset_list_new.append(dataset)

			context_dataset_new = ConcatDataset(dataset_list_new)

			batch_size_expert = int(math.ceil(split_ratio * batch_size))
			batch_size_new = int(batch_size - batch_size_expert)

			data_loader_expert = DataLoader(context_dataset, batch_size=batch_size_expert, shuffle=True)
			data_loader_new = DataLoader(context_dataset_new, batch_size=batch_size_new, shuffle=True)

			x_train_expert, y_train_expert = next(iter(data_loader_expert))
			x_train_new, y_train_new = next(iter(data_loader_new))

			x_train = torch.cat((x_train_expert, x_train_new), dim=0)
			y_train = torch.cat((y_train_expert, y_train_new), dim=0)

			dataset = ContextDatasetMomentum(x_train, *y_train)

		context_model.train()
		best_model, best_mse = train(model=context_model, loss_fn=loss_fn, opt=opt, train_dataset=dataset, n_epochs=n_epochs, batch_size=batch_size,
			  shuffle=shuffle,
			  clip_gradient=clip_gradient, clip_value_grad=clip_value_grad,
			  clip_weight=clip_weight, clip_value_weight=clip_value_weight,
			  log_freq=log_freq, logger=logger, loss_clip=loss_clip, stop_threshold=stop_threshold)

		# context_model.eval()
		#
		# traj_mse = 0.
		# for n in range(n_experts):
		# 	q0 = q0_list[n]
		# 	t_eval = expert_time_list[n]
		# 	t_final = t_eval[-1]
		# 	learner_traj = generate_trajectories(rmptree_net_list[n], q0, time_dependent=True,
		# 								order=1, return_label=False, t_step=dt, t_final=t_final, method='euler')
		#
		# 	traj_mse = traj_mse + traj_criterion(learner_traj, expert_traj_list[n]).item()
		# traj_mse = traj_mse/n_experts
		#
		# if traj_mse < best_mse:
		# 	best_mse = traj_mse
		# 	best_iteration = i
		# 	best_model = copy.deepcopy(context_model)
		#
		# print('Current Trajectory MSE: {}'.format(traj_mse))
		# print('Best Trajectory MSE: {}, Best Iteration: {}'.format(best_mse, best_iteration))

	return best_model, best_mse


# ---------------------------------------------

class MixtureNet(nn.Module):
	def __init__(self, model_1, model_2, beta):
		super(MixtureNet, self).__init__()
		self.model_1 = model_1
		self.model_2 = model_2
		self.beta = beta

	def forward(self, *args, **kwargs):
		alpha = np.random.binomial(1, self.beta)
		return float(alpha)*self.model_1(*args, **kwargs) + (1.-alpha)*self.model_2(*args, **kwargs)

	def set_beta(self, beta):
		self.beta = beta

	def set_model_1(self, model_1):
		self.model_1 = model_1

	def set_model_2(self, model_2):
		self.model_2 = model_2


# ---------------------------------------
# Move outside learning!
class TrajectoryTrackingTimeDependentController(nn.Module):
	def __init__(self, x_traj, xd_traj, dt, kp=4., kd=1.):
		self.x_traj = x_traj   # list of positions dt spaced in time
		self.xd_traj = xd_traj # list of velocities dt spaced in time
		self.kp = kp
		self.kd = kd
		self.dt = dt
		super(TrajectoryTrackingTimeDependentController, self).__init__()

	def forward(self, t, x):
		# assuming robot starts from zero velocity!
		idx = np.round(t / self.dt).astype(int)
		xdd = -self.kp*(x-self.x_traj[idx]) -self.kd*(0.0 - self.xd_traj[idx])
		return xdd


class TimeWrapperNet(nn.Module):
	def __init__(self, net):
		super(TimeWrapperNet, self).__init__()
		self.net = net

	def forward(self, t, x):
		if isinstance(x, torch.Tensor):
			return self.net(x)
		elif isinstance(x, dict):
			return self.net(**x)
		else:
			raise ValueError


