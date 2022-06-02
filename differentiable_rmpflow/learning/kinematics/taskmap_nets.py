import torch

import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import numpy as np

from differentiable_rmpflow.rmpflow.kinematics import TaskMap, ComposedTaskMap


class EuclideanizingFlow(ComposedTaskMap):
	"""
	A sequential container of flows.
	num_dims: dimensions of input and output
	num_blocks: number of modules in the flow
	num_hidden: hidden layer dimensions or number of features for scaling and translation functions
	s_act: (only for coupling_layer_type='fcnn') activation fcn for scaling
	t_act: (only for coupling_layer_type:'fcnn') activation fcn for translation
	sigma: (only for coupling_layer_type:'rfnn') length scale for random for fourier features
	flow_type: (realnvp/glow) selects the submodules in each module depending on the architecture
	coupling_layer_type: (fcnn/rfnn) representatation for scaling and translation

	NOTE: Orginal RealNVP and GLOW uses normalization routines, skipped for now!
	"""
	def __init__(self, n_inputs, n_blocks, n_hidden, s_act=None, t_act=None, sigma=None,
				 flow_type='realnvp',
				 coupling_network_type='fcnn',
				 goal=None,
				 normalization_scaling=None,
				 normalization_bias=None,
				 device=torch.device('cpu')):

		super(EuclideanizingFlow, self).__init__(device=device)

		taskmap_list = []
		# data normalization step
		input_norm_taskmap = ElementwiseAffineTaskMap(n_inputs=n_inputs, bias=normalization_bias,
													  scaling=normalization_scaling, device=device)
		taskmap_list.append(input_norm_taskmap)

		# space warping step
		print('Using the {} for coupling layer'.format(coupling_network_type))
		if flow_type == 'realnvp':
			mask = torch.arange(0, n_inputs) % 2  # alternating inputs
			mask = mask.to(device=device, dtype=torch.get_default_dtype())
			for _ in range(n_blocks):   # TODO: Try batchnorm again
				taskmap_list += [
					CouplingLayer(
						n_inputs=n_inputs, n_hidden=n_hidden, mask=mask,
						s_act=s_act, t_act=t_act, sigma=sigma, base_network=coupling_network_type, device=device),
				]
				# mask = 1 - mask  # flipping mask
				mask = torch.roll(mask, shifts=-1, dims=0)
		elif flow_type == 'glow':
			mask = torch.arange(0, n_inputs) % 2  # alternating inputs
			mask = mask.to(device=device, dtype=torch.get_default_dtype())
			for _ in range(n_blocks):		# TODO: Try ActNorm again
				taskmap_list += [
					InvertibleMM(n_inputs),
					CouplingLayer(
						n_inputs=n_inputs, n_hidden=n_hidden, mask=mask,
						s_act=s_act, t_act=t_act, sigma=sigma, base_network=coupling_network_type, device=device),
				]
				# mask = 1 - mask  # flipping mask			# TODO: Not sure if this mask needs flipping
				# mask = torch.roll(mask, shifts=-1, dims=0)
		else:
			raise TypeError('Unknown Flow Type!')

		# defining a composed map for normalization and warping
		latent_taskmap = ComposedTaskMap(taskmaps=taskmap_list, device=device)

		# setting up the overall composed taskmap (euclidenization + fixing origin)
		self.taskmaps.append(latent_taskmap)

		# fixing the origin to goal in latent space
		latent_target_taskmap = LatentTargetTaskMap(n_inputs=n_inputs, latent_taskmap=latent_taskmap,
													goal=goal, device=device)

		self.taskmaps.append(latent_target_taskmap)

		if flow_type == 'realnvp' and coupling_network_type == 'rfnn':	# use analytic jacobian for rfnn type
			print('Using Analytic Jacobian!')
			self.use_numerical_jacobian = False
		else:
			print('Using Numerical Jacobian!')
			self.use_numerical_jacobian = True


# -----------------------------------------------------------------------------------------------

class ElementwiseAffineTaskMap(TaskMap):
	def __init__(self, n_inputs, scaling=None, bias=None, device=torch.device('cpu')):
		super(ElementwiseAffineTaskMap, self).__init__(n_inputs=n_inputs, n_outputs=n_inputs, psi=self.psi,
													   J=self.J,
													   J_dot=self.J_dot,
													   device=device)
		if scaling is not None:
			if scaling.dim() == 1:
				scaling = scaling.reshape(1, -1)
			self.register_buffer('scaling', scaling.to(device=device, dtype=torch.get_default_dtype()))
		else:
			self.register_buffer('scaling', torch.ones(1, n_inputs, device=device))

		if bias is not None:
			if bias.dim() == 1:
				bias = bias.reshape(1, -1)
			self.register_buffer('bias', bias.to(device=device, dtype=torch.get_default_dtype()))
		else:
			self.register_buffer('bias', torch.zeros(1, n_inputs, device=device))

	def psi(self, x):
		return x * self.scaling + self.bias

	def J(self, x):
		return torch.diag_embed(self.scaling).repeat(x.shape[0], 1, 1)

	def J_dot(self, x, xd):
		return torch.zeros(self.n_inputs, self.n_inputs, device=self.device).repeat(x.shape[0], 1, 1)


# --------------------------------------------------------------------------------------------
class LatentTargetTaskMap(TaskMap):
	def __init__(self, n_inputs, latent_taskmap, goal=None, device=torch.device('cpu')):
		super(LatentTargetTaskMap, self).__init__(n_inputs=n_inputs, n_outputs=n_inputs, psi=self.psi,
												  J = self.J,
												  J_dot=self.J_dot,
												  device=device)
		self.latent_taskmap = latent_taskmap

		if goal is not None:
			if goal.ndim == 1:
				goal = goal.reshape(1, -1)
			self.register_buffer('goal', goal.to(device=device, dtype=torch.get_default_dtype()))
		else:
			self.register_buffer('goal', torch.zeros(1, self.n_inputs, device=device))

	def psi(self, x):
		return x - self.latent_taskmap.psi(self.goal)

	def J(self, x):
		return torch.eye(self.n_inputs).repeat(x.shape[0], 1, 1)

	def J_dot(self, x, xd):
		return torch.zeros(self.n_inputs, self.n_inputs,  device=self.device).repeat(x.shape[0], 1, 1)


# -----------------------------------------------------------------------------------------

class CouplingLayer(TaskMap):
	""" An implementation of a coupling layer
	from RealNVP (https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self, n_inputs, n_hidden, mask,
				 base_network='rfnn', s_act='elu', t_act='elu', sigma=0.45, device=torch.device('cpu')):

		super(CouplingLayer, self).__init__(n_inputs=n_inputs, n_outputs=n_inputs, psi=self.psi, J=self.jacobian,
											device=device)

		self.num_inputs = n_inputs
		self.register_buffer('mask', mask.to(device=device, dtype=torch.get_default_dtype()))

		if base_network == 'fcnn':
			self.scale_net = \
				FCNN(in_dim=n_inputs, out_dim=n_inputs, hidden_dim=n_hidden, act=s_act).to(device=device)
			self.translate_net = \
				FCNN(in_dim=n_inputs, out_dim=n_inputs, hidden_dim=n_hidden, act=t_act).to(device=device)
			print('neural network initialized with identity map!')

			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.translate_net.network[-1].bias.data)

			nn.init.zeros_(self.scale_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].bias.data)

		elif base_network == 'rfnn':
			print('Random fouier feature bandwidth = {}. Change it as per data!'.format(sigma))
			self.scale_net = \
				RFFN(in_dim=n_inputs, out_dim=n_inputs, nfeat=n_hidden, sigma=sigma).to(device=device)
			self.translate_net = \
				RFFN(in_dim=n_inputs, out_dim=n_inputs, nfeat=n_hidden, sigma=sigma).to(device=device)

			print('Initializing coupling layers as identity!')
			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].weight.data)
		else:
			raise TypeError('The network type has not been defined')

	def psi(self, x, mode='direct'):
		mask = self.mask
		masked_inputs = x * mask

		log_s = self.scale_net(masked_inputs) * (1 - mask)
		t = self.translate_net(masked_inputs) * (1 - mask)

		if mode == 'direct':
			s = torch.exp(log_s)
			return x * s + t
		else:
			s = torch.exp(-log_s)
			return (x - t) * s

	def jacobian(self, x, mode='direct'):
		mask = self.mask
		masked_inputs = x * mask
		if mode == 'direct':
			log_s = self.scale_net(masked_inputs) * (1 - mask)
			s = torch.exp(log_s)									#TODO: remove redundancy!
			J_s = self.scale_net.jacobian(masked_inputs)
			J_t = self.translate_net.jacobian(masked_inputs)

			J = (J_s * (x * s * (1 - mask)).unsqueeze(2) + J_t * ((1 - mask).unsqueeze(1))) * mask
			J = J + torch.diag_embed(s)
			return J
		else:
			raise NotImplementedError


class InvertibleMM(TaskMap):
	""" An implementation of a invertible matrix multiplication
	layer from Glow: Generative Flow with Invertible 1x1 Convolutions
	(https://arxiv.org/abs/1807.03039).
	"""

	def __init__(self, n_inputs, device=torch.device('cpu')):
		self.device = device
		self.num_inputs = n_inputs
		self.W_nn = nn.Parameter(torch.zeros(n_inputs, n_inputs, device=self.device), requires_grad=True)
		print('Initializing invertible convolution layer as identity!')
		super(InvertibleMM, self).__init__(n_inputs=n_inputs, n_outputs=n_inputs, psi=self.psi, device=device)

	@property
	def W(self):
		return self.W_nn + torch.eye(self.num_inputs, device=self.device)

	def psi(self, x, mode='direct'):
		if mode == 'direct':
			return x.matmul(self.W)
		else:
			return x.matmul(torch.inverse(self.W))


class RFFN(nn.Module):
	"""
	Random Fourier features network.
	"""

	def __init__(self, in_dim, out_dim, nfeat, sigma=10.):
		super(RFFN, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.sigma = np.ones(in_dim) * sigma
		self.coeff = np.random.normal(0.0, 1.0, (nfeat, in_dim))
		self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
		self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

		self.register_buffer('coeff_tensor', torch.tensor(self.coeff).to(dtype=torch.get_default_dtype()))
		self.register_buffer('offset_tensor', torch.tensor(self.offset).to(dtype=torch.get_default_dtype()))


		self.network = nn.Sequential(
			LinearClamped(in_dim, nfeat, self.coeff, self.offset),
			Cos(),
			nn.Linear(nfeat, out_dim, bias=False)
		)

		self.jacobian = self.jacobian_analytic

	def forward(self, x):
		return self.network(x)

	def jacobian_analytic(self, x):
		n = x.shape[0]
		y = F.linear(x, self.coeff_tensor, self.offset_tensor)
		y = -torch.sin(y)
		y = y.repeat_interleave(self.in_dim, dim=0)
		y = y * self.coeff_tensor.t().repeat(n, 1)
		J = F.linear(y, self.network[-1].weight, bias=None)
		J = J.reshape(n, self.out_dim, self.in_dim).permute(0,2,1)
		return J

	def jacobian_numeric(self, inputs):
		if inputs.ndimension() == 1:
			n = 1
		else:
			n = inputs.size()[0]
		inputs = inputs.repeat(1, self.in_dim).view(-1, self.in_dim)
		inputs.requires_grad_(True)
		y = self(inputs)
		mask = torch.eye(self.in_dim).repeat(n, 1)
		J = autograd.grad(y, inputs, mask, create_graph=True)[0]
		J = J.reshape(n, self.in_dim, self.in_dim)
		return J


class FCNN(nn.Module):
	'''
	2-layer fully connected neural network
	'''

	def __init__(self, in_dim, out_dim, hidden_dim, act='tanh'):
		super(FCNN, self).__init__()
		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
					   'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}

		act_func = activations[act]
		self.network = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, out_dim)
		)

	def forward(self, x):
		return self.network(x)


class LinearClamped(nn.Module):
	'''
	Linear layer with user-specified parameters (not to be learrned!)
	'''

	__constants__ = ['bias', 'in_features', 'out_features']

	def __init__(self, in_features, out_features, weights, bias_values=None):
		super(LinearClamped, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.register_buffer('weight', torch.tensor(weights).to(dtype=torch.get_default_dtype()))
		if bias_values is not None:
			self.register_buffer('bias', torch.tensor(bias_values).to(dtype=torch.get_default_dtype()))
		else:
			self.register_buffer('bias', None)

	def forward(self, input):
		if input.dim() == 1:
			return F.linear(input.view(1, -1), self.weight, self.bias)
		return F.linear(input, self.weight, self.bias)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)


class Sin(nn.Module):
	"""
	Applies the cosine element-wise function
	"""

	def forward(self, inputs):
		return torch.sin(inputs)


class Cos(nn.Module):
	"""
	Applies the cosine element-wise function
	"""

	def forward(self, inputs):
		return torch.cos(inputs)


# class FirstOrderDataset(Dataset):
# 	def __init__(self, x_train, xd_train, do_normalize=True):
# 		self.x_train = x_train
# 		self.xd_train = xd_train
# 		self.do_normalize = do_normalize
#
# 		minx = np.min(x_train, axis=0).reshape(1, -1)
# 		maxx = np.max(x_train, axis=0).reshape(1, -1)
# 		scaling = 1. / (maxx - minx)
# 		translation = -minx / (maxx - minx) - 0.5
#
# 		self.scaling = torch.from_numpy(scaling).float()
# 		self.translation = torch.from_numpy(translation).float()
#
# 	def __getitem__(self, idx):
# 		if torch.is_tensor(idx):
# 			idx = idx.tolist()
#
# 		x_batch = self.x_train[idx]
# 		xd_batch = self.xd_train[idx]
#
# 		if self.do_normalize:
# 			x_batch = self.normalize_pos(x_batch)
# 			xd_batch = self.normalize_vel(xd_batch)
#
# 		return x_batch, xd_batch
#
# 	def normalize_pos(self, x):
# 		return x * self.scaling + self.translation
#
# 	def normalize_vel(self, xd):
# 		return xd * self.scaling
#
# 	def denormalize_pos(self, x):
# 		return (x - self.translation)/self.scaling
#
# 	def denormalize_vel(self, xd):
# 		return xd / self.scaling

# ------------------------------------------------------------------------


