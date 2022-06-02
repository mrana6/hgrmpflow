import torch
from torch.nn import functional as F

# TODO: convert these to nn.Module
class Potential(object):
	def __call__(self, q):
		pass

	def grad(self, q):
		pass

	def to(self, device):
		pass

	def save(self, filename):
		pass

	def load(self, filename):
		pass


class ZeroPotential(Potential):
	def __init__(self, device=torch.device('cpu')):
		self.device = device

	def __call__(self, q):
		return torch.zeros(q.size(0), device=self.device)

	def grad(self, q):
		return torch.zeros_like(q, device=self.device)

	def to(self, device):
		self.device = device


class QuadraticPotential(Potential):
	def __init__(self, n_dims=None, hessian=None, device=torch.device('cpu')):
		assert n_dims is not None or hessian is not None

		if n_dims is not None:
			self.n_dims = n_dims
		else:
			self.n_dims = hessian.size()[0]

		if hessian is not None:
			assert hessian.size()[0] == self.n_dims
			self.hessian = hessian.to(device)
		else:
			self.hessian = torch.eye(self.n_dims, device=device)

		self.device = device

	def __call__(self, q):
		potential = 0.5 * torch.einsum('bi, ij, bj->b', q, self.hessian, q)
		return potential

	def grad(self, q):
		gradient = torch.einsum('ij, bj->bi', self.hessian, q)
		return gradient

	def to(self, device):
		self.device = device
		self.hessian.to(self.device)

	def set_parameters(self, hessian):
		assert hessian.size()[0] == self.n_dims
		self.hessian = hessian.to(self.device)


class LogCoshPotential(Potential):
	def __init__(self, scaling=100., p_gain=1.):
		self.scaling = scaling * 1.
		self.p_gain = p_gain

	def __call__(self, q):
		q_norm = torch.norm(q, dim=1)
		potential = self.p_gain / self.scaling * torch.log(torch.cosh(q_norm) * 2.)
		return potential

	def grad(self, q):
		n_dims = q.size()[1]
		q_norm = torch.norm(q, dim=1).repeat(n_dims, 1).t()
		s_alpha = torch.tanh(self.scaling * q_norm)
		q_hat = F.normalize(q)
		return self.p_gain * s_alpha * q_hat

	def set_parameters(self, scaling):
		self.scaling = scaling * 1.


class L2NormPotential(Potential):
	def __init__(self, scaling=1.):
		self.scaling = scaling

	def __call__(self, q):
		return torch.norm(q, dim=1)

	def grad(self, q):
		return F.normalize(q)

	def set_parameters(self, scaling):
		self.scaling = scaling * 1.


class BarrierPotential(Potential):
	'''
	Barrier Potential
	'''
	def __init__(self, proportional_gain=1e-5, slope_order=4, scale=1., device=torch.device('cpu')):
		self.proportional_gain = proportional_gain
		self.slope_order = slope_order
		self.scale = scale
		self.device = device

	def __call__(self, x):
		phi = 0.5 / self.scale * self.proportional_gain * self.barrier_scalar(self.scale * x) ** 2
		return phi

	def grad(self, x):
		del_phi = self.proportional_gain * self.barrier_scalar(x * self.scale) * self.del_barrier_scalar(x * self.scale)
		return del_phi

	def barrier_scalar(self, x):
		w = 1. / x ** self.slope_order
		w[x <= 0.] = 1e15
		return w

	def del_barrier_scalar(self, x):
		del_w = -1.0 * self.slope_order / x ** (self.slope_order + 1)
		del_w[x <= 0.] = 0.
		return del_w

	def to(self, device):
		self.device = device


class SmoothBarrierPotential(Potential):
	'''
	Smooth Barrier Potential
	'''

	def __init__(self, alpha=1.0, maximum=1e15):
		self.alpha = alpha
		self.maximum = maximum

	def __call__(self, x):
		phi = self.maximum * self.barrier_scalar(x)
		return phi

	def barrier_scalar(self, x):
		w = 1. / torch.cosh(self.alpha * x)
		return w
