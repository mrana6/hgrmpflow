import torch

# TODO: convert these to nn.Module
class Metric(object):
	def __call__(self, q, qd):
		pass

	def load_state_dict(self, filename):
		pass


class IdentityMetric(Metric):
	def __init__(self, scaling=1., device=torch.device('cpu')):
		self.scaling = scaling
		self.device = device

	def __call__(self, q, qd=None):
		n, d = q.size()
		metric = torch.eye(d, device=self.device).repeat(n, 1, 1) * self.scaling
		return metric

	def to(self, device):
		self.device = device


class StretchMetric(Metric):
	def __init__(self, rotation_matrix, sigma=0.5, scale=0.5, device=torch.device('cpu')):
		self.n_dims = rotation_matrix.size()[0]
		self.sigma = sigma
		self.scale = scale
		self.device = device
		self.rotation_matrix = rotation_matrix.to(device=self.device)

	def __call__(self, q, qd=None):
		assert q.size()[1] == self.n_dims

		n_samples = q.size()[0]

		q_norm = q.norm(dim=1)
		# s = torch.tanh(q_norm / sigma) * scale
		s = q_norm * self.scale + 0.01

		S = torch.eye(self.n_dims, device=self.device).repeat(n_samples, 1, 1)
		S[:, 0, 0] = s

		metric = torch.einsum('ij,bjk,lk->bil', self.rotation_matrix, S, self.rotation_matrix)
		return metric

	def to(self, device):
		self.device = device

	def set_parameters(self, rotation_matrix=None, scale=None, sigma=None):
		if rotation_matrix is not None:
			self.rotation_matrix = rotation_matrix.to(self.device)
		if scale is not None:
			self.scale = scale
		if sigma is not None:
			self.sigma = sigma


class OuterProductMetric(Metric):
	def __init__(self, n_dims, inner_product_gain=0.5, bias=0.1, device=torch.device('cpu')):
		self.n_dims = n_dims
		self.inner_product_gain = inner_product_gain
		self.bias = bias
		self.device = device

	def __call__(self, q, qd=None):
		if q.dim() == 1:
			q = q.unsqueeze(0)
		elif q.dim() == 3:
			q = q.squeeze(2)

		assert q.dim() == 2
		assert self.n_dims == q.size()[1]

		metric = self.bias * \
				 torch.eye(self.n_dims, device=self.device) + \
				 self.inner_product_gain * torch.einsum('bi,bj->bij', q, q)
		return metric

	def to(self, device):
		self.device = device

	def set_parameters(self, inner_product_gain=None, bias=None):
		if inner_product_gain is not None:
			self.inner_product_gain = inner_product_gain
		if bias is not None:
			self.bias = bias


class ScaledUniformMetric(Metric):
	'''
	Metric that is scaled with a position-varying weight function
	'''
	def __init__(self, w_u=10.0, w_l=1.0, sigma=1.0, device=torch.device('cpu')):
		self.w_u_ = w_u
		self.w_l_ = w_l
		self.sigma_ = sigma
		self.device = device

	def __call__(self, x, xd=None):
		n, d = x.shape
		beta = torch.exp(-torch.norm(x, dim=1).reshape(-1, 1) ** 2 / (2.0 * self.sigma_ ** 2))
		w = (self.w_u_ - self.w_l_) * beta + self.w_l_
		G = torch.eye(d, device=self.device).repeat(n, 1, 1) * w.repeat(1, d**2).reshape(n, d, d)
		return G

	def to(self, device):
		self.device = device


class DirectionallyStretchedMetric(Metric):
	'''
	Metric that stretches in the direction of the goal
	'''

	def __init__(self, w_u=10.0, w_l=1.0, sigma_gamma=1., sigma_alpha=1.0,
				 alpha_softmax=1.0, eps=1e-12, dist_offset=0.4, dist_alpha=10.,
				 isotropic_mass=0.0075, device=torch.device('cpu')):
		self.w_u = w_u
		self.w_l = w_l
		self.sigma_gamma = sigma_gamma
		self.sigma_alpha = sigma_alpha
		self.alpha_softmax = alpha_softmax
		self.eps = eps
		self.dist_offset = dist_offset
		self.dist_alpha = dist_alpha
		self.isotropic_mass = isotropic_mass
		self.device = device

	def __call__(self, x, xd=None):
		n, d = x.shape
		x_norm = torch.norm(x, dim=1).reshape(-1, 1)
		s = torch.tanh(self.alpha_softmax * x_norm)
		x_hat = x / (x_norm + self.eps)
		# n = s * x_hat
		# S = np.dot(n, n.T) # NOTE: not really sure what this is adding
		S = torch.einsum('bi, bj->bij', x_hat, x_hat)

		# NOTE: old formulation
		# gamma = 1./cosh_(x_norm * self.sigma_gamma) ** 2
		# alpha = 1./cosh_(x_norm * self.sigma_alpha) ** 2

		# min_weight sets some isotropic mass which provides inertia against movement in the nullspace
		# of S, the anisotropic component
		min_weight = self.isotropic_mass
		max_weight = 1. - min_weight
		gamma = max_weight * 1. / torch.cosh(x_norm * self.sigma_gamma) ** 2 + min_weight
		alpha = max_weight * 1. / torch.cosh(x_norm * self.sigma_alpha) ** 2 + min_weight

		w = (self.w_u - self.w_l) * gamma + self.w_l
		I = torch.eye(d, device=self.device).repeat(n, 1, 1)
		w_diag = I * w.repeat(1, d**2).reshape(n, d, d)
		alpha_diag = I * alpha.repeat(1, d ** 2).reshape(n, d, d)
		G = w_diag * ((1. - alpha_diag) * S + alpha_diag + self.eps*I)

		return G


class BarrierWithVelocityToggleMetric(Metric):
	'''
	Barrier metric that turns off when velocity is negative (on 1D task space)
	'''
	def __init__(self, epsilon=1e-3, slope_order=4,  device=torch.device('cpu')):
		self.epsilon = epsilon
		self.slope_order = slope_order
		self.device = device

	def __call__(self, x, xd):
		w = self.barrier_scalar(x)
		u = self.velocity_scalar(xd, epsilon=self.epsilon)
		G = w * u
		return G.unsqueeze(2)

	def barrier_scalar(self, x):
		w = 1. / x ** self.slope_order
		w[x <= 0.] = 1e15
		return w

	def velocity_scalar(self, xd, epsilon):
		u = epsilon + torch.min(xd, torch.tensor(0., device=self.device))*xd
		return u

	def to(self, device):
		self.device = device


class BarrierWithVelocityToggleSoftMetric(Metric):
	'''
	Barrier metric that turns off when velocity is negative (on 1D task space)
	'''
	def __init__(self, epsilon=1e-3, slope_order=4, alpha=10., device=torch.device('cpu')):
		self.epsilon = epsilon
		self.slope_order = slope_order
		self.alpha = alpha
		self.device = device

	def __call__(self, x, xd):
		w = self.barrier_scalar(x)
		u = self.velocity_scalar(xd, alpha=self.alpha)
		G = w * u + self.epsilon
		return G.unsqueeze(2)

	def barrier_scalar(self, x):
		w = 1. / x ** self.slope_order
		w[x <= 0.] = 1e15
		return w

	def velocity_scalar(self, xd, alpha):
		# NOTE: tanh implementation of sigmoid function is more stable
		u = 0.5 * (torch.tanh(-0.5 * alpha * xd) + 1.)
		return u

	def to(self, device):
		self.device = device


class BarrierMetric(Metric):
	'''
	Barrier Metric on 1D task space
	'''
	def __init__(self, epsilon=0.2, slope_order=1, scale=1., device=torch.device('cpu')):
		self.device = device
		self.epsilon = epsilon
		self.slope_order = slope_order
		self.scale = scale

	def __call__(self, x, xd=None):
		return (self.barrier_scalar(self.scale * x) + self.epsilon).unsqueeze(2)

	def barrier_scalar(self, x):
		w = 1. / x ** self.slope_order
		w[x <= 0.] = 1e15
		return w

	def to(self, device):
		self.device = device