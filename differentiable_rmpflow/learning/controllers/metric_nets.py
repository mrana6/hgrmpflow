import torch
import torch.nn as nn
import torch.nn.functional as F


class MetricCholNet(nn.Module):
	def __init__(self, n_dims, n_hidden_1, n_hidden_2, activation_fn=F.relu, device=torch.device('cpu'), return_cholesky=True):
		super(MetricCholNet, self).__init__()
		self.n_dims = n_dims
		self.fc1 = nn.Linear(n_dims, n_hidden_1)
		self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
		n_diag = n_dims
		n_off_diag = n_dims * (n_dims - 1) // 2
		self.fc3_ld = nn.Linear(n_hidden_2, n_diag)
		# # intializing diagonal elements
		nn.init.zeros_(self.fc3_ld.weight.data)
		nn.init.zeros_(self.fc3_ld.bias.data)

		if n_dims > 1:
			self.fc3_lo = nn.Linear(n_hidden_2, n_off_diag)
			# # intializing off-diagonal elements
			nn.init.zeros_(self.fc3_lo.weight.data)
			nn.init.zeros_(self.fc3_lo.bias.data)

		self.activation_fn = activation_fn
		self.device = device
		self.return_cholesky = return_cholesky

	def reset_parameters(self):
		self.fc1.reset_parameters()
		self.fc2.reset_parameters()
		self.fc3_ld.reset_parameters()
		if self.fc3_lo:
			self.fc3_lo.reset_parameters()

	def forward(self, x, xd=None, offset=1e-5):
		n_samples = x.size()[0]
		x = self.activation_fn(self.fc1(x))
		x = self.activation_fn(self.fc2(x))
		# ld = F.relu(self.fc3_ld(x)) + offset
		# ld = torch.abs(self.fc3_ld(x)) + offset

		ld = torch.abs(self.fc3_ld(x) + 1) + offset
		# ld = self.fc3_ld(x) + 1
		# ld = self.output_activation(ld, torch.zeros_like(ld)) + offset

		if self.n_dims > 1:
			lo = self.fc3_lo(x)
			lm = torch.zeros(n_samples, self.n_dims, self.n_dims, device=self.device)
			diag_ind = torch.arange(self.n_dims)
			off_diag_ind = torch.tril_indices(self.n_dims, self.n_dims, -1)
			lm[:, diag_ind, diag_ind] = ld
			lm[:, off_diag_ind[0], off_diag_ind[1]] = lo
			gm = torch.matmul(lm, lm.transpose(1, 2))
		# x_norm = torch.norm(x, dim=1)
		else:
			lm = ld.reshape(-1, self.n_dims, self.n_dims)
			gm = lm * lm

		if self.return_cholesky:		# add a new flag to retun just the metric instead
			return gm, lm
		else:
			return gm


class ScalarMetric(nn.Module):
	def __init__(self, n_dims, n_hidden, eps=1e-12):
		super(ScalarMetric, self).__init__()
		self.scalar_network = nn.Sequential(
			nn.Linear(n_dims, n_hidden), nn.ReLU(),
			nn.Linear(n_hidden, n_hidden), nn.ReLU(),
			nn.Linear(n_hidden, 1), Exp(eps=torch.tensor(eps)))
		self.n_dims = n_dims

	def forward(self, input):
		scalar = self.scalar_network(input)
		scalar = torch.diag_embed(scalar.repeat(1, self.n_dims))
		return scalar


class MetricSquared(nn.Module):
	def __init__(self, metric_root_fn, outer_product=False):
		super(MetricSquared, self).__init__()
		self.metric_root_fn = metric_root_fn                # square root of the metric
		self.outer_product = outer_product

	def forward(self, q):
		G_root = self.metric_root_fn(q)
		G_root_T = G_root.permute(0, 2, 1)

		if self.outer_product:   # takes outerproduct
			G = torch.matmul(G_root,  G_root_T)
		else:                   # takes innerproduct
			G = torch.matmul(G_root_T, G_root)
		return G


class ProductMetric(nn.Module):
	def __init__(self, metric_fn_1, metric_fn_2):
		super(ProductMetric, self).__init__()
		self.metric_fn_1 = metric_fn_1
		self.metric_fn_2 = metric_fn_2

	def forward(self, q):
		return torch.matmul(self.metric_fn_1(q), self.metric_fn_2(q))


class Exp(nn.Module):
	def __init__(self, eps=torch.tensor(1e-12)):
		super(Exp, self).__init__()
		self.register_buffer('eps', eps)

	def forward(self, inputs):
		return torch.exp(inputs) + self.eps