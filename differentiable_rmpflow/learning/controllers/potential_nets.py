import torch
import torch.nn as nn
from ..kinematics.taskmap_nets import ElementwiseAffineTaskMap


class ComposedPotentialNet(nn.Module):
	def __init__(self, n_inputs,
				 latent_taskmap,
				 latent_potential,
				 goal,
				 output_scaling_fn=None,
				 normalization_scaling=None,
				 normalization_bias=None,
				 device=torch.device('cpu')):

		super(ComposedPotentialNet, self).__init__()
		self.latent_taskmap = latent_taskmap
		self.latent_potential = latent_potential
		self.goal = goal
		self.device = device
		self.latent_taskmap.to(self.device)
		self.latent_potential.to(self.device)

		self.input_norm_taskmap = ElementwiseAffineTaskMap(n_inputs=n_inputs, bias=normalization_bias,
													  scaling=normalization_scaling, device=self.device)

		if output_scaling_fn is None:
			self.output_scaling_fn = lambda x: 1.
		else:
			self.output_scaling_fn = output_scaling_fn
			self.output_scaling_fn.to(self.device)

	def forward(self, x):
		x = self.input_norm_taskmap.psi(x)
		x = self.latent_taskmap.psi(x)

		goal = self.input_norm_taskmap.psi(self.goal)
		goal = self.latent_taskmap.psi(goal)

		x = x - goal

		phi = self.latent_potential(x)
		return phi

	def grad(self, x):
		goal = self.input_norm_taskmap.psi(self.goal)
		goal = self.latent_taskmap.psi(goal)

		y, J_y = self.input_norm_taskmap(x, order=1)
		z, J_z = self.latent_taskmap(y, order=1)
		z = z - goal

		J = torch.bmm(J_z, J_y)
		del_Phi = torch.bmm(self.latent_potential.grad(z).unsqueeze(1), J).squeeze(1)
		# del_Phi = 100. * self.output_scaling_fn(x)*del_Phi

		M = torch.bmm(J.permute(0, 2, 1), J)
		trace_M_inv = torch.einsum('bii->b', torch.inverse(M)).reshape(-1, 1)
		del_Phi = self.output_scaling_fn(x)*trace_M_inv*del_Phi

		# M = torch.bmm(J.permute(0, 2, 1), J)
		# M_inv = torch.inverse(M)
		# del_Phi = torch.bmm(del_Phi.unsqueeze(1), M_inv.permute(0, 2, 1)).squeeze(1)

		# J_pinv = torch.inverse(J)
		# del_Phi = torch.einsum('bij, bj -> bi', J_pinv, self.latent_potential.grad(z))
		return del_Phi

