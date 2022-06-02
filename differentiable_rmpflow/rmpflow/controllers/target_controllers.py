from .force_controllers import *
from .momentum_controllers import *
from .metrics import *
from .potentials import *


class TargetForceControllerUniform(NaturalGradientDescentForceController):
	'''
	Target controller with uniform metric and soft-norm type potential
	'''
	def __init__(self, damping_gain=2.0, proportional_gain=1.0, norm_robustness_eps=1e-12,
				 alpha=1.0, w_u=10.0, w_l=1.0, sigma=1.0, ds_type='gds', device=torch.device('cpu')):
		self.damping_gain_ = damping_gain
		self.proportional_gain_ = proportional_gain
		self.eps_ = norm_robustness_eps
		self.alpha_ = alpha
		self.w_u_ = w_u
		self.w_l_ = w_l
		self.sigma_ = sigma
		self.device = device

		G = ScaledUniformMetric(w_u=self.w_u_, w_l=self.w_l_ , sigma=self.sigma_, device=device)   # Note: G doesnt depend on xd
		Phi_tilde = LogCoshPotential(p_gain=self.proportional_gain_, scaling=self.alpha_)
		del_Phi_tilde = Phi_tilde.grad
		del_Phi = lambda x: torch.einsum('bij, bj-> bi', G(x), del_Phi_tilde(x))
		B = lambda x, xd: G(x) * self.damping_gain_
		super(TargetForceControllerUniform, self).__init__(G=G, B=B, del_Phi=del_Phi, ds_type=ds_type, device=device)


# ------------------------------------------------
class TargetForceControllerStretched(NaturalGradientDescentForceController):
	def __init__(self, damping_gain=2.0, proportional_gain=1.0, w_u=10.0, w_l=1.0, sigma_gamma=20., sigma_alpha=0.1,
				 alpha_softmax =1.0, eps=1e-12, ds_type='gds', device=torch.device('cpu')):
		self.damping_gain = damping_gain
		self.proportional_gain = proportional_gain
		self.w_u = w_u
		self.w_l = w_l
		self.sigma_gamma = sigma_gamma
		self.sigma_alpha = sigma_alpha
		self.alpha_softmax = alpha_softmax
		self.eps = eps

		G = DirectionallyStretchedMetric(w_u=self.w_u, w_l=self.w_l, sigma_gamma=self.sigma_gamma,
										 sigma_alpha=self.sigma_alpha, alpha_softmax =self.alpha_softmax, eps=self.eps,
										 device=device)

		Phi_tilde = LogCoshPotential(p_gain=self.proportional_gain, scaling=self.alpha_softmax)
		del_Phi_tilde = Phi_tilde.grad
		del_Phi = lambda x: torch.einsum('bij, bj-> bi', G(x), del_Phi_tilde(x))
		B = lambda x, xd: G(x) * self.damping_gain
		super(TargetForceControllerStretched, self).__init__(G=G, B=B, del_Phi=del_Phi, ds_type=ds_type, device=device)


# -------------------------------------------------
class TargetMomentumControllerUniform(NaturalGradientDescentMomentumController):
	'''
	Target controller with uniform metric and soft-norm type potential
	'''
	def __init__(self, proportional_gain=1.0, norm_robustness_eps=1e-12,
				 alpha=1.0, w_u=10.0, w_l=1.0, sigma=1.0, device=torch.device('cpu')):
		self.proportional_gain_ = proportional_gain
		self.eps_ = norm_robustness_eps
		self.alpha_ = alpha
		self.w_u_ = w_u
		self.w_l_ = w_l
		self.sigma_ = sigma

		G = ScaledUniformMetric(w_u=self.w_u_, w_l=self.w_l_ , sigma=self.sigma_, device=device)   # Note: G doesnt depend on xd
		Phi_tilde = LogCoshPotential(p_gain=self.proportional_gain_, scaling=self.alpha_)
		del_Phi_tilde = Phi_tilde.grad
		del_Phi = lambda x: torch.einsum('bij, bj-> bi', G(x), del_Phi_tilde(x))
		super(TargetMomentumControllerUniform, self).__init__(G=G, del_Phi=del_Phi, device=device)


# ------------------------------------------------
class TargetMomentumControllerStretched(NaturalGradientDescentMomentumController):
	def __init__(self, damping_gain=2.0, proportional_gain=1.0, w_u=10.0, w_l=1.0, sigma_gamma=20., sigma_alpha=0.1,
				 alpha_softmax =1.0, eps=1e-12, ds_type='gds', device=torch.device('cpu')):
		self.damping_gain = damping_gain
		self.proportional_gain = proportional_gain
		self.w_u = w_u
		self.w_l = w_l
		self.sigma_gamma = sigma_gamma
		self.sigma_alpha = sigma_alpha
		self.alpha_softmax = alpha_softmax
		self.eps = eps

		G = DirectionallyStretchedMetric(w_u=self.w_u, w_l=self.w_l, sigma_gamma=self.sigma_gamma,
										 sigma_alpha=self.sigma_alpha, alpha_softmax =self.alpha_softmax, eps=self.eps,
										 device=device)
		Phi_tilde = LogCoshPotential(p_gain=self.proportional_gain, scaling=self.alpha_softmax)
		del_Phi_tilde = Phi_tilde.grad
		del_Phi = lambda x: torch.einsum('bij, bj-> bi', G(x), del_Phi_tilde(x))
		super(TargetMomentumControllerStretched, self).__init__(G=G, del_Phi=del_Phi, device=device)


# -------------------------------------------
#
# class TrajectoryTrackingTimeDependentController(nn.Module):
# 	def __init__(self, x_traj, xd_traj, dt, kp=4., kd=1.):
# 		self.x_traj = x_traj   # list of positions dt spaced in time
# 		self.xd_traj = xd_traj # list of velocities dt spaced in time
# 		self.kp = kp
# 		self.kd = kd
# 		self.dt = dt
# 		super(TrajectoryTrackingTimeDependentController, self).__init__()
#
# 	def forward(self, t, x):
# 		# assuming robot starts from zero velocity!
# 		idx = np.round(t / self.dt).astype(int)
# 		xdd = -self.kp*(x-self.x_traj[idx]) -self.kd*(0.0 - self.xd_traj[idx])
# 		return xdd
#
#
# class TimeWrapperNet(nn.Module):
# 	def __init__(self, net):
# 		super(TimeWrapperNet, self).__init__()
# 		self.net = net
#
# 	def forward(self, t, x):
# 		if isinstance(x, torch.Tensor):
# 			return self.net(x)
# 		elif isinstance(x, dict):
# 			return self.net(**x)
# 		else:
# 			raise ValueError
