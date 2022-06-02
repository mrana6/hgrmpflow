from .force_controllers import *
from .momentum_controllers import *
from .metrics import *
from .potentials import *


# -----------------------------------------------------------------------
class ObstacleAvoidanceForceController(NaturalGradientDescentForceController):
	def __init__(self, proportional_gain=1e-5, damping_gain=0.0, epsilon=1e-3, alpha_switch=None, ds_type='gds',
				 device=torch.device('cpu')):
		'''
		Obstacle avoidance controller
		:param proportional_gain:
		:param damping_gain:
		:param epsilon:
		:param alpha_switch: (defines slope of switch) set to None if hard switch desired
		:param ds_type:
		'''
		if alpha_switch is None:  #sets a hard velocity based toggle switch
			G = BarrierWithVelocityToggleMetric(epsilon=epsilon, device=device)
		else:  # soft switch with softness given by alpha_switch
			G = BarrierWithVelocityToggleSoftMetric(epsilon=epsilon, alpha=alpha_switch, device=device)

		Phi = BarrierPotential(proportional_gain=proportional_gain, device=device)
		del_Phi = Phi.grad
		B = lambda x, xd: damping_gain * G(x, xd)   # such that inv(G)*B = constant
		super(ObstacleAvoidanceForceController, self).__init__(G=G, B=B, del_Phi=del_Phi, ds_type=ds_type, device=device)


# ---------------------------------------------------------------------
class ObstacleAvoidanceMomentumController(NaturalGradientDescentMomentumController):
	def __init__(self, proportional_gain=1e-5, scale=1., device=torch.device('cpu')):
		G = BarrierMetric(slope_order=1, scale=scale)
		Phi = BarrierPotential(proportional_gain=proportional_gain, scale=scale, slope_order=1)
		del_Phi = Phi.grad
		super(ObstacleAvoidanceMomentumController, self).__init__(G=G, del_Phi=del_Phi, device=device)



