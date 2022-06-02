from .taskmaps import TaskMap
import torch

def get_distance_taskmap(type='sphere', **kwargs):
	if type == 'sphere':
		return SphereDistanceTaskMap(**kwargs)
	elif type == 'point':
		return PointDistanceTaskMap(**kwargs)
	elif type == 'box':
		return BoxDistanceTaskMap(**kwargs)
	elif type == 'cylinder':
		return CylinderDistanceTaskMap(**kwargs)
	else:
		# todo: add more cases
		raise ValueError


class SphereDistanceTaskMap(TaskMap):
	'''
	Task map for spherical type obstacles
	'''
	def __init__(self, n_inputs, center=None, radius=1.0, device=torch.device('cpu')):
		self.device = device
		if center is None:
			self.center = torch.zeros(1, n_inputs, device=self.device)
		else:
			self.center = center.to(device=self.device, dtype=torch.get_default_dtype())
		self.radius = radius
		self.n_inputs = n_inputs

		psi = lambda x: torch.norm(x-self.center, dim=1).reshape(-1, 1)/self.radius - 1.
		J = lambda x: (1.0 / torch.norm(x-self.center, dim=1).reshape(-1, 1) * (x - self.center)/ self.radius).unsqueeze(1)

		def J_dot(x, xd):
			n = x.shape[0]
			x_norm = torch.norm(x - self.center).reshape(-1, 1)
			Jd = -1.0 / x_norm ** 3 * torch.einsum('bi,bj->bij', x - self.center, x - self.center) \
				 + 1.0 / x_norm * torch.eye(self.n_inputs, device = self.device).repeat(n,1,1)
			Jd = torch.einsum('bi, bij->bj', xd, Jd).unsqueeze(1)
			return Jd

		super(SphereDistanceTaskMap, self).__init__(n_inputs=self.n_inputs, n_outputs=1, psi=psi, J=J, J_dot=J_dot,
													device=self.device)


class PointDistanceTaskMap(TaskMap):
	'''
	Task map for point type obstacles
	'''
	def __init__(self, n_inputs, center=None, device=torch.device('cpu')):
		self.device = device
		if center is None:
			self.center = torch.zeros(1, n_inputs, device=self.device)
		else:
			self.center = center.to(device=self.device, dtype=torch.get_default_dtype())
		psi = lambda x: 0.5 * torch.norm(x - self.center, dim=1).reshape(-1, 1)**2
		J = lambda x: (x - self.center).unsqueeze(1)
		J_dot = lambda x, xd: xd.unsqueeze(1)

		super(PointDistanceTaskMap, self).__init__(n_inputs=self.n_inputs, n_outputs=1, psi=psi, J=J, J_dot=J_dot,
													device=self.device)


class BoxDistanceTaskMap(TaskMap):
	def __init__(self, n_inputs, box_center=None, box_size=None, scale=1., device=torch.device('cpu')):
		self.device = device
		self.scale = scale
		if box_center is None:
			self.box_center = torch.zeros(1, n_inputs, device=self.device)
		else:
			self.box_center = box_center.to(device=self.device, dtype=torch.get_default_dtype())

		if box_size is None:
			self.box_size = 2. * torch.ones(1, 2, device=self.device)
		else:
			self.box_size = box_size.to(device=self.device, dtype=torch.get_default_dtype())

		def psi(y):
			d = torch.abs(y - self.box_center) - self.box_size / 2.
			sdf = torch.norm(torch.max(d, torch.tensor(0., device=self.device)), dim=1)\
				  + torch.min(torch.max(d, dim=1)[0], torch.tensor(0., device=self.device))
			return sdf.reshape(-1, 1) * self.scale

		super(BoxDistanceTaskMap, self).__init__(n_inputs=self.n_inputs, n_outputs=1, psi=psi, device=self.device)


class CylinderDistanceTaskMap(TaskMap):
	'''
	Cylinder task map essentially
	'''
	def __init__(self, x1, x2, radius=0., device=torch.device('cpu')):
		self.x1 = x1.to(device=self.device, dtype=torch.get_default_dtype())
		self.x2 = x2.to(device=self.device, dtype=torch.get_default_dtype())
		self.radius = radius
		self.device = device
		D = x1.shape[1]

		def psi(x):
			return PointLineSegmentDistance(self.x1, self.x2, x) - self.radius

		# def J(x):
		# 	_, alpha = PointLineSegmentDistance(self.x1, self.x2, x, get_alpha_ret=True)
		# 	return torch.nn.functional.normalize(x - ((1. - alpha) * self.x1 + alpha * self.x2)).unsqueeze(2)

		super(CylinderDistanceTaskMap, self).__init__(n_inputs=D, n_outputs=1, psi=psi, device=self.device)


# -----------------------------------------------
# Helper functions

def PointLineSegmentDistance(x1, x2, y, get_alpha_ret=False):
	'''
	Calculates and returns the shortest distance between a point y and the  line segment defined by (x1, x2).
	Optionally returns the alpha blending between the two as a return parameter defined as:
	x* = (1 - alpha) x1 + alpha x2 = x1 + alpha (x2 - x1)
	'''
	dist, alpha_ = PointLineDistance(x1, x2 - x1, y, get_alpha=True)

	if alpha_ < 0.:
		alpha = 0.
		dist = torch.norm(y - x1, dim=1).reshape(-1, 1)
	elif alpha_ > 1.:
		alpha = 1.
		dist = torch.norm(y - x2, dim=1).reshape(-1, 1)
	else:
		alpha = alpha_

	if get_alpha_ret:
		alpha_ret = alpha
		return dist, alpha_ret
	return dist


def PointLineDistance(x, v_x, y, get_alpha = False):
	# Calculates and returns the shortest distance between a point y and
	# the line defined by x + alpha v_x.
	v_x_norm = torch.norm(v_x, dim=1).reshape(-1, 1)

	v_x_hat = v_x / v_x_norm
	v = y - x
	v_dot_vx = torch.einsum('bi,bi -> b', v, v_x_hat).reshape(-1, 1)
	v_x_parallel = v_dot_vx * v_x_hat
	dist = torch.norm(v - v_x_parallel, dim=1).reshape(-1, 1)

	if get_alpha:
		alpha = v_dot_vx / v_x_norm
		return dist, alpha
	return dist




