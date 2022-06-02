import torch
from urdf_parser_py.urdf import URDF, Pose
from .taskmaps import TaskMap
from collections import OrderedDict
import numpy as np


class JointLimit(object):
	def __init__(self, lower=None, upper=None):
		self.lower = lower
		self.upper = upper


class Robot(object):
	def __init__(self, urdf_path=None, workspace_dim=3):
		self.workspace_dim = workspace_dim
		self.link_names = []
		self.joint_names = []
		self.joint_limits = []

		if urdf_path is None:
			try:
				self.robot_model = URDF.from_parameter_server(key='robot_description')
			except:
				raise RuntimeError("Could not create kinematic tree from /robot_description.")
		else:
			try:
				self.robot_model = URDF.from_xml_file(file_path=urdf_path)
			except:
				raise RuntimeError(
					"Could not create kinematic tree from urdf file")

		self.base_link_ = self.robot_model.get_root()
		self.set_properties_from_model()
		self.sort_link_names()   #TODO: if you add a link to tree, make sure to sort link names!
		self.print_robot_description()

		self.task_maps = self.get_all_task_maps() #dict of all task maps

	@property
	def num_joints(self):
		return len(self.joint_names)

	@property
	def num_links(self):
		return len(self.link_names)

	@property
	def cspace_dim(self):
		return len(self.joint_names)

	def forward_kinematics(self, q):
		'''
		Gives a DxN of array of positions for all the links
		:param q:
		:return:
		'''
		fk = torch.zeros(self.workspace_dim, self.num_links, device=q.device)
		n = 0
		for link_name, task_map in self.task_maps.items():
			fk[:, n] = task_map.psi(q).flatten()
			n += 1

		return fk

	def set_properties_from_model(self):
		for key, value in self.robot_model.link_map.items():
			self.link_names.append(value.name)

		for i in range(len(self.robot_model.joints)):
			if self.robot_model.joints[i].joint_type != 'fixed':
				self.joint_names.append(self.robot_model.joints[i].name)
				if self.robot_model.joints[i].limit is not None:
					lower = self.robot_model.joints[i].limit.lower
					upper = self.robot_model.joints[i].limit.upper
				else:
					lower = None
					upper = None
				self.joint_limits.append(JointLimit(lower=lower, upper=upper))

	def sort_link_names(self):
		# sorting by name
		sorted_idx = np.argsort(self.link_names)
		self.link_names = [self.link_names[i] for i in sorted_idx]

		num_segments_list = []
		for i in range(len(self.link_names)):
			chain = self.robot_model.get_chain(self.base_link_, self.link_names[i])
			num_segments = len(chain)
			num_segments_list.append(num_segments)

		# sorting by chain size
		sorted_idx = np.argsort(num_segments_list)
		self.link_names = [self.link_names[i] for i in sorted_idx]

	def print_robot_description(self):
		print("URDF non-fixed joints: %d" % len(self.joint_names))
		print("URDF total joints: %d" % len(self.robot_model.joints))
		print("URDF links: %d" % len(self.robot_model.links))
		print("Non-fixed joints: " + str(self.joint_names))
		print("Links: " + str(self.link_names))

	def get_all_task_maps(self, base_link=None):
		'''
		finds the kinematics as a dict for all the links with the root link as the base of the robot by default
		:return:
		'''
		task_maps = OrderedDict()
		if base_link is None:
			base_link = self.base_link_

		for i in range(len(self.link_names)):
			target_link = self.link_names[i]
			task_maps[target_link] = self.get_task_map(target_link=target_link, base_link=base_link)
		return task_maps

	def get_task_map(self, target_link, base_link=None, device=torch.device('cpu')):
		'''
		Gives the task map to be used by RMPflow
		:param target_link:
		:param base_link:
		:param np_joint_names: list of joint names in order
		:return:
		'''
		if base_link is None:
			base_link = self.base_link_

		chain = self.robot_model.get_chain(base_link, target_link)
		nvar = 0
		joint_list = []
		actuated_types = ["prismatic", "revolute", "continuous"]
		actuated_names = []

		# The chain sometimes have repeated links (no idea why!). Removing repetitions!
		unique_chain = []
		for elem in chain:
			if elem not in unique_chain:
				unique_chain.append(elem)

		chain = unique_chain
		for item in chain:
			if item in self.robot_model.joint_map:
				joint = self.robot_model.joint_map[item]
				joint_list += [joint]
				if joint.type in actuated_types:
					nvar += 1
					actuated_names += [joint.name]
					if joint.axis is None:
						joint.axis = [1., 0., 0.]
					if joint.origin is None:
						joint.origin = Pose(xyz=[0., 0., 0.],
											rpy=[0., 0., 0.])
					elif joint.origin.xyz is None:
						joint.origin.xyz = [0., 0., 0.]
					elif joint.origin.rpy is None:
						joint.origin.rpy = [0., 0., 0.]

		def fk(q):
			if q.ndimension() == 1:
				q = q.reshape(1,-1)
			batch_size = q.shape[0]
			T_fk = torch.eye(4, 4, device=device).repeat(batch_size, 1, 1)
			i=0
			for joint in joint_list:
				if joint.type == "fixed":
					xyz = torch.tensor(joint.origin.xyz, device=device)
					rpy = torch.tensor(joint.origin.rpy, device=device)
					joint_frame = T_rpy(xyz, rpy, device=device)
					T_fk = torch.matmul(T_fk, joint_frame)
				elif joint.type == "prismatic":
					if joint.axis is None:
						axis = torch.tensor([1., 0., 0.], device=device)
					axis = torch.tensor(joint.axis, device=device)
					joint_frame = T_prismatic(torch.tensor(joint.origin.xyz, device=device),
											  torch.tensor(joint.origin.rpy, device=device),
											  torch.tensor(joint.axis, device=device), q[:, i],
											  batch_size=batch_size)
					T_fk = torch.bmm(T_fk, joint_frame)
					i += 1
				elif joint.type in ["revolute", "continuous"]:
					if joint.axis is None:
						axis = torch.tensor([1., 0., 0.], device=device)
					axis = torch.tensor(joint.axis, device=device)
					axis = (1. / torch.norm(axis)) * axis
					joint_frame = T_revolute(torch.tensor(joint.origin.xyz, device=device),
											 torch.tensor(joint.origin.rpy, device=device),
											 torch.tensor(joint.axis, device=device), q[:, i],
											 batch_size=batch_size)
					T_fk = torch.bmm(T_fk, joint_frame)
					i += 1
			return T_fk[:, 0:self.workspace_dim, -1]
		return TaskMap(n_inputs=self.num_joints, n_outputs=self.workspace_dim, psi=fk, device=device)


# helper functions
# ----------------------------------------------------------
def rotation_rpy(rpy, device=torch.device('cpu')):
	"""Returns a rotation matrix from roll pitch yaw. ZYX convention."""
	cr = torch.cos(rpy[0])
	sr = torch.sin(rpy[0])
	cp = torch.cos(rpy[1])
	sp = torch.sin(rpy[1])
	cy = torch.cos(rpy[2])
	sy = torch.sin(rpy[2])
	return torch.tensor([[cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
						 [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
						 [  -sp,             cp*sr,             cp*cr]], device=device)


def T_rpy(displacement, rpy, device=torch.device('cpu')):
	"""Homogeneous transformation matrix with roll pitch yaw."""
	T = torch.zeros(4, 4, device=device)
	T[:3, :3] = rotation_rpy(rpy)
	T[:3, 3] = displacement
	T[3, 3] = 1.0
	return T


def T_prismatic(xyz, rpy, axis, qi, batch_size=1, device=torch.device('cpu')):
	T = torch.zeros(batch_size, 4, 4, device=device)

	# Origin rotation from RPY ZYX convention
	cr = torch.cos(rpy[0])
	sr = torch.sin(rpy[0])
	cp = torch.cos(rpy[1])
	sp = torch.sin(rpy[1])
	cy = torch.cos(rpy[2])
	sy = torch.sin(rpy[2])
	r00 = cy*cp
	r01 = cy*sp*sr - sy*cr
	r02 = cy*sp*cr + sy*sr
	r10 = sy*cp
	r11 = sy*sp*sr + cy*cr
	r12 = sy*sp*cr - cy*sr
	r20 = -sp
	r21 = cp*sr
	r22 = cp*cr
	p0 = r00*axis[0]*qi + r01*axis[1]*qi + r02*axis[2]*qi
	p1 = r10*axis[0]*qi + r11*axis[1]*qi + r12*axis[2]*qi
	p2 = r20*axis[0]*qi + r21*axis[1]*qi + r22*axis[2]*qi

	# Homogeneous transformation matrix
	T[:, 0, 0] = r00
	T[:, 0, 1] = r01
	T[:, 0, 2] = r02
	T[:, 1, 0] = r10
	T[:, 1, 1] = r11
	T[:, 1, 2] = r12
	T[:, 2, 0] = r20
	T[:, 2, 1] = r21
	T[:, 2, 2] = r22
	T[:, 0, 3] = xyz[0] + p0
	T[:, 1, 3] = xyz[1] + p1
	T[:, 2, 3] = xyz[2] + p2
	T[:, 3, 3] = 1.0
	return T


def T_revolute(xyz, rpy, axis, qi, batch_size=1, device=torch.device('cpu')):
	T = torch.zeros(batch_size, 4, 4, device=device)

	# Origin rotation from RPY ZYX convention
	cr = torch.cos(rpy[0])
	sr = torch.sin(rpy[0])
	cp = torch.cos(rpy[1])
	sp = torch.sin(rpy[1])
	cy = torch.cos(rpy[2])
	sy = torch.sin(rpy[2])
	r00 = cy*cp
	r01 = cy*sp*sr - sy*cr
	r02 = cy*sp*cr + sy*sr
	r10 = sy*cp
	r11 = sy*sp*sr + cy*cr
	r12 = sy*sp*cr - cy*sr
	r20 = -sp
	r21 = cp*sr
	r22 = cp*cr

	# joint rotation from skew sym axis angle
	cqi = torch.cos(qi)
	sqi = torch.sin(qi)
	s00 = (1 - cqi)*axis[0]*axis[0] + cqi
	s11 = (1 - cqi)*axis[1]*axis[1] + cqi
	s22 = (1 - cqi)*axis[2]*axis[2] + cqi
	s01 = (1 - cqi)*axis[0]*axis[1] - axis[2]*sqi
	s10 = (1 - cqi)*axis[0]*axis[1] + axis[2]*sqi
	s12 = (1 - cqi)*axis[1]*axis[2] - axis[0]*sqi
	s21 = (1 - cqi)*axis[1]*axis[2] + axis[0]*sqi
	s20 = (1 - cqi)*axis[0]*axis[2] - axis[1]*sqi
	s02 = (1 - cqi)*axis[0]*axis[2] + axis[1]*sqi

	# Homogeneous transformation matrix
	T[:, 0, 0] = r00*s00 + r01*s10 + r02*s20
	T[:, 1, 0] = r10*s00 + r11*s10 + r12*s20
	T[:, 2, 0] = r20*s00 + r21*s10 + r22*s20

	T[:, 0, 1] = r00*s01 + r01*s11 + r02*s21
	T[:, 1, 1] = r10*s01 + r11*s11 + r12*s21
	T[:, 2, 1] = r20*s01 + r21*s11 + r22*s21

	T[:, 0, 2] = r00*s02 + r01*s12 + r02*s22
	T[:, 1, 2] = r10*s02 + r11*s12 + r12*s22
	T[:, 2, 2] = r20*s02 + r21*s12 + r22*s22

	T[:, 0, 3] = xyz[0]
	T[:, 1, 3] = xyz[1]
	T[:, 2, 3] = xyz[2]
	T[:, 3, 3] = 1.0
	return T

# --------------------------------------------------------

if __name__ == '__main__':
	r = Robot(urdf_path='../../../urdf/lula_franka_gen_fixed_gripper.urdf')
	taskmap = r.get_task_map(target_link='panda_leftfingertip', base_link='base_link')
	x = torch.zeros(1, 7)
	xd = torch.ones(1, 7)
	y, yd, J, Jd = taskmap(x, xd)