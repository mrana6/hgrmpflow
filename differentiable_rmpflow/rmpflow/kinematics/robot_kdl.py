import numpy as np
import os

import PyKDL as kdl
from kdl_parser_py import urdf
from urdf_parser_py.urdf import URDF
from termcolor import colored
from collections import OrderedDict

import torch
from .taskmaps import TaskMap


class JointLimit(object):
	def __init__(self, lower=None, upper=None):
		self.lower_ = lower
		self.upper_ = upper

	@property
	def lower(self):
		return self.lower_

	@property
	def upper(self):
		return self.upper_


class Robot(object):
	# TODO: Add a function to add a new link for obstacle interpolation. Resort links when you do this!
	def __init__(self, in_joint_names=None, urdf_path=None, workspace_dim=3):
		self.workspace_dim = workspace_dim
		self.link_names = []
		self.joint_names = []
		self.joint_limits = []
		self.in_joint_names = in_joint_names  # Provide the active joints and IN ORDER !

		if self.in_joint_names is None:
			print(colored('WARNING: Joint name mapping not provided. Using kdl order!', 'green', 'on_red'))

		if urdf_path is None:
			self.robot_model = URDF.from_parameter_server(key='robot_description')
			success, self.kdl_tree = urdf.treeFromUrdfModel(self.robot_model)
			if not success:
				raise RuntimeError(
					"Could not create kinematic tree from /robot_description.")
		else:
			self.robot_model = URDF.from_xml_file(file_path=urdf_path)
			success, self.kdl_tree = urdf.treeFromUrdfModel(self.robot_model)
			if not success:
				raise RuntimeError(
					"Could not create kinematic tree from urdf file")

		self.base_link = self.robot_model.get_root()
		self.set_properties_from_model()
		self.sort_link_names()   #TODO: if you add a link to tree, make sure to sort link names!
		self.print_robot_description()

		self.task_maps = self.get_all_task_maps() #dict of all task maps


	@property
	def cspace_dim(self):
		return len(self.joint_names)

	@property
	def num_joints(self):
		return len(self.joint_names)

	@property
	def num_links(self):
		return len(self.link_names)

	def forward_kinematics(self, q):
		'''
		Gives a DxN of array of positions for all the links
		:param q:
		:return:
		'''
		fk = torch.zeros(self.workspace_dim, self.num_links, device=q.device)
		n=0
		for link_name, task_map in self.task_maps.items():
			fk[:, n] = task_map.psi(q).flatten()
			n+=1

		return fk

	def set_properties_from_model(self):
		self.link_names = []
		for key, value in self.robot_model.link_map.items():
			self.link_names.append(value.name)

		self.joint_names = []
		self.joint_limits = []
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
			num_segments = int(self.kdl_tree.getChain(self.base_link, self.link_names[i]).getNrOfSegments())
			num_segments_list.append(num_segments)

		# sorting by chain size
		sorted_idx = np.argsort(num_segments_list)
		self.link_names = [self.link_names[i] for i in sorted_idx]

	def print_robot_description(self):
		print("URDF non-fixed joints: %d" % len(self.joint_names))
		print("URDF total joints: %d" % len(self.robot_model.joints))
		print("URDF links: %d" % len(self.robot_model.links))
		print("KDL joints: %d" % self.kdl_tree.getNrOfJoints())
		print("KDL segments: %d" % self.kdl_tree.getNrOfSegments())
		print("Non-fixed joints: " + str(self.joint_names))
		print("Links: " + str(self.link_names))

	def get_all_task_maps(self, base_link=None):
		'''
		finds the kinematics as a dict for all the links with the root link as the base of the robot by default
		:return:
		'''
		task_maps = OrderedDict()
		if base_link is None:
			base_link = self.base_link

		for i in range(len(self.link_names)):
			target_link = self.link_names[i]
			task_maps[target_link] = self.get_task_map(target_link=target_link, base_link=base_link)
		return task_maps

	def get_task_map(self, target_link, base_link=None, verbose=False, device=torch.device('cpu')):
		'''
		Gives the task map to be used by RMPflow
		:param target_link:
		:param base_link:
		:param np_joint_names: list of joint names in order
		:param verbose:
		:return:
		'''
		if base_link is None:
			base_link = self.base_link

		kdl_chain = self.kdl_tree.getChain(base_link, target_link)
		kdl_fk = kdl.ChainFkSolverPos_recursive(kdl_chain)
		kdl_jacobian = kdl.ChainJntToJacSolver(kdl_chain)
		kdl_jacobian_dot = kdl.ChainJntToJacDotSolver(kdl_chain)
		D = int(kdl_chain.getNrOfJoints())

		if self.in_joint_names is None:
			np_to_kdl_idx_map = list(range(kdl_chain.getNrOfJoints()))
		else:
			np_to_kdl_idx_map = self.find_np_to_kdl_map(kdl_chain, self.in_joint_names)

		if verbose:
			self.print_kdl_chain(kdl_chain)

		def forward_kinematcs(q):
			assert q.shape[-1] == self.num_joints, "Input dimensions mismatch!"
			q_ = q[np_to_kdl_idx_map]
			task_frame = kdl.Frame()
			kdl_fk.JntToCart(self.torch_to_kdl(q_), task_frame)
			p = task_frame.p
			p_array = torch.tensor([p.x(), p.y(), p.z()], device=device)
			return p_array[0:self.workspace_dim]

		def jacobian(q):
			assert q.shape[-1] == self.num_joints, "Input dimension mismatch!"
			q_ = q[np_to_kdl_idx_map]
			jacobian_ = kdl.Jacobian(D)
			kdl_jacobian.JntToJac(self.torch_to_kdl(q_), jacobian_)
			jacobian = torch.zeros(self.workspace_dim, self.num_joints, device=device)
			jacobian[:, np_to_kdl_idx_map] = \
				self.kdl_to_torch(jacobian_, device=device)[0:self.workspace_dim, :]  # converting to np and extracting only the position part!
			return jacobian

		def jacobian_dot(q, q_dot):
			assert q.shape[-1] == self.num_joints, "Input dimensions mismatch!"
			q_ = q[np_to_kdl_idx_map]
			qd_ = q_dot[np_to_kdl_idx_map]

			jacobian_dot_ = kdl.Jacobian(D)
			kdl_jacobian_dot.JntToJacDot(self.torch_to_kdl_vel(q_, qd_), jacobian_dot_)
			jacobian_dot = torch.zeros(self.workspace_dim, self.num_joints, device=device)
			jacobian_dot[:, np_to_kdl_idx_map] = \
				self.kdl_to_torch(jacobian_dot_, device=device)[0:self.workspace_dim, :]
			return jacobian_dot

		def batch_forward_kinematics(q):
			if q.ndim == 1:
				q = q.reshape(1, -1)

			N = q.shape[0]
			fk = torch.zeros(N, self.workspace_dim, device=device)
			for n in range(N):
				fk[n, :] = forward_kinematcs(q[n,:])
			return fk

		def batch_jacobian(q):
			if q.ndim == 1:
				q = q.reshape(1, -1)

			N = q.shape[0]
			J = torch.zeros(N, self.workspace_dim, self.cspace_dim, device=device)
			for n in range(N):
				J[n] = jacobian(q[n, :])
			return J

		def batch_jacobian_dot(q, q_dot):
			if q.ndim == 1:
					q = q.reshape(1, -1)
					if q_dot is not None and q_dot.ndim == 1:
						q_dot = q_dot.reshape(1, -1)
			N = q.shape[0]
			Jd = torch.zeros(N, self.workspace_dim, self.cspace_dim, device=device)
			for n in range(N):
				Jd[n] = jacobian_dot(q[n,:], q_dot[n,:])
			return Jd

		def batch_forward(x, xd=None, order=2):
			if x.ndim == 1:
				x = x.reshape(1, -1)
				if xd is not None and xd.ndim == 1:
					xd = xd.reshape(1, -1)

			N = x.shape[0]
			y = torch.zeros(N, self.workspace_dim, device=device)
			J = torch.zeros(N, self.workspace_dim, self.cspace_dim, device=device)

			if order == 1:
				for n in range(N):
					y[n, :] = forward_kinematcs(x[n, :])
					J[n] = jacobian(x[n, :])
				return y, J

			Jd = torch.zeros(N, self.workspace_dim, self.cspace_dim, device=device)
			for n in range(N):
				y[n, :] = forward_kinematcs(x[n, :])
				J[n] = jacobian(x[n, :])
				Jd[n] = jacobian_dot(x[n,:], xd[n,:])

			yd = torch.einsum('bij,bj->bi', J, xd)
			return y, yd, J, Jd

		link_task_map = TaskMap(n_inputs=self.cspace_dim, n_outputs=self.workspace_dim,
								psi=batch_forward_kinematics, J=batch_jacobian, J_dot=batch_jacobian_dot, device=device)
		link_task_map.forward = batch_forward
		return link_task_map

	def find_kdl_to_np_map(self, kdl_chain, in_joint_names):
		num_segments = kdl_chain.getNrOfSegments()
		chain_joint_names = []
		for n in range(num_segments):
			kdl_joint_name = kdl_chain.getSegment(n).getJoint().getName()
			if kdl_joint_name in self.joint_names:  # kdl_joint_name can be fixed too. This is to check we get only non-fixed.
				chain_joint_names.append(kdl_joint_name)

		joint_map = []
		for n in range(len(in_joint_names)):
			in_joint_name = in_joint_names[n]
			if in_joint_name in chain_joint_names:
				idx = chain_joint_names.index(in_joint_name)
				joint_map.append(idx)

		return joint_map

	def find_np_to_kdl_map(self, kdl_chain, np_joint_names):
		joint_map = []
		num_segments = kdl_chain.getNrOfSegments()
		for n in range(num_segments):
			kdl_joint_name = kdl_chain.getSegment(n).getJoint().getName()
			if kdl_joint_name in np_joint_names:
				idx = np_joint_names.index(kdl_joint_name)
				joint_map.append(idx)

		return joint_map

	def torch_to_kdl(self, q):
		q_kdl = kdl.JntArray(q.shape[-1])
		for i, q_i in enumerate(q):
			q_kdl[i] = q_i.item()
		return q_kdl

	def torch_to_kdl_vel(self, q, q_dot):
		q_kdl = kdl.JntArray(q.shape[-1])
		q_dot_kdl = kdl.JntArray(q_dot.shape[-1])

		for (i, q_i), (_, qd_i) in zip(enumerate(q), enumerate(q_dot)):
			q_kdl[i] = q_i.item()
			q_dot_kdl[i] = qd_i.item()
		return kdl.JntArrayVel(q_kdl, q_dot_kdl)

	def kdl_to_torch(self, data, device=torch.device('cpu')):
		array_torch = torch.zeros(data.rows(), data.columns(), device=device)
		for i in range(data.rows()):
			for j in range(data.columns()):
				array_torch[i, j] = data[i, j]
		return array_torch

	def print_kdl_chain(self, chain):
		for idx in range(chain.getNrOfSegments()):
			print('* ' + chain.getSegment(idx).getName())


if __name__ == '__main__':
	filename = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../urdf/point_planar_robot.urdf'))
	np_joint_names = ['joint1', 'joint2', 'joint3']
	robot = Robot(urdf_path=filename, workspace_dim=2)#, in_joint_names=np_joint_names)
	task_map = robot.get_task_map(target_link='link3', verbose=True)

	a = task_map(torch.zeros(1,2), torch.zeros(1,2), order=2)


