from torch.utils.data import TensorDataset, Dataset
import torch


class FirstOrderDynamicsDataset(Dataset):
	def __init__(self, x, xd):
		self.x = x
		self.xd = xd

	def __getitem__(self, index):
		x_batch = self.x[index]
		xd_batch = self.xd[index]
		return x_batch, xd_batch

	def __len__(self):
		return self.x.size(0)


class SecondOrderDynamicsDataset(Dataset):
	def __init__(self, x, xd, xdd):
		self.x = x
		self.xd = xd
		self.xdd = xdd

	def __getitem__(self, index):
		x_batch = self.x[index]
		xd_batch = self.xd[index]
		state_batch = {'x': x_batch, 'xd': xd_batch}
		xdd_batch = self.xdd[index]
		return state_batch, xdd_batch

	def __len__(self):
		return self.x.size(0)


class PullbackDataset(Dataset):
	def __init__(self, q, qd, qdd, psi_list=None, J_list=None, Jd_list=None, dphi_list=None):
		assert q.size(0) == qd.size(0)
		assert q.size(0) == qdd.size(0)
		if psi_list is not None:
			assert all(q.size(0) == psi.size(0) for psi in psi_list)
		if J_list is not None:
			assert all(q.size(0) == J.size(0) for J in J_list)
		if Jd_list is not None:
			assert all(q.size(0) == Jd.size(0) for Jd in Jd_list)
			assert len(psi_list) == len(J_list) and len(J_list) == len(Jd_list)
		if dphi_list is not None:
			assert all(q.size(0) == dphi.size(0) for dphi in dphi_list)
			assert len(psi_list) == len(dphi_list)

		self.x = torch.cat((q, qd), dim=1)
		self.qdd = qdd
		self.psi_list = psi_list
		self.J_list = J_list
		self.Jd_list = Jd_list
		self.dphi_list = dphi_list

	def __getitem__(self, index):
		x_batch = {
			'x': self.x[index],
			'psi_list': [psi[index] for psi in self.psi_list] if self.psi_list is not None else None,
			'J_list': [J[index] for J in self.J_list] if self.J_list is not None else None,
			'Jd_list': [Jd[index] for Jd in self.Jd_list] if self.Jd_list is not None else None,
			'dphi_list': [dphi[index] for dphi in self.dphi_list] if self.dphi_list is not None else None
		}
		y_batch = self.qdd[index]
		return x_batch, y_batch

	def __len__(self):
		return self.x.size(0)


class ContextDataset(Dataset):
	def __init__(self, q_config, qd_config, qdd_config,
				 q_leaf_list, qd_leaf_list, J_list, Jd_list, force_list, metric_list):

		self.state = torch.cat((q_config, qd_config), dim=1)
		self.qdd_config = qdd_config
		self.q_leaf_list = q_leaf_list
		self.qd_leaf_list = qd_leaf_list
		self.J_list = J_list
		self.Jd_list = Jd_list
		self.force_list = force_list
		self.metric_list = metric_list

	def __getitem__(self, index):
		x_batch = {
			'state': self.state[index],
			'q_leaf_list': [q_leaf[index] for q_leaf in self.q_leaf_list],
			'qd_leaf_list': [qd_leaf[index] for qd_leaf in self.qd_leaf_list],
			'J_list': [J[index] for J in self.J_list],
			'Jd_list': [Jd[index] for Jd in self.Jd_list],
			'force_list': [force[index] for force in self.force_list],
			'metric_list': [metric[index] for metric in self.metric_list]
		}
		y_batch = self.qdd_config[index]
		return x_batch, y_batch

	def __len__(self):
		return self.state.size(0)


class ContextDatasetMomentum(Dataset):
	def __init__(self, q_config, qd_config, q_leaf_list, J_list, momentum_list, metric_list):

		self.state = q_config
		self.qd_config = qd_config
		self.q_leaf_list = q_leaf_list
		self.J_list = J_list
		self.momentum_list = momentum_list
		self.metric_list = metric_list

	def __getitem__(self, index):
		x_batch = {
			'state': self.state[index],
			'q_leaf_list': [q_leaf[index] for q_leaf in self.q_leaf_list],
			'J_list': [J[index] for J in self.J_list],
			'momentum_list': [momentum[index] for momentum in self.momentum_list],
			'metric_list': [metric[index] for metric in self.metric_list]
		}
		y_batch = self.qd_config[index]
		return x_batch, y_batch

	def __len__(self):
		return self.qd_config.size(0)

