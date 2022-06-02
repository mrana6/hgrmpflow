import torch
from torch import nn
import abc

# TODO: rmp order should be an input to the initializer
class Rmp(nn.Module):
	def __init__(self, return_natural=True):
		super(Rmp, self).__init__()
		# By default return natural form, otherwise acceleration/velocity
		self.return_natural = return_natural

	def forward(self, x, xd=None, t=None):
		if x.ndim == 1:
			x = x.reshape(1, -1)
			if xd is not None and xd.ndim == 1:
				xd = xd.reshape(1, -1)
		f, M = self.eval_natural(x=x, xd=xd, t=t)

		if not self.training:
			f = f.detach()
			M = M.detach()

		if self.return_natural:
			return f, M
		xdd = torch.einsum('bij,bj->bi', torch.pinverse(M), f)
		return xdd

	@abc.abstractmethod
	def eval_natural(self, x, xd, t=None):
		pass

	def eval_canonical(self, x, xd, t=None):
		f, M = self(x=x, xd=xd, t=t)
		xdd = torch.einsum('bij,bj->bi', torch.pinverse(M), f)
		return xdd, M


class RmpTreeNode(Rmp):
	'''
	Node in the RMP tree
	TODO: Add extra_repr for printing out the tree branching out of it
	'''
	def __init__(self, n_dim, name="", order=2, return_natural=True, device=torch.device('cpu')):
		super(RmpTreeNode, self).__init__(return_natural=return_natural)
		self.n_dim = n_dim     # dimension of the node task space
		self.name = name
		self.edges = torch.nn.ModuleList()   	# list of edges connecting other nodes
		self.rmps = torch.nn.ModuleList()		# list of leaf rmps
		self.order = order # order of rmp node (1: momentum-based, 2: force-based)
		self.device = device

	def add_rmp(self, rmp):
		self.rmps.append(rmp)

	def add_task_space(self, task_map, name=""):
		assert(self.n_dim == task_map.n_inputs), ValueError('Dimension mismatch!')
		child_node = RmpTreeNode(n_dim=task_map.n_outputs, name=name, order=self.order, device=self.device)
		edge = RmpTreeEdge(task_map=task_map, child_node=child_node, order=self.order, device=self.device)
		self.edges.append(edge)
		return child_node

	def eval_natural(self, x, xd=None, t=None):
		assert(x.shape[-1] == self.n_dim), ValueError('Dimension mismatch!')

		n_pts = x.shape[0]
		f = torch.zeros(n_pts, self.n_dim, device=self.device)
		M = torch.zeros(n_pts, self.n_dim, self.n_dim, device=self.device)

		for i in range((len(self.edges))):
			f_i, M_i = self.edges[i](x=x, xd=xd, t=t)
			f += f_i
			M += M_i

		for i in range((len(self.rmps))):
			f_i, M_i = self.rmps[i](x=x, xd=xd, t=t)
			f += f_i
			M += M_i

		return f, M

	@property
	def n_edges(self):
		return len(self.edges)

	@property
	def n_rmps(self):
		return len(self.rmps)


class RmpTreeEdge(Rmp):
	def __init__(self, task_map, child_node, order=2, return_natural=True, device=torch.device('cpu')):
		super(RmpTreeEdge, self).__init__(return_natural=return_natural)
		self.task_map = task_map		# mapping from parent to child node
		self.child_node = child_node    # child node
		self.order = order
		self.device = device

		assert self.order in [1, 2], TypeError('Invalid RMP order!')

	def eval_natural(self, x, xd=None, t=None):
		if self.order == 2:
			# pushforward
			y, yd, J, Jd = self.task_map(x=x, xd=xd, order=self.order)
			f_y, M_y = self.child_node(x=y, xd=yd, t=t)

			# pullback
			M = torch.einsum('bji, bjk, bkl->bil', J, M_y, J)
			f = torch.einsum('bji, bj->bi', J, f_y) - torch.einsum('bji, bjk, bkl, bl->bi', J, M_y, Jd, xd)
			return f, M

		elif self.order == 1:
			# pushforward
			y, J = self.task_map(x=x, order=self.order)
			p_y, M_y = self.child_node(x=y, t=t)

			# pullback
			M = torch.einsum('bji, bjk, bkl->bil', J, M_y, J)
			p = torch.einsum('bji, bj->bi', J, p_y)
			return p, M



