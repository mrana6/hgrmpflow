import torch
import torch.nn as nn



def get_metric_chol_net_loss(loss_fn=nn.MSELoss()):

	def metric_chol_net_loss(g_pred, g_target):
		if isinstance(g_pred, tuple):
			g_pred = g_pred[0]
		if isinstance(g_target, tuple):
			g_target = g_target[0]
		return loss_fn(g_pred, g_target)

	return metric_chol_net_loss


def get_lagrangian_accel_net_loss(loss_fn=nn.MSELoss()):

	def lagrangian_accel_net_loss(y_pred, y_target):
		if isinstance(y_pred, tuple):
			y_pred = y_pred[0]
		if isinstance(y_target, tuple):
			y_target = y_target[0]
		return loss_fn(y_pred, y_target)
	return lagrangian_accel_net_loss


def get_lagrangian_force_net_loss(loss_fn=nn.MSELoss()):

	def lagrangian_force_net_loss(y_pred, y_target):
		f_pred, g_pred = y_pred
		f_target = torch.einsum('bij,bj->bi', g_pred, y_target)
		return loss_fn(f_pred, f_target)

	return lagrangian_force_net_loss


def get_lagrangian_force_net_loss2(loss_fn=nn.MSELoss()):

	def lagrangian_force_net_loss(y_pred, y_target):
		f_pred = y_pred[0]
		m_pred = y_pred[1]
		m_inv_pred = torch.inverse(m_pred)
		y_pred = torch.einsum('bij,bj->bi', m_inv_pred, f_pred)
		return loss_fn(y_pred, y_target)

	return lagrangian_force_net_loss


def get_lagrangian_force_net_loss3():

	def lagrangian_force_net_loss(y_pred, y_target):
		f_pred = y_pred[0]
		m_pred = y_pred[1]
		m_inv_pred = torch.inverse(m_pred)
		y_pred = torch.einsum('bij,bj->bi', m_inv_pred, f_pred)
		diff = y_pred - y_target
		loss = torch.mean(torch.einsum('bi, bj, bij->b', diff, diff, m_pred))
		return loss

	return lagrangian_force_net_loss


def get_lagrangian_force_net_loss4(criterion=nn.MSELoss()):

	def lagrangian_force_net_loss(y_pred, y_target):
		f_pred, M_pred, J_subtasks = y_pred
		M_inv_pred = torch.inverse(M_pred)    #TODO: This was regular inverse before!
		y_pred = torch.einsum('bij, bj -> bi', M_inv_pred, f_pred)
		diff_cspace = y_pred - y_target

		n_subtasks = len(J_subtasks)
		loss = 0.
		for n in range(n_subtasks):
			diff_subtask = torch.einsum('bij, bj -> bi', J_subtasks[n], diff_cspace)
			loss += criterion(diff_subtask, torch.zeros_like(diff_subtask))
		loss = loss/n_subtasks
		return loss
		#
		# n_subtasks = len(J_subtasks)
		# losses = []
		# for n in range(n_subtasks):
		# 	diff_subtask = torch.einsum('bij, bj -> bi', J_subtasks[n], diff_cspace)
		# 	loss_n = criterion(diff_subtask, torch.zeros_like(diff_subtask))
		# 	losses.append(loss_n.reshape(1,-1))
		# 	# subtask_losses = torch.cat((subtask_losses, loss))
		# subtask_losses = torch.cat(losses)
		# loss = torch.mean(subtask_losses)
		# return loss

	return lagrangian_force_net_loss