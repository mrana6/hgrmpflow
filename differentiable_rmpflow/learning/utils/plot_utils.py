import torch
import numpy as np
import matplotlib.pyplot as plt
from differentiable_rmpflow.rmpflow.utils import *


# ----------------------------------------------
def visualize_tensor(x1_coords, x2_coords, z_tensor,
					 levels=20, cmap='viridis',
					 vmin=None, vmax=None, colorbar=False):
	'''
	plot the contour of each entry of z_tensor, see matplotlib.pyplot.contourf for more information
	:param x1_coords (np.ndarray): x1 coordinates, created by torch.meshgrid
	:param x2_coords (np.ndarray): x2_coordinates, created by torch.meshgrid
	:param z_tensor (np.ndarray): the tensor to be visualized
	:param levels (int): numbers of contour regions
	:param cmap (colormap): colormap for the contour
	:param vmin (float): maximal value used for normalization, default=None
	:param vmax (float): minimal value used for normalization, default=None
	:return: None
	'''

	# reshape the tensor so that it has the same shape as x1_coords and x2_coords
	n_rows, n_cols = x1_coords.shape
	z_tensor = z_tensor.reshape(n_rows, n_cols, -1)
	n_dims = z_tensor.shape[-1]

	# plot the contour for each element of z
	for d in range(n_dims):
		plt.figure()
		cf = plt.contourf(
			x1_coords, x2_coords, z_tensor[:, :, d],
			levels, cmap=cmap, vmin=vmin, vmax=vmax
		)
		if colorbar:
			plt.colorbar(cf)
		plt.show()

# ----------------------------------------------------------------------

def visualize_field(x1_coords, x2_coords, z_coords,
					cmap = 'viridis', type='stream', color=None):
	'''
	plot the streamplot or quiver plot for 2-dim vector fields
	see matplotlib.pyplot.streamplot or matplotlib.pyplot.quiver for more information
	:param x1_coords (np.ndarray): x1 coordinates, created by torch.meshgrid
	:param x2_coords (np.ndarray): x2_coordinates, created by torch.meshgrid
	:param z_tensor (np.ndarray): the vector field to be visualized
	:param cmap (colormap): colormap of the plot
	:param type ('stream or 'quiver'): type of plot, streamplot or quiverplot
	:return:
	'''

	# reshape the tensor so that it has the same shape as x1_coords and x2_coords
	n_rows, n_cols = x1_coords.shape
	z_coords = z_coords.reshape(n_rows, n_cols, -1)
	n_dims = z_coords.shape[-1]

	# make sure that z has at least 2 dims to visualize
	assert n_dims >= 2

	# visualize the first 2 dimensions of z
	if type == 'stream':
		# create streamplot
		if color is None:
			color = np.linalg.norm(z_coords, axis=2)
		if cmap is not None:
			plt.streamplot(
				x1_coords, x2_coords,
				z_coords[:, :, 0], z_coords[:, :, 1],
				color=color, cmap=cmap
			)
		else:
			if color is None:
				color = '0.4'
			plt.streamplot(
				x1_coords, x2_coords,
				z_coords[:, :, 0], z_coords[:, :, 1],
				color=color, cmap=cmap
			)
	elif type == 'quiver':
		# create quiverplot
		if color is None:
			color = np.linalg.norm(z_coords, axis=2)
		plt.quiver(
			x1_coords, x2_coords,
			z_coords[:, :, 0], z_coords[:, :, 1],
			color, units='width', cmap=cmap
		)
	# plt.show()
	# plt.ion()

# ----------------------------------------------------------------

def visualize_metric(model, x_lim=[[0, 1], [0, 1]], delta=0.05):
	'''
	visualize the metric of the model
	:param model (torch.nn.Module): the metric model to be visualized
	:param x_lim (array-like): the range of the state-space (positions) to be sampled over
	:param delta (float): the step size for sampled positions
	:return: None
	'''

	# generate a meshgrid of coordinates to test
	x2_coords, x1_coords = torch.meshgrid(
		[torch.arange(x_lim[1][0], x_lim[1][1], delta),
		 torch.arange(x_lim[0][0], x_lim[0][1], delta)])

	# create a flat version of the coordinates for forward pass
	x_test = torch.zeros(x1_coords.nelement(), 2)
	x_test[:, 0] = x1_coords.reshape(-1)
	x_test[:, 1] = x2_coords.reshape(-1)

	# evaluate the metric at every sample point
	g_pred = model(x_test)

	# for MetricCholNet, both the metric and its cholesky
	# decomposition is returned, take the metric only
	if isinstance(g_pred, tuple):
		g_pred = g_pred[0].detach() # detach the tensor from the graph
	else:
		g_pred = g_pred.detach()

	# visualize the metric tensor
	visualize_tensor(
		x1_coords.numpy(),
		x2_coords.numpy(),
		g_pred.numpy()
	)

# -----------------------------------------------------------------------------

def visualize_accel(model, x_lim=[[0, 1], [0, 1]], delta=0.05, cmap=None):
	'''
	visualize the acceleration model
	:param model (torch.nn.Module): the acceleration model to be visualized
	:param x_lim (array-like): the range of the state-space (positions) to be sampled over
	:param delta (float): the step size for the sampled positions
	:return: None
	'''

	# generate a meshgrid of coordinates to test
	# note that we do x2, x1 due to a special requirement of the streamplot function
	# see matplotlib.pyplot.streamplot for more information
	x2_coords, x1_coords = torch.meshgrid(
		[torch.arange(x_lim[1][0], x_lim[1][1], delta),
		 torch.arange(x_lim[0][0], x_lim[0][1], delta)])

	# generate test samples with zero velocity
	x_test = torch.zeros(x1_coords.nelement(), 4)
	x_test[:, 0] = x1_coords.reshape(-1)
	x_test[:, 1] = x2_coords.reshape(-1)

	# forward pass
	y_pred = model(x_test)

	# for force model, both the force and the metric are returned
	if isinstance(y_pred, tuple):
		f_pred, g_pred = y_pred
		# compute the acceleration: a = inv(G)*f
		y_pred = torch.einsum('bij,bj->bi', torch.inverse(g_pred), f_pred).detach()
	else:
		y_pred = y_pred.detach()

	# visualize each element of acceleration(contour plot)
	# visualize_tensor(
	#     x1_coords.numpy(),
	#     x2_coords.numpy(),
	#     y_pred.numpy()
	# )

	# visualize the acceleration as a vector field (equivalent to the warped potential)
	visualize_field(
		x1_coords.numpy(),
		x2_coords.numpy(),
		y_pred.numpy(),
		cmap=cmap
	)

# ------------------------------------------------------------------

def visualize_vel(model, x_lim=[[0, 1], [0, 1]], delta=0.05, cmap=None, color=None):
	'''
	visualize the velocity model (first order)
	similar to visualize_accel
	:param model (torch.nn.Module): the velocity model to be visualized
	:param x_lim (array-like): the range of the state-space (positions) to be sampled over
	:param delta (float): the step size for the sampled positions
	:return: None
	'''

	# generate a meshgrid of coordinates to test
	# note that we do x2, x1 due to a special requirement of the streamplot function
	# see matplotlib.pyplot.streamplot for more information
	x2_coords, x1_coords = torch.meshgrid(
		[torch.arange(x_lim[1][0], x_lim[1][1], delta),
		 torch.arange(x_lim[0][0], x_lim[0][1], delta)])

	# generate a flat version of the coordinates for forward pass
	x_test = torch.zeros(x1_coords.nelement(), 2)
	x_test[:, 0] = x1_coords.reshape(-1)
	x_test[:, 1] = x2_coords.reshape(-1)

	# forward pass
	y_pred = model(x_test).detach()

	# visualize each element of velocity (contour plot)
	# visualize_tensor(
	#     x1_coords.numpy(),
	#     x2_coords.numpy(),
	#     y_pred.numpy()
	# )

	# visualize the velocity as a vector field (equivalent to the warped potential)
	visualize_field(
		x1_coords.numpy(),
		x2_coords.numpy(),
		y_pred.numpy(),
		cmap=cmap,
		color=color
	)

# -----------------------------------------------------------------------------

def visualize_training_set(
		model, x_train,
		order=2, zero_vel_init=True,
		n_samples=40, t_final=2., t_step=None, color='r', integration_method='euler'):
	'''
	plot the training set and the roll-out trajectories of the
	learned model from initial positions from the training sets
	:param model (torch.nn.Module): the learned model to be evaluated
	:param x_train (torch.Tensor): the training trajectory
	:param order (1 or 2): whether the system is first order (order=1) or second order (order=2)
	:param zero_vel_init (bool): whether rolling out the trajectories from zero velocity
				(useful when testing second order systems with metrics learned by a first order system)
	:param n_samples (int): number of points to sample from
	:param t_final (float): time horizon of the roll-out trajectories
	:return:
	'''

	# plot the trajectories from the training set
	plt.plot(x_train[:, 0].numpy(), x_train[:, 1].numpy(), color)

	# evenly sample n_samples # of points from the training set as initial position
	step = len(x_train) // n_samples

	x_inits = x_train[step * np.arange(0, n_samples), :].reshape(n_samples, -1)

	if order == 2 and zero_vel_init:
		if x_inits.shape[-1] == 2:
			x_inits = torch.cat((x_inits, torch.zeros(n_samples, 2)), dim=1)
		elif x_inits.shape[-1] == 4:
			x_inits[:, 2:] = 0.
		else:
			raise TypeError('Plotting routine only works for 2D positions or position-velocity pairs')

	x_trajs = generate_trajectories(
		model,
		x_inits,
		order=order,
		return_label=False,
		t_final=t_final/n_samples,
		t_step=t_step,
		method=integration_method
	)

	for i in range(n_samples):
		x_traj = x_trajs[:, i, :].numpy()
		# plot the roll-out trajectory
		plt.plot(x_traj[:, 0], x_traj[:, 1], 'b')

		# plot the initial position
		plt.plot([x_traj[0,0]], [x_traj[0,1]], 'go')


def visualize_training_set_3d(
		model, x_train, ax,
		order=2, zero_vel_init=True,
		n_samples=40, t_final=2., t_step=None, color='r', integration_method='euler'):
	'''
	plot the training set and the roll-out trajectories of the
	learned model from initial positions from the training sets
	:param model (torch.nn.Module): the learned model to be evaluated
	:param x_train (torch.Tensor): the training trajectory
	:param order (1 or 2): whether the system is first order (order=1) or second order (order=2)
	:param zero_vel_init (bool): whether rolling out the trajectories from zero velocity
				(useful when testing second order systems with metrics learned by a first order system)
	:param n_samples (int): number of points to sample from
	:param t_final (float): time horizon of the roll-out trajectories
	:return:
	'''

	# plot the trajectories from the training set
	ax.plot(x_train[:, 0].numpy(), x_train[:, 1].numpy(), x_train[:, 2].numpy(), color)

	# evenly sample n_samples # of points from the training set as initial position
	step = len(x_train) // n_samples

	x_inits = x_train[step * np.arange(0, n_samples), :].reshape(n_samples, -1)

	if order == 2 and zero_vel_init:
		if x_inits.shape[-1] == 3:
			x_inits = torch.cat((x_inits, torch.zeros(n_samples, 3)), dim=1)
		elif x_inits.shape[-1] == 6:
			x_inits[:, 3:] = 0.
		else:
			raise TypeError('Plotting routine only works for 3D positions or position-velocity pairs')

	x_trajs = generate_trajectories(
		model,
		x_inits,
		order=order,
		return_label=False,
		t_final=t_final/n_samples,
		t_step=t_step,
		method=integration_method
	)

	for i in range(n_samples):
		x_traj = x_trajs[:, i, :].numpy()
		# plot the roll-out trajectory
		ax.plot(x_traj[:,0], x_traj[:, 1], x_traj[:, 2], 'b')

		# plot the initial position
		ax.plot([x_traj[0,0]], [x_traj[0,1]], [x_traj[0,2]], 'go')

