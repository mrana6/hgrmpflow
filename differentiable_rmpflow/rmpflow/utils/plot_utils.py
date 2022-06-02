import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from differentiable_rmpflow.rmpflow.utils.torchdiffeq import odeint
import torch


def generate_trajectories(
		model, x_init, time_dependent=False, order=2,
		return_label=False, filename=None,
		t_init=0., t_final=10., t_step=0.01, method='rk4'):
	'''
	generate roll-out trajectories of the given dynamical model
	:param model (torch.nn.Module): dynamical model
	:param x_init (torch.Tensor): initial condition
	:param order (int): 1 if first order system, 2 if second order system
	:param return_label (bool): set True to return the velocity (order=1) / acceleration (order=2) along the trajectory
	:param filename (string): if not None, save the generated trajectories to the file
	:param t_init (float): initial time for numerical integration
	:param t_final (float): final time for numerical integration
	:param t_step (float): step for t_eval for numerical integration
	:param method (string): method of integration (ivp or euler)
	:return: state (torch.Tensor) if return_label is False
	:return: state (torch.Tensor), control (torch.Tensor) if return label is True
	'''

	# make sure that the initial condition has dim 1
	if x_init.ndim == 1:
		x_init = x_init.reshape(1, -1)

	# dynamics for numerical integration
	if order == 2:
		# dynamics for the second order system
		def dynamics(t, state):
			# for second order system, the state includes
			# both position and velocity
			n_dims = state.shape[-1] // 2
			x = state[:, :n_dims]
			x_dot = state[:, n_dims:]

			# compute the acceleration under the given model
			if time_dependent:
				y_pred = model(t=t, x=x, xd=x_dot)
			else:
				y_pred = model(x, x_dot)

			# for force model, both the force and the metric are returned
			if isinstance(y_pred, tuple):
				f_pred, g_pred = y_pred

				# compute the acceleration: a = inv(G)*f
				x_ddot = torch.einsum(
					'bij,bj->bi',
					torch.inverse(g_pred),
					f_pred).detach() # detach the tensor from the computational graph
			else:
				# otherwise, only the acceleration is returned
				x_ddot = y_pred.detach()

			# the time-derivative of the state includes
			# both velocity and acceleration
			state_dot = torch.cat(
				(x_dot, x_ddot), dim=1)
			return state_dot

	elif order == 1:
		# dynamics for the first order system
		def dynamics(t, state):
			# for first order systems, the state is the position

			# compute the velocity under the given model
			if time_dependent:
				y_pred = model(t=t, x=state)
			else:
				y_pred = model(state)
			if isinstance(y_pred, tuple):
				p_pred, g_pred = y_pred
				x_dot = torch.einsum(
					'bij,bj->bi',
					torch.inverse(g_pred),
					p_pred).detach()
			else:
				x_dot = y_pred.detach()
			return x_dot
	else:
		raise TypeError('Unknown order!')

	# the times at which the trajectory is computed
	t_eval = torch.arange(t_init, t_final, t_step)

	x_data = odeint(dynamics, x_init, t_eval, method=method)

	# if the control inputs along the trajectory are also needed,
	# compute the control inputs (useful for generating datasets)
	if return_label:
		y_data = dynamics(t_eval, x_data)
		data = (x_data, y_data)
	else:
		data = x_data

	if filename is not None:
		torch.save(data, filename)

	return data

# -----------------------------------------

def plot_traj_2D(traj, ls, color, order=2):
	plt.plot(traj[:, 0], traj[:, 1], linestyle=ls, linewidth=2, color=color)
	plt.plot(traj[0, 0], traj[0, 1], 'ko')
	plt.plot(traj[-1, 0], traj[-1, 1], 'x', color=color)
	if order ==2:
		plt.quiver(traj[0, 0], traj[0, 1], traj[0, 2] + 1e-20, traj[0, 3] + 1e-20, color='k',scale_units='xy', scale=1.)


# --------------------------------------

def plot_trajectories_3D(traj_list, ls, color, ax_handle=None, zorder=100):
	if ax_handle is None:
		ax = plt.gca()
	else:
		ax = ax_handle

	for i in range(len(traj_list)):
		traj = traj_list[i]
		ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], linestyle=ls, linewidth=2, color=color, zorder=zorder+i)
		ax.scatter3D(traj[0, 0], traj[0, 1], traj[0, 2], color = 'green', marker='o')
		ax.scatter3D(traj[-1, 0], traj[-1, 1], traj[-1, 2], color = 'red', marker='x')
		# plt.quiver(traj[0, 0], traj[1, 0], traj[2, 0] + 1e-20, traj[3, 0] + 1e-20, color


# ---------------------------------------

def plot_traj_3D(traj, ls, color, ax_handle=None):
	if ax_handle is None:
		ax = plt.gca()
	else:
		ax = ax_handle
	ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], linestyle=ls, linewidth=2, color=color)
	ax.scatter3D(traj[0, 0], traj[0, 1], traj[0, 2], color='green', marker='o')
	ax.scatter3D(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='red', marker='x')
	# plt.quiver(traj[0, 0], traj[1, 0], traj[2, 0] + 1e-20, traj[3, 0] + 1e-20, color='k')

# ----------------------------------------

def plot_traj_time(time, traj, ls='--', color='k', axs_handles=None, lw=2):
	shape = traj.shape
	if shape[1] > shape[0]:
		traj = traj.T

	num_dim = traj.shape[1]
	if axs_handles is None:
		axs_handles = [None] * num_dim
	else:
		assert (len(axs_handles) == num_dim)

	for i in range(num_dim):
		if axs_handles[i] is None:
			ax = plt.subplot(num_dim, 1, i + 1)
		else:
			ax = axs_handles[i]
		ax.plot(time, traj[:, i], linestyle=ls, linewidth=lw, color=color)
		axs_handles[i] = ax

	return axs_handles

# ---------------------------------------

def plot_robot_2D(robot, q, lw=2, handle_list=None, link_order=None):
	link_pos = robot.forward_kinematics(q)
	num_links = link_pos.shape[1]

	if link_order is None:
		link_order = [(i, j) for i, j in zip(range(0, num_links), range(1, num_links))]

	if handle_list is None:
		handle_list = []

		for n in range(len(link_order)):
			pt1 = link_pos[:, link_order[n][0]]
			pt2 = link_pos[:, link_order[n][1]]

			h1 = plt.plot(pt1[0], pt1[1], marker='o', color='black', linewidth=3 * lw)
			handle_list.append(h1[0])
			h2 = plt.plot(pt2[0], pt2[1], marker='o', color='black', linewidth=3 * lw)
			handle_list.append(h2[0])
			h3 = plt.plot(np.array([pt1[0], pt2[0]]), np.array([pt1[1], pt2[1]]), color='blue', linewidth=lw)
			handle_list.append(h3[0])
	else:
		m = 0
		for n in range(len(link_order)):
			pt1 = link_pos[:, link_order[n][0]]
			pt2 = link_pos[:, link_order[n][1]]

			handle_list[m].set_data(pt1[0], pt1[1])
			m += 1
			handle_list[m].set_data(pt2[0], pt2[1])
			m += 1
			handle_list[m].set_data(np.array([pt1[0], pt2[0]]), np.array([pt1[1], pt2[1]]))
			m += 1
	return handle_list,

# ------------------------------

def plot_robot_3D(robot, q, lw, handle_list=None, link_order=None):
	link_pos = robot.forward_kinematics(q)

	if handle_list is None:
		fig = plt.gcf()
		ax = plt.gca()
		handle_list = []

		if link_order is not None:
			for n in range(len(link_order)):
				pt1 = link_pos[:, link_order[n][0]]
				pt2 = link_pos[:, link_order[n][1]]
				h3 = ax.plot3D(np.array([pt1[0], pt2[0]]), np.array([pt1[1], pt2[1]]), np.array([pt1[2], pt2[2]]),
							   color='blue', linewidth=lw,  marker='o', markerfacecolor='black', markersize=3*lw)
				handle_list.append(h3[0])
		else:
			for n in range(link_pos.shape[1] - 1):
				pt1 = link_pos[:, n]
				pt2 = link_pos[:, n + 1]
				h1 = ax.scatter3D(pt1[0], pt1[1], pt1[2], color='black', linewidths=3 * lw)
				handle_list.append(h1)
				h2 = ax.scatter3D(pt2[0], pt2[1], pt2[2], color='black', linewidths=3 * lw)
				handle_list.append(h2)
				h3 = ax.plot3D(np.array([pt1[0], pt2[0]]), np.array([pt1[1], pt2[1]]), np.array([pt1[2], pt2[2]]),
							   color='blue', linewidth=lw,  marker='o', markerfacecolor='black', markersize=3*lw)
				handle_list.append(h3)

	else:
		if link_order is not None:
			m = 0
			for n in range(len(link_order)):
				pt1 = link_pos[:, link_order[n][0]]
				pt2 = link_pos[:, link_order[n][1]]

				# handle_list[m].set_data(pt1[0], pt1[1], pt1[2])
				# m += 1
				# handle_list[m].set_data(pt2[0], pt2[1], pt2[2])
				# m += 1
				handle_list[m].set_data(np.array([pt1[0], pt2[0]]), np.array([pt1[1], pt2[1]]))
				handle_list[m].set_3d_properties(np.array([pt1[2], pt2[2]]))
				m += 1
		else:
			m = 0
			for n in range(link_pos.shape[1] - 1):
				pt1 = link_pos[:, n]
				pt2 = link_pos[:, n + 1]

				handle_list[m].set_data(pt1[0], pt1[1], pt1[2])
				m += 1
				handle_list[m].set_data(pt2[0], pt2[1], pt2[2])
				m += 1
				handle_list[m].set_data(np.array([pt1[0], pt2[0]]), np.array([pt1[1], pt2[1]]),
										np.array([pt1[2], pt2[2]]))
				m += 1

	return handle_list,

# ---------------------------------------

def plot_tf_from_mat(transformation_mat, axis_scale=0.02, ax_handle=None):
	if ax_handle is None:
		ax = plt.gca()
	else:
		ax = ax_handle
	axis_colors = ['r', 'g', 'b']
	for d in range(3):
		e_d_master = axis_scale * np.eye(3)[:, d].reshape(-1, 1)
		e_d_obj = transform_pt(e_d_master, transformation_mat)
		origin_obj = transformation_mat[:3, -1].reshape(-1, 1)
		line_points = np.concatenate((origin_obj, e_d_obj), axis=1)
		ax.plot3D(line_points[0, :], line_points[1, :], line_points[2, :], linewidth=4, color=axis_colors[d])


# -------------------------------
def transform_pt(pt, transform_mat):
	pt = np.concatenate((pt, np.ones((1,1))))
	pt_transformed = np.dot(transform_mat, pt)[0:3].reshape(-1,1)

	return pt_transformed

# --------------------------------
def set_axes_radius(ax, origin, radius):
	ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
	ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
	ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
	'''Make axes of 3D plot have equal scale so that spheres appear as spheres,
	cubes as cubes, etc..  This is one possible solution to Matplotlib's
	ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

	Input
	  ax: a matplotlib axis, e.g., as output from plt.gca().
	'''

	limits = np.array([
		ax.get_xlim3d(),
		ax.get_ylim3d(),
		ax.get_zlim3d(),
	])

	origin = np.mean(limits, axis=1)
	radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
	set_axes_radius(ax, origin, radius)


# ------------------------------------------
# cube plotting routines

def cuboid_data2(o, size=(1,1,1)):
	X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
		 [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
		 [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
		 [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
		 [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
		 [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
	X = np.array(X).astype(float)
	for i in range(3):
		X[:,:,i] *= size[i]
	X += np.array(o)
	return X

def plotCubeAt2(positions,sizes=None,colors=None, **kwargs):
	if not isinstance(colors,(list,np.ndarray)): colors=["C0"]*len(positions)
	if not isinstance(sizes,(list,np.ndarray)): sizes=[(1,1,1)]*len(positions)
	g = []
	for p,s,c in zip(positions,sizes,colors):
		g.append( cuboid_data2(p, size=s) )
	return Poly3DCollection(np.concatenate(g),
							facecolors=np.repeat(colors,6), **kwargs)


#-------------------------------------------------------

def plot_sphere(c, r, axis = None, alpha=1.0):
	if axis == None:
		axis = plt.gca()
	N = 50
	stride = 2
	u = np.linspace(0, 2 * np.pi, N)
	v = np.linspace(0, np.pi, N)
	x = c[0] + r*np.outer(np.cos(u), np.sin(v))
	y = c[1] + r*np.outer(np.sin(u), np.sin(v))
	z = c[2] + r*np.outer(np.ones(np.size(u)), np.cos(v))
	axis.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride, alpha=alpha)

# ----------------------------------------------------
from scipy.linalg import norm
def plot_cylinder(p0, p1, R, ax = None):
	if ax == None:
		ax = plt.gca()

	v = p1 - p0
	# find magnitude of vector
	mag = norm(v)
	# unit vector in direction of axis
	v = v / mag
	# make some vector not in the same direction as v
	not_v = np.array([1, 0, 0])
	if (v == not_v).all():
		not_v = np.array([0, 1, 0])
	# make vector perpendicular to v
	n1 = np.cross(v, not_v)
	# normalize n1
	n1 /= norm(n1)
	# make unit vector perpendicular to v and n1
	n2 = np.cross(v, n1)
	# surface ranges over t from 0 to length of axis and 0 to 2*pi
	t = np.linspace(0, mag, 100)
	theta = np.linspace(0, 2 * np.pi, 100)
	# use meshgrid to make 2d arrays
	t, theta = np.meshgrid(t, theta)
	# generate coordinates for surface
	X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
	ax.plot_surface(X, Y, Z)
	# plot axis
	ax.plot(*zip(p0, p1), color='red')
	# ax.set_xlim(0, 10)
	# ax.set_ylim(0, 10)
	# ax.set_zlim(0, 10)
	# plt.show()



