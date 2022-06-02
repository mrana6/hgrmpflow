from differentiable_rmpflow.learning.utils import *
from differentiable_rmpflow.learning.controllers import ScalarMetric, MetricCholNet, ContextMomentumNet2
from differentiable_rmpflow.learning.kinematics import EuclideanizingFlow
from differentiable_rmpflow.learning.utils import Params
from differentiable_rmpflow.learning.losses import *

from differentiable_rmpflow.rmpflow.controllers import LogCoshPotential, NaturalGradientDescentMomentumController, \
	DampingMomemtumController, QuadraticPotential, IdentityMetric
from differentiable_rmpflow.rmpflow.rmptree import RmpTreeNode
from differentiable_rmpflow.rmpflow.kinematics import DimSelectorTaskMap

import torch.optim as optim
import torch
torch.set_default_dtype(torch.float32)

import os
import numpy as np
import time
import scipy.io as sio
import pickle
import matplotlib.animation as animation
from itertools import chain

# ------------------------------------------

PARAMS = Params(n_hidden_1				= 128,		# number of hidden units in hidden layer 1 in metric cholesky net
				n_hidden_2				= 64,		# number of hidden units in hidden layer 2 in metric cholseky net
				n_blocks_flow			= 10,    	# number of blocks in diffeomorphism net
				n_hidden_flow			= 200,  	# number of hidden units in the two hidden layers in diffeomorphism net
				s_act_flow				='elu',		# (fcnn only) activation fcn in scaling network of coupling layers
				t_act_flow				='elu',		# (fcnn only) activation fcn in scaling network of coupling layers
				sigma_flow				=0.45,		# (for rfnn only) length scale
				flow_type				='realnvp',	# (realnvp/glow) architecture of flow
				coupling_network_type	='rfnn',	# (rfnn/fcnn) coupling network parameterization
				eps						= 1e-12,

				# Optimization params
				n_epochs 				=5000,  	# number of epochs
				stop_threshold			=250,
				batch_size 				=None,  	# size of minibatches
				learning_rate 			= 0.0001,  	# learning rate
				weight_decay 			= 1e-6,  	# weight for regularization

				# pre-processing params
				downsample_rate 		=4,			# data downsampling
				smoothing_window_size	=25,		# smoothing window size for savizky golay filter
				start_cut				=10,		# how much data to cut at the beginning
				end_cut					=5,		# how much data to cut at the end
				)


learning_method = 'combined'
load_models = True
save_models = False
plot_models = True
load_pretrained_models = True

robot_name = 'five_point_planar_robot'
robot_urdf = robot_name + '.urdf'

# params
dataset_name = 'lasa_handwriting_dataset_v2'
data_name = 'Sshape'
link_names = ['link4', 'link8']

demo_test = 2
workspace_dim = 2
joint_damping_gain = 1e-4
rmp_order = 1

use_kdl = True

if use_kdl:
	try:
		from differentiable_rmpflow.rmpflow.kinematics.robot_kdl import Robot
	except:
		raise ValueError('Error Importing KDL! Make sure KDL is correctly installed!')
else:
	from differentiable_rmpflow.rmpflow.kinematics.robot import Robot

# -----------------------------------------

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'models'))
models_path = os.path.join(models_dir, robot_name, data_name)
if save_models and not os.path.exists(models_path):
	print('create directory: {}'.format(models_path))
	os.makedirs(models_path)

# ------------------------------------------------

package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
data_root = os.path.join(package_path, 'data')

# Creating the robot model
urdf_dir = os.path.join(package_path, 'urdf')
urdf_path = os.path.join(urdf_dir, robot_urdf)
robot = Robot(urdf_path=urdf_path, workspace_dim=workspace_dim)
cspace_dim = robot.cspace_dim

# Load data
file_name = data_name + '.mat'
file_path = os.path.join(data_root, dataset_name, file_name)

data = sio.loadmat(file_path)
data = data['demos']
n_demos = int(data.shape[1])
n_dims = data[0, 0]['pos'][0][0].shape[0]

dt = data[0, 0]['t'][0][0][0,1] - data[0, 0]['t'][0][0][0, 0]
dt = np.round(dt * PARAMS.downsample_rate, 2)

demo_traj_list = [data[0, i]['pos'][0][0].T for i in range(n_demos)]

torch_traj_datasets = preprocess_dataset(demo_traj_list, dt=dt,
										 start_cut=PARAMS.start_cut,
										 end_cut=PARAMS.end_cut,
										 downsample_rate=PARAMS.downsample_rate,
										 smoothing_window_size=PARAMS.smoothing_window_size,
										 vel_thresh=1.,
										 goal_at_origin = True)


joint_traj_list = [traj_dataset.tensors[0].numpy() for traj_dataset in torch_traj_datasets]

# add fake zeros for the rest of joints
joint_traj_list = [np.concatenate( (traj, torch.zeros((traj.shape[0], cspace_dim-2)) ), axis=1) for traj in joint_traj_list]
time_list = [np.arange(0., joint_traj.shape[0])*dt for joint_traj in joint_traj_list]

leaf_goals = dict()
leaf_goal_biases = dict()
for link_name in link_names:
	mean_goal, goal_biases = find_mean_goal(robot, joint_traj_list, link_name,
								  base_to_tracked_frame_transforms=None)

	leaf_goals[link_name] = mean_goal
	leaf_goal_biases[link_name] = goal_biases


# ------------------------------------------------------
print('Setting up tree')
root = RmpTreeNode(n_dim=cspace_dim, name="cspace_root", order=rmp_order, return_natural=True)
root.eval()
# --------------------------------
# Adding damping to each joint

for i in range(cspace_dim):
	joint_task_map = DimSelectorTaskMap(n_inputs=cspace_dim, selected_dims=i)
	joint_node = root.add_task_space(joint_task_map, name="joint"+str(i))
	damping_rmp = DampingMomemtumController(damping_gain=joint_damping_gain)
	joint_node.add_rmp(damping_rmp)

# ------------------------------------------------------

demo_goal_list = []
dataset_list = []
for demo in range(n_demos):
	cspace_traj = joint_traj_list[demo]
	cspace_vel = np.diff(cspace_traj, axis=0) / dt
	cspace_vel = np.concatenate((cspace_vel, np.zeros((1, cspace_dim))), axis=0)

	cspace_traj = torch.from_numpy(cspace_traj).to(torch.get_default_dtype())
	cspace_vel = torch.from_numpy(cspace_vel).to(torch.get_default_dtype())

	N = cspace_traj.shape[0]

	x_list = []
	J_list = []
	p_list = []
	m_list = []

	# NOTE: Taking the original goal! (WITHOUT CUTTING!)
	cspace_goal = torch.from_numpy(joint_traj_list[demo][-1].reshape(1, -1)).to(torch.get_default_dtype())
	demo_goal_list.append(cspace_goal)

	p, m = root(x=cspace_traj)

	x_list.append(cspace_traj)
	J_list.append(torch.eye(cspace_dim).repeat((N, 1, 1)))
	p_list.append(p)
	m_list.append(m)

	for link_name in link_names:
		link_task_map = robot.get_task_map(target_link=link_name)

		link_task_map.eval()
		x, J = link_task_map(cspace_traj, order=rmp_order)

		x_list.append(x)
		J_list.append(J)
		p_list.append(torch.zeros(N, 1))
		m_list.append(torch.zeros(N, 1))

		local_goal = link_task_map.psi(cspace_goal)

	dataset = ContextDatasetMomentum(
		cspace_traj,
		cspace_vel,
		x_list, J_list, p_list, m_list
	)

	dataset_list.append(dataset)

cat_dataset = ConcatDataset(dataset_list)

# ----------------------------------------------------------------
lagrangian_nets = [None]
n=0
for link_name in link_names:
	# for tracked_frame in tracked_frames:
	x_train = torch.cat([dataset_.q_leaf_list[n + 1] for dataset_ in dataset_list], dim=0)

	minx = torch.min(x_train, dim=0)[0].reshape(1,-1)
	maxx = torch.max(x_train, dim=0)[0].reshape(1,-1)
	scaling = 1./(maxx - minx)
	translation = -minx / (maxx - minx) - 0.5

	leaf_goal = torch.from_numpy(leaf_goals[link_name]).to(dtype=torch.get_default_dtype())

	leaf_rmp = RmpTreeNode(n_dim=workspace_dim, order=rmp_order, return_natural=True)
	latent_taskmap = EuclideanizingFlow(n_inputs=workspace_dim, n_blocks=PARAMS.n_blocks_flow, n_hidden=PARAMS.n_hidden_flow,
										s_act=PARAMS.s_act_flow, t_act=PARAMS.t_act_flow,
										sigma=PARAMS.sigma_flow,
										flow_type=PARAMS.flow_type,
										coupling_network_type=PARAMS.coupling_network_type,
										goal=leaf_goal,
										normalization_scaling=scaling,
										normalization_bias=translation)


	latent_space = leaf_rmp.add_task_space(task_map=latent_taskmap)

	latent_pot_fn = LogCoshPotential()
	# latent_metric_fn = IdentityMetric()
	latent_metric_fn = MetricCholNet(n_dims=workspace_dim, n_hidden_1=PARAMS.n_hidden_1,
									 n_hidden_2=PARAMS.n_hidden_2, return_cholesky=False)

	latent_rmp = NaturalGradientDescentMomentumController(G=latent_metric_fn, del_Phi=latent_pot_fn.grad)
	latent_space.add_rmp(latent_rmp)

	lagrangian_nets.append(leaf_rmp)
	n+=1

# Loading/Training learner
# --------------------------------------------------------------------------------------

if load_models:
	print('--------------------------------------------')
	print('------------------loading-------------------')
	print('--------------------------------------------')
	n=0
	for link_name in link_names:
		model_filename = 'model_{}_{}.pt'.format(link_name, learning_method)
		lagrangian_nets[n+1].load_state_dict(torch.load(os.path.join(models_path, model_filename)))
		n+=1
else:
	print('--------------------------------------------')
	print('------------------training------------------')
	print('--------------------------------------------')
	if learning_method == 'combined':
		# NOTE: The taskmap is kept fixed, while the metric is learned!
		print('Training metrics only!')
		learnable_params = []
		if load_pretrained_models:
			n = 0
			for link_name in link_names:
				model_filename = 'model_{}_{}.pt'.format(link_name, 'independent')
				leaf_rmp = lagrangian_nets[n + 1]
				leaf_rmp.load_state_dict(torch.load(os.path.join(models_path, model_filename)))
				latent_metric_params = leaf_rmp.edges[0].child_node.parameters()
				learnable_params.append(latent_metric_params)
				n += 1
		learnable_params = chain(*learnable_params)
		model = ContextMomentumNet2(lagrangian_nets, cspace_dim, metric_scaling=[1. for net in lagrangian_nets])
		model.train()

		criterion = nn.SmoothL1Loss()
		# criterion = nn.MSELoss()
		loss_fn = get_lagrangian_force_net_loss4(criterion=criterion)
		optimizer = optim.Adam(learnable_params, lr=PARAMS.learning_rate, weight_decay=PARAMS.weight_decay)

		t_start = time.time()
		best_model, best_traj_loss = \
			train(model=model, loss_fn=loss_fn, opt=optimizer, train_dataset=cat_dataset, n_epochs=PARAMS.n_epochs,
				  batch_size=PARAMS.batch_size, stop_threshold=PARAMS.stop_threshold)
		print('time elapsed: {} seconds'.format(time.time() - t_start))
		print('\n')
	elif learning_method == 'independent':
		# NOTE: The metric is kept fixed (identity) while the taskmap is learned!
		print('Training taskmaps only! Using identity latent space metric!')
		n = 0
		for link_name in link_names:
			x_train = torch.cat([dataset_.q_leaf_list[n + 1] for dataset_ in dataset_list], dim=0)
			J_train = torch.cat([dataset_.J_list[n + 1] for dataset_ in dataset_list], dim=0)
			qd_train = torch.cat([dataset_.qd_config for dataset_ in dataset_list], dim=0)
			xd_train = torch.bmm(qd_train.unsqueeze(1), J_train.permute(0, 2, 1)).squeeze(1)
			leaf_dataset = TensorDataset(x_train, xd_train)

			leaf_rmp = lagrangian_nets[n + 1]
			leaf_rmp.return_natural = False
			leaf_rmp.train()

			# for the independent version, only train the taskmap
			learnable_params = leaf_rmp.edges[0].task_map.parameters()

			criterion = nn.SmoothL1Loss()
			# criterion = nn.MSELoss()
			loss_fn = criterion
			optimizer = optim.Adam(learnable_params, lr=PARAMS.learning_rate, weight_decay=PARAMS.weight_decay)

			t_start = time.time()
			best_model, best_traj_loss = \
				train(model=leaf_rmp, loss_fn=loss_fn, opt=optimizer, train_dataset=leaf_dataset, n_epochs=PARAMS.n_epochs,
					  batch_size=PARAMS.batch_size, stop_threshold=PARAMS.stop_threshold)
			print('time elapsed: {} seconds'.format(time.time() - t_start))
			print('\n')
			leaf_rmp.return_natural = True
			n += 1
	else:
		raise NotImplementedError('Unknown learning method. Valid methods are: 1) combined, 2) independent')

if save_models:
	print('--------------------------------------------')
	print('------------------Saving------------------')
	print('--------------------------------------------')

	n=0
	for link_name in link_names:
		model_filename = 'model_{}_{}.pt'.format(link_name, learning_method)
		torch.save(lagrangian_nets[n+1].state_dict(), os.path.join(models_path, model_filename))
		n += 1

	params_filename = 'params_{}.pt'.format(learning_method)
	with open(os.path.join(models_path, params_filename), 'wb') as handle:
		pickle.dump(PARAMS, handle, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------------------------------------------------------------------
if plot_models:
	print('--------------------------------------------')
	print('-------------------testing------------------')
	print('--------------------------------------------')

	n=0
	for link_name in link_names:
		link_task_map = robot.get_task_map(target_link=link_name)
		leaf_node = root.add_task_space(link_task_map, name=link_name)
		leaf_rmp = lagrangian_nets[n+1]
		leaf_rmp.eval()
		leaf_node.add_rmp(leaf_rmp)
		n+=1

# ---------------------------------------------------------------
root.eval()
# Rolling out
T = time_list[demo_test][-1]
t0 = 0.
print('--------------------------------------------')
print('Rolling out for demo: {}, from t0: {} to T:{}'.format(demo_test, t0, T))
print('--------------------------------------------')

ref_cspace_traj = torch.from_numpy(joint_traj_list[demo_test]).to(dtype=torch.get_default_dtype()) # dataset_list[demo_test].state.numpy().T
q0 = copy.deepcopy(ref_cspace_traj[int(t0 / dt), :].reshape(1, -1))

# q0[0, 0] = q0[0, 0] + 10.
# q0[0, 1] = q0[0, 1] + 3.

# dt = 0.1
root.return_natural = False

cspace_traj = generate_trajectories(root, x_init=q0, t_init=0., t_final=T+dt, t_step=dt, order=rmp_order, method='euler',
							 return_label=False)

time_traj = np.arange(0., T+dt, dt)

cspace_traj = cspace_traj.squeeze()
cspace_traj_np = cspace_traj.numpy()

N_ref = ref_cspace_traj

ref_link_trajs = []
link_trajs = []
for link_name in link_names:
	link_task_map = robot.get_task_map(target_link=link_name)

	# Finding the reference trajectory (with the bias!)
	ref_link_traj = link_task_map.psi(ref_cspace_traj) # NOTE: We are not composing with the tracked frame task map!
	ref_link_trajs.append(ref_link_traj)

	link_traj = link_task_map.psi(cspace_traj)
	link_trajs.append(link_traj)


# ------------------------------------------
# Plotting
diff_link_traj = link_trajs[1] - link_trajs[0]
ref_diff_link_traj = ref_link_trajs[1] - ref_link_trajs[0]

for i in range(len(link_names)):
	fig = plt.figure()
	plot_traj_time(time_traj, link_trajs[i].numpy(), '--', 'b')
	plot_traj_time(time_list[demo_test], ref_link_trajs[i].numpy(), ':', 'r')
	# plot_traj_time(time_traj, diff_link_traj.numpy(), '--', 'b')
	# plot_traj_time(time_list[demo_test], ref_diff_link_traj.numpy(), ':', 'r')

fig = plt.figure()
plt.axis(np.array([-20, 70, -20, 70]))
plt.gca().set_aspect('equal', 'box')

for i in range(len(link_names)):
	plot_traj_2D(link_trajs[i].numpy(), '--', 'b', order=1)
	plot_traj_2D(ref_link_trajs[i].numpy(), ':', 'r', order=1)

link_order = [(i, j) for i, j in zip(range(2, robot.num_links), range(3, robot.num_links))]
handles, = plot_robot_2D(robot, q0, 2, link_order=link_order)

h, = plot_robot_2D(robot, cspace_traj[0, :cspace_dim], 2, link_order=link_order)
plot_robot_2D(robot, cspace_traj[0, :cspace_dim], 2, handle_list=h, link_order=link_order)


def init():
	return h,


def animate(i):
	nsteps = cspace_traj.shape[0]
	handle, = plot_robot_2D(robot, cspace_traj[i % nsteps, :cspace_dim], 2, h, link_order=link_order)
	return handle,


ani = animation.FuncAnimation(
	fig, animate, init_func=init, interval=30, blit=False)

plt.show()


