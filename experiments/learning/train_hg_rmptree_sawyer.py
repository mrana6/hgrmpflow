from differentiable_rmpflow.learning.controllers import ContextMomentumNet2
from differentiable_rmpflow.learning.controllers import MetricCholNet
from differentiable_rmpflow.learning.kinematics import EuclideanizingFlow
from differentiable_rmpflow.learning.losses import *
from differentiable_rmpflow.learning.utils import *
from differentiable_rmpflow.rmpflow.controllers import LogCoshPotential, DampingMomemtumController, \
	NaturalGradientDescentMomentumController
from differentiable_rmpflow.rmpflow.kinematics import DimSelectorTaskMap, TargetTaskMapTF, ComposedTaskMap, TargetTaskMap
from differentiable_rmpflow.rmpflow.rmptree import RmpTreeNode

import torch
import torch.optim as optim
torch.set_default_dtype(torch.float32)

import os
import pickle
import time

import numpy as np
from matplotlib import animation


PARAMS = Params(n_hidden_1				= 128,		# number of hidden units in hidden layer 1 in metric cholesky net
				n_hidden_2				= 64,		# number of hidden units in hidden layer 2 in metric cholseky net
				n_blocks_flow			= 10,   	# number of blocks in diffeomorphism net
				n_hidden_flow			= 200,  	# number of hidden units in the two hidden layers in diffeomorphism net
				s_act_flow				='elu',		# (fcnn only) activation fcn in scaling network of coupling layers
				t_act_flow				='elu',		# (fcnn only) activation fcn in scaling network of coupling layers
				sigma_flow				=0.45,		# (for rfnn only) length scale
				flow_type				='realnvp',	# (realnvp/glow) architecture of flow
				coupling_network_type	='rfnn',	# (rfnn/fcnn) coupling network parameterization
				eps						= 1e-12,

				# Optimization params
				n_epochs 				= 5000,  	# number of epochs
				stop_threshold			= 250,
				batch_size 				= None,  	# size of minibatches
				learning_rate 			= 0.0001,  	# learning rate
				weight_decay 			= 1e-10,  	# weight for regularization

				# pre-processing params
				downsample_rate 		=1,			# data downsampling
				smoothing_window_size	=31,		# smoothing window size for savizky golay filter
				start_cut				=15,		# how much data to cut at the beginning
				end_cut					=10,		# how much data to cut at the end

				# Dagger params
				n_dagger_iterations 	= 1,  		# number of dagger iterations ( NOTE: regular training if no dagger used)
				tracker_kp 				= 2.,  		# proportional gain for tracking controller
				split_ratio 			= 0.32,  	# ratio of original out of aggregated dataset
				)


# ------------------------------------------------------------------

load_models = True
save_models = False
test_models = True
use_kdl = False
learning_method = 'combined'
joint_damping_gain = 0.50
rmp_order = 1
workspace_dim = 3
cspace_dim = 7

demo_test = 8

if use_kdl:
	try:
		from differentiable_rmpflow.rmpflow.kinematics.robot_kdl import Robot
	except:
		raise ValueError('Error Importing KDL! Make sure KDL is correctly installed!')
else:
	from differentiable_rmpflow.rmpflow.kinematics.robot import Robot


robot_name = 'sawyer'#'franka'
robot_urdf = 'sawyer_fixed_head_gripper.urdf'#'lula_franka_gen_fixed_gripper.urdf'
skill_name = 'drawer_closing'#'chewie_door_reaching'#'synthetic_reaching'
base_link = 'base'#'base_link'
link_names = ['right_l6', 'right_gripper_r_finger_tip', 'right_gripper_l_finger_tip']#['panda_link7', 'panda_leftfingertip', 'panda_rightfingertip']		# TODO: Make sure order is same as data parser

if skill_name == 'chewie_door_reaching'and robot_name == 'franka':
	tracked_frame = 'chewie_door_right_handle'
elif skill_name == 'drawer_closing' and robot_name == 'franka':
	tracked_frame = 'drawer'
elif skill_name == 'synthetic_reaching':
	tracked_frame = 'virtual_obj'
else:
	tracked_frame = 'virtual_obj'
	print('WARNING! The skill hasnt been preprocessed!')

# -----------------------------------------

package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

models_dir = os.path.join(package_path, 'models')
models_path = os.path.join(models_dir, robot_name, skill_name)
if save_models and not os.path.exists(models_path):
	print('create directory: {}'.format(models_path))
	os.makedirs(models_path)

# ------------------------------------------
# Loading data
print('loading data')

data_dir = os.path.join(package_path, 'data')
file_name = skill_name + '_smooth.pickle'
file_path = os.path.join(data_dir, robot_name, file_name)

with open(file_path, 'rb') as f:
	u = pickle._Unpickler(f)
	u.encoding = 'latin1'
	dataset_smooth = u.load()

# Creating the robot model
urdf_dir = os.path.join(package_path, 'urdf')
urdf_path = os.path.join(urdf_dir, robot_urdf)
robot = Robot(urdf_path=urdf_path, workspace_dim=workspace_dim)

# ----------------------------------------
time_list = dataset_smooth['time']
num_demos = len(time_list)

dt = time_list[0][1,0] - time_list[0][0, 0]
dt = dt*PARAMS.downsample_rate
dt = np.round(dt, 2)

dataset = dict()
joint_traj_list = dataset_smooth['joint_pos']
joint_traj_list = [joint_traj[PARAMS.start_cut:-PARAMS.end_cut:PARAMS.downsample_rate, :cspace_dim] for joint_traj in joint_traj_list]
time_list = [torch.arange(0., joint_traj.shape[0])*dt for joint_traj in joint_traj_list]

if tracked_frame is not None:
	base_to_tracked_frame_transforms = dataset_smooth['transforms'][tracked_frame]
else:
	base_to_tracked_frame_transforms = [np.eye(workspace_dim+1, workspace_dim+1) for _ in range(len(joint_traj_list))]

leaf_goals = dict()
leaf_goal_biases = dict()
for link_name in link_names:
	mean_goal, goal_biases = find_mean_goal(robot, joint_traj_list, link_name,
								  base_to_tracked_frame_transforms=base_to_tracked_frame_transforms)

	leaf_goals[link_name] = mean_goal
	leaf_goal_biases[link_name] = goal_biases

# -----------------------------------------
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

# -----------------------------------------

dataset_list = []
for demo in range(num_demos):
	cspace_traj = joint_traj_list[demo]
	cspace_vel = np.diff(cspace_traj, axis=0) / dt
	cspace_vel = np.concatenate((cspace_vel, np.zeros((1, cspace_dim))), axis=0)

	cspace_traj = torch.from_numpy(cspace_traj).to(dtype=torch.get_default_dtype())
	cspace_vel = torch.from_numpy(cspace_vel).to(dtype=torch.get_default_dtype())

	N = cspace_traj.shape[0]

	x_list = []
	J_list = []
	p_list = []
	m_list = []
	demo_goal_list = []

	p, m = root(x=cspace_traj)

	x_list.append(cspace_traj)
	J_list.append(torch.eye(cspace_dim).repeat((N, 1, 1)))
	p_list.append(p)
	m_list.append(m)

	for link_name in link_names:
		link_taskmap = robot.get_task_map(target_link=link_name)

		T_frame_to_base = \
			torch.from_numpy(transform_inv(base_to_tracked_frame_transforms[demo])).to(dtype=torch.get_default_dtype())
		frame_taskmap = TargetTaskMapTF(T_frame_to_base, n_inputs=workspace_dim)

		goal_bias_taskmap = \
			TargetTaskMap(goal=torch.from_numpy(leaf_goal_biases[link_name][demo]).to(dtype=torch.get_default_dtype()))

		composed_task_map = ComposedTaskMap(taskmaps=[link_taskmap, frame_taskmap, goal_bias_taskmap],
											use_numerical_jacobian=False)

		composed_task_map.eval()
		x, J = composed_task_map(cspace_traj, order=rmp_order)

		x_list.append(x)
		J_list.append(J)
		p_list.append(torch.zeros(N, 1))
		m_list.append(torch.zeros(N, 1))

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
		model_filename = 'model_{}_in_{}_{}.pt'.format(link_name, tracked_frame, learning_method)
		lagrangian_nets[n+1].load_state_dict(torch.load(os.path.join(models_path, model_filename)))
		n+=1
else:
	print('--------------------------------------------')
	print('------------------training------------------')
	print('--------------------------------------------')
	if learning_method == 'combined':
		model = ContextMomentumNet2(lagrangian_nets, cspace_dim, metric_scaling=[1. for net in lagrangian_nets])
		model.train()

		criterion = nn.SmoothL1Loss()
		loss_fn = get_lagrangian_force_net_loss4(criterion=criterion)
		optimizer = optim.Adam(model.parameters(), lr=PARAMS.learning_rate, weight_decay=PARAMS.weight_decay)

		t_start = time.time()
		best_model, best_traj_loss = \
			train(model=model, loss_fn=loss_fn, opt=optimizer, train_dataset=cat_dataset, n_epochs=PARAMS.n_epochs,
				  batch_size=PARAMS.batch_size, stop_threshold=PARAMS.stop_threshold)
		print('time elapsed: {} seconds'.format(time.time() - t_start))
		print('\n')
	elif learning_method == 'independent':
		n = 0
		for link_name in link_names:
			x_train = torch.cat([dataset_.q_leaf_list[n + 1] for dataset_ in dataset_list], dim=0)
			J_train = torch.cat([dataset_.J_list[n + 1] for dataset_ in dataset_list], dim=0)
			qd_train = torch.cat([dataset_.qd_config for dataset_ in dataset_list], dim=0)
			xd_train = torch.bmm(qd_train.unsqueeze(1), J_train.permute(0, 2, 1)).squeeze(1)
			leaf_dataset = TensorDataset(x_train, xd_train)

			model = lagrangian_nets[n + 1]
			model.return_natural = False
			model.train()

			criterion = nn.SmoothL1Loss()
			loss_fn = criterion
			optimizer = optim.Adam(model.parameters(), lr=PARAMS.learning_rate, weight_decay=PARAMS.weight_decay)

			t_start = time.time()
			best_model, best_traj_loss = \
				train(model=model, loss_fn=loss_fn, opt=optimizer, train_dataset=leaf_dataset, n_epochs=PARAMS.n_epochs,
					  batch_size=PARAMS.batch_size, stop_threshold=PARAMS.stop_threshold)
			print('time elapsed: {} seconds'.format(time.time() - t_start))
			print('\n')
			model.return_natural = True
			n += 1
	else:
		raise NotImplementedError('Unknown learning method. Valid methods are: 1) combined, 2) independent')

if save_models:
	print('--------------------------------------------')
	print('------------------Saving------------------')
	print('--------------------------------------------')

	n=0
	for link_name in link_names:
		model_filename = 'model_{}_in_{}_{}.pt'.format(link_name, tracked_frame, learning_method)
		torch.save(lagrangian_nets[n+1].state_dict(), os.path.join(models_path, model_filename))
		n += 1

	params_filename = 'params_{}.pt'.format(learning_method)
	with open(os.path.join(models_path, params_filename), 'wb') as handle:
		pickle.dump(PARAMS, handle, protocol=pickle.HIGHEST_PROTOCOL)


# --------------------------------------------------------------------------------------

if test_models:
	print('--------------------------------------------')
	print('-------------------testing------------------')
	print('--------------------------------------------')

	n=0
	for link_name in link_names:
		link_taskmap = robot.get_task_map(target_link=link_name)

		T_frame_to_base = \
			torch.from_numpy(transform_inv(base_to_tracked_frame_transforms[demo_test])).to(dtype=torch.get_default_dtype())
		frame_taskmap = TargetTaskMapTF(T_frame_to_base, n_inputs=workspace_dim)

		goal_bias_taskmap = \
			TargetTaskMap(goal=torch.from_numpy(leaf_goal_biases[link_name][demo_test]).to(dtype=torch.get_default_dtype()))

		composed_task_map = ComposedTaskMap(taskmaps=[link_taskmap, frame_taskmap, goal_bias_taskmap],
											use_numerical_jacobian=False)

		leaf_node = root.add_task_space(composed_task_map, name=tracked_frame)
		leaf_rmp = lagrangian_nets[n+1]
		leaf_rmp.eval()
		leaf_rmp.return_natural = True
		leaf_node.add_rmp(leaf_rmp)
		n+=1

# ---------------------------------------------------------------
root.eval()
# Rolling out
T = time_list[demo_test][-1] + dt*150.
t0 = 0.
print('--------------------------------------------')
print('Rolling out for demo: {}, from t0: {} to T:{}'.format(demo_test, t0, T))
print('--------------------------------------------')

ref_cspace_traj = torch.from_numpy(joint_traj_list[demo_test]).to(dtype=torch.get_default_dtype()) # dataset_list[demo_test].state.numpy().T
# q0 = copy.deepcopy(ref_cspace_traj[int(t0 / dt), :].reshape(1, -1))

q0 = torch.tensor([0.5928310546875, 0.2513330078125, -2.05405078125,
				   0.842865234375, 2.146251953125, 1.7718466796875, 2.9842958984375]).reshape(1,-1)
# q0[0, -1] = q0[0, -1] + np.pi/8
# q0[0, -2] = q0[0, -2]
# q0[0, -3] = q0[0, -3] + np.pi/8
# q0[0, 2] = q0[0, 2] + np.pi/8

# dt = 0.
root.return_natural = False

cspace_traj = generate_trajectories(root, x_init=q0, t_init=0., t_final=T+dt, t_step=dt, order=rmp_order, method='euler',
							 return_label=False)

time_traj = np.arange(0, cspace_traj.shape[0])*dt
# time_traj = np.arange(0., T+dt, dt)

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

np.savetxt('rep.txt', np.concatenate((time_traj.reshape(-1,1), cspace_traj_np), axis=1), delimiter=',')

# ------------------------------------------
# Plotting

for i in range(len(link_names)):
	fig = plt.figure()
	plot_traj_time(time_traj, link_trajs[i].numpy(), '--', 'b')
	plot_traj_time(time_list[demo_test], ref_link_trajs[i].numpy(), ':', 'r')


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlim3d(-0.8, 1)
ax.set_ylim3d(-0.8, 1)
ax.set_zlim3d(-0.1, 1)

for i in range(len(link_names)):
	plot_traj_3D(link_trajs[i].numpy(), '--', 'b')
	plot_traj_3D(ref_link_trajs[i].numpy(), ':', 'r')

if robot_name == 'franka':
	link_order = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 8), (8, 9), (9, 11),
				  (11, 13), (11, 14), (14, 17), (11, 15), (15, 16)]
elif robot_name == 'sawyer':
	link_order = [(0, 6), (6, 9), (9, 11), (11, 13), (13, 16), (16, 18), (18, 19), (19, 22), # right_l6
				  (22, 23), (23, 25)]

handles, = plot_robot_3D(robot, q0, 2, link_order=link_order)

h, = plot_robot_3D(robot, cspace_traj[0, :cspace_dim], 2, link_order=link_order)
plot_robot_3D(robot, cspace_traj[0, :cspace_dim], 2, handle_list=h, link_order=link_order)


def init():
	return h,


def animate(i):
	nsteps = cspace_traj.shape[0]
	handle, = plot_robot_3D(robot, cspace_traj[i % nsteps, :cspace_dim], 2, h, link_order=link_order)
	return handle,


ani = animation.FuncAnimation(
	fig, animate, init_func=init, interval=30, blit=False)

plt.show()
