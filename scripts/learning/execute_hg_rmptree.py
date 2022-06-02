from differentiable_rmpflow.learning.controllers import MetricCholNet
from differentiable_rmpflow.learning.kinematics import EuclideanizingFlow
from differentiable_rmpflow.learning.utils import *
from differentiable_rmpflow.rmpflow.controllers import LogCoshPotential, DampingMomemtumController, \
	NaturalGradientDescentMomentumController
from differentiable_rmpflow.rmpflow.kinematics import DimSelectorTaskMap, TargetTaskMapTF, ComposedTaskMap, TargetTaskMap
from differentiable_rmpflow.rmpflow.rmptree import RmpTreeNode

import torch
torch.set_default_dtype(torch.float32)

import os
import pickle
import numpy as np
from matplotlib import animation

# -----------------------------------------

model_suffix = 'combined'
demo_test = 0
workspace_dim = 3
cspace_dim = 7
joint_damping_gain = 0.50
rmp_order = 1
use_kdl = True

if use_kdl:
	try:
		from differentiable_rmpflow.rmpflow.kinematics.robot_kdl import Robot
	except:
		raise ValueError('Error Importing KDL! Make sure KDL is correctly installed!')
else:
	from differentiable_rmpflow.rmpflow.kinematics.robot import Robot

robot_name = 'franka'
robot_urdf = 'lula_franka_gen_fixed_gripper.urdf'
skill_name = 'chewie_door_reaching'#'synthetic_reaching'
base_link = 'base_link'
link_names = ['panda_link7', 'panda_leftfingertip', 'panda_rightfingertip']		# TODO: Make sure order is same as data parser

if skill_name == 'chewie_door_reaching':
	tracked_frame = 'chewie_door_right_handle'
elif skill_name == 'drawer_closing':
	tracked_frame = 'drawer'
elif skill_name == 'synthetic_reaching':
	tracked_frame = 'virtual_obj'
else:
	tracked_frame = None
	print('WARNING! The skill hasnt been preprocessed!')

# -------------------------------------------------------------
package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
models_dir = os.path.join(package_path, 'models')
models_path = os.path.join(models_dir, robot_name, skill_name)

# ---------------------------------------------------------------
# Loading params

params_filename = 'params_{}.pt'.format(model_suffix)

with open(os.path.join(models_path, params_filename), 'rb') as f:
	PARAMS = pickle.load(f)

# -----------------------------------------------------------------
# Creating the robot model
urdf_dir = os.path.join(package_path, 'urdf')
urdf_path = os.path.join(urdf_dir, robot_urdf)
robot = Robot(urdf_path=urdf_path, workspace_dim=workspace_dim)

# -----------------------------------------------------------------
# Loading data

data_dir = os.path.join(package_path, 'data')
file_name = skill_name + '_smooth.pickle'
file_path = os.path.join(data_dir, robot_name, file_name)
with open(file_path, 'rb') as f:
	u = pickle._Unpickler(f)
	u.encoding = 'latin1'
	dataset_smooth = u.load()

time_list = dataset_smooth['time']
num_demos = len(time_list)

dt = time_list[0][1,0] - time_list[0][0, 0]
dt = dt*PARAMS.downsample_rate
dt = np.round(dt, 2)

dataset = dict()
joint_traj_list = dataset_smooth['joint_pos']
joint_traj_list = [joint_traj[PARAMS.start_cut:-PARAMS.end_cut:PARAMS.downsample_rate, :cspace_dim]
				   for joint_traj in joint_traj_list]
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
root.return_natural = False
# --------------------------------
# Adding damping to each joint

for i in range(cspace_dim):
	joint_task_map = DimSelectorTaskMap(n_inputs=cspace_dim, selected_dims=i)
	joint_node = root.add_task_space(joint_task_map, name="joint"+str(i))
	damping_rmp = DampingMomemtumController(damping_gain=joint_damping_gain)
	joint_node.add_rmp(damping_rmp)

# --------------------------------
# Adding leaf RMPs

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

	# adding learned RMPs
	# ---------------------------------------------------------------

	leaf_rmp = RmpTreeNode(n_dim=workspace_dim, order=rmp_order, return_natural=True)
	latent_taskmap = EuclideanizingFlow(n_inputs=workspace_dim, n_blocks=PARAMS.n_blocks_flow,
										n_hidden=PARAMS.n_hidden_flow,
										s_act=PARAMS.s_act_flow, t_act=PARAMS.t_act_flow,
										sigma=PARAMS.sigma_flow,
										flow_type=PARAMS.flow_type,
										coupling_network_type=PARAMS.coupling_network_type,
										goal=None,
										normalization_scaling=None,
										normalization_bias=None)
	latent_space = leaf_rmp.add_task_space(task_map=latent_taskmap)

	latent_pot_fn = LogCoshPotential()
	latent_metric_fn = MetricCholNet(n_dims=workspace_dim, n_hidden_1=PARAMS.n_hidden_1,
									 n_hidden_2=PARAMS.n_hidden_2, return_cholesky=False)
	latent_rmp = NaturalGradientDescentMomentumController(G=latent_metric_fn, del_Phi=latent_pot_fn.grad)
	latent_space.add_rmp(latent_rmp)

	model_filename = 'model_{}_in_{}_{}.pt'.format(link_name, tracked_frame, model_suffix)
	leaf_rmp.load_state_dict(torch.load(os.path.join(models_path, model_filename)))

	# ----------------------------------------------------------------
	leaf_node.add_rmp(leaf_rmp)

# ---------------------------------------------------------------
root.eval()
# Rolling out
t0 = 0.
T = time_list[demo_test][-1]
print('--------------------------------------------')
print('Rolling out for demo: {}, from t0: {} to T:{}'.format(demo_test, t0, T))
print('--------------------------------------------')

ref_cspace_traj = torch.from_numpy(joint_traj_list[demo_test]).to(dtype=torch.get_default_dtype()) # dataset_list[demo_test].state.numpy().T
q0 = copy.deepcopy(ref_cspace_traj[int(t0 / dt), :].reshape(1, -1))
# q0[0, -1] = q0[0, -1] + np.pi/8
# q0[0, -2] = q0[0, -2]
# q0[0, -3] = q0[0, -3] + np.pi/8
# q0[0, 2] = q0[0, 2] + np.pi/8

# dt = 0.
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

link_order = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 7), (7, 8), (8, 9), (9, 11),
			  (11, 13), (11, 14), (14, 17), (11, 15), (15, 16)]
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