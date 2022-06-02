from differentiable_rmpflow.rmpflow.utils import *
from differentiable_rmpflow.rmpflow.rmptree import *
from differentiable_rmpflow.rmpflow.controllers import TargetMomentumControllerUniform, ObstacleAvoidanceMomentumController, DampingMomemtumController
from differentiable_rmpflow.rmpflow.kinematics import SphereDistanceTaskMap, TargetTaskMap, DimSelectorTaskMap

import torch
torch.set_default_dtype(torch.float32)   # IMP: high precision can be required for this!

import os
import numpy as np
import time

import matplotlib.animation as animation


order = 1
wspace_dim = 2
joint_damping_gain = 0.5
obstacle_center = torch.tensor([-5, 10]).reshape(1, -1)
obstacle_radius = 1.0
obs_alpha = 1e-5
xg = torch.tensor([-10, 15]).reshape(1,-1)
use_kdl = True

if use_kdl:
	try:
		from differentiable_rmpflow.rmpflow.kinematics.robot_kdl import Robot
	except:
		raise ValueError('Error Importing KDL! Make sure KDL is correctly installed!')
else:
	from differentiable_rmpflow.rmpflow.kinematics.robot import Robot

# -------------------------------------------
# setting up robot

urdf_name = 'snake_robot.urdf'
package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
urdf_dir = os.path.join(package_path, 'urdf')
urdf_path = os.path.join(urdf_dir, urdf_name)
robot = Robot(workspace_dim=wspace_dim, urdf_path=urdf_path)
cspace_dim = robot.cspace_dim

# ---------------------------------------------
# setting up tree
root = RmpTreeNode(n_dim=cspace_dim, name='root', order=order)

for i in range(cspace_dim):
	joint_task_map = DimSelectorTaskMap(n_inputs=cspace_dim, selected_dims=i)
	joint_node = root.add_task_space(joint_task_map, name="joint"+str(i))
	damping_rmp = DampingMomemtumController(damping_gain=joint_damping_gain)
	joint_node.add_rmp(damping_rmp)


# last link
link_taskmap = robot.task_maps[list(robot.task_maps.keys())[-1]]
link_node = root.add_task_space(task_map=link_taskmap, name='link')

# attractor
target_taskmap = TargetTaskMap(xg)
controller = TargetMomentumControllerUniform(proportional_gain=1.0, norm_robustness_eps=1e-12, alpha=1.0, w_u=10.0, w_l=1.0, sigma=1.0)
leaf = link_node.add_task_space(task_map=target_taskmap, name='target')
leaf.add_rmp(controller)


# second-last link
link_taskmap = robot.task_maps[list(robot.task_maps.keys())[-2]]
link_node = root.add_task_space(task_map=link_taskmap, name='link1')

# attractor
xg2 = xg + torch.tensor([2, 0.]).reshape(1, -1)
target_taskmap = TargetTaskMap(xg2)
controller = TargetMomentumControllerUniform(proportional_gain=1.0, norm_robustness_eps=1e-12, alpha=1.0, w_u=10.0, w_l=1.0, sigma=1.0)
leaf = link_node.add_task_space(task_map=target_taskmap, name='target')
leaf.add_rmp(controller)

# -----------------------------------------------
# rolling out
root.eval()
q0 = torch.zeros(1, cspace_dim)

q0[0, 2] = np.pi/2
q0[0, 3] = -np.pi/2
q0[0, 4] = np.pi/2
q0[0, 5] = -np.pi/2
q0[0, 6] = np.pi/2
q0[0, 7] = -np.pi/2
q0[0, 8] = np.pi/2
q0[0, 9] = -np.pi/2

T = 50.
dt = 0.1
root.return_natural = False
t0 = time.time()
traj = generate_trajectories(root, x_init=q0, t_init=0., t_final=T, t_step=dt, order=order, method='rk4', return_label=False)
tf = time.time()
print('avg compute time per step = {}'.format((tf-t0)*dt/T))


traj = traj.squeeze()
link_traj = link_taskmap.psi(traj)

traj_np = traj.numpy()

# -----------------------------------------------
# plotting
fig = plt.figure()
plt.axis(np.array([-20, 20, -20, 20]))
plt.gca().set_aspect('equal', 'box')

link_order = [(i, j) for i, j in zip(range(2, robot.num_links), range(3, robot.num_links))]
handles, = plot_robot_2D(robot, q0, 2, link_order=link_order)

h, = plot_robot_2D(robot, traj[0, :cspace_dim], 2, link_order=link_order)
plot_robot_2D(robot, traj[0, :cspace_dim], 2, handle_list=h, link_order=link_order)


def init():
	return h,


def animate(i):
	nsteps = traj.shape[0]
	handle, = plot_robot_2D(robot, traj[i % nsteps, :cspace_dim], 2, h, link_order=link_order)
	return handle,


ani = animation.FuncAnimation(
	fig, animate, init_func=init, interval=30, blit=False)

plt.show()
