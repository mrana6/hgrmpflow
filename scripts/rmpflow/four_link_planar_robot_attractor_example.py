from differentiable_rmpflow.rmpflow.utils import *
from differentiable_rmpflow.rmpflow.rmptree import *
from differentiable_rmpflow.rmpflow.controllers import TargetForceControllerUniform, DampingForceController
from differentiable_rmpflow.rmpflow.kinematics import TargetTaskMap, DimSelectorTaskMap

import torch
torch.set_default_dtype(torch.float32)

import os
import numpy as np
import matplotlib.animation as animation


order = 2
wspace_dim = 2
urdf_name = 'four_link_planar_robot.urdf'
use_kdl = False

joint_damping_gain = 1e-2
eef_link_name = 'link5'
eef_goal = torch.tensor([-5., 10.]).reshape(1, -1)

if use_kdl:
	try:
		from differentiable_rmpflow.rmpflow.kinematics.robot_kdl import Robot
	except:
		raise ValueError('Error Importing KDL! Make sure KDL is correctly installed!')
else:
	from differentiable_rmpflow.rmpflow.kinematics.robot import Robot


# -------------------------------------------
# setting up robot

package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
urdf_dir = os.path.join(package_path, 'urdf')
urdf_path = os.path.join(urdf_dir, urdf_name)
robot = Robot(workspace_dim=wspace_dim, urdf_path=urdf_path)
cspace_dim = robot.cspace_dim
# ---------------------------------------------
# setting up tree
root = RmpTreeNode(n_dim=cspace_dim, name='root')
link_taskmap = robot.get_task_map(base_link='link1', target_link=eef_link_name)
link_node = root.add_task_space(task_map=link_taskmap, name='link')

# joint dampings
for i in range(cspace_dim):
	joint_task_map = DimSelectorTaskMap(n_inputs=cspace_dim, selected_dims=i)
	joint_node = root.add_task_space(joint_task_map, name="joint"+str(i))
	damping_rmp = DampingForceController(damping_gain=joint_damping_gain)
	joint_node.add_rmp(damping_rmp)

# attractor
target_taskmap = TargetTaskMap(eef_goal)
controller = TargetForceControllerUniform(damping_gain=2., proportional_gain=1.0,
										  norm_robustness_eps=1e-12, alpha=1.0, w_u=10.0, w_l=1.0, sigma=1.0)

leaf = link_node.add_task_space(task_map=target_taskmap, name='target')
leaf.add_rmp(controller)

# -----------------------------------------------
# rolling out
print('Rolling out...')
root.eval()

q0 = torch.zeros(1, cspace_dim)
qd0 = torch.zeros(1, cspace_dim)
s0 = torch.cat((q0, qd0), dim=1)

T = 30.
dt = 0.1

root.return_natural = False
traj = generate_trajectories(root, x_init=s0, t_init=0., t_final=T, t_step=dt, order=order, method='rk4', return_label=False)

traj = traj.squeeze()
link_traj = link_taskmap.psi(traj[:, :cspace_dim]).numpy()

traj_np = traj.numpy()

# -----------------------------------------------
# plotting
fig = plt.figure()
plt.axis(np.array([-20, 20, -20, 20]))
plt.gca().set_aspect('equal', 'box')

link_order = [(i, j) for i, j in zip(range(0, robot.num_links), range(1, robot.num_links))]
handles, = plot_robot_2D(robot, q0, 2, link_order=link_order)

h, = plot_robot_2D(robot, traj[0, :cspace_dim], 2, link_order=link_order)
plot_robot_2D(robot, traj[0, :cspace_dim], 2, handle_list=h, link_order=link_order)
plot_traj_2D(link_traj, ls='--', color='m', order=1)

def init():
	return h,


def animate(i):
	nsteps = traj.shape[0]
	handle, = plot_robot_2D(robot, traj[i % nsteps, :cspace_dim], 2, h, link_order=link_order)
	return handle,

ani = animation.FuncAnimation(
	fig, animate, init_func=init, interval=30, blit=False)

plt.show()