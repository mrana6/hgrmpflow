from differentiable_rmpflow.rmpflow.utils import *
from differentiable_rmpflow.rmpflow.rmptree import *
from differentiable_rmpflow.rmpflow.controllers import TargetMomentumControllerUniform, ObstacleAvoidanceMomentumController
from differentiable_rmpflow.rmpflow.kinematics import SphereDistanceTaskMap, TargetTaskMap

import torch
torch.set_default_dtype(torch.float32)

import os
import numpy as np
import time



order = 1
cspace_dim = 2
wspace_dim = 2
obstacle_center = torch.zeros(1, wspace_dim)
obstacle_radius = 1.0
obs_alpha = 1e-5
xg = torch.tensor([-3, 3]).reshape(1,-1)
use_kdl = False

if use_kdl:
	try:
		from differentiable_rmpflow.rmpflow.kinematics.robot_kdl import Robot
	except:
		raise ValueError('Error Importing KDL! Make sure KDL is correctly installed!')
else:
	from differentiable_rmpflow.rmpflow.kinematics.robot import Robot

# -------------------------------------------
# setting up robot

urdf_name = 'point_planar_robot.urdf'
package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
urdf_dir = os.path.join(package_path, 'urdf')
urdf_path = os.path.join(urdf_dir, urdf_name)
robot = Robot(workspace_dim=wspace_dim, urdf_path=urdf_path)

# ---------------------------------------------
# setting up tree
root = RmpTreeNode(n_dim=cspace_dim, name='root', order=order)
link_taskmap = robot.get_task_map(base_link='link1', target_link='link3')
link_node = root.add_task_space(task_map=link_taskmap, name='link')

# attractor
target_taskmap = TargetTaskMap(xg)
controller = TargetMomentumControllerUniform(proportional_gain=1.0, norm_robustness_eps=1e-12, alpha=1.0, w_u=10.0, w_l=1.0, sigma=1.0)
leaf = link_node.add_task_space(task_map=target_taskmap, name='target')
leaf.add_rmp(controller)

# obstacle
obstacle_taskmap = SphereDistanceTaskMap(n_inputs=wspace_dim, radius=obstacle_radius, center=obstacle_center)
obstacle_controller = ObstacleAvoidanceMomentumController(proportional_gain=obs_alpha)
leaf = link_node.add_task_space(task_map=obstacle_taskmap, name='obstacle')
leaf.add_rmp(obstacle_controller)

# -----------------------------------------------
# rolling out
root.eval()
q = torch.tensor([2.5, -3.2]).reshape(1, -1)
qd = torch.zeros(1, 2)


T = 20.
dt = 0.1
root.return_natural = False
t0 = time.time()
traj = generate_trajectories(root, x_init=q, t_init=0., t_final=T, t_step=dt, order=order, method='rk4', return_label=False)
tf = time.time()
print('avg compute time per step = {}'.format((tf-t0)*dt/T))

traj_np = traj.squeeze().numpy()

# -----------------------------------------------
# plotting
fig = plt.figure()
plt.axis(np.array([-5, 5, -5, 5]))
plt.gca().set_aspect('equal', 'box')

plot_traj_2D(traj_np, '--', 'b', order=order)

circle = plt.Circle((obstacle_center[0,0], obstacle_center[0,1]), obstacle_radius, color='k', fill=False)
plt.gca().add_artist(circle)

plt.show()