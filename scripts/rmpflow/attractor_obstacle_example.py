from differentiable_rmpflow.rmpflow.utils import *
from differentiable_rmpflow.rmpflow.rmptree import RmpTreeNode
from differentiable_rmpflow.rmpflow.controllers import TargetForceControllerUniform, ObstacleAvoidanceForceController
from differentiable_rmpflow.rmpflow.kinematics import SphereDistanceTaskMap, TargetTaskMap

import torch
torch.set_default_dtype(torch.float32)

cspace_dim = 2
wspace_dim = 2
obstacle_center = torch.zeros(1, wspace_dim)
obstacle_radius = 1.0
obs_alpha = 1e-5

xg = torch.tensor([-3, 3]).reshape(1,-1)

# ---------------------------------------------
# setting up tree
root = RmpTreeNode(n_dim=cspace_dim, name='root')

# attractor
target_taskmap = TargetTaskMap(xg)
controller = TargetForceControllerUniform(damping_gain=2., proportional_gain=1.0, norm_robustness_eps=1e-12, alpha=1.0, w_u=10.0, w_l=1.0, sigma=1.0)
leaf = root.add_task_space(task_map=target_taskmap, name='target')
leaf.add_rmp(controller)

# obstacle
obstacle_taskmap = SphereDistanceTaskMap(n_inputs=wspace_dim, radius=obstacle_radius, center=obstacle_center)
obstacle_controller = ObstacleAvoidanceForceController(proportional_gain=obs_alpha, damping_gain=0.0, epsilon=0.2)
leaf = root.add_task_space(task_map=obstacle_taskmap, name='obstacle')
leaf.add_rmp(obstacle_controller)

# -----------------------------------------------
# rolling out

root.eval()

q = torch.tensor([2.5, -3.2]).reshape(1, -1)
qd = torch.tensor([-1.0, 1.0]).reshape(1, -1)

T = 20.
dt = 0.01

traj = q
tt = torch.tensor([0.0])
import time
t0 = time.time()
N = int(T / dt)
for i in range(1, N):
    qdd, M = root.eval_canonical(q, qd)
    q = q + dt*qd
    qd = qd + dt*qdd
    traj = torch.cat((traj, q), dim=0)
    tt = torch.cat((tt, torch.tensor([dt*i])))
tf = time.time()
print('avg compute time per step = {}'.format((tf-t0)/N))

time_np = tt.numpy()
traj_np = traj.numpy()

fig = plt.figure()
plt.axis(np.array([-5, 5, -5, 5]))
plt.gca().set_aspect('equal', 'box')

plot_traj_2D(traj_np, '--', 'b', order=1)

circle = plt.Circle((obstacle_center[0,0], obstacle_center[0,1]), obstacle_radius, color='k', fill=False)
plt.gca().add_artist(circle)
plt.show()