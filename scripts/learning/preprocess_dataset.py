import pickle
import os
from collections import OrderedDict
import matplotlib.pyplot as plt

from differentiable_rmpflow.learning.utils.preprocess_utils import *
from differentiable_rmpflow.rmpflow.utils.plot_utils import *

import torch

torch.set_default_dtype(torch.float64)

use_kdl = False

if use_kdl:
    try:
        from differentiable_rmpflow.rmpflow.kinematics.robot_kdl import Robot
    except:
        raise ValueError('Error Importing KDL! Make sure KDL is correctly installed!')
else:
    from differentiable_rmpflow.rmpflow.kinematics.robot import Robot

# ------------------------------------------
# Create a dict of smoothed dataset
# Loads the bag file parsed in the format given by bag parser
# Run the parser first

# TODO: The transforms to tracked frames assume the tracked frames are STATIC!


# --------------------------------------------
# Params!
robot_name = 'sawyer'
skill_name = 'drawer_closing'

if robot_name == 'franka':
    eef_name = 'right_gripper'  # only for plotting
    base_link = 'base_link'
    robot_urdf_name = 'lula_franka_gen.urdf'  # TODO: can use the param server instead
    joint_idx = np.arange(7)
elif robot_name == 'jaco':
    eef_name = 'j2s7s300_end_effector'
    base_link = 'j2s7s300_link_base'
    robot_urdf_name = 'j2s7s300.urdf'
    joint_idx = np.arange(7)
elif robot_name == 'sawyer':
    eef_name = 'right_hand'
    base_link = 'base'
    robot_urdf_name = 'sawyer_fixed_head_gripper.urdf'
    joint_idx = np.arange(1, 8)  # removing head_pan and torso joints
else:
    base_link = 'base_link'
    print('WARNING! robot config doesnt exist!')

if skill_name == 'chewie_door_reaching':
    world_link = 'world'
    base_link_bag = '01_base_link'  # name of the base frame in the bag file, gives the transform in 'world' frame
    tracked_frames = ['chewie_door_right_handle']  # track frames bag file should have this!
elif skill_name == 'drawer_closing':
    world_link = None
    base_link_bag = base_link  # name of the base frame in the bag file (if it has been remapped,  otherwise set to base_link)
    tracked_frames = ['drawer']  # track frames bag file should have this!
elif skill_name == 'cup_reaching_obs':
    world_link = None
    base_link_bag = base_link  # name of the base frame in the bag file (if it has been remapped,  otherwise set to base_link)
    tracked_frames = ['cup']  # track frames bag file should have this!
else:
    world_link = None
    base_link_bag = base_link
    tracked_frames = ['drawer']
    print("WARNING! Skill config doesnt exist!")

# ----------------------------------------
# Smoothing pararms

align_flag = False
dtw_source_idx = 3
crop_tol = 1e-3
monotonic_tol = 1e-3
smooth_window_size = 21
num_samples = 250  # 200
start_cut = 10
end_cut = 5

# ------------------------------

package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
data_root = os.path.join(package_path, 'data')

dataset_dir = os.path.join(data_root, robot_name, skill_name)
file_name = skill_name + '.pickle'
file_path = os.path.join(data_root, robot_name, file_name)

with open(file_path, 'rb') as handle:
    dataset = pickle.load(handle)

# -----------------------------------------
# loading joint trajectories

num_demos = len(dataset)
dataset_smooth = dict()

joint_names = dataset[0]['joint_state']['name']
joint_traj_list = []

for i in range(num_demos):
    joint_traj_list.append(np.array(dataset[i]['joint_state']['position'])[:, joint_idx])

joint_traj_list_raw = joint_traj_list

if skill_name == 'reaching_two_cylinder':
    joint_traj_list_raw = crop_trajectories(joint_traj_list_raw, tol=crop_tol)

joint_traj_list = crop_trajectories(joint_traj_list, tol=crop_tol)
joint_traj_list = trim_trajectories(joint_traj_list, start_cut, end_cut)
joint_traj_list = monotonize_trajectories(joint_traj_list, tol=monotonic_tol)

if align_flag:
    joint_traj_list = align_trajectories(joint_traj_list,
                                         source_idx=dtw_source_idx)  # this also resamples the trajectories

joint_traj_list = smooth_trajectories(joint_traj_list, window_size=smooth_window_size)
joint_traj_list_resampled = resample_trajectories(joint_traj_list, num_samples=num_samples)

raw_time_list = []
time_list = []
for i in range(num_demos):
    dt = dataset[i]['time'][1] - dataset[i]['time'][0]  # assuming uniform time interval
    num_samples_old = joint_traj_list[i].shape[0]
    dt_new = dt * num_samples_old / num_samples
    time = dt_new * np.arange(0, num_samples).reshape(-1, 1)
    time_old = dt * np.arange(0, num_samples_old).reshape(-1, 1)
    raw_time_list.append(time_old)
    time_list.append(time)

joint_traj_list = joint_traj_list_resampled

dataset_smooth['time'] = time_list
dataset_smooth['joint_pos'] = joint_traj_list
dataset_smooth['joint_names'] = joint_names
dataset_smooth['transforms'] = dict()

dataset_raw = dict()
dataset_raw['time'] = raw_time_list
dataset_raw['joint_names'] = joint_names
dataset_raw['transforms'] = dict()
dataset_raw['joint_pos'] = joint_traj_list_raw

# ----------------------------------------
# Finding link trajectories wrt base using forward kinematics


robot_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'urdf'))
robot = Robot(urdf_path=os.path.join(robot_dir, robot_urdf_name))
link_names = robot.link_names
num_links = len(link_names)

dataset_smooth[base_link] = OrderedDict()
dataset_raw[base_link] = OrderedDict()
for i in range(num_links):
    link_task_map = robot.task_maps[link_names[i]]
    link_traj_list = [link_task_map.psi(torch.from_numpy(joint_traj).to(torch.get_default_dtype())).numpy()
                      for joint_traj in joint_traj_list]

    dataset_smooth[base_link][link_names[i]] = link_traj_list

    link_traj_list_raw = [link_task_map.psi(torch.from_numpy(joint_traj).to(torch.get_default_dtype())).numpy()
                      for joint_traj in joint_traj_list_raw]
    dataset_raw[base_link][link_names[i]] = link_traj_list_raw

# ----------------------------------------
# Transforming data to other links

tracked_frame = 'virtual_obj' # object frame at the last points of trajectories
dataset_smooth[tracked_frame] = OrderedDict()
dataset_smooth['transforms'][tracked_frame] = []

R = np.eye(3)
T_base_to_drawer = np.zeros((4, 4))
T_base_to_drawer[-1, -1] = 1.0
T_base_to_drawer[0:3, 0:3] = R

for j in range(num_links):
    link_name = link_names[j]
    link_traj_list = []

    for i in range(num_demos):
        t = dataset_smooth[base_link][eef_name][i][-1,
            :]  # taking the final position of the robot as the drawer position
        T_base_to_drawer[0:3, -1] = t
        T_drawer_to_base = transform_inv(T_base_to_drawer)

        link_base_traj = dataset_smooth[base_link][link_name][i]
        link_traj = transform_traj(link_base_traj, T_drawer_to_base)
        link_traj_list.append(link_traj)

        if j == 0:
            dataset_smooth['transforms'][tracked_frame].append(T_base_to_drawer)

    dataset_smooth[tracked_frame][link_name] = link_traj_list



# ---------------------------------------

# TODO: THIS USES ONLY THE FIRST FRAME OF THE TRACKED OBJECT

if world_link is not None:
    T_world_to_tracked_link_list = []
    dataset_smooth[world_link] = OrderedDict()
    for j in range(num_links):
        # transform each link trajectory from base to world frame
        link_name = link_names[j]
        link_traj_list = []

        for i in range(num_demos):
            T_world_to_base = dataset[i][base_link_bag][0]  # TODO: NOTE: Picking only the first transform!
            link_base_traj = dataset_smooth[base_link][link_name][i]
            link_traj = transform_traj(link_base_traj, T_world_to_base)
            link_traj_list.append(link_traj)

        dataset_smooth['world'][link_name] = link_traj_list

    for k in range(len(tracked_frames)):
        # transform each link trajectory from base to tracked frame (e.g. door handle)
        tracked_frame = tracked_frames[k]
        dataset_smooth[tracked_frame] = OrderedDict()
        dataset_smooth[world_link][tracked_frame] = []

        dataset_raw[tracked_frame] = OrderedDict()
        # dataset_raw[world_link][tracked_frame] = []

        dataset_raw['transforms'][tracked_frame] = []
        dataset_smooth['transforms'][tracked_frame] = []

        for j in range(num_links):
            link_name = link_names[j]
            link_traj_list = []
            link_traj_list_raw = []
            for i in range(num_demos):
                T_world_to_base = dataset[i][base_link_bag][0]
                T_world_to_frame = dataset[i][tracked_frame][0]  # TODO: NOTE: Picking only the first transform!
                T_frame_to_world = transform_inv(T_world_to_frame)
                T_frame_to_base = np.dot(T_frame_to_world, T_world_to_base)
                T_base_to_frame = transform_inv(T_frame_to_base)

                link_base_traj = dataset_smooth[base_link][link_name][i]
                link_traj = transform_traj(link_base_traj, T_frame_to_base)
                link_traj_list.append(link_traj)

                link_base_traj_raw = dataset_raw[base_link][link_name][i]
                link_traj_raw = transform_traj(link_base_traj_raw, T_frame_to_base)
                link_traj_list_raw.append(link_traj_raw)

                if j == 0:
                    dataset_smooth['transforms'][tracked_frame].append(T_base_to_frame)
                    dataset_raw['transforms'][tracked_frame].append(T_base_to_frame)

            dataset_smooth[tracked_frame][link_name] = link_traj_list
            dataset_raw[tracked_frame][link_name] = link_traj_list_raw

if skill_name == 'drawer_closing' and robot_name == 'franka':
    tracked_frame = tracked_frames[0]  # drawer only
    dataset_smooth[tracked_frame] = OrderedDict()
    dataset_smooth['transforms'][tracked_frame] = []

    R = np.eye(3)
    T_base_to_drawer = np.zeros((4, 4))
    T_base_to_drawer[-1, -1] = 1.0
    T_base_to_drawer[0:3, 0:3] = R

    for j in range(num_links):
        link_name = link_names[j]
        link_traj_list = []

        for i in range(num_demos):
            t = dataset_smooth[base_link]['right_gripper'][i][-1,
                :]  # taking the final position of the robot as the drawer position
            T_base_to_drawer[0:3, -1] = t
            T_drawer_to_base = transform_inv(T_base_to_drawer)

            link_base_traj = dataset_smooth[base_link][link_name][i]
            link_traj = transform_traj(link_base_traj, T_drawer_to_base)
            link_traj_list.append(link_traj)

            if j == 0:
                dataset_smooth['transforms'][tracked_frame].append(T_base_to_drawer)

        dataset_smooth[tracked_frame][link_name] = link_traj_list

# --------------------------------------------------
outfile_name = skill_name + '_raw.pickle'
outfile_path = os.path.join(data_root, robot_name, outfile_name)
with open(outfile_path, 'wb') as handle:
    pickle.dump(dataset_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

outfile_name = skill_name + '_smooth.pickle'
outfile_path = os.path.join(data_root, robot_name, outfile_name)
with open(outfile_path, 'wb') as handle:
    pickle.dump(dataset_smooth, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ---------------------------------------
# Plotting
#
eef_traj_list = dataset_smooth[base_link][eef_name]

axs_handles = plot_traj_time(dataset_smooth['time'][0], dataset_smooth[base_link][eef_name][0][:, 0:3], '--', 'k')
for n in range(1, num_demos):
    axs_handles = plot_traj_time(dataset_smooth['time'][n], dataset_smooth[base_link][eef_name][n][:, 0:3], '--', 'k',
                                 axs_handles=axs_handles)

# axs_handles = plot_traj_time(dataset_raw['time'][0], dataset_raw[tracked_frames[0]][eef_name][0][:,0:3], '--', 'k')
# for n in range(1,num_demos):
# 	axs_handles = plot_traj_time(dataset_raw['time'][n], dataset_raw[tracked_frames[0]][eef_name][n][:,0:3], '--', 'k', axs_handles=axs_handles)

# plot in 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
# eef_traj_list_T = [traj.T for traj in eef_traj_list]
plot_trajectories_3D(eef_traj_list, '-', 'b')
plt.show()
