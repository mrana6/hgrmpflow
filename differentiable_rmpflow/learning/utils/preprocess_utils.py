from torch.utils.data import TensorDataset
from scipy.signal import savgol_filter
import torch

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import numpy as np
from scipy import interpolate
from scipy.signal import medfilt
from termcolor import colored


# ---------------------------------
def preprocess_dataset(traj_list, dt, start_cut=15, end_cut=10, downsample_rate=1,
					   smoothing_window_size=25, vel_thresh=0., goal_at_origin = False):
	n_dims = traj_list[0].shape[1]
	n_demos = len(traj_list)
	# dt = round(dt * downsample_rate, 2)
	torch_datasets = []
	for i in range(n_demos):
		demo_pos = traj_list[i]

		for j in range(n_dims):
			demo_pos[:, j] = savgol_filter(demo_pos[:, j], smoothing_window_size, 3)

		demo_pos = demo_pos[::downsample_rate, :]
		demo_vel = np.diff(demo_pos, axis=0) / dt
		demo_vel_norm = np.linalg.norm(demo_vel, axis=1)
		ind = np.where(demo_vel_norm > vel_thresh)
		demo_pos = demo_pos[np.min(ind):(np.max(ind) + 2), :]

		for j in range(n_dims):
			demo_pos[:, j] = savgol_filter(demo_pos[:, j], smoothing_window_size, 3)

		demo_pos = demo_pos[start_cut:-end_cut, :]

		if goal_at_origin:
			demo_pos = demo_pos - demo_pos[-1]

		demo_vel = np.diff(demo_pos, axis=0) / dt
		demo_vel = np.concatenate((demo_vel, np.zeros((1, n_dims))), axis=0)

		torch_datasets.append(TensorDataset(torch.from_numpy(demo_pos).to(torch.get_default_dtype()),
											torch.from_numpy(demo_vel).to(torch.get_default_dtype())))
	return torch_datasets


# -------------------------------------
def align_traj(source_traj, target_traj):
	distance, path = fastdtw(source_traj, target_traj, dist=euclidean)
	target_traj_aligned = np.zeros_like(source_traj)

	for i in range(len(path)):
		target_traj_aligned[path[i][0], :] = target_traj[path[i][1], :]

	return target_traj_aligned


def crop_traj(traj, tol):
	init = traj[0, :]
	goal = traj[-1, :]
	try:
		idx_min = np.where(np.linalg.norm(traj - init, axis=1) > tol)[0][0]
		idx_max = np.where(np.linalg.norm(traj - goal, axis=1) > tol)[0][-1]
	except IndexError:
		return np.array([])
	return traj[idx_min:idx_max, :]


def monotonize_traj(traj, tol):
	monotonic_traj = []
	monotonic_traj.append(traj[0,:])
	for n in range(1,len(traj)):
		if np.linalg.norm(traj[n,:] - monotonic_traj[-1]) > tol:
			monotonic_traj.append(traj[n,:])

	monotonic_traj = np.array(monotonic_traj)
	return monotonic_traj


def smooth_traj(traj, window_size, polyorder=3):
	smoothed_traj = np.zeros_like(traj)

	for j in range(traj.shape[1]):
		smoothed_traj[:, j] = savgol_filter(traj[:, j], window_size, polyorder)

	return smoothed_traj


def differentiate_traj(pos_traj, dt, normalize_vel=False):
	num_dim = pos_traj.shape[1]
	vel_traj = np.diff(pos_traj, axis=0) / dt
	static_threshold = 1e-6
	static_flag = np.any(np.linalg.norm(vel_traj, axis=1) <= static_threshold)

	if static_flag:
		print(colored('WARNING: Found velocities less than {}. Consider Monotonizing!'.format(static_threshold), 'red', 'on_blue'))

	if normalize_vel:
		vel_norms = np.maximum(np.linalg.norm(vel_traj, axis=1), 1e-16).reshape(-1,1)  # lower bounding vel magnitudes to 1e-15
		vel_traj = vel_traj/vel_norms

	vel_traj = np.concatenate((vel_traj, np.zeros((1, num_dim))), axis=0)
	acc_traj = np.diff(vel_traj, axis=0) / dt
	acc_traj = np.concatenate((acc_traj, np.zeros((1, num_dim))), axis=0)

	return vel_traj, acc_traj


def transform_traj(traj, transform_mat):
	'''
	Transform a 3D position trajectory using a homogenous transformation

	:param traj: Nx3 array   # Works for 3-D position trajectories only
	:param transform_mat: homogeneous transformation (list or single transform)
	:return:
	'''
	traj = np.concatenate((traj, np.ones((traj.shape[0],1))), axis=1) # appending a 1 at the end for transformation
	traj_transformed = np.zeros_like(traj)

	if type(transform_mat) is list:
		if len(transform_mat) == len(traj):
			for i in range(len(traj)):
				traj_transformed[i,:] = np.dot(transform_mat[i], traj[i,:].reshape(-1,1)).T
		else:
			print("Error! Length of transforms list != lenght of trajectory")
			return

	else:
		traj_transformed = np.dot(transform_mat, traj.T).T

	traj_transformed = traj_transformed[:,:-1]   # removing the appened 1

	return traj_transformed


def find_link_trajectory(joint_traj, fk_func, output_dim=3):
	'''
	Carries out forward kinematics
	:param joint_traj:
	:param fk_func:  forward kinematics function (can come from the task map)
	:return:
	'''

	link_traj = np.zeros((joint_traj.shape[0], output_dim))
	for i in range(joint_traj.shape[0]):
		link_traj[i, :] = fk_func(joint_traj[i, :]).flatten()

	return link_traj


def transform_inv(T):
	'''
	Inverts a rigid body transformation
	:param T:
	:return:
	'''
	T_inv = np.zeros_like(T)
	T_inv[-1,-1] = 1.0
	R = T[0:3, 0:3]
	p = T[0:3, -1].reshape(-1,1)

	T_inv[0:3, 0:3] = R.T
	T_inv[0:3, -1] = -np.dot(R.T, p).flatten()

	return T_inv


def transform_pt(pt, transform_mat):
	pt = np.concatenate((pt, np.ones((1,1))))
	pt_transformed = np.dot(transform_mat, pt)[0:3].reshape(-1,1)

	return pt_transformed


def resample_traj(traj, num_samples):
	resampled_traj = np.zeros((num_samples, traj.shape[1]))
	x = np.linspace(0, 1, traj.shape[0], endpoint=True)
	x_new = np.linspace(0, 1, num_samples, endpoint=True)

	for i in range(traj.shape[1]):
		f = interpolate.interp1d(x, traj[:,i], 'cubic')
		resampled_traj[:,i] = f(x_new)

	return resampled_traj

def trim_traj(traj, start_cut, end_cut):
	# trimmed_traj = traj[start_cut:, :]
	# trimmed_traj = trimmed_traj[:-(end_cut+1),:]

	num_pts = traj.shape[0]
	trimmed_traj = traj[start_cut:num_pts - end_cut, :]

	return trimmed_traj



def median_filter_traj(traj, kernel_size):
	filtered_traj = np.zeros_like(traj)
	for i in range(traj.shape[1]):
		filtered_traj[:,i] = medfilt(traj[:,i], kernel_size=kernel_size)

	return filtered_traj



# ------------------------------------
# Batch processing versions

def scale_trajectories(traj_list, scale_factor):
	scaled_traj_list = []
	for i in range(len(traj_list)):
		scaled_traj_list.append(scale_factor*traj_list[i])

	return scaled_traj_list


def trim_trajectories(traj_list, start_cut, end_cut):
	trimmed_traj_list = []
	for i in range(len(traj_list)):
		traj = traj_list[i]
		trimmed_traj = trim_traj(traj, start_cut, end_cut)
		trimmed_traj_list.append(trimmed_traj)

	return trimmed_traj_list


def resample_trajectories(traj_list, num_samples):
	resampled_traj_list = []
	for i in range(len(traj_list)):
		traj = traj_list[i]
		resampled_traj = resample_traj(traj, num_samples)
		resampled_traj_list.append(resampled_traj)

	return resampled_traj_list




def align_trajectories(traj_list, source_idx):
	'''
	Time aligns trajectories using dynamic time warping
	:param traj_list:
	:param source_idx:
	:return:
	'''
	aligned_traj_list = []
	source_traj = traj_list[source_idx]

	for i in range(len(traj_list)):
		if i != source_idx:
			target_traj = traj_list[i]
			target_traj_aligned = align_traj(source_traj, target_traj)

			aligned_traj_list.append(target_traj_aligned)
		else:
			aligned_traj_list.append(source_traj)

	return aligned_traj_list


def crop_trajectories(traj_list, tol):
	'''
	Removes static points at the corners of the trajectories
	:param traj_list:
	:param tol:
	:return:
	'''
	cropped_traj_list = []
	for i in range(len(traj_list)):
		traj = traj_list[i]
		cropped_traj_list.append(crop_traj(traj, tol))

	return cropped_traj_list

def smooth_trajectories(traj_list, window_size):
	'''
	Smooths using savizky golay filter
	:param traj_list:
	:param window_size:
	:return:
	'''
	smoothed_traj_list = []

	for i in range(len(traj_list)):
		traj = traj_list[i]
		smoothed_traj_list.append(smooth_traj(traj, window_size))

	return smoothed_traj_list


def monotonize_trajectories(traj_list, tol):
	monotonic_traj_list = []
	for i in range(len(traj_list)):
		traj = traj_list[i]
		monotonic_traj_list.append(monotonize_traj(traj, tol))

	return monotonic_traj_list


def find_link_trajectories(joint_traj_list, fk_func, output_dim=3):
	link_traj_list = []

	for i in range(len(joint_traj_list)):
		joint_traj = joint_traj_list[i]
		link_traj = find_link_trajectory(joint_traj, fk_func, output_dim)
		link_traj_list.append(link_traj)

	return link_traj_list


def min_dtw_traj_idx(traj_list):
	mean_dist_list = []
	for i in range(len(traj_list)):
		source_traj = traj_list[i]
		dist_list = []
		for j in range(len(traj_list)):
			if i != j:
				target_traj = traj_list[j]
				distance, _ = fastdtw(source_traj, target_traj, dist=euclidean)
				dist_list.append(distance)

		mean_dist = np.mean(np.array(dist_list))
		mean_dist_list.append(mean_dist)

	return mean_dist_list.index(min(mean_dist_list))


def transform_trajectories(traj_list, transform_mat):
	'''
	Transforms a list of trajectories given either a
		1) single transformation matrix for all trajectories OR
		2) list of transformation matrices (1 per trajectory) OR
		3) list of list of transformation matrices (1 per trajectory point)
	:param traj_list: list of trajectories
	:param transform_mat: transformation matrices
	:return: list of transformed trajectories
	'''
	transformed_traj_list = []

	if type(transform_mat) is list:
		if len(transform_mat) == len(traj_list):
			for i in range(len(traj_list)):
				traj = traj_list[i]
				transformed_traj = transform_traj(traj, transform_mat[i])
				transformed_traj_list.append(transformed_traj)
		else:
			print("Error! Length of transforms list != length of trajectory list")
			return
	else:
		for i in range(len(traj_list)):
			traj = traj_list[i]
			transformed_traj = transform_traj(traj, transform_mat)
			transformed_traj_list.append(transformed_traj)



	return transformed_traj_list


def median_filter_trajectories(traj_list, kernel_size):
	filtered_traj_list = []
	for n in range(len(traj_list)):
		filtered_traj = median_filter_traj(traj_list[n], kernel_size)
		filtered_traj_list.append(filtered_traj)

	return filtered_traj_list


def differentiate_trajectories(traj_list, dt,  normalize_vel=False):
	vel_traj_list = []
	acc_traj_list = []

	for i in range(len(traj_list)):
		pos_traj = traj_list[i]

		vel_traj, acc_traj = differentiate_traj(pos_traj, dt, normalize_vel=normalize_vel)
		vel_traj_list.append(vel_traj)
		acc_traj_list.append(acc_traj)

	return vel_traj_list, acc_traj_list


def fix_pt_trajectories(traj_list, fix_idx=None):
	if fix_idx is not None:
		fixed_pt = traj_list[0][fix_idx,:]
		traj_list_fixed = traj_list

		for i in range(len(traj_list)):
			diff = fixed_pt - traj_list[i][fix_idx,:]
			for j in range(traj_list[i].shape[0]):
				traj_list_fixed[i][j,:] = traj_list[i][j,:] + diff
	else:
		traj_list_fixed = traj_list

	return traj_list_fixed


# ------------------------------

def find_mean_goal(robot, joint_traj_list, link_name, base_to_tracked_frame_transforms=None):
	'''
	Find the goals as the mean of the end point of demos for each link.
	NOTE: If the base_to_tracked_frame_transforms is provides, the goals will be in reference of the tracked frames
	:param robot:
	:param joint_traj_list:
	:param link_names:
	:param tracked_frame_transforms:
	:return:
	'''
	n_demos = len(joint_traj_list)
	link_task_map = robot.get_task_map(target_link=link_name)

	link_traj_list_base = []
	for i in range(len(joint_traj_list)):
		joint_traj = torch.from_numpy(joint_traj_list[i]).to(dtype=torch.get_default_dtype())
		link_traj = link_task_map.psi(joint_traj).numpy()
		link_traj_list_base.append(link_traj)

	if base_to_tracked_frame_transforms is None:
		# use the base link reference frame
		leaf_traj_list = link_traj_list_base
	else:
		# inverting the transforms
		tracked_frame_to_base_transforms = \
			[transform_inv(T_base_to_frame) for T_base_to_frame in base_to_tracked_frame_transforms]

		# transforming to the tracked frame space
		leaf_traj_list = transform_trajectories(link_traj_list_base, tracked_frame_to_base_transforms)

	# Extracting goals in the leaf coordinates
	goals = np.array([traj[-1].tolist() for traj in leaf_traj_list])

	# Finding the mean in the tracked frame coordinates
	mean_goal = np.mean(goals, axis=0).reshape(1, -1)

	# find goal biases (translations)
	demo_goal_biases = []
	for n in range(n_demos):
		# adding a virtual link which ends at the goal
		bias = leaf_traj_list[n][-1].reshape(1, -1) - mean_goal
		demo_goal_biases.append(bias)

	return mean_goal, demo_goal_biases


