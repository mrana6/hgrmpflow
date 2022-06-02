from differentiable_rmpflow.learning.utils import *
from differentiable_rmpflow.learning.controllers import ScalarMetric, MetricCholNet
from differentiable_rmpflow.learning.kinematics import EuclideanizingFlow
from differentiable_rmpflow.learning.utils import Params

from differentiable_rmpflow.rmpflow.controllers import LogCoshPotential, NaturalGradientDescentMomentumController, IdentityMetric
from differentiable_rmpflow.rmpflow.rmptree import RmpTreeNode

import os
import torch
torch.set_default_dtype(torch.float32)
import torch.optim as optim


import numpy as np
import time
import scipy.io as sio


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

				# Dagger params
				n_dagger_iterations 	= 1,  		# number of dagger iterations ( NOTE: regular training if no dagger used)
				tracker_kp 				= 2.,  		# proportional gain for tracking controller
				split_ratio 			= 0.32,  	# ratio of original out of aggregated dataset
				)


load_saved_model = False
save_models = True
plot_models = True


# params
dataset_name = 'lasa_handwriting_dataset_v2'
data_name = 'heee'


# -----------------------------------------

models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'models'))
models_path = os.path.join(models_dir, dataset_name)
if save_models and not os.path.exists(models_path):
	print('create directory: {}'.format(models_path))
	os.makedirs(models_path)

# ------------------------------------------------

package_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
data_root = os.path.join(package_path, 'data')

file_name = data_name + '.mat'
file_path = os.path.join(data_root, dataset_name, file_name)

data = sio.loadmat(file_path)
data = data['demos']
n_demos = int(data.shape[1])
n_dims = data[0, 0]['pos'][0][0].shape[0]

demo_traj_list = [data[0, i]['pos'][0][0].T for i in range(n_demos)]
dt = data[0, 0]['t'][0][0][0,1] - data[0, 0]['t'][0][0][0, 0]
dt = np.round(dt * PARAMS.downsample_rate, 2)

torch_traj_datasets = preprocess_dataset(demo_traj_list, dt=dt,
										 start_cut=PARAMS.start_cut,
										 end_cut=PARAMS.end_cut,
										 downsample_rate=PARAMS.downsample_rate,
										 smoothing_window_size=PARAMS.smoothing_window_size,
										 vel_thresh=1.,
										 goal_at_origin = True)

train_dataset = ConcatDataset(torch_traj_datasets)

# normalization factors
minx, maxx = find_data_range(torch_traj_datasets, idx=0)
xrange = (maxx - minx)
scaling = 1. / xrange
translation = -minx / xrange - 0.5


# --------------------------------------------------------------------------------
# Learner setup

latent_taskmap = EuclideanizingFlow(n_inputs=n_dims, n_blocks=PARAMS.n_blocks_flow, n_hidden=PARAMS.n_hidden_flow,
								 s_act=PARAMS.s_act_flow, t_act=PARAMS.t_act_flow,
								 sigma=PARAMS.sigma_flow,
								 flow_type=PARAMS.flow_type,
								 coupling_network_type=PARAMS.coupling_network_type,
								 goal=None,
								 normalization_scaling=scaling,
								 normalization_bias=translation)


latent_pot_fn = LogCoshPotential()
# latent_metric_fn = IdentityMetric()
latent_metric_fn = ScalarMetric(n_dims=2, n_hidden=100)
# latent_metric_fn = MetricCholNet(n_dims=n_dims, n_hidden_1=PARAMS.n_hidden_1,
# 								 n_hidden_2=PARAMS.n_hidden_2, return_cholesky=False)
latent_rmp = NaturalGradientDescentMomentumController(G=latent_metric_fn, del_Phi=latent_pot_fn.grad)
learner_model = RmpTreeNode(n_dim=2, order=1, return_natural=False)
latent_space = learner_model.add_task_space(task_map=latent_taskmap)
latent_space.add_rmp(latent_rmp)
learner_model.train()

# x_train = torch.cat([dataset_.tensors[0] for dataset_ in torch_traj_datasets], dim=0)
# y = learner_model(x_train)
# ------------------------------------------------------------------------------
# Training learner
optimizer = optim.Adam(learner_model.parameters(), lr=PARAMS.learning_rate, weight_decay=PARAMS.weight_decay)
criterion = nn.SmoothL1Loss()
loss_fn = criterion

if load_saved_model:
	print('--------------------------------------------')
	print('------------------loading-------------------')
	print('--------------------------------------------')
	model_filename = '{}.pt'.format(data_name)
	learner_model.load_state_dict(torch.load(os.path.join(models_path, model_filename)))
else:
	print('--------------------------------------------')
	print('------------------training------------------')
	print('--------------------------------------------')

	t_start = time.time()

	learner_model.train()
	best_model, best_traj_loss = \
		train(model=learner_model, loss_fn=loss_fn, opt=optimizer, train_dataset=train_dataset,
			  n_epochs=PARAMS.n_epochs, batch_size=PARAMS.batch_size,
			  stop_threshold=PARAMS.stop_threshold, shuffle=False)
	print('time elapsed: {} seconds'.format(time.time() - t_start))
	print('\n')

if save_models:
	print('--------------------------------------------')
	print('------------------Saving------------------')
	print('--------------------------------------------')

	model_filename = '{}.pt'.format(data_name)
	torch.save(learner_model.state_dict(), os.path.join(models_path, model_filename))


if plot_models:
	print('--------------------------------------------')
	print('-------------------testing------------------')
	print('--------------------------------------------')

	learner_model.eval()
	#  ------------------------------------------
	# defing plot limits

	minx = minx.numpy()
	maxx = maxx.numpy()
	xrange = xrange.numpy()
	if n_dims == 2:
		x_lim = [[minx[0, 0] - xrange[0, 0] / 10., maxx[0, 0] + xrange[0, 0] / 10.],
				 [minx[0, 1] - xrange[0, 1] / 10., maxx[0, 1] + xrange[0, 1] / 10.]]

		fig = plt.figure()

		ax = plt.gca()
		ax.set_xlim(x_lim[0])
		ax.set_ylim(x_lim[1])
		plt.xticks([])
		plt.yticks([])
		plt.tight_layout()
		visualize_vel(learner_model, x_lim=x_lim, delta=0.5, cmap=None, color='0.4')

		for torch_dataset in torch_traj_datasets:
			t_final = dt * (torch_dataset.tensors[0].shape[0] - 1)
			visualize_training_set(
				learner_model,
				torch_dataset.tensors[0],
				order=1,
				n_samples=1,
				t_final=t_final,
				t_step=dt,
				integration_method='euler'
			)

	plt.show()
