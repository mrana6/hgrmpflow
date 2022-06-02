import torch
import torch.autograd as autograd
import time
from differentiable_rmpflow.rmpflow import TaskMap


# model = nn.Sequential()
# model.add_module('W0', nn.Linear(8, 16))
# model.add_module('tanh', nn.Tanh())
# model.add_module('W1', nn.Linear(16, 1))
#
# # x = torch.randn(1,8)
#
# make_dot(model, params=dict(model.named_parameters()))




seed = 1

torch.manual_seed(seed)

n_inputs = 2
n_outputs = 2
device = 'cpu'

psi = lambda x: x/torch.norm(x, dim=1).reshape(-1,1)

x = torch.rand(10, 2, device=device)
xd = torch.ones(10, 2, device=device)

taskmap = TaskMap(n_inputs=n_inputs, n_outputs=n_outputs, psi=psi)
# time.sleep(2)
y = psi(x)
N = 1000

t0 = time.time()
for i in range(N):
	n = x.size()[0]
	x_m = x.repeat(1, n_outputs).view(-1, n_inputs)
	x_m.requires_grad_(True)
	y_m = psi(x_m)
	y = y_m[::n_outputs, :].detach()

	if y_m.requires_grad:
		mask = torch.eye(n_outputs, device=device).repeat(n, 1)
		J, = autograd.grad(y_m, x_m, mask, create_graph=True, allow_unused=True, retain_graph=True)
		# make_dot(J).view()
		if J is None:
			J = torch.zeros(n, n_outputs, n_inputs, device=device)
		else:
			J = J.reshape(n, n_outputs, n_inputs)
	else:  # if requires grad is False, then output has no dependence of input
		J = torch.zeros(n, n_outputs, n_inputs, device=device)

	Jd_ = torch.zeros(n, n_outputs, n_inputs, device=device)
	if J.requires_grad:  # if requires grad is False, then J has no dependence of input
		# Finding jacobian of each column and applying chain rule
		for i in range(n_inputs):
			Ji = J[:, :, i]
			mask = torch.eye(n_outputs, n_inputs, device=device).repeat(n, 1)
			Ji_m = Ji.repeat(1, n_inputs).view(-1, n_inputs)
			Ji_dx = autograd.grad(Ji_m, x_m, mask, create_graph=False, retain_graph=True)[0]
			Ji_dx = Ji_dx.reshape(n, n_outputs, n_inputs)
			Jd_[:, :, i] = torch.einsum('bij,bj->bi', Ji_dx, xd)
		Jd_ = Jd_.detach()
	J.detach()
	Jd_.detach()
tf = time.time()
print('Avg compute time: {}'.format((tf-t0)/N))

# ----------------------------------------------------------------



def J_analytic(x):
	x_norm = torch.norm(x, dim=1).reshape(-1,1)
	I = torch.eye(x.shape[1]).repeat(x.shape[0],1,1)
	J = torch.diag_embed(1./x_norm)*I - torch.diag_embed(1./x_norm**3)*torch.einsum('bi,bj->bij',x,x)
	return J

J_a = J_analytic(x)

n = x.size()[0]
x_m = x.repeat(1, n_outputs * n_inputs).view(-1, n_inputs)
x_m.requires_grad_(True)
J_m = J_analytic(x_m)
J_m_vec = J_m.reshape(-1, n_outputs * n_inputs)
mask = torch.eye(n_outputs * n_inputs, device=device).repeat(n, 1)
dJ_m_vec, = autograd.grad(J_m_vec, x_m, mask, create_graph=False, allow_unused=True)
dJ_m_xd_vec = dJ_m_vec * xd.repeat(1, n_outputs * n_inputs).view(-1, n_inputs)
dJ_m_xd = dJ_m_xd_vec.sum(dim=1)
Jd = dJ_m_xd.reshape(-1, n_outputs, n_inputs).detach()




# y = psi(x)
# N = 1000
#
# for i in range(10):
# 	t0 = time.time()
# 	taskmap(x, xd)
#
# 	tf = time.time()
# 	print('Avg compute time: {}'.format((tf-t0)))


# N = 1000
# t0 = time.time()
# n = x.size()[0]
# x_m = x.repeat(1, n_outputs * n_inputs**2).view(-1, n_inputs)
# x_m.requires_grad_(True)
# y_m = psi(x_m)
# y = y_m[::n_outputs*n_inputs**2, :].detach()
#
# if y_m.requires_grad:
# 	mask = torch.eye(n_outputs, device=device).repeat(n*n_inputs**2, 1)
# 	J_m, = autograd.grad(y_m, x_m, mask, create_graph=True, allow_unused=True, retain_graph=True)
# 	if J_m is None:
# 		J = torch.zeros(n, n_outputs, n_inputs, device=device)
# 	else:
# 		J_m = J_m.reshape(-1, n_outputs, n_inputs)
# 		J = J_m[::n_outputs*n_inputs, :].detach()
# else:  # if requires grad is False, then output has no dependence of input
# 	J = torch.zeros(n, n_outputs, n_inputs, device=device)
#
#
# J_m_vec = J_m.reshape(-1, n_outputs*n_inputs).repeat_interleave(2, dim=0)
# mask = torch.eye(n_outputs*n_inputs, device=device).repeat(n, 1)
# dJ_m_vec, = autograd.grad(J_m_vec, x_m, mask, create_graph=True, allow_unused=True, retain_graph=True)
# make_dot(dJ_m_vec).view()
# dJ_m_vec = dJ_m_vec[::n_inputs]
# dJ_m_xd_vec = dJ_m_vec * xd.repeat(1, n_outputs*n_inputs).view(-1, n_inputs)
# dJ_m_xd = dJ_m_xd_vec.sum(dim=1)
# Jd = dJ_m_xd.reshape(-1, n_outputs, n_inputs).detach()
