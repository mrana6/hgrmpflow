import time
import torch.autograd.functional
import torch.autograd as autograd


def Jacobian(f, x, n_inputs, n_outputs):
    n = x.size()[0]
    x_m = x.repeat(1, n_outputs).view(-1, n_inputs)
    x_m.requires_grad_(True)
    y_m = f(x_m)
    if y_m.requires_grad:
        mask = torch.eye(n_outputs).repeat(n, 1)
        J = torch.autograd.grad(y_m, x_m, mask, create_graph=True, allow_unused=True)[0]
        if J is None:
            J = torch.zeros(n, n_outputs, n_inputs)
        else:
            J = J.reshape(n, n_outputs, n_inputs)
    else:  # if requires grad is False, then output has no dependence of input
        J = torch.zeros(n, n_outputs, n_inputs)

    return J

alpha = 1e-2
radius = 0.2
goal = torch.zeros(1, 2)

def psi_map(x):
	stretch = torch.tanh(alpha * (torch.norm(x, dim=1) / radius - 1.0)).reshape(-1,1) * x
	return stretch

def psi(x):
	return psi_map(x) - psi_map(goal)

# psi = lambda x: 0.5*(torch.norm(x, dim=1)**2).reshape(-1, 1)

N = 10
x = torch.rand(N, 2)
K = 1000
n_outputs = 2
n_inputs = 2

# -----------------------------------------------------------------

t0 = time.time()
for _ in range(K):
	J_ = torch.autograd.functional.jacobian(psi, x, create_graph=True)
	idx = torch.arange(N)
	J1 = J_[idx, :, idx, :]
tf = time.time()
print('avg. time for torch jacobian: {}'.format((tf-t0)/K))

# ------------------------------------------------------------------

t0 = time.time()
for _ in range(K):
	J = Jacobian(psi, x, n_inputs=n_inputs, n_outputs=n_outputs)
tf = time.time()
print('avg. time for self-defined jacobian: {}'.format((tf-t0)/K))



