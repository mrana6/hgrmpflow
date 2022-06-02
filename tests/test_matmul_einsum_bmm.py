import torch
import time

batch_size = 1000
d = 3
J = torch.eye(d, d).repeat(batch_size, 1, 1)
xd = torch.ones(batch_size, d)


N = 500

t0 = time.time()
for _ in range(N):
	yd = torch.matmul(xd.unsqueeze(1), J.permute(0, 2, 1)).squeeze()
tf = time.time()
print('avg. time for torch matmul: {}, total time: {}'.format((tf-t0)/N, tf-t0))


t0 = time.time()
for _ in range(N):
	yd = torch.bmm(xd.unsqueeze(1), J.permute(0, 2, 1)).squeeze()
tf = time.time()
print('avg. time for torch bmm: {}, total time: {}'.format((tf-t0)/N, tf-t0))


t0 = time.time()
for _ in range(N):
	yd = torch.einsum('bij,bj->bi', J, xd)
tf = time.time()
print('avg. time for torch einsum: {}, total time: {}'.format((tf-t0)/N, tf-t0))

