import time
import numpy as np
import torch
import torch.nn as nn


A = torch.rand(1, 2, 2)
b = torch.rand(1, 2)

class TestModel(nn.Module):
	# def __call__(self):
	# 	c = torch.einsum('bij, bj ->bi', A, b)
	# 	return c
	def forward(self):
		c = torch.einsum('bij, bj ->bi', A, b)
		return c


N = 10000


print('-------------------------------------')
print('batch mode comparisons')
print('-------------------------------------')

t0 = time.time()
for _ in range(N):
	c = torch.einsum('bij, bj ->bi', A, b)
tf = time.time()
print('avg. time for torch einsum: {}'.format((tf-t0)/N))

model = TestModel()
t0 = time.time()
for _ in range(N):
	c = model()
tf = time.time()
print('avg. time for torch nn.model: {}'.format((tf-t0)/N))

t0 = time.time()
for _ in range(N):
	c = torch.matmul(A, b.unsqueeze(2)).squeeze(2)
tf = time.time()
print('avg. time for torch matmul: {}'.format((tf-t0)/N))

A_np = A.numpy()
b_np = b.numpy()
t0 = time.time()
for _ in range(N):
	c = np.matmul(A_np, np.expand_dims(b_np, axis=2)).squeeze(axis=2)
tf = time.time()
print('avg. time for np matmul: {}'.format((tf-t0)/N))

print('-------------------------------------')
print('non-batch mode comparisons')
print('-------------------------------------')

A_ = A[0]
b_ = b[0].reshape(-1,1)
t0 = time.time()
for _ in range(N):
	c = torch.einsum('ij, jk -> ik', A_, b_)
tf = time.time()
print('avg. time for torch non-batch einsum: {}'.format((tf-t0)/N))

A_ = A[0]
b_ = b[0].reshape(-1,1)
t0 = time.time()
for _ in range(N):
	c = torch.mm(A_, b_)
tf = time.time()
print('avg. time for torch non-batch mm: {}'.format((tf-t0)/N))

A_ = A[0]
b_ = b[0].reshape(-1,1)
t0 = time.time()
for _ in range(N):
	c = torch.matmul(A_, b_)
tf = time.time()
print('avg. time for torch non-batch matmul: {}'.format((tf-t0)/N))


A_np = A.numpy()[0]
b_np = b.numpy()[0].reshape(-1, 1)
t0 = time.time()
for _ in range(N):
	c = np.dot(A_np, b_np)
tf = time.time()
print('avg. time for non-batch np dot: {}'.format((tf-t0)/N))

A_np = A.numpy()[0]
b_np = b.numpy()[0].reshape(-1, 1)
t0 = time.time()
for _ in range(N):
	c = np.matmul(A_np, b_np)
tf = time.time()
print('avg. time for non-batch np matmul: {}'.format((tf-t0)/N))