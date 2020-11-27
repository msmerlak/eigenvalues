import torch
import time
import perturbation_theory as pt
#from jacobi import jacobi
#import cupy as cp
import sys
import pyscf
import scipy.sparse.linalg
import numpy as np
#import cProfile, pstats, StringIO
#pr = cProfile.Profile()


n = int(sys.argv[1])
l = float(sys.argv[2])
dtype = sys.argv[3]
device = sys.argv[4]

if dtype == 'double':
    dtype = torch.double
elif dtype == 'float':
    dtype = torch.float
elif dtype == 'half':
    dtype = torch.half


# S = l*torch.ones(n, n, dtype = dtype, device = device)
# S = S - torch.diag(torch.diag(S)) + torch.diag(torch.tensor(range(n), dtype = dtype, device = device))


R = torch.diag(torch.tensor(range(n), dtype = dtype, device = device)) + l*torch.randn(n,n, dtype = dtype, device = device)
S = R #+ R.t()
QW = pt.Partitioning(S).QW()

#R0 = ev.P(10, .1, dtype).cuda()
#S0 = R0 + R0.t()
#DPT(S0).eig()

print('----------------- ARNOLDI -----------------')

begin = time.time()

#with torch.autograd.profiler.profile(use_cuda=True) as prof
e, v = scipy.sparse.linalg.eigs(S.numpy(), k = 1)
#print(prof.key_averages().table(sort_by='cuda_time_total',  row_limit=3))
print('Residual:', scipy.linalg.norm(S@v - e*v))
end = time.time()


print('Time: ', end - begin)



print('----------------- IPT -----------------')

begin = time.time()

#with torch.autograd.profiler.profile(use_cuda=True) as prof
D, V = pt.IPT(S).eigs()
#print(prof.key_averages().table(sort_by='cuda_time_total',  row_limit=3))

end = time.time()


print('Time: ', end - begin)

print('----------------- IPT with QW partiting -----------------')

begin = time.time()

#with torch.autograd.profiler.profile(use_cuda=True) as prof:
D, V = pt.IPT(S, diagonal =QW).eigs()
#print(prof.key_averages().table(sort_by='cuda_time_total',  row_limit=3))

end = time.time()


print('Time: ', end - begin)


print('----------------- DAVIDSON -----------------')

s = S.numpy()
Z0 = torch.eye(n, dtype = dtype)[:,0]
z0 = Z0.numpy()


tic = time.time()
e1, v1 = pyscf.lib.eig(lambda x: s @ x, x0 = z0, precond = np.diag(s), tol = 1e-30, lindep=1e-30)
toc = time.time()
print('Residual Davidson:', scipy.linalg.norm(s @ v1 - e1*v1))
print('Time Davidson:', toc - tic)
#
#
#
# print('----------------- RSPT -----------------')
#
# begin = time.time()
#
# #with torch.autograd.profiler.profile(use_cuda=True) as prof:
# D, V = pt.RSPT(S).eigs()
# #print(prof.key_averages().table(sort_by='cuda_time_total',  row_limit=3))
#
# end = time.time()
#
# print(D)
# print('Time: ', end - begin)
#
#
#
# print('----------------- RSPT with QW preconditioning-----------------')
# torch.cuda.synchronize()
#
# begin = time.time()
#
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     D, V = pt.RSPT(S, diagonal =QW).eigs()
# print(prof.key_averages().table(sort_by='cuda_time_total',  row_limit=3))
#
# end = time.time()
# torch.cuda.synchronize()
#
# print(D)
# print('Time: ', end - begin)
