import torch
import time
import ipt
import cupy as cp
import sys

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
S = R + R.t()

print('')
print('--------- SYMMETRIC MATRIX ---------')
print('')
print('--- IPT ---')


begin = time.time()

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    D, V = ipt.IPT(S).eig()
print(prof.key_averages().table(sort_by='cuda_time_total',  row_limit=3))
end = time.time()

res = torch.linalg.norm(S @ V - V @ D)/torch.linalg.norm(S)
print('Residual: ', res.item())
print('Time: ', end - begin)

print('')
print('--- SYEV (via pytorch) ---')



begin = time.time()
D, V = torch.symeig(S, eigenvectors = True)
end = time.time()



D = torch.diag(D)
res = torch.linalg.norm(S @ V - V @ D)/torch.linalg.norm(S)
print('Residual: ', res.item())
print('Time: ', end - begin)


print('')
print('--- syevj (via cupy) ---')
S_cp = cp.asarray(S.cpu().numpy())
d, V = cp.cusolver.syevj(S_cp[:10, :10], with_eigen_vector = True)

torch.cuda.synchronize()
begin = time.perf_counter()
d, V = cp.cusolver.syevj(S_cp, with_eigen_vector = True)
end = time.perf_counter()
torch.cuda.synchronize()


d, V = torch.as_tensor(d, device = device), torch.as_tensor(V, device = device)
D = torch.diag(d)
res = torch.linalg.norm(S @ V - V @ D)/torch.linalg.norm(S)
print('Residual: ', res.item())
print('Time:', end-begin)


print('')
print('--------- NON-SYMMETRIC MATRIX ---------')
print('')


print('--- IPT ---')

begin = time.time()

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    D, V = ipt.IPT(R).eig()
print(prof.key_averages().table(sort_by='cuda_time_total',  row_limit=3))
end = time.time()


res = torch.linalg.norm(S @ V - V @ D)/torch.linalg.norm(S)
print('Residual: ', res.item())
print('Time: ', end - begin)

print('')
print('--- GEEV (via pytorch) ---')




begin = time.time()
D, V = torch.eig(R, eigenvectors = True)
end = time.time()




print('Time: ', end - begin)
