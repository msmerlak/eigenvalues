import torch
import time
import sys
import numpy as np

# Recursive QR from:
# https://arxiv.org/pdf/1912.05508.pdf


device = sys.argv[3]

def RMGSQR(A, cutoff = 128):
    m, n = A.size()
    if n == cutoff:
        if n == 1:
            norm = torch.linalg.norm(A)
            Q = A/norm
            R = torch.tensor([[norm]], device = device)
        else:
            if device == 'cuda':
                ## it looks like it's better to perform the tall QR on the CPU
                Q, R = torch.qr(A.cpu())
                Q = Q.cuda()
                R = R.cuda()

            else:
                Q, R = torch.qr(A)
        return(Q,R)
    else:
        q = int(n/2)
        Q1, R11 = RMGSQR(A[:,:q], cutoff)
        R12 = Q1.t() @ A[:, q:]
        Q2, R22 = RMGSQR(A[:, q:] - Q1 @ R12, cutoff)

        Q = torch.cat((Q1, Q2), dim = 1)

        Rup = torch.cat((R11, R12), dim = 1)
        Rdown = torch.cat((torch.zeros(q, q, device = device), R22), dim = 1)
        R = torch.cat((Rup, Rdown), dim = 0)
        return(Q, R)

n = int(sys.argv[1])
cutoff = float(sys.argv[2])

dtype = sys.argv[4]
if dtype == 'double':
    dtype = torch.double
elif dtype == 'float':
    dtype = torch.float

M = torch.randn(n,n, device = device, dtype = dtype)

print('--- recursive QR ---')
begin = time.time()
Q, R = RMGSQR(M, cutoff)
end = time.time()

print('Time:', end - begin)
print('Error:', torch.linalg.norm(M - Q @ R).item())
print('Orthogonality:', torch.linalg.norm(torch.eye(n,n, device = device) - Q @ Q.t()).item())


print('--- torch.qr ---')
begin = time.time()
Q, R = torch.qr(M)
end = time.time()

print('Time:', end - begin)
print('Error:', torch.linalg.norm(M - Q @ R).item())
print('Orthogonality:', torch.linalg.norm(torch.eye(n,n, device = device) - Q @ Q.t()).item())



if device == 'cpu':

    print('--- LAPACK geqrf + orgqr ---')
    begin = time.time()
    a, tau = torch.geqrf(M)
    Q = torch.orgqr(a, tau)
    end = time.time()
    print('Time:', end - begin)

    #print('--- NUMPY linalg.qr ---')
    #M = M.numpy()
    #begin = time.time()
    #Q, R = np.linalg.qr(M)
    #end = time.time()
    #print('Time:', end - begin)
