import torch

class IPT:

    def __init__(self, matrix, diagonal = None, device = 'cpu', dtype = torch.double):
        if torch.is_tensor(matrix):
            self.matrix = matrix
        else:
            self.matrix = torch.tensor(matrix, device = device, dtype = dtype)
        if torch.is_tensor(diagonal):
            self.diagonal = diagonal
        else:
            self.diagonal = torch.diag(self.matrix)
        self.dtype = dtype
        self.device = device

    def eig(self, V0 = None, max_iter = 1000, return_eigenvalues = True, normalized_eigenvectors = True):

        M = self.matrix
        d = self.diagonal

        def F(V, id, theta, delta):
            n = V.size()[0]
            deltaV = torch.matmul(delta, V)
            W = deltaV - torch.matmul(V, torch.diag(torch.diag(deltaV)))
            return(id - torch.mul(theta, W))


        theta = torch.triu(1/(d.unsqueeze(1) - d), diagonal = 1)
        theta = theta - theta.t()

        delta = M - torch.diag(d)

        id = torch.eye(M.size(0), dtype = M.dtype, device = M.device)

        if V0 == None:
            V = id
        else:
            V = V0

        err = 1
        it = 0

        while err > 2*torch.finfo(M.dtype).eps and it < max_iter:


            it += 1
            W = F(V, id, theta, delta)
            err = torch.max(torch.abs(W-V))/torch.max(torch.abs(V))
            V = W

        D = torch.diag(d + torch.diag(torch.matmul(delta, V)))
        if normalized_eigenvectors:
            V = V / torch.linalg.norm(V, dim = 1)

        print('Iterations IPT:', it)
        print('Residual IPT:', torch.linalg.norm(M @ V - V @ D).item())

        if return_eigenvalues:
            return(D, V)
        else:
            return(V)

    def eigs(self, k = 0, order = 1000, d = None, v0 = None, trace = False):

        M = self.matrix
        D = self.diagonal

        def f(v, e, t, delta):
            deltav = torch.matmul(delta, v)
            return(e + torch.mul(t, deltav - deltav[k]*v))

        n = M.size(0)

        delta = M - torch.diag(D)
        t = 1/(D[k] - D)
        t[k] = 0
        e = torch.zeros(n, dtype = M.dtype, device = M.device)
        e[k]  = 1

        if v0 == None:
            v = e
        else:
            v = v0

        err = 1
        l = 0
        d = []
        while err > 2*torch.finfo(M.dtype).eps and l <= order:
            # On the GPU the error condition is the time bottleneck
            l += 1
            w = f(v, e, t, delta)
            err = torch.max(torch.abs(w-v))
            v = w
            d.append(D[k] + torch.matmul(delta, v)[k])
        print('Iterations DPT:',  l)
        print('Residual DPT:', torch.linalg.norm(M @ v - d[-1] * v).item())

        if trace:
            return(d[-1], v, d)
        else:
            return(d[-1], v)

    def test(self):

        M = self.matrix
        d = self.diagonal
        n = M.size(0)

        theta = torch.triu(1/(d.unsqueeze(1) - d), diagonal = 1)
        theta = theta - theta.t()

        delta = M - torch.diag(d)

        b = torch.linalg.norm(delta, 'fro')*torch.linalg.norm(theta, 'fro')/n
        print('Bound:', b.item())
        if b > .5:
            return(False)
        else:
            return(True)

    def eigh(self, approximate_eigenvectors = None):
        ## assuming orthogonal eigenvectors
        # if approximate_eigenvectors == None:
        #     Q = torch.eye(self.matrix.size(0), device = self.device, dtype = self.matrix.dtype)
        #     rotated = self.matrix
        if not torch.is_tensor(approximate_eigenvectors):
            Q = torch.tensor(approximate_eigenvectors, device = self.device, dtype = self.matrix.dtype)
            rotated = Q.t() @ self.matrix @ Q
        else:
            Q = approximate_eigenvectors
            rotated = Q.t() @ self.matrix @ Q
        ipt = IPT(rotated, dtype = self.dtype)
        test = ipt.test()
        if test:
            print('Test passed, using IPT.')
            D, V = ipt.eig()
        else:
            print('Test failed, using SYEV.')
            D, V = ipt.syev()
        return(D, Q @ V)

    def syev(self):
        D, V = torch.symeig(self.matrix, eigenvectors = True)
        return(D, V)

class RSPT:

    def __init__(self, matrix, diagonal = None):
        if torch.is_tensor(matrix):
            self.matrix = matrix
        else:
            self.matrix = torch.tensor(matrix)
        if torch.is_tensor(diagonal):
            self.diagonal = diagonal
        else:
            self.diagonal = torch.diag(self.matrix)

    def eigs(self, k = 0, v0 = None, order = 500, trace = False):

        M = self.matrix
        D = self.diagonal
        n = M.size(0)

        delta = M - torch.diag(D)
        t = 1/(D - D[k])
        t[k] = 0

        e = torch.zeros(n, dtype = M.dtype, device = M.device)
        e[k]  = 1

        if v0 == None:
            a = e
        else:
            a = v0

        a_list=[a]
        v = a
        l = 0

        d = []
        while torch.linalg.norm(a) > 2*torch.finfo(M.dtype).eps and l <= order:
            # On the GPU the error condition is the time bottleneck
            l += 1
            a = torch.zeros(M.size(0), dtype = M.dtype, device = M.device)
            for s in range(l):
                a = a + a_list[l-1-s] * (delta @ a_list[s])[k]
            a = a - delta @ a_list[l-1]
            a = torch.mul(t, a)
            a_list.append(a)

            v = v + a
            d.append(D[k] + torch.matmul(delta, v)[k])
        print('Residual RSPT:', torch.linalg.norm(M @ v - d[-1] * v).item())

        if trace:
            return(d[-1], v, d)
        else:
            return(d[-1], v)

class Partitioning:

    def __init__(self, matrix, diagonal = None):
        if torch.is_tensor(matrix):
            self.matrix = matrix
        else:
            self.matrix = torch.tensor(matrix)
        if torch.is_tensor(diagonal):
            self.diagonal = diagonal
        else:
            self.diagonal = torch.diag(self.matrix)

    def QW(self, k = 0):
        #Mihalka, Szabados, Surjan, J. Chem. Phys. 146, 124121 (2017)

        D = self.diagonal
        W = self.matrix - torch.diag(self.diagonal)

        num = torch.diag(W @ W) + torch.mul(torch.diag(W), D-D[k])
        denom = torch.diag(W) + D-D[k]
        eta = torch.div(num, denom)
        eta[k] = 0
        return(D + eta)

    # def optimized(self, k = 0):
    #     # Szabados-Surjan, Chem. Phys. Lett. 308 (1999) 303-309
    #     D = self.diagonal
    #     W = self.matrix - torch.diag(self.diagonal)
    #
    #     A = torch.mul(torch.eye(D.size(0), device = D.device, dtype = D.dtype), D-D[k]-W[k,k])
    #     A = A + torch.diag(1/W[k, :]) @ W @ torch.diag(W[:,k])
    #     Delta_inv, LU = torch.solve(torch.ones(D.size(), device = D.device, dtype = D.dtype), A)
    #     Delta = 1/Delta_inv
    #     Delta[k] = 0
    #     return(Delta - D-D[k])
