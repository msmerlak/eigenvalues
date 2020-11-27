function [ev, eVector] = IPT_one(M, m)

tic

precision = 10*eps;

n  = size(M,1);
v = zeros(n, 1);
v(m) = 1;


if strcmp(class(M), 'gpuArray')
    G = gather(M);
    Theta = 1./(G(m,m) - diag(G).').';
    Theta(~isfinite(Theta))=0;

    Theta = gpuArray(Theta);

    v = gpuArray(v);

else
    Theta = 1./(M(m,m) - diag(M).').';
    Theta(~isfinite(Theta))=0;
end

Delta = offDiag(M);


error = 1;
i=0;

while error > precision

    if error > 1e10
        fprintf("blow up!")
        break
    end

    i=i+1;

    z = Delta*v;

    w = Theta.*(z - z(m)*v);
    w(m) = w(m) + 1;

    error = norm(v-w, Inf);

    v = w;
end

eVector = v;
z = gather(z);
M = gather(M);
ev = M(m,m) + z(m);

fprintf("Converged in %d iterations. \n", i)
toc

end
