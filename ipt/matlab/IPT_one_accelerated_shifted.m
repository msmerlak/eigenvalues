function [ev, eVector] = IPT_one_accelerated_shifted(M, m, mMax, l, device)

n  = size(M,1);


if issparse(M)
    id = speye(n);
else
    id = eye(n);
end

v0 = id(:, m);

L = l*v0;
D = diag(M);


Theta = 1./(M(m,m) + l - D.').';
Theta(~isfinite(Theta))=0;
Theta(m,m) = 0;


Delta = M - diag(diag(M));
Delta(:, m) = Delta(:, m) - L;

if strcmp(device, 'cuda')
    v = fixed_point(@(w) F(w, gpuArray(Theta), gpuArray(Delta), m), gpuArray(v0), mMax, 5000);
else
    v = fixed_point(@(w) F(w, Theta, Delta, m), v0, mMax, 5000);
end


eVector = v;
z = Delta*v;
ev = M(m,m) + l + z(m);

end
