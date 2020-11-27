function [V, D] = IPT_full(M, tol, over)

n  = size(M,1);


%
% iteration: F(V) = I - Θ ∗ (Δ*V - diag(Δ*V)·V)
%

D = diag(diag(M));
Delta = M-D;
Theta = theta(M);

%fprintf("Bound = %d. \n", norm(Theta)*norm(Delta));

if nargin < 2, tol =  2*eps(class(gather(M))); end
if nargin < 3, over = 1e3; end

if strcmp(class(M), 'gpuArray')
    id = eye(n, 'gpuArray');
else
    id = eye(n);
end

i=0;
V = id;

%residuals = [max(vecnorm(M*V-V*D))];
error = 1;

while error > tol
    
    if error > over
        fprintf("Blow up! \n")
        break
    end
    
    i=i+1;
    
    T = Delta*V;
    T = T - V*diag(diag(T));
    T = id - Theta .* T;
    error = norm(T-V, inf)/norm(V, inf);
    V = T;
    
end

D = diag(diag(M+Delta*V));
V = V./vecnorm(V);


if error <= tol*norm(M, 'fro')
    fprintf("dPT converged in %d iterations. \n", i)
end

fprintf("Largest residual is %d. \n", max(vecnorm(M*V-V*D)))

end
