function t = theta(M)

if strcmp(class(M), 'gpuArray')
    m = gather(M);
else
    m = M;
end

D = diag(m);
n = size(m,1);

if issparse(m)
    [row, col] = find(m.*~speye(size(m)));
    g = 1 ./ ( D(row) - D(col) );
    t = sparse(row, col, g, n, n);
    t(~isfinite(t)) = 0;
else
    t=1./(D-D.');
    t(~isfinite(t)) = 0;
end

if strcmp(class(m), 'gpuArray')
    t = gpuArray(t);
end

end