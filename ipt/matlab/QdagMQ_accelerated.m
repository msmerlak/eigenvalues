function ev = QdagMQ_accelerated(M, mMax)

n = size(M,1);



if isreal(M) && ~ishermitian(M)
for k=1:n
    M(k,k) = 1i + M(k,k);
end
    realButNotH = true;
else
    realButNotH = false;
end

m = fixedPoint(@G_QR, M(:), mMax);

M = reshape(m, [n,n]);

ev = diag(M);

if realButNotH
    ev = ev - 1i;
end

ev = sort(ev);
end