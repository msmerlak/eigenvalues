function M = G(M)

if isvector(M)
    testVector = true;
    n=sqrt(length(M));
    M = reshape(M, [n,n]);
else
    testVector = false;
    n = size(M, 1);
end

    
    S = - theta(M).*M;
    
    normS = norm(S, Inf);
    
    if normS > 1
        S = 3*S/normS;
    end
    
    for k=1:n
        S(k,k) = 1 + S(k,k);
    end
    
    M = (S\M)*S;

if testVector
    M = M(:);
end

end