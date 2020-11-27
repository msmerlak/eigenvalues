function [ev, errors] = QdagMQ(A)
M= full(A);

tic

n = size(M,1);
accuracy = 12;

i = 0;

j = 1;
l = 0;

%errors = [norm(theta(M))*norm(M-diag(diag(M)))];

ev=[];


if strcmp(class(M), 'gpuArray')
    %errors = gpuArray(errors);
    id = eye(n, 'gpuArray');
else
    id = eye(n);
end


if isreal(M) && ~ishermitian(M)
    M = M + 1i*id;
    realButNotH = true;
else
    realButNotH = false;
end



while length(M) > 1
    
    if norm(tril(M,-1),'fro')< 10^(-accuracy)*norm(tril(A,-2),'fro')
        ev = diag(M);
        %fprintf("Error is %d. \n",max(sort(real(ev))-sort(real(eig(A))))+max(sort(imag(ev))-sort(imag(eig(A)))))
        fprintf("Converged in %d iterations. \n",i)
        return
    end
    
    i = i+1;
    
    
    
    t = theta(M);
    
    if norm(t, 'fro')==0
        fprintf('theta is zero :(\n')
        return
    end
    
    q = - t.*M;
    
    if norm(q, Inf)>1
        q = 0.5*q;
    end
    

    q = id + q;
    [Q, ~] = qr(q);

    M = Q'*M*Q;

    while n > 1 && norm(M(j+1:end,1), 1) < 10^(-accuracy)
        ev(j) = M(j,j); j = j+1;
        n = n-1;
    end
    M = M(j:end, j:end);
    id=id(j:end, j:end);
    
    
%     while n > 1 && abs(M(end-l,end-l-1)) < 10^(-accuracy)
%         %M(end-l,end-l-1)=0; l = l+1;
%         ev(n) = M(end - l, end - l); l = l + 1;
%         n = n-1;
%     end
%     M = M(1:end - l, 1:end - l);
%     id=id(1:end - l, 1:end - l);
    

%         pause(.01);
%         imagesc(log(abs(M)));

end

ev(end+1) = M(end, end);

if realButNotH
    ev = ev - 1i;
end

fprintf("Converged in %d iterations. \n",i)
%fprintf("Error is %d. \n",max(sort(real(ev))-sort(real(eig(full(A)))))+max(sort(imag(ev))-sort(imag(eig(full(A))))))

toc

end