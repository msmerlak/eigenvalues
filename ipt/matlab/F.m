function v = F(v, Theta, Delta, m)

% Delta = offDiag(M);
% Theta = 1./(M(m,m) - diag(M).').';
% Theta(~isfinite(Theta))=0;

z = Delta*v;

v = Theta.*(z - z(m)*v);
v(m) = v(m) + 1;

end