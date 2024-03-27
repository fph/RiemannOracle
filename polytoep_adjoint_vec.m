function v = polytoep_adjoint_vec(A, d, w)
% compute T_d(A)'*w, an "adjoint convolution", without forming T_d(A)
%
% The input A comes in the form of a mxnxp array containing the
% coefficients; A(:,:,end) is the leading term.
% 
% The input w can come in the form of either W = [w1 w2... w_{k+d+1}],
% or a long vector [w1;w2;...;w_{k+d+1}], with each block of length m.

[m, n, kplus1] = size(A);
Astar = reshape(pagectranspose(A), [n,m*kplus1]);
w = reshape(w, [m*(kplus1+d),1]);
v = zeros(n*(d+1),1);
for i = 0:d
    v(i*n+1:(i+1)*n) = Astar * w(i*m+1:(i+kplus1)*m);
end
