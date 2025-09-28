function [x0, lambda0, row, col] = nearest_defective_heuristic_starting_point(A, lambda0, k)
% find a starting point for nearest_defective problems
% following [Alam, Bora, Byers, Overton '11], uses
% eigenvalue condition numbers
%
%  [x0, lambda0] = nearest_defective_heuristic_starting_point(A)
%  return x0 for the best lambda according to the ABBO heuristic
%
%  [x0, lambda0] = nearest_defective_heuristic_starting_point(A, []. 10)
%  return the 10th best x0
%
%  [x0, lambda0] = nearest_defective_heuristic_starting_point(A, 3.5)
%  uses a fixed lambda=3.5
%
%  [x0, lambda0] = nearest_defective_heuristic_starting_point(A, [3 4])
%  uses the eigenvalue pair ev(3), ev(4) from ev = eig(A)

n = length(A);

if not(exist('lambda0', 'var')) || isempty(lambda0)
    if not(exist('k', 'var'))
        k = 1;
    end
    [V,D,c] = condeig(A);
    lambdas = diag(D);
    % find the nearest pair of eigenvalues
    distmatrix = abs(lambdas - transpose(lambdas));
    scaleddist = distmatrix ./ (c+c');
    [I,J,V] = find(distmatrix);
    tokeep = I<J;
    I = I(tokeep); J = J(tokeep); V = V(tokeep);
    [~, perm] = sort(V, 'ascend');
    I = I(perm); J = J(perm); V = V(perm);
    row = I(k);
    col = J(k);
    lambda0 = (c(col)*lambdas(row)+c(row)*lambdas(col)) / (c(col)+c(row));
elseif length(lambda0)==2
    % user specified a pair of eigenvalues
    [V,D,c] = condeig(A);
    lambdas = diag(D);
    row = lambda0(1); col = lambda0(2);
    lambda0 = (c(col)*lambdas(row)+c(row)*lambdas(col)) / (c(col)+c(row));
else
    % lambda0 provided
    row = nan; col = nan;
end
[U, ~, V] = svd(A - lambda0*eye(n));
[x0, ~] = qr([U(:,end), V(:,end)], 0);