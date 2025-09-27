function [x0, lambda0] = nearest_defective_heuristic_starting_point(A, lambda0)
% find a starting point for nearest_defective problems
% following [Alam, Bora, Byers, Overton '11], uses
% eigenvalue condition numbers

n = length(A);

if not(exist('lambda0', 'var')) || isempty(lambda0)
    [V,D,c] = condeig(A);
    lambdas = diag(D);
    % find the nearest pair of eigenvalues
    distmatrix = abs(lambdas - transpose(lambdas));
    scaleddist = distmatrix ./ (c+c');
    distmatrix(1:n+1:end)=inf; % replaces diagonal with inf
    m = min(distmatrix);
    [row,col] = find(distmatrix==m, 1);
    lambda0 = (c(col)*lambdas(row)+c(row)*lambdas(col)) / (c(col)+c(row));
elseif length(lambda0)==2
    % we specified a pair of eigenvalues
    [V,D,c] = condeig(A);
    lambdas = diag(D);
    row = lambda0(1); col = lambda0(2);
    lambda0 = (c(col)*lambdas(row)+c(row)*lambdas(col)) / (c(col)+c(row));
end
[U, ~, V] = svd(A - lambda0*eye(n));
[x0, ~] = qr([U(:,end), V(:,end)], 0);