function [alpha, R] = basis_coordinates(P, A)
% compute coordinates of A in a perturbation basis
%
% [alpha, R] = basis_coordinates(P, A)
%
% R is the remainder

[m, n, p] = size(P);

alpha = zeros(p, 1);
for k = 1:p
    alpha(k) = trace(P(:,:,k)' * A);
end
R = A;
for k = 1:p
    R = R - alpha(k) * P(:,:,k);
end