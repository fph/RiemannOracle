function T = polytoep(P, k)
% (block) Toeplitz matrix associated to the product with a (matrix) polynomial
%
% T = polytoep(P, k)
%
% Input: the coefficients P of a matrix polynomial; P(:,:,end) is the
% leading coefficient.
% If P is a vector, works with a scalar polynomial instead.
%
% Output: the block Toeplitz matrix such that vec P(.)*v(.) = T(P(.)) * vec(v(.))
% for each vector polynomial of degree k

if isvector(P)
    P = reshape(P, [1,1,length(P)]);
end

[m, n, p] = size(P);

Astack = reshape(permute(P, [1,3,2]), m*p, n);

T = zeros(m*(p+k), n*(k+1));

for i = 1:(k+1)
    T(m*(i-1)+1:m*(i+p-1), (i-1)*n+1:i*n) = Astack;
end
