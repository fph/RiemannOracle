function P = autobasis(A, normalized)
% Construct a perturbation basis that preserves the structure of A
%
% Equal values are kept equal, and zeros are kept zero.
%
% If normalized==true (default), normalize to norm(P(:,:,k),'fro')==1

if not(exist('normalized', 'var')) || isempty(normalized)
    normalized = true;
end

[m, n] = size(A);
[I, J, V] = find(A);

[U, ~, IV] = unique(V);

p = length(U);

P = zeros(m, n, p);
for k = 1:length(V)
    P(I(k), J(k), IV(k)) = 1;
end
if normalized
    for k = 1:p
            P(:,:,k) = P(:,:,k) / norm(P(:,:,k), 'fro');
    end
end
