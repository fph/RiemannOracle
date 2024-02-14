function P = perturbation_basis_from_sparse(A)
%
% Generates the P corresponding to perturbations with the same sparsity
% pattern as A.
%
% This is useful to test specialized solvers: for instance,
% optimizing on nearest_singular_sparse(P, A) should give the same
% result as
% nearest_singular_structured_dense(perturbation_basis_from_sparse(P), A);

[II, JJ, VV] = find(A);
nnz = length(VV);
[m, n] = size(A);
P = zeros(m, n, nnz);
for k = 1:nnz
    P(II(k),JJ(k), k) = 1;
end
