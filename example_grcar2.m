%
% Variant of example_grcar, in which A is given via basis coefficients.
%

n = 8;
% A is given via coordinates in the basis P
A = [-sqrt(n-1); sqrt(n); sqrt(n-1); sqrt(n-2); sqrt(n-3)];

P = [];
P(:,:,1) = diag(ones(n-1,1), -1);
P(:,:,2) = diag(ones(n,1), 0);
P(:,:,3) = diag(ones(n-1,1), 1);
P(:,:,4) = diag(ones(n-2,1), 2);
P(:,:,5) = diag(ones(n-3,1), 3);

for k = 1:size(P,3)
    P(:,:,k) = P(:,:,k) / norm(P(:,:,k), 'fro');
end

problem = nearest_singular_structured_dense(P, A, true);

% regproblem = apply_regularization(problem, 0.2, randn(size(A,1),1));
% checkgradient(regproblem)
% checkhessian(regproblem)

options = struct();
options.max_outer_iterations = 30;
options.verbosity = 0;
options.y = 0;
[x, cost, info, results] = penalty_method(problem, [], options);
regproblem = apply_regularization(problem, info.last_epsilon, info.y);
[Delta, AplusDelta, store] = regproblem.minimizer(x, struct());

fprintf('Cost: %e, sigma_min: %e\n', cost, min(svd(AplusDelta)));