%
% Nearest Hurwitz-unstable matrix to -grcar(6) with the same structure
% 

n = 6;
A = gallery('grcar', n);
A = -A;

P = [];
P(:,:,1) = diag(ones(n-1,1), -1);
P(:,:,2) = diag(ones(n,1), 0);
P(:,:,3) = diag(ones(n-1,1), 1);
P(:,:,4) = diag(ones(n-2,1), 2);
P(:,:,5) = diag(ones(n-3,1), 3);

for k = 1:size(P,3)
    P(:,:,k) = P(:,:,k) / norm(P(:,:,k), 'fro');
end

problem = nearest_unstable_structured_dense(P, 'Hurwitz', A);

% regproblem = apply_regularization(problem, 0.2, randn(size(A,1),1));
% checkgradient(regproblem)
% checkhessian(regproblem)

options = struct();
options.max_outer_iterations = 30;
options.maxiter = 10000;
options.verbosity = 1;
options.y = 0;
options.epsilon_decrease = 0.5;
[x, cost, info, results] = penalty_method(problem, [], options);
regproblem = apply_regularization(problem, info.last_epsilon, info.y);
[Delta, AplusDelta, store] = regproblem.minimizer(x, struct());

fprintf('Cost: %e, sigma_min: %e\n', cost, min(svd(A+Delta)));
