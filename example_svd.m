% Nearest singular matrix to a random matrix
% Simple example to test nearest_singular_structured_dense
% This is going to be slow, since it does not exploit the Kronecker structure in M.

n = 4;
A = randn(n);

P = autobasis(reshape(1:n^2, n,n));

problem = nearest_singular_structured_dense(P, A, true);

options = struct();
options.max_outer_iterations = 30;
options.verbosity = 0;
options.y = 0;
[x, cost, info] = penalty_method(problem, [], options);
regproblem = apply_regularization(problem, info.last_epsilon, info.y);
[Delta, AplusDelta, store] = regproblem.minimizer(x, struct());

cost = norm(Delta, 'fro')^2;
exact_sol = min(svd(A))^2;
fprintf('Cost: %e, sigma_min(A)^2: %e, difference = %e\n', cost, exact_sol, cost-exact_sol);
