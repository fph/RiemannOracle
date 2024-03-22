% Variant of example_svd to test more cases 
%
% (1) rectangular, and (2) we use the vector of basis coefficients of A
% in the algorithm, i.e., A(:)

m = 4;
n = 3;
A = randn(m, n);

P = autobasis(reshape(1:m*n, m,n));

problem = nearest_singular_structured_dense(P, A(:), true);

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
