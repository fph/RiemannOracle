%
% Example script: find the nearest sparse singular matrix to 
% Matrix Market's benchmark matrix orani678.
%

A = mmread('orani678.mtx');

problem = nearest_singular_sparse([], A, true);

options = struct();
options.maxiter = 10000;
options.verbosity = 1;
options.solver = @trustregions;
% specifies an initial value for y in the augmented Lagrangian method. 
% If isempty(options.y), the vanilla penalty method is used.
options.y = zeros(size(A,1), 1);

[x, cost, info, results] = penalty_method(problem, [], options);
x_reg = problem.recover_exact(x, 1/eps); 
cost_reg = problem.cost(x_reg, struct());
fprintf('Optimal value of f_{eps} with eps=%e: %e.\n', info.last_epsilon, cost);
fprintf('Optimal value of f after inserting zeros in A*v: %e. sqrt(cost) = %e\n', cost_reg, sqrt(cost_reg));
