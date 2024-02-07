addpath(genpath('manopt/manopt'));

A = mmread('orani678.mtx');

problem = nearest_singular_sparse([], A, true);

options = struct();
options.maxiter = 10000;
options.verbosity = 1;
options.solver = @trustregions;
% specifies an initial value for y in the augmented Lagrangian method. 
% If isempty(options.y), the vanilla penalty method is used.
options.y = 0; 

x = penalty_method(problem, [], options);
x_reg = problem.recover_exact(x, 1/eps); 
cost_reg = problem.cost(x_reg, struct());
fprintf('Optimal cost function after inserting zeros in A*v: %e.\n', cost_reg);